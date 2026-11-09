"""Tests for the two-phase ZIP replace and the shared venv-layout helpers.

``_atomic_replace_dir`` (#49145) made each *individual* directory swap safe,
but the ZIP update replaced ~70 top-level entries in a loop with no atomicity
across iterations. An interruption partway left some entries at the new
version and the rest at the old one -- every file valid Python, the
combination unbootable. That is the mechanism behind the ``ImportError`` in
#76091 and the field report in #63717.

Reference: issues #76104 (ZIP atomicity) and #76105 (venv-helper duplication).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hermes_cli import update_cmd, update_cmd_zip
from hermes_constants import venv_bin_dir, venv_python_path

# ---------------------------------------------------------------------------
# Two-phase replace
# ---------------------------------------------------------------------------

def _live_tree(root: Path, names: dict[str, str]) -> None:
    for name, marker in names.items():
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "version.txt").write_text(marker)

def _stage_all(root: Path, new: Path, names: list[str]) -> list[tuple[str, str]]:
    return [
        (
            update_cmd._stage_replacement(str(new / n), str(root / n)),
            str(root / n),
        )
        for n in names
    ]

def test_staging_touches_nothing_live(tmp_path):
    """Phase 1 must not modify the install -- a failure there is a no-op."""
    live, new = tmp_path / "live", tmp_path / "new"
    _live_tree(live, {"agent": "old", "tools": "old"})
    _live_tree(new, {"agent": "new", "tools": "new"})

    _stage_all(live, new, ["agent", "tools"])

    assert (live / "agent" / "version.txt").read_text() == "old"
    assert (live / "tools" / "version.txt").read_text() == "old"

def test_commit_swaps_every_entry(tmp_path):
    live, new = tmp_path / "live", tmp_path / "new"
    _live_tree(live, {"agent": "old", "tools": "old"})
    _live_tree(new, {"agent": "new", "tools": "new"})

    update_cmd._commit_staged_replacements(_stage_all(live, new, ["agent", "tools"]))

    assert (live / "agent" / "version.txt").read_text() == "new"
    assert (live / "tools" / "version.txt").read_text() == "new"
    # No staging/backup litter left behind.
    assert not [p for p in os.listdir(live) if "hermes-update" in p]

def test_failed_swap_rolls_back_every_earlier_swap(tmp_path, monkeypatch):
    """The regression: a mid-loop failure must not leave a mixed-version tree.

    Before the two-phase split this produced `agent/` new + `tools/` stale --
    the exact shape that yields
    `ImportError: cannot import name 'TODO_INJECTION_HEADER'`.
    """
    live, new = tmp_path / "live", tmp_path / "new"
    _live_tree(live, {"agent": "old", "tools": "old"})
    _live_tree(new, {"agent": "new", "tools": "new"})
    staged = _stage_all(live, new, ["agent", "tools"])

    real_rename = os.rename
    calls = {"n": 0}

    def flaky_rename(src, dst):
        calls["n"] += 1
        # Let the first entry swap fully (2 renames), then break the second.
        if calls["n"] == 4:
            raise OSError("simulated AV interference")
        return real_rename(src, dst)

    monkeypatch.setattr(update_cmd.os, "rename", flaky_rename)

    with pytest.raises(OSError):
        update_cmd._commit_staged_replacements(staged)

    monkeypatch.undo()
    # Both entries must be back at the OLD version -- not one new, one old.
    versions = {
        n: (live / n / "version.txt").read_text() for n in ("agent", "tools")
    }
    assert versions == {"agent": "old", "tools": "old"}, (
        f"mixed-version tree after rollback: {versions}"
    )

def test_commit_handles_entries_absent_from_the_install(tmp_path):
    """A brand-new top-level dir has no live counterpart to move aside."""
    live, new = tmp_path / "live", tmp_path / "new"
    live.mkdir()
    _live_tree(new, {"brand_new": "new"})

    update_cmd._commit_staged_replacements(_stage_all(live, new, ["brand_new"]))

    assert (live / "brand_new" / "version.txt").read_text() == "new"

def test_staging_sets_aside_leftovers_it_cannot_prove_are_its_own(tmp_path):
    """A staging-suffix entry nothing journaled (a pre-journal crash's, or a user's) is neither staged
    over nor deleted (F78): it is kept aside byte for byte and the update proceeds."""
    live, new = tmp_path / "live", tmp_path / "new"
    _live_tree(live, {"agent": "old"})
    _live_tree(new, {"agent": "new"})
    stale = Path(f"{live / 'agent'}.hermes-update-staging")
    stale.mkdir()
    (stale / "junk.txt").write_text("from a previous crash")

    update_cmd._commit_staged_replacements(_stage_all(live, new, ["agent"]))

    assert (live / "agent" / "version.txt").read_text() == "new"
    assert not (live / "agent" / "junk.txt").exists()
    assert [p.read_text() for p in live.glob("agent.hermes-update-staging.hermes-update-kept/*")] == [
        "from a previous crash"]

# ---------------------------------------------------------------------------
# Shared venv helpers (#76105)
# ---------------------------------------------------------------------------

def test_venv_helpers_accept_str_and_path():
    assert venv_python_path("/opt/x/venv") == venv_python_path(Path("/opt/x/venv"))

# ---------------------------------------------------------------------------
# Top-level FILES must be atomic too (#76104 review, C1)
# ---------------------------------------------------------------------------

def test_top_level_files_are_swapped_atomically(tmp_path):
    """The repo root holds 20 first-party modules (run_agent.py, cli.py,
    hermes_constants.py, ...). Covering only directories would leave exactly
    the bug class this PR closes."""
    live, new = tmp_path / "live", tmp_path / "new"
    live.mkdir()
    new.mkdir()
    (live / "run_agent.py").write_text("old")
    (new / "run_agent.py").write_text("new")

    staged = [
        (
            update_cmd._stage_replacement(
                str(new / "run_agent.py"), str(live / "run_agent.py")
            ),
            str(live / "run_agent.py"),
        )
    ]
    update_cmd._commit_staged_replacements(staged)

    assert (live / "run_agent.py").read_text() == "new"
    assert not [p for p in os.listdir(live) if "hermes-update" in p]

def test_file_swap_failure_restores_the_original_file(tmp_path, monkeypatch):
    """A mid-swap failure must not leave a stale-or-corrupt root module."""
    live, new = tmp_path / "live", tmp_path / "new"
    live.mkdir()
    new.mkdir()
    for name in ("cli.py", "run_agent.py"):
        (live / name).write_text("old")
        (new / name).write_text("new")

    staged = [
        (update_cmd._stage_replacement(str(new / n), str(live / n)), str(live / n))
        for n in ("cli.py", "run_agent.py")
    ]

    real_replace = os.replace
    calls = {"n": 0}

    def flaky_replace(src, dst):  # a root file swaps in by os.replace over its hardlinked backup
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("simulated AV interference")
        return real_replace(src, dst)

    monkeypatch.setattr(update_cmd.os, "replace", flaky_replace)
    with pytest.raises(OSError):
        update_cmd._commit_staged_replacements(staged)
    monkeypatch.undo()

    versions = {n: (live / n).read_text() for n in ("cli.py", "run_agent.py")}
    assert versions == {"cli.py": "old", "run_agent.py": "old"}, (
        f"mixed/corrupt root modules after rollback: {versions}"
    )

def test_failed_staging_leaves_no_orphaned_copies(tmp_path, monkeypatch):
    """#76104 review C2: orphaned staging dirs make the retry we recommend
    fail harder than the original attempt (less free space each time)."""
    live, new = tmp_path / "live", tmp_path / "new"
    _live_tree(live, {"agent": "old", "tools": "old", "gateway": "old"})
    _live_tree(new, {"agent": "new", "tools": "new", "gateway": "new"})

    real_copytree = update_cmd.shutil.copytree
    calls = {"n": 0}

    def flaky_copytree(src, dst, *a, **kw):
        calls["n"] += 1
        if calls["n"] == 3:
            raise OSError(28, "No space left on device")
        return real_copytree(src, dst, *a, **kw)

    monkeypatch.setattr(update_cmd.shutil, "copytree", flaky_copytree)

    staged: list[tuple[str, str]] = []
    with pytest.raises(OSError):
        try:
            for n in ("agent", "tools", "gateway"):
                staged.append(
                    (
                        update_cmd._stage_replacement(
                            str(new / n), str(live / n)
                        ),
                        str(live / n),
                    )
                )
        except Exception:
            update_cmd._discard_staged(staged)
            raise
    monkeypatch.undo()

    leftovers = [p for p in os.listdir(live) if "hermes-update" in p]
    assert leftovers == [], f"orphaned staging copies: {leftovers}"
    # And nothing live was touched.
    for n in ("agent", "tools", "gateway"):
        assert (live / n / "version.txt").read_text() == "old"

def test_venv_helpers_honour_an_explicit_platform_verdict():
    """Callers must be able to override the platform check (#76107 CI).

    The suite exercises Windows paths on Linux CI by patching predicates like
    `hermes_main._is_windows`. A helper that reads `sys.platform`
    unconditionally silently drops those paths out of coverage -- and broke
    `test_verify_core_dependencies.py::test_uses_virtual_env_from_environment`,
    which patches `_is_windows` and then asserts on a `Scripts/python.exe`
    path.
    """
    v = Path("/opt/proj/venv")
    assert venv_bin_dir(v, windows=True).name == "Scripts"
    assert venv_bin_dir(v, windows=False).name == "bin"
    assert venv_python_path(v, windows=True).name == "python.exe"
    assert venv_python_path(v, windows=False).name == "python"
    # Halves must stay consistent under an explicit verdict.
    for flag in (True, False):
        assert venv_python_path(v, windows=flag).parent == venv_bin_dir(
            v, windows=flag
        )

# ---------------------------------------------------------------------------
# Crash between "move dst aside" and "move staging in" (Phase 2 review HIGH)
# ---------------------------------------------------------------------------

def test_staging_restores_backup_when_dst_is_missing(tmp_path, monkeypatch):
    """A previous run that died mid-swap leaves dst missing and the backup as
    the ONLY copy of that entry. On retry, _stage_replacement must restore
    the backup to dst BEFORE clearing leftovers — otherwise a staging failure
    right after (disk exhaustion is likeliest exactly then) leaves a hole in
    the install with nothing to roll back to."""
    live, new = tmp_path / "live", tmp_path / "new"
    live.mkdir()
    _live_tree(new, {"agent": "new"})
    # Simulate the crashed state: dst gone, backup holds the old tree.
    backup = live / "agent.hermes-update-old"
    backup.mkdir()
    (backup / "version.txt").write_text("old")

    # Staging fails (disk full) on the fresh copy.
    def boom(src, dst, *a, **kw):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(update_cmd.shutil, "copytree", boom)
    with pytest.raises(OSError):
        update_cmd._stage_replacement(str(new / "agent"), str(live / "agent"))
    monkeypatch.undo()

    # The old tree must have been restored to dst before the failure.
    assert (live / "agent" / "version.txt").read_text() == "old"
    assert not backup.exists()

    # And a clean retry completes the update normally.
    staged = _stage_all(live, new, ["agent"])
    update_cmd._commit_staged_replacements(staged)
    assert (live / "agent" / "version.txt").read_text() == "new"
    assert not [p for p in os.listdir(live) if "hermes-update" in p]

def test_staging_restores_a_dangling_symlink_backup_instead_of_deleting_it(tmp_path):
    """The leftover-backup restore tested ``exists()``: a dangling symlink backup (the only copy of a
    tracked symlink entry) read as absent and the leftover sweep deleted it (review Z3)."""
    live, new = tmp_path / "live", tmp_path / "new"
    live.mkdir()
    new.mkdir()
    (new / "alpha").write_text("NEW")
    backup = live / "alpha.hermes-update-old"
    try:
        backup.symlink_to("gone.txt")
    except OSError:
        pytest.skip("symlinks need privileges here")
    update_cmd._stage_replacement(str(new / "alpha"), str(live / "alpha"))
    assert (live / "alpha").is_symlink() and os.readlink(live / "alpha") == "gone.txt"
    assert not os.path.lexists(backup)


@pytest.mark.parametrize("kind", ["file", "dir"])
def test_a_symlink_planted_at_the_staging_path_after_the_sweep_is_never_written_through(tmp_path, monkeypatch, kind):
    """_stage_replacement swept the fixed staging name, checked it was gone, then copy2'd to that same
    path: a symlink planted in between carried the extracted bytes outside the install (review Z2).
    The stage now fails closed on the planted entry; the external file keeps its bytes."""
    live, new, outside = tmp_path / "live", tmp_path / "new", tmp_path / "outside"
    for d in (live, new, outside):
        d.mkdir()
    victim = outside / ("sentinel.txt" if kind == "file" else "sentinel")
    if kind == "file":
        victim.write_text("SENTINEL")
        (new / "alpha").write_text("NEW")
    else:
        victim.mkdir()
        (new / "alpha").mkdir()
        (new / "alpha" / "x.py").write_text("NEW")
    (live / "alpha").write_text("OLD") if kind == "file" else (live / "alpha").mkdir()
    staging = live / "alpha.hermes-update-staging"
    real_isdir = os.path.isdir

    def plant_then_isdir(path):  # the barrier: after the sweep and its lexists check, before the copy
        if str(path) == str(new / "alpha") and not os.path.lexists(staging):
            try:
                staging.symlink_to(victim)
            except OSError:
                pytest.skip("symlinks need privileges here")
        return real_isdir(path)

    monkeypatch.setattr(update_cmd_zip.os.path, "isdir", plant_then_isdir)
    with pytest.raises(OSError):
        update_cmd._stage_replacement(str(new / "alpha"), str(live / "alpha"))
    monkeypatch.undo()
    if kind == "file":
        assert victim.read_text() == "SENTINEL"
    else:
        assert list(victim.iterdir()) == []
    staging.unlink()  # the planted link (only the link) goes; a healthy stage then proceeds
    update_cmd._commit_staged_replacements([(update_cmd._stage_replacement(str(new / "alpha"), str(live / "alpha")),
                                             str(live / "alpha"))])
    assert (live / "alpha" if kind == "file" else live / "alpha" / "x.py").read_text() == "NEW"


def test_commit_failure_plus_discard_leaves_no_staging_litter(tmp_path, monkeypatch):
    """Phase-2 failure must not orphan staging copies for unswapped entries.

    _update_via_zip calls _discard_staged when _commit_staged_replacements
    raises. The rollback restores every swapped entry, but staging copies for
    the not-yet-swapped entries (potentially most of a full tree) would
    otherwise survive — and the retry's up-front free-space check runs BEFORE
    the lazy per-entry leftover cleanup, so the litter makes the retry fail
    harder than the original attempt. This pins the combination: rollback +
    discard leaves the old tree intact and ZERO update litter."""
    live, new = tmp_path / "live", tmp_path / "new"
    _live_tree(live, {"agent": "old", "tools": "old", "gateway": "old"})
    _live_tree(new, {"agent": "new", "tools": "new", "gateway": "new"})
    staged = _stage_all(live, new, ["agent", "tools", "gateway"])

    real_rename = os.rename
    calls = {"n": 0}

    def flaky_rename(src, dst):
        calls["n"] += 1
        if calls["n"] == 4:  # first entry fully swapped, second breaks
            raise OSError("simulated AV interference")
        return real_rename(src, dst)

    monkeypatch.setattr(update_cmd.os, "rename", flaky_rename)
    with pytest.raises(OSError):
        try:
            update_cmd._commit_staged_replacements(staged)
        except OSError:
            # Mirrors the _update_via_zip wiring.
            update_cmd._discard_staged(staged)
            raise
    monkeypatch.undo()

    # Old tree intact...
    for n in ("agent", "tools", "gateway"):
        assert (live / n / "version.txt").read_text() == "old"
    # ...and zero litter of any kind (staging OR backup).
    litter = [p for p in os.listdir(live) if "hermes-update" in p]
    assert litter == [], f"orphaned update litter: {litter}"


@pytest.mark.parametrize("hardlinks", [True, False])
def test_root_files_never_go_missing_mid_swap(tmp_path, monkeypatch, hardlinks):
    """Every launcher imports ``hermes_constants``/``hermes_bootstrap`` before anything else, and
    ``hermes_bootstrap`` is what reaches the restore after a killed swap: a kill between any two
    filesystem steps must find every root file present (old or new bytes), only directories moved.
    That holds on filesystems without hardlinks too (FAT32/exFAT/SMB)."""
    if not hardlinks:
        def no_link(*_a, **_k):
            raise OSError(1, "Operation not permitted")
        monkeypatch.setattr(update_cmd_zip.os, "link", no_link)
    live, new = tmp_path / "live", tmp_path / "new"
    for side, text in ((live, "old"), (new, "new")):
        (side / "hermes_cli").mkdir(parents=True)
        (side / "hermes_cli" / "main.py").write_text(text, encoding="utf-8")
        for name in ("hermes_constants.py", "hermes_bootstrap.py"):
            (side / name).write_text(text, encoding="utf-8")
    names = ("hermes_constants.py", "hermes_cli", "hermes_bootstrap.py")
    staged = [(update_cmd._stage_replacement(str(new / n), str(live / n)), str(live / n)) for n in names]
    missing: list[str] = []

    def probing(real):
        def step(src, dst):
            real(src, dst)
            missing.extend(n for n in ("hermes_constants.py", "hermes_bootstrap.py") if not (live / n).is_file())
        return step

    monkeypatch.setattr(update_cmd.os, "rename", probing(os.rename))
    monkeypatch.setattr(update_cmd.os, "replace", probing(os.replace))
    update_cmd._commit_staged_replacements(staged)
    monkeypatch.undo()
    assert not missing, f"root modules absent mid-swap: {missing}"
    assert {n: (live / n).read_text(encoding="utf-8-sig") for n in ("hermes_constants.py", "hermes_bootstrap.py")} == {
        "hermes_constants.py": "new", "hermes_bootstrap.py": "new"}
    assert not [p for p in os.listdir(live) if "hermes-update" in p]


# ---------------------------------------------------------------------------
# ZIP swap owner lock and staging journal (_early_recovery_zip / _journaled_stage_and_swap)
# ---------------------------------------------------------------------------

_LOCK_ROLE = r'''
import json, os, sys, time
from pathlib import Path
from hermes_cli import _early_recovery_zip as erz
role, work = sys.argv[1], Path(sys.argv[2])
live, lock_path = work / "live", work / "live" / ".hermes-update-zip-swap.lock"
def wait_for(name):
    end = time.time() + 30
    while not (work / name).exists():
        if time.time() > end:
            raise SystemExit(f"{role}: no {name}")
        time.sleep(0.01)
def mark(name, owned):
    (work / name).write_text(json.dumps({"owned": bool(owned), "inode": os.stat(lock_path).st_ino
                                         if lock_path.exists() else None}))
if role == "a":  # owner whose release is caught right after its unlock: B already holds the inode
    real = erz._lock_fd
    def lock_fd(fd, lock):
        done = real(fd, lock)
        if not lock:
            wait_for("b")
        return done
    erz._lock_fd = lock_fd
    with erz.zip_swap_owner_lock(live) as owned:
        mark("a", owned)
        wait_for("b-waiting"); time.sleep(0.3)
    (work / "a-done").touch()
elif role == "b":  # waiter that wins the lock as A lets go
    wait_for("a"); (work / "b-waiting").touch()
    with erz.zip_swap_owner_lock(live, wait=20) as owned:
        mark("b", owned)
        wait_for("c")
else:  # newcomer after A's release completed
    wait_for("a-done")
    with erz.zip_swap_owner_lock(live) as owned:
        mark("c", owned)
'''


def test_zip_swap_lock_is_one_inode_across_a_release(tmp_path):
    """Three real processes: A releases while B waits; B wins. C must then be refused: a release that
    unlinks the lock path lets C lock a fresh inode next to B's (two "exclusive" owners)."""
    import json
    import subprocess
    import sys

    (tmp_path / "live").mkdir()
    repo = os.path.realpath(Path(update_cmd.__file__).parent.parent)
    procs = [subprocess.Popen([sys.executable, "-c", _LOCK_ROLE, role, str(tmp_path)], cwd=repo,
                              env={**os.environ, "PYTHONPATH": repo}) for role in "abc"]
    assert [p.wait(timeout=90) for p in procs] == [0, 0, 0]
    a, b, c = (json.loads((tmp_path / r).read_text()) for r in "abc")
    assert a["owned"] and b["owned"]
    assert not c["owned"], f"two processes held the ZIP swap lock at once (inodes {b['inode']} and {c['inode']})"
    assert a["inode"] == b["inode"] == c["inode"]


@pytest.mark.skipif(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
                    reason="POSIX permission bits; root ignores them")
def test_zip_swap_lock_refuses_admission_without_a_lock(tmp_path):
    """A root where the lock file cannot be created grants nothing: no lock, no swap, and a named reason."""
    from hermes_cli._early_recovery_zip import zip_swap_owner_lock

    live = tmp_path / "live"
    live.mkdir()
    live.chmod(0o555)
    try:
        with zip_swap_owner_lock(live) as owned:
            assert not owned, "admitted to the ZIP swap without holding any lock"
            assert owned.reason.startswith("cannot open the ZIP swap lock")
        with pytest.raises(RuntimeError, match="no ZIP swap lock"):
            update_cmd_zip._journaled_stage_and_swap(str(tmp_path), [], live, None)
    finally:
        live.chmod(0o755)


def _unreadable_release(tmp_path: Path, *, read_only_dir: bool) -> tuple[Path, Path, list[Path]]:
    live, extracted = tmp_path / "live", tmp_path / "extracted"
    live.mkdir()
    extracted.mkdir()
    (live / "keep.txt").write_text("live data")
    (extracted / "first.txt").write_text("new first")
    (extracted / "second").mkdir()
    (extracted / "second" / "good.txt").write_text("new copied data")
    locked = []
    if read_only_dir:  # copytree copies its mode: the partial stage gets a directory nobody can empty
        ro = extracted / "second" / "aaa_ro"
        ro.mkdir()
        (ro / "x.txt").write_text("x")
        locked.append(ro)
    blocked = extracted / "second" / "zzz_no_read.txt"
    blocked.write_text("unreadable")
    locked.append(blocked)
    for path in locked:
        path.chmod(0o555 if path.is_dir() else 0)
    return live, extracted, locked


_POSIX_MODES = pytest.mark.skipif(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
                                  reason="POSIX permission bits; root ignores them")


@_POSIX_MODES
def test_unreadable_source_file_leaves_no_partial_stage(tmp_path):
    """The copy of ``second`` dies on an unreadable file after copying the rest: that partial staging
    tree is the failing entry's and must be dropped like the finished ones (nothing live changes)."""
    from hermes_cli._early_recovery_zip import ZIP_SWAP_JOURNAL, restore_interrupted_zip_swap

    live, extracted, locked = _unreadable_release(tmp_path, read_only_dir=False)
    try:
        with pytest.raises(OSError):
            update_cmd_zip._journaled_stage_and_swap(str(extracted), ["first.txt", "second"], live, None)
    finally:
        for path in locked:
            path.chmod(0o755)
    assert not list(live.glob("*.hermes-update-staging")), "a partial stage leaked past the failed copy"
    assert not (live / ZIP_SWAP_JOURNAL).exists()
    assert restore_interrupted_zip_swap(live) is False
    assert sorted(p.name for p in live.iterdir()) == [".hermes-update-zip-swap.lock", "keep.txt"]


@_POSIX_MODES
def test_a_stage_cleanup_that_cannot_finish_keeps_the_journal_for_recovery(tmp_path):
    """When the updater cannot remove its partial stage, the journal is that stage's only record: it
    stays, and the next launch's recovery removes the stage, then the journal."""
    from hermes_cli._early_recovery_zip import ZIP_SWAP_JOURNAL, restore_interrupted_zip_swap

    live, extracted, locked = _unreadable_release(tmp_path, read_only_dir=True)
    try:
        with pytest.raises(OSError):
            update_cmd_zip._journaled_stage_and_swap(str(extracted), ["first.txt", "second"], live, None)
    finally:
        for path in locked:
            path.chmod(0o755)
    assert (live / "second.hermes-update-staging").exists(), "harness: the cleanup was expected to fail"
    assert (live / ZIP_SWAP_JOURNAL).exists(), "the journal went while its staging path was still on disk"
    assert restore_interrupted_zip_swap(live) is False  # nothing live moved: no relaunch
    assert not list(live.glob("*.hermes-update-staging")) and not (live / ZIP_SWAP_JOURNAL).exists()
    assert (live / "keep.txt").read_text() == "live data"


def _swap_or_recover(extracted: Path, entries: list[str], live: Path) -> None:
    """``_download_and_swap_zip``'s wiring: a failed swap is settled from its journal right after."""
    from hermes_cli._early_recovery_zip import restore_interrupted_zip_swap

    try:
        update_cmd_zip._journaled_stage_and_swap(str(extracted), entries, live, None)
    except Exception:
        restore_interrupted_zip_swap(live)


@_POSIX_MODES
def test_a_stale_backup_never_stands_in_for_the_live_entry(tmp_path, monkeypatch):
    """A leftover ``<entry>.hermes-update-old`` this swap did not make (a pre-journal crash holding a
    read-only directory) must be removed or refuse the swap; it must never become the live entry."""
    from hermes_cli import update_cmd_commit

    monkeypatch.setattr(update_cmd_commit, "arm_commit_obligations", lambda *a, **k: None)
    live, extracted = tmp_path / "live", tmp_path / "extracted"
    (live / "pkg").mkdir(parents=True)
    (live / "pkg" / "m.py").write_text("LIVE_OLD", encoding="utf-8")
    (extracted / "pkg").mkdir(parents=True)
    (extracted / "pkg" / "m.py").write_text("NEW", encoding="utf-8")
    stale = live / "pkg.hermes-update-old" / "ro"
    stale.mkdir(parents=True)
    (stale / "f").write_text("stale remnant", encoding="utf-8")
    stale.chmod(0o555)
    try:
        _swap_or_recover(extracted, ["pkg"], live)
    finally:
        for path in live.rglob("*"):
            if path.is_dir():
                path.chmod(0o755)
    module = live / "pkg" / "m.py"
    assert module.is_file() and module.read_text(encoding="utf-8-sig") in ("LIVE_OLD", "NEW"), sorted(
        str(p.relative_to(live)) for p in live.rglob("*"))


def test_a_failed_swap_keeps_a_file_the_user_made_at_a_never_installed_entry(tmp_path, monkeypatch):
    """The swap fails before ``brand_new`` is installed; a file the user created there meanwhile is
    theirs, and neither the rollback nor the journal recovery may delete it."""
    from hermes_cli import update_cmd_commit

    live, extracted = tmp_path / "live", tmp_path / "extracted"
    for side, text in ((live, "old"), (extracted, "new")):
        (side / "a").mkdir(parents=True)
        (side / "a" / "v.txt").write_text(text, encoding="utf-8")
    (extracted / "brand_new").write_text("new entry", encoding="utf-8")
    user_file = live / "brand_new"
    monkeypatch.setattr(update_cmd_commit, "arm_commit_obligations",
                        lambda *a, **k: user_file.write_text("USER NOTE", encoding="utf-8"))
    real_rename = os.rename

    def refuse_moving_a(src, dst):  # Windows AV holding ``a`` open
        if os.fspath(src) == str(live / "a") and os.fspath(dst).endswith(".hermes-update-old"):
            raise PermissionError(13, "in use")
        return real_rename(src, dst)

    monkeypatch.setattr(update_cmd_zip.os, "rename", refuse_moving_a)
    _swap_or_recover(extracted, ["a", "brand_new"], live)
    monkeypatch.undo()
    assert user_file.is_file() and user_file.read_text(encoding="utf-8-sig") == "USER NOTE"
    assert (live / "a" / "v.txt").read_text(encoding="utf-8-sig") == "old"
    assert not [p.name for p in live.iterdir() if "hermes-update-staging" in p.name or p.name.endswith("-old")]

def test_a_journal_that_cannot_be_dropped_after_the_commit_never_fails_the_update(tmp_path, monkeypatch):
    """The swap committed: an AV scan holding the journal must not turn it into a reported failure
    that disarms the new tree's completion obligations (F19)."""
    from hermes_cli import update_cmd_commit
    from hermes_cli.update_host_obligation import read_host_obligation
    from hermes_cli.venv_sync import completion_pending_path

    live, extracted = tmp_path / "live", tmp_path / "extracted"
    _live_tree(live, {"payload": "old"})
    _live_tree(extracted, {"payload": "new"})
    real = Path.unlink

    def held(self, *args, **kwargs):
        if self.name == update_cmd_zip.ZIP_SWAP_JOURNAL:
            raise PermissionError(13, "being used by another process", str(self))
        return real(self, *args, **kwargs)

    update_cmd_commit.begin_update_attempt()
    monkeypatch.setattr(Path, "unlink", held)
    try:
        update_cmd_zip._journaled_stage_and_swap(str(extracted), ["payload"], live, "b" * 40)
    finally:
        update_cmd_commit.begin_update_attempt()
    assert (live / "payload" / "version.txt").read_text(encoding="utf-8-sig") == "new"
    assert completion_pending_path(live).is_file()
    assert (read_host_obligation() or {}).get("expected_sha") == "b" * 40


_KILLED_BACKUP_COPY = r'''
import os, shutil, sys
from pathlib import Path
from hermes_cli import update_cmd_commit, update_cmd_zip
live, extracted = Path(sys.argv[1]), Path(sys.argv[2])
update_cmd_commit.arm_commit_obligations = lambda *a, **k: None
update_cmd_zip._hardlink_backup = lambda path, backup: False  # FAT32/exFAT/SMB: no hardlinks
real = shutil.copyfileobj
def killed_inside_the_backup_copy(source, out, length=0):
    if source.name == str(live / "a.py"):
        out.write(source.read(2))
        out.flush()
        os._exit(9)  # a SIGKILL: no handler runs, the half-written temp stays
    return real(source, out, length)
shutil.copyfileobj = killed_inside_the_backup_copy
update_cmd_zip._journaled_stage_and_swap(str(extracted), ["a.py"], live, None)
'''


def test_a_backup_copy_killed_before_its_rename_is_cleared_by_the_recovery(tmp_path):
    """On a file system without hardlinks ``_file_backup`` copies to ``<entry>.hermes-update-old.<gen>.tmp``
    first. The real swap is killed inside that copy: the temp it journaled on creation is its own and
    goes, or every later ZIP update refuses on "uncommitted changes" (review C4)."""
    import subprocess
    import sys

    from hermes_cli._early_recovery_zip import ZIP_SWAP_JOURNAL, restore_interrupted_zip_swap

    live, extracted = tmp_path / "live", tmp_path / "extracted"
    live.mkdir()
    extracted.mkdir()
    (live / "a.py").write_text("old\n", encoding="utf-8")
    (extracted / "a.py").write_text("new\n", encoding="utf-8")
    repo = os.path.realpath(Path(update_cmd.__file__).parent.parent)
    killed = subprocess.run([sys.executable, "-c", _KILLED_BACKUP_COPY, str(live), str(extracted)], cwd=repo,
                            env={**os.environ, "PYTHONPATH": repo}, capture_output=True, text=True, timeout=120, check=False)
    temps = list(live.glob("a.py.hermes-update-old.*.tmp"))
    assert killed.returncode == 9 and [p.read_bytes() for p in temps] == [b"ol"], (
        f"harness: not killed inside the backup copy: {killed.returncode} {killed.stderr[-2000:]}")

    restore_interrupted_zip_swap(live)

    assert not (live / ZIP_SWAP_JOURNAL).exists()
    assert sorted(p.name for p in live.iterdir() if not p.name.endswith(".lock")) == ["a.py"]
    assert (live / "a.py").read_text(encoding="utf-8") == "old\n"


@pytest.mark.parametrize("content", ["USER FILE", "old", "old\n", ""])
def test_a_file_at_the_backup_temp_name_that_is_not_the_killed_copy_is_kept(tmp_path, content):
    """The run's tag names the temp but proves nothing, and neither do its bytes: a swap killed before
    its backup copy existed leaves the name free, and a file there afterwards is not Hermes', even an
    empty one, a prefix or a full copy of the live file. Only the identity the swap journaled when it
    created the temp may delete it; anything else is kept aside, the journal still retiring (F78-R)."""
    from hermes_cli._early_recovery_zip import (
        ZIP_SWAP_JOURNAL, restore_interrupted_zip_swap, write_zip_swap_journal, zip_entry_identity)

    live = tmp_path / "live"
    live.mkdir()
    (live / "a.py").write_text("old\n", encoding="utf-8")
    (live / "a.py.hermes-update-staging").write_text("new\n", encoding="utf-8")
    write_zip_swap_journal(live, "swapping", [["a.py", True, zip_entry_identity(live / "a.py.hermes-update-staging"),
                                              zip_entry_identity(live / "a.py")]], "0123456789ab")
    (live / "a.py.hermes-update-old.0123456789ab.tmp").write_text(content, encoding="utf-8")

    restore_interrupted_zip_swap(live)

    kept = [p for p in live.iterdir() if ".hermes-update-kept" in p.name]
    assert [p.read_text(encoding="utf-8") for p in kept] == [content], sorted(p.name for p in live.iterdir())
    assert not (live / ZIP_SWAP_JOURNAL).exists()
    assert (live / "a.py").read_text(encoding="utf-8") == "old\n"


def _swap_killed(tmp_path, monkeypatch, *, live: dict, new: dict, install_first: bool) -> Path:
    """Run the real journaled stage+swap and kill it inside the swap: nothing renamed yet, or only the
    first staged entry renamed into place. The journal the real writer left is what recovery reads."""
    from hermes_cli import update_cmd_commit
    from hermes_cli._early_recovery import ZIP_SWAP_JOURNAL

    root, extracted = tmp_path / "live", tmp_path / "extracted"
    for d, files in ((root, live), (extracted, new)):
        d.mkdir()
        for name, text in files.items():
            (d / name).write_text(text, encoding="utf-8")
    monkeypatch.setattr(update_cmd_commit, "arm_commit_obligations", lambda *a, **k: None)

    def killed(staged, **_kw):
        if install_first:  # the real swap's first step: backup (hardlink) when live, then the rename
            staging, dst = staged[0]
            if os.path.lexists(dst):
                update_cmd_zip._file_backup(dst, dst + ".hermes-update-old")
            os.replace(staging, dst)
        raise RuntimeError("killed mid-swap")

    with monkeypatch.context() as fault:
        fault.setattr(update_cmd_zip, "_commit_staged_replacements", killed)
        with pytest.raises(RuntimeError, match="killed"):
            update_cmd_zip._journaled_stage_and_swap(str(extracted), sorted(new), root, None)
    assert (root / ZIP_SWAP_JOURNAL).exists(), "harness: the kill must leave the journal"
    return root


def test_recovery_never_installs_or_deletes_a_backup_suffix_created_after_the_kill(tmp_path, monkeypatch):
    """A swap killed before any rename; then a user file appears at ``cli.py.hermes-update-old``. The
    path-only journal took it for the swap's backup: it overwrote the live cli.py with it, and the old
    bytes were gone (review Z1). Recovery now proves a backup is the entry it recorded before using or
    deleting it: the live file stays, the user's file is kept aside, byte for byte."""
    from hermes_cli._early_recovery_zip import ZIP_SWAP_JOURNAL, restore_interrupted_zip_swap

    root = _swap_killed(tmp_path, monkeypatch, live={"cli.py": "LIVE"}, new={"cli.py": "NEW"}, install_first=False)
    (root / "cli.py.hermes-update-old").write_text("USER LATER", encoding="utf-8")

    restore_interrupted_zip_swap(root)

    assert (root / "cli.py").read_text(encoding="utf-8") == "LIVE"
    kept = [p for p in root.iterdir() if p.read_bytes() == b"USER LATER"]
    assert len(kept) == 1 and kept[0].name != "cli.py"
    assert not (root / "cli.py.hermes-update-staging").exists() and not (root / ZIP_SWAP_JOURNAL).exists()


def test_recovery_never_deletes_a_user_file_that_replaced_an_installed_entry(tmp_path, monkeypatch):
    """A new entry (notes.md) was renamed into place, then the swap was killed; the user then replaced
    notes.md with a file of their own. Recovery deleted whatever sat at the journaled name (review Z1);
    only the entry the swap installed (its recorded staging identity) may go."""
    from hermes_cli._early_recovery_zip import ZIP_SWAP_JOURNAL, restore_interrupted_zip_swap

    root = _swap_killed(tmp_path, monkeypatch, live={}, new={"notes.md": "NEW"}, install_first=True)
    (root / "notes.md").unlink()
    (root / "notes.md").write_text("USER NOTES", encoding="utf-8")

    restore_interrupted_zip_swap(root)

    assert [p.read_text(encoding="utf-8") for p in root.iterdir() if p.name.startswith("notes.md")] == ["USER NOTES"]
    assert not (root / ZIP_SWAP_JOURNAL).exists()


def test_recovery_still_rolls_back_an_authentic_killed_swap(tmp_path, monkeypatch):
    """The control: a swap killed after installing its first entry rolls back to the old tree, and every
    staging copy and backup the swap itself made is deleted (provenance proven), with nothing kept aside."""
    from hermes_cli._early_recovery_zip import ZIP_SWAP_JOURNAL, restore_interrupted_zip_swap

    root = _swap_killed(tmp_path, monkeypatch, live={"a.py": "OLD_A", "b.py": "OLD_B"},
                        new={"a.py": "NEW_A", "b.py": "NEW_B", "c.py": "NEW_C"}, install_first=True)
    assert (root / "a.py").read_text(encoding="utf-8") == "NEW_A", "harness: a.py was installed"
    assert restore_interrupted_zip_swap(root) is True
    assert {p.name: p.read_text(encoding="utf-8") for p in root.iterdir() if not p.name.startswith(".")} == {
        "a.py": "OLD_A", "b.py": "OLD_B"}
    assert not (root / ZIP_SWAP_JOURNAL).exists()


@pytest.mark.parametrize("requires", [">=3.8", ">=3.99"])
def test_the_zip_gate_never_admits_a_conflict_marker_even_for_a_newer_python(tmp_path, monkeypatch, requires):
    """A release whose requires-python excludes this interpreter skipped the ZIP pre-commit gate
    entirely, so a startup module with a merge-conflict marker (broken under EVERY Python) was
    swapped in (review C5). Its syntax may be a newer Python's, the marker never is."""
    from hermes_cli import update_cmd_commit

    armed = []
    monkeypatch.setattr(update_cmd_commit, "arm_commit_obligations", lambda *a, **k: armed.append(a))
    live, extracted = tmp_path / "live", tmp_path / "extracted"
    (live / "hermes_cli").mkdir(parents=True)
    (live / "hermes_cli" / "main.py").write_text("OLD = 1\n", encoding="utf-8")
    (extracted / "hermes_cli").mkdir(parents=True)
    (extracted / "hermes_cli" / "main.py").write_text(
        "<<<<<<< HEAD\nA = 1\n=======\nA = 2\n>>>>>>> branch\n", encoding="utf-8")
    (extracted / "pyproject.toml").write_text(
        f'[project]\nname = "x"\nrequires-python = "{requires}"\n', encoding="utf-8")

    with pytest.raises(SyntaxError):
        update_cmd_zip._journaled_stage_and_swap(str(extracted), ["hermes_cli", "pyproject.toml"], live, None)

    assert not armed  # refused before the commit point
    assert (live / "hermes_cli" / "main.py").read_text(encoding="utf-8") == "OLD = 1\n"


def _git_checkout(root: Path, files: dict[str, str]) -> None:
    import subprocess

    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text, encoding="utf-8")
    for args in (["init", "-q"], ["add", "."], ["-c", "user.name=t", "-c", "user.email=t@example.invalid",
                                                 "commit", "-qm", "pre"]):
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


@_POSIX_MODES
def test_a_committed_swap_keeps_its_journal_until_the_backup_is_gone(tmp_path, monkeypatch):
    """Q1/F79: the swap committed but its backup could not be removed (a read-only directory in the old
    tree here; an AV scan holding it on Windows). The journal is that backup's only ownership record: it
    stays ``committed``, the next launch's recovery drops the backup and then the journal, keeping the
    new tree, and the retry's dirty-tree gate is clean again."""
    import json

    from hermes_cli import update_cmd_commit
    from hermes_cli._early_recovery_zip import ZIP_SWAP_JOURNAL, restore_interrupted_zip_swap

    monkeypatch.setattr(update_cmd_commit, "arm_commit_obligations", lambda *a, **k: None)
    live, extracted = tmp_path / "live", tmp_path / "extracted"
    for side in (live, extracted):  # identical payloads: only the swap's own leftovers can dirty the tree
        side.mkdir()
    _git_checkout(live, {"payload/ro/f.txt": "same"})
    (extracted / "payload" / "ro").mkdir(parents=True)
    (extracted / "payload" / "ro" / "f.txt").write_text("same", encoding="utf-8")
    (live / "payload" / "ro").chmod(0o555)  # git does not track it; the backup's rmtree cannot empty it
    try:
        assert update_cmd_zip._journaled_stage_and_swap(str(extracted), ["payload"], live, "b" * 40)
        assert (live / "payload.hermes-update-old").exists(), "harness: the backup cleanup was expected to fail"
        journal = live / ZIP_SWAP_JOURNAL
        assert journal.is_file(), "the journal went while the swap's backup was still on disk"
        assert json.loads(journal.read_text(encoding="utf-8"))["phase"] == "committed"
        assert restore_interrupted_zip_swap(live) is False  # the new tree stays: nothing to relaunch
    finally:
        for path in live.rglob("*"):
            if path.is_dir() and not path.is_symlink():
                path.chmod(0o755)
    assert not (live / "payload.hermes-update-old").exists() and not (live / ZIP_SWAP_JOURNAL).exists()
    assert update_cmd_zip._zip_overlay_block_reason(live) is None

@pytest.mark.parametrize("planted", ["cli.py.hermes-update-staging", "cli.py.hermes-update-old"])
def test_a_suffix_path_that_appears_after_the_preflight_is_never_deleted(tmp_path, monkeypatch, planted):
    """F78: a file at the staging suffix (created during the download, after the clean-tree preflight)
    or at the backup suffix (created after the pre-swap recheck) is nothing this transaction made. The
    stage dropped the first unconditionally and the hardlink backup the second; both survive now, byte
    for byte, and the swap is refused with the live tree left old."""
    from hermes_cli import update_cmd_commit
    from hermes_cli._early_recovery_zip import ZIP_SWAP_JOURNAL, restore_interrupted_zip_swap

    live, extracted = tmp_path / "live", tmp_path / "extracted"
    live.mkdir()
    extracted.mkdir()
    _git_checkout(live, {"cli.py": "LIVE"})
    (extracted / "cli.py").write_text("NEW", encoding="utf-8")
    user = live / planted
    if planted.endswith("-staging"):
        user.write_text("USER NOTE", encoding="utf-8")
    monkeypatch.setattr(update_cmd_commit, "arm_commit_obligations",  # the last step before the swap
                        lambda *a, **k: user.exists() or user.write_text("USER NOTE", encoding="utf-8"))
    with pytest.raises((SystemExit, OSError)):
        update_cmd_zip._journaled_stage_and_swap(str(extracted), ["cli.py"], live, "b" * 40)
    restore_interrupted_zip_swap(live)
    assert (live / "cli.py").read_text(encoding="utf-8") == "LIVE"
    kept = [p.name for p in live.iterdir() if p.is_file() and p.read_bytes() == b"USER NOTE"]
    assert len(kept) == 1, sorted(p.name for p in live.iterdir())
    assert not (live / ZIP_SWAP_JOURNAL).exists()


def test_a_stage_journals_each_directory_before_its_fill_and_not_every_file(tmp_path, monkeypatch):
    """The staging journal is an fsync'd rewrite, ~9 ms on ext4: one per entry was ~1 s of a 123-entry
    stage (review K132361). A directory's id is still on disk before copytree fills it, and a kill
    right after staging still leaves every copy provably the update's: recovery drops them all."""
    import json

    from hermes_cli import update_cmd_commit
    from hermes_cli._early_recovery_zip import ZIP_SWAP_JOURNAL, restore_interrupted_zip_swap

    live, extracted = tmp_path / "live", tmp_path / "extracted"
    _live_tree(live, {"agent": "old", "tools": "old"})
    _live_tree(extracted, {"agent": "new", "tools": "new"})
    entries = ["a.py", "agent", "b.py", "c.py", "tools", "d.py", "e.py"]
    for name in (e for e in entries if e.endswith(".py")):
        (extracted / name).write_text(f"new {name}", encoding="utf-8")
    durable_before_fill, staging_writes = set(), []
    real = update_cmd_zip.write_zip_swap_journal

    def journal(root, phase, *args, **kwargs):
        real(root, phase, *args, **kwargs)
        staging_writes.append(phase == "staging")
        for name, _existed, staged_id, _live in json.loads((root / ZIP_SWAP_JOURNAL).read_text("utf-8"))["entries"]:
            staging = root / f"{name}.hermes-update-staging"
            if staged_id and staging.is_dir() and not any(staging.iterdir()):
                durable_before_fill.add(name)

    class Killed(BaseException):
        pass

    def killed(*_args, **_kwargs):
        raise Killed  # the process dies after staging, before the swapping record

    monkeypatch.setattr(update_cmd_zip, "write_zip_swap_journal", journal)
    monkeypatch.setattr(update_cmd_commit, "tree_syntax_error", killed)
    with pytest.raises(Killed):
        update_cmd_zip._journaled_stage_and_swap(str(extracted), entries, live, None)

    assert durable_before_fill == {"agent", "tools"}
    assert sum(staging_writes) == 4, "the opening record, one per directory, one for the last entry"
    restore_interrupted_zip_swap(live)
    assert sorted(p.name for p in live.iterdir() if not p.name.startswith(".")) == ["agent", "tools"]
    assert (live / "agent" / "version.txt").read_text() == "old"
    assert not (live / ZIP_SWAP_JOURNAL).exists()
