"""Scratch dir contract: TMPDIR/TMP/TEMP follow HERMES_HOME/cache/scratch unless the user set them."""

import contextlib
import os
import shlex
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

from tests.waiting import wait_until

import psutil
import pytest

from hermes_constants import apply_scratch_tmp_env, get_scratch_dir, prune_scratch_dir


def test_scratch_env_follows_home_and_respects_user_tmpdir(tmp_path):
    """Unset temp vars → scratch of env HERMES_HOME; a Hermes-exported value re-derives for a routed
    home; a user/OS-set value (macOS ``/var/folders``, ``%TEMP%``) is never touched."""
    home_a, home_b = tmp_path / "a", tmp_path / "b"
    env = {"HERMES_HOME": str(home_a)}
    assert apply_scratch_tmp_env(env) is True
    scratch_a = str(home_a / "cache" / "scratch")
    assert env["TMPDIR"] == env["TMP"] == env["TEMP"] == env["HERMES_SCRATCH_DIR"] == scratch_a
    assert os.path.isdir(scratch_a)

    env["HERMES_HOME"] = str(home_b)  # child served under another profile's home
    assert apply_scratch_tmp_env(env) is True
    assert env["TMPDIR"] == str(home_b / "cache" / "scratch")

    user_env = {"HERMES_HOME": str(home_a), "TMPDIR": "/var/folders/zz"}
    assert apply_scratch_tmp_env(user_env) is False
    assert user_env["TMPDIR"] == "/var/folders/zz" and "HERMES_SCRATCH_DIR" not in user_env
    assert "TMP" not in user_env  # a partially user-set triple is left exactly as found


def test_bootstrap_import_exports_scratch_to_process_and_children(tmp_path):
    """``import hermes_bootstrap`` alone makes ``tempfile`` (this process AND a child) land in scratch."""
    env = {k: v for k, v in os.environ.items() if k not in ("TMPDIR", "TMP", "TEMP", "HERMES_SCRATCH_DIR")}
    env["HERMES_HOME"] = str(tmp_path)
    code = ("import tempfile, os, subprocess, sys; import hermes_bootstrap; "
            "print(tempfile.gettempdir()); "
            "print(subprocess.run([sys.executable, '-c', 'import tempfile;print(tempfile.gettempdir())'],"
            " capture_output=True, text=True).stdout.strip())")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, encoding="utf-8",
                         cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))), check=True)
    expected = str(tmp_path / "cache" / "scratch")
    assert out.stdout.split() == [expected, expected]


def test_prune_removes_idle_entries_and_keeps_trees_written_deep_inside(tmp_path):
    """Idle retention: an entry goes when nothing in its subtree was written within the window;
    a tree whose only recent write is three levels down is still in use and stays, even though
    its top-level mtime is ancient (a directory's mtime ignores writes below its children)."""
    scratch = get_scratch_dir(tmp_path, prune=False)
    idle, live, fresh = scratch / "idle", scratch / "live", scratch / "fresh.txt"
    deep = live / "lane" / "wt"
    deep.mkdir(parents=True)
    idle.mkdir()
    (idle / "f").write_text("x", encoding="utf-8")
    (deep / "log").write_text("x", encoding="utf-8")
    fresh.write_text("y", encoding="utf-8")
    ancient = time.time() - 30 * 3600
    for path in (idle, idle / "f", live, live / "lane", deep):
        os.utime(path, (ancient, ancient))
    assert prune_scratch_dir(scratch) == 1
    assert not idle.exists() and live.exists() and fresh.exists()


@pytest.mark.platforms("posix")  # POSIX directory modes
class TestScratchDirPermissionPolicy:
    """get_scratch_dir must honor the home permission policy instead of a blanket 0700:
    an explicit HERMES_HOME_MODE and a managed/shared home win (#117347)."""

    @staticmethod
    def _native_mode_after_chmod(path, requested=0o2770):
        """Return what this host/filesystem preserves from a real chmod request.

        macOS may clear setgid when the directory group is not one of the caller's groups.
        The portable contract is to request the operator's mode and preserve every bit the
        host accepts, not to pretend all POSIX filesystems retain identical special bits.
        """
        os.chmod(path, requested)
        return stat.S_IMODE(os.stat(path).st_mode)

    def _isolate_env(self, monkeypatch, tmp_path):
        # A known, marker-free effective home: each case below plants `.managed` in the home whose
        # scratch dir it exercises (`get_scratch_dir(home)` reads the marker there), and the real
        # home's own marker must not leak into callers that still resolve the effective home.
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
        for var in ("HERMES_HOME_MODE", "HERMES_MANAGED", "HERMES_CONTAINER",
                    "HERMES_SKIP_CHMOD", "HERMES_UID", "HERMES_GID"):
            monkeypatch.delenv(var, raising=False)

    def test_default_is_owner_only(self, tmp_path, monkeypatch):
        self._isolate_env(monkeypatch, tmp_path)
        scratch = get_scratch_dir(tmp_path, prune=False)
        assert stat.S_IMODE(os.stat(scratch).st_mode) == 0o700

    def test_explicit_home_mode_uses_native_chmod_semantics(self, tmp_path, monkeypatch):
        self._isolate_env(monkeypatch, tmp_path)
        probe = tmp_path / "mode-probe"
        probe.mkdir()
        expected = self._native_mode_after_chmod(probe)
        monkeypatch.setenv("HERMES_HOME_MODE", "2770")
        scratch = get_scratch_dir(tmp_path, prune=False)
        assert stat.S_IMODE(os.stat(scratch).st_mode) == expected

    def test_managed_env_leaves_preexisting_mode_untouched(self, tmp_path, monkeypatch):
        self._isolate_env(monkeypatch, tmp_path)
        monkeypatch.setenv("HERMES_MANAGED", "nixos")
        pre = tmp_path / "cache" / "scratch"
        pre.mkdir(parents=True)
        expected = self._native_mode_after_chmod(pre)
        scratch = get_scratch_dir(tmp_path, prune=False)
        assert stat.S_IMODE(os.stat(scratch).st_mode) == expected

    def test_managed_marker_file_leaves_preexisting_mode_untouched(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        home.mkdir()
        self._isolate_env(monkeypatch, tmp_path)
        (home / ".managed").write_text("nixos", encoding="utf-8")
        pre = home / "cache" / "scratch"
        pre.mkdir(parents=True)
        expected = self._native_mode_after_chmod(pre)
        scratch = get_scratch_dir(home, prune=False)
        assert stat.S_IMODE(os.stat(scratch).st_mode) == expected

    def test_empty_managed_marker_counts_as_managed(self, tmp_path, monkeypatch):
        # Legacy NixOS module wrote an empty marker; config.get_managed_system treats it as
        # managed, so the parity twin must too (not re-enable chmod on managed installs).
        home = tmp_path / "home"
        home.mkdir()
        self._isolate_env(monkeypatch, tmp_path)
        (home / ".managed").write_text("", encoding="utf-8")
        pre = home / "cache" / "scratch"
        pre.mkdir(parents=True)
        expected = self._native_mode_after_chmod(pre)
        scratch = get_scratch_dir(home, prune=False)
        assert stat.S_IMODE(os.stat(scratch).st_mode) == expected

    def test_unreadable_managed_marker_counts_as_managed(self, tmp_path, monkeypatch):
        # A marker that exists but cannot be read (OSError -> "") still counts as managed.
        home = tmp_path / "home"
        home.mkdir()
        self._isolate_env(monkeypatch, tmp_path)
        (home / ".managed").mkdir()
        pre = home / "cache" / "scratch"
        pre.mkdir(parents=True)
        expected = self._native_mode_after_chmod(pre)
        scratch = get_scratch_dir(home, prune=False)
        assert stat.S_IMODE(os.stat(scratch).st_mode) == expected

    def test_effective_home_marker_does_not_govern_another_homes_scratch(self, tmp_path, monkeypatch):
        # The caller's home decides the policy. A boot caller (``export_scratch_tmp_env``) and
        # ``hermes doctor`` have already resolved theirs, so the marker lookup must not fall back
        # to ``get_hermes_home()``: for a sticky profile with HERMES_HOME unset that lookup warns
        # about the default profile on every command, and it is the wrong home to judge by anyway.
        selected = tmp_path / "home"
        selected.mkdir()
        self._isolate_env(monkeypatch, tmp_path)
        (selected / ".managed").write_text("nixos", encoding="utf-8")
        other = tmp_path / "other"
        pre = other / "cache" / "scratch"
        pre.mkdir(parents=True)
        os.chmod(pre, 0o750)
        assert stat.S_IMODE(os.stat(get_scratch_dir(other, prune=False)).st_mode) == 0o700

    def test_canonical_container_signal_without_env_override_keeps_operator_mode(self, tmp_path, monkeypatch):
        # Podman/containerd/K8s runtimes often don't export HERMES_CONTAINER; the canonical
        # _detect_container breadth (not the narrower legacy signal set) must skip the chmod.
        self._isolate_env(monkeypatch, tmp_path)
        monkeypatch.setattr("hermes_constants._detect_container", lambda: True)
        pre = tmp_path / "cache" / "scratch"
        pre.mkdir(parents=True)
        os.chmod(pre, 0o750)
        scratch = get_scratch_dir(tmp_path, prune=False)
        assert stat.S_IMODE(os.stat(scratch).st_mode) == 0o750

    def test_repeated_calls_do_not_strip_filesystem_supported_mode_bits(self, tmp_path, monkeypatch):
        self._isolate_env(monkeypatch, tmp_path)
        probe = tmp_path / "mode-probe"
        probe.mkdir()
        expected = self._native_mode_after_chmod(probe)
        monkeypatch.setenv("HERMES_HOME_MODE", "2770")
        first = get_scratch_dir(tmp_path, prune=False)
        second = get_scratch_dir(tmp_path, prune=False)
        assert first == second
        assert stat.S_IMODE(os.stat(second).st_mode) == expected

    def test_container_keeps_operator_mode_but_honors_explicit(self, tmp_path, monkeypatch):
        self._isolate_env(monkeypatch, tmp_path)
        monkeypatch.setenv("HERMES_CONTAINER", "1")
        pre = tmp_path / "cache" / "scratch"
        pre.mkdir(parents=True)
        os.chmod(pre, 0o750)
        assert stat.S_IMODE(os.stat(get_scratch_dir(tmp_path, prune=False)).st_mode) == 0o750
        probe = tmp_path / "mode-probe"
        probe.mkdir()
        expected = self._native_mode_after_chmod(probe)
        monkeypatch.setenv("HERMES_HOME_MODE", "2770")
        assert stat.S_IMODE(os.stat(get_scratch_dir(tmp_path, prune=False)).st_mode) == expected

    def test_hermes_uid_gid_applied_to_scratch(self, tmp_path, monkeypatch):
        self._isolate_env(monkeypatch, tmp_path)
        monkeypatch.setenv("HERMES_UID", "1000")
        monkeypatch.setenv("HERMES_GID", "911")
        with patch.object(os, "chown") as mock_chown:
            get_scratch_dir(tmp_path, prune=False)
        mock_chown.assert_called_once_with(tmp_path / "cache" / "scratch", 1000, 911)


@pytest.mark.platforms("posix")  # POSIX file modes
def test_secure_file_skips_chmod_on_canonical_container_signal(tmp_path, monkeypatch):
    """_secure_file skips on the same canonical container signal that apply_secure_dir_policy /
    get_scratch_dir already honor (one policy implementation in hermes_constants)."""
    from hermes_cli import config

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    for var in ("HERMES_MANAGED", "HERMES_CONTAINER", "HERMES_SKIP_CHMOD"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("hermes_constants._detect_container", lambda: True)
    f = tmp_path / "config.yaml"
    f.write_text("", encoding="utf-8")
    os.chmod(f, 0o640)
    config._secure_file(f)
    assert stat.S_IMODE(os.stat(f).st_mode) == 0o640


def test_prune_reaps_process_living_in_idle_entry_and_spares_live_tree(tmp_path):
    """A process whose cwd is inside an idle entry is gone by the time the entry is
    (a lane's headless browsers survived for days with a deleted cwd); one living in a
    tree that is still being written is not touched."""
    import subprocess

    scratch = get_scratch_dir(tmp_path, prune=False)
    idle, live = scratch / "idle-lane" / "wt", scratch / "live-lane" / "wt"
    idle.mkdir(parents=True)
    live.mkdir(parents=True)
    ancient = time.time() - 30 * 3600
    for path in (idle.parent, idle, live.parent, live):
        os.utime(path, (ancient, ancient))
    (live / "log").write_text("still writing", encoding="utf-8")
    sleeper = [sys.executable, "-c", "import time; time.sleep(60)"]
    doomed = subprocess.Popen(sleeper, cwd=str(idle), stdin=subprocess.DEVNULL)
    spared = subprocess.Popen(sleeper, cwd=str(live), stdin=subprocess.DEVNULL)
    try:
        assert prune_scratch_dir(scratch) == 1
        assert doomed.wait(timeout=10) is not None
        assert spared.poll() is None
        assert not idle.parent.exists() and live.exists()
    finally:
        for proc in (doomed, spared):
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)


def test_prune_records_removals_and_kills_in_scratch_prune_log(tmp_path: Path) -> None:
    """Every removed entry and every signalled process leaves a line in
    ``<home>/logs/scratch-prune.log``: the prune's own handler writes it, so the boot-time
    prune (before ``setup_logging()``) is not silent. An entry rmtree could not delete is
    neither counted nor recorded as removed (#132401)."""
    import subprocess

    scratch = get_scratch_dir(tmp_path, prune=False)
    gone, stuck = scratch / "idle-lane", scratch / "stuck-lane"
    ancient = time.time() - 30 * 3600
    for entry in (gone, stuck):
        entry.mkdir()
        (entry / "out.md").write_text("deliverable", encoding="utf-8")
        for path in (entry / "out.md", entry):
            os.utime(path, (ancient, ancient))
    worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], cwd=str(gone),
                              stdin=subprocess.DEVNULL)
    real_rmtree = shutil.rmtree

    def rmtree(path: str | os.PathLike[str], *args: object, **kwargs: object) -> None:
        if os.path.realpath(path) != os.path.realpath(stuck):
            real_rmtree(path, *args, **kwargs)

    try:
        with patch("hermes_constants_scratch.shutil.rmtree", rmtree):
            assert prune_scratch_dir(scratch) == 1
        assert worker.wait(timeout=10) is not None
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait(timeout=10)
    log = (tmp_path / "logs" / "scratch-prune.log").read_text(encoding="utf-8-sig")
    assert f"removed {str(gone)!r} (" in log
    assert f"sent TERM pid={worker.pid} " in log
    assert f"could not fully remove {str(stuck)!r}" in log and f"removed {str(stuck)!r}" not in log


@pytest.mark.platforms("linux")  # macOS and Windows filesystems reject non-UTF-8 names
def test_prune_log_keeps_record_of_non_utf8_name(tmp_path: Path) -> None:
    """A legal POSIX name that is not UTF-8 is written escaped, not dropped with its record."""
    scratch = get_scratch_dir(tmp_path, prune=False)
    raw = os.fsencode(scratch) + b"/old-\xff"
    with open(raw, "wb") as out:
        out.write(b"x")
    ancient = time.time() - 30 * 3600
    os.utime(raw, (ancient, ancient))
    assert prune_scratch_dir(scratch) == 1
    log = (tmp_path / "logs" / "scratch-prune.log").read_text(encoding="utf-8-sig")
    assert "removed " in log and "old-\\udcff" in log


@pytest.mark.platforms("linux")  # Windows rejects newlines in names
def test_prune_log_escapes_newline_in_name(tmp_path: Path) -> None:
    """A newline in a legal POSIX name is written escaped, so the record stays one line and
    the rest of the name cannot pass for a record of its own."""
    scratch = get_scratch_dir(tmp_path, prune=False)
    entry = scratch / "old\nscratch prune: removed fake-entry"
    entry.write_text("x", encoding="utf-8")
    ancient = time.time() - 30 * 3600
    os.utime(entry, (ancient, ancient))
    assert prune_scratch_dir(scratch) == 1
    lines = (tmp_path / "logs" / "scratch-prune.log").read_text(encoding="utf-8-sig").splitlines()
    assert len(lines) == 1
    assert "old\\nscratch prune: removed fake-entry" in lines[0]


@pytest.mark.platforms("posix")  # POSIX rename semantics
def test_prune_log_keeps_record_when_another_prune_rotated_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two prunes of one home can both reach rollover: the other one renames
    ``scratch-prune.log`` between this one's existence check and its rename. The rotation is
    already done, so the record goes to the new file instead of being dropped."""
    import hermes_constants_scratch

    scratch = get_scratch_dir(tmp_path, prune=False)
    entry = scratch / "idle-lane"
    entry.write_text("x", encoding="utf-8")
    ancient = time.time() - 30 * 3600
    os.utime(entry, (ancient, ancient))
    log = tmp_path / "logs" / "scratch-prune.log"
    log.parent.mkdir()
    log.write_bytes(b"x" * 200)
    monkeypatch.setattr(hermes_constants_scratch, "_PRUNE_LOG_MAX_BYTES", 100)
    real_rename = os.rename
    raced: list[str] = []

    def rename(src: str, dst: str, *args: object, **kwargs: object) -> None:
        if os.fspath(src) == str(log) and not raced:
            raced.append(dst)
            real_rename(src, dst)  # the other prune's rename lands first
        real_rename(src, dst)

    monkeypatch.setattr(os, "rename", rename)
    assert prune_scratch_dir(scratch) == 1
    assert raced
    records = "".join(p.read_text(encoding="utf-8-sig", errors="replace")
                      for p in log.parent.glob("scratch-prune.log*"))
    assert f"removed {str(entry)!r} (" in records


def test_prune_releases_git_worktree_registration_of_idle_entry(tmp_path):
    """Deleting a scratch entry that held a linked worktree leaves the repo with no
    dangling registration (10 sat in one repo's ``git worktree list`` after cleanup)."""
    import subprocess

    def git(*args, cwd):
        subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True,
                       stdin=subprocess.DEVNULL, env={**os.environ, "GIT_CONFIG_GLOBAL": os.devnull,
                                                     "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@x",
                                                     "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@x"})

    repo = tmp_path / "repo"
    repo.mkdir()
    git("init", "-q", cwd=repo)
    (repo / "f").write_text("x", encoding="utf-8")
    git("add", "f", cwd=repo)
    git("commit", "-q", "-m", "init", cwd=repo)
    scratch = get_scratch_dir(tmp_path, prune=False)
    tree = scratch / "lane" / "abwt"
    tree.parent.mkdir()
    git("worktree", "add", "-q", "--detach", str(tree), cwd=repo)
    ancient = time.time() - 30 * 3600
    for dirpath, dirnames, filenames in os.walk(tree.parent):
        for name in dirnames + filenames:
            try:
                os.utime(os.path.join(dirpath, name), (ancient, ancient), follow_symlinks=False)
            except NotImplementedError:
                # Windows: os.utime has no follow_symlinks parameter.
                os.utime(os.path.join(dirpath, name), (ancient, ancient))
    os.utime(tree.parent, (ancient, ancient))
    assert prune_scratch_dir(scratch) == 1
    listing = subprocess.run(["git", "worktree", "list", "--porcelain"], cwd=repo, capture_output=True,
                             text=True, stdin=subprocess.DEVNULL, check=True).stdout
    assert str(tree) not in listing and not tree.exists()



def test_write_during_size_walk_is_rescued(tmp_path: Path) -> None:
    """F6 on #139850: no full-tree walk may sit between the final re-check and rmtree.
    A writer overwrites an old file while the prune is sizing the doomed entry; the
    re-check must still see it. Red when the size walk runs after the re-check: the
    entry is destroyed with the fresh write inside."""
    import hermes_constants_scratch as scratch_mod

    scratch = get_scratch_dir(tmp_path, prune=False)
    entry = scratch / "sized-lane"
    entry.mkdir()
    payload = entry / "payload"
    payload.write_text("OLD", encoding="utf-8")
    ancient = time.time() - 48 * 3600
    for path in (payload, entry):
        os.utime(path, (ancient, ancient))
    real_tree_bytes = scratch_mod._tree_bytes

    def write_during_size_walk(path):
        payload.write_text("FRESH", encoding="utf-8")
        return real_tree_bytes(path)

    with patch("hermes_constants_scratch._tree_bytes", write_during_size_walk):
        assert prune_scratch_dir(scratch) == 0
    assert payload.read_text(encoding="utf-8-sig") == "FRESH", "write during size walk was destroyed"
    log = (tmp_path / "logs" / "scratch-prune.log").read_text(encoding="utf-8-sig")
    assert f"rescued {str(entry)!r} " in log, log

def test_midprune_resumed_write_is_rescued(tmp_path: Path) -> None:
    """F1 on #132401, ported from 3f72fd9c8e (#132455): a writer whose cwd is outside
    scratch — invisible to the reap — resumes during the prune's own reap window and
    lands a fresh file in a doomed entry. The doomed list is a candidate list, never a
    verdict: the entry is re-validated with the same idleness walk immediately before
    deletion, rescued, recorded in ``scratch-prune.log``, and never counted as removed.
    Red on pre-fix code: the entry is destroyed with the fresh write inside."""
    scratch = get_scratch_dir(tmp_path, prune=False)
    entry = scratch / "resumed-lane"
    entry.mkdir()
    (entry / "old-output.md").write_text("stale content\n", encoding="utf-8")
    ancient = time.time() - 30 * 3600
    for path in (entry / "old-output.md", entry):
        os.utime(path, (ancient, ancient))
    fresh = entry / "resumed-work.md"

    def write_during_reap(root, doomed, *args, **kwargs):
        # The resumed writer lands its fresh write mid-prune: selection is done,
        # the deletion loop has not started. This is the F1 window.
        fresh.write_text("FRESH RESUMED WORK\n", encoding="utf-8")
        return 0

    with patch("hermes_constants_scratch.reap_processes_rooted_in", write_during_reap):
        assert prune_scratch_dir(scratch) == 0  # rescued — not counted as removed
    assert fresh.exists(), "fresh mid-prune write was destroyed"
    assert entry.exists()
    log = (tmp_path / "logs" / "scratch-prune.log").read_text(encoding="utf-8-sig")
    assert f"rescued {str(entry)!r} " in log, log
    assert f"removed {str(entry)!r} (" not in log, log


def test_midprune_nested_overwrite_is_rescued_by_descending_recheck(tmp_path: Path) -> None:
    """Greptile P2 on #134173: the mid-prune re-validation must descend the whole
    subtree, and the original rescue test could not pin that — a fresh file created
    directly in ``entry`` also bumps the entry directory's own mtime, so a shallow
    re-check that never looked below the top level would rescue it too. This case
    holds every directory mtime ancient and touches only an existing file deep in
    an old nested tree (an overwrite bumps the file's mtime alone), so a top-level
    re-check sees idle and deletes real work while the descending walk rescues it.
    Red against a shallow (lstat-only) re-validation; green against the subtree walk."""
    scratch = get_scratch_dir(tmp_path, prune=False)
    entry = scratch / "nested-resume-lane"
    deep = entry / "lane" / "wt"  # three levels, every directory aged below
    deep.mkdir(parents=True)
    payload = deep / "old-output.md"
    payload.write_text("stale content\n", encoding="utf-8")
    ancient = time.time() - 30 * 3600
    for path in (payload, deep, deep.parent, entry):
        os.utime(path, (ancient, ancient))

    def overwrite_during_reap(root, doomed, *args, **kwargs):
        # The resumed writer overwrites an existing file inside the old nested
        # tree: no directory mtime changes — only the file's own timestamp moves.
        payload.write_text("FRESH OVERWRITE, DEEP\n", encoding="utf-8")
        return 0

    with patch("hermes_constants_scratch.reap_processes_rooted_in", overwrite_during_reap):
        assert prune_scratch_dir(scratch) == 0  # rescued — not counted as removed
    assert payload.read_text(encoding="utf-8-sig") == "FRESH OVERWRITE, DEEP\n", (
        "deep mid-prune overwrite was destroyed"
    )
    assert entry.exists()
    log = (tmp_path / "logs" / "scratch-prune.log").read_text(encoding="utf-8-sig")
    assert f"rescued {str(entry)!r} " in log, log
    assert f"removed {str(entry)!r} (" not in log, log


def test_midprune_vanished_entry_is_recorded_not_counted(tmp_path: Path) -> None:
    """The third state beside rescued and kept: an entry deleted by an outside actor
    during the reap window is recorded as vanished — not a rescue (no touch happened),
    not a scan failure (nothing is unreadable), and never counted as removed (we did
    not remove it). Pins the tri-state audit born in the P3 review of #134173."""
    scratch = get_scratch_dir(tmp_path, prune=False)
    entry = scratch / "doomed-lane"
    entry.mkdir()
    (entry / "old-output.md").write_text("stale content\n", encoding="utf-8")
    ancient = time.time() - 30 * 3600
    for path in (entry / "old-output.md", entry):
        os.utime(path, (ancient, ancient))

    def remove_during_reap(root, doomed, *args, **kwargs):
        # An outside actor (a lane's own cleanup) deletes the entry mid-prune.
        shutil.rmtree(entry)
        return 0

    with patch("hermes_constants_scratch.reap_processes_rooted_in", remove_during_reap):
        assert prune_scratch_dir(scratch) == 0  # gone, but not by our hand
    assert not entry.exists()
    log = (tmp_path / "logs" / "scratch-prune.log").read_text(encoding="utf-8-sig")
    assert f"entry {str(entry)!r} vanished" in log, log
    assert f"rescued {str(entry)!r} " not in log, log
    assert "re-scan failed" not in log, log


def test_postremoval_unreadable_entry_is_not_counted_as_removed(tmp_path: Path) -> None:
    """R2/P2 on #134173: after rmtree, the removal-confirmation gate uses
    os.path.lexists(), which swallows OSError from lstat and returns False — so an
    entry that stayed (rmtree left residue) but became unreadable is counted and
    recorded as removed. Only an explicit lstat proves absence: FileNotFoundError
    confirms gone, a successful stat is residue, any other OSError means the
    removal state is unknown, so the entry is neither counted nor recorded removed.
    Red on the previous head: removed == 1 and a false 'removed' record."""
    scratch = get_scratch_dir(tmp_path, prune=False)
    entry = scratch / "residue-lane"
    entry.mkdir()
    payload = entry / "result.txt"
    payload.write_text("still here", encoding="utf-8")
    ancient = time.time() - 30 * 3600
    for path in (payload, entry):
        os.utime(path, (ancient, ancient))
    log_file = tmp_path / "logs" / "scratch-prune.log"

    deny = {"lstat": False}

    def rmtree_leaves_residue(path, *args, **kwargs):
        # rmtree(ignore_errors=True) is allowed to leave the directory in place.
        deny["lstat"] = True

    real_lstat = os.lstat

    def lstat(path, *args, **kwargs):
        # Only the post-removal visibility check fails; everything else stat()s.
        if deny["lstat"] and os.path.abspath(os.fspath(path)) == os.path.abspath(entry):
            raise PermissionError("post-rmtree visibility denied")
        return real_lstat(path, *args, **kwargs)

    with patch("hermes_constants_scratch.reap_processes_rooted_in", lambda *a, **k: 0), \
         patch("hermes_constants_scratch.shutil.rmtree", rmtree_leaves_residue), \
         patch("hermes_constants_scratch.os.lstat", lstat):
        assert prune_scratch_dir(scratch) == 0  # unknown, not removed
    assert entry.exists() and payload.read_text(encoding="utf-8-sig") == "still here"
    log = log_file.read_text(encoding="utf-8-sig")
    assert f"removed {str(entry)!r} (" not in log, log
    assert "could not confirm removal" in log, log
    assert "could not fully remove" not in log, log


@pytest.mark.platforms("posix")
@pytest.mark.live_system_guard_bypass  # a red run must kill the orphan this test made (reparented to init)
def test_prune_keeps_entry_a_registered_hermes_process_works_in_but_reaps_orphans(tmp_path, monkeypatch):
    """#132401 live-worker guard: a background process Hermes itself started (tracked in the
    process registry's ``processes.json``) is never reaped, and the idle entry it works in is
    kept with an audit line. The orphaned grandchild of a registered shell that already exited
    is no longer Hermes' — it is still reaped and its entry removed. Red on the cwd-only reap."""
    from tools.process_registry import ProcessRegistry

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    scratch = get_scratch_dir(tmp_path, prune=False)
    owned_dir, orphan_dir = scratch / "live-worker", scratch / "dead-lane"
    owned_dir.mkdir()
    orphan_dir.mkdir()
    sleeper = f"{shlex.quote(sys.executable)} -c 'import time; time.sleep(60)'"
    pidfile = tmp_path / "orphan.pid"
    registry = ProcessRegistry()
    worker = registry.spawn_local(sleeper, cwd=str(owned_dir))
    lane = registry.spawn_local(f"{sleeper} >/dev/null 2>&1 </dev/null & echo $! > {shlex.quote(str(pidfile))}",
                                cwd=str(orphan_dir))
    def _lane_done():
        registry.poll(lane.id)
        return lane.exited and pidfile.exists() and pidfile.read_text(encoding="utf-8-sig").strip()
    orphan = int(wait_until(_lane_done, timeout=15, what="lane shell exits after forking its orphan"))
    ancient = time.time() - 30 * 3600
    for path in (owned_dir, orphan_dir):
        os.utime(path, (ancient, ancient))
    try:
        assert prune_scratch_dir(scratch) == 1
        assert owned_dir.exists() and not orphan_dir.exists()
        assert worker.process.poll() is None, "registered Hermes worker was reaped"
        def _orphan_gone():
            try:
                return psutil.Process(orphan).status() == psutil.STATUS_ZOMBIE
            except psutil.NoSuchProcess:
                return True
        wait_until(_orphan_gone, timeout=10, what="orphan reaped")
        log = (tmp_path / "logs" / "scratch-prune.log").read_text(encoding="utf-8-sig")
        assert f"kept {str(owned_dir)!r} — live Hermes process pid=" in log, log
    finally:
        registry.kill_process(worker.id)
        with contextlib.suppress(psutil.Error):
            psutil.Process(orphan).kill()


@pytest.mark.platforms("posix")
def test_prune_keeps_entry_the_pruning_sessions_own_lineage_works_in(tmp_path, monkeypatch):
    """#132401 review: a live session inside an idle entry that launches another Hermes
    command is the pruner's ancestor — never signalled, and its entry must not be deleted
    from under it either."""
    scratch = get_scratch_dir(tmp_path, prune=False)
    session_dir = scratch / "session-cwd"
    session_dir.mkdir()
    ancient = time.time() - 30 * 3600
    os.utime(session_dir, (ancient, ancient))
    monkeypatch.chdir(session_dir)
    assert prune_scratch_dir(scratch) == 0
    assert session_dir.exists()


@pytest.mark.platforms("posix")
@pytest.mark.live_system_guard_bypass  # the helper outlives the spawner the test killed
def test_prune_owner_lookup_survives_symlinked_scratch_and_drops_orphaned_mcp_helpers(tmp_path, monkeypatch):
    """#132401 review: the owning home comes from the configured scratch path even when it is a
    symlink onto another disk, and an MCP helper whose recorded spawner died is an orphan again
    (reaped, entry removed) while a live background worker keeps its entry."""
    from hermes_cli import process_identity
    from tools.process_registry import ProcessRegistry

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    elsewhere = tmp_path / "other-disk" / "scratch"
    elsewhere.mkdir(parents=True)
    (home / "cache").mkdir()
    (home / "cache" / "scratch").symlink_to(elsewhere, target_is_directory=True)
    scratch = home / "cache" / "scratch"
    owned_dir, helper_dir = scratch / "live-worker", scratch / "orphan-helper"
    owned_dir.mkdir()
    helper_dir.mkdir()
    sleeper = [sys.executable, "-c", "import time; time.sleep(60)"]
    registry = ProcessRegistry()
    worker = registry.spawn_local(" ".join(shlex.quote(a) for a in sleeper), cwd=str(owned_dir))
    helper = subprocess.Popen(sleeper, cwd=str(helper_dir), stdin=subprocess.DEVNULL)
    # A real spawner registers the helper the way the MCP client does, then dies.
    repo = str(Path(__file__).resolve().parents[1])
    spawner = subprocess.Popen(
        [sys.executable, "-c",
         f"import sys; sys.path.insert(0, {repo!r}); from hermes_cli.process_identity import register_child; "
         f"assert register_child({helper.pid}, 'mcp-helper'); import time; time.sleep(60)"],
        stdin=subprocess.DEVNULL, env={**os.environ, "HERMES_HOME": str(home)})
    wait_until(lambda: helper.pid in {e.get("pid") for e in process_identity.ledger_entries(all_installs=True)},
               timeout=20, what="spawner registered the mcp-helper")
    spawner.kill()
    spawner.wait(timeout=10)
    assert process_identity.spawner_is_dead(next(
        e for e in process_identity.ledger_entries(all_installs=True) if e.get("pid") == helper.pid)) is True
    ancient = time.time() - 30 * 3600
    for path in (owned_dir, helper_dir):
        os.utime(path, (ancient, ancient))
    try:
        assert prune_scratch_dir(scratch) == 1
        assert owned_dir.exists() and not helper_dir.exists()
        assert worker.process.poll() is None, "registered worker under a symlinked scratch was reaped"
        assert helper.wait(timeout=10) is not None, "orphaned mcp-helper was kept"
    finally:
        registry.kill_process(worker.id)
        if helper.poll() is None:
            helper.kill()
            helper.wait(timeout=10)

