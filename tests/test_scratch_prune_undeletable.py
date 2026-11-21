"""An idle scratch entry this user cannot read or delete is reported once and not re-walked hourly.

A tree this user sealed read-only itself (0555 dirs, 0444 files: a release builder's published
generation under test) is not "undeletable": the prune unseals it (u+rwx, own dirs inside the
entry only, symlinks never followed) and removes it on the first pass. Only what another uid
owns keeps the record/skip/weekly re-check behaviour.

Without root a real foreign-owned dir cannot be made, so :func:`_foreign` simulates one: it marks
chosen own dirs as owned by someone else for the ownership check, and the dir's own read-only
mode then produces the same EACCES a root-owned container write does.
"""

import json
import os
import re
import stat
import time

import pytest

import hermes_constants_scratch as scratch
import hermes_constants_scratch_stuck as stuck

pytestmark = [
    pytest.mark.platforms("posix"),
    pytest.mark.skipif(
        not hasattr(os, "getuid") or os.getuid() == 0,
        reason="needs a non-root user: root deletes through a read-only dir",
    ),
]

OLD = time.time() - 48 * 3600
# Today's record line for a blocker another uid owns, kept byte for byte (an optional
# "(or without sudo: docker run ...)" suffix follows when docker is installed).
FOREIGN_LINE = (
    "scratch prune: cannot remove {entry!r}: it holds files this user (uid {uid}) may not read or "
    "delete, usually root-owned output of a Docker container; skipping it until that is "
    "fixed (re-checked weekly). Remove it with: sudo rm -rf {entry}"
)


def _foreign(monkeypatch, *paths):
    """Treat *paths* as owned by another uid in the prune's ownership check (by inode, so a
    rename cannot dodge it). ``raising=False``: the seam does not exist before the fix, where
    every EACCES already counts as foreign."""
    marked = {(st.st_dev, st.st_ino) for st in (os.lstat(p) for p in paths)}

    def owned(st):
        return (st.st_dev, st.st_ino) not in marked and st.st_uid == os.geteuid()

    monkeypatch.setattr(stuck, "_owned_by_this_user", owned, raising=False)


def _age(*paths):
    for p in paths:
        os.utime(p, (OLD, OLD), follow_symlinks=False)


def _undeletable_entry(root, name="lane"):
    """``<name>/ro/blob`` with ``ro`` mode 0500: rmtree removes everything else and fails
    on ``blob`` with EACCES, the same residue a root-owned container write leaves (once
    ``ro`` is marked foreign with :func:`_foreign`)."""
    entry = root / name
    ro = entry / "ro"
    ro.mkdir(parents=True)
    (entry / "deletable.txt").write_text("x", encoding="utf-8")
    (ro / "blob").write_text("left by a container", encoding="utf-8")
    _age(ro / "blob", entry / "deletable.txt", ro, entry)
    ro.chmod(stat.S_IRUSR | stat.S_IXUSR)
    return entry, ro


def _sealed_generation(entry):
    """An immutable release generation as builders publish it: every dir 0555, every file
    0444, the entry's top dir included. Returns the dirs (deepest first)."""
    gen = entry / "release" / ".generations" / "g-20261113T181142Z-1-abc" / "OneilBudget-Mac"
    for sub in ("rollback-0.9.1", "rollback-0.10.7", "wheels"):
        (gen / sub).mkdir(parents=True)
        (gen / sub / "payload.bin").write_bytes(b"x" * 64)
    (gen / "manifest.json").write_text("{}", encoding="utf-8")
    paths = sorted(entry.rglob("*"), key=lambda p: -len(p.parts)) + [entry]
    _age(*paths)
    dirs = []
    for p in paths:
        if p.is_dir():
            p.chmod(0o555)
            dirs.append(p)
        else:
            p.chmod(0o444)
    return dirs


def _open_up(*tops):
    """Test cleanup: give back u+rwx on whatever is left (symlinks never followed)."""
    for top in tops:
        if not os.path.lexists(top) or os.path.islink(top):
            continue
        os.chmod(top, 0o700)
        for dirpath, dirnames, _ in os.walk(top):
            for d in dirnames:
                full = os.path.join(dirpath, d)
                if not os.path.islink(full):
                    os.chmod(full, 0o700)


def _walk_spy(monkeypatch, root):
    """Every directory listing under *root* other than root itself (idle scan, size walk, rmtree)."""
    walked = []
    real_scandir = os.scandir

    def scandir(path=".", *a, **k):
        if isinstance(path, (str, os.PathLike)) and os.fspath(path).startswith(str(root) + os.sep):
            walked.append(os.fspath(path))
        return real_scandir(path, *a, **k)

    monkeypatch.setattr(scratch.os, "scandir", scandir)
    monkeypatch.setattr(scratch, "reap_processes_rooted_in", lambda *_a, **_k: 0)
    return walked


def _lines(log_file, entry):
    if not log_file.exists():
        return []
    return [ln for ln in log_file.read_text(encoding="utf-8-sig").splitlines() if repr(str(entry)) in ln]


def _walked(walked, entry):
    return any(p == str(entry) or p.startswith(str(entry) + os.sep) for p in walked)


def _state(root):
    path = root / stuck.STUCK_STATE_NAME
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {}


def _mode(path):
    return stat.S_IMODE(os.lstat(path).st_mode)


# --- today's behaviour for foreign-owned residue (kept) ------------------------------------


def test_undeletable_or_unreadable_entry_logged_once_then_skipped_without_walk(tmp_path, monkeypatch):
    root = tmp_path / "scratch"
    entry, ro = _undeletable_entry(root)
    # A dir this user may not even list (root-owned 0700): the idle scan itself fails.
    sealed = root / "sealed"
    locked = sealed / "locked"
    locked.mkdir(parents=True)
    (locked / "blob").write_text("unreadable", encoding="utf-8")
    for p in (locked / "blob", locked, sealed):
        os.utime(p, (OLD, OLD))
    locked.chmod(0)
    _foreign(monkeypatch, ro, locked)
    log_file = tmp_path / "logs" / "scratch-prune.log"
    walked = _walk_spy(monkeypatch, root)
    try:
        assert scratch.prune_idle_entries(root, 24, frozenset(), log_file) == 0
        assert _walked(walked, entry) and _walked(walked, sealed)
        walked.clear()
        # A day later: pass 1's partial delete no longer counts as recent activity, so on
        # main both are walked again (and the sealed one never gets a hint) every hour.
        later = time.time() + 48 * 3600
        monkeypatch.setattr(scratch.time, "time", lambda: later)
        assert scratch.prune_idle_entries(root, 24, frozenset(), log_file) == 0
        for e in (entry, sealed):
            assert not _walked(walked, e), f"a known-stuck entry must not be re-walked every pass: {e}"
            lines = _lines(log_file, e)
            assert len(lines) == 1, lines
            assert f"sudo rm -rf {e}" in lines[0], lines[0]
        assert (ro / "blob").exists()
    finally:
        ro.chmod(0o700)
        locked.chmod(0o700)


def test_fixed_expired_or_other_uid_entry_is_judged_again(tmp_path, monkeypatch):
    root = tmp_path / "scratch"
    entry, ro = _undeletable_entry(root)
    other, other_ro = _undeletable_entry(root, "other")
    _foreign(monkeypatch, ro, other_ro)
    log_file = tmp_path / "logs" / "scratch-prune.log"
    walked = _walk_spy(monkeypatch, root)
    now = time.time()
    try:
        assert scratch.prune_idle_entries(root, 24, frozenset(), log_file) == 0
        # The user fixes the nested dir only; the top dir's own metadata is not touched.
        top_before = os.lstat(entry)
        ro.chmod(0o700)
        assert os.lstat(entry).st_ctime_ns == top_before.st_ctime_ns
        walked.clear()
        monkeypatch.setattr(scratch.time, "time", lambda: now + 48 * 3600)
        assert scratch.prune_idle_entries(root, 24, frozenset(), log_file) == 1
        assert not entry.exists()
        assert not _walked(walked, other)
        # Still stuck after a week: re-checked once (bounded retry) and reported again.
        monkeypatch.setattr(scratch.time, "time", lambda: now + 9 * 24 * 3600)
        assert scratch.prune_idle_entries(root, 24, frozenset(), log_file) == 0
        assert _walked(walked, other) and len(_lines(log_file, other)) == 2
        # A record made under another uid is not trusted either.
        state_file = root / ".prune_stuck.json"
        state = json.loads(state_file.read_text(encoding="utf-8-sig"))
        state["other"]["uid"] += 1  # a record made by another uid
        state_file.write_text(json.dumps(state), encoding="utf-8")
        walked.clear()
        assert scratch.prune_idle_entries(root, 24, frozenset(), log_file) == 0
        assert _walked(walked, other)
        other_ro.chmod(0o700)
        monkeypatch.setattr(scratch.time, "time", lambda: now + 11 * 24 * 3600)
        assert scratch.prune_idle_entries(root, 24, frozenset(), log_file) == 1
        assert not state_file.exists()
    finally:
        for d in (ro, other_ro):
            if d.exists():
                d.chmod(0o700)


# --- (a) a tree this user sealed read-only is unsealed and removed on the first pass -------


def test_a_own_sealed_tree_removed_on_first_prune_without_stuck_record(tmp_path, monkeypatch):
    root = tmp_path / "scratch"
    entry = root / "k4fix-rel-abc"
    dirs = _sealed_generation(entry)
    log_file = tmp_path / "logs" / "scratch-prune.log"
    _walk_spy(monkeypatch, root)
    try:
        assert scratch.prune_idle_entries(root, 24, frozenset(), log_file) == 1
        assert not os.path.lexists(entry)
        assert _state(root) == {}
        assert not (root / stuck.STUCK_STATE_NAME).exists()
        lines = _lines(log_file, entry)
        assert not any("root-owned" in ln or "cannot remove" in ln for ln in lines), lines
        unsealed = [ln for ln in lines if "scratch prune: unsealed and removed" in ln]
        assert len(unsealed) == 1, lines
        assert (
            f"scratch prune: unsealed and removed {str(entry)!r}: {len(dirs)} read-only dir(s) "
            f"sealed by this user (uid {os.geteuid()}) were made writable (u+rwx) so it could be deleted"
        ) in unsealed[0]
        assert any(re.search(r"scratch prune: removed .* \(\d+ bytes\)$", ln) for ln in lines), lines
    finally:
        _open_up(entry)


def test_a_own_unopenable_subdir_is_unsealed_by_the_rmtree_handler(tmp_path):
    """rmtree cannot even open a mode-0 dir of this user (a race the idle scan cannot see):
    the handler unseals it, walks it, and reports no blocker."""
    entry = tmp_path / "scratch" / "lane"
    shut = entry / "a" / "shut"
    (shut / "inner").mkdir(parents=True)
    (shut / "inner" / "f").write_text("x", encoding="utf-8")
    (entry / "a").chmod(0o555)
    shut.chmod(0)
    unsealed = []
    try:
        assert stuck.rmtree_collecting_permission_errors(entry, unsealed=unsealed) == []
        assert not os.path.lexists(entry)
        assert str(shut) in unsealed and str(entry / "a") in unsealed
    finally:
        _open_up(entry)


# --- (b) a foreign-owned blocker is still recorded and skipped exactly as today -------------


def test_b_foreign_blocker_recorded_and_skipped_exactly_as_today(tmp_path, monkeypatch):
    root = tmp_path / "scratch"
    entry, ro = _undeletable_entry(root)
    _foreign(monkeypatch, ro)
    log_file = tmp_path / "logs" / "scratch-prune.log"
    walked = _walk_spy(monkeypatch, root)
    try:
        assert scratch.prune_idle_entries(root, 24, frozenset(), log_file) == 0
        # Nothing of another uid's is chmod'ed: the blocker keeps its mode and its file.
        assert _mode(ro) == 0o500
        assert (ro / "blob").read_text(encoding="utf-8") == "left by a container"
        rec = _state(root)["lane"]
        assert rec["blockers"] == [str(ro)]
        assert rec["uid"] == os.geteuid()
        assert rec["fp"] == stuck._fingerprint(entry)
        lines = _lines(log_file, entry)
        assert len(lines) == 1, lines
        assert FOREIGN_LINE.format(entry=str(entry), uid=os.geteuid()) in lines[0], lines[0]
        # Skipped (not re-walked) on the next pass, and not logged again.
        walked.clear()
        later = time.time() + 48 * 3600
        monkeypatch.setattr(scratch.time, "time", lambda: later)
        assert scratch.prune_idle_entries(root, 24, frozenset(), log_file) == 0
        assert not _walked(walked, entry)
        assert len(_lines(log_file, entry)) == 1
    finally:
        ro.chmod(0o700)


# --- (c) symlinks are never followed: a read-only target outside scratch is untouched ------


def _readonly_target(tmp_path):
    target = tmp_path / "outside" / "published"
    (target / "sub").mkdir(parents=True)
    (target / "keep.txt").write_text("do not touch", encoding="utf-8")
    (target / "sub" / "keep2.txt").write_text("nor this", encoding="utf-8")
    for f in (target / "keep.txt", target / "sub" / "keep2.txt"):
        f.chmod(0o444)
    (target / "sub").chmod(0o555)
    target.chmod(0o555)
    return target


def _snapshot(target):
    """Mode, ctime (moves on any chmod) and bytes of everything under *target*."""
    out = {}
    for dirpath, dirnames, filenames in os.walk(target):
        for name in [*dirnames, *filenames, ""]:
            p = os.path.join(dirpath, name) if name else dirpath
            st = os.lstat(p)
            if stat.S_ISREG(st.st_mode):
                with open(p, "rb") as f:
                    body = f.read()
            else:
                body = None
            out[os.path.relpath(p, target)] = (stat.S_IMODE(st.st_mode), st.st_ctime_ns, body)
    return out


def test_c_symlink_to_readonly_dir_outside_scratch_is_never_followed(tmp_path, monkeypatch):
    root = tmp_path / "scratch"
    target = _readonly_target(tmp_path)
    before = _snapshot(target)
    entry = root / "lane"
    sealed = entry / "sealed"
    sealed.mkdir(parents=True)
    (sealed / "link").symlink_to(target, target_is_directory=True)
    (entry / "toplink").symlink_to(target, target_is_directory=True)
    (sealed / "f").write_text("x", encoding="utf-8")
    _age(sealed / "link", entry / "toplink", sealed / "f", sealed, entry)
    sealed.chmod(0o555)
    entry.chmod(0o555)
    log_file = tmp_path / "logs" / "scratch-prune.log"
    _walk_spy(monkeypatch, root)
    try:
        assert scratch.prune_idle_entries(root, 24, frozenset(), log_file) == 1
        assert not os.path.lexists(entry)
        assert _snapshot(target) == before, "the symlink target's modes or contents changed"
        assert _state(root) == {}
    finally:
        _open_up(entry)
        _open_up(target)


def test_c_unseal_refuses_symlinked_components_and_paths_outside_the_entry(tmp_path):
    target = _readonly_target(tmp_path)
    before = _snapshot(target)
    entry = tmp_path / "scratch" / "lane"
    entry.mkdir(parents=True)
    (entry / "link").symlink_to(target, target_is_directory=True)
    try:
        # Through a symlinked component, at the symlink itself, and outside the entry.
        assert stuck._unseal_around(str(entry), str(entry / "link" / "sub" / "keep2.txt")) == []
        assert stuck._unseal_around(str(entry), str(entry / "link" / "sub")) == []
        assert stuck._unseal_around(str(entry), str(entry / "link")) == []
        assert stuck._unseal_around(str(entry), str(target / "sub")) == []
        assert stuck._unseal_around(str(entry), str(entry) + "-sibling") == []
        assert stuck._unseal_around(str(entry), str(entry / ".." / "x")) == []
        assert _snapshot(target) == before
        # A sibling entry whose name extends this one's ("lane" / "lane-sibling") is outside it,
        # even when the entry holds a dir named like the suffix ("lane/-sibling").
        sibling = entry.parent / (entry.name + "-sibling")
        decoy = entry / "-sibling"
        for d in (sibling, decoy):
            d.mkdir()
            d.chmod(0o555)
        try:
            assert stuck._unseal_around(str(entry), str(sibling)) == []
            assert _mode(sibling) == 0o555 and _mode(decoy) == 0o555
        finally:
            for d in (sibling, decoy):
                d.chmod(0o700)
        # The scratch root (the entry's parent) is never chmod'ed either.
        entry.parent.chmod(0o555)
        try:
            assert stuck._unseal_around(str(entry), str(entry)) == []
            assert _mode(entry.parent) == 0o555
        finally:
            entry.parent.chmod(0o700)
    finally:
        _open_up(target)


# --- (d) mixed trees: own parts unsealed; recorded only if foreign blockers remain ---------


def _mixed_entry(root):
    entry = root / "mixed"
    dirs = _sealed_generation(entry)  # own, sealed (top included)
    foreign = entry / "docker-out"
    entry.chmod(0o755)
    foreign.mkdir()
    (foreign / "blob").write_text("root wrote this", encoding="utf-8")
    _age(foreign / "blob", foreign, entry)
    foreign.chmod(0o500)
    entry.chmod(0o555)
    return entry, foreign, dirs


def test_d_mixed_tree_own_parts_unsealed_and_recorded_for_the_foreign_blocker(tmp_path, monkeypatch):
    root = tmp_path / "scratch"
    entry, foreign, _dirs = _mixed_entry(root)
    _foreign(monkeypatch, foreign)
    log_file = tmp_path / "logs" / "scratch-prune.log"
    walked = _walk_spy(monkeypatch, root)
    try:
        assert scratch.prune_idle_entries(root, 24, frozenset(), log_file) == 0
        # Own parts: unsealed and gone. Foreign part: untouched, mode and file.
        assert sorted(os.listdir(entry)) == ["docker-out"]
        assert _mode(entry) & stat.S_IRWXU == stat.S_IRWXU
        assert _mode(foreign) == 0o500
        assert (foreign / "blob").read_text(encoding="utf-8") == "root wrote this"
        assert _state(root)["mixed"]["blockers"] == [str(foreign)]
        lines = _lines(log_file, entry)
        assert len(lines) == 2, lines
        assert (
            f"scratch prune: unsealed {len(_dirs)} read-only dir(s) sealed by this user "
            f"(uid {os.geteuid()}) in {str(entry)!r}; what is left is owned by another user"
        ) in lines[0], lines[0]
        assert FOREIGN_LINE.format(entry=str(entry), uid=os.geteuid()) in lines[1], lines[1]
        walked.clear()
        later = time.time() + 48 * 3600
        monkeypatch.setattr(scratch.time, "time", lambda: later)
        assert scratch.prune_idle_entries(root, 24, frozenset(), log_file) == 0
        assert not _walked(walked, entry)
    finally:
        _open_up(entry)


def test_d_mixed_tree_with_no_foreign_content_left_is_removed_not_recorded(tmp_path, monkeypatch):
    """A foreign-owned dir that is empty goes once its own sealed parent is unsealed: with no
    foreign blocker left, nothing is recorded."""
    root = tmp_path / "scratch"
    entry, foreign, _dirs = _mixed_entry(root)
    foreign.chmod(0o700)
    (foreign / "blob").unlink()
    _age(foreign)
    foreign.chmod(0o500)
    _foreign(monkeypatch, foreign)
    log_file = tmp_path / "logs" / "scratch-prune.log"
    _walk_spy(monkeypatch, root)
    try:
        assert scratch.prune_idle_entries(root, 24, frozenset(), log_file) == 1
        assert not os.path.lexists(entry)
        assert _state(root) == {}
        assert any("unsealed and removed" in ln for ln in _lines(log_file, entry))
    finally:
        _open_up(entry)


# --- records and lines stay honest about who owns the blocker ------------------------------


def test_legacy_record_for_an_own_sealed_tree_is_judged_again_at_once(tmp_path, monkeypatch):
    """Records filed before this fix called this user's own 0555 generations "root-owned" and
    would skip them for up to a week: such a record is dropped and the entry removed now."""
    root = tmp_path / "scratch"
    entry = root / "v010-build-rc-xyz"
    dirs = _sealed_generation(entry)
    root.joinpath(stuck.STUCK_STATE_NAME).write_text(json.dumps({
        entry.name: {
            "blockers": [str(d) for d in dirs[:2]],
            "fp": stuck._fingerprint(entry),
            "since": time.time(),
            "uid": os.geteuid(),
        },
    }), encoding="utf-8")
    log_file = tmp_path / "logs" / "scratch-prune.log"
    _walk_spy(monkeypatch, root)
    try:
        assert scratch.prune_idle_entries(root, 24, frozenset(), log_file) == 1
        assert not os.path.lexists(entry)
        assert _state(root) == {}
    finally:
        _open_up(entry)


def test_own_dir_the_scan_cannot_open_is_not_blamed_on_docker(tmp_path, monkeypatch):
    """A dir of this user's at mode 0 fails the idle scan (kept, fail-closed, as before), but
    the one line it gets names this user's own dir and a sudo-free fix, not root/Docker."""
    root = tmp_path / "scratch"
    entry = root / "lane"
    locked = entry / "locked"
    locked.mkdir(parents=True)
    (locked / "f").write_text("x", encoding="utf-8")
    _age(locked / "f", locked, entry)
    locked.chmod(0)
    log_file = tmp_path / "logs" / "scratch-prune.log"
    _walk_spy(monkeypatch, root)
    try:
        assert scratch.prune_idle_entries(root, 24, frozenset(), log_file) == 0
        lines = _lines(log_file, entry)
        assert len(lines) == 1, lines
        assert "root-owned" not in lines[0] and "Docker" not in lines[0], lines[0]
        assert f"owned by this user (uid {os.geteuid()})" in lines[0], lines[0]
        assert f"chmod -R u+rwx {entry} && rm -rf {entry}" in lines[0], lines[0]
        assert _state(root)["lane"]["blockers"] == [str(locked)]
    finally:
        locked.chmod(0o700)
