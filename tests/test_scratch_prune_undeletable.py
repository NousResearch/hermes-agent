"""An idle scratch entry this user cannot read or delete is reported once and not re-walked hourly."""

import json
import os
import stat
import time

import pytest

import hermes_constants_scratch as scratch

pytestmark = [
    pytest.mark.platforms("posix"),
    pytest.mark.skipif(
        not hasattr(os, "getuid") or os.getuid() == 0,
        reason="needs a non-root user: root deletes through a read-only dir",
    ),
]

OLD = time.time() - 48 * 3600


def _undeletable_entry(root, name="lane"):
    """``<name>/ro/blob`` with ``ro`` mode 0500: rmtree removes everything else and fails
    on ``blob`` with EACCES, the same residue a root-owned container write leaves."""
    entry = root / name
    ro = entry / "ro"
    ro.mkdir(parents=True)
    (entry / "deletable.txt").write_text("x", encoding="utf-8")
    (ro / "blob").write_text("left by a container", encoding="utf-8")
    for p in (ro / "blob", entry / "deletable.txt", ro, entry):
        os.utime(p, (OLD, OLD))
    ro.chmod(stat.S_IRUSR | stat.S_IXUSR)
    return entry, ro


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
    return [ln for ln in log_file.read_text(encoding="utf-8-sig").splitlines() if repr(str(entry)) in ln]


def _walked(walked, entry):
    return any(p == str(entry) or p.startswith(str(entry) + os.sep) for p in walked)


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
