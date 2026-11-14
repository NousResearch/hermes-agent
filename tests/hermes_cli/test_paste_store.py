"""Retention tests for the local collapsed-paste store (``$HERMES_HOME/pastes``)."""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_cli import paste_store


def _age(path, seconds):
    """Backdate a file's mtime by *seconds*."""
    stamp = time.time() - seconds
    os.utime(path, (stamp, stamp))


@pytest.fixture(autouse=True)
def _fresh_session_tracking():
    """Each test starts with an empty per-process paste record."""
    paste_store._session_pastes.clear()
    yield
    paste_store._session_pastes.clear()


def test_write_paste_lands_in_pastes_dir(tmp_path):
    path = paste_store.write_paste("hello\nworld", 3, home=tmp_path)
    assert path.parent == tmp_path / "pastes"
    assert path.name.startswith("paste_3_")
    assert path.suffix == ".txt"
    assert path.read_text(encoding="utf-8") == "hello\nworld"


def test_write_paste_never_clobbers_another_process(tmp_path):
    first = paste_store.write_paste("one", 1, home=tmp_path)
    second = paste_store.write_paste("two", 1, home=tmp_path)
    assert first != second
    assert first.read_text(encoding="utf-8") == "one"
    assert second.read_text(encoding="utf-8") == "two"


def test_sweep_deletes_only_expired_pastes(tmp_path):
    old = paste_store.write_paste("old", 1, home=tmp_path)
    fresh = paste_store.write_paste("fresh", 2, home=tmp_path)
    _age(old, 3 * 3600)

    deleted, remaining = paste_store.sweep_pastes(minutes=60, home=tmp_path)

    assert (deleted, remaining) == (1, 1)
    assert not old.exists()
    assert fresh.exists()


def test_sweep_leaves_debug_pending_json_alone(tmp_path):
    """``pending.json`` is ``hermes debug share`` state; it must survive every sweep."""
    directory = tmp_path / "pastes"
    directory.mkdir()
    pending = directory / paste_store.DEBUG_PENDING_NAME
    pending.write_text("[]", encoding="utf-8")
    _age(pending, 30 * 86400)
    old = paste_store.write_paste("old", 1, home=tmp_path)
    _age(old, 30 * 86400)

    deleted, _ = paste_store.sweep_pastes(minutes=60, home=tmp_path)

    assert deleted == 1
    assert pending.exists()
    assert not old.exists()


def test_retention_zero_keeps_everything(tmp_path):
    old = paste_store.write_paste("old", 1, home=tmp_path)
    _age(old, 90 * 86400)

    assert paste_store.sweep_pastes(minutes=0, home=tmp_path) == (0, 1)
    assert old.exists()


def test_purge_all_ignores_age(tmp_path):
    fresh = paste_store.write_paste("fresh", 1, home=tmp_path)
    old = paste_store.write_paste("old", 2, home=tmp_path)
    _age(old, 10 * 86400)

    deleted, remaining = paste_store.sweep_pastes(purge_all=True, home=tmp_path)

    assert (deleted, remaining) == (2, 0)
    assert not fresh.exists() and not old.exists()


def test_sweep_of_missing_dir_is_a_noop(tmp_path):
    assert paste_store.sweep_pastes(minutes=60, home=tmp_path) == (0, 0)


@pytest.mark.parametrize("raw,expected", [
    (60, 60.0),
    ("30", 30.0),
    (0, 0.0),
    (-5, 60.0),          # negative is nonsense → default, not "keep forever"
    ("nonsense", 60.0),
    (None, 60.0),
])
def test_retention_minutes_reads_config(raw, expected):
    assert paste_store.retention_minutes({"paste_retention_minutes": raw}) == expected


def test_retention_minutes_defaults_when_key_absent():
    assert paste_store.retention_minutes({}) == paste_store.DEFAULT_RETENTION_MINUTES


def test_maybe_sweep_is_throttled_but_still_sweeps_once(tmp_path, monkeypatch):
    monkeypatch.setattr(paste_store, "_last_sweep_at", 0.0)
    old = paste_store.write_paste("old", 1, home=tmp_path)
    _age(old, 7200)

    paste_store.maybe_sweep_pastes(config={"paste_retention_minutes": 60}, home=tmp_path)
    assert not old.exists()

    second = paste_store.write_paste("second", 2, home=tmp_path)
    _age(second, 7200)
    paste_store.maybe_sweep_pastes(config={"paste_retention_minutes": 60}, home=tmp_path)
    assert second.exists(), "the throttle should skip the immediate second sweep"


def test_list_pastes_is_newest_first_and_skips_pending_json(tmp_path):
    first = paste_store.write_paste("first", 1, home=tmp_path)
    _age(first, 600)
    second = paste_store.write_paste("second", 2, home=tmp_path)
    (tmp_path / "pastes" / paste_store.DEBUG_PENDING_NAME).write_text("[]", encoding="utf-8")

    assert paste_store.list_pastes(home=tmp_path) == [second, first]


def test_clear_session_pastes_removes_only_this_process_writes(tmp_path):
    from_this_process = paste_store.write_paste("mine", 1, home=tmp_path)
    other = paste_store.write_paste("theirs", 2, home=tmp_path)
    paste_store._session_pastes.discard(other)  # simulate another session's file

    assert paste_store.clear_session_pastes() == 1
    assert not from_this_process.exists()
    assert other.exists()


def test_clear_session_pastes_honors_keep_forever(monkeypatch, tmp_path):
    mine = paste_store.write_paste("mine", 1, home=tmp_path)
    monkeypatch.setattr(paste_store, "retention_minutes", lambda config=None: 0.0)

    assert paste_store.clear_session_pastes() == 0
    assert mine.exists()


def test_session_pastes_die_with_the_process(tmp_path):
    """The atexit hook is what stops a finished session's pasted text from outliving it."""
    repo_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, HERMES_HOME=str(tmp_path), PYTHONPATH=str(repo_root))
    code = "from hermes_cli import paste_store; print(paste_store.write_paste('secret', 1))"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            env=env, cwd=str(repo_root), timeout=120, check=False)
    assert result.returncode == 0, result.stderr

    written = Path(result.stdout.strip().splitlines()[-1])
    assert written.parent == tmp_path / "pastes"
    assert not written.exists(), "the exiting process must delete the paste it wrote"


def test_is_paste_file_rejects_other_entries():
    assert paste_store.is_paste_file("paste_1_101010.txt")
    assert not paste_store.is_paste_file("pending.json")
    assert not paste_store.is_paste_file("paste_1_101010.txt.lock")
    assert not paste_store.is_paste_file("notes.txt")
