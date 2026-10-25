"""Holder evidence is diagnostic only; bounded init still proceeds."""
import json
import os
import threading
from pathlib import Path

from hermes_cli import kanban_db_connect as kbc


def test_holder_warning_identifies_pid_profile(tmp_path, monkeypatch, caplog):
    home = tmp_path / 'profiles' / 'worker-a'
    home.mkdir(parents=True)
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr(kbc, '_INIT_LOCK_TIMEOUT_SECONDS', 0.1)
    path = tmp_path / 'board.db'
    ready, release = threading.Event(), threading.Event()

    def holder():
        with kbc._cross_process_init_lock(path):
            ready.set()
            release.wait(5)

    thread = threading.Thread(target=holder)
    thread.start()
    try:
        assert ready.wait(3)
        with kbc._cross_process_init_lock(path):
            assert f'pid={os.getpid()} profile=worker-a' in caplog.text
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    # A second owner replaces rather than appends JSON under the same lock file.
    with kbc._cross_process_init_lock(path):
        with path.with_name('board.db.init.lock').open('rb') as f:
            f.seek(1)
            assert json.load(f)['profile'] == 'worker-a'


def test_legacy_holder_warning_is_unknown(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(kbc, '_try_lock_nb', lambda handle: False)
    monkeypatch.setattr(kbc, '_INIT_LOCK_TIMEOUT_SECONDS', 0)
    with kbc._cross_process_init_lock(tmp_path / 'board.db'):
        assert 'unknown (legacy or unreadable lock metadata)' in caplog.text
