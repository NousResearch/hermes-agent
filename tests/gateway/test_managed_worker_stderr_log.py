"""Managed-worker stderr (its tracebacks and redirected prints) lands in the owning profile's
private, redacted, size-capped ``logs/managed-worker.log`` instead of being discarded.

A worker bound to a Kanban task mirrors the same redacted lines into that task's card log — the
dashboard's "Worker log" — which otherwise carried only the client's ``Session:`` line and the
exit trailer, making a running managed worker look dead.
"""
import asyncio
import io
import os
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

from gateway.session_contract import SessionRef

SECRET = 'sk-proj-' + 'Q' * 40
CRASH = f"""import sys
sys.stderr.write('Traceback (most recent call last):\\n')
sys.stderr.write('RuntimeError: provider refused api_key={SECRET} Authorization: Bearer {SECRET}\\n')
sys.stderr.flush()
raise SystemExit(1)
"""


@pytest.mark.platforms("linux")
def test_worker_stderr_reaches_a_private_redacted_profile_log(tmp_path, monkeypatch):
    from gateway import session_managed_worker as managed
    original = subprocess.Popen

    def crashing_child(args, **kwargs):
        return original([sys.executable, '-c', CRASH], **kwargs)
    monkeypatch.setattr(managed.subprocess, 'Popen', crashing_child)
    monkeypatch.setattr(managed, '_worker_env', lambda authority: None)
    authority = SimpleNamespace(profile_id=str(tmp_path), pending_results={}, waiters={},
                                adopt_agent=lambda session_id, generation, worker: None)
    row = {'admission_id': 'adm', 'principal_id': 'owner', 'generation': 3, 'payload': {'text': 'go'}}
    with pytest.raises(EOFError):
        asyncio.run(asyncio.wait_for(managed.execute_managed(authority, SessionRef('p', 's'), row, None), 20))
    log = tmp_path / 'logs' / 'managed-worker.log'
    text = log.read_text()
    assert 'Traceback (most recent call last)' in text and 'RuntimeError: provider refused' in text
    assert SECRET not in text and 'QQQQQQQQQQ' not in text
    assert stat.S_IMODE(log.stat().st_mode) == 0o600


def test_worker_log_is_bounded_and_scrubs_the_workers_exact_secrets(tmp_path):
    from gateway.session_managed_worker_log import LOG_NAME, MAX_LOG_BYTES, drain_worker_stderr
    path = tmp_path / 'logs' / LOG_NAME
    assignment = 'opaque-assignment-secret-value'
    noise = (f'line {assignment} ' + 'x' * 200 + '\n') * 12_000 + 'y' * 300_000 + '\nlast\n'
    drain_worker_stderr(io.BytesIO(noise.encode()), path, 7, lambda: [assignment])
    files = [path.with_name(LOG_NAME + '.1'), path]  # oldest first
    assert all(f.stat().st_size <= MAX_LOG_BYTES for f in files)
    assert not list(path.parent.glob(LOG_NAME + '.2'))
    text = ''.join(f.read_text() for f in files)
    assert assignment not in text and '[truncated]' in text and text.endswith('[pid 7] last\n')


def _card_log(tmp_path, task_id='t_00000001'):
    """The dispatcher's card log: it created the directory before spawning the worker."""
    directory = tmp_path / 'kanban' / 'logs'
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f'{task_id}.log'


def test_kanban_bound_worker_stderr_reaches_the_card_log_redacted_and_capped(tmp_path):
    from gateway.session_managed_worker_log import KEPT_LINE_CHARS, MAX_LINE_BYTES, drain_worker_stderr
    profile_log = tmp_path / 'logs' / 'managed-worker.log'
    card_log = _card_log(tmp_path)
    assignment = 'opaque-assignment-secret-value'
    # Over MAX_LINE_BYTES so the drain's own whole-line cap truncates it, as in the private log.
    oversized = 'z' * (MAX_LINE_BYTES + 1_000)
    stream = io.BytesIO(
        f'Traceback (most recent call last):\n'
        f'RuntimeError: provider refused api_key={SECRET} Authorization: Bearer {SECRET}\n'
        f'{assignment} then {oversized}\n'.encode())
    drain_worker_stderr(stream, profile_log, 7, lambda: [assignment], card_log)
    text = card_log.read_text()
    assert 'Traceback (most recent call last)' in text and 'RuntimeError: provider refused' in text
    assert SECRET not in text and 'QQQQQQQQQQ' not in text
    assert assignment not in text
    assert '[truncated]' in text and oversized not in text
    assert max(len(line) for line in text.splitlines()) <= KEPT_LINE_CHARS + len('[pid 7]  [truncated]')
    assert card_log.read_text() == profile_log.read_text()


def test_private_profile_log_still_receives_every_line_beside_the_card_log(tmp_path):
    from gateway.session_managed_worker_log import drain_worker_stderr
    profile_log = tmp_path / 'logs' / 'managed-worker.log'
    card_log = _card_log(tmp_path)
    drain_worker_stderr(io.BytesIO(b'first\nsecond\n'), profile_log, 7, list, card_log)
    assert profile_log.read_text() == '[pid 7] first\n[pid 7] second\n'
    assert card_log.read_text() == profile_log.read_text()


def test_non_kanban_worker_stderr_leaves_the_card_log_untouched(tmp_path):
    from gateway.session_managed_worker_log import drain_worker_stderr
    profile_log = tmp_path / 'logs' / 'managed-worker.log'
    card_log = _card_log(tmp_path)
    drain_worker_stderr(io.BytesIO(b'ordinary session noise\n'), profile_log, 7, list)
    assert 'ordinary session noise' in profile_log.read_text()
    assert not card_log.exists()


def test_card_log_path_follows_the_dispatchers_task_context(tmp_path, monkeypatch):
    from gateway.session_managed_worker_log import kanban_card_log_path
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(tmp_path))
    assert kanban_card_log_path(SimpleNamespace(kanban_json=None)) is None
    assert kanban_card_log_path(SimpleNamespace(kanban_json='null')) is None
    assert kanban_card_log_path(SimpleNamespace(kanban_json='{"board": "default"}')) is None
    assert (kanban_card_log_path(SimpleNamespace(kanban_json='{"board": "default", "task_id": "t_abc"}'))
            == tmp_path / 'kanban' / 'logs' / 't_abc.log')


@pytest.mark.platforms("linux")
def test_managed_worker_wiring_mirrors_stderr_into_the_card_log(tmp_path, monkeypatch):
    from gateway import session_managed_worker as managed
    original = subprocess.Popen

    def crashing_child(args, **kwargs):
        return original([sys.executable, '-c', CRASH], **kwargs)
    monkeypatch.setattr(managed.subprocess, 'Popen', crashing_child)
    monkeypatch.setattr(managed, '_worker_env', lambda authority: None)
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(tmp_path))
    authority = SimpleNamespace(profile_id=str(tmp_path), pending_results={}, waiters={},
                                adopt_agent=lambda session_id, generation, worker: None)
    policy = SimpleNamespace(kanban_json='{"board": "default", "task_id": "t_abc"}')
    card_log = _card_log(tmp_path, 't_abc')
    card_log.write_bytes(b'Session: local-deadbeef\n')
    row = {'admission_id': 'adm', 'principal_id': 'owner', 'generation': 3, 'payload': {'text': 'go'}}
    with pytest.raises(EOFError):
        asyncio.run(asyncio.wait_for(
            managed.execute_managed(authority, SessionRef('p', 's'), row, policy), 20))
    text = card_log.read_text()
    assert text.startswith('Session: local-deadbeef\n')
    assert 'Traceback (most recent call last)' in text and 'RuntimeError: provider refused' in text
    assert SECRET not in text and 'QQQQQQQQQQ' not in text
    assert os.path.exists(tmp_path / 'logs' / 'managed-worker.log')
