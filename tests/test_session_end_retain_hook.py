import importlib.util
import json
import sqlite3
from pathlib import Path


def _load_handler_module():
    path = Path('/home/ht/.hermes/hooks/session-end-retain/handler.py')
    spec = importlib.util.spec_from_file_location('session_end_retain_handler', path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _seed_state_db(db_path: Path):
    conn = sqlite3.connect(db_path)
    conn.execute(
        '''
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            source TEXT,
            user_id TEXT,
            ended_at REAL,
            end_reason TEXT
        )
        '''
    )
    conn.commit()
    conn.close()


def test_find_old_session_id_prefers_exact_session_key_match(tmp_path, monkeypatch):
    mod = _load_handler_module()
    db_path = tmp_path / 'state.db'
    queue_path = tmp_path / 'queue.jsonl'
    _seed_state_db(db_path)

    conn = sqlite3.connect(db_path)
    conn.execute(
        'INSERT INTO sessions (id, source, user_id, ended_at, end_reason) VALUES (?, ?, ?, ?, ?)',
        ('sess_target_001', 'feishu', 'u1', 100.0, 'user_exit'),
    )
    conn.execute(
        'INSERT INTO sessions (id, source, user_id, ended_at, end_reason) VALUES (?, ?, ?, ?, ?)',
        ('sess_other_newer', 'feishu', 'u2', 200.0, 'user_exit'),
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(mod, 'STATE_DB', db_path)
    monkeypatch.setattr(mod, 'QUEUE_FILE', queue_path)

    found = mod._find_old_session_id('sess_target_001')
    assert found == 'sess_target_001'


def test_handle_enqueues_exact_session_key_not_latest_ended_session(tmp_path, monkeypatch):
    mod = _load_handler_module()
    db_path = tmp_path / 'state.db'
    queue_path = tmp_path / 'queue.jsonl'
    _seed_state_db(db_path)

    conn = sqlite3.connect(db_path)
    conn.execute(
        'INSERT INTO sessions (id, source, user_id, ended_at, end_reason) VALUES (?, ?, ?, ?, ?)',
        ('sess_target_001', 'feishu', 'u1', 100.0, 'user_exit'),
    )
    conn.execute(
        'INSERT INTO sessions (id, source, user_id, ended_at, end_reason) VALUES (?, ?, ?, ?, ?)',
        ('sess_other_newer', 'feishu', 'u2', 200.0, 'user_exit'),
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(mod, 'STATE_DB', db_path)
    monkeypatch.setattr(mod, 'QUEUE_FILE', queue_path)

    mod.handle('session:end', {'session_key': 'sess_target_001', 'platform': 'feishu'})

    lines = queue_path.read_text(encoding='utf-8').splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload['session_id'] == 'sess_target_001'
    assert payload['platform'] == 'feishu'
