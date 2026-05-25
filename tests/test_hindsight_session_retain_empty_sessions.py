import importlib.util
from pathlib import Path


def _load_retain_module():
    path = Path('/home/ht/.hermes/hooks/session-end-retain/hindsight_session_retain.py')
    spec = importlib.util.spec_from_file_location('hindsight_session_retain', path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_process_session_marks_empty_session_processed(monkeypatch):
    mod = _load_retain_module()
    marked = []
    monkeypatch.setattr(mod, '_get_session_messages', lambda session_id: [])
    monkeypatch.setattr(mod, '_mark_processed', lambda session_id: marked.append(session_id))

    ok = mod.process_session('sess-empty', 'telegram', dry_run=False)

    assert ok is True
    assert marked == ['sess-empty']


def test_fallback_filters_zero_message_sessions(monkeypatch):
    mod = _load_retain_module()
    monkeypatch.setattr(mod, '_load_processed_ids', lambda: set())
    monkeypatch.setattr(mod, '_read_queue', lambda: [])
    monkeypatch.setattr(mod, '_query_db', lambda *_args, **_kwargs: [
        {'id': 'sess-empty', 'source': 'telegram', 'user_id': 'u1', 'ended_at': 1, 'title': None, 'message_count': 0},
        {'id': 'sess-real-1', 'source': 'feishu', 'user_id': 'u2', 'ended_at': 2, 'title': 'a', 'message_count': 3},
        {'id': 'sess-real-2', 'source': 'feishu', 'user_id': 'u3', 'ended_at': 3, 'title': 'b', 'message_count': 2},
        {'id': 'sess-real-3', 'source': 'feishu', 'user_id': 'u4', 'ended_at': 4, 'title': 'c', 'message_count': 1},
    ])
    monkeypatch.setattr(mod, '_get_session_messages', lambda sid: [{'role': 'user', 'content': sid}])
    monkeypatch.setattr(mod, '_extract_items', lambda messages: [{'content': messages[0]['content'], 'role': 'user'}])
    seen = []
    monkeypatch.setattr(mod, '_retain_items', lambda items, session_id, platform, dry_run: seen.append(session_id) or True)

    class _Resp:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False
        def read(self):
            return b'{"status": "healthy"}'

    monkeypatch.setattr(mod.urllib.request, 'urlopen', lambda *args, **kwargs: _Resp())
    monkeypatch.setattr(mod.sys, 'argv', ['hindsight_session_retain.py'])

    mod.main()

    assert seen == ['sess-real-1', 'sess-real-2', 'sess-real-3']
