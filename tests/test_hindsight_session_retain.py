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


def test_fallback_day_scan_batch_size_is_three_when_queue_empty(monkeypatch):
    mod = _load_retain_module()

    monkeypatch.setattr(mod, '_load_processed_ids', lambda: set())
    monkeypatch.setattr(mod, '_read_queue', lambda: [])
    monkeypatch.setattr(mod, 'process_session', lambda sid, platform, dry_run: True)

    sessions = [
        {'id': f'sess_{i}', 'source': 'feishu', 'title': f't{i}', 'message_count': i}
        for i in range(5)
    ]
    monkeypatch.setattr(mod, '_get_today_sessions', lambda target_date=None: sessions)

    captured = []

    def _capture_process(sid, platform, dry_run):
        captured.append((sid, platform, dry_run))
        return True

    monkeypatch.setattr(mod, 'process_session', _capture_process)
    monkeypatch.setattr(mod, 'QUEUE_FILE', Path('/tmp/nonexistent-retain-queue.jsonl'))
    monkeypatch.setattr(mod, 'HINDSIGHT_URL', 'http://fake')

    class _Resp:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False
        def read(self):
            return b'{"status": "healthy"}'

    monkeypatch.setattr(mod.urllib.request, 'urlopen', lambda *args, **kwargs: _Resp())
    monkeypatch.setattr(mod.sys, 'argv', ['hindsight_session_retain.py', '--dry-run'])

    mod.main()

    assert [sid for sid, _platform, _dry_run in captured] == ['sess_1', 'sess_2', 'sess_3']
