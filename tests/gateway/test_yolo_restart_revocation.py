"""An explicitly revoked --yolo launch stays off in a fresh gateway interpreter."""
import subprocess
import sys


def test_persisted_off_is_not_overridden_by_frozen_launch_after_restart(tmp_path):
    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionStore, SessionSource
    store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
    store._db = None
    source = SessionSource(platform=Platform.LOCAL, chat_id='revoked', user_id='human')
    entry = store.get_or_create_session(source)
    store.set_session_yolo(entry.session_key, True)
    store.set_session_yolo(entry.session_key, False)
    probe = '''
import sys
from pathlib import Path
from gateway.config import GatewayConfig
from gateway.session import SessionStore
from types import SimpleNamespace
from gateway.session_managed_worker import _session_yolo
from tools.approval import is_session_yolo_enabled
store = SessionStore(sessions_dir=Path(sys.argv[1]), config=GatewayConfig())
store._db = None
entry = store.lookup_by_session_key(sys.argv[2])
assert entry is not None and entry.yolo is False
_session_yolo(SimpleNamespace(runner=SimpleNamespace(session_store=store)),
              entry.session_key, SimpleNamespace(yolo=True))
assert not is_session_yolo_enabled(entry.session_key), 'revoked launch was re-enabled'
'''
    result = subprocess.run([sys.executable, '-c', probe, str(tmp_path), entry.session_key],
                            capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_live_toggle_survives_failed_persistence_after_initial_restore():
    from tools.approval import clear_session, is_session_yolo_enabled
    from tools.approval_yolo import restore_gateway_yolo, toggle_session_yolo
    key = 'gateway-yolo-persistence-failure'
    def unavailable(enabled):
        raise OSError('routing store unavailable')
    try:
        restore_gateway_yolo(key, False)
        toggle_session_yolo(key, True, persist=unavailable)
        restore_gateway_yolo(key, False)
        assert is_session_yolo_enabled(key) is True
        toggle_session_yolo(key, False, persisted=True, persist=unavailable)
        restore_gateway_yolo(key, True)
        assert is_session_yolo_enabled(key) is False
    finally:
        clear_session(key)
