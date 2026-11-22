"""Routing entries nobody toggled follow the ``--yolo`` launch policy; only an explicit OFF revokes it."""
import subprocess
import sys
from types import SimpleNamespace

_RESTARTED_OWNER = '''
import sys
from pathlib import Path
from types import SimpleNamespace
from gateway.config import GatewayConfig
from gateway.session import SessionStore
from gateway.session_managed_worker import _session_yolo
store = SessionStore(sessions_dir=Path(sys.argv[1]), config=GatewayConfig())
store._db = None
print('ENABLED', _session_yolo(SimpleNamespace(runner=SimpleNamespace(session_store=store)),
                               sys.argv[2], SimpleNamespace(yolo=True)))
'''


def _store(tmp_path):
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
    store._db = None
    return store


def test_untoggled_entry_keeps_launch_yolo_after_restart_but_explicit_off_stays_off(tmp_path):
    from gateway.config import Platform
    from gateway.session import SessionSource
    store = _store(tmp_path)
    # Branch/reset/migration/cron/new-session entries are written without a toggle.
    untouched = store.get_or_create_session(SessionSource(platform=Platform.LOCAL, chat_id='branched', user_id='h'))
    revoked = store.get_or_create_session(SessionSource(platform=Platform.LOCAL, chat_id='revoked', user_id='h'))
    store.set_session_yolo(revoked.session_key, False)

    def restarted(key):
        result = subprocess.run([sys.executable, '-c', _RESTARTED_OWNER, str(tmp_path), key],
                                capture_output=True, text=True, timeout=30, check=False)
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()
    assert restarted(untouched.session_key) == 'ENABLED True', 'launch --yolo lost for an entry nobody revoked'
    assert restarted(revoked.session_key) == 'ENABLED False', 'owner /yolo OFF revived by the launch policy'


def test_launch_yolo_reseeds_after_a_session_boundary(tmp_path):
    """``apply_launch_yolo``: seeded once per session boundary; /new clears the toggle, not the launch."""
    import gateway.run as gateway_run
    from gateway.config import Platform
    from gateway.session import SessionSource
    from gateway.session_managed_worker import _session_yolo
    from tools.approval import clear_session
    store = _store(tmp_path)
    entry = store.get_or_create_session(SessionSource(platform=Platform.LOCAL, chat_id='bnd', user_id='h'))
    authority = SimpleNamespace(runner=SimpleNamespace(session_store=store))
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.session_store = store
    launch = SimpleNamespace(yolo=True)
    try:
        assert _session_yolo(authority, entry.session_key, launch) is True
        store.set_session_yolo(entry.session_key, False)  # owner /yolo off, then /new
        clear_session(entry.session_key)
        assert _session_yolo(authority, entry.session_key, launch) is False
        runner._clear_session_boundary_security_state(entry.session_key)
        assert _session_yolo(authority, entry.session_key, launch) is True, 'launch --yolo not re-seeded after /new'
    finally:
        clear_session(entry.session_key)
