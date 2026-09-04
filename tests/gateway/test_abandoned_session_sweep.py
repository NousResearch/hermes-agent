from types import SimpleNamespace
import threading
import time

import pytest

from gateway.config import GatewayConfig
from gateway.run import GatewayRunner, _SESSION_DB_UNPINNED
from hermes_state import SessionDB


class _SessionDBStub:
    def __init__(self, result=0, error=None):
        self.result = result
        self.error = error
        self.calls = []

    async def finalize_abandoned_sessions(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


def _runner(session_db, *, idle_minutes=90):
    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(
        default_reset_policy=SimpleNamespace(idle_minutes=idle_minutes)
    )
    runner._session_db_pinned = session_db
    return runner


@pytest.mark.asyncio
async def test_sweep_uses_configured_idle_window():
    session_db = _SessionDBStub(result=3)
    runner = _runner(session_db, idle_minutes=37)

    assert await runner._sweep_abandoned_sessions() == 3
    assert session_db.calls == [{"idle_minutes": 37}]


@pytest.mark.asyncio
async def test_sweep_is_best_effort_when_database_is_unavailable():
    runner = _runner(None)
    assert await runner._sweep_abandoned_sessions() == 0

    session_db = _SessionDBStub(error=RuntimeError("locked"))
    runner = _runner(session_db)
    assert await runner._sweep_abandoned_sessions() == 0


@pytest.mark.asyncio
async def test_multiplex_sweep_finalizes_each_profile_store(
    tmp_path, monkeypatch
):
    import gateway.run as gateway_run
    import hermes_state

    root = tmp_path / "hermes"
    secondary = root / "profiles" / "secondary"
    root.mkdir(parents=True)
    secondary.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(
        hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH
    )
    monkeypatch.setattr(
        gateway_run,
        "_multiplex_profile_homes",
        lambda _config: [("default", root), ("secondary", secondary)],
    )

    old = time.time() - 2 * 86400
    for home, session_id in (
        (root, "root-abandoned"),
        (secondary, "secondary-abandoned"),
    ):
        db = SessionDB(db_path=home / "state.db")
        db.create_session(session_id, "telegram")
        db.append_message(session_id, "user", "hello")
        with db._lock:
            db._conn.execute(
                "UPDATE sessions SET started_at = ?, last_activity_at = NULL "
                "WHERE id = ?",
                (old, session_id),
            )
            db._conn.commit()
        db.close()

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner._session_db_pinned = _SESSION_DB_UNPINNED
    runner._session_db_handles = {}
    runner._session_db_handles_lock = threading.Lock()

    try:
        assert await runner._sweep_abandoned_sessions() == 2
        root_db = SessionDB(db_path=root / "state.db")
        secondary_db = SessionDB(db_path=secondary / "state.db")
        try:
            assert root_db.get_session("root-abandoned")["end_reason"] == "abandoned"
            assert (
                secondary_db.get_session("secondary-abandoned")["end_reason"]
                == "abandoned"
            )
            assert root_db.get_session("secondary-abandoned") is None
            assert secondary_db.get_session("root-abandoned") is None
        finally:
            root_db.close()
            secondary_db.close()
    finally:
        runner.close_all_session_db_handles()
