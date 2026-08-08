"""Regression coverage for the gateway's long-lived SQLite FD budget."""

from gateway.config import GatewayConfig
from gateway.run import GatewayRunner


def test_gateway_runner_reuses_session_store_database(monkeypatch, tmp_path):
    """One runner must own one SessionDB, not one per gateway subsystem."""
    import hermes_state

    created = []

    class FakeSessionDB:
        def __init__(self, *args, **kwargs):
            created.append(self)

    monkeypatch.setattr(hermes_state, "SessionDB", FakeSessionDB)

    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))

    assert len(created) == 1
    assert runner._session_db is not None
    assert runner._session_db._db is runner.session_store._db
