from unittest.mock import Mock

import pytest

from gateway.config import GatewayConfig
from gateway.run import GatewayRunner


@pytest.mark.parametrize("raw_value", ["false", False])
def test_gateway_runner_respects_vacuum_after_prune_flag(monkeypatch, tmp_path, raw_value):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    fake_db = Mock()
    monkeypatch.setattr("hermes_state.SessionDB", lambda: fake_db)
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "sessions": {
                "auto_prune": True,
                "retention_days": 45,
                "min_interval_hours": 6,
                "vacuum_after_prune": raw_value,
            }
        },
    )

    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))

    assert runner._session_db is fake_db
    fake_db.maybe_auto_prune_and_vacuum.assert_called_once_with(
        retention_days=45,
        min_interval_hours=6,
        vacuum=False,
    )
