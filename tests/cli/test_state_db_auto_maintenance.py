from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("raw_value", ["false", False])
def test_run_state_db_auto_maintenance_respects_vacuum_flag(monkeypatch, tmp_path, raw_value):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    import cli as cli_mod

    session_db = Mock()
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "sessions": {
                "auto_prune": True,
                "retention_days": 30,
                "min_interval_hours": 12,
                "vacuum_after_prune": raw_value,
            }
        },
    )

    cli_mod._run_state_db_auto_maintenance(session_db)

    session_db.maybe_auto_prune_and_vacuum.assert_called_once_with(
        retention_days=30,
        min_interval_hours=12,
        vacuum=False,
    )
