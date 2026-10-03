from pathlib import Path
from unittest.mock import MagicMock








def test_explicit_auto_prune_false_is_respected(monkeypatch, tmp_path: Path):
    """Migration guard: an install that explicitly opted out keeps its choice."""
    import cli
    import hermes_cli.config
    import hermes_constants

    session_db = MagicMock()
    session_db.get_meta.return_value = "already-done"
    monkeypatch.setattr(
        hermes_cli.config,
        "load_config",
        lambda: {"sessions": {"auto_prune": False, "retention_days": 90}},
    )
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)

    cli._run_state_db_auto_maintenance(session_db)

    session_db.maybe_auto_prune_and_vacuum.assert_not_called()






def test_negative_retention_days_in_config_deletes_nothing(monkeypatch, tmp_path: Path):
    """E2E through the real startup path: ``sessions.retention_days: -1`` reaches
    ``maybe_auto_prune_and_vacuum`` from this config loader; without validation it
    builds a future cutoff and deletes every ended session."""
    import cli
    import hermes_cli.config
    import hermes_constants
    from hermes_state import SessionDB

    session_db = SessionDB(db_path=tmp_path / "state.db")
    try:
        session_db.create_session(session_id="ended", source="cli")
        session_db.end_session("ended", "done")
        monkeypatch.setattr(
            hermes_cli.config,
            "load_config",
            lambda: {"sessions": {"auto_prune": True, "retention_days": -1}},
        )
        monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)

        cli._run_state_db_auto_maintenance(session_db)

        assert session_db.get_session("ended") is not None
    finally:
        session_db.close()


def test_non_positive_min_interval_hours_in_config_falls_back_to_default(monkeypatch, tmp_path: Path):
    """``sessions.min_interval_hours: 0`` (or negative) does not disable the throttle — it makes
    ``now - last < min_interval_hours * 3600`` false on every call, so the sweep this gates would
    run on every housekeeping tick instead of at most once per interval. Must fall back to the
    documented default (24) instead of reaching the DB layer un-floored."""
    import cli
    import hermes_cli.config
    import hermes_constants

    session_db = MagicMock()
    session_db.get_meta.return_value = "already-done"
    monkeypatch.setattr(
        hermes_cli.config,
        "load_config",
        lambda: {
            "sessions": {
                "auto_archive": True, "auto_archive_days": 3,
                "auto_prune": True, "retention_days": 90, "min_interval_hours": 0,
            }
        },
    )
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)

    cli._run_state_db_auto_maintenance(session_db)

    session_db.maybe_auto_archive.assert_called_once_with(idle_days=3.0, min_interval_hours=24)
    session_db.maybe_auto_prune_and_vacuum.assert_called_once_with(
        retention_days=90,
        min_interval_hours=24,
        min_vacuum_interval_days=30,
        vacuum=True,
        sessions_dir=tmp_path / "sessions",
    )


def test_cli_auto_maintenance_forwards_vacuum_interval(monkeypatch, tmp_path: Path):
    import cli
    import hermes_cli.config
    import hermes_constants

    session_db = MagicMock()
    session_db.get_meta.return_value = "already-done"
    monkeypatch.setattr(
        hermes_cli.config,
        "load_config",
        lambda: {
            "sessions": {
                "auto_prune": True,
                "retention_days": 90,
                "vacuum_after_prune": True,
                "min_interval_hours": 24,
                "min_vacuum_interval_days": 17,
            }
        },
    )
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)

    cli._run_state_db_auto_maintenance(session_db)

    session_db.maybe_auto_prune_and_vacuum.assert_called_once_with(
        retention_days=90,
        min_interval_hours=24,
        min_vacuum_interval_days=17,
        vacuum=True,
        sessions_dir=tmp_path / "sessions",
    )
