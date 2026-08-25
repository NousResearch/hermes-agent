from pathlib import Path
from unittest.mock import MagicMock


def test_default_config_exposes_vacuum_interval():
    from hermes_cli.config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["sessions"]["min_vacuum_interval_days"] == 30


def test_default_config_auto_prune_on_with_90_day_retention():
    """#54189: state.db retention is ON by default (ended sessions, 90 days)."""
    from hermes_cli.config import DEFAULT_CONFIG

    sessions = DEFAULT_CONFIG["sessions"]
    assert sessions["auto_prune"] is True
    assert sessions["retention_days"] == 90
    assert sessions["vacuum_after_prune"] is True


def test_fresh_config_runs_auto_prune_at_startup(monkeypatch, tmp_path: Path):
    """A config.yaml with NO ``sessions:`` keys must reach the prune call with the
    new defaults (the loader deep-merges DEFAULT_CONFIG)."""
    import cli
    import hermes_cli.config
    import hermes_constants
    from hermes_cli.config import DEFAULT_CONFIG

    session_db = MagicMock()
    session_db.get_meta.return_value = "already-done"
    # Simulate load_config() on a fresh home: only defaults for the section.
    monkeypatch.setattr(
        hermes_cli.config,
        "load_config",
        lambda: {"sessions": dict(DEFAULT_CONFIG["sessions"])},
    )
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)

    cli._run_state_db_auto_maintenance(session_db)

    session_db.maybe_auto_prune_and_vacuum.assert_called_once_with(
        retention_days=90,
        min_interval_hours=24,
        min_vacuum_interval_days=30,
        vacuum=True,
        sessions_dir=tmp_path / "sessions",
    )


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


def test_shipped_template_does_not_pin_sessions_keys():
    """Installers copy cli-config.yaml.example verbatim into config.yaml, so any
    uncommented ``sessions:`` value there becomes an EXPLICIT user setting that
    would freeze the retention defaults. The template must leave them commented
    so code defaults (and future flips) apply."""
    import yaml

    template = Path(__file__).resolve().parents[1] / "cli-config.yaml.example"
    data = yaml.safe_load(template.read_text(encoding="utf-8")) or {}
    assert "sessions" not in data


def test_shipped_template_keeps_pre_update_backup_quick(tmp_path, monkeypatch):
    """A fresh install seeded from cli-config.yaml.example must keep the
    pre-update safety net on.

    Same seeding rule as ``test_shipped_template_does_not_pin_sessions_keys``:
    installers (scripts/install.sh, docker/stage2-hook.sh, hermes doctor
    --fix) copy the template verbatim, so its ``updates.pre_update_backup``
    value becomes an EXPLICIT setting that overrides the code default.
    After #65754 made "quick" the default, the template still shipped
    the legacy boolean ``false``, which resolves to "off" — every new
    install silently lost the #48200 pre-update safety net (#94944).
    """
    from types import SimpleNamespace

    from hermes_cli.update_cmd_maint import _resolve_pre_update_backup_mode

    template = Path(__file__).resolve().parents[1] / "cli-config.yaml.example"
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        template.read_text(encoding="utf-8"), encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    assert _resolve_pre_update_backup_mode(SimpleNamespace()) == "quick"


def test_loader_yields_new_defaults_for_fresh_home(monkeypatch, tmp_path: Path):
    """Real load_config() against an empty HERMES_HOME → auto_prune on, 90 days."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_cli.config import load_config

    sessions = load_config().get("sessions") or {}
    assert sessions.get("auto_prune") is True
    assert sessions.get("retention_days") == 90


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
