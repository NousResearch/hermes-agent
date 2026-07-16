"""CLI and config-cutover tests for ``hermes migrate state-postgres``."""
from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import sys

import pytest

from hermes_cli.migrate import cmd_migrate_state_postgres
from hermes_cli.state_postgres_migration import MigrationPhase, MigrationReport


class _Source:
    def __init__(self, path: Path):
        self.path = path


class _Target:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _Engine:
    calls: list[dict] = []

    def __init__(self, source, target, *, cutover=None):
        self.source = source
        self.target = target
        self.cutover = cutover

    def run(self, request):
        type(self).calls.append(
            {
                "source": self.source,
                "target": self.target,
                "cutover": self.cutover,
                "request": request,
            }
        )
        report = MigrationReport(
            run_id=request.run_id or "generated",
            target_identity="postgres schema=hermes dsn_env=HERMES_STATE_POSTGRES_DSN",
            apply=request.apply,
            phase=MigrationPhase.COMPLETE if request.apply else MigrationPhase.DRY_RUN,
        )
        if request.apply and self.cutover is not None:
            self.cutover(report)
        return report


def test_state_postgres_dry_run_has_no_cutover_and_output_has_no_dsn(
    monkeypatch, capsys, tmp_path: Path
):
    import hermes_cli.config as config
    import hermes_cli.state_postgres_migration as migration
    import state_store.postgres.migration_adapter as postgres_adapter
    import state_store.sqlite.migration_adapter as sqlite_adapter

    _Engine.calls.clear()
    monkeypatch.setattr(config, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(config, "load_config", lambda: {"sessions": {"state": {"sqlite_path": "legacy.db"}}})
    monkeypatch.setattr(sqlite_adapter, "SQLiteMigrationSourceAdapter", _Source)
    monkeypatch.setattr(postgres_adapter, "PostgresMigrationTargetAdapter", _Target)
    monkeypatch.setattr(migration, "StatePostgresMigration", _Engine)

    result = cmd_migrate_state_postgres(
        Namespace(
            apply=False,
            dsn_env="HERMES_STATE_POSTGRES_DSN",
            schema="hermes",
            batch_size=25,
            run_id="dry-run",
        )
    )

    assert result == 0
    assert _Engine.calls[0]["source"].path == tmp_path / "legacy.db"
    assert _Engine.calls[0]["cutover"] is None
    output = capsys.readouterr().out
    assert "HERMES_STATE_POSTGRES_DSN" in output
    assert "postgresql://" not in output


def test_apply_switches_config_only_from_engine_cutover(
    monkeypatch, capsys, tmp_path: Path
):
    import hermes_cli.config as config
    import hermes_cli.state_postgres_migration as migration
    import state_store.postgres.migration_adapter as postgres_adapter
    import state_store.sqlite.migration_adapter as sqlite_adapter

    _Engine.calls.clear()
    switches: list[tuple[str, str]] = []
    monkeypatch.setattr(config, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(config, "load_config", lambda: {})
    monkeypatch.setattr(
        config,
        "atomic_switch_state_to_postgres",
        lambda *, dsn_env, schema: switches.append((dsn_env, schema)),
    )
    monkeypatch.setattr(sqlite_adapter, "SQLiteMigrationSourceAdapter", _Source)
    monkeypatch.setattr(postgres_adapter, "PostgresMigrationTargetAdapter", _Target)
    monkeypatch.setattr(migration, "StatePostgresMigration", _Engine)

    result = cmd_migrate_state_postgres(
        Namespace(
            apply=True,
            dsn_env="PG_STATE_URL",
            schema="hermes_state",
            batch_size=100,
            run_id="apply-run",
        )
    )

    assert result == 0
    assert switches == [("PG_STATE_URL", "hermes_state")]
    assert _Engine.calls[0]["request"].apply
    assert _Engine.calls[0]["target"].kwargs["dsn_env"] == "PG_STATE_URL"
    assert "postgresql://" not in capsys.readouterr().out


def test_atomic_state_cutover_backs_up_raw_config_and_never_writes_dsn(
    monkeypatch, tmp_path: Path
):
    from hermes_cli import config

    config_path = tmp_path / "config.yaml"
    original = (
        "model:\n"
        "  default: test\n"
        "sessions:\n"
        "  state:\n"
        "    sqlite_path: custom-state.db\n"
    )
    config_path.write_text(original, encoding="utf-8")
    monkeypatch.setattr(config, "get_config_path", lambda: config_path)
    monkeypatch.setattr(config, "is_managed", lambda: False)

    backup = config.atomic_switch_state_to_postgres(
        dsn_env="HERMES_STATE_POSTGRES_DSN",
        schema="hermes_state",
    )

    rendered = config_path.read_text(encoding="utf-8")
    assert backup.read_text(encoding="utf-8") == original
    assert "backend: postgres" in rendered
    assert "dsn_env: HERMES_STATE_POSTGRES_DSN" in rendered
    assert "schema: hermes_state" in rendered
    assert "postgresql://" not in rendered


def test_cli_rejects_a_plaintext_dsn_option(monkeypatch):
    from hermes_cli import main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hermes",
            "migrate",
            "state-postgres",
            "--schema",
            "hermes_state",
            "--dsn",
            "postgresql://must-not-be-accepted",
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        main.main()

    assert exit_info.value.code == 2
