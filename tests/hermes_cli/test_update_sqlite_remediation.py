"""Post-update reporting for unresolved SQLite WAL-reset risk."""

from pathlib import Path
from types import SimpleNamespace

from hermes_cli import update_cmd
import hermes_cli.update_cmd_maint as update_cmd_maint


def test_runtime_status_probes_running_venv_outside_checkout(tmp_path, monkeypatch):
    running_python = tmp_path / "venv312" / "bin" / "python"
    observed = []
    vulnerable = SimpleNamespace(wal_reset_vulnerable=True)
    monkeypatch.setattr("hermes_constants.project_venv_dir", lambda _root: None)
    monkeypatch.setattr(update_cmd.sys, "executable", str(running_python))
    monkeypatch.setattr(
        "hermes_cli.sqlite_runtime.probe_sqlite_runtime",
        lambda python: observed.append(Path(python)) or vulnerable,
    )

    safe, info = update_cmd._post_update_sqlite_runtime_status()

    assert observed == [running_python]
    assert safe is False
    assert info is vulnerable


def test_summary_withholds_success_when_sqlite_remediation_failed(capsys, monkeypatch):
    monkeypatch.setattr(
        update_cmd,
        "_post_update_sqlite_runtime_status",
        lambda: (False, SimpleNamespace(sqlite_version_string="3.46.1")),
        raising=False,
    )
    monkeypatch.setattr(
        update_cmd_maint,
        "_post_update_sqlite_runtime_status",
        lambda: (False, SimpleNamespace(sqlite_version_string="3.46.1")),
        raising=False,
    )
    monkeypatch.setattr(
        update_cmd,
        "_update_complete_message",
        lambda _version: "✓ Update complete! (v0.20.5)",
    )
    monkeypatch.setattr(
        update_cmd_maint,
        "_update_complete_message",
        lambda _version: "✓ Update complete! (v0.20.5)",
    )

    complete = update_cmd_maint._print_verified_update_completion(
        update_cmd_maint._update_complete_message("0.20.4"),
    )

    out = capsys.readouterr().out
    assert complete is False
    assert "Update complete" not in out
    assert "SQLite 3.46.1" in out
    assert "WAL-reset" in out
    assert "uv-managed Python" in out
    assert "hermes doctor" in out


def test_current_checkout_completion_is_verified_before_success(capsys, monkeypatch):
    monkeypatch.setattr(
        update_cmd,
        "_post_update_sqlite_runtime_status",
        lambda: (False, SimpleNamespace(sqlite_version_string="3.46.1")),
    )
    monkeypatch.setattr(
        update_cmd_maint,
        "_post_update_sqlite_runtime_status",
        lambda: (False, SimpleNamespace(sqlite_version_string="3.46.1")),
    )

    complete = update_cmd._print_verified_update_completion("✓ Already up to date!")

    out = capsys.readouterr().out
    assert complete is False
    assert "Already up to date" not in out
    assert "SQLite 3.46.1" in out


def test_current_checkout_repair_returns_verified_completion_result(monkeypatch):
    monkeypatch.setattr(update_cmd, "_prepare_updated_checkout", lambda *a, **k: None)
    monkeypatch.setattr(update_cmd, "_check_and_apply_config_migration", lambda **k: None)
    monkeypatch.setattr(
        update_cmd,
        "_print_verified_update_completion",
        lambda _message: False,
    )
    complete = update_cmd._repair_current_checkout(
        assume_yes=True,
        gateway_mode=False,
        pre_update_snapshot_id=None,
        had_desktop_app_before_update=False,
        upstream_checked=True,
    )

    assert complete is False
