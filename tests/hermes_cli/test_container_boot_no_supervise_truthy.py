"""HERMES_GATEWAY_NO_SUPERVISE must accept shared truthy aliases."""

from __future__ import annotations

from hermes_cli.container_boot import _maybe_migrate_legacy_gateway_run_state


def test_no_supervise_on_alias_skips_legacy_migrate(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_GATEWAY_NO_SUPERVISE", "on")
    result = _maybe_migrate_legacy_gateway_run_state(
        tmp_path,
        container_argv=("gateway", "run"),
        dry_run=True,
    )
    assert result is None
    assert not (tmp_path / "gateway_state.json").exists()


def test_no_supervise_1_alias_skips_legacy_migrate(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_GATEWAY_NO_SUPERVISE", "1")
    result = _maybe_migrate_legacy_gateway_run_state(
        tmp_path,
        container_argv=("gateway", "run"),
        dry_run=True,
    )
    assert result is None


def test_no_supervise_unset_still_migrates_legacy(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_GATEWAY_NO_SUPERVISE", raising=False)
    result = _maybe_migrate_legacy_gateway_run_state(
        tmp_path,
        container_argv=("gateway", "run"),
        dry_run=True,
    )
    assert result == "running"


def test_no_supervise_off_does_not_opt_out(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_GATEWAY_NO_SUPERVISE", "off")
    result = _maybe_migrate_legacy_gateway_run_state(
        tmp_path,
        container_argv=("gateway", "run"),
        dry_run=True,
    )
    assert result == "running"
