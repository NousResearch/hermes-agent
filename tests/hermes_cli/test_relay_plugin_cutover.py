"""Regression tests for migration from the removed Hermes Relay plugin."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
import yaml

from hermes_cli.config import migrate_config
from hermes_cli.doctor import collect_relay_plugin_cutover_findings
from hermes_cli.relay_plugin_cutover import RELAY_PLUGINS_CONFIG_ENV


def test_v38_migration_removes_only_legacy_relay_plugin_keys(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "_config_version": 37,
                "plugins": {
                    "enabled": [
                        "keep-me",
                        "observability/nemo_relay",
                        "nemo_relay",
                    ]
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        results = migrate_config(interactive=False, quiet=True)

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert raw["_config_version"] == 38
    assert raw["plugins"]["enabled"] == ["keep-me"]
    assert any(
        "observability/nemo_relay" in warning
        and RELAY_PLUGINS_CONFIG_ENV in warning
        for warning in results["warnings"]
    )


def test_doctor_reports_stale_relay_plugin_key():
    findings = dict(
        collect_relay_plugin_cutover_findings(
            {"plugins": {"enabled": ["observability/nemo_relay"]}},
            {},
        )
    )

    replacement = findings["plugins.enabled: observability/nemo_relay"]
    assert "remove it" in replacement
    assert RELAY_PLUGINS_CONFIG_ENV in replacement


def test_doctor_reports_legacy_exporter_env_without_new_config(monkeypatch):
    monkeypatch.delenv(RELAY_PLUGINS_CONFIG_ENV, raising=False)
    findings = dict(
        collect_relay_plugin_cutover_findings(
            {},
            {
                "HERMES_NEMO_RELAY_ATIF_ENABLED": "true",
                "HERMES_NEMO_RELAY_ATIF_EXPORT_TIMEOUT_S": "30",
            },
        )
    )

    assert "now ignored" in findings["HERMES_NEMO_RELAY_ATIF_ENABLED"]
    assert RELAY_PLUGINS_CONFIG_ENV in findings["HERMES_NEMO_RELAY_ATIF_ENABLED"]
    assert "now ignored" in findings["HERMES_NEMO_RELAY_ATIF_EXPORT_TIMEOUT_S"]
    assert (
        RELAY_PLUGINS_CONFIG_ENV
        in findings["HERMES_NEMO_RELAY_ATIF_EXPORT_TIMEOUT_S"]
    )


@pytest.mark.parametrize("name", ["nemo_relay", "observability/nemo_relay"])
def test_enable_rejects_removed_relay_plugin_without_discovery(name, capsys):
    with (
        patch("hermes_cli.plugins_cmd._resolve_plugin_key_and_source") as resolve,
        patch("hermes_cli.plugins_cmd._save_enabled_set") as save_enabled,
    ):
        from hermes_cli.plugins_cmd import cmd_enable

        with pytest.raises(SystemExit) as exc_info:
            cmd_enable(name, allow_tool_override=False)

    assert exc_info.value.code == 1
    resolve.assert_not_called()
    save_enabled.assert_not_called()
    output = capsys.readouterr().out
    assert name in output
    assert RELAY_PLUGINS_CONFIG_ENV in output


def test_enable_rejects_alias_resolving_to_removed_relay_plugin(capsys):
    with (
        patch(
            "hermes_cli.plugins_cmd._resolve_plugin_key_and_source",
            return_value=("observability/nemo_relay", "user"),
        ),
        patch("hermes_cli.plugins_cmd._save_enabled_set") as save_enabled,
    ):
        from hermes_cli.plugins_cmd import cmd_enable

        with pytest.raises(SystemExit) as exc_info:
            cmd_enable("relay-copy", allow_tool_override=False)

    assert exc_info.value.code == 1
    save_enabled.assert_not_called()
    output = capsys.readouterr().out
    assert "observability/nemo_relay" in output
    assert RELAY_PLUGINS_CONFIG_ENV in output


def test_doctor_does_not_warn_for_legacy_env_after_new_config_is_selected(
    monkeypatch,
):
    monkeypatch.delenv(RELAY_PLUGINS_CONFIG_ENV, raising=False)
    findings = collect_relay_plugin_cutover_findings(
        {},
        {
            RELAY_PLUGINS_CONFIG_ENV: "/tmp/plugins.toml",
            "HERMES_NEMO_RELAY_ATIF_ENABLED": "true",
        },
    )

    assert findings == []
