"""Behavior contracts for model-provider package inventory and lifecycle state."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import yaml

from hermes_cli import plugins_cmd as pc

KEY = "model-providers/acme-provider"


def _entry(
    *,
    name="acme-provider",
    source="entrypoint",
    path: str | Path = "acme_provider:register",
    key=KEY,
):
    return (name, "1.2.3", "Acme provider", source, path, key)


def _args(**overrides):
    values = {
        "json": True,
        "plain": False,
        "enabled": False,
        "no_bundled": False,
        "user": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_dedicated_package_inventory_uses_canonical_namespace(monkeypatch, tmp_path):
    user = tmp_path / "user"
    bundled = tmp_path / "bundled"
    user.mkdir()
    bundled.mkdir()
    eps = importlib.metadata.EntryPoints([
        importlib.metadata.EntryPoint(
            name="generic", value="generic:register", group="hermes_agent.plugins"
        ),
        importlib.metadata.EntryPoint(
            name="acme-provider",
            value="acme_provider:register",
            group="hermes_agent.model_providers",
        ),
    ])
    monkeypatch.setattr(pc, "_plugins_dir", lambda: user)
    monkeypatch.setattr("hermes_cli.plugins.get_bundled_plugins_dir", lambda: bundled)
    monkeypatch.setattr(pc.importlib.metadata, "entry_points", lambda: eps)

    entries = {entry[5]: entry for entry in pc._discover_all_plugins()}

    assert entries["generic"][3] == "entrypoint"
    assert entries[KEY][0] == "acme-provider"
    assert entries[KEY][4] == "acme_provider:register"


def test_generic_entrypoint_inventory_accepts_list_api(monkeypatch):
    entry = SimpleNamespace(
        name="generic",
        value="generic:register",
        group="hermes_agent.plugins",
        dist=None,
    )
    monkeypatch.setattr(pc.importlib.metadata, "entry_points", lambda: [entry])

    assert pc._discover_entrypoint_plugins() == [
        ("generic", "", "", "generic:register")
    ]


def test_list_shows_package_opt_in_and_directory_default_on(monkeypatch, capsys):
    entries = [
        _entry(),
        _entry(
            name="local-provider",
            source="user",
            path=Path("/plugins/model-providers/local-provider"),
            key="model-providers/local-provider",
        ),
    ]
    monkeypatch.setattr(pc, "_discover_all_plugins", lambda: entries)
    monkeypatch.setattr(pc, "_get_enabled_set", lambda: set())
    monkeypatch.setattr(pc, "_get_disabled_set", lambda: set())

    pc.cmd_list(_args())

    rows = {row["name"]: row for row in json.loads(capsys.readouterr().out)}
    assert rows["acme-provider"]["status"] == "not enabled"
    assert rows["local-provider"]["status"] == "enabled"


@pytest.mark.parametrize("enabled", [True, False])
def test_activation_write_canonicalizes_aliases(monkeypatch, tmp_path, enabled):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    aliases = {KEY, "acme-provider", "acme"}
    config = home / "config.yaml"
    config.write_text(
        yaml.safe_dump({
            "plugins": {
                "enabled": ["unrelated", "acme", "acme-provider"],
                "disabled": ["old-deny", "acme", "acme-provider"],
            }
        })
    )

    pc._write_plugin_activation(KEY, aliases, enabled)

    plugins = yaml.safe_load(config.read_text())["plugins"]
    allow, deny = set(plugins["enabled"]), set(plugins["disabled"])
    assert not (allow | deny) & (aliases - {KEY})
    assert allow == ({"unrelated", KEY} if enabled else {"unrelated"})
    assert deny == ({"old-deny"} if enabled else {"old-deny", KEY})


def test_bulk_selection_is_canonical_and_preserves_unrelated(monkeypatch, tmp_path):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    config = home / "config.yaml"
    config.write_text(
        yaml.safe_dump({
            "plugins": {
                "enabled": ["unrelated", "acme-provider"],
                "disabled": ["old-deny", "local-provider"],
            }
        })
    )
    local_key = "model-providers/local-provider"
    entries = {
        KEY: _entry(),
        local_key: _entry(name="Local Provider", source="user", key=local_key),
    }
    monkeypatch.setattr(pc, "_resolve_plugin_entry", entries.get)

    pc._write_plugin_selection(list(entries), {local_key})

    plugins = yaml.safe_load(config.read_text())["plugins"]
    assert set(plugins["enabled"]) == {"unrelated", local_key}
    assert set(plugins["disabled"]) == {"old-deny", KEY}


@pytest.mark.parametrize(
    "operation", [pc.dashboard_update_user_plugin, pc.dashboard_remove_user_plugin]
)
def test_pip_provider_update_and_remove_are_pip_owned(monkeypatch, operation):
    monkeypatch.setattr(pc, "_resolve_plugin_entry", lambda _name: _entry())

    result = operation("acme-provider")

    assert result["ok"] is False
    assert "pip" in result["error"].lower()


@pytest.mark.parametrize(
    ("name", "source", "enabled", "disabled", "expected"),
    [
        ("Manifest Name", "user", set(), {"leaf"}, "disabled"),
        ("cloudflare", "entrypoint", {"cloudflare"}, set(), "not enabled"),
        ("cloudflare", "entrypoint", {"model-providers/leaf"}, set(), "enabled"),
    ],
)
def test_status_matches_runtime_identities(name, source, enabled, disabled, expected):
    assert (
        pc._plugin_status(
            name,
            enabled,
            disabled,
            key="model-providers/leaf",
            source=source,
        )
        == expected
    )


def test_toggle_selects_default_active_user_model_provider(monkeypatch):
    entry = _entry(name="Manifest Name", source="user", key="model-providers/leaf")
    run_ui = MagicMock()
    monkeypatch.setattr(pc, "_discover_all_plugins", lambda: [entry])
    monkeypatch.setattr(pc, "_get_enabled_set", lambda: set())
    monkeypatch.setattr(pc, "_get_disabled_set", lambda: set())
    monkeypatch.setattr(pc.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(pc, "_run_composite_ui", run_ui)

    pc.cmd_toggle()

    assert run_ui.call_args.args[3] == {0}
