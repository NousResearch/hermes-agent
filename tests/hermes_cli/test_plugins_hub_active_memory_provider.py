"""Dashboard plugins tab must not report the live memory provider as inactive.

Regression test for #82898. `_merged_plugins_hub()` powers
``/api/dashboard/plugins/hub``; it derived ``runtime_status`` purely from
``plugins.enabled`` / ``plugins.disabled``, so a plugin selected through
``memory.provider`` — which populates neither set — rendered as ``inactive``
with an "Enable" button while it was already serving memory.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from hermes_cli import plugins_cmd, web_server
from tools import registry as tools_registry


def _patch_hub(monkeypatch, *, rows, active_provider_dir, enabled=None, disabled=None):
    monkeypatch.setattr(web_server, "_get_dashboard_plugins", lambda force_rescan=False: [])
    monkeypatch.setattr(web_server, "_discover_memory_provider_statuses", lambda: [])
    monkeypatch.setattr(web_server, "get_hermes_home", lambda: Path("/tmp/hermes-home"))
    monkeypatch.setattr(
        web_server, "load_config", lambda: {"dashboard": {"hidden_plugins": []}}
    )

    monkeypatch.setattr(plugins_cmd, "_discover_all_plugins", lambda: list(rows))
    monkeypatch.setattr(plugins_cmd, "_get_current_context_engine", lambda: "compressor")
    monkeypatch.setattr(plugins_cmd, "_get_current_memory_provider", lambda: "mnemosyne")
    monkeypatch.setattr(plugins_cmd, "_discover_context_engines", lambda: [])
    monkeypatch.setattr(plugins_cmd, "_get_disabled_set", lambda: disabled or set())
    monkeypatch.setattr(plugins_cmd, "_get_enabled_set", lambda: enabled or set())
    monkeypatch.setattr(
        plugins_cmd, "_active_memory_provider_dir", lambda: active_provider_dir
    )
    monkeypatch.setattr(
        plugins_cmd, "_read_manifest", lambda _path: {"provides_tools": []}
    )
    monkeypatch.setattr(
        tools_registry.registry,
        "get_entry",
        lambda _name: SimpleNamespace(check_fn=None),
    )
    web_server._invalidate_plugins_hub_cache()


def _row_for(name):
    return {row["name"]: row for row in name["plugins"]}


def test_active_memory_provider_is_not_reported_inactive(monkeypatch, tmp_path):
    provider_dir = tmp_path / "plugins" / "mnemosyne"
    provider_dir.mkdir(parents=True)
    rows = [
        ("mnemosyne-hermes", "0.5.0", "Memory", "user", str(provider_dir), "mnemosyne-hermes"),
    ]
    _patch_hub(monkeypatch, rows=rows, active_provider_dir=provider_dir)

    payload = web_server._merged_plugins_hub(force_refresh=True)

    row = _row_for(payload)["mnemosyne-hermes"]
    # "active" is a distinct state, typed in web/src/lib/api.ts alongside
    # disabled/enabled/inactive. It must not be "enabled": the Plugins page
    # offers Disable for enabled rows, and disabling only edits
    # `plugins.enabled` — it would not stop the provider.
    assert row["runtime_status"] == "active"


def test_unrelated_plugin_still_reports_inactive(monkeypatch, tmp_path):
    provider_dir = tmp_path / "plugins" / "mnemosyne"
    other_dir = tmp_path / "plugins" / "unrelated"
    provider_dir.mkdir(parents=True)
    other_dir.mkdir(parents=True)
    rows = [
        ("mnemosyne-hermes", "0.5.0", "Memory", "user", str(provider_dir), "mnemosyne-hermes"),
        ("unrelated", "1.0.0", "Other", "user", str(other_dir), "unrelated"),
    ]
    _patch_hub(monkeypatch, rows=rows, active_provider_dir=provider_dir)

    payload = web_server._merged_plugins_hub(force_refresh=True)

    by_name = _row_for(payload)
    assert by_name["mnemosyne-hermes"]["runtime_status"] == "active"
    assert by_name["unrelated"]["runtime_status"] == "inactive"


def test_explicitly_disabled_provider_still_reports_disabled(monkeypatch, tmp_path):
    """An explicit opt-out keeps precedence over the active-provider signal."""
    provider_dir = tmp_path / "plugins" / "mnemosyne"
    provider_dir.mkdir(parents=True)
    rows = [
        ("mnemosyne-hermes", "0.5.0", "Memory", "user", str(provider_dir), "mnemosyne-hermes"),
    ]
    _patch_hub(
        monkeypatch,
        rows=rows,
        active_provider_dir=provider_dir,
        disabled={"mnemosyne-hermes"},
    )

    payload = web_server._merged_plugins_hub(force_refresh=True)

    assert _row_for(payload)["mnemosyne-hermes"]["runtime_status"] == "disabled"


def test_e2e_hub_resolves_provider_without_mocking_the_lookup(monkeypatch, tmp_path):
    """End-to-end: the hub resolves the provider itself, against a real tree.

    Only the dashboard-extension scan and the tool-registry probe are stubbed
    (unrelated scaffolding, and the probe must stay off the request path).
    Plugin discovery, config loading and the whole `memory.provider` ->
    `find_provider_dir` chain run for real, so this cannot pass on a mocked
    lookup alone.
    """
    home = tmp_path / "hermes_home"
    plugin_dir = home / "plugins" / "mnemosyne"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        "name: mnemosyne-hermes\nversion: 0.5.0\ndescription: Memory provider\n",
        encoding="utf-8",
    )
    (plugin_dir / "__init__.py").write_text(
        "from plugins.memory import MemoryProvider\n\n\n"
        "class Provider(MemoryProvider):\n    pass\n",
        encoding="utf-8",
    )
    (home / "config.yaml").write_text(
        "memory:\n  provider: mnemosyne\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    monkeypatch.setattr(web_server, "_get_dashboard_plugins", lambda force_rescan=False: [])
    monkeypatch.setattr(web_server, "_discover_memory_provider_statuses", lambda: [])
    monkeypatch.setattr(
        tools_registry.registry,
        "get_entry",
        lambda _name: SimpleNamespace(check_fn=None),
    )
    web_server._invalidate_plugins_hub_cache()

    payload = web_server._merged_plugins_hub(force_refresh=True)

    row = _row_for(payload)["mnemosyne-hermes"]
    assert row["runtime_status"] == "active"


def test_no_configured_provider_leaves_rows_inactive(monkeypatch, tmp_path):
    other_dir = tmp_path / "plugins" / "unrelated"
    other_dir.mkdir(parents=True)
    rows = [("unrelated", "1.0.0", "Other", "user", str(other_dir), "unrelated")]
    _patch_hub(monkeypatch, rows=rows, active_provider_dir=None)

    payload = web_server._merged_plugins_hub(force_refresh=True)

    assert _row_for(payload)["unrelated"]["runtime_status"] == "inactive"


def test_provider_also_in_enabled_set_agrees_with_the_cli(monkeypatch, tmp_path):
    """The hub and the CLI must not disagree when a provider is *also* enabled.

    A plugin can be listed in `plugins.enabled` and selected through
    `memory.provider` at once. The CLI's `_plugin_status` checks the provider
    directory before `plugins.enabled`, so it reports "active"; the hub checked
    `plugins.enabled` first, so it reported "enabled" and the Plugins page put
    the Disable button back — the exact control this change removes, since
    disabling only edits `plugins.enabled` and would not stop the provider.
    """
    provider_dir = tmp_path / "plugins" / "mnemosyne"
    provider_dir.mkdir(parents=True)
    rows = [
        ("mnemosyne-hermes", "0.5.0", "Memory", "user", str(provider_dir), "mnemosyne-hermes"),
    ]
    _patch_hub(
        monkeypatch,
        rows=rows,
        active_provider_dir=provider_dir,
        enabled={"mnemosyne-hermes"},
    )

    payload = web_server._merged_plugins_hub(force_refresh=True)
    hub_status = _row_for(payload)["mnemosyne-hermes"]["runtime_status"]

    cli_status = plugins_cmd._plugin_status(
        "mnemosyne-hermes",
        {"mnemosyne-hermes"},
        set(),
        key="mnemosyne-hermes",
        dir_path=provider_dir,
        active_provider_dir=provider_dir,
    )

    assert cli_status == plugins_cmd.PLUGIN_STATUS_ACTIVE
    assert hub_status == cli_status
