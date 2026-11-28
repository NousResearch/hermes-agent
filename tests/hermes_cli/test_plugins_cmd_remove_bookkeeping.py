"""``plugins remove`` (CLI, dashboard, ``plugins.manage`` RPC) leaves no config residue and never
follows a symlink (#54336, plugin-scan L2-F1/F2/F5).

Real removal core against a temp HERMES_HOME; the RPC path goes through ``tui_gateway.server``.
"""

from __future__ import annotations

import pytest
import hermes_yaml as yaml

from hermes_cli import plugins_cmd
from tui_gateway import server


def _write_plugin(root, rel, name, extra=""):
    d = root / rel
    d.mkdir(parents=True)
    (d / "plugin.yaml").write_text(f"name: {name}\nversion: '1.0'\n{extra}", encoding="utf-8")
    (d / "__init__.py").write_text("def register(ctx):\n    pass\n", encoding="utf-8")
    return d


@pytest.fixture
def home(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes-home"
    (hermes_home / "plugins").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    return hermes_home


def _config(home):
    return yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8")) or {}


def test_remove_forgets_every_config_trace_and_resets_the_memory_provider(home):
    """After an uninstall, plugins.enabled/disabled/entries no longer name the plugin (a reinstall
    must start from the "Enable now?" decision) and a ``memory.provider`` pointing at it is reset —
    otherwise the next agent init re-clones the provider from the catalog, undoing the uninstall."""
    _write_plugin(home / "plugins", "fakemem", "fakemem", "kind: exclusive\n")
    (home / "config.yaml").write_text(yaml.safe_dump({
        "plugins": {"enabled": ["fakemem", "keep-me"], "disabled": ["other"],
                    "entries": {"fakemem": {"allow_tool_override": True}, "keep-me": {}}},
        "memory": {"provider": "fakemem"},
    }), encoding="utf-8")

    resp = server.handle_request({"id": "1", "method": "plugins.manage",
                                  "params": {"action": "remove", "name": "fakemem"}})

    assert resp["result"] == {"ok": True, "name": "fakemem", "cleared_memory_provider": True}
    assert not (home / "plugins" / "fakemem").exists()
    cfg = _config(home)
    assert cfg["plugins"]["enabled"] == ["keep-me"] and cfg["plugins"]["disabled"] == ["other"]
    assert cfg["plugins"]["entries"] == {"keep-me": {}}
    assert not cfg["memory"]["provider"]


def test_remove_clears_the_toolset_of_a_plugin_without_provides_tools(home):
    """``remove`` drops the plugin's toolset from a saved ``platform_toolsets`` list even when
    plugin.yaml never declared ``provides_tools``; otherwise ``hermes config check`` warns about an
    unknown toolset forever."""
    from tools.registry import registry

    d = _write_plugin(home / "plugins", "zznodecl", "zznodecl")
    (d / "__init__.py").write_text(
        "def register(ctx):\n"
        "    ctx.register_tool(name='zznodecl_tool', toolset='zznodecl_ts', handler=lambda args, **kw: '{}',\n"
        "                      schema={'name': 'zznodecl_tool', 'description': 'probe',\n"
        "                              'parameters': {'type': 'object', 'properties': {}}})\n",
        encoding="utf-8")
    (home / "config.yaml").write_text(yaml.safe_dump({
        "plugins": {"enabled": ["zznodecl"]},
        "platform_toolsets": {"cli": ["hermes-cli", "zznodecl_ts"], "telegram": ["hermes-telegram"]},
    }), encoding="utf-8")
    try:
        assert plugins_cmd.dashboard_remove_user_plugin("zznodecl")["ok"] is True
    finally:
        registry.deregister("zznodecl_tool")

    assert _config(home)["platform_toolsets"] == {"cli": ["hermes-cli"], "telegram": ["hermes-telegram"]}


def test_remove_of_a_symlink_inside_the_plugins_dir_unlinks_only_the_link(home):
    """A dev alias pointing at a sibling install resolves INSIDE the plugins dir, so the containment
    check passes; removing the alias must not delete the sibling (or its install metadata)."""
    victim = _write_plugin(home / "plugins", "victim", "victim")
    (home / "plugins" / "alias").symlink_to(victim, target_is_directory=True)
    (home / "plugins" / ".install-metadata.json").write_text('{"victim": {"pinned_sha": "%s"}}' % ("a" * 40),
                                                             encoding="utf-8")
    (home / "config.yaml").write_text("plugins:\n  enabled: [victim]\n", encoding="utf-8")

    result = plugins_cmd.dashboard_remove_user_plugin("alias")

    assert result == {"ok": True, "name": "alias"}
    assert not (home / "plugins" / "alias").is_symlink()
    assert (victim / "plugin.yaml").exists()
    assert "victim" in (home / "plugins" / ".install-metadata.json").read_text(encoding="utf-8")
    assert _config(home)["plugins"]["enabled"] == ["victim"]


@pytest.mark.parametrize(
    ("web", "remaining", "expected"),
    [
        (
            {"backend": "exa", "search_backend": " Test-Web ", "extract_backend": "firecrawl"},
            False,
            {"backend": "exa", "extract_backend": "firecrawl"},
        ),
        (
            {"backend": "test-web", "search_backend": "exa", "extract_backend": "test-web"},
            False,
            {"search_backend": "exa"},
        ),
        (
            {"backend": "test-web", "search_backend": "test-web", "extract_backend": "test-web"},
            True,
            {"backend": "test-web", "search_backend": "test-web", "extract_backend": "test-web"},
        ),
    ],
    ids=["search-overrides-shared", "shared-and-extract", "provider-still-installed"],
)
def test_remove_clears_only_web_selections_for_removed_providers(home, web, remaining, expected):
    from tools.tool_backend_helpers import read_web_capability_selection

    plugins = home / "plugins"
    manifest = "kind: backend\nprovides_web_providers: [test-web]\n"
    _write_plugin(plugins, "web-plugin", "web-plugin", manifest)
    if remaining:
        _write_plugin(plugins, "other-web-plugin", "other-web-plugin", manifest)
    (home / "config.yaml").write_text(
        yaml.safe_dump({"web": {**web, "keyless_fallback": False}}), encoding="utf-8",
    )

    assert plugins_cmd.dashboard_remove_user_plugin("web-plugin")["ok"] is True

    assert not (plugins / "web-plugin").exists()
    assert read_web_capability_selection("search") == expected.get("search_backend", expected.get("backend"))
    assert read_web_capability_selection("extract") == expected.get("extract_backend", expected.get("backend"))
    assert _config(home)["web"] == {**expected, "keyless_fallback": False}
