"""Plugin activation writes and reads the SAME key on every surface (#27548, #73131, #82898).

Runs the real ``hermes_cli.plugins_cmd`` commands against a temp HERMES_HOME and checks the verdict
through the loader's own gate (``plugins_discovery.gate_manifest``) — never by re-reading the list
the command just wrote.
"""

from __future__ import annotations

import pytest

from hermes_cli import plugins_cmd
from hermes_cli.config import load_config, save_config
from hermes_cli.plugins_discovery import collect_directory_manifests, gate_manifest


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
    (hermes_home / "config.yaml").write_text("plugins:\n  enabled: []\n  disabled: []\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    return hermes_home


@pytest.fixture(autouse=True)
def _commit_plugin_selection_without_building_an_environment(monkeypatch):
    """These tests cover selection keys; PM's real publication path has its own integration tests."""
    def admit(enabled, disabled, **_kwargs):
        cfg = load_config()
        cfg["plugins"] = {"enabled": sorted(enabled), "disabled": sorted(disabled)}
        save_config(cfg)

    monkeypatch.setattr("hermes_cli.plugins_admission.admit_plugin_set_change", admit)


def _lists():
    plugins = load_config().get("plugins") or {}
    return set(plugins.get("enabled") or []), set(plugins.get("disabled") or [])


def test_disable_bundled_platform_by_manifest_name_gates_the_loader(home):
    """`hermes plugins disable photon-platform` must produce a key the loader's gate matches; the
    CLI wrote ``platforms/photon`` while discovery keyed the adapter ``photon-platform``."""
    plugins_cmd.cmd_disable("photon-platform")

    enabled, disabled = _lists()
    photon = next(m for m in collect_directory_manifests() if m.name == "photon-platform")
    gate = gate_manifest(photon, disabled, enabled)
    assert gate.action == "placeholder" and gate.error == "disabled via config"
    # And the key the CLI persists IS the loader's key — not merely an alias of it.
    assert photon.key in disabled


def test_dashboard_toggle_writes_canonical_key_and_clears_stale_aliases(home):
    """A toggle by bare leaf or manifest name must land on the canonical key; a stale key in
    ``disabled`` outranks a manifest-name entry in ``enabled``, so writing the raw identifier
    reported success while the plugin stayed off."""
    _write_plugin(home / "plugins", "obs/zzprobe", "zz-probe-manifest")
    cfg = load_config()
    cfg["plugins"]["disabled"] = ["obs/zzprobe"]
    save_config(cfg)

    result = plugins_cmd.dashboard_set_agent_plugin_enabled("zzprobe", enabled=True)

    # The enable also loads the plugin now (#87770): with no gateway answering, a restart is still the
    # honest hint and the activation summary rides along.
    assert {k: result[k] for k in ("ok", "name", "unchanged", "restart_required")} == {
        "ok": True, "name": "obs/zzprobe", "unchanged": False, "restart_required": True}
    assert result["gateway_reloaded"] is False
    enabled, disabled = _lists()
    assert enabled == {"obs/zzprobe"} and disabled == set()
    manifest = next(m for m in collect_directory_manifests() if m.name == "zz-probe-manifest")
    assert gate_manifest(manifest, disabled, enabled).action == "load"
    # Same state requested again by manifest name: nothing to write, no restart to announce.
    again = plugins_cmd.dashboard_set_agent_plugin_enabled("zz-probe-manifest", enabled=True)
    assert again["unchanged"] is True and again["restart_required"] is False


def test_status_reports_bundled_defaults_and_the_live_memory_provider(home):
    """Bundled backends (auto-load) and the plugin selected by ``memory.provider`` run without a
    ``plugins.enabled`` entry; status must not call them "not enabled" (#73131, #82898)."""
    _write_plugin(home / "plugins", "fakemem", "fakemem", "kind: exclusive\n")
    cfg = load_config()
    cfg["memory"] = {"provider": "fakemem"}
    save_config(cfg)
    enabled, disabled = plugins_cmd._get_enabled_set(), plugins_cmd._get_disabled_set()
    active = plugins_cmd._category_active_names()

    by_key = {e[5]: e for e in plugins_cmd._discover_all_plugins()}
    for key in ("web/firecrawl", "fakemem"):
        name, _v, _d, source, dir_path, _k = by_key[key]
        status = plugins_cmd._plugin_status(name, enabled, disabled, key=key, source=source, dir_path=dir_path,
                                            active=active)
        assert status == "enabled", key
    # An explicit disable still wins over both defaults.
    assert plugins_cmd._plugin_status("fakemem", enabled, {"fakemem"}, key="fakemem", source="user",
                                      active=active) == "disabled"


def test_status_matches_the_loader_for_a_user_installed_model_provider(home):
    """``gate_manifest`` loads a model provider from any source without a ``plugins.enabled`` entry, so
    status (and the Desktop Plugins switch) must not report a user-installed one as "not enabled"; a
    user standalone plugin stays opt-in."""
    _write_plugin(home / "plugins", "fakeprov", "fakeprov", "kind: model-provider\n")
    _write_plugin(home / "plugins", "fakestand", "fakestand", "")
    enabled, disabled = plugins_cmd._get_enabled_set(), plugins_cmd._get_disabled_set()
    by_key = {e[5]: e for e in plugins_cmd._discover_all_plugins()}
    status = {}
    for key in ("fakeprov", "fakestand"):
        name, _v, _d, source, dir_path, _k = by_key[key]
        status[key] = plugins_cmd._plugin_status(name, enabled, disabled, key=key, source=source, dir_path=dir_path)
    assert status == {"fakeprov": "enabled", "fakestand": "not enabled"}
    name, _v, _d, source, dir_path, _k = by_key["fakeprov"]
    assert plugins_cmd._plugin_status(name, enabled, {"fakeprov"}, key="fakeprov", source=source,
                                      dir_path=dir_path) == "disabled"


def test_status_enables_a_user_model_provider_only_where_provider_discovery_imports_it(home):
    """providers/ discovery imports every ``plugins/model-providers/<name>/`` child, but a flat
    ``plugins/<name>/`` one only when it declares exactly ``kind: model-provider``. A provider in a
    category directory, or flat with a differently-cased kind, never loads, so status must not call
    it enabled. A non-provider under ``model-providers/`` is imported but never registered."""
    import providers

    for rel, kind in (("flatprov", "model-provider"), ("model-providers/subprov", "model-provider"),
                      ("model-providers/subcaps", "MODEL-PROVIDER"), ("custom/nestedprov", "model-provider"),
                      ("model-providers/substand", "standalone"), ("capsprov", "MODEL-PROVIDER")):
        d = _write_plugin(home / "plugins", rel, rel.rsplit("/", 1)[-1], f"kind: {kind}\n")
        (d / "__init__.py").write_text(f"open({str(d / 'imported')!r}, 'w').close()\n", encoding="utf-8")
    providers._scan_home_layer(providers._HomeLayer(), str(home))

    rows = {e[5]: e for e in plugins_cmd._discover_all_plugins() if e[3] == "user"}
    imported = {key: (row[4] / "imported").exists() for key, row in rows.items()}
    assert imported == {"flatprov": True, "model-providers/subprov": True, "model-providers/subcaps": True,
                        "model-providers/substand": True, "custom/nestedprov": False, "capsprov": False}
    status = {key: plugins_cmd._plugin_status(row[0], set(), set(), key=key, source=row[3], dir_path=row[4])
              == "enabled" for key, row in rows.items()}
    assert status == {**imported, "model-providers/substand": False}
