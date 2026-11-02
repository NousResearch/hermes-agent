"""`hermes config get/set` must recognize keys the runtime reads but DEFAULT_CONFIG leaves unseeded:
core provider selections (``browser.cloud_provider``, ``<category>.use_gateway``), MoA preset fields
(``moa.presets.<name>.fanout``), core-owned keys with a known reader (``computer_use.backend``), and keys
an ACTIVE plugin declares through ``config_keys``.
Declarations never weaken the schema's typo notices."""

import argparse
import json
import os
from unittest.mock import patch

import pytest

from hermes_cli.config import config_command, set_config_value

NOTICE = "not a recognized config key"


@pytest.fixture(autouse=True)
def hermes_home(tmp_path):
    home = tmp_path / "hermes"
    home.mkdir()
    (home / ".env").touch()
    with patch.dict(os.environ, {"HERMES_HOME": str(home)}):
        os.environ.pop("HERMES_ENABLE_PROJECT_PLUGINS", None)
        yield home


def _write_config(home, data):
    (home / "config.yaml").write_text(json.dumps(data))


def _plugin(root, name, config_keys, init_source=None):
    plugin = root / name
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(json.dumps({"name": name, "config_keys": config_keys}))
    if init_source is not None:
        (plugin / "__init__.py").write_text(init_source)
    return plugin


def _set_notice(key, value, capsys):
    set_config_value(key, value)
    out = capsys.readouterr().out
    assert f"Set {key}" in out
    return NOTICE in out


def test_config_get_browser_selection_keys_print_value_without_notice(hermes_home, capsys):
    _write_config(hermes_home, {"browser": {"cloud_provider": "browser-use", "use_gateway": True}})

    for key, expected in (("browser.cloud_provider", "browser-use"), ("browser.use_gateway", True)):
        config_command(argparse.Namespace(config_command="get", key=key, json=True))
        captured = capsys.readouterr()
        assert json.loads(captured.out) == expected
        assert NOTICE not in captured.err, key


@pytest.mark.parametrize("key,value", [
    ("browser.cloud_provider", "browserbase"),   # core: tools.tool_backend_helpers.read_selection
    ("web.use_gateway", "true"),                 # core: legacy managed-gateway flag, every category
    ("dashboard.kanban.default_tenant", "t1"),   # core: bundled kanban dashboard preferences
])
def test_config_set_runtime_read_key_is_silent(key, value, hermes_home, capsys):
    assert not _set_notice(key, value, capsys)


def _moa_key(name, field):
    return "moa.presets." + name.replace(".", r"\.") + "." + field


@pytest.mark.parametrize("name", ["default", "review", "review.v2", "code review"])
def test_moa_preset_fields_set_and_get_without_notice(name, hermes_home, capsys):
    from hermes_cli.config import get_config_value, load_config
    from hermes_cli.moa_config import resolve_moa_preset

    slot = {"provider": "openrouter", "model": "test/actor"}
    _write_config(hermes_home, {"moa": {"presets": {name: {"reference_models": [slot], "aggregator": slot}}}})
    for field, value, expected in (("fanout", "every_n:5", "every_n:5"), ("reference_timeout", "900", 900)):
        assert not _set_notice(_moa_key(name, field), value, capsys), field
        get_config_value(_moa_key(name, field), as_json=True)
        captured = capsys.readouterr()
        assert json.loads(captured.out) == expected
        assert NOTICE not in captured.err, field
    preset = resolve_moa_preset(load_config()["moa"], name)
    assert (preset["fanout"], preset["reference_timeout"]) == ("every_n:5", 900)


@pytest.mark.parametrize("name", ["default", "review"])
def test_moa_aggregator_slot_fields_set_without_notice(name, hermes_home, capsys):
    from hermes_cli.config import load_config
    from hermes_cli.moa_config import resolve_moa_preset

    slot = {"provider": "openrouter", "model": "test/actor"}
    _write_config(hermes_home, {"moa": {"presets": {name: {"reference_models": [slot], "aggregator": slot}}}})
    for field, value in (("provider", "anthropic"), ("model", "test/judge"), ("reasoning_effort", "high")):
        assert not _set_notice(_moa_key(name, "aggregator." + field), value, capsys), field
    aggregator = resolve_moa_preset(load_config()["moa"], name)["aggregator"]
    assert aggregator == {"provider": "anthropic", "model": "test/judge", "reasoning_effort": "high"}


@pytest.mark.parametrize("key", [
    pytest.param("moa.presets.default.fanoutt", id="field-typo"),
    pytest.param("moa.presets.review.reference_timeot", id="field-typo-custom"),
    pytest.param(r"moa.presets.review\.v2.fanoutt", id="field-typo-escaped-name"),
    pytest.param("moa.presets.review.unexpected.fanout", id="extra-segment-before-field"),
    pytest.param("moa.presets.review.fanout.unexpected", id="extra-segment-after-field"),
    # The normalizer strips names, so this entry would replace the ``review`` preset.
    pytest.param("moa.presets. review .fanout", id="space-padded-name"),
    pytest.param("moa.presets.review", id="whole-preset"),
    pytest.param("moa.presets.review.aggregator.bogus", id="unknown-slot-field"),
    pytest.param("moa.presets.review.enabled.x", id="scalar-field-subpath"),
    # ``n`` only takes effect beside ``mode: every_n``; alone it normalizes to user_turn.
    pytest.param("moa.presets.review.fanout.n", id="fanout-mapping-half"),
])
def test_moa_non_field_paths_still_notice(key, hermes_home, monkeypatch, capsys):
    # Only the schema walk and the MoA rule decide these keys; skip the plugin scan they would fall through to.
    monkeypatch.setattr("hermes_cli.config_runtime_keys.plugin_declared_config_keys", frozenset)
    assert _set_notice(key, "5", capsys)


def test_every_selection_category_key_set_is_what_read_selection_returns(hermes_home, capsys):
    from tools.tool_backend_helpers import read_selection, selection_config_keys

    name_keys: dict[str, set] = {}
    for dotted in selection_config_keys():
        section, key = dotted.split(".")
        name_keys.setdefault(section, set()).add(key)
    # Every `hermes tools` / Tool Gateway category read_selection serves.
    assert {"web", "image_gen", "video_gen", "tts", "stt", "browser"} <= set(name_keys)
    for section, keys in name_keys.items():
        key = next(k for k in ("cloud_provider", "backend", "provider") if k in keys)
        _write_config(hermes_home, {})
        assert not _set_notice(f"{section}.{key}", f"vendor-{section}", capsys)
        assert read_selection(section) == f"vendor-{section}"


def test_computer_use_backend_set_is_what_the_selector_reads(hermes_home, capsys):
    from plugins.computer_use import configured_backend_name

    assert not _set_notice("computer_use.backend", "widget-driver", capsys)
    assert configured_backend_name() == "widget-driver"


def test_plugin_config_keys_count_only_while_the_plugin_is_enabled(hermes_home, capsys):
    _plugin(hermes_home / "plugins", "widget-plugin", ["display.widget"])

    assert _set_notice("display.widget.size", "3", capsys)  # installed, not enabled
    _write_config(hermes_home, {"plugins": {"enabled": ["widget-plugin"]}})
    assert not _set_notice("display.widget.size", "4", capsys)
    # A sibling the plugin did not declare, and a typo of a core selection key, still warn.
    assert _set_notice("display.widgetx", "3", capsys)
    assert _set_notice("browser.cloud_providr", "browserbase", capsys)
    _write_config(hermes_home, {"plugins": {"enabled": ["widget-plugin"], "disabled": ["widget-plugin"]}})
    assert _set_notice("display.widget.size", "5", capsys)  # disabled wins


def test_portable_plugin_declares_config_keys_in_its_hermes_extension(hermes_home, capsys):
    plugin = hermes_home / "plugins" / "portwidget"
    plugin.mkdir(parents=True)
    (plugin / "plugin.json").write_text(json.dumps({
        "$schema": "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json", "name": "portwidget",
        "version": "1.0.0", "extensions": {"com.nousresearch.hermes": {"config_keys": ["display.portwidget"]}},
    }))

    assert _set_notice("display.portwidget.size", "3", capsys)
    _write_config(hermes_home, {"plugins": {"enabled": ["portwidget"]}})
    assert not _set_notice("display.portwidget.size", "4", capsys)


def test_plugin_cannot_silence_typos_in_a_whole_core_section(hermes_home, capsys):
    _plugin(hermes_home / "plugins", "hostile", ["agent", "agent.turn_liveness"])
    _write_config(hermes_home, {"plugins": {"enabled": ["hostile"]}})

    for typo, suggestion in (("agent.max_turnz", "agent.max_turns"),
                             ("agent.turn_liveness.timeout", "agent.turn_liveness.timeout_s")):
        set_config_value(typo, "1")
        out = capsys.readouterr().out
        assert NOTICE in out, typo
        assert suggestion in out


def test_selected_memory_provider_declares_its_config_keys(hermes_home, capsys):
    _plugin(hermes_home / "plugins", "widgetmem", ["memory.widgetmem"],
            init_source="from agent.memory_provider import MemoryProvider\n\nclass WidgetMem(MemoryProvider):\n    pass\n")

    assert _set_notice("memory.widgetmem.project", "p1", capsys)  # installed, not selected
    _write_config(hermes_home, {"memory": {"provider": "widgetmem"}})
    assert not _set_notice("memory.widgetmem.project", "p2", capsys)
    _write_config(hermes_home, {"memory": {"provider": "widgetmem"}, "plugins": {"disabled": ["widgetmem"]}})
    assert _set_notice("memory.widgetmem.project", "p3", capsys)  # selected but disabled


def test_enabled_project_plugin_declares_its_config_keys(hermes_home, tmp_path, monkeypatch, capsys):
    _plugin(tmp_path / "proj" / ".hermes" / "plugins", "widget-plugin", ["display.widget"])
    _write_config(hermes_home, {"plugins": {"enabled": ["widget-plugin"]}})
    monkeypatch.chdir(tmp_path / "proj")

    assert _set_notice("display.widget.size", "3", capsys)  # project plugins are opt-in
    monkeypatch.setenv("HERMES_ENABLE_PROJECT_PLUGINS", "1")
    assert not _set_notice("display.widget.size", "4", capsys)
