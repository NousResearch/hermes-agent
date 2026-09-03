from types import SimpleNamespace
from unittest.mock import MagicMock

import hermes_cli.memory_setup as memory_setup
from hermes_cli.memory_setup import _CANCELLED, _curses_select








def test_cmd_setup_generic_choice_cancel_writes_nothing(tmp_path, monkeypatch):
    class ChoiceProvider:
        def __init__(self):
            self.save_config = MagicMock()

        def get_config_schema(self):
            return [{
                "key": "mode",
                "description": "Mode",
                "default": "one",
                "choices": ["one", "two"],
            }]

    provider = ChoiceProvider()
    selections = iter([0, _CANCELLED])
    save_config = MagicMock()
    install_dependencies = MagicMock()

    monkeypatch.setattr(memory_setup, "_get_available_providers", lambda: [("fake", "local", provider)])
    monkeypatch.setattr(memory_setup, "_curses_select", lambda *args, **kwargs: next(selections))
    monkeypatch.setattr(memory_setup, "_install_dependencies", install_dependencies)
    monkeypatch.setattr(memory_setup, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"memory": {}})
    monkeypatch.setattr("hermes_cli.config.save_config", save_config)

    memory_setup.cmd_setup(SimpleNamespace())

    install_dependencies.assert_called_once_with("fake")
    save_config.assert_not_called()
    provider.save_config.assert_not_called()
    assert not (tmp_path / ".env").exists()


# _write_env_vars's CR/LF-stripping, denylist, and plain-value-roundtrip
# behavior is covered by tests/hermes_cli/test_memory_setup_env_denylist.py,
# which exercises the current save_env_value-routed signature
# (env_writes, hermes_home=None) \u2014 these three tests pinned the prior direct
# Path.write_text(env_path, env_writes) signature/implementation and were
# removed along with it (#60587).


# ---------------------------------------------------------------------------
# _provider_pip_dependencies — mode-aware dep expansion (#70636)
# ---------------------------------------------------------------------------





def test_install_dependencies_force_reinstalls_versioned_specs(tmp_path, monkeypatch):
    """force=True hands every declared spec (version ranges intact) to pip,
    so a downgraded/stripped bridge package is restored on hermes update."""
    import yaml as _yaml

    plugin_dir = tmp_path / "mem0"
    plugin_dir.mkdir()
    (plugin_dir / "plugin.yaml").write_text(
        _yaml.safe_dump({"pip_dependencies": ["mem0ai>=2.0.10,<3"]}), encoding="utf-8"
    )
    monkeypatch.setattr(
        "plugins.memory.find_provider_dir", lambda name: plugin_dir
    )

    installed = []

    def fake_install_specs(specs, timeout=120):
        installed.append(list(specs))
        return SimpleNamespace(ok=True, blocked=False, reason="", stderr="")

    monkeypatch.setattr("tools.lazy_deps.install_specs", fake_install_specs)

    memory_setup._install_dependencies("mem0", force=True)

    assert installed, "force=True must reach the install step"
    assert any("mem0ai>=2.0.10,<3" in specs for specs in installed)


def test_install_dependencies_reads_plugin_yml(tmp_path, monkeypatch):
    """Setup must install dependencies from plugin.yml providers."""
    import yaml as _yaml

    plugin_dir = tmp_path / "ymlprovider"
    plugin_dir.mkdir()
    (plugin_dir / "plugin.yml").write_text(
        _yaml.safe_dump({"pip_dependencies": ["yml-dependency>=1,<2"]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "plugins.memory.find_provider_dir", lambda name: plugin_dir
    )

    installed = []
    monkeypatch.setattr(
        "tools.lazy_deps.install_specs",
        lambda specs, timeout=120: installed.append(list(specs))
        or SimpleNamespace(ok=True, blocked=False, reason="", stderr=""),
    )

    memory_setup._install_dependencies("ymlprovider", force=True)

    assert installed == [["yml-dependency>=1,<2"]]


def test_plugin_yml_description_reaches_memory_setup(tmp_path, monkeypatch):
    """Setup picker must show plugin.yml description metadata."""
    plugin_dir = tmp_path / "plugins" / "ymlprovider"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yml").write_text(
        "name: ymlprovider\ndescription: YAML description\n",
        encoding="utf-8",
    )
    (plugin_dir / "__init__.py").write_text(
        "from agent.memory_provider import MemoryProvider\n"
        "class YmlProvider(MemoryProvider):\n"
        "    @property\n"
        "    def name(self): return 'ymlprovider'\n"
        "    def is_available(self): return True\n"
        "    def initialize(self, **kw): pass\n"
        "    def sync_turn(self, *a, **kw): pass\n"
        "    def get_tool_schemas(self): return []\n"
        "    def handle_tool_call(self, *a, **kw): return '{}'\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "plugins.memory._get_user_plugins_dir",
        lambda: tmp_path / "plugins",
    )

    from plugins.memory import discover_memory_providers

    providers = discover_memory_providers()

    assert next(desc for name, desc, _ in providers if name == "ymlprovider") == "YAML description"


def test_plugin_yaml_wins_over_plugin_yml_setup_metadata(tmp_path, monkeypatch):
    """Setup must use plugin.yaml before plugin.yml, matching discovery."""
    import yaml as _yaml

    plugin_dir = tmp_path / "plugins" / "precedenceprovider"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        _yaml.safe_dump(
            {
                "name": "precedenceprovider",
                "description": "YAML description",
                "pip_dependencies": ["yaml-dependency>=1,<2"],
            }
        ),
        encoding="utf-8",
    )
    (plugin_dir / "plugin.yml").write_text(
        _yaml.safe_dump(
            {
                "name": "precedenceprovider",
                "description": "YML description",
                "pip_dependencies": ["yml-dependency>=1,<2"],
            }
        ),
        encoding="utf-8",
    )
    (plugin_dir / "__init__.py").write_text(
        "from agent.memory_provider import MemoryProvider\n"
        "class PrecedenceProvider(MemoryProvider):\n"
        "    @property\n"
        "    def name(self): return 'precedenceprovider'\n"
        "    def is_available(self): return True\n"
        "    def initialize(self, **kw): pass\n"
        "    def sync_turn(self, *a, **kw): pass\n"
        "    def get_tool_schemas(self): return []\n"
        "    def handle_tool_call(self, *a, **kw): return '{}'\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "plugins.memory.find_provider_dir", lambda name: plugin_dir
    )
    monkeypatch.setattr(
        "plugins.memory._get_user_plugins_dir",
        lambda: tmp_path / "plugins",
    )

    installed = []
    monkeypatch.setattr(
        "tools.lazy_deps.install_specs",
        lambda specs, timeout=120: installed.append(list(specs))
        or SimpleNamespace(ok=True, blocked=False, reason="", stderr=""),
    )
    memory_setup._install_dependencies("precedenceprovider", force=True)
    from plugins.memory import discover_memory_providers

    providers = discover_memory_providers()

    assert installed == [["yaml-dependency>=1,<2"]]
    assert next(desc for name, desc, _ in providers if name == "precedenceprovider") == "YAML description"
def test_cmd_status_memory_tool_gate_disabled(capsys, monkeypatch):
    """When both memory stores are disabled, Memory status reports memory tool as disabled."""
    _cfg = {"memory": {"memory_enabled": False, "user_profile_enabled": False}}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: _cfg)
    # check_memory_requirements() reads the readonly loader, not load_config.
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly", lambda: _cfg, raising=False
    )
    monkeypatch.setattr(memory_setup, "_get_available_providers", lambda: [])

    memory_setup.cmd_status(SimpleNamespace())

    captured = capsys.readouterr().out
    assert "Memory tool:        disabled ✗" in captured
    assert "Memory injection:   disabled ✗" in captured
    assert "User profile:       disabled ✗" in captured


def test_cmd_status_memory_tool_gate_enabled(capsys, monkeypatch):
    """When at least one memory store is enabled, Memory status reports memory tool as enabled."""
    _cfg = {"memory": {"memory_enabled": True, "user_profile_enabled": False}}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: _cfg)
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly", lambda: _cfg, raising=False
    )
    monkeypatch.setattr(memory_setup, "_get_available_providers", lambda: [])

    memory_setup.cmd_status(SimpleNamespace())

    captured = capsys.readouterr().out
    assert "Memory tool:        enabled ✓" in captured
    assert "Memory injection:   enabled ✓" in captured
    assert "User profile:       disabled ✗" in captured
