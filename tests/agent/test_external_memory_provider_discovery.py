from pathlib import Path
from unittest.mock import patch


def _write_memory_plugin(path: Path, *, description: str = "User plugin", cli: bool = False) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "plugin.yaml").write_text(f'name: {path.name}\ndescription: "{description}"\n')
    (path / "__init__.py").write_text(
        "from agent.memory_provider import MemoryProvider\n\n"
        "class DemoMemoryProvider(MemoryProvider):\n"
        "    @property\n"
        "    def name(self):\n"
        f"        return '{path.name}'\n\n"
        "    def is_available(self):\n"
        "        return True\n\n"
        "    def initialize(self, session_id, **kwargs):\n"
        "        return None\n\n"
        "    def get_tool_schemas(self):\n"
        "        return []\n\n"
        "def register(ctx):\n"
        "    ctx.register_memory_provider(DemoMemoryProvider())\n"
    )
    if cli:
        (path / "cli.py").write_text(
            "def demo_command(args):\n"
            "    return None\n\n"
            "def register_cli(subparser):\n"
            "    subparser.set_defaults(func=demo_command)\n"
        )


def test_discover_memory_providers_includes_user_plugins(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    user_plugins = hermes_home / "plugins"
    bundled = tmp_path / "bundled_memory"
    _write_memory_plugin(user_plugins / "user-memory", description="User-scoped provider")
    bundled.mkdir(parents=True)

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from plugins import memory as memory_plugins
    monkeypatch.setattr(memory_plugins, "_MEMORY_PLUGINS_DIR", bundled)

    providers = memory_plugins.discover_memory_providers()
    names = [name for name, _, _ in providers]

    assert "user-memory" in names


def test_load_memory_provider_falls_back_to_user_plugin_dir(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    user_plugins = hermes_home / "plugins"
    bundled = tmp_path / "bundled_memory"
    _write_memory_plugin(user_plugins / "user-memory")
    bundled.mkdir(parents=True)

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from plugins import memory as memory_plugins
    monkeypatch.setattr(memory_plugins, "_MEMORY_PLUGINS_DIR", bundled)

    provider = memory_plugins.load_memory_provider("user-memory")

    assert provider is not None
    assert provider.name == "user-memory"


def test_load_memory_provider_prefers_bundled_plugin_over_user_plugin(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    user_plugins = hermes_home / "plugins"
    bundled = tmp_path / "bundled_memory"
    _write_memory_plugin(user_plugins / "shared-memory", description="user copy")
    _write_memory_plugin(bundled / "shared-memory", description="bundled copy")

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from plugins import memory as memory_plugins
    monkeypatch.setattr(memory_plugins, "_MEMORY_PLUGINS_DIR", bundled)

    provider = memory_plugins.load_memory_provider("shared-memory")

    assert provider is not None
    assert provider.name == "shared-memory"
    providers = memory_plugins.discover_memory_providers()
    descriptions = {name: desc for name, desc, _ in providers}
    assert descriptions["shared-memory"] == "bundled copy"


def test_discover_plugin_cli_commands_uses_active_user_memory_plugin(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    user_plugins = hermes_home / "plugins"
    bundled = tmp_path / "bundled_memory"
    _write_memory_plugin(user_plugins / "user-memory", cli=True)
    bundled.mkdir(parents=True)
    (hermes_home / "config.yaml").write_text("memory:\n  provider: user-memory\n")

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from plugins import memory as memory_plugins
    monkeypatch.setattr(memory_plugins, "_MEMORY_PLUGINS_DIR", bundled)

    commands = memory_plugins.discover_plugin_cli_commands()

    assert len(commands) == 1
    assert commands[0]["name"] == "user-memory"


def test_install_dependencies_reads_user_plugin_manifest(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    user_plugin = hermes_home / "plugins" / "user-memory"
    _write_memory_plugin(user_plugin)
    (user_plugin / "plugin.yaml").write_text(
        'name: user-memory\npip_dependencies:\n  - fake-dep\n'
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from hermes_cli import memory_setup

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "fake_dep":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(memory_setup, "__import__", fake_import, raising=False)

    with patch("shutil.which", return_value="/usr/bin/uv"), \
         patch("subprocess.run") as mock_run:
        memory_setup._install_dependencies("user-memory")

    assert mock_run.called
    args = mock_run.call_args[0][0]
    assert args[:5] == ["/usr/bin/uv", "pip", "install", "--python", memory_setup.sys.executable]
    assert "fake-dep" in args
