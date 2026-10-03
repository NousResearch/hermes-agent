"""A failed uninstall must leave a retained plugin's tool selection intact."""

import json
from pathlib import Path

import hermes_yaml as yaml
import pytest

from hermes_cli import plugins_cmd as pc
from hermes_cli.plugins import discover_plugins
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def _install(home):
    plugin = home / "plugins" / "removalprobe"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(
        'name: removalprobe\nversion: "1.0.0"\nprovides_tools: [removal_probe]\n',
        encoding="utf-8",
    )
    (plugin / "__init__.py").write_text(
        "def register(ctx):\n"
        "    ctx.register_tool(name='removal_probe', toolset='removal_tools',\n"
        "        schema={'name': 'removal_probe', 'parameters': {'type': 'object',\n"
        "        'properties': {}}}, handler=lambda *a, **kw: '{}')\n",
        encoding="utf-8",
    )
    (home / "config.yaml").write_text(
        "# Keep my tool selection if uninstall fails.\n"
        "plugins:\n  enabled: [removalprobe]\n"
        "  entries:\n    removalprobe: {custom: preserved}\n"
        "platform_toolsets:\n  cli: [removal_tools, terminal]\n"
        "  telegram: \"['removal_tools', 'web']\"\n",
        encoding="utf-8",
    )
    return plugin


@pytest.mark.parametrize("failure", ["tree", "metadata"])
@pytest.mark.parametrize("surface", ["cli", "dashboard"])
def test_failed_remove_preserves_retained_plugin(
    tmp_path, monkeypatch, failure, surface
):
    home = tmp_path / "home"
    plugin = _install(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    discover_plugins(force=True)
    assert pc._get_plugin_toolset_key("removalprobe") == "removal_tools"
    config = home / "config.yaml"
    before = config.read_bytes()
    metadata = home / "plugins" / ".install-metadata.json"

    def deny_write(*args, **kwargs):
        raise PermissionError("test removal boundary is not writable")

    if failure == "metadata":
        metadata.write_text(
            json.dumps({"removalprobe": {"revision": "a" * 40}}), encoding="utf-8"
        )
        monkeypatch.setattr(pc, "_write_install_metadata", deny_write)
    else:
        monkeypatch.setattr(pc, "rmtree_readonly", deny_write)

    if surface == "cli":
        with pytest.raises(SystemExit) as exc:
            pc.cmd_remove("removalprobe")
        assert exc.value.code == 1
    else:
        result = pc.dashboard_remove_user_plugin("removalprobe")
        assert result["ok"] is False and "not writable" in result["error"]

    assert (plugin / "plugin.yaml").is_file()
    assert config.read_bytes() == before
    if failure == "metadata":
        assert "removalprobe" in json.loads(metadata.read_text(encoding="utf-8-sig"))
        assert not list(plugin.parent.glob(".removalprobe.remove-*"))


def test_successful_remove_cleans_tool_selection_only_in_its_profile(
    tmp_path, monkeypatch
):
    home_a, home_b = tmp_path / "a", tmp_path / "b"
    plugin_a, plugin_b = _install(home_a), _install(home_b)
    monkeypatch.setenv("HERMES_HOME", str(home_a))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    before_a = (home_a / "config.yaml").read_bytes()

    for home in (home_a, home_b, home_a):
        token = set_hermes_home_override(home)
        try:
            discover_plugins(force=True)
            assert pc._get_plugin_toolset_key("removalprobe") == "removal_tools"
            if home == home_b:
                result = pc.dashboard_remove_user_plugin("removalprobe")
                assert result["ok"] is True
                config = yaml.safe_load(
                    (home / "config.yaml").read_text(encoding="utf-8-sig")
                )
                assert config["platform_toolsets"]["cli"] == ["terminal"]
                assert config["platform_toolsets"]["telegram"] == ["web"]
                assert "removalprobe" not in config["plugins"]["enabled"]
                assert "removalprobe" not in config["plugins"]["entries"]
                assert not plugin_b.exists()
            else:
                assert plugin_a.is_dir()
                assert (home / "config.yaml").read_bytes() == before_a
        finally:
            reset_hermes_home_override(token)
