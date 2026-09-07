"""A manifest-name alias cannot replace the canonical enablement target."""
from pathlib import Path

import pytest

from hermes_cli import plugins_cmd as cmd


@pytest.mark.parametrize("name", ["wanted-plugin", "b/target"])
@pytest.mark.parametrize("surface", ["dashboard", "cli"])
def test_setup_gate_keeps_the_resolved_canonical_plugin(tmp_path, monkeypatch, name, surface):
    home = tmp_path / "profile"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(cmd, "_is_tty", lambda: False)
    alias = home / "plugins/a/alias"
    target = home / "plugins/b/target"
    alias.mkdir(parents=True)
    target.mkdir(parents=True)
    (alias / "plugin.yaml").write_text("name: b/target\n", encoding="utf-8")
    (target / "plugin.yaml").write_text(
        "name: wanted-plugin\nsetup: {entrypoint: setup.py}\n", encoding="utf-8"
    )
    (target / "setup.py").write_text(
        'def describe(home):\n'
        '    return {"revision": "v1", "ready": False, "summary": "Setup required", "details": []}\n'
        'def run(home):\n'
        '    raise AssertionError("No setup consent was granted")\n',
        encoding="utf-8",
    )

    if surface == "cli":
        with pytest.raises(SystemExit):
            cmd.cmd_enable(name, allow_tool_override=False)
    else:
        result = cmd.dashboard_set_agent_plugin_enabled(name, enabled=True)
        assert result.get("status") == "consent_required", result
        assert result["consent"]["key"] == "b/target"
    assert not cmd._get_enabled_set()
    assert not (home / "config.yaml").exists()
