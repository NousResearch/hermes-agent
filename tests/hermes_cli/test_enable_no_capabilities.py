"""Enable asks for declared privileges, not an unsolicited override grant."""
from unittest.mock import MagicMock

import pytest
import yaml


@pytest.mark.parametrize(
    "caps,flag,existing,answer,expected,prompts",
    [
        (None, None, None, "n", None, 0),
        ([], None, None, "n", None, 0),
        (None, True, None, "n", True, 0),
        (None, False, True, "n", False, 0),
        (None, None, True, "n", True, 0),
        (["tools.override"], None, None, "y", True, 1),
    ],
)
def test_enable_consent(tmp_path, monkeypatch, caps, flag, existing, answer, expected, prompts):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    plugin = tmp_path / "plugins" / "hook-only-probe"
    plugin.mkdir(parents=True)
    manifest = {"name": "hook-only-probe", "version": "0.1.0", "hooks": ["transform_llm_output"]}
    if caps is not None:
        manifest["capabilities"] = caps
    (plugin / "plugin.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    (plugin / "__init__.py").write_text("def register(ctx):\n    pass\n", encoding="utf-8")
    initial = {} if existing is None else {
        "plugins": {"entries": {"hook-only-probe": {"allow_tool_override": existing}}}
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(initial), encoding="utf-8")
    from hermes_cli import plugins_cmd

    console = MagicMock()
    console.input.return_value = answer
    monkeypatch.setattr(plugins_cmd, "_console", lambda: console)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    plugins_cmd.cmd_enable("hook-only-probe", allow_tool_override=flag)
    assert console.input.call_count == prompts
    config = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert "hook-only-probe" in config["plugins"]["enabled"]
    entry = config["plugins"].get("entries", {}).get("hook-only-probe", {})
    assert entry.get("allow_tool_override") is expected
