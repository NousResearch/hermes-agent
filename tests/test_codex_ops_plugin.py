from __future__ import annotations

import importlib.util
import json
import sys
import tomllib
from pathlib import Path


def _load_core_module():
    root = Path(__file__).resolve().parents[1]
    pkg_dir = root / "plugins" / "codex_ops"
    module_name = "_test_codex_ops_core"
    spec = importlib.util.spec_from_file_location(
        module_name,
        pkg_dir / "core.py",
        submodule_search_locations=[str(pkg_dir)],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_redact_text_masks_high_confidence_tokens():
    core = _load_core_module()
    openai_like = "sk-" + ("a" * 24)
    github_like = "ghp_" + ("b" * 24)
    text = f"api_key={openai_like} token={github_like}"
    redacted = core.redact_text(text)

    assert openai_like not in redacted
    assert github_like not in redacted
    assert "[REDACTED" in redacted


def test_compact_output_preserves_head_tail_and_signal_lines(monkeypatch):
    core = _load_core_module()
    monkeypatch.setattr(core, "_cfg_bool", lambda key, default: default)
    monkeypatch.setattr(
        core,
        "_cfg_int",
        lambda key, default, minimum=1, maximum=1_000_000: {
            "compact_threshold_chars": 100,
            "compact_head_lines": 2,
            "compact_tail_lines": 2,
            "compact_signal_limit": 5,
        }.get(key, default),
    )
    output = "\n".join(["head-1", "head-2"] + [f"noise-{i}" for i in range(20)] + ["ERROR important failure", "tail-1", "tail-2"])
    compacted = core.compact_output("codex exec --cd . --sandbox workspace-write 'do work'", output, returncode=1)
    assert "[codex-ops] terminal output compacted" in compacted
    assert "head-1" in compacted
    assert "ERROR important failure" in compacted
    assert "tail-2" in compacted
    assert "omitted" in compacted


def test_compact_output_respects_zero_signal_limit(monkeypatch):
    core = _load_core_module()
    monkeypatch.setattr(core, "_cfg_bool", lambda key, default: default)
    monkeypatch.setattr(
        core,
        "_cfg_int",
        lambda key, default, minimum=1, maximum=1_000_000: {
            "compact_threshold_chars": 100,
            "compact_head_lines": 2,
            "compact_tail_lines": 2,
            "compact_signal_limit": 0,
        }.get(key, default),
    )
    output = "\n".join(["head-1", "head-2"] + [f"ERROR hidden-{i}" for i in range(20)] + ["tail-1", "tail-2"])
    compacted = core.compact_output("codex exec --cd . --sandbox workspace-write 'do work'", output, returncode=1)
    assert "signal lines outside head/tail" not in compacted
    assert "ERROR hidden-0" not in compacted
    assert "tail-2" in compacted


def test_guard_blocks_codex_danger_full_access(monkeypatch):
    core = _load_core_module()
    monkeypatch.setattr(core, "_cfg_bool", lambda key, default: False if key == "allow_danger_full_access" else default)
    commands = [
        "codex exec --cd /repo --danger-full-access 'fix it'",
        "codex exec --cd /repo --dangerously-bypass-approvals-and-sandbox 'fix it'",
        "codex exec --cd /repo --sandbox danger-full-access 'fix it'",
        "codex exec --cd /repo --sandbox=danger-full-access 'fix it'",
        "codex exec --cd /repo -s danger-full-access 'fix it'",
        "codex exec --cd /repo -s=danger-full-access 'fix it'",
        "codex exec --cd /repo -sdanger-full-access 'fix it'",
        "codex exec --cd /repo --yolo 'fix it'",
    ]
    for command in commands:
        decision = core.guard_pre_tool_call(tool_name="terminal", args={"command": command})
        assert decision is not None, command
        assert decision["action"] == "block"
        assert "workspace-write" in decision["message"]


def test_guard_allows_codex_danger_full_access_when_explicit(monkeypatch):
    core = _load_core_module()
    monkeypatch.setattr(core, "_cfg_bool", lambda key, default: True if key == "allow_danger_full_access" else default)
    decision = core.guard_pre_tool_call(
        tool_name="terminal",
        args={"command": "codex exec --cd /repo --sandbox danger-full-access 'fix it'"},
    )
    assert decision is None


def test_guard_ignores_non_codex_commands(monkeypatch):
    core = _load_core_module()
    monkeypatch.setattr(core, "_cfg_bool", lambda key, default: False if key == "allow_danger_full_access" else default)
    decision = core.guard_pre_tool_call(
        tool_name="terminal",
        args={"command": "python -m pytest --sandbox danger-full-access"},
    )
    assert decision is None


def test_record_tool_call_writes_profile_scoped_ledger(tmp_path, monkeypatch):
    core = _load_core_module()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(core, "_cfg_bool", lambda key, default: False)
    result = json.dumps({"output": "Traceback: boom\n", "exit_code": 1, "error": None})
    command = "codex exec --cd /repo --sandbox workspace-write 'fix tests with proprietary roadmap details'"
    core.record_tool_call(
        tool_name="terminal",
        args={"command": command, "workdir": "/repo"},
        result=result,
        duration_ms=123,
        status="error",
        session_id="s1",
    )
    info = core.status()
    assert info["total"] == 1
    rows = core.list_runs(limit=5)
    assert rows[0]["exit_code"] == 1
    shown = core.get_run(rows[0]["id"])
    assert shown is not None
    assert "Traceback" in shown["summary"]
    assert "proprietary roadmap details" not in shown["command"]
    assert "<prompt sha256=" in shown["command"]
    assert "--sandbox workspace-write" in shown["command"]
    assert str(tmp_path) in info["db"]


def test_plugin_integration_registers_hooks_and_cli(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "_config_version: 32\nplugins:\n  enabled:\n    - codex-ops\n",
        encoding="utf-8",
    )

    import hermes_cli.config as config
    import hermes_cli.plugins as plugins

    config._LOAD_CONFIG_CACHE.clear()
    config._RAW_CONFIG_CACHE.clear()
    monkeypatch.setattr(plugins, "_plugin_manager", None)

    manager = plugins.get_plugin_manager()
    manager.discover_and_load(force=True)

    loaded = manager._plugins.get("codex-ops")
    assert loaded is not None
    assert loaded.enabled is True
    assert loaded.error is None
    assert manager.has_hook("pre_tool_call")
    assert manager.has_hook("post_tool_call")
    assert manager.has_hook("transform_terminal_output")
    assert "codex-ops" in manager._cli_commands
    skill_path = manager.find_plugin_skill("codex-ops:codex-operations")
    assert skill_path is not None
    assert skill_path.name == "SKILL.md"

    block = plugins.get_pre_tool_call_block_message(
        "terminal",
        {"command": "codex exec --cd /repo --sandbox danger-full-access 'fix'"},
    )
    assert block is not None
    assert "no-sandbox" in block


def test_packaging_includes_plugin_skill_data():
    root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    plugin_package_data = pyproject["tool"]["setuptools"]["package-data"]["plugins"]

    assert "**/skills/**/*.md" in plugin_package_data
    assert (root / "plugins" / "codex_ops" / "skills" / "codex-operations" / "SKILL.md").is_file()
