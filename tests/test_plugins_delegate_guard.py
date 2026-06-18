import importlib.util
import sys
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_plugin(name: str):
    plugin_path = ROOT / "plugins" / name / "__init__.py"
    module_name = f"test_plugin_{name.replace('-', '_')}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, plugin_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class HookCollector:
    def __init__(self):
        self.hooks = {}

    def register_hook(self, name, callback):
        self.hooks[name] = callback


def configure_tool_guard(plugin, cooldown_seconds=300):
    plugin._config = {
        "guarded_tools": ["claude-code", "claude code", "codex"],
        "failure_patterns": ["connection refused", "proxy error"],
        "cooldown_seconds": cooldown_seconds,
        "max_consecutive_failures": 1,
    }
    plugin._failure_regexes = None
    plugin._failure_state.clear()


def test_code_modification_guard_blocks_delegate_code_changes():
    plugin = load_plugin("code-modification-guard")

    result = plugin.on_pre_tool_call(
        tool_name="delegate_task",
        args={
            "goal": (
                "Please implement the fix in run_agent.py, update the function, "
                "and add tests for the changed behavior."
            )
        },
    )

    assert result["action"] == "block"
    assert "delegate_task" in result["message"]
    assert "codex exec" in result["message"]


def test_code_modification_guard_allows_research_tasks():
    plugin = load_plugin("code-modification-guard")

    result = plugin.on_pre_tool_call(
        tool_name="delegate_task",
        args={
            "goal": (
                "Research how run_agent.py handles code modification requests "
                "and summarize the implementation options."
            )
        },
    )

    assert result is None


def test_code_modification_guard_allows_short_goals_without_file_paths():
    plugin = load_plugin("code-modification-guard")

    result = plugin.on_pre_tool_call(
        tool_name="delegate_task",
        args={"goal": "fix bug"},
    )

    assert result is None


def test_code_modification_guard_allows_non_delegate_task_tools():
    plugin = load_plugin("code-modification-guard")

    result = plugin.on_pre_tool_call(
        tool_name="read_file",
        args={
            "goal": (
                "Please implement the fix in run_agent.py, update the function, "
                "and add tests for the changed behavior."
            )
        },
    )

    assert result is None


def test_tool_guard_registers_hooks_with_importlib_loaded_plugin():
    plugin = load_plugin("tool-guard")
    configure_tool_guard(plugin)
    collector = HookCollector()

    plugin.register(collector)

    assert set(collector.hooks) == {"pre_tool_call", "post_tool_call"}


def test_tool_guard_blocks_after_recorded_failure_and_clears_after_success():
    plugin = load_plugin("tool-guard")
    configure_tool_guard(plugin)
    args = {"goal": "Use Codex to inspect this issue"}

    plugin._post_tool_call(
        tool_name="delegate_task",
        args=args,
        result="proxy error: could not connect to upstream",
    )
    blocked = plugin._pre_tool_call(tool_name="delegate_task", args=args)

    assert blocked["action"] == "block"
    assert "codex" in blocked["message"]
    assert "proxy error" in blocked["message"]

    plugin._post_tool_call(tool_name="delegate_task", args=args, result="completed")

    assert plugin._pre_tool_call(tool_name="delegate_task", args=args) is None


def test_tool_guard_clears_failure_after_cooldown(monkeypatch):
    plugin = load_plugin("tool-guard")
    configure_tool_guard(plugin)
    args = {"goal": "Use Codex to inspect this issue"}

    now = 1000.0
    monkeypatch.setattr(plugin.time, "monotonic", lambda: now)
    plugin._post_tool_call(
        tool_name="delegate_task",
        args=args,
        result="connection refused by proxy",
    )

    monkeypatch.setattr(plugin.time, "monotonic", lambda: now + 301.0)

    assert plugin._pre_tool_call(tool_name="delegate_task", args=args) is None
    assert plugin._failure_state == {}
