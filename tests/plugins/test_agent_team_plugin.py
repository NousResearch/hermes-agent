import importlib.util
import json
from pathlib import Path


PLUGIN_PATH = Path(__file__).resolve().parents[2] / "plugins" / "agent-team" / "__init__.py"


def load_plugin():
    spec = importlib.util.spec_from_file_location("agent_team_plugin", PLUGIN_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StubContext:
    def __init__(self, result=None):
        self.commands = {}
        self.dispatched = []
        self.result = result or json.dumps(
            {
                "results": [
                    {"task_index": 0, "status": "completed", "summary": "research done", "duration_seconds": 1.0},
                    {"task_index": 1, "status": "completed", "summary": "build done", "duration_seconds": 2.0},
                    {"task_index": 2, "status": "completed", "summary": "review done", "duration_seconds": 3.0},
                ],
                "total_duration_seconds": 3.0,
            }
        )

    def register_command(self, name, handler, description="", args_hint=""):
        self.commands[name] = {
            "handler": handler,
            "description": description,
            "args_hint": args_hint,
        }

    def dispatch_tool(self, tool_name, args, **kwargs):
        self.dispatched.append((tool_name, args, kwargs))
        return self.result


def registered_handler(result=None):
    plugin = load_plugin()
    ctx = StubContext(result=result)
    plugin.register(ctx)
    return ctx.commands["team"]["handler"], ctx


def test_registers_team_slash_command():
    plugin = load_plugin()
    ctx = StubContext()

    plugin.register(ctx)

    assert "team" in ctx.commands
    assert ctx.commands["team"]["args_hint"] == "[full|research|build|review] <task>"


def test_help_does_not_dispatch():
    plugin = load_plugin()
    ctx = StubContext()
    plugin.register(ctx)

    output = ctx.commands["team"]["handler"]("help")

    assert output.startswith("/team")
    assert ctx.dispatched == []


def test_full_mode_dispatches_three_roles():
    plugin = load_plugin()
    ctx = StubContext()
    plugin.register(ctx)

    output = ctx.commands["team"]["handler"]("full build a reporting workflow")

    assert ctx.dispatched[0][0] == "delegate_task"
    tasks = ctx.dispatched[0][1]["tasks"]
    assert [task["team_role"] for task in tasks] == ["researcher", "builder", "reviewer"]
    assert tasks[0]["toolsets"] == ["web"]
    assert tasks[1]["toolsets"] == ["terminal", "file"]
    assert ctx.dispatched[0][1]["max_iterations"] == 20
    assert "Researcher" in output
    assert "Builder" in output
    assert "Reviewer" in output


def test_single_mode_dispatches_one_role():
    plugin = load_plugin()
    ctx = StubContext()
    plugin.register(ctx)

    ctx.commands["team"]["handler"]("review audit this change")

    tasks = ctx.dispatched[0][1]["tasks"]
    assert len(tasks) == 1
    assert tasks[0]["team_role"] == "reviewer"
    assert "max_iterations" not in ctx.dispatched[0][1]


def test_default_mode_without_explicit_mode_dispatches_full_team():
    handler, ctx = registered_handler()

    handler("create a reporting workflow")

    tasks = ctx.dispatched[0][1]["tasks"]
    assert [task["team_role"] for task in tasks] == ["researcher", "builder", "reviewer"]
    assert "create a reporting workflow" in tasks[0]["context"]


def test_mode_aliases_dispatch_expected_roles():
    cases = {
        "all": ["researcher", "builder", "reviewer"],
        "r": ["researcher"],
        "b": ["builder"],
        "v": ["reviewer"],
    }

    for alias, expected_roles in cases.items():
        handler, ctx = registered_handler()

        handler(f"{alias} inspect this")

        tasks = ctx.dispatched[0][1]["tasks"]
        assert [task["team_role"] for task in tasks] == expected_roles


def test_empty_task_returns_help_without_dispatch():
    handler, ctx = registered_handler()

    assert handler("").startswith("/team")
    assert handler("review").startswith("/team")
    assert ctx.dispatched == []


def test_non_json_delegate_result_is_passed_through():
    handler, _ctx = registered_handler(result="plain delegate output")

    output = handler("review inspect this")

    assert output == "plain delegate output"


def test_parent_context_error_gets_actionable_message():
    result = json.dumps({"error": "delegate_task requires an active Hermes agent context"})
    handler, _ctx = registered_handler(result=result)

    output = handler("review inspect this")

    assert output.startswith("Cannot run /team here")
    assert "interactive Hermes session" in output


def test_generic_delegate_error_is_formatted():
    result = json.dumps({"error": "upstream failed"})
    handler, _ctx = registered_handler(result=result)

    output = handler("review inspect this")

    assert output == "Agent team failed: upstream failed"


def test_result_formatting_uses_role_order_when_team_role_missing():
    result = json.dumps(
        {
            "results": [
                {"status": "completed", "summary": "review done", "duration_seconds": 1.5}
            ],
            "total_duration_seconds": 1.5,
        }
    )
    handler, _ctx = registered_handler(result=result)

    output = handler("review inspect this")

    assert "Reviewer — completed (1.5s)" in output
    assert "review done" in output
