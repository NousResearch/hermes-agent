"""
test_register_contract — verifies the agent-side plugin contract.

Uses a fake PluginContext to assert that register() wires exactly:
  - 5 tools (workflow_list, workflow_run, workflow_status, workflow_approve, workflow_cancel)
  - 0 hooks
  - 1 CLI command named "workflow"
  - no call to ctx.include_router (absence of AttributeError = pass,
    because FakeCtx deliberately omits include_router)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import MagicMock, patch


@dataclass
class FakeCtx:
    """Minimal fake PluginContext — deliberately does NOT have include_router."""

    tools: List[Dict[str, Any]] = field(default_factory=list)
    hooks: List[Dict[str, Any]] = field(default_factory=list)
    cli_commands: List[Dict[str, Any]] = field(default_factory=list)

    def register_tool(
        self,
        name: str,
        toolset: str,
        schema: dict,
        handler: Callable,
        check_fn: Optional[Callable] = None,
        is_async: bool = False,
        description: str = "",
        emoji: str = "",
        **kwargs: Any,
    ) -> None:
        self.tools.append({"name": name, "toolset": toolset})

    def register_hook(self, hook_name: str, callback: Callable) -> None:
        self.hooks.append({"hook": hook_name})

    def register_cli_command(
        self,
        name: str,
        help: str,
        setup_fn: Callable,
        handler_fn: Optional[Callable] = None,
        description: str = "",
    ) -> None:
        self.cli_commands.append({"name": name})


_EXPECTED_TOOLS = {
    "workflow_list",
    "workflow_run",
    "workflow_status",
    "workflow_approve",
    "workflow_cancel",
}


def test_register_tools_count(monkeypatch):
    """register() must register exactly 5 tools in the workflow toolset."""
    # Patch get_engine so no real DB is opened during import
    fake_engine = MagicMock()
    monkeypatch.setattr(
        "plugins.workflow_engine._shared._engine", fake_engine
    )

    ctx = FakeCtx()
    import plugins.workflow_engine as we  # noqa: PLC0415

    we.register(ctx)

    assert len(ctx.tools) == 5, (
        f"Expected 5 tools, got {len(ctx.tools)}: {[t['name'] for t in ctx.tools]}"
    )
    assert {t["name"] for t in ctx.tools} == _EXPECTED_TOOLS


def test_register_no_hooks(monkeypatch):
    """register() must not register any hooks."""
    fake_engine = MagicMock()
    monkeypatch.setattr(
        "plugins.workflow_engine._shared._engine", fake_engine
    )

    ctx = FakeCtx()
    import plugins.workflow_engine as we  # noqa: PLC0415

    we.register(ctx)

    assert len(ctx.hooks) == 0, (
        f"Expected 0 hooks, got {len(ctx.hooks)}: {ctx.hooks}"
    )


def test_register_one_cli_command(monkeypatch):
    """register() must register exactly 1 CLI command named 'workflow'."""
    fake_engine = MagicMock()
    monkeypatch.setattr(
        "plugins.workflow_engine._shared._engine", fake_engine
    )

    ctx = FakeCtx()
    import plugins.workflow_engine as we  # noqa: PLC0415

    we.register(ctx)

    assert len(ctx.cli_commands) == 1, (
        f"Expected 1 CLI command, got {len(ctx.cli_commands)}: {ctx.cli_commands}"
    )
    assert ctx.cli_commands[0]["name"] == "workflow"


def test_register_no_include_router(monkeypatch):
    """register() must not call ctx.include_router (FakeCtx has none; no AttributeError = pass)."""
    fake_engine = MagicMock()
    monkeypatch.setattr(
        "plugins.workflow_engine._shared._engine", fake_engine
    )

    ctx = FakeCtx()
    # Verify FakeCtx truly has no include_router
    assert not hasattr(ctx, "include_router"), "FakeCtx must not have include_router"

    import plugins.workflow_engine as we  # noqa: PLC0415

    # If register() tries to call ctx.include_router, it will raise AttributeError
    try:
        we.register(ctx)
    except AttributeError as exc:
        raise AssertionError(
            f"register() called ctx.include_router which does not exist: {exc}"
        ) from exc


def test_all_tools_have_check_fn():
    """Every tool module must export a callable check function."""
    import plugins.workflow_engine.tools.list_workflows as lw  # noqa: PLC0415
    import plugins.workflow_engine.tools.run_workflow as rw  # noqa: PLC0415
    import plugins.workflow_engine.tools.workflow_status as ws  # noqa: PLC0415
    import plugins.workflow_engine.tools.approve_workflow as aw  # noqa: PLC0415
    import plugins.workflow_engine.tools.cancel_workflow as cw  # noqa: PLC0415

    for mod in (lw, rw, ws, aw, cw):
        assert callable(getattr(mod, "check", None)), (
            f"{mod.__name__} must export a callable 'check' function"
        )
        assert callable(getattr(mod, "handler", None)), (
            f"{mod.__name__} must export a callable 'handler' function"
        )
        assert isinstance(getattr(mod, "SCHEMA", None), dict), (
            f"{mod.__name__} must export a SCHEMA dict"
        )


def test_check_fns_read_only_always_true():
    """workflow_list and workflow_status check() must always return True."""
    from plugins.workflow_engine.tools.list_workflows import check as list_check  # noqa: PLC0415
    from plugins.workflow_engine.tools.workflow_status import check as status_check  # noqa: PLC0415

    assert list_check() is True
    assert status_check() is True
