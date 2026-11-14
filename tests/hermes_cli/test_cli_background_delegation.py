"""Classic-CLI /bg: a background delegate_task joins inside the bg_* run (#140569).

The /bg answer is printed once and nothing drains completions for its bg_* session, so a detached
child's result would sit in the ledger forever.
"""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cli as cli_module
from tests.hermes_cli.test_cli_approval_ui import _make_background_cli_stub


def test_background_task_delegation_runs_inline(monkeypatch):
    import tools.delegate_tool as dt

    monkeypatch.setattr(dt, "_build_child_agent", lambda **kw: SimpleNamespace(_subagent_id="child"))
    monkeypatch.setattr(dt, "_resolve_delegation_credentials", lambda *a, **kw: {
        "model": "test", "provider": None, "base_url": None, "api_key": None,
        "api_mode": None, "command": None, "args": None,
    })
    monkeypatch.setattr(dt, "_run_single_child", lambda **kw: {
        "task_index": 0, "status": "completed", "summary": "child work product",
        "api_calls": 1, "duration_seconds": 0, "model": "test", "exit_reason": "completed",
    })
    detached = MagicMock(return_value={"delegation_id": "detached"})
    monkeypatch.setattr("tools.async_delegation.dispatch_async_delegation_batch", detached)
    seen = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            self._print_fn = None
            self.thinking_callback = None

        def run_conversation(self, **kwargs):
            parent = SimpleNamespace(_delegate_depth=0, _subagent_id=None)
            seen["tool"] = dt.delegate_task(goal="inspect workspace", background=True, parent_agent=parent)
            return {"final_response": "done", "messages": []}

        def close(self):
            pass

    cli = _make_background_cli_stub()
    with patch("run_agent.AIAgent", FakeAgent), patch.object(cli_module, "_cprint"), \
            patch.object(cli_module, "ChatConsole"):
        cli._handle_background_command("/bg inspect workspace")
        for thread in list(cli._background_tasks.values()):
            thread.join(timeout=10)

    detached.assert_not_called()
    assert "child work product" in json.dumps(seen["tool"])
