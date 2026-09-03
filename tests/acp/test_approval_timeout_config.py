"""Server-level wiring tests for the ACP approval timeout (#63538).

The approval factories in ``acp_adapter.permissions`` and
``acp_adapter.edit_approval`` accept a ``timeout`` argument, but
``acp_adapter/server.py`` used to omit it, so both ACP approval paths fell
back to their 60 s default regardless of the configured
``approvals.timeout``. These tests drive the real ``prompt()`` wiring and
assert both constructors receive the configured value.
"""

import asyncio
import threading
import types

import pytest

import acp_adapter.edit_approval as edit_approval_module
import acp_adapter.server as server_module
from acp.schema import TextContentBlock


class _StopAfterWiring(Exception):
    """Raised from the stub state to stop prompt() right after callback wiring."""


class _StubState:
    def __init__(self):
        self.runtime_lock = threading.Lock()
        self.is_running = False
        self.queued_prompts = []
        self.cancel_event = None
        self.current_prompt_text = ""
        self.interrupted_prompt_text = ""

    @property
    def agent(self):
        raise _StopAfterWiring


def _make_agent(monkeypatch, captured):
    agent = object.__new__(server_module.HermesACPAgent)
    agent._conn = types.SimpleNamespace(request_permission=object())
    agent.session_manager = types.SimpleNamespace(get_session=lambda _sid: _StubState())

    monkeypatch.setattr(server_module, "make_tool_progress_cb", lambda *a, **k: (lambda text: None))
    monkeypatch.setattr(server_module, "make_thinking_cb", lambda *a, **k: (lambda text: None))
    monkeypatch.setattr(server_module, "make_step_cb", lambda *a, **k: (lambda step: None))
    monkeypatch.setattr(server_module, "make_message_cb", lambda *a, **k: (lambda text: None))
    monkeypatch.setattr(
        server_module,
        "make_approval_callback",
        lambda *a, **k: captured.setdefault("command", (a, k)) or "approval_cb",
    )
    monkeypatch.setattr(
        edit_approval_module,
        "make_acp_edit_approval_requester",
        lambda *a, **k: captured.setdefault("edit", (a, k)) or "edit_requester",
    )
    monkeypatch.setattr("tools.approval._get_approval_timeout", lambda: 137)
    return agent


def test_prompt_wires_configured_approval_timeout_to_both_constructors(monkeypatch):
    captured: dict = {}
    agent = _make_agent(monkeypatch, captured)

    with pytest.raises(_StopAfterWiring):
        asyncio.run(agent.prompt([TextContentBlock(type="text", text="hello")], "session-1"))

    assert captured["command"][1]["timeout"] == 137.0
    assert captured["edit"][1]["timeout"] == 137.0
