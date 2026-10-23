"""An MCP elicitation is a question the *server* wrote for the person in the chat, not a command
the agent wants to run. It must reach them as that question and a Yes / No, without the
command-approval wrapper (header, code fence, "why it was flagged", session/always tiers)."""

import sys
from typing import Any, Dict

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, ExecApprovalPrompt, SendResult
from gateway.run import _format_exec_approval_fallback

QUESTION = "Add “Ouvre-moi” by Claire McGowan to Kavita (Livres)?"


class _Buttons(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True), Platform.TELEGRAM)
        self.prompts: list = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        pass

    async def send(self, *a: Any, **k: Any) -> SendResult:
        return SendResult(success=True)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {}

    async def _send_exec_approval_prompt(self, prompt: ExecApprovalPrompt) -> SendResult:
        self.prompts.append(prompt)
        return SendResult(success=True, message_id="m1")


@pytest.mark.asyncio
async def test_plain_prompt_is_the_question_and_yes_no():
    adapter = _Buttons()
    await adapter.send_exec_approval(
        "chat", QUESTION, "sess", description="Approval requested by MCP server 'livrarr'.",
        allow_permanent=False, allow_session=False, plain=True)
    (prompt,) = adapter.prompts
    assert prompt.plain is True
    assert prompt.text == QUESTION
    assert prompt.actions == [("Yes", "once", "primary"), ("No", "deny", "danger")]


@pytest.mark.asyncio
async def test_command_approvals_keep_their_wrapper():
    adapter = _Buttons()
    await adapter.send_exec_approval("chat", "rm -rf /tmp/x", "sess", description="cleanup")
    (prompt,) = adapter.prompts
    assert prompt.plain is False
    assert prompt.text != "rm -rf /tmp/x" and "cleanup" in prompt.text
    assert prompt.choices == ["once", "session", "always", "deny"]


def test_plain_text_fallback_is_the_question_and_how_to_answer():
    text = _format_exec_approval_fallback(
        QUESTION, "Approval requested by MCP server 'livrarr'.", "/",
        allow_permanent=False, allow_session=False, plain=True)
    assert text == f"{QUESTION}\n\nReply `/approve` for yes or `/deny` for no."


def test_elicitation_consent_asks_for_a_plain_one_shot_prompt(monkeypatch):
    from tools import approval_prompt

    seen = []
    monkeypatch.setattr(approval_prompt._ctx, "_is_gateway_approval_context", lambda: True)
    monkeypatch.setattr(approval_prompt._ctx, "get_current_session_key", lambda: "sess")
    monkeypatch.setattr("tools.approval._gateway_notify_cb", lambda _key: lambda _data: None)
    monkeypatch.setattr(
        approval_prompt._gw, "_await_gateway_decision",
        lambda _key, _cb, data, **_kw: seen.append(data) or {"resolved": True, "choice": "once"})
    assert approval_prompt.request_elicitation_consent(QUESTION, "desc") == "accept"
    (data,) = seen
    assert (data["allow_session"], data["allow_permanent"], data["plain"]) == (False, False, True)


@pytest.mark.asyncio
async def test_matrix_plain_prompt_offers_yes_and_no_reactions_only(monkeypatch):
    sys.path.insert(0, "tests/gateway")
    import test_matrix_approval_reaction_fail_closed  # noqa: F401 -- stubs mautrix
    from plugins.platforms.matrix.adapter import MatrixAdapter

    adapter = object.__new__(MatrixAdapter)
    adapter._client = object()
    adapter._approval_prompts_by_event = {}
    sent = {}

    async def _capture(chat_id, text, metadata, make_prompt, registry, emojis, label):
        sent.update(text=text, emojis=tuple(emojis))
        return SendResult(success=True, message_id="e1")

    monkeypatch.setattr(adapter, "_send_reaction_prompt", _capture)
    prompt = ExecApprovalPrompt(
        chat_id="!room", session_key="sess", text=QUESTION, command=QUESTION, description="d",
        smart_denied=False, actions=[("Yes", "once", "primary"), ("No", "deny", "danger")], plain=True)
    await adapter._send_exec_approval_prompt(prompt)
    assert sent == {"text": f"{QUESTION}\n\n✅ Yes · ❌ No", "emojis": ("✅", "❌")}
