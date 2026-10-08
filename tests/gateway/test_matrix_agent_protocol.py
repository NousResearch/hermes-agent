"""Matrix adapter: fluxchat agent protocol fields (org.fluxchat.agent) on approvals."""
import types

import pytest
from unittest.mock import AsyncMock, patch

from gateway.config import PlatformConfig
from gateway.platforms.base import ExecApprovalPrompt, SendResult

KEY = "org.fluxchat.agent"


def _adapter(monkeypatch):
    monkeypatch.setenv("MATRIX_ALLOWED_USERS", "@me:example.org")
    from plugins.platforms.matrix.adapter import MatrixAdapter
    adapter = MatrixAdapter(PlatformConfig(enabled=True, token="tok", extra={"homeserver": "https://matrix.example.org"}))
    adapter._user_id = "@bot:example.org"
    adapter._client = object()
    return adapter


@pytest.mark.asyncio
async def test_send_carries_agent_status_from_metadata(monkeypatch):
    adapter = _adapter(monkeypatch)
    sent = []
    adapter._send_room_message = AsyncMock(side_effect=lambda chat, content: sent.append(content) or "$e1")
    await adapter.send("!r:example.org", "hi", metadata={"fluxchat_agent": {"v": 1, "state": "working"}})
    assert sent[0][KEY] == {"v": 1, "state": "working"}
    # Without it, nothing extra.
    await adapter.send("!r:example.org", "plain")
    assert KEY not in sent[1]


@pytest.mark.asyncio
async def test_approval_prompt_offers_its_choices_then_settles(monkeypatch):
    adapter = _adapter(monkeypatch)
    sent = []
    adapter._send_room_message = AsyncMock(side_effect=lambda chat, content: sent.append(content) or "$prompt")
    adapter._send_reaction = AsyncMock(return_value="$r")
    prompt = ExecApprovalPrompt(
        chat_id="!r:example.org", session_key="sess-1", text="Run `rm -rf build`?",
        actions=[("Allow once", "once", "primary"), ("Always allow", "always", ""), ("Deny", "deny", "danger")],
        command="rm -rf build", description="", smart_denied=False, metadata={"thread_id": "$root"})
    result = await adapter._send_exec_approval_prompt(prompt)
    assert result.message_id == "$prompt"
    status = sent[0][KEY]
    assert status["state"] == "waiting"
    assert [c["label"] for c in status["choices"]] == ["Allow once", "Always allow", "Deny"]
    assert sent[0]["m.relates_to"]["event_id"] == "$root", "thread metadata still applies"

    # The user approves: the prompt is edited out of "waiting".
    adapter.edit_message = AsyncMock(return_value=SendResult(success=True))
    adapter._redact_bot_approval_reactions = AsyncMock()
    event = types.SimpleNamespace(sender="@me:example.org", event_id="$react", room_id="!r:example.org",
                                  content={"m.relates_to": {"event_id": "$prompt", "key": "✅"}})
    with patch("tools.approval.resolve_gateway_approval", return_value=1):
        await adapter._on_reaction(event)
    args, kwargs = adapter.edit_message.call_args
    assert args[1] == "$prompt"
    assert kwargs["agent_status"]["state"] == "working"
    assert kwargs["agent_status"]["choices"] == status["choices"]


@pytest.mark.asyncio
async def test_edit_puts_status_in_new_content(monkeypatch):
    adapter = _adapter(monkeypatch)
    captured = []
    adapter._send_content_event = AsyncMock(side_effect=lambda chat, content: captured.append(content) or SendResult(success=True))
    await adapter.edit_message("!r:example.org", "$m", "Working on it", agent_status={"v": 1, "state": "done"})
    assert captured[0]["m.new_content"][KEY] == {"v": 1, "state": "done"}
    assert captured[0][KEY] == {"v": 1, "state": "done"}

def test_matrix_hint_teaches_the_result_headline():
    """The Matrix platform prompt tells the agent to open its final reply with a
    one-line outcome: the adapter's default `result` is the reply's first line, so
    the headline has to come from the agent, not from trimming the transcript."""
    from agent.prompt_builder import PLATFORM_HINTS
    hint = PLATFORM_HINTS["matrix"]
    assert "first line" in hint
    assert "result" in hint
