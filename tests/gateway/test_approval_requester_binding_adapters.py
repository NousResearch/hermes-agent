"""Native approval buttons/reactions on Teams, Matrix and WhatsApp Cloud honour ``approvals.approvers``.

Alice's turn raised the approval; Bob (admitted to the chat) clicks. With the opt-in (``approvers: []``)
Bob's click is refused with "Only Alice can answer this approval." and the approval stays pending; by
default (key absent) Bob's click resolves it, as before.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import asyncio

import pytest

from gateway.config import PlatformConfig
from tests.gateway.test_approval_requester_binding import DEFAULT, pending_from, write_approvers

REFUSAL = "Only Alice can answer this approval."
CASES = [(DEFAULT, True), ([], False)]


@pytest.fixture(autouse=True)
def _clean_queues():
    from tools import approval as approval_mod

    approval_mod._gateway_queues.clear()
    yield
    approval_mod._gateway_queues.clear()


@pytest.mark.asyncio
@pytest.mark.parametrize(("approvers", "allowed"), CASES)
async def test_teams_card_click(approvers, allowed, monkeypatch):
    import tests.gateway.test_teams
    from plugins.platforms.teams import adapter as teams_mod

    write_approvers(approvers)
    entry = pending_from("aad-alice", "Alice", "sess-t", platform="teams")
    adapter = object.__new__(teams_mod.TeamsAdapter)
    adapter.config = PlatformConfig(enabled=True, extra={})
    adapter._approval_previews = {}
    monkeypatch.setattr(adapter, "_card_action_denied", lambda _from: None)
    monkeypatch.setattr(adapter, "_invoke_message", lambda text: ("message", text))
    monkeypatch.setattr(adapter, "_invoke_card", lambda body: ("card", body))
    monkeypatch.setattr(teams_mod, "TextBlock", lambda **kw: kw)  # the mocked SDK leaves it None
    ctx = SimpleNamespace(activity=SimpleNamespace(
        from_=SimpleNamespace(aad_object_id="aad-bob", id="29:bob"),
        value=SimpleNamespace(action=SimpleNamespace(data={"hermes_action": "approve_once", "session_key": "sess-t",
                                                  "request_id": entry.data["request_id"]}))))

    response = await adapter._on_card_action(ctx)

    assert entry.event.is_set() is allowed
    if not allowed:
        assert response == ("message", REFUSAL)


@pytest.mark.asyncio
@pytest.mark.parametrize(("approvers", "allowed"), CASES)
async def test_matrix_reaction(approvers, allowed):
    import tests.gateway.test_matrix_approval_reaction_fail_closed as mx  # installs the mautrix stub

    write_approvers(approvers)
    entry = pending_from("@alice:hs", "Alice", "sess-m", platform="matrix")
    adapter = mx._make_adapter(allowed_user_ids={"@alice:hs", "@bob:hs"})
    adapter.config = PlatformConfig(enabled=True, extra={})
    adapter._approval_require_sender = False  # the adapter's own stricter gate off: the global setting decides
    adapter._redact_bot_approval_reactions = AsyncMock()
    adapter.send = AsyncMock()
    prompt = mx._MatrixApprovalPrompt(session_key="sess-m", chat_id="!room:hs", message_id="$p",
                                      request_id=entry.data["request_id"])
    adapter._approval_prompts_by_event["$p"] = prompt

    assert await adapter._handle_approval_reaction("!room:hs", "$p", "✅", "@bob:hs") is True

    assert entry.event.is_set() is allowed
    if not allowed:
        assert adapter.send.await_args.args[1] == REFUSAL
        assert adapter._approval_prompts_by_event.get("$p") is prompt and not prompt.resolved


@pytest.mark.asyncio
@pytest.mark.parametrize(("approvers", "allowed"), CASES)
async def test_whatsapp_cloud_tap(approvers, allowed, monkeypatch):
    from tests.gateway.test_whatsapp_cloud import _make_adapter, _mock_httpx_response

    monkeypatch.setenv("WHATSAPP_ALLOW_ALL_USERS", "true")
    write_approvers(approvers)
    entry = pending_from("15550000001", "Alice", "sess-w", platform="whatsapp_cloud")
    adapter = _make_adapter()
    adapter._exec_approval_state["a1"] = "sess-w"
    adapter._approval_request_ids()["a1"] = entry.data["request_id"]
    adapter._http_client = MagicMock(post=AsyncMock(return_value=_mock_httpx_response(200, {"messages": [{"id": "x"}]})))
    raw = {"from": "15550000002", "type": "interactive",
           "interactive": {"type": "button_reply", "button_reply": {"id": "appr:a1:approve", "title": "Approve"}}}

    assert await adapter._dispatch_interactive_reply(raw, {}) is True

    assert entry.event.is_set() is allowed
    if not allowed:
        assert adapter._http_client.post.call_args.kwargs["json"]["text"]["body"] == REFUSAL
        assert adapter._exec_approval_state.get("a1") == "sess-w"

