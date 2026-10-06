"""Photon question delivery, poll-bound clarify answers, and vote notifications."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent, MessageType
from plugins.platforms.photon.adapter import PhotonAdapter
from tools import clarify_gateway as cg


@pytest.fixture(autouse=True)
def _isolate_clarify(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cg, "_entries", {})
    monkeypatch.setattr(cg, "_session_index", {})
    monkeypatch.setattr(cg, "_notify_cbs", {})


def _make_adapter(monkeypatch: pytest.MonkeyPatch) -> PhotonAdapter:
    monkeypatch.setenv("PHOTON_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "test-project-secret")
    cfg = PlatformConfig(enabled=True, token="", extra={})
    adapter = PhotonAdapter(cfg)
    adapter.set_authorization_check(lambda *_args: True)
    return adapter


def _capture(
    adapter: PhotonAdapter, monkeypatch: pytest.MonkeyPatch
) -> List[MessageEvent]:
    captured: List[MessageEvent] = []

    async def fake_handle(event: MessageEvent) -> None:
        captured.append(event)

    monkeypatch.setattr(adapter, "handle_message", fake_handle)
    return captured


async def _gateway_intercept(event: MessageEvent, session_key: str) -> Optional[str]:
    from gateway.run_inbound import GatewayInboundMixin

    runner = GatewayInboundMixin.__new__(GatewayInboundMixin)

    async def prepare_text(inbound):
        return inbound.text

    runner._hm_update_prompt_reply = lambda *_args: None
    runner._pending_event_audio_paths = lambda _event: []
    runner._prepare_clarify_reply_text = prepare_text
    runner._delivery_adapter_for = lambda _source: None
    return await runner._hm_pending_reply_intercepts(event, event.source, session_key)


def _poll_option_event(
    *, title: str, selected: bool = True, msg_id: str = "spc-msg-vote",
    poll_id: Optional[str] = "spc-msg-poll", group: bool = False,
) -> Dict[str, Any]:
    return {
        "messageId": msg_id,
        "platform": "iMessage",
        "space": {"id": "+155****4567", "type": "group" if group else "dm", "phone": "+155****4567"},
        "sender": {"id": "+155****4567"},
        "content": {
            "type": "poll_option",
            "title": title,
            "selected": selected,
            "pollTitle": "Pick one",
            "pollId": poll_id,
        },
        "timestamp": "2026-05-14T19:06:32.000Z",
    }


# ---------------------------------------------------------------------------
# Inbound: a poll vote becomes the clarify answer.


@pytest.mark.asyncio
async def test_poll_vote_dispatched_as_choice_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)
    _stub_sidecar_text(adapter, monkeypatch)
    _stub_sidecar_poll(adapter, monkeypatch)
    entry = cg.register("clar-1", "sess-1", "Pick one", ["Route", "Calendar"])
    await adapter.send_clarify("+155****4567", "Pick one", entry.choices, "clar-1", "sess-1")

    vote = _poll_option_event(title="Route")
    vote["content"].update(tally={"Route": 1, "Calendar": 0}, voters=1)
    await adapter._dispatch_inbound(vote)

    assert entry.event.is_set()
    assert entry.response == "Route"
    assert captured == []  # DM answer reaches the agent through the clarify result.


@pytest.mark.asyncio
async def test_poll_vote_and_later_text_remain_distinct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)

    await adapter._dispatch_inbound(_poll_option_event(title="Route"))
    await adapter._dispatch_inbound(
        _poll_option_event(title="Route", selected=False, msg_id="spc-msg-unvote")
    )
    later = _poll_option_event(title="", msg_id="spc-msg-text")
    later["content"] = {"type": "text", "text": "Did you get my choice?"}
    await adapter._dispatch_inbound(later)

    assert [(event.message_id, event.text) for event in captured] == [
        ("spc-msg-vote", "[poll vote] Route (poll: Pick one)"),
        ("spc-msg-text", "Did you get my choice?"),
    ]
    assert not captured[0].allow_gateway_control
    assert captured[1].allow_gateway_control


# ---------------------------------------------------------------------------
# Outbound: send_clarify renders a native poll for choices.


def _stub_sidecar_poll(
    adapter: PhotonAdapter, monkeypatch: pytest.MonkeyPatch, *, ok: bool = True,
    poll_id: str = "spc-msg-poll",
) -> List[Tuple[str, str, list]]:
    calls: List[Tuple[str, str, list]] = []

    async def fake_send_poll(space_id: str, title: str, options: list):
        calls.append((space_id, title, list(options)))
        return SendResult(
            success=ok,
            message_id=poll_id if ok else None,
            error=None if ok else "boom",
        )

    monkeypatch.setattr(adapter, "_sidecar_send_poll", fake_send_poll)
    return calls


def _stub_sidecar_text(
    adapter: PhotonAdapter, monkeypatch: pytest.MonkeyPatch
) -> List[Tuple[str, str]]:
    sends: List[Tuple[str, str]] = []

    async def fake_send(space_id: str, text: str):
        sends.append((space_id, text))
        return SendResult(success=True, message_id="spc-msg-text")

    monkeypatch.setattr(adapter, "_sidecar_send", fake_send)
    return sends


@pytest.mark.asyncio
async def test_send_clarify_with_choices_sends_native_poll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    poll_calls = _stub_sidecar_poll(adapter, monkeypatch)
    text_calls = _stub_sidecar_text(adapter, monkeypatch)

    marked: List[str] = []
    monkeypatch.setattr(cg, "mark_awaiting_text", lambda cid: marked.append(cid))

    result = await adapter.send_clarify(
        chat_id="+155****4567",
        question="Pick one",
        choices=["A", "B", "C"],
        clarify_id="clar-1",
        session_key="sess-1",
    )

    assert result.success
    assert len(poll_calls) == 1
    space_id, title, options = poll_calls[0]
    assert space_id == "+155****4567"
    assert title == "Pick one"
    assert options == ["A", "B", "C"]
    assert text_calls == [("+155****4567", "Pick one")]
    # The vote returns as text, so text-capture must be enabled.
    assert marked == ["clar-1"]


@pytest.mark.asyncio
async def test_question_precedes_poll(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch)
    calls = []

    async def text(chat_id, content):
        calls.append(("text", content))
        return SendResult(success=True)

    async def poll(chat_id, title, options):
        calls.append(("poll", title))
        return SendResult(success=True, message_id="poll-1")

    monkeypatch.setattr(adapter, "_sidecar_send", text)
    monkeypatch.setattr(adapter, "_sidecar_send_poll", poll)
    await adapter.send_clarify("chat", "Which vault?", ["A", "B"], "clar-1", "sess-1")
    assert calls == [("text", "Which vault?"), ("poll", "Which vault?")]


@pytest.mark.asyncio
async def test_failed_poll_sends_numbered_fallback_without_repeating_question(monkeypatch) -> None:
    adapter = _make_adapter(monkeypatch)
    calls = _stub_sidecar_text(adapter, monkeypatch)
    _stub_sidecar_poll(adapter, monkeypatch, ok=False)
    entry = cg.register("clar-1", "sess-1", "Which vault?", ["A", "B"])

    result = await adapter.send_clarify("chat", entry.question, entry.choices, "clar-1", "sess-1")

    assert result.success
    assert len(calls) == 2
    assert calls[0] == ("chat", "Which vault?")
    assert sum(text.count("Which vault?") for _, text in calls) == 1
    assert "1. A" in calls[1][1] and "2. B" in calls[1][1]
    assert entry.awaiting_text
    assert cg.resolve_text_response_for_session("sess-1", "2")
    assert entry.response == "B"


@pytest.mark.asyncio
async def test_question_failure_prevents_poll(monkeypatch) -> None:
    adapter = _make_adapter(monkeypatch)
    poll_calls = _stub_sidecar_poll(adapter, monkeypatch)

    async def failed_text(chat_id, text):
        return SendResult(success=False, error="cannot send question")

    monkeypatch.setattr(adapter, "_sidecar_send", failed_text)
    result = await adapter.send_clarify("chat", "Pick one", ["A", "B"], "clar-1", "sess-1")
    assert not result.success
    assert poll_calls == []


@pytest.mark.asyncio
async def test_open_ended_clarify_remains_text(monkeypatch) -> None:
    adapter = _make_adapter(monkeypatch)
    text_calls = _stub_sidecar_text(adapter, monkeypatch)
    poll_calls = _stub_sidecar_poll(adapter, monkeypatch)
    result = await adapter.send_clarify("chat", "Tell me more", None, "clar-1", "sess-1")
    assert result.success
    assert "Tell me more" in text_calls[0][1]
    assert poll_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("poll_id", ["unknown", None])
async def test_unknown_poll_cannot_answer_pending_clarify(monkeypatch, poll_id) -> None:
    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)
    entry = cg.register("new", "sess-1", "New question", ["Route", "Calendar"])
    cg.mark_awaiting_text("new")
    await adapter._dispatch_inbound(_poll_option_event(title="Route", poll_id=poll_id))

    assert len(captured) == 1
    outcome = await _gateway_intercept(captured[0], "sess-1")
    assert not entry.event.is_set()
    assert outcome is None
    assert captured[0].text == "[poll vote] Route (poll: Pick one)"
    assert captured[0].message_type == MessageType.TEXT


@pytest.mark.asyncio
async def test_vote_resolves_its_poll_even_with_another_pending_prompt(monkeypatch) -> None:
    adapter = _make_adapter(monkeypatch)
    _stub_sidecar_text(adapter, monkeypatch)
    _stub_sidecar_poll(adapter, monkeypatch)
    older = cg.register("older", "sess-1", "Older", ["Route", "Calendar"])
    newer = cg.register("newer", "sess-1", "Newer", ["Route", "Calendar"])
    await adapter.send_clarify("+155****4567", "Newer", newer.choices, "newer", "sess-1")
    await adapter._dispatch_inbound(_poll_option_event(title="Route"))
    assert newer.response == "Route"
    assert newer.event.is_set()
    assert not older.event.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("expired", [False, True])
async def test_stale_vote_cannot_resolve_new_prompt(monkeypatch, expired) -> None:
    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)
    _stub_sidecar_text(adapter, monkeypatch)
    _stub_sidecar_poll(adapter, monkeypatch)
    old = cg.register("old", "sess-1", "Old question", ["Route", "Calendar"])
    await adapter.send_clarify("+155****4567", old.question, old.choices, "old", "sess-1")
    if expired:
        cg.wait_for_response("old", timeout=0.001)
    else:
        cg.resolve_gateway_clarify("old", "Calendar")
        assert cg.wait_for_response("old", timeout=1) == "Calendar"
    new = cg.register("new", "sess-1", "New question", ["Route", "Calendar"])
    _stub_sidecar_poll(adapter, monkeypatch, poll_id="new-poll")
    await adapter.send_clarify("+155****4567", new.question, new.choices, "new", "sess-1")
    await adapter._dispatch_inbound(_poll_option_event(title="Route"))
    outcome = await _gateway_intercept(captured[0], "sess-1")
    assert not new.event.is_set()
    assert outcome is None
    assert captured[0].text == "[poll vote] Route (poll: Old question)"


@pytest.mark.asyncio
async def test_wrong_chat_cannot_resolve_poll(monkeypatch) -> None:
    adapter = _make_adapter(monkeypatch)
    _capture(adapter, monkeypatch)
    _stub_sidecar_text(adapter, monkeypatch)
    _stub_sidecar_poll(adapter, monkeypatch)
    entry = cg.register("clar-1", "sess-1", "Pick one", ["Route", "Calendar"])
    await adapter.send_clarify("other-chat", entry.question, entry.choices, "clar-1", "sess-1")
    await adapter._dispatch_inbound(_poll_option_event(title="Route"))
    assert not entry.event.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("authorized", [False, None])
async def test_unverified_voter_cannot_resolve_poll(monkeypatch, authorized) -> None:
    adapter = _make_adapter(monkeypatch)
    adapter.set_authorization_check(lambda *_args: authorized)
    captured = _capture(adapter, monkeypatch)
    _stub_sidecar_text(adapter, monkeypatch)
    _stub_sidecar_poll(adapter, monkeypatch)
    entry = cg.register("clar-1", "sess-1", "Pick one", ["Route", "Calendar"])
    await adapter.send_clarify("+155****4567", entry.question, entry.choices, "clar-1", "sess-1")
    await adapter._dispatch_inbound(_poll_option_event(title="Route"))
    assert not entry.event.is_set()
    assert captured[0].allow_gateway_control is False


@pytest.mark.asyncio
async def test_deselect_then_reselect_resolves_once(monkeypatch) -> None:
    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)
    _stub_sidecar_text(adapter, monkeypatch)
    _stub_sidecar_poll(adapter, monkeypatch)
    entry = cg.register("clar-1", "sess-1", "Pick one", ["Route", "Calendar"])
    await adapter.send_clarify("+155****4567", entry.question, entry.choices, "clar-1", "sess-1")
    await adapter._dispatch_inbound(_poll_option_event(title="Calendar", selected=False))
    assert not entry.event.is_set()
    await adapter._dispatch_inbound(_poll_option_event(title="Route", msg_id="selected"))
    assert entry.response == "Route"
    await adapter._dispatch_inbound(_poll_option_event(title="Route", selected=False, msg_id="removed"))
    await adapter._dispatch_inbound(_poll_option_event(title="Calendar", msg_id="changed"))
    assert entry.response == "Route"
    assert [(event.message_id, event.allow_gateway_control) for event in captured] == [("changed", False)]


@pytest.mark.asyncio
@pytest.mark.parametrize("selected", [True, False])
async def test_group_vote_includes_totals_without_changing_clarify_answer(monkeypatch, selected) -> None:
    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)
    _stub_sidecar_text(adapter, monkeypatch)
    _stub_sidecar_poll(adapter, monkeypatch)
    entry = cg.register("clar-1", "sess-1", "Which vault?", ["Route", "Calendar"])
    await adapter.send_clarify("+155****4567", entry.question, entry.choices, "clar-1", "sess-1")
    vote = _poll_option_event(title="Route", selected=selected, group=True)
    vote["content"].update(tally={"Route": 2, "Calendar": 1}, voters=3)
    await adapter._dispatch_inbound(vote)
    assert entry.response == ("Route" if selected else None)
    assert captured[0].text == (
        f"[{'poll vote' if selected else 'poll vote removed'}] Route (poll: Which vault?)"
        "\nTotals: Route: 2, Calendar: 1 (3 voters)"
    )
    assert not captured[0].allow_gateway_control
