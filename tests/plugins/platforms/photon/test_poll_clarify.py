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
async def test_group_vote_answering_clarify_is_not_forwarded(monkeypatch) -> None:
    """The clarify result is the agent's input; a second forwarded copy would double it."""
    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)
    _stub_sidecar_text(adapter, monkeypatch)
    _stub_sidecar_poll(adapter, monkeypatch)
    entry = cg.register("clar-1", "sess-1", "Which vault?", ["Route", "Calendar"])
    await adapter.send_clarify("+155****4567", entry.question, entry.choices, "clar-1", "sess-1")
    vote = _poll_option_event(title="Route", group=True)
    vote["content"].update(tally={"Route": 2, "Calendar": 1}, voters=3)
    await adapter._dispatch_inbound(vote)
    assert entry.response == "Route"
    assert captured == []


@pytest.mark.asyncio
@pytest.mark.parametrize("selected", [True, False])
async def test_stale_group_vote_includes_totals(monkeypatch, selected) -> None:
    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)
    _stub_sidecar_text(adapter, monkeypatch)
    _stub_sidecar_poll(adapter, monkeypatch)
    entry = cg.register("clar-1", "sess-1", "Which vault?", ["Route", "Calendar"])
    await adapter.send_clarify("+155****4567", entry.question, entry.choices, "clar-1", "sess-1")
    assert cg.resolve_gateway_clarify("clar-1", "Calendar")
    vote = _poll_option_event(title="Route", selected=selected, group=True)
    vote["content"].update(tally={"Route": 2, "Calendar": 1}, voters=3)
    await adapter._dispatch_inbound(vote)
    assert entry.response == "Calendar"
    assert captured[0].text == (
        f"[{'poll vote' if selected else 'poll vote removed'}] Route (poll: Which vault?)"
        "\nTotals: Route: 2, Calendar: 1 (3 voters)"
    )
    assert not captured[0].allow_gateway_control


@pytest.mark.asyncio
async def test_partial_group_totals_are_marked(monkeypatch) -> None:
    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)
    vote = _poll_option_event(title="Route", group=True)
    vote["content"].update(tally={"Route": 1, "Calendar": 0}, voters=1, partial=True)
    await adapter._dispatch_inbound(vote)
    assert captured[0].text.endswith("\nTotals (partial): Route: 1, Calendar: 0 (1 voters)")


@pytest.mark.asyncio
@pytest.mark.parametrize("poll_title", ["Poll", "", None])
async def test_placeholder_poll_title_is_omitted(monkeypatch, poll_title) -> None:
    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)
    vote = _poll_option_event(title="Route")
    vote["content"]["pollTitle"] = poll_title
    await adapter._dispatch_inbound(vote)
    assert captured[0].text == "[poll vote] Route"


@pytest.mark.asyncio
@pytest.mark.parametrize("bound", [False, True])
async def test_unrelated_group_vote_obeys_require_mention(monkeypatch, bound) -> None:
    """Ported from the adversarial run: votes carry no mention, so a gated group drops them."""
    adapter = _make_adapter(monkeypatch)
    adapter.require_mention = True
    captured = _capture(adapter, monkeypatch)
    _stub_sidecar_text(adapter, monkeypatch)
    _stub_sidecar_poll(adapter, monkeypatch)
    if bound:  # an answered prompt's poll is as unrelated as a stranger's
        entry = cg.register("clar-1", "sess-1", "Which vault?", ["Route", "Calendar"])
        await adapter.send_clarify("+155****4567", entry.question, entry.choices, "clar-1", "sess-1")
        cg.resolve_gateway_clarify("clar-1", "Calendar")
    await adapter._dispatch_inbound(_poll_option_event(
        title="Route", group=True, poll_id="spc-msg-poll" if bound else "unrelated-poll"))
    assert captured == []


@pytest.mark.asyncio
async def test_bound_group_vote_answers_despite_require_mention(monkeypatch) -> None:
    adapter = _make_adapter(monkeypatch)
    adapter.require_mention = True
    captured = _capture(adapter, monkeypatch)
    _stub_sidecar_text(adapter, monkeypatch)
    _stub_sidecar_poll(adapter, monkeypatch)
    entry = cg.register("clar-1", "sess-1", "Which vault?", ["Route", "Calendar"])
    await adapter.send_clarify("+155****4567", entry.question, entry.choices, "clar-1", "sess-1")
    await adapter._dispatch_inbound(_poll_option_event(title="Route", group=True))
    assert entry.response == "Route"
    assert captured == []


@pytest.mark.asyncio
@pytest.mark.parametrize("group", [False, True])
async def test_unrelated_vote_does_not_reach_a_busy_session(monkeypatch, caplog, group) -> None:
    """busy_input_mode=interrupt would abort the running turn; there is no non-interrupting
    user-event path, so the informational vote is dropped while the session runs."""
    import asyncio
    import logging

    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)
    vote = _poll_option_event(title="Route", group=group, poll_id="unrelated-poll")
    probe = adapter.build_source(chat_id="+155****4567", chat_name="+155****4567",
                                 chat_type="group" if group else "dm", user_id="+155****4567",
                                 user_name="+155****4567")
    adapter._active_sessions[adapter._source_session_key(probe)] = asyncio.Event()
    with caplog.at_level(logging.INFO, logger="plugins.platforms.photon.adapter"):
        await adapter._dispatch_inbound(vote)
    assert captured == []
    assert "dropping poll vote while the agent is busy" in caplog.text


@pytest.mark.asyncio
async def test_unmatched_poll_id_cannot_answer_the_only_open_poll_clarify(monkeypatch, caplog) -> None:
    import logging

    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)
    _stub_sidecar_text(adapter, monkeypatch)
    _stub_sidecar_poll(adapter, monkeypatch, poll_id="sent-id")
    entry = cg.register("clar-1", "sess-1", "Pick one", ["Route", "Calendar"])
    await adapter.send_clarify("+155****4567", entry.question, entry.choices, "clar-1", "sess-1")
    with caplog.at_level(logging.WARNING, logger="plugins.platforms.photon.adapter"):
        await adapter._dispatch_inbound(_poll_option_event(title="Calendar", poll_id="vote-id"))
    assert not entry.event.is_set()
    assert len(captured) == 1 and not captured[0].allow_gateway_control
    assert "unmatched poll vote-id" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["two-open", "unknown-choice", "other-chat"])
async def test_unmatched_poll_id_does_not_guess(monkeypatch, case) -> None:
    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)
    _stub_sidecar_text(adapter, monkeypatch)
    entries = []
    for index, chat in enumerate(["+155****4567", "other-chat" if case == "other-chat" else "+155****4567"]):
        if index and case == "unknown-choice":
            break
        entry = cg.register(f"clar-{index}", f"sess-{index}", "Pick one", ["Route", "Calendar"])
        _stub_sidecar_poll(adapter, monkeypatch, poll_id=f"sent-{index}")
        await adapter.send_clarify(chat, entry.question, entry.choices, entry.clarify_id, entry.session_key)
        entries.append(entry)
    vote = _poll_option_event(title="Other" if case == "unknown-choice" else "Route", poll_id="vote-id")
    if case == "other-chat":
        vote["space"]["id"] = "third-chat"
    await adapter._dispatch_inbound(vote)
    assert not any(entry.event.is_set() for entry in entries)
    assert len(captured) == 1 and not captured[0].allow_gateway_control


@pytest.mark.asyncio
async def test_vote_emitted_inside_send_poll_answers_clarify(monkeypatch) -> None:
    """Ported from the adversarial run: the live stream can deliver a tap before /send-poll returns."""
    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)
    _stub_sidecar_text(adapter, monkeypatch)
    entry = cg.register("early", "sess-1", "Early vote?", ["Route", "Calendar"])

    async def slow_send(chat, title, options):
        await adapter._dispatch_inbound(_poll_option_event(title="Route", poll_id="spc-msg-early"))
        return SendResult(success=True, message_id="spc-msg-early")

    monkeypatch.setattr(adapter, "_sidecar_send_poll", slow_send)
    await adapter.send_clarify("+155****4567", entry.question, entry.choices, "early", "sess-1")
    assert entry.event.is_set() and entry.response == "Route"
    assert captured == []


@pytest.mark.asyncio
async def test_early_vote_waits_for_its_poll_binding(monkeypatch) -> None:
    """With two prompts open the fallback cannot pick one; the early vote waits for its id."""
    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)
    _stub_sidecar_text(adapter, monkeypatch)
    _stub_sidecar_poll(adapter, monkeypatch, poll_id="first-poll")
    first = cg.register("first", "sess-1", "First?", ["Route", "Calendar"])
    await adapter.send_clarify("+155****4567", first.question, first.choices, "first", "sess-1")
    second = cg.register("second", "sess-1", "Second?", ["Route", "Calendar"])

    async def slow_send(chat, title, options):
        await adapter._dispatch_inbound(_poll_option_event(title="Calendar", poll_id="second-poll"))
        assert not second.event.is_set()
        return SendResult(success=True, message_id="second-poll")

    monkeypatch.setattr(adapter, "_sidecar_send_poll", slow_send)
    await adapter.send_clarify("+155****4567", second.question, second.choices, "second", "sess-1")
    assert second.response == "Calendar"
    assert not first.event.is_set()
    assert captured == []


@pytest.mark.asyncio
async def test_parallel_polls_in_different_chats(monkeypatch) -> None:
    """Ported from the adversarial run: a poll id only answers in the chat it was sent to."""
    import asyncio

    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)
    _stub_sidecar_text(adapter, monkeypatch)
    count = 0

    async def poll(chat, title, options):
        nonlocal count
        count += 1
        return SendResult(success=True, message_id=f"poll-{count}")

    monkeypatch.setattr(adapter, "_sidecar_send_poll", poll)
    first = cg.register("a", "sess-a", "A?", ["Route", "Calendar"])
    second = cg.register("b", "sess-b", "B?", ["Route", "Calendar"])
    await asyncio.gather(
        adapter.send_clarify("chat-a", first.question, first.choices, "a", "sess-a"),
        adapter.send_clarify("chat-b", second.question, second.choices, "b", "sess-b"))
    first_poll = adapter._open_poll_clarifies["a"].poll_id
    second_poll = adapter._open_poll_clarifies["b"].poll_id
    wrong = _poll_option_event(title="Route", poll_id=first_poll)
    wrong["space"]["id"] = "chat-b"
    await adapter._dispatch_inbound(wrong)
    assert not first.event.is_set() and not second.event.is_set()
    for chat, poll_id, title in [("chat-a", first_poll, "Route"), ("chat-b", second_poll, "Calendar")]:
        vote = _poll_option_event(title=title, poll_id=poll_id, msg_id=f"vote-{chat}")
        vote["space"]["id"] = chat
        await adapter._dispatch_inbound(vote)
    assert (first.response, second.response) == ("Route", "Calendar")
    assert len(captured) == 1 and not captured[0].allow_gateway_control


@pytest.mark.asyncio
async def test_multi_select_clarify_uses_numbered_text(monkeypatch) -> None:
    """A poll tap would finish a multi-select prompt on the first choice."""
    adapter = _make_adapter(monkeypatch)
    sends = _stub_sidecar_text(adapter, monkeypatch)
    poll_calls = _stub_sidecar_poll(adapter, monkeypatch)
    entry = cg.register("multi", "sess-1", "Pick any", ["Route", "Calendar", "Mail"], multi_select=True)
    result = await adapter.send_clarify("chat", entry.question, entry.choices, "multi", "sess-1")
    assert result.success
    assert poll_calls == []
    assert len(sends) == 1 and "Pick any" in sends[0][1] and "1. Route" in sends[0][1]
    assert entry.awaiting_text
    await adapter._dispatch_inbound(_poll_option_event(title="Route"))
    assert not entry.event.is_set()
    assert cg.resolve_text_response_for_session("sess-1", "1, 3")
    assert entry.response is not None and "Route" in entry.response and "Mail" in entry.response
