"""Native-poll clarify tests for PhotonAdapter.

iMessage has a native poll bubble (spectrum-ts `poll()` builder). A
multiple-choice ``clarify`` renders as that poll; the user taps a choice and
the vote streams back inbound as a ``poll_option`` event. These tests cover
both directions without spawning the Node sidecar or binding ports:

  * outbound — ``send_clarify`` with choices POSTs ``/send-poll`` and flips the
    clarify into text-capture mode; with no choices it stays plain text;
  * inbound — a ``poll_option`` selection is dispatched as a plain-text message
    carrying the chosen option (so the gateway clarify-intercept resolves it),
    a deselection is dropped, and an empty-title vote is dropped.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, SendResult
from plugins.platforms.photon.adapter import PhotonAdapter


def _make_adapter(monkeypatch: pytest.MonkeyPatch) -> PhotonAdapter:
    monkeypatch.setenv("PHOTON_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "test-project-secret")
    cfg = PlatformConfig(enabled=True, token="", extra={})
    return PhotonAdapter(cfg)


def _capture(
    adapter: PhotonAdapter, monkeypatch: pytest.MonkeyPatch
) -> List[MessageEvent]:
    captured: List[MessageEvent] = []

    async def fake_handle(event: MessageEvent) -> None:
        captured.append(event)

    monkeypatch.setattr(adapter, "handle_message", fake_handle)
    return captured


def _poll_option_event(
    *, title: str, selected: bool = True, msg_id: str = "spc-msg-vote"
) -> Dict[str, Any]:
    return {
        "messageId": msg_id,
        "platform": "iMessage",
        "space": {"id": "+155****4567", "type": "dm", "phone": "+155****4567"},
        "sender": {"id": "+155****4567"},
        "content": {
            "type": "poll_option",
            "title": title,
            "selected": selected,
            "pollTitle": "Pick one",
        },
        "timestamp": "2026-05-14T19:06:32.000Z",
    }


# ---------------------------------------------------------------------------
# Inbound: a poll vote becomes the clarify answer.


@pytest.mark.asyncio
async def test_poll_vote_dispatched_as_choice_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A poll selection is forwarded as a plain-text message carrying the
    chosen option, so the gateway clarify text-intercept can resolve it."""
    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)

    await adapter._dispatch_inbound(
        _poll_option_event(title="Yes — native tappable buttons")
    )

    assert len(captured) == 1
    ev = captured[0]
    assert ev.text == "Yes — native tappable buttons"
    assert ev.message_type == MessageType.TEXT
    assert ev.source.chat_id == "+155****4567"


@pytest.mark.asyncio
async def test_poll_deselection_is_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Un-tapping a choice (selected=false) carries no answer — dropped."""
    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)

    await adapter._dispatch_inbound(
        _poll_option_event(title="Yes", selected=False)
    )
    assert captured == []


@pytest.mark.asyncio
async def test_poll_vote_empty_title_is_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    captured = _capture(adapter, monkeypatch)

    await adapter._dispatch_inbound(_poll_option_event(title="   "))
    assert captured == []


# ---------------------------------------------------------------------------
# Outbound: send_clarify renders a native poll for choices.


def _stub_sidecar_poll(
    adapter: PhotonAdapter, monkeypatch: pytest.MonkeyPatch, *, ok: bool = True
) -> List[Tuple[str, str, list]]:
    calls: List[Tuple[str, str, list]] = []

    async def fake_send_poll(space_id: str, title: str, options: list):
        calls.append((space_id, title, list(options)))
        return SendResult(
            success=ok,
            message_id="spc-msg-poll" if ok else None,
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

    marked: List[str] = []
    import tools.clarify_gateway as cg

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
    # The vote returns as text, so text-capture must be enabled.
    assert marked == ["clar-1"]


@pytest.mark.asyncio
async def test_send_clarify_open_ended_stays_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No choices → plain text question, never a poll."""
    adapter = _make_adapter(monkeypatch)
    poll_calls = _stub_sidecar_poll(adapter, monkeypatch)
    sends = _stub_sidecar_text(adapter, monkeypatch)

    result = await adapter.send_clarify(
        chat_id="+155****4567",
        question="What's your name?",
        choices=None,
        clarify_id="clar-2",
        session_key="sess-2",
    )

    assert result.success
    assert poll_calls == []  # no poll for open-ended
    assert len(sends) == 1
    assert "What's your name?" in sends[0][1]


@pytest.mark.asyncio
async def test_send_clarify_falls_back_to_text_when_poll_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An old sidecar without /send-poll (or a send error) degrades to the
    numbered-text clarify so the user can still answer."""
    adapter = _make_adapter(monkeypatch)
    poll_calls = _stub_sidecar_poll(adapter, monkeypatch, ok=False)
    sends = _stub_sidecar_text(adapter, monkeypatch)

    import tools.clarify_gateway as cg

    monkeypatch.setattr(cg, "mark_awaiting_text", lambda cid: None)

    result = await adapter.send_clarify(
        chat_id="+155****4567",
        question="Pick one",
        choices=["A", "B"],
        clarify_id="clar-3",
        session_key="sess-3",
    )

    assert result.success  # text fallback succeeded
    assert len(poll_calls) == 1  # poll was attempted
    assert len(sends) == 1  # then text list sent
    body = sends[0][1]
    assert "Pick one" in body
    assert "A" in body and "B" in body
