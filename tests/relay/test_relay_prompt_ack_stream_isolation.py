"""Prompt-ack sends must never seal an open draft stream.

Live finding (rc.4 staging, 100% reproducible on approval turns): the
"✅ Approved once" acknowledgement the adapter sends after resolving an
exec-approval prompt_response carried only placement metadata (thread_id)
— no per-turn message identity and no interim marker. send()'s
single-open-stream fallback (review B2: "a chat with exactly one open
stream absorbs a turn-final arriving without identity") therefore matched
the approval turn's OWN live draft and sealed it with the ack text:

  * every subsequent append hit the post-seal tombstone (silently
    swallowed by design — built for millisecond stragglers), freezing the
    visible draft mid-word with zero log lines;
  * the turn-final then found no open draft and fell through to a plain
    send — the duplicate (fallback) message the user saw. Also silent:
    no suppression line, no seal-failed warning.

The fix marks every prompt-lifecycle system send (approval ack, slash-
confirm ack, expiry notice) as an interim send, which bypasses draft
matching entirely. These tests pin the whole class, plus the regression
contract that real turn-finals still absorb into their stream.
"""

import json

import pytest

from gateway.config import PlatformConfig
from gateway.relay.adapter import RelayAdapter
from gateway.relay.descriptor import CapabilityDescriptor


def _descriptor(**overrides):
    base = dict(
        contract_version=1,
        platform="slack",
        label="Slack",
        max_message_length=4000,
        supports_draft_streaming=True,
        supports_edit=True,
        supports_threads=True,
        markdown_dialect="mrkdwn",
        len_unit="chars",
        supported_ops=("send", "edit", "draft", "prompt"),
    )
    base.update(overrides)
    return CapabilityDescriptor(**base)


class FakeTransport:
    def __init__(self):
        self.frames = []
        self._identities = [("slack", "hermes")]

    def descriptor_for_platform(self, platform):
        return None

    async def send_outbound(self, frame, platform=None):
        self.frames.append((frame, platform))
        return {"success": True, "message_id": "1.2"}


class FakeSource:
    chat_id = "D01"
    thread_id = "111.222"


class FakeEvent:
    def __init__(self, prompt_id, option_id):
        self.prompt_response = {"prompt_id": prompt_id, "option_id": option_id}
        self.source = FakeSource()


def _adapter():
    config = PlatformConfig(enabled=True, extra={})
    return RelayAdapter(config, _descriptor(), transport=FakeTransport())


async def _open_turn_draft(a, chat_id="D01", draft_id=7):
    """Open a live draft the way a streaming turn does."""
    await a.send_draft(chat_id, draft_id, "streaming partial…")
    key = a._draft_key(chat_id, None)
    candidates = [k for k in a._open_draft_by_chat if k.startswith(f"{chat_id}:")]
    assert candidates, "test setup: draft did not arm interception"
    return candidates[0]


class TestPromptAckDoesNotSealDraft:
    @pytest.mark.asyncio
    async def test_approval_ack_leaves_open_draft_untouched(self):
        """The exact live failure: exec-approval tap resolves while the
        turn's draft stream is open; the ack must go out as its own plain
        send and the draft must STILL be armed afterwards."""
        a = _adapter()
        draft_key = await _open_turn_draft(a)

        prompt_id = a._mint_prompt(
            "exec_approval", {"session_key": "sess-1", "chat_id": "D01"}
        )
        # resolve_gateway_approval is imported inside the handler; a session
        # with no waiting entry returns 0 -> "expired" label. Either label
        # shape exercises the same send path.
        consumed = await a._consume_prompt_response(
            FakeEvent(prompt_id, "once")
        )
        assert consumed is True

        # The draft interception must still be armed for the real turn-final.
        assert draft_key in a._open_draft_by_chat, (
            "prompt ack sealed the open draft — the live stuck-stream/"
            "duplicate-final bug"
        )
        # The ack egressed as a plain send op, not a draft seal.
        ack_frames = [
            f for f, _ in a._transport.frames if f["op"] == "send"
        ]
        assert ack_frames, "ack was not sent at all"
        seal_frames = [
            f
            for f, _ in a._transport.frames
            if f["op"] == "draft" and f.get("final") is True
        ]
        assert not seal_frames, "ack egressed as a draft seal"

    @pytest.mark.asyncio
    async def test_expired_prompt_notice_leaves_open_draft_untouched(self):
        """Same class: the expiry notice for an unknown prompt id must not
        absorb into an open stream either."""
        a = _adapter()
        draft_key = await _open_turn_draft(a)
        consumed = await a._consume_prompt_response(
            FakeEvent("prompt-never-minted", "once")
        )
        assert consumed is True
        assert draft_key in a._open_draft_by_chat

    @pytest.mark.asyncio
    async def test_turn_final_still_absorbs_into_single_open_stream(self):
        """Regression control (review B2 contract): a REAL turn-final
        without identity metadata, in a chat with exactly one open stream,
        must still seal that stream."""
        a = _adapter()
        draft_key = await _open_turn_draft(a)
        result = await a.send("D01", "the full final answer")
        assert result.success
        assert draft_key not in a._open_draft_by_chat, (
            "turn-final no longer absorbs into its stream — B2 fallback broken"
        )
        seal_frames = [
            f
            for f, _ in a._transport.frames
            if f["op"] == "draft" and f.get("final") is True
        ]
        assert len(seal_frames) == 1
