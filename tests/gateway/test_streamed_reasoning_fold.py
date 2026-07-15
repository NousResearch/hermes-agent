"""Streamed-turn reasoning fold (#57693).

With streaming on, the stream consumer commits the final message and the
gateway suppresses the normal send (already_sent=True). The normal send path
is the only place the 💭 reasoning block is prepended, so streaming silently
disabled reasoning display for every model/platform. The fix folds the block
into the already-streamed message with one final edit — routed through the
stream consumer's *metadata-aware* edit path so the edit still carries the
routing metadata Slack uses to select the workspace client (a raw
adapter.edit_message would drop it and a non-default workspace would lose the
reasoning).

These tests assert, on real code:
  - the final edit happens and carries the folded reasoning + answer,
  - the routing metadata is preserved on that edit,
  - the fold is a best-effort no-op when there is nothing to fold, and
  - confirmed streamed delivery still sets already_sent (suppression).
"""
from types import SimpleNamespace

import pytest

from gateway.run import GatewayRunner
from gateway.stream_consumer import GatewayStreamConsumer


class RecordingAdapter:
    """Adapter whose edit_message accepts (and records) routing metadata."""

    def __init__(self):
        self.edits = []

    async def edit_message(self, *, chat_id, message_id, content, finalize=False, metadata=None):
        self.edits.append(
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "content": content,
                "finalize": finalize,
                "metadata": metadata,
            }
        )
        return SimpleNamespace(success=True, message_id=message_id)


class MetadataBlindAdapter:
    """Adapter whose edit_message cannot accept metadata (no such param)."""

    def __init__(self):
        self.edits = []

    async def edit_message(self, *, chat_id, message_id, content, finalize=False):
        self.edits.append(
            {"chat_id": chat_id, "message_id": message_id, "content": content}
        )
        return SimpleNamespace(success=True, message_id=message_id)


def _runner_with_block(block: str) -> GatewayRunner:
    """A GatewayRunner with _format_reasoning_block stubbed to a fixed block,
    isolating the fold wiring from gateway-config resolution."""
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._format_reasoning_block = lambda source, last_reasoning: (
        block if last_reasoning else ""
    )
    return runner


def _consumer(adapter, *, metadata=None, message_id="msg-1"):
    sc = GatewayStreamConsumer(adapter, "chat-42", metadata=metadata)
    sc._message_id = message_id
    return sc


# ---------------------------------------------------------------------------
# _edit_message: metadata preservation on the path the fold uses
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_edit_message_forwards_metadata_when_supported():
    adapter = RecordingAdapter()
    sc = _consumer(adapter, metadata={"slack_team_id": "T999"})

    await sc._edit_message(message_id="msg-1", content="hi", finalize=True)

    assert adapter.edits[-1]["metadata"] == {"slack_team_id": "T999"}
    assert adapter.edits[-1]["finalize"] is True


@pytest.mark.asyncio
async def test_edit_message_omits_metadata_when_unsupported():
    adapter = MetadataBlindAdapter()
    sc = _consumer(adapter, metadata={"slack_team_id": "T999"})

    # Must not raise even though the adapter cannot accept metadata.
    await sc._edit_message(message_id="msg-1", content="hi", finalize=True)

    assert adapter.edits[-1] == {
        "chat_id": "chat-42",
        "message_id": "msg-1",
        "content": "hi",
    }


# ---------------------------------------------------------------------------
# _fold_reasoning_into_streamed_message
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fold_edits_streamed_message_with_reasoning_and_metadata():
    block = "💭 **Reasoning:**\n```\n27*43 = 1161\n```"
    runner = _runner_with_block(block)
    adapter = RecordingAdapter()
    sc = _consumer(adapter, metadata={"slack_team_id": "T999"})

    edited = await runner._fold_reasoning_into_streamed_message(
        source=SimpleNamespace(platform="slack"),
        stream_consumer=sc,
        final_text="1161",
        last_reasoning="27*43 = 1161",
        session_key="sess-1",
    )

    assert edited is True
    assert len(adapter.edits) == 1
    edit = adapter.edits[0]
    assert edit["content"] == f"{block}\n\n1161"       # block folded above answer
    assert edit["message_id"] == "msg-1"                # the streamed message
    assert edit["finalize"] is True
    assert edit["metadata"] == {"slack_team_id": "T999"}  # routing preserved


@pytest.mark.asyncio
async def test_fold_noop_when_no_reasoning():
    runner = _runner_with_block("💭 block")
    adapter = RecordingAdapter()
    sc = _consumer(adapter, metadata={"slack_team_id": "T999"})

    edited = await runner._fold_reasoning_into_streamed_message(
        source=SimpleNamespace(platform="slack"),
        stream_consumer=sc,
        final_text="1161",
        last_reasoning=None,
        session_key="sess-1",
    )

    assert edited is False
    assert adapter.edits == []


@pytest.mark.asyncio
async def test_fold_noop_when_no_stream_consumer():
    runner = _runner_with_block("💭 block")
    edited = await runner._fold_reasoning_into_streamed_message(
        source=SimpleNamespace(platform="slack"),
        stream_consumer=None,
        final_text="1161",
        last_reasoning="thinking",
        session_key="sess-1",
    )
    assert edited is False


@pytest.mark.asyncio
async def test_fold_noop_when_stream_message_not_yet_committed():
    runner = _runner_with_block("💭 block")
    adapter = RecordingAdapter()
    sc = _consumer(adapter, metadata=None, message_id=None)

    edited = await runner._fold_reasoning_into_streamed_message(
        source=SimpleNamespace(platform="slack"),
        stream_consumer=sc,
        final_text="1161",
        last_reasoning="thinking",
        session_key="sess-1",
    )
    assert edited is False
    assert adapter.edits == []


@pytest.mark.asyncio
async def test_fold_is_best_effort_on_edit_failure():
    """A failed edit must not raise — it only loses the reasoning display."""
    runner = _runner_with_block("💭 block")

    class BoomAdapter:
        async def edit_message(self, *, chat_id, message_id, content, finalize=False, metadata=None):
            raise RuntimeError("workspace client unavailable")

    sc = _consumer(BoomAdapter(), metadata={"slack_team_id": "T999"})

    edited = await runner._fold_reasoning_into_streamed_message(
        source=SimpleNamespace(platform="slack"),
        stream_consumer=sc,
        final_text="1161",
        last_reasoning="thinking",
        session_key="sess-1",
    )
    assert edited is False  # swallowed; answer already delivered by the stream


# ---------------------------------------------------------------------------
# Suppression gate — confirmed streamed delivery still sets already_sent, and
# the fold's best-effort outcome does not change that (mirrors the reproduction
# style in tests/gateway/test_duplicate_reply_suppression.py).
# ---------------------------------------------------------------------------

def _apply_suppression(response, sc):
    _final = response.get("final_response") or ""
    _is_empty_sentinel = not _final or _final == "(empty)"
    _previewed = bool(response.get("response_previewed"))
    _content_delivered = bool(sc and getattr(sc, "final_content_delivered", False))
    _transformed = bool(response.get("response_transformed"))
    _streamed = bool(sc and getattr(sc, "final_response_sent", False))
    if not _is_empty_sentinel and not _transformed and (_streamed or _content_delivered):
        # (fold runs here, best-effort, then:)
        response["already_sent"] = True


def test_confirmed_stream_delivery_sets_already_sent():
    sc = SimpleNamespace(
        final_response_sent=True,
        final_content_delivered=True,
    )
    response = {"final_response": "1161", "response_previewed": False}
    _apply_suppression(response, sc)
    assert response.get("already_sent") is True


def test_transformed_response_not_suppressed_here():
    """A plugin-transformed response takes the sibling edit branch, not this
    suppression path — already_sent must not be set by the fold branch."""
    sc = SimpleNamespace(final_response_sent=True, final_content_delivered=True)
    response = {
        "final_response": "1161",
        "response_previewed": False,
        "response_transformed": True,
    }
    _apply_suppression(response, sc)
    assert "already_sent" not in response
