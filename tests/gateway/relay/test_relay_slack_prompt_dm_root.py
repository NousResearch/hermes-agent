"""Slack relay: interactive prompts follow the turn's thread stamp (QA-5).

The threading MODE (flat DM vs thread-per-message) is decided in exactly ONE
place: run.py's ``_resolve_progress_thread_id``, which reads
``platforms.slack.extra.reply_in_thread`` and encodes the verdict into the
outbound ``metadata`` stamp:

  * flat mode  -> the synthetic self-anchor is suppressed in run.py, so prompt
    metadata arrives with NO ``thread_id`` and the card posts at the DM root;
  * thread-per-message (default) -> ``metadata.thread_id`` is stamped for the
    whole turn; on the FIRST turn it legitimately equals the triggering
    message's ts (the synthetic root IS the thread).

The prompt lane must TRUST that stamp, like ``_resolve_reply_to_for_send``
does. Re-deriving the mode here (the old unconditional
``thread_id == message_id`` strip) exiled the approval card and its
resolved-state swap to the DM root while progress bubbles honoured the thread
(the 2026-07-27 mixed-placement report).

These are behaviour-contract tests: they assert how the outbound ``prompt``
frame relates to the inherited thread metadata (the invariant the connector
depends on), not a snapshot. They drive the REAL ``RelayAdapter`` +
``StubConnector`` end to end.
"""

from __future__ import annotations

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.relay.adapter import RelayAdapter
from gateway.relay.descriptor import CONTRACT_VERSION, CapabilityDescriptor
from gateway.session import SessionSource

from tests.gateway.relay.stub_connector import StubConnector

FULL_OPS = ("send", "edit", "typing", "get_chat_info", "send_media", "prompt", "react")


def _slack_desc(**kw) -> CapabilityDescriptor:
    base = dict(
        contract_version=CONTRACT_VERSION,
        platform="slack",
        label="Slack",
        max_message_length=4000,
        supports_draft_streaming=False,
        supports_edit=True,
        supports_threads=True,
        markdown_dialect="mrkdwn",
        len_unit="chars",
        supported_ops=FULL_OPS,
    )
    base.update(kw)
    return CapabilityDescriptor(**base)


def _wire(
    chat_id: str,
    chat_type: str,
    *,
    user_id="U1",
    scope_id=None,
    platform=Platform.SLACK,
):
    """A RelayAdapter fronting Slack, with inbound scope + chat_type captured."""
    stub = StubConnector(_slack_desc())
    adapter = RelayAdapter(PlatformConfig(), _slack_desc(), transport=stub)
    src = SessionSource(
        platform=platform,
        chat_id=chat_id,
        chat_type=chat_type,
        user_id=user_id,
        scope_id=scope_id,
    )
    adapter._capture_scope(
        MessageEvent(text="hi", source=src, message_type=MessageType.TEXT)
    )
    return adapter, stub


def _last_prompt(stub) -> dict:
    prompts = [f for f in stub.sent if f["op"] == "prompt"]
    assert prompts, "expected a prompt op on the wire"
    return prompts[-1]


# ---------------------------------------------------------------------------
# Flat mode: run.py stamps NO thread_id -> the card posts at the DM root.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_exec_approval_flat_mode_posts_at_dm_root():
    """Flat-DM turn (reply_in_thread=false): run.py suppressed the synthetic
    anchor upstream, so prompt metadata has no thread_id and none appears on
    the wire — the card posts at the DM root."""
    adapter, stub = _wire("D1", "dm", scope_id="T1")
    md = {"message_id": "1700000000.000100", "scope_id": "T1"}
    result = await adapter.send_exec_approval(
        "D1", "rm -rf /tmp/x", "sess:1", description="deletes files", metadata=md
    )
    assert result.success is True
    frame = _last_prompt(stub)
    meta = frame["metadata"] or {}
    assert "thread_id" not in meta
    assert "thread_ts" not in meta
    # reply_to on the outbound action stays unset — a root-level post.
    assert frame["reply_to"] is None
    # Tenant scope is preserved untouched (egress routing must not break).
    assert meta.get("scope_id") == "T1"


# ---------------------------------------------------------------------------
# Thread-per-message mode: the first-turn self-anchor (thread_id == message_id)
# IS the thread root — the prompt must stay in the thread (QA-5 regression).
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_exec_approval_first_turn_self_anchor_stays_in_thread():
    """Thread-per-message first turn: run.py stamps thread_id = the triggering
    message's own ts. The approval card must post INTO that thread — stripping
    it exiled the card to the home channel (2026-07-27 report)."""
    adapter, stub = _wire("D1", "dm", scope_id="T1")
    md = {
        "thread_id": "1700000000.000100",
        "message_id": "1700000000.000100",
        "scope_id": "T1",
    }
    result = await adapter.send_exec_approval(
        "D1", "rm -rf /tmp/x", "sess:1", description="deletes files", metadata=md
    )
    assert result.success is True
    frame = _last_prompt(stub)
    meta = frame["metadata"] or {}
    assert meta.get("thread_id") == "1700000000.000100", (
        "first-turn self-anchor is the thread root; the prompt must honour it"
    )
    assert meta.get("scope_id") == "T1"


@pytest.mark.asyncio
async def test_clarify_first_turn_self_anchor_stays_in_thread():
    adapter, stub = _wire("D1", "dm", scope_id="T1")
    md = {
        "thread_id": "1700000000.000200",
        "message_id": "1700000000.000200",
        "scope_id": "T1",
    }
    result = await adapter.send_clarify(
        "D1", "Which env?", ["prod", "staging"], "cl-1", "sess:1", metadata=md
    )
    assert result.success is True
    frame = _last_prompt(stub)
    meta = frame["metadata"] or {}
    assert meta.get("thread_id") == "1700000000.000200"
    assert meta.get("scope_id") == "T1"


@pytest.mark.asyncio
async def test_slash_confirm_first_turn_self_anchor_stays_in_thread():
    """The stamp-trusting rule covers every prompt surface (single
    _send_prompt choke point)."""
    adapter, stub = _wire("D1", "dm")
    md = {"thread_id": "1700000000.000300", "message_id": "1700000000.000300"}
    await adapter.send_slash_confirm(
        "D1", "Reload MCP", "invalidates cache", "s", "cf-1", metadata=md
    )
    frame = _last_prompt(stub)
    assert (frame["metadata"] or {}).get("thread_id") == "1700000000.000300"


# ---------------------------------------------------------------------------
# Regression guards: a REAL thread and non-DM / non-Slack chats are untouched
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_exec_approval_in_real_thread_keeps_thread_id():
    """A DM prompt raised inside a REAL thread (thread_id distinct from the
    triggering message ts) stays in that thread."""
    adapter, stub = _wire("D1", "dm", scope_id="T1")
    md = {
        "thread_id": "1699000000.999000",
        "message_id": "1700000000.000100",
        "scope_id": "T1",
    }
    await adapter.send_exec_approval("D1", "cmd", "s", metadata=md)
    frame = _last_prompt(stub)
    assert frame["metadata"]["thread_id"] == "1699000000.999000"


@pytest.mark.asyncio
async def test_channel_approval_keeps_thread_id():
    """A Slack CHANNEL prompt keeps its thread_id (autoThread / real thread)."""
    adapter, stub = _wire("C1", "channel", scope_id="T1")
    md = {
        "thread_id": "1700000000.000400",
        "message_id": "1700000000.000400",
        "scope_id": "T1",
    }
    await adapter.send_exec_approval("C1", "cmd", "s", metadata=md)
    frame = _last_prompt(stub)
    assert frame["metadata"]["thread_id"] == "1700000000.000400"


@pytest.mark.asyncio
async def test_non_slack_dm_approval_keeps_thread_id():
    """A non-Slack relay DM keeps thread_id (its connector owns its own
    threading semantics)."""
    adapter, stub = _wire("dc1", "dm", platform=Platform.DISCORD)
    md = {"thread_id": "9000", "message_id": "9000"}
    await adapter.send_exec_approval("dc1", "cmd", "s", metadata=md)
    frame = _last_prompt(stub)
    assert frame["metadata"]["thread_id"] == "9000"


# ---------------------------------------------------------------------------
# QA-1 rich status: the relay advertises Slack's text status line and carries
# the live per-tool phrase on the typing frame (native set_status_text parity).
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_slack_relay_advertises_status_text():
    adapter, _stub = _wire("D1", "dm")
    assert adapter.supports_status_text is True


@pytest.mark.asyncio
async def test_non_slack_relay_does_not_advertise_status_text():
    stub = StubConnector(_slack_desc(platform="discord"))
    adapter = RelayAdapter(
        PlatformConfig(), _slack_desc(platform="discord"), transport=stub
    )
    assert adapter.supports_status_text is False


@pytest.mark.asyncio
async def test_typing_carries_live_status_phrase():
    """set_status_text() -> the next typing frame carries the phrase as
    content; clearing it (None) reverts to a content-less heartbeat frame
    (never an empty string, which is Slack's explicit clear)."""
    adapter, stub = _wire("D1", "dm", scope_id="T1")
    adapter.set_status_text("D1", "is running pytest…")
    await adapter.send_typing("D1", metadata={"scope_id": "T1"})
    typing = [f for f in stub.sent if f["op"] == "typing"]
    assert typing and typing[-1].get("content") == "is running pytest…"

    adapter.set_status_text("D1", None)
    await adapter.send_typing("D1", metadata={"scope_id": "T1"})
    typing = [f for f in stub.sent if f["op"] == "typing"]
    assert "content" not in typing[-1], (
        "cleared phrase must omit content (empty string means CLEAR on Slack)"
    )
