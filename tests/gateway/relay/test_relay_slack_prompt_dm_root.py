"""Slack relay: interactive prompts (approval / clarify) must post FLAT at the DM root.

Reported symptom (live): an approval / clarify Block Kit card raised mid-turn in
a Slack DM was posted THREADED under the triggering message instead of at the DM
root ("the approval block was put in a thread and did not follow the setting").

Root cause: a clarify/approval prompt is emitted in reply to the triggering
inbound event, so the metadata handed to ``_send_prompt`` carries that event's
thread context — run.py's ``_thread_metadata_for_source`` stamps
``metadata["thread_id"]`` (for a Slack DM, the triggering message's own ts, used
only as a session-keying fallback), plus ``metadata["message_id"]`` = that same
ts. Forwarding ``thread_id`` makes the connector's slackRestSender thread the
prompt card UNDER the user's message. Native Slack Hermes suppresses this
synthetic DM thread anchor (``SlackAdapter._resolve_thread_ts``); the relay lane
had no such disambiguation.

These are behaviour-contract tests: they assert how the outbound ``prompt`` frame
relates to the chat type + inherited thread metadata (the invariant the connector
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
# DM-root: the synthetic self-anchor is stripped
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_exec_approval_posts_flat_at_dm_root():
    """A Slack DM approval prompt must NOT inherit the triggering message's
    synthetic thread_id — it posts flat at the DM root, matching native."""
    adapter, stub = _wire("D1", "dm", scope_id="T1")
    # run.py hands the prompt the triggering message's thread context: for a DM
    # with no real thread, thread_id == message_id (the synthetic self-anchor).
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
    # The inherited synthetic thread anchor is dropped so it posts at the DM root.
    assert "thread_id" not in meta, (
        "approval prompt must NOT inherit the triggering message thread_id"
    )
    assert "thread_ts" not in meta
    # reply_to on the outbound action stays unset — a root-level post.
    assert frame["reply_to"] is None
    # Tenant scope is preserved untouched (egress routing must not break).
    assert meta.get("scope_id") == "T1"
    # The caller's original metadata dict was not mutated in place.
    assert md.get("thread_id") == "1700000000.000100"


@pytest.mark.asyncio
async def test_clarify_posts_flat_at_dm_root():
    """A Slack DM clarify prompt (with choices) also posts flat at the DM root."""
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
    assert "thread_id" not in meta
    assert "thread_ts" not in meta
    assert frame["reply_to"] is None
    assert meta.get("scope_id") == "T1"


@pytest.mark.asyncio
async def test_slash_confirm_posts_flat_at_dm_root():
    """The DM-root rule covers every prompt surface (single _send_prompt choke)."""
    adapter, stub = _wire("D1", "dm")
    md = {"thread_id": "1700000000.000300", "message_id": "1700000000.000300"}
    await adapter.send_slash_confirm(
        "D1", "Reload MCP", "invalidates cache", "s", "cf-1", metadata=md
    )
    frame = _last_prompt(stub)
    assert "thread_id" not in (frame["metadata"] or {})


# ---------------------------------------------------------------------------
# Regression guards: a REAL thread and non-DM / non-Slack chats are untouched
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_exec_approval_in_real_thread_keeps_thread_id():
    """A DM prompt raised inside a REAL thread (thread_id distinct from the
    triggering message ts) must STAY in that thread — only the synthetic
    self-anchor is stripped."""
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
    """A Slack CHANNEL prompt keeps its thread_id (autoThread / real thread);
    the DM-only guard must not touch a non-DM chat."""
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
    """The disambiguation is Slack-scoped: a non-Slack relay DM keeps thread_id
    (its connector owns its own threading semantics)."""
    adapter, stub = _wire("dc1", "dm", platform=Platform.DISCORD)
    md = {"thread_id": "9000", "message_id": "9000"}
    await adapter.send_exec_approval("dc1", "cmd", "s", metadata=md)
    frame = _last_prompt(stub)
    assert frame["metadata"]["thread_id"] == "9000"
