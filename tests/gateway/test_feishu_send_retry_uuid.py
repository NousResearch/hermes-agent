"""Regression tests for stable UUID threading in Feishu send retry.

Sweeper review on #46361 flagged that the original PR's uuid5 fix was on the
right track but lived at the wrong layer: ``_send_raw_message`` computed it,
but ``_feishu_send_with_retry`` calls ``_send_raw_message`` once per attempt,
so without threading the same uuid through every attempt, retries still
generated fresh uuids and Feishu's server-side idempotency could not dedupe
the "succeeded server-side, lost client-side" case.

These tests pin the contract: ONE uuid5(payload) per logical
``_feishu_send_with_retry`` call, shared across every retry attempt and the
reply→create fallback path.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from plugins.platforms.feishu.adapter import FeishuAdapter


def _bare_adapter():
    """Bare FeishuAdapter (no __init__) with _send_raw_message replaced by a
    capturing mock. Caller wires the mock's side_effect / return_value."""
    adapter = object.__new__(FeishuAdapter)
    return adapter


@pytest.mark.asyncio
async def test_retry_attempts_share_same_uuid_value():
    """When the first send attempt raises a transient exception, the second
    attempt must arrive at _send_raw_message with the SAME uuid_value as the
    first — never a fresh uuid4. That shared idempotency key is what lets
    Feishu dedupe the case where attempt 1 actually landed server-side."""
    adapter = _bare_adapter()
    captured_uuids: list[str] = []

    async def _capture(**kwargs):
        captured_uuids.append(kwargs["uuid_value"])
        if len(captured_uuids) == 1:
            raise RuntimeError("simulated transient network error")
        return MagicMock(success=lambda: True, data=MagicMock(message_id="m1"))

    adapter._send_raw_message = _capture

    await adapter._feishu_send_with_retry(
        chat_id="oc_test",
        msg_type="text",
        payload='{"text":"hello"}',
        reply_to=None,
        metadata=None,
    )

    assert len(captured_uuids) == 2, "retry did not happen"
    assert captured_uuids[0] == captured_uuids[1], (
        f"retry used a different uuid_value: {captured_uuids!r} — Feishu "
        "cannot dedupe a server-side-success-but-client-side-loss duplicate."
    )


@pytest.mark.asyncio
async def test_uuid_value_is_deterministic_from_payload():
    """The idempotency key is uuid5(NAMESPACE_DNS, payload) — deterministic
    across processes and across attempts for the same content. Two sends of
    the same payload produce the same uuid_value."""
    import uuid as _uuid

    adapter = _bare_adapter()
    captured: list[str] = []

    async def _capture(**kwargs):
        captured.append(kwargs["uuid_value"])
        return MagicMock(success=lambda: True, data=MagicMock(message_id="m1"))

    adapter._send_raw_message = _capture
    payload = '{"text":"same content"}'
    await adapter._feishu_send_with_retry(
        chat_id="oc_a", msg_type="text", payload=payload,
        reply_to=None, metadata=None,
    )
    await adapter._feishu_send_with_retry(
        chat_id="oc_b", msg_type="text", payload=payload,
        reply_to=None, metadata=None,
    )

    expected = _uuid.uuid5(_uuid.NAMESPACE_DNS, payload).hex
    assert captured == [expected, expected], (
        f"uuid_value must be uuid5(payload); got {captured!r}"
    )


@pytest.mark.asyncio
async def test_different_payloads_get_different_uuid_values():
    """Sanity: distinct payloads produce distinct uuid_values. Guards against
    a constant-uuid regression that would cause Feishu to dedupe distinct
    messages as if they were duplicates."""
    adapter = _bare_adapter()
    captured: list[str] = []

    async def _capture(**kwargs):
        captured.append(kwargs["uuid_value"])
        return MagicMock(success=lambda: True, data=MagicMock(message_id="m1"))

    adapter._send_raw_message = _capture
    await adapter._feishu_send_with_retry(
        chat_id="oc_a", msg_type="text", payload='{"text":"one"}',
        reply_to=None, metadata=None,
    )
    await adapter._feishu_send_with_retry(
        chat_id="oc_b", msg_type="text", payload='{"text":"two"}',
        reply_to=None, metadata=None,
    )

    assert captured[0] != captured[1], (
        f"distinct payloads must produce distinct uuid_values; got {captured!r}"
    )


@pytest.mark.asyncio
async def test_reply_to_create_fallback_shares_uuid():
    """When the reply path returns a 'message withdrawn/missing' code, the
    fallback _send_raw_message (reply_to=None) must reuse the SAME uuid_value
    as the original reply attempt. Otherwise Feishu would see two unrelated
    messages for one logical send."""
    adapter = _bare_adapter()
    captured: list[tuple[Optional[str], str]] = []

    async def _capture(**kwargs):
        captured.append((kwargs.get("reply_to"), kwargs["uuid_value"]))
        if kwargs.get("reply_to"):
            # Reply path returns a withdrawn-message failure code
            # (see _FEISHU_REPLY_FALLBACK_CODES in adapter).
            return MagicMock(success=lambda: False, code=230011)
        return MagicMock(success=lambda: True, data=MagicMock(message_id="m_new"))

    adapter._send_raw_message = _capture

    await adapter._feishu_send_with_retry(
        chat_id="oc_test",
        msg_type="text",
        payload='{"text":"hi"}',
        reply_to="om_withdrawn",
        metadata=None,
    )

    assert len(captured) == 2, "fallback path was not invoked"
    (reply1, uuid1), (reply2, uuid2) = captured
    assert reply1 == "om_withdrawn", "first attempt lost its reply_to"
    assert reply2 is None, "fallback attempt did not clear reply_to"
    assert uuid1 == uuid2, (
        f"reply→create fallback used a different uuid_value: {captured!r}"
    )
