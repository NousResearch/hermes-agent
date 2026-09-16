"""Config-gated WhatsApp read-only mode (``send_policy``).

Behaviour contract for the adapter half of the read-only boundary:
- ``send_policy: disabled`` refuses every adapter-level outbound path WITHOUT any HTTP
  request reaching the bridge (send, standalone send, read receipts);
- the default/``open`` policy preserves today's behaviour byte-for-byte;
- the bridge-adopt check refuses a bridge whose policy differs from the adapter's
  (an old bridge without the flag must be restarted, never adopted against a
  read-only adapter).
"""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.whatsapp.adapter import WhatsAppAdapter, _standalone_send


def _adapter(**extra) -> WhatsAppAdapter:
    return WhatsAppAdapter(PlatformConfig(enabled=True, extra={"send_policy": "disabled", **extra}))


def test_send_policy_default_is_open():
    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={}))
    assert adapter._sends_disabled is False
    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={"send_policy": "open"}))
    assert adapter._sends_disabled is False


def test_send_policy_any_non_open_value_fails_closed():
    for value in ("disabled", "DISABLED", "read-only", "typo"):
        assert _adapter(send_policy=value)._sends_disabled is True, value


def test_disabled_send_returns_refusal_without_bridge_call():
    """The bite test: with a bridge AVAILABLE, a refused send must produce ZERO transport
    calls and the policy refusal — not a transport error.

    If the guard is inert (or only present in a description), send() falls
    through to the bridge transport and this probe counts a call — the test fails.
    """
    adapter = _adapter()
    transport_calls = []

    async def _bridge_available():
        return None  # simulate a healthy bridge

    async def _counting_transport(*a, **k):
        transport_calls.append(a)
        from gateway.platforms.base import SendResult

        return SendResult(success=True, message_id="x")

    adapter._bridge_unavailable = _bridge_available
    adapter._post_bridge_message = _counting_transport

    result = asyncio.run(adapter.send("15550000001", "hello"))

    assert result.success is False
    assert "send_policy is disabled" in (result.error or "")
    assert transport_calls == [], "a read-only send must never reach the bridge transport"


def test_disabled_send_read_receipt_skips_transport():
    adapter = _adapter()
    # Receipt path guards on _send_read_receipts/_http_session; both enabled here so the
    # policy guard is the ONLY reason the call returns without transport.
    adapter._send_read_receipts = True
    adapter._http_session = object()

    awaited = []

    async def _fail_bridge_req(*a, **k):
        awaited.append(a)
        raise AssertionError("read receipt must not reach the bridge in read-only mode")

    adapter._bridge_req = _fail_bridge_req

    asyncio.run(adapter._send_read_receipt({"readReceiptKey": {"id": "x", "remoteJid": "y"}}))
    assert awaited == []


def test_standalone_send_refused_when_policy_disabled():
    result = asyncio.run(
        _standalone_send(SimpleNamespace(extra={"send_policy": "disabled"}), "15550000001", "hello")
    )
    assert "send_policy is disabled" in str(result)


def test_standalone_send_open_policy_reaches_bridge():
    """Control: with the default policy the standalone path must NOT short-circuit on policy
    (it proceeds to the transport and fails there if the bridge is absent — never with the
    policy refusal)."""
    import aiohttp  # noqa: F401  (guard the ImportError path out of the way)

    result = asyncio.run(_standalone_send(SimpleNamespace(extra={}), "15550000001", "hello"))
    assert "send_policy is disabled" not in str(result)


def test_reuse_running_bridge_refuses_policy_mismatch():
    """An old bridge (no sendsDisabled in /health) must be restarted, never adopted,
    when the adapter is read-only — otherwise the HTTP layer runs open."""
    adapter = _adapter()

    async def _probe():
        return True, {
            "status": "connected",
            "scriptHash": "abc123",
            "sendReadReceipts": False,
            # no "sendsDisabled" key: simulates a pre-flag bridge
        }

    adapter._probe_bridge_health = _probe
    import plugins.platforms.whatsapp.adapter as adapter_module

    def _fake_hash(_p):
        return "abc123"

    original = adapter_module._file_content_hash
    adapter_module._file_content_hash = _fake_hash
    try:
        adopted = asyncio.run(adapter._reuse_running_bridge(Path("bridge.js")))
    finally:
        adapter_module._file_content_hash = original

    assert adopted is False


def test_reuse_running_bridge_adopts_matching_policy():
    """Control: a connected bridge reporting the SAME policy (read-only) is adopted."""
    adapter = _adapter()

    async def _probe():
        return True, {
            "status": "connected",
            "scriptHash": "abc123",
            "sendReadReceipts": False,
            "sendsDisabled": True,
        }

    adapter._probe_bridge_health = _probe
    import plugins.platforms.whatsapp.adapter as adapter_module

    def _fake_hash(_p):
        return "abc123"

    original = adapter_module._file_content_hash
    adapter_module._file_content_hash = _fake_hash
    try:
        adopted = asyncio.run(adapter._reuse_running_bridge(Path("bridge.js")))
    finally:
        adapter_module._file_content_hash = original

    assert adopted is True
