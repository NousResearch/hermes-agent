"""The egress guardrail must hold at the adapter methods themselves.

Regression tests for the review finding that guarding individual call sites
(_send_with_retry) left the streaming path, media captions, and ledger
redelivery unguarded: every concrete adapter's ``send``/``edit_message`` is
wrapped at subclass creation, so ANY caller is covered.
"""

import asyncio

import pytest

from gateway.platforms.base import BasePlatformAdapter, SendResult
from hermes_durability.egress import BLOCK_ERROR

# Synthetic fixtures, split so the source holds no token-shaped literal.
GHP = "ghp" + "_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789"


class RecordingAdapter(BasePlatformAdapter):
    """Minimal concrete adapter capturing what the platform would receive."""

    def __init__(self):
        self.sent = []
        self.edited = []

    @property
    def name(self):
        return "recording"

    @property
    def platform(self):
        return "recording"

    async def connect(self, *, is_reconnect=False):  # pragma: no cover
        return True

    async def disconnect(self):  # pragma: no cover - not exercised
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(content)
        return SendResult(success=True, message_id="1")

    async def edit_message(self, chat_id, message_id, content, *,
                           finalize=False):
        self.edited.append(content)
        return SendResult(success=True, message_id=message_id)

    async def get_chat_info(self, chat_id):  # pragma: no cover
        return {}


@pytest.fixture
def adapter():
    return RecordingAdapter()


def test_send_is_wrapped(adapter):
    result = asyncio.run(adapter.send("chat", f"token {GHP} end"))
    assert result.success
    assert len(adapter.sent) == 1
    assert GHP not in adapter.sent[0]
    assert "end" in adapter.sent[0]


def test_edit_message_is_wrapped(adapter):
    # The streaming path delivers model text via edit_message, not send.
    result = asyncio.run(
        adapter.edit_message("chat", "42", f"stream {GHP} tail"))
    assert result.success
    assert GHP not in adapter.edited[0]


def test_blocked_send_returns_stable_error(adapter, monkeypatch):
    import agent.redact as redact_mod

    def boom(*a, **k):
        raise RuntimeError("redactor broke")

    monkeypatch.setattr(redact_mod, "redact_sensitive_text", boom)
    result = asyncio.run(adapter.send("chat", "anything"))
    assert not result.success
    assert result.error == BLOCK_ERROR
    assert not result.retryable
    assert adapter.sent == []


def test_wrapping_not_doubled_in_subclasses():
    class Child(RecordingAdapter):
        pass

    # Child doesn't define its own send: it inherits the already-wrapped
    # method, and __init_subclass__ must not wrap it again (plugin
    # middleware would run twice per body).
    assert Child.send is RecordingAdapter.send
    assert getattr(RecordingAdapter.__dict__["send"], "_egress_guarded", False)


def test_clean_content_passes_unchanged(adapter):
    asyncio.run(adapter.send("chat", "hello world"))
    assert adapter.sent == ["hello world"]


# ── outbound_message category contract ──────────────────────────────────────
# Plugins key on ``category``; the documented values must be exactly the ones
# the boundaries that call plugin middleware actually pass.

def _record_categories(monkeypatch):
    import hermes_cli.middleware as mw
    from hermes_cli.middleware import OutboundMessageResult

    seen = []

    def fake(text, **context):
        seen.append(context.get("category"))
        return OutboundMessageResult(text=text, original_text=text)

    monkeypatch.setattr(mw, "apply_outbound_message_middleware", fake)
    return seen


def _relay_send(text):
    from gateway.config import Platform
    from gateway.delivery import DeliveryTransport

    class Relay:
        async def send_for_platform(self, platform, chat_id, content, metadata=None):
            return SendResult(success=True, message_id="r1")

    transport = DeliveryTransport(Relay(), None, Platform.RELAY)
    return asyncio.run(transport.send(Platform.TELEGRAM, "chat", text, None))


def test_middleware_categories_match_the_boundaries(adapter, monkeypatch):
    seen = _record_categories(monkeypatch)
    asyncio.run(adapter.send("chat", "one"))
    asyncio.run(adapter.edit_message("chat", "42", "two"))
    _relay_send("three")
    assert seen == ["adapter_send", "adapter_edit_message", "delivery_relay"]


def test_redaction_only_passes_never_reach_plugins(monkeypatch):
    from hermes_durability.egress import guard_outbound_text

    seen = _record_categories(monkeypatch)
    for category in ("final_response", "send_message_tool"):
        guard_outbound_text("body", platform="telegram", category=category,
                            apply_middleware=False)
    assert seen == []


def test_documented_categories_are_the_passed_ones():
    import re
    from pathlib import Path

    doc = (Path(__file__).resolve().parents[2]
           / "website" / "docs" / "developer-guide" / "middleware.md").read_text()
    rows = re.findall(r"^\| `([a-z_]+)` \| (?:the wrapped|`DeliveryTransport)", doc, re.M)
    assert sorted(rows) == ["adapter_edit_message", "adapter_send", "delivery_relay"]
