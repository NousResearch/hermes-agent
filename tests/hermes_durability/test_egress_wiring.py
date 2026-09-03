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

GHP = "ghp_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789"


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
