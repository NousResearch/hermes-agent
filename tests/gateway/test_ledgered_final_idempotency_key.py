"""A ledgered final's sends carry its obligation id as ``metadata["idempotency_key"]``.

Every re-send of one final reply (``_send_with_retry``'s retries, its plain-text fallback, the
delivery ledger's reconnect/boot redelivery) presents the same key, so an adapter whose platform
dedups by a client token (Discord's message nonce) can return the message the first send already
created instead of posting a second copy. A send that is not the reply itself (the delivery-failure
notice) must not carry the key, or the reply's later redelivery would be deduplicated onto it.
"""

import os
from typing import Any

import pytest

from gateway import delivery_ledger as dl
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource

SESSION_KEY = "agent:main:telegram:dm:42"


class _Adapter(BasePlatformAdapter):
    """Scripted send results; records every send's content and metadata."""

    def __init__(self, results=()):
        super().__init__(PlatformConfig(enabled=True), Platform.TELEGRAM)
        self._results = list(results)
        self.sent: list[tuple[str, Any]] = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        pass

    async def get_chat_info(self, chat_id: str) -> dict[str, Any]:
        return {}

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append((content, metadata))
        return self._results.pop(0) if self._results else SendResult(success=True, message_id="ok")


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(dl, "_db_path", lambda: home / "state.db")
    monkeypatch.setattr(dl, "_owner_stamp", lambda: (os.getpid(), 202))
    enabled = {"on": True}
    monkeypatch.setattr(dl, "ledger_enabled", lambda config=None: enabled["on"])

    async def _no_sleep(_delay):
        return None
    monkeypatch.setattr("gateway.platforms.base.asyncio.sleep", _no_sleep)
    return enabled


def _obligation_ids() -> list:
    with dl._connect() as conn:
        return [r[0] for r in conn.execute("SELECT obligation_id FROM delivery_obligations")]


def _event() -> MessageEvent:
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="42", chat_type="dm")
    return MessageEvent(text="hi", source=source, message_id="7")


@pytest.mark.asyncio
async def test_every_send_of_a_ledgered_final_carries_its_obligation_id(ledger):
    """Retry and plain-text fallback included; the caller's metadata is not mutated."""
    adapter = _Adapter([
        SendResult(success=False, error="ConnectionError: reset", retryable=True),
        SendResult(success=False, error="Bad Request: can't parse entities"),  # -> plain fallback
    ])
    metadata = {"thread_id": "9"}

    result, _ = await adapter.send_final_ledgered(_event(), SESSION_KEY, "the answer", metadata, reply_to=None)

    assert result.success
    [obligation_id] = _obligation_ids()
    assert len(adapter.sent) == 3  # first send, one retry, the plain-text fallback
    assert [m["idempotency_key"] for _, m in adapter.sent] == [obligation_id] * 3
    assert all(m["thread_id"] == "9" for _, m in adapter.sent)
    assert metadata == {"thread_id": "9"}


@pytest.mark.asyncio
async def test_ledger_off_sends_the_metadata_unchanged(ledger):
    ledger["on"] = False
    adapter = _Adapter()
    metadata = {"thread_id": "9"}

    await adapter.send_final_ledgered(_event(), SESSION_KEY, "the answer", metadata, reply_to=None)

    assert adapter.sent == [("the answer", {"thread_id": "9"})]


@pytest.mark.asyncio
async def test_delivery_failure_notice_does_not_reuse_the_reply_key(ledger):
    down = SendResult(success=False, error="ConnectionError: reset", retryable=True)
    adapter = _Adapter([down, down, down])

    result, _ = await adapter.send_final_ledgered(_event(), SESSION_KEY, "the answer", {}, reply_to=None)

    assert not result.success
    *reply_sends, (notice, notice_metadata) = adapter.sent
    assert [content for content, _ in reply_sends] == ["the answer"] * 3
    assert all("idempotency_key" in m for _, m in reply_sends)
    assert notice != "the answer"
    assert "idempotency_key" not in (notice_metadata or {})
