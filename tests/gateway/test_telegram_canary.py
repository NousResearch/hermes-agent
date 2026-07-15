"""Synthetic, non-private Telegram delivery canary tests."""

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from hermes_cli.telegram_canary import (
    CANARY_SCHEMA,
    SyntheticRetryProbe,
    append_private_receipt,
    redact_receipt,
    run_canary,
)
from plugins.platforms.telegram.adapter import TelegramAdapter


class _Message:
    def __init__(self, message_id: int):
        self.message_id = message_id


class _Bot:
    def __init__(self):
        self.calls = 0

    async def send_message(self, **_kwargs):
        self.calls += 1
        return _Message(1000 + self.calls)


@pytest.fixture
def strict_allowlist(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "owner-1")
    for key in (
        "GATEWAY_ALLOWED_USERS",
        "TELEGRAM_ALLOW_ALL_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.mark.asyncio
async def test_canary_exercises_auth_retry_split_dedup_and_receipt(
    tmp_path: Path,
    strict_allowlist,
    monkeypatch,
):
    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter.asyncio.sleep", AsyncMock()
    )
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="synthetic"))
    bot = _Bot()
    probe = SyntheticRetryProbe(bot)
    adapter._bot = probe

    receipt_path = tmp_path / "receipts" / "telegram-canary.jsonl"
    state_path = tmp_path / "receipts" / "telegram-canary-state.json"
    receipt, receipt_sha256 = await run_canary(
        adapter=adapter,
        probe=probe,
        chat_id="@not_recorded",
        destination_alias="offline-fixture",
        receipt_path=receipt_path,
        state_path=state_path,
        runtime_sha="a" * 40,
        run_id="00000000-0000-4000-8000-000000000001",
        created_at="2026-07-15T12:00:00Z",
    )

    assert receipt["schema"] == CANARY_SCHEMA
    assert receipt["synthetic"] is True
    assert receipt["private_data"] is False
    assert receipt["qualifies_for_health_p6"] is False
    assert receipt["result"] == "pass"
    assert receipt["checks"]["authentication"] == {
        "gateway_path_exercised": True,
        "strict_single_owner_allowlist": True,
        "owner_allowed": True,
        "unknown_denied": True,
    }
    assert receipt["checks"]["retry"]["injected_failures"] == 1
    assert receipt["checks"]["retry"]["retried"] is True
    assert receipt["checks"]["idempotency"]["delivery_attempts"] == 2
    assert receipt["checks"]["idempotency"]["actual_deliveries"] == 1
    assert receipt["checks"]["idempotency"]["duplicates_suppressed"] == 1
    assert receipt["checks"]["length"]["chunk_count"] >= 2
    assert receipt["checks"]["length"]["no_truncation"] is True
    assert all(
        units <= 4096
        for units in receipt["checks"]["length"]["chunk_utf16_units"]
    )
    assert len(receipt["checks"]["delivery"]["message_ids"]) == receipt["checks"]["length"]["chunk_count"]
    assert probe.attempts == bot.calls + 1

    assert receipt_path.stat().st_mode & 0o777 == 0o600
    assert receipt_path.parent.stat().st_mode & 0o777 == 0o700
    assert state_path.stat().st_mode & 0o777 == 0o600
    stored = json.loads(receipt_path.read_text(encoding="utf-8").strip())
    assert stored == receipt
    assert "@not_recorded" not in receipt_path.read_text(encoding="utf-8")
    assert len(receipt_sha256) == 64


@pytest.mark.asyncio
async def test_canary_fails_closed_without_single_owner_allowlist(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "*")
    monkeypatch.delenv("GATEWAY_ALLOWED_USERS", raising=False)
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="synthetic"))
    probe = SyntheticRetryProbe(_Bot())
    adapter._bot = probe

    receipt, _ = await run_canary(
        adapter=adapter,
        probe=probe,
        chat_id="@not_recorded",
        destination_alias="offline-fixture",
        receipt_path=tmp_path / "receipt.jsonl",
        state_path=tmp_path / "state.json",
        runtime_sha="b" * 40,
        run_id="00000000-0000-4000-8000-000000000002",
        created_at="2026-07-15T12:00:00Z",
    )

    assert receipt["result"] == "fail"
    assert receipt["checks"]["authentication"]["strict_single_owner_allowlist"] is False
    assert probe.attempts == 0


def test_private_receipt_is_append_only_and_redaction_removes_message_ids(tmp_path):
    path = tmp_path / "private" / "receipt.jsonl"
    first = {
        "schema": CANARY_SCHEMA,
        "checks": {"delivery": {"message_ids": ["1", "2"]}},
    }
    second = {
        "schema": CANARY_SCHEMA,
        "checks": {"delivery": {"message_ids": ["3"]}},
    }

    append_private_receipt(path, first)
    append_private_receipt(path, second)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [first, second]
    assert path.stat().st_mode & 0o777 == 0o600
    redacted = redact_receipt(second)
    assert redacted["checks"]["delivery"]["message_ids"] == ["<redacted>"]


@pytest.mark.asyncio
async def test_partial_multichunk_send_never_retries_full_payload(monkeypatch):
    from telegram.error import NetworkError

    class _PartialBot:
        def __init__(self):
            self.calls = 0

        async def send_message(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return _Message(1001)
            raise NetworkError("synthetic later-chunk connection failure")

    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter.asyncio.sleep", AsyncMock()
    )
    monkeypatch.setattr("gateway.platforms.base.asyncio.sleep", AsyncMock())
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="synthetic"))
    bot = _PartialBot()
    adapter._bot = bot

    result = await adapter._send_with_retry("@fixture", "A" * 5000)

    assert result.success is False
    assert result.retryable is False
    assert result.raw_response["partial_send"] is True
    assert result.raw_response["delivered_chunks"] == 1
    assert result.raw_response["total_chunks"] == 2
    assert result.raw_response["message_ids"] == ["1001"]
    # One successful first chunk plus the adapter's three attempts for chunk 2.
    # The base retry layer must not start the payload from chunk 1 again.
    assert bot.calls == 4
