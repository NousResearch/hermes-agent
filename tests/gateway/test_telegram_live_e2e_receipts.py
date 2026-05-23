"""Tests for sanitized Telegram live E2E receipt emission."""

import json
from pathlib import Path

from gateway.config import Platform
from gateway.platforms.base import (
    MessageEvent,
    MessageType,
    SendResult,
    _telegram_live_e2e_delivery_record,
    append_telegram_live_e2e_receipt,
)
from gateway.session import SessionSource, build_session_key


def _event() -> MessageEvent:
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001234567890",
        chat_type="group",
        user_id="8558418840",
        thread_id="42",
        message_id="1001",
    )
    return MessageEvent(
        text="secret-like raw user text should never be stored sk-test-123",
        message_type=MessageType.TEXT,
        source=source,
        message_id="1001",
        platform_update_id=2002,
    )


def test_append_telegram_live_e2e_receipt_redacts_raw_text_and_ids(tmp_path, monkeypatch):
    receipts_path = tmp_path / "receipts.jsonl"
    monkeypatch.setenv("HERMES_TELEGRAM_LIVE_E2E_RECEIPTS_PATH", str(receipts_path))
    event = _event()
    session_key = build_session_key(event.source)

    delivery_record = _telegram_live_e2e_delivery_record(SendResult(success=True, message_id="9001"))
    assert delivery_record is not None
    delivery_record["message_id"] = "RAW_DELIVERY_ID_SHOULD_NOT_SURVIVE"
    delivery_record["message_id_sha256"] = "RAW_DELIVERY_ID_UNDER_ALLOWED_KEY_SHOULD_NOT_SURVIVE"
    delivery_record["message_ids_sha256"] = ["RAW_LIST_ID_SHOULD_NOT_SURVIVE", "d" * 64]
    delivery_record["error_sha256"] = "RAW_ERROR_TEXT_SHOULD_NOT_SURVIVE sk-test-789"
    delivery_record["text"] = "RAW DELIVERY TEXT sk-test-456 SHOULD NOT SURVIVE"
    written = append_telegram_live_e2e_receipt(
        event=event,
        session_key=session_key,
        phase="processing_complete",
        delivery_attempted=True,
        delivery_succeeded=True,
        delivery_results=[delivery_record],
    )

    assert written == receipts_path
    payload = json.loads(receipts_path.read_text(encoding="utf-8").strip())
    rendered = json.dumps(payload, sort_keys=True)
    assert payload["record_type"] == "telegram_live_e2e_receipt"
    assert payload["delivery_attempted"] is True
    assert payload["delivery_succeeded"] is True
    assert payload["raw_text_stored"] is False
    assert payload["raw_private_identifiers_recorded"] is False
    assert payload["secret_material_stored"] is False
    assert "secret-like raw user text" not in rendered
    assert "sk-test-123" not in rendered
    assert "-1001234567890" not in rendered
    assert "8558418840" not in rendered
    assert "9001" not in rendered
    assert "RAW_DELIVERY_ID_SHOULD_NOT_SURVIVE" not in rendered
    assert "RAW_DELIVERY_ID_UNDER_ALLOWED_KEY" not in rendered
    assert "RAW_LIST_ID_SHOULD_NOT_SURVIVE" not in rendered
    assert "RAW_ERROR_TEXT_SHOULD_NOT_SURVIVE" not in rendered
    assert "RAW DELIVERY TEXT" not in rendered
    assert "sk-test-456" not in rendered
    assert "sk-test-789" not in rendered
    assert payload["chat_id_sha256"]
    delivery_result = payload["delivery_results"][0]
    assert delivery_result["message_id_sha256"] is None
    assert delivery_result["error_sha256"] is None
    assert delivery_result["message_ids_sha256"] == ["d" * 64]


def test_append_telegram_live_e2e_receipt_ignores_non_telegram(tmp_path, monkeypatch):
    receipts_path = tmp_path / "receipts.jsonl"
    monkeypatch.setenv("HERMES_TELEGRAM_LIVE_E2E_RECEIPTS_PATH", str(receipts_path))
    event = _event()
    event.source.platform = Platform.DISCORD

    written = append_telegram_live_e2e_receipt(
        event=event,
        session_key="discord:dm:test",
        phase="processing_complete",
    )

    assert written is None
    assert not receipts_path.exists()
