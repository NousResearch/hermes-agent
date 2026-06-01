"""Tests for AWF Telegram local-file integration v0."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, SendResult
from gateway.session import SessionSource


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _write_jsonl(path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in items), encoding="utf-8")


@pytest.fixture
def awf_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "awf" / "details").mkdir(parents=True)
    _write_json(
        tmp_path / "awf" / "status.json",
        {
            "summary": "2 gates pending",
            "runs": [
                {"id": "RUN-7", "status": "pending", "stage": "approval"},
            ],
        },
    )
    _write_json(
        tmp_path / "awf" / "details" / "RUN-7.json",
        {
            "id": "RUN-7",
            "issue": "ENG-123",
            "stage": "approval",
            "summary": "Ship synthesis plan",
            "proof_commands": ["pytest tests/gateway/test_awf_local_file_contract.py -q"],
            "denied_commands": ["git push", "curl https://linear.app/graphql"],
            "linear_url": "https://linear.app/acme/issue/ENG-123",
        },
    )
    _write_json(
        tmp_path / "awf" / "pending-gates.json",
        {
            "gates": [
                {
                    "gate_id": "gate-1",
                    "issue": "ENG-123",
                    "run": "RUN-7",
                    "stage": "approval",
                    "summary": "Ship synthesis plan",
                    "proof_commands": ["pytest tests/gateway/test_awf_local_file_contract.py -q"],
                    "denied_commands": ["git push", "curl https://linear.app/graphql"],
                    "linear_url": "https://linear.app/acme/issue/ENG-123",
                }
            ]
        },
    )
    return tmp_path


def _awf_config(home, approvers=("42",), **overrides):
    config = {
        "status_path": "awf/status.json",
        "details_dir": "awf/details",
        "pending_gates_path": "awf/pending-gates.json",
        "approval_events_path": "awf/approval-events.jsonl",
        "approver_telegram_user_ids": list(approvers),
    }
    config.update(overrides)
    return config


def _event(text, user_id="42"):
    return MessageEvent(
        text=text,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            user_id=user_id,
            chat_id="100",
            user_name="Ada",
        ),
        message_id="m1",
    )


def _runner(awf_cfg):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(awf=awf_cfg)
    return runner


@pytest.mark.asyncio
async def test_awf_status_reads_configured_status_file(awf_home):
    runner = _runner(_awf_config(awf_home))

    result = await runner._handle_awf_command(_event("/awf status"))

    assert "AWF status" in result
    assert "2 gates pending" in result
    assert "RUN-7" in result


@pytest.mark.asyncio
async def test_awf_details_reads_issue_or_run_from_details_dir(awf_home):
    runner = _runner(_awf_config(awf_home))

    result = await runner._handle_awf_command(_event("/awf details RUN-7"))

    assert "AWF details" in result
    assert "ENG-123" in result
    assert "Ship synthesis plan" in result
    assert "pytest tests/gateway/test_awf_local_file_contract.py -q" in result
    assert "git push" in result


@pytest.mark.asyncio
async def test_awf_approve_appends_event_and_duplicate_is_already_resolved(awf_home):
    runner = _runner(_awf_config(awf_home))

    first = await runner._handle_awf_command(_event("/awf approve ENG-123 approval"))
    second = await runner._handle_awf_command(_event("/awf approve ENG-123 approval"))

    assert "approved" in first.lower()
    assert "already resolved" in second.lower()
    events = [json.loads(line) for line in (awf_home / "awf" / "approval-events.jsonl").read_text().splitlines()]
    assert len(events) == 1
    assert events[0]["decision"] == "approved"
    assert events[0]["issue"] == "ENG-123"
    assert events[0]["stage"] == "approval"
    assert events[0]["actor"]["telegram_user_id"] == "42"


@pytest.mark.asyncio
async def test_awf_reject_requires_approver_and_reason(awf_home):
    runner = _runner(_awf_config(awf_home, approvers=("42",)))

    missing_reason = await runner._handle_awf_command(_event("/awf reject ENG-123 approval"))
    unauthorized = await runner._handle_awf_command(
        _event("/awf reject ENG-123 approval not enough proof", user_id="99")
    )
    allowed = await runner._handle_awf_command(
        _event("/awf reject ENG-123 approval not enough proof", user_id="42")
    )

    assert "Usage: /awf reject" in missing_reason
    assert "not authorized" in unauthorized.lower()
    assert "rejected" in allowed.lower()


@pytest.mark.asyncio
async def test_telegram_awf_card_uses_awf_callback_prefixes_and_loads_gate_details_for_callback(awf_home, monkeypatch):
    from gateway.platforms.telegram import TelegramAdapter
    import gateway.platforms.telegram as telegram_mod
    from gateway.config import PlatformConfig

    class FakeInlineKeyboardButton:
        def __init__(self, text, callback_data=None, url=None):
            self.text = text
            self.callback_data = callback_data
            self.url = url
        def to_dict(self):
            data = {"text": self.text}
            if self.callback_data is not None:
                data["callback_data"] = self.callback_data
            if self.url is not None:
                data["url"] = self.url
            return data

    class FakeInlineKeyboardMarkup:
        def __init__(self, inline_keyboard):
            self.inline_keyboard = inline_keyboard

    monkeypatch.setattr(telegram_mod, "InlineKeyboardButton", FakeInlineKeyboardButton)
    monkeypatch.setattr(telegram_mod, "InlineKeyboardMarkup", FakeInlineKeyboardMarkup)

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="dummy"))
    adapter._bot = SimpleNamespace(send_message=AsyncMock(return_value=SimpleNamespace(message_id=55)))
    adapter._reply_to_mode = "first"
    adapter._disable_link_previews = True
    adapter._is_callback_user_authorized = lambda *a, **kw: True
    adapter._metadata_thread_id = lambda metadata: None
    adapter._reply_to_message_id_for_send = lambda *a, **kw: None
    adapter._thread_kwargs_for_send = lambda *a, **kw: {}
    adapter._link_preview_kwargs = lambda: {}
    adapter._send_message_with_thread_fallback = AsyncMock(return_value=SimpleNamespace(message_id=55))

    gate = {
        "gate_id": "gate-1",
        "issue": "ENG-123",
        "stage": "approval",
        "summary": "Ship synthesis plan",
        "proof_commands": ["pytest -q"],
        "denied_commands": ["git push"],
        "linear_url": "https://linear.app/acme/issue/ENG-123",
    }
    send_result = await adapter.send_awf_approval_card("100", gate)

    assert send_result == SendResult(success=True, message_id="55")
    kwargs = adapter._send_message_with_thread_fallback.call_args.kwargs
    keyboard = kwargs["reply_markup"].inline_keyboard
    def _button_dict(button):
        return button.to_dict() if hasattr(button, "to_dict") else button.__dict__

    callback_data = [
        _button_dict(button).get("callback_data")
        for row in keyboard
        for button in row
        if _button_dict(button).get("callback_data")
    ]
    assert "awf:a:gate-1" in callback_data, repr(keyboard)
    assert "awf:r:gate-1" in callback_data, repr(keyboard)
    assert "awf:d:gate-1" in callback_data, repr(keyboard)
    assert any((_button_dict(button).get("url") == "https://linear.app/acme/issue/ENG-123") for row in keyboard for button in row)

    config_path = awf_home / "config.yaml"
    config_path.write_text("awf:\n  status_path: awf/status.json\n  details_dir: awf/details\n  pending_gates_path: awf/pending-gates.json\n  approval_events_path: awf/approval-events.jsonl\n  approver_telegram_user_ids: ['42']\n", encoding="utf-8")

    query = SimpleNamespace(
        data="awf:a:gate-1",
        from_user=SimpleNamespace(id=42, first_name="Ada", username="ada"),
        message=SimpleNamespace(chat_id=100, chat=SimpleNamespace(type="private"), message_thread_id=None, message_id=55),
        answer=AsyncMock(),
        edit_message_text=AsyncMock(),
    )
    update = SimpleNamespace(callback_query=query)

    await adapter._handle_callback_query(update, SimpleNamespace())
    await adapter._handle_callback_query(update, SimpleNamespace())

    query.answer.assert_any_await(text="✅ AWF approved")
    query.answer.assert_any_await(text="This gate is already resolved.")
    events = [json.loads(line) for line in (awf_home / "awf" / "approval-events.jsonl").read_text().splitlines()]
    assert len(events) == 1
    assert events[0]["gate_id"] == "gate-1"
    assert events[0]["decision"] == "approved"


def test_awf_config_derives_card_outbox_paths_from_pending_gates(awf_home):
    from gateway.awf import load_awf_config

    cfg = load_awf_config(SimpleNamespace(awf=_awf_config(awf_home, auto_send_cards=True)))

    assert cfg.auto_send_cards is True
    assert cfg.card_requests_path == awf_home / "awf" / "telegram-card-requests.jsonl"
    assert cfg.card_results_path == awf_home / "awf" / "telegram-card-results.jsonl"
    assert cfg.card_chat_id == "42"


@pytest.mark.asyncio
async def test_awf_card_request_watcher_sends_card_once_and_records_result(awf_home):
    from gateway.run import GatewayRunner

    card_request = {
        "type": "awf.telegram.card.requested",
        "request_id": "card-gate-1",
        "gate_id": "gate-1",
        "status": "pending",
        "chat_id": "100",
        "thread_id": "55",
    }
    _write_jsonl(awf_home / "awf" / "telegram-card-requests.jsonl", [card_request])

    runner = _runner(_awf_config(awf_home, auto_send_cards=True))
    adapter = SimpleNamespace(
        send_awf_approval_card=AsyncMock(return_value=SendResult(success=True, message_id="77"))
    )
    runner.adapters = {Platform.TELEGRAM: adapter}

    first = await GatewayRunner._send_pending_awf_cards_once(runner)
    second = await GatewayRunner._send_pending_awf_cards_once(runner)

    assert first == 1
    assert second == 0
    adapter.send_awf_approval_card.assert_awaited_once()
    args, kwargs = adapter.send_awf_approval_card.call_args
    assert args[0] == "100"
    assert args[1]["gate_id"] == "gate-1"
    assert kwargs["metadata"] == {"thread_id": "55"}

    events = [
        json.loads(line)
        for line in (awf_home / "awf" / "telegram-card-results.jsonl").read_text().splitlines()
    ]
    assert len(events) == 1
    assert events[0]["type"] == "awf.telegram.card.send_result"
    assert events[0]["result"] == "sent"
    assert events[0]["request_id"] == "card-gate-1"
    assert events[0]["message_id"] == "77"
