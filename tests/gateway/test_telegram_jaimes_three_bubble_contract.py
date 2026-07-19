import asyncio
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from plugins.platforms.telegram import adapter as telegram_adapter
from plugins.platforms.telegram.adapter import TelegramAdapter
from gateway.platforms.base import BasePlatformAdapter


class FakeButton:
    def __init__(self, text, callback_data=None):
        self.text = text
        self.callback_data = callback_data


class FakeMarkup:
    def __init__(self, rows):
        self.inline_keyboard = rows


def _adapter():
    return object.__new__(TelegramAdapter)


def test_jaimes_final_prose_never_infers_buttons():
    content = """🤖 gpt-5.6-sol (Codex subscription)

**Objective Complete:** Yes

**TLDR:**
- Work completed

**Challenges/Blockers:**
None

**Next steps for approval:**
1. Run live canary
2. Keep current configuration
"""
    with patch.object(telegram_adapter, "InlineKeyboardButton", FakeButton), patch.object(
        telegram_adapter, "InlineKeyboardMarkup", FakeMarkup
    ):
        markup = _adapter()._jaimes_topic17_reply_markup(
            content,
            "17",
            {"notify": True},
        )
    assert markup is None


def test_jaimes_final_no_action_has_no_buttons():
    content = """🤖 gpt-5.6-sol (Codex subscription)

**Objective Complete:** Yes

**Next steps for approval:**
No action needed
"""
    with patch.object(telegram_adapter, "InlineKeyboardButton", FakeButton), patch.object(
        telegram_adapter, "InlineKeyboardMarkup", FakeMarkup
    ):
        markup = _adapter()._jaimes_topic17_reply_markup(
            content,
            "17",
            {"notify": True},
        )
    assert markup is None


def test_jaimes_final_no_action_with_explanation_has_no_buttons():
    content = """🤖 gpt-5.6-sol (Codex subscription)

**Objective Complete:** Yes

**Next steps for approval:**
- No action needed; future alerts use this layout
"""
    with patch.object(telegram_adapter, "InlineKeyboardButton", FakeButton), patch.object(
        telegram_adapter, "InlineKeyboardMarkup", FakeMarkup
    ):
        markup = _adapter()._jaimes_topic17_reply_markup(
            content,
            "17",
            {"notify": True},
        )
    assert markup is None


def test_jaimes_final_formatter_unavailable_fails_closed(tmp_path):
    adapter = _adapter()
    adapter.platform = type("Platform", (), {"value": "telegram"})()
    with patch.object(telegram_adapter._Path, "home", return_value=tmp_path), \
         patch.object(telegram_adapter.logger, "warning"):
        with pytest.raises(RuntimeError, match="validation unavailable"):
            asyncio.run(adapter._jaimes_canonical_final_before_send(
                "-1003589561528",
                "Complete: Yes - unverified raw result",
                "17",
                {"notify": True},
            ))


def test_jaimes_final_builds_pre_delivery_edit_for_current_active_card(tmp_path):
    state_path = tmp_path / ".openclaw" / "telegram" / "jaimes_fast_ack_state.json"
    script = tmp_path / ".openclaw" / "workspace" / "mission-control" / "scripts" / "jaimes_work_card.py"
    state_path.parent.mkdir(parents=True)
    script.parent.mkdir(parents=True)
    script.write_text("# test\n")
    state_path.write_text(json.dumps({"active_cards": {
        "current": {
            "status": "active",
            "key": "turn-123",
            "ack_message_id": "44",
            "objective": "Verify the KALEIDO dip against tactical entry gates",
            "telegram_chat_id": "-1003589561528",
            "telegram_thread_id": "17",
            "work_id": "work-current",
            "ledger_run_id": "run-current",
            "task_started_at": "2026-07-12T23:21:59Z",
            "started_at": "2026-07-12T23:22:00Z",
        }
    }}))
    content = "Model: openai-codex/gpt-5.6-sol | Route: live market check | Why: verified\nComplete: Yes"
    with patch.object(telegram_adapter._Path, "home", return_value=tmp_path):
        command = _adapter()._jaimes_pre_final_card_command(
            "-1003589561528", content, "17", {"notify": True}
        )
    assert command is not None
    assert command[2] == "update"
    assert command[command.index("--key") + 1] == "turn-123"
    assert command[command.index("--now") + 1] == "Final summary validated; sending now"
    assert command[command.index("--model") + 1] == "openai-codex/gpt-5.6-sol"
    assert command[command.index("--work-id") + 1] == "work-current"
    assert command[command.index("--run-id") + 1] == "run-current"
    assert command[command.index("--task-started-at") + 1] == "2026-07-12T23:21:59Z"


def test_jaimes_post_delivery_command_links_exact_final_message_before_close(tmp_path):
    state_path = tmp_path / ".openclaw" / "telegram" / "jaimes_fast_ack_state.json"
    script = tmp_path / ".openclaw" / "workspace" / "mission-control" / "scripts" / "jaimes_work_card.py"
    state_path.parent.mkdir(parents=True)
    script.parent.mkdir(parents=True)
    script.write_text("# test\n")
    state_path.write_text(json.dumps({"active_cards": {
        "current": {
            "status": "active",
            "key": "turn-123",
            "ack_message_id": "44",
            "objective": "Verify Topic 17 delivery",
            "telegram_chat_id": "-1003589561528",
            "telegram_thread_id": "17",
            "work_id": "work-current",
            "ledger_run_id": "run-current",
            "task_started_at": "2026-07-18T19:39:36Z",
            "started_at": "2026-07-18T19:39:40Z",
        }
    }}))
    content = "Model: openai-codex/gpt-5.6-sol | Route: current run | Why: verified\nComplete: Yes"
    with patch.object(telegram_adapter._Path, "home", return_value=tmp_path):
        command = _adapter()._jaimes_post_final_card_command(
            "-1003589561528", content, "17", {"notify": True}, "3914"
        )
        missing = _adapter()._jaimes_post_final_card_command(
            "-1003589561528", content, "17", {"notify": True}, None
        )
    assert command is not None
    assert command[2] == "done"
    assert command[command.index("--final-message-id") + 1] == "3914"
    assert command[command.index("--final-delivery-verified-by") + 1] == "hermes-adapter-success"
    assert command[command.index("--work-id") + 1] == "work-current"
    assert command[command.index("--run-id") + 1] == "run-current"
    assert missing is None


def test_jaimes_final_send_canonicalizes_then_closes_only_after_delivery():
    source = inspect.getsource(TelegramAdapter.send)
    canonical_at = source.index("await self._jaimes_canonical_final_before_send")
    ready_at = source.index("await self._jaimes_finalize_card_before_final")
    rich_complete_at = source.index("await self._jaimes_complete_card_after_final")
    rich_return_at = source.index("return rich_result", rich_complete_at)
    legacy_complete_at = source.rindex("await self._jaimes_complete_card_after_final")
    legacy_return_at = source.index("return SendResult(", legacy_complete_at)
    assert canonical_at < ready_at < rich_complete_at < rich_return_at
    assert legacy_complete_at < legacy_return_at
    assert "Final/direct sends may not have an active gateway refresh loop" not in source
    assert "jaimes_reply_markup = None" in source


def test_jaimes_final_formatter_receives_private_payload_over_stdin(tmp_path):
    state_path = tmp_path / ".openclaw" / "telegram" / "jaimes_fast_ack_state.json"
    script = tmp_path / ".openclaw" / "workspace" / "mission-control" / "scripts" / "jaimes_telegram_fast_ack.py"
    state_path.parent.mkdir(parents=True)
    script.parent.mkdir(parents=True)
    state_path.write_text(json.dumps({"active_cards": {
        "current": {
            "status": "active",
            "key": "turn-123",
            "ack_message_id": "44",
            "objective": "Assess Agent RH safely",
            "model": "openai-codex/gpt-5.6-sol",
            "route": "JAIMES verified execution",
            "telegram_chat_id": "-1003589561528",
            "telegram_thread_id": "17",
            "work_id": "work-current",
            "ledger_run_id": "run-current",
            "task_started_at": "2026-07-12T23:21:59Z",
            "started_at": "2026-07-12T23:22:00Z",
        }
    }}))
    script.write_text(
        "import json,sys\n"
        "payload=json.load(sys.stdin)\n"
        "assert payload['objective']=='Assess Agent RH safely'\n"
        "assert payload['work_id']=='work-current'\n"
        "assert payload['run_id']=='run-current'\n"
        "assert payload['task_started_at']=='2026-07-12T23:21:59Z'\n"
        "print('<pre>canonical final</pre>')\n"
    )
    adapter = _adapter()
    with patch.object(telegram_adapter._Path, "home", return_value=tmp_path):
        rendered = asyncio.run(adapter._jaimes_canonical_final_before_send(
            "-1003589561528",
            "private findings",
            "17",
            {"notify": True},
        ))
    assert rendered == "```\ncanonical final\n```"


def test_gateway_stops_typing_before_final_delivery():
    source = inspect.getsource(BasePlatformAdapter._process_message_background)
    response_branch = source.index("if response:")
    stop_at = source.index("await _stop_typing_task()", response_branch)
    send_at = source.index("result = await self._send_with_retry", response_branch)
    assert stop_at < send_at


def test_active_card_ignores_blank_zombie_and_normalizes_chat_prefix(tmp_path):
    state_path = tmp_path / ".openclaw" / "telegram" / "jaimes_fast_ack_state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(json.dumps({"active_cards": {
        "zombie": {
            "status": "active",
            "key": "zombie",
            "ack_message_id": "",
            "objective": "",
            "telegram_chat_id": "-1001",
            "telegram_thread_id": "17",
            "started_at": "2099-01-01T00:00:00Z",
        },
        "bound": {
            "status": "active",
            "key": "bound",
            "ack_message_id": "44",
            "objective": "Verify exact final delivery",
            "telegram_chat_id": "telegram:-1001",
            "telegram_thread_id": "17",
            "started_at": "2026-07-18T20:00:00Z",
        },
    }}), encoding="utf-8")

    with patch.object(telegram_adapter._Path, "home", return_value=tmp_path):
        card = _adapter()._jaimes_topic17_active_card(
            "-1001", "17", {"notify": True}
        )
    assert card and card["key"] == "bound"


def test_unbound_final_preserves_verified_why_in_private_formatter_payload(tmp_path):
    script = (
        tmp_path / ".openclaw" / "workspace" / "mission-control" / "scripts"
        / "jaimes_telegram_fast_ack.py"
    )
    script.parent.mkdir(parents=True)
    script.write_text(
        "import json,sys\n"
        "payload=json.load(sys.stdin)\n"
        "assert payload['model']=='openai-codex/gpt-5.6-sol'\n"
        "assert payload['route']=='JAIMES verified execution'\n"
        "assert payload['why']=='300/300 checks passed'\n"
        "print('<pre>canonical final</pre>')\n",
        encoding="utf-8",
    )
    content = (
        "Model: openai-codex/gpt-5.6-sol | Route: JAIMES verified execution "
        "| Why: 300/300 checks passed\nComplete: Yes"
    )

    with patch.object(telegram_adapter._Path, "home", return_value=tmp_path):
        rendered = asyncio.run(_adapter()._jaimes_canonical_final_before_send(
            "-1001", content, "17", {"notify": True}
        ))
    assert rendered == "```\ncanonical final\n```"


def test_unbound_final_only_uses_literal_objective_label(tmp_path):
    script = (
        tmp_path / ".openclaw" / "workspace" / "mission-control" / "scripts"
        / "jaimes_telegram_fast_ack.py"
    )
    script.parent.mkdir(parents=True)
    script.write_text(
        "import json,sys\n"
        "payload=json.load(sys.stdin)\n"
        "expected=('Assess the current agent workflow' "
        "if '\\nObjective: Assess the current agent workflow' in payload['text'] "
        "else 'Complete the current Telegram task')\n"
        "assert payload['objective']==expected\n"
        "assert payload['objective'] not in {'Yes','No'}\n"
        "print('<pre>canonical final</pre>')\n",
        encoding="utf-8",
    )
    adapter = _adapter()
    contents = (
        "Model: test | Route: test | Why: verified\nObjective Complete: Yes",
        "Model: test | Route: test | Why: verified\nObjective Complete: No",
        (
            "Model: test | Route: test | Why: verified\n"
            "Objective: Assess the current agent workflow"
        ),
    )

    with patch.object(telegram_adapter._Path, "home", return_value=tmp_path):
        rendered = [
            asyncio.run(adapter._jaimes_canonical_final_before_send(
                "-1001", content, "17", {"notify": True}
            ))
            for content in contents
        ]
    assert rendered == ["```\ncanonical final\n```"] * 3


def test_post_delivery_command_persists_exact_adapter_message_id(tmp_path):
    state_path = tmp_path / ".openclaw" / "telegram" / "jaimes_fast_ack_state.json"
    script = (
        tmp_path / ".openclaw" / "workspace" / "mission-control" / "scripts"
        / "jaimes_work_card.py"
    )
    state_path.parent.mkdir(parents=True)
    script.parent.mkdir(parents=True)
    script.write_text("# test\n", encoding="utf-8")
    state_path.write_text(json.dumps({"active_cards": {
        "bound": {
            "status": "active",
            "key": "bound",
            "ack_message_id": "44",
            "objective": "Verify exact final delivery",
            "telegram_chat_id": "-1001",
            "telegram_thread_id": "17",
            "started_at": "2026-07-18T20:00:00Z",
        }
    }}), encoding="utf-8")
    content = (
        "Model: openai-codex/gpt-5.6-sol | Route: JAIMES verified execution "
        "| Why: verified\nComplete: Yes"
    )

    with patch.object(telegram_adapter._Path, "home", return_value=tmp_path):
        command = _adapter()._jaimes_post_final_card_command(
            "-1001", content, "17", {"notify": True}, "3914"
        )
    assert command is not None
    assert command[2] == "done"
    assert command[command.index("--final-message-id") + 1] == "3914"
    assert command[
        command.index("--final-delivery-verified-by") + 1
    ] == "hermes-adapter-success"
    assert "--blocker" not in command
    assert "--next" not in command


def test_topic17_final_skips_rich_and_closes_with_exact_ptb_message_id():
    adapter = _adapter()

    class FakeBot:
        async def send_message(self, **_kwargs):
            return SimpleNamespace(message_id=777)

    adapter._bot = FakeBot()
    adapter.platform = type("Platform", (), {"value": "telegram"})()
    adapter._send_path_degraded = False
    adapter._reply_to_mode = "off"
    adapter._metadata_thread_id = lambda _metadata: "17"
    adapter._metadata_reply_to_message_id = lambda _metadata: None
    adapter._message_thread_id_for_send = lambda thread_id: thread_id
    adapter._is_private_dm_topic_send = lambda *_args, **_kwargs: False
    adapter._should_thread_reply = lambda *_args, **_kwargs: False
    adapter._thread_kwargs_for_send = lambda *_args, **_kwargs: {
        "message_thread_id": 17
    }
    adapter._link_preview_kwargs = lambda: {}
    adapter._notification_kwargs = lambda _metadata: {}
    adapter._should_attempt_rich = lambda *_args, **_kwargs: True
    adapter._try_send_rich = AsyncMock(
        side_effect=AssertionError("Topic 17 final must not use rich delivery")
    )
    adapter._jaimes_canonical_final_before_send = AsyncMock(
        return_value="```\ncanonical final\n```"
    )
    adapter._jaimes_finalize_card_before_final = AsyncMock()
    adapter._jaimes_complete_card_after_final = AsyncMock()
    adapter.format_message = lambda content: content
    adapter.truncate_message = lambda content, *_args, **_kwargs: [content]

    result = asyncio.run(adapter.send(
        "-1001",
        "Model: test | Route: test | Why: test\nComplete: Yes",
        metadata={"notify": True, "message_thread_id": "17"},
    ))

    assert result.success is True
    assert result.message_id == "777"
    adapter._try_send_rich.assert_not_awaited()
    adapter._jaimes_complete_card_after_final.assert_awaited_once()
    assert adapter._jaimes_complete_card_after_final.await_args.args[-1] == "777"


def test_topic17_contract_does_not_add_a_task_header():
    source = Path(telegram_adapter.__file__).read_text(encoding="utf-8")
    assert "--header-message-id" not in source
    assert "JAIMES_TELEGRAM_TASK_HEADERS" not in source
