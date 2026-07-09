"""Telegram group-Q&A boundary tests.

These tests exercise the adapter boundary only: group commands must be
answered by neo.group_qa and never enter the normal Gateway handler.
"""
from __future__ import annotations

import asyncio
import builtins
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from gateway.config import Platform, PlatformConfig


def _adapter():
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="fake", extra={})
    adapter._bot = SimpleNamespace(id=999, username="hermes_bot")
    adapter._reply_to_mode = "first"
    adapter._message_handler = AsyncMock()
    adapter._is_user_authorized_from_message = lambda _message: True
    adapter.handle_message = AsyncMock()
    return adapter


def _root(tmp_path):
    day = tmp_path / "data/days/2026/2026-07-10.json"
    day.parent.mkdir(parents=True)
    (tmp_path / "data/indexes").mkdir(parents=True)
    day.write_text(json.dumps({"date": "2026-07-10", "wake_at": "2026-07-10T08:00:00+09:00", "sleep_at": None, "meals": [], "outings": [], "work_sessions": [], "todolist": []}), encoding="utf-8")
    (tmp_path / "data/indexes/current.json").write_text(json.dumps({"current_day": {"path": "data/days/2026/2026-07-10.json"}}), encoding="utf-8")
    return tmp_path


def _message(text, chat_id=-100, reply_to_bot=False):
    message = SimpleNamespace(
        text=text,
        chat=SimpleNamespace(id=chat_id, type="supergroup", is_forum=False, title=None),
        from_user=SimpleNamespace(id=111, full_name="Test"),
        message_id=123,
        date=None,
        reply_text=AsyncMock(),
        reply_to_message=SimpleNamespace(from_user=SimpleNamespace(id=999)) if reply_to_bot else None,
    )
    return SimpleNamespace(update_id=1, message=message, effective_message=None), message


def test_enabled_group_qa_answers_without_gateway_dispatch(monkeypatch, tmp_path):
    monkeypatch.setenv("TELEGRAM_GROUP_QA_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_ALLOWED_GROUP_CHATS", "-100")
    monkeypatch.setenv("NEO_MARIERIE_ROOT", str(_root(tmp_path)))
    adapter = _adapter()
    update, message = _message("/ask@hermes_bot 마리 오늘 일어났어?")

    asyncio.run(adapter._handle_command(update, SimpleNamespace()))

    message.reply_text.assert_awaited_once()
    assert "일어난 기록" in message.reply_text.await_args.args[0]
    adapter.handle_message.assert_not_awaited()


def test_disabled_or_mutation_group_command_is_silent(monkeypatch, tmp_path):
    monkeypatch.setenv("TELEGRAM_GROUP_QA_ENABLED", "false")
    monkeypatch.setenv("TELEGRAM_ALLOWED_GROUP_CHATS", "-100")
    monkeypatch.setenv("NEO_MARIERIE_ROOT", str(_root(tmp_path)))
    adapter = _adapter()
    update, message = _message("/question 오늘 잤어?")
    asyncio.run(adapter._handle_command(update, SimpleNamespace()))
    message.reply_text.assert_not_awaited()
    adapter.handle_message.assert_not_awaited()

    monkeypatch.setenv("TELEGRAM_GROUP_QA_ENABLED", "true")
    update, message = _message("/reset")
    asyncio.run(adapter._handle_command(update, SimpleNamespace()))
    message.reply_text.assert_not_awaited()
    adapter.handle_message.assert_not_awaited()


def test_sensitive_group_question_is_refused(monkeypatch, tmp_path):
    monkeypatch.setenv("TELEGRAM_GROUP_QA_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_ALLOWED_GROUP_CHATS", "-100")
    monkeypatch.setenv("NEO_MARIERIE_ROOT", str(_root(tmp_path)))
    adapter = _adapter()
    update, message = _message("/status private spark 기록 있어?")
    asyncio.run(adapter._handle_command(update, SimpleNamespace()))
    assert "말할 수 없어" in message.reply_text.await_args.args[0]
    adapter.handle_message.assert_not_awaited()


def test_group_mention_and_reply_use_qa_without_gateway(monkeypatch, tmp_path):
    monkeypatch.setenv("TELEGRAM_GROUP_QA_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_ALLOWED_GROUP_CHATS", "-100")
    monkeypatch.setenv("NEO_MARIERIE_ROOT", str(_root(tmp_path)))
    adapter = _adapter()
    update, message = _message("@hermes_bot 마리 오늘 일어났어?")
    asyncio.run(adapter._handle_text_message(update, SimpleNamespace()))
    assert "일어난 기록" in message.reply_text.await_args.args[0]
    adapter.handle_message.assert_not_awaited()

    update, message = _message("오늘 밥 먹었어?", reply_to_bot=True)
    asyncio.run(adapter._handle_text_message(update, SimpleNamespace()))
    message.reply_text.assert_awaited_once()
    adapter.handle_message.assert_not_awaited()


def test_group_menu_is_empty_while_private_commands_remain_out_of_scope():
    adapter = _adapter()
    commands = adapter._group_qa_menu_commands(lambda name, description: SimpleNamespace(command=name))
    assert commands == []


def test_trusted_mention_uses_isolated_read_only_model_route(monkeypatch, tmp_path):
    monkeypatch.setenv("TELEGRAM_GROUP_QA_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_GROUP_QA_MODEL_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_GROUP_QA_DETAIL_LEVEL", "trusted")
    monkeypatch.setenv("TELEGRAM_ALLOWED_GROUP_CHATS", "-100")
    monkeypatch.setenv("NEO_MARIERIE_ROOT", str(_root(tmp_path)))
    adapter = _adapter()
    update, _message_obj = _message("@hermes_bot 오늘 밥 뭐 먹었어?")
    asyncio.run(adapter._handle_text_message(update, SimpleNamespace()))
    event = adapter.handle_message.await_args.args[0]
    assert event.metadata["trusted_group_model_route"] is True
    assert event.source.read_only_route is True
    assert event.source.session_key_override == "telegram:trusted_group:-100"
    assert event.source.session_id_override.startswith("telegram:trusted_group:-100:")


def test_private_spark_is_refused_before_model_dispatch(monkeypatch, tmp_path):
    monkeypatch.setenv("TELEGRAM_GROUP_QA_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_GROUP_QA_MODEL_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_GROUP_QA_DETAIL_LEVEL", "trusted")
    monkeypatch.setenv("TELEGRAM_ALLOWED_GROUP_CHATS", "-100")
    monkeypatch.setenv("NEO_MARIERIE_ROOT", str(_root(tmp_path)))
    adapter = _adapter()
    update, message = _message("@hermes_bot spark 기록 있어?")
    asyncio.run(adapter._handle_text_message(update, SimpleNamespace()))
    assert "여기선 말 못 해" in message.reply_text.await_args.args[0]
    adapter.handle_message.assert_not_awaited()


def test_missing_group_qa_core_fails_closed(monkeypatch):
    monkeypatch.setenv("TELEGRAM_GROUP_QA_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_ALLOWED_GROUP_CHATS", "-100")
    real_import = builtins.__import__

    def no_group_qa(name, *args, **kwargs):
        if name == "neo.group_qa":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_group_qa)
    adapter = _adapter()
    update, message = _message("/ask 오늘 일어났어?")
    asyncio.run(adapter._handle_command(update, SimpleNamespace()))
    message.reply_text.assert_not_awaited()
    adapter.handle_message.assert_not_awaited()


def test_group_brief_prefers_rich_message_and_falls_back(monkeypatch, tmp_path):
    monkeypatch.setenv("TELEGRAM_GROUP_QA_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_ALLOWED_GROUP_CHATS", "-100")
    monkeypatch.setenv("TELEGRAM_GROUP_QA_DETAIL_LEVEL", "trusted")
    monkeypatch.setenv("NEO_MARIERIE_ROOT", str(_root(tmp_path)))
    adapter = _adapter()
    adapter._bot.do_api_request = AsyncMock(return_value={"result": {"message_id": 456}})
    update, message = _message("/brief@hermes_bot")

    asyncio.run(adapter._handle_command(update, SimpleNamespace()))

    adapter._bot.do_api_request.assert_awaited_once()
    endpoint, = adapter._bot.do_api_request.await_args.args
    assert endpoint == "sendRichMessage"
    payload = adapter._bot.do_api_request.await_args.kwargs["api_kwargs"]
    assert "# 🌤 오늘 브리프" in payload["rich_message"]["markdown"]
    assert "## 기상" in payload["rich_message"]["markdown"]
    message.reply_text.assert_not_awaited()
    adapter.handle_message.assert_not_awaited()

    adapter = _adapter()
    class BadRequest(Exception):
        pass
    adapter._bot.do_api_request = AsyncMock(side_effect=BadRequest("bad request"))
    update, message = _message("/brief")
    asyncio.run(adapter._handle_command(update, SimpleNamespace()))
    message.reply_text.assert_awaited_once()
    assert "🌤 오늘 브리프" in message.reply_text.await_args.args[0]


def test_trusted_group_model_override_applied_and_cleaned(monkeypatch, tmp_path):
    monkeypatch.setenv("TELEGRAM_GROUP_QA_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_GROUP_QA_MODEL_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_GROUP_QA_DETAIL_LEVEL", "trusted")
    monkeypatch.setenv("TELEGRAM_ALLOWED_GROUP_CHATS", "-100")
    monkeypatch.setenv("TELEGRAM_GROUP_QA_MODEL", "gpt-5.6-luna")
    monkeypatch.setenv("TELEGRAM_GROUP_QA_REASONING_EFFORT", "low")
    monkeypatch.setenv("NEO_MARIERIE_ROOT", str(_root(tmp_path)))
    adapter = _adapter()

    class FakeRunner:
        _session_model_overrides = {}
        _session_reasoning_overrides = {}

        def _set_session_reasoning_override(self, key, val):
            if val is None:
                self._session_reasoning_overrides.pop(key, None)
            else:
                self._session_reasoning_overrides[key] = val

    runner = FakeRunner()
    adapter._message_handler.__self__ = runner
    seen = {}

    async def capture_dispatch(_event):
        seen["model"] = runner._session_model_overrides.copy()
        seen["reasoning"] = runner._session_reasoning_overrides.copy()

    adapter.handle_message.side_effect = capture_dispatch
    update, _ = _message("@hermes_bot test question?")
    asyncio.run(adapter._handle_text_message(update, SimpleNamespace()))
    session_key = "telegram:trusted_group:-100"
    assert seen["model"] == {session_key: {"model": "gpt-5.6-luna"}}
    assert session_key in seen["reasoning"]
    assert session_key not in runner._session_model_overrides
    assert session_key not in runner._session_reasoning_overrides
    adapter.handle_message.assert_awaited_once()


def test_no_model_override_when_env_unset(monkeypatch, tmp_path):
    monkeypatch.setenv("TELEGRAM_GROUP_QA_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_GROUP_QA_MODEL_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_GROUP_QA_DETAIL_LEVEL", "trusted")
    monkeypatch.setenv("TELEGRAM_ALLOWED_GROUP_CHATS", "-100")
    monkeypatch.delenv("TELEGRAM_GROUP_QA_MODEL", raising=False)
    monkeypatch.delenv("TELEGRAM_GROUP_QA_REASONING_EFFORT", raising=False)
    monkeypatch.setenv("NEO_MARIERIE_ROOT", str(_root(tmp_path)))
    adapter = _adapter()

    class FakeRunner:
        _session_model_overrides = {}
        _session_reasoning_overrides = {}

        def _set_session_reasoning_override(self, key, val):
            if val is None:
                self._session_reasoning_overrides.pop(key, None)
            else:
                self._session_reasoning_overrides[key] = val

    runner = FakeRunner()
    adapter._message_handler.__self__ = runner
    update, _ = _message("@hermes_bot test question?")
    asyncio.run(adapter._handle_text_message(update, SimpleNamespace()))
    assert len(runner._session_model_overrides) == 0
    assert len(runner._session_reasoning_overrides) == 0
    adapter.handle_message.assert_awaited_once()
