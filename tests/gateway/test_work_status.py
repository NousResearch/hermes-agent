import asyncio

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, ProcessingOutcome, SendResult
from gateway.session import SessionSource
from gateway.work_status import (
    WorkStatusConfig,
    WorkStatusHandle,
    fallback_status_text,
    infer_status_mode,
    interpret_status_request,
    maybe_ai_status_text,
    resolve_work_status_config,
)


class DummyAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True), Platform.TELEGRAM)
        self.sent = []
        self.pinned = []
        self.unpinned = []
        self.deleted = []
        self.edited = []
        self.typed = []
        self._next_id = 100

    async def connect(self):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self._next_id += 1
        self.sent.append((chat_id, content, metadata or {}))
        return SendResult(success=True, message_id=str(self._next_id))

    async def send_typing(self, chat_id, metadata=None):
        self.typed.append((chat_id, metadata))

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "group"}

    async def pin_message(self, chat_id, message_id, *, metadata=None):
        self.pinned.append((chat_id, message_id, metadata))
        return True

    async def unpin_message(self, chat_id, message_id, *, metadata=None):
        self.unpinned.append((chat_id, message_id, metadata))
        return True

    async def delete_message(self, chat_id, message_id):
        self.deleted.append((chat_id, message_id))
        return True

    async def edit_message(self, chat_id, message_id, content, *, finalize=False):
        self.edited.append((chat_id, message_id, content, finalize))
        return SendResult(success=True, message_id=message_id)


def event(text="please build the modular pinned summary", *, chat_id="42"):
    return MessageEvent(
        text=text,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id=chat_id, user_id="u1"),
    )


def test_work_status_config_supports_structured_and_legacy_keys():
    structured = {
        "display": {
            "work_status": {
                "enabled": True,
                "delay_seconds": 0,
                "cleanup": {"on_success": "delete", "delete_delay_seconds": 0},
                "chats": {"allow": ["42"], "deny": ["99"]},
                "platforms": {"telegram": {"mode": "pinned"}},
            }
        }
    }
    cfg = resolve_work_status_config(structured, "telegram", chat_id="42")
    assert cfg.enabled is True
    assert cfg.mode == "pinned"
    assert cfg.delay_seconds == 0
    assert cfg.cleanup_on_success == "delete"

    assert resolve_work_status_config(structured, "telegram", chat_id="99").enabled is False
    assert resolve_work_status_config(structured, "telegram", chat_id="7").enabled is False

    legacy = {"display": {"pinned_work_summary": True, "pinned_work_summary_delay_seconds": 3}}
    cfg = resolve_work_status_config(legacy, "telegram", chat_id="42")
    assert cfg.enabled is True
    assert cfg.delay_seconds == 3


def test_fallback_status_uses_reply_context_for_short_followups():
    ev = event("do this")
    ev.reply_to_text = "Investigate the busy Telegram keyboard and make it modular"
    text = fallback_status_text(ev)
    assert text.startswith("📌 [Research] From reply:")
    assert "busy Telegram keyboard" in text


def test_fallback_status_for_short_message_is_not_prefixed_with_work_on():
    assert fallback_status_text(event("Test")) == "📌 [Verify] Run tests"


def test_status_mode_replaces_working_prefix_with_request_interpretation():
    text = fallback_status_text(event("Our pinned message is dumb. Replace Working with the mode and summarize the request interpretation."))
    assert text == "📌 [Debug] Investigate pinned message code"
    assert "Working:" not in text
    assert infer_status_mode("Can you explain what broke?") == "Ask"
    assert interpret_status_request("How did the implementation go", "Ask") == "Summarize implementation outcome"
    assert interpret_status_request("What's left?", "Ask") == "Summarize remaining work"
    assert interpret_status_request("Whats left", "Ask") == "Summarize remaining work"
    assert interpret_status_request("What is left?", "Ask") == "Summarize remaining work"
    assert infer_status_mode("Do we need to restart") == "Ask"
    assert interpret_status_request("Do we need to restart", "Ask") == "Assess restart requirement"
    assert fallback_status_text(event("Itsnot")) == "📌 [Debug] Investigate reported issue"
    assert infer_status_mode("Plan the migration") == "Plan"
    assert interpret_status_request("Our pinned message is dumb", "Debug") == "Investigate pinned message code"
    assert fallback_status_text(event("It is not working")) == "📌 [Debug] Investigate reported issue"


@pytest.mark.asyncio
async def test_ai_status_does_not_echo_ask_question_with_answer_prefix(monkeypatch):
    async def fake_call_llm(*args, **kwargs):
        return {"choices": [{"message": {"content": "Answer: How did the implementation go"}}]}

    def fake_extract(response):
        return response["choices"][0]["message"]["content"]

    monkeypatch.setattr("agent.auxiliary_client.async_call_llm", fake_call_llm)
    monkeypatch.setattr("agent.auxiliary_client.extract_content_or_reasoning", fake_extract)

    text = await maybe_ai_status_text(WorkStatusConfig(ai_summary=True), event("How did the implementation go"))
    assert text == "📌 [Ask] Summarize implementation outcome"


@pytest.mark.asyncio
async def test_start_and_finish_work_status_pins_then_edits_unpins_and_deletes(monkeypatch):
    adapter = DummyAdapter()
    cfg_dict = {
        "display": {
            "work_status": {
                "enabled": True,
                "mode": "pinned",
                "delay_seconds": 0,
                "cleanup": {"on_success": "edit_done_then_delete", "delete_delay_seconds": 0},
            }
        }
    }
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg_dict)
    gen = {"value": 5}

    handle = await adapter._start_work_status(event(), None, "session-1", lambda: gen["value"])
    assert handle is not None
    assert handle.pinned is True
    assert adapter.sent[0][1].startswith("📌 [Build]")
    assert adapter.pinned == [("42", handle.message_id, {"notify": False})]

    await adapter._finish_work_status(
        handle,
        adapter._resolve_work_status_config(event()),
        ProcessingOutcome.SUCCESS,
        lambda: gen["value"],
    )
    assert adapter.edited == [("42", handle.message_id, "✅ Done", True)]
    assert adapter.unpinned[-1][0:2] == ("42", handle.message_id)
    assert adapter.deleted == [("42", handle.message_id)]


@pytest.mark.asyncio
async def test_persistent_pinned_status_stays_until_description_changes(monkeypatch):
    adapter = DummyAdapter()
    cfg_dict = {
        "display": {
            "work_status": {
                "enabled": True,
                "mode": "pinned",
                "delay_seconds": 0,
                "cleanup": {"on_success": "keep", "delete_delay_seconds": 0},
            }
        }
    }
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg_dict)
    gen = {"value": 1}

    first = await adapter._start_work_status(event("summarize pinned work status"), None, "session-1", lambda: gen["value"])
    assert first is not None
    await adapter._finish_work_status(
        first,
        adapter._resolve_work_status_config(event()),
        ProcessingOutcome.SUCCESS,
        lambda: gen["value"],
    )
    assert len(adapter.sent) == 1
    assert adapter.pinned == [("42", first.message_id, {"notify": False})]
    assert adapter.unpinned == []
    assert adapter.deleted == []

    gen["value"] = 2
    same = await adapter._start_work_status(event("summarize pinned work status"), None, "session-2", lambda: gen["value"])
    assert same is first
    assert len(adapter.sent) == 1
    assert adapter.edited == []

    gen["value"] = 3
    changed = await adapter._start_work_status(event("review dirty repository cleanup"), None, "session-3", lambda: gen["value"])
    assert changed is first
    assert len(adapter.sent) == 1
    assert adapter.edited[-1][0:2] == ("42", first.message_id)
    assert "dirty repository cleanup" in adapter.edited[-1][2]
    assert adapter.edited[-1][3] is False


@pytest.mark.asyncio
async def test_finish_work_status_skips_stale_generation():
    adapter = DummyAdapter()
    handle = WorkStatusHandle(
        status_id="s1",
        platform="telegram",
        chat_id="42",
        thread_id=None,
        session_key="session-1",
        run_generation=1,
        message_id="101",
        mode="pinned",
        pinned=True,
    )
    await adapter._finish_work_status(
        handle,
        WorkStatusConfig(cleanup_on_success="delete", delete_delay_seconds=0),
        ProcessingOutcome.SUCCESS,
        lambda: 2,
    )
    assert adapter.unpinned == []
    assert adapter.deleted == []


@pytest.mark.asyncio
async def test_short_turn_cancels_delayed_status_before_sending(monkeypatch):
    adapter = DummyAdapter()
    cfg_dict = {"display": {"work_status": {"enabled": True, "delay_seconds": 60}}}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg_dict)

    task = asyncio.create_task(adapter._start_work_status(event(), None, "session-1", lambda: 1))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert adapter.sent == []
