"""DEAD path: not imported by gateway/run.py — contract-only unit tests.

See gateway/RUNTIME_SERVICES.md. Marked dead_runtime_service so suites can
optionally filter with ``-m "not dead_runtime_service"``; default still runs.
"""
import pytest

pytestmark = pytest.mark.dead_runtime_service

from unittest.mock import MagicMock, call

from gateway.transcript_persistence_runtime_service import (
    build_gateway_session_meta_entry,
    did_gateway_agent_fail_early,
    did_gateway_turn_send_visible_reply,
    extract_new_gateway_transcript_messages,
    persist_gateway_agent_transcript,
)


def test_did_gateway_agent_fail_early_requires_failed_without_final_response():
    assert did_gateway_agent_fail_early({"failed": True, "final_response": ""}) is True
    assert did_gateway_agent_fail_early({"failed": True, "final_response": "ok"}) is False
    assert did_gateway_agent_fail_early({"failed": False, "final_response": ""}) is False


def test_did_gateway_turn_send_visible_reply_requires_real_visible_response():
    assert (
        did_gateway_turn_send_visible_reply(
            agent_result={"response_state": "sent", "suppress_reply": False},
            visible_final_response="正常回复",
        )
        is True
    )
    assert (
        did_gateway_turn_send_visible_reply(
            agent_result={"response_state": "qq_explicit_fallback", "synthetic_fallback": True},
            visible_final_response="我在，你继续说。",
        )
        is False
    )
    assert (
        did_gateway_turn_send_visible_reply(
            agent_result={"response_state": "suppressed_no_reply", "suppress_reply": True},
            visible_final_response="",
        )
        is False
    )


def test_extract_new_gateway_transcript_messages_uses_history_offset_and_sync():
    seen = {}

    def _sync(messages, *, raw_final_response, visible_final_response):
        seen["messages"] = messages
        seen["raw"] = raw_final_response
        seen["visible"] = visible_final_response
        return messages + [{"role": "assistant", "content": visible_final_response}]

    result = extract_new_gateway_transcript_messages(
        agent_result={"history_offset": 2, "final_response": "raw"},
        agent_messages=[
            {"role": "user", "content": "old-1"},
            {"role": "assistant", "content": "old-2"},
            {"role": "user", "content": "new-1"},
        ],
        history_len=5,
        visible_final_response="visible",
        sync_visible_final_response=_sync,
    )

    assert seen["messages"] == [{"role": "user", "content": "new-1"}]
    assert seen["raw"] == "raw"
    assert seen["visible"] == "visible"
    assert result[-1] == {"role": "assistant", "content": "visible"}


def test_build_gateway_session_meta_entry_defaults_empty_tools():
    entry = build_gateway_session_meta_entry(
        tool_defs=None,
        model="gpt-test",
        platform="qq",
        timestamp="2026-04-18T00:00:00",
    )

    assert entry == {
        "role": "session_meta",
        "tools": [],
        "model": "gpt-test",
        "platform": "qq",
        "timestamp": "2026-04-18T00:00:00",
    }


def test_persist_gateway_agent_transcript_skips_failed_early_transcript_growth():
    session_store = MagicMock()
    logger = MagicMock()

    result = persist_gateway_agent_transcript(
        session_store=session_store,
        session_id="sess-1",
        session_key="key-1",
        platform="qq",
        history=[{"role": "user", "content": "older"}],
        agent_result={
            "failed": True,
            "final_response": "",
            "last_prompt_tokens": 12,
        },
        agent_messages=[],
        message_text="hello",
        visible_final_response="",
        resolve_gateway_model=lambda: "gpt-test",
        sync_visible_final_response=lambda messages, **_: messages,
        session_db_present=True,
        logger=logger,
        timestamp="2026-04-18T01:00:00",
    )

    assert result.agent_failed_early is True
    assert result.persisted_messages == 0
    session_store.append_to_transcript.assert_not_called()
    session_store.update_session.assert_called_once_with(
        "key-1",
        last_prompt_tokens=12,
        mark_visible_reply=False,
    )
    logger.info.assert_called_once()


def test_persist_gateway_agent_transcript_writes_session_meta_and_backup_entries():
    session_store = MagicMock()

    result = persist_gateway_agent_transcript(
        session_store=session_store,
        session_id="sess-1",
        session_key="key-1",
        platform="qq",
        history=[],
        agent_result={
            "tools": [{"name": "terminal"}],
            "history_offset": 0,
            "final_response": "visible",
            "last_prompt_tokens": 42,
        },
        agent_messages=[
            {"role": "system", "content": "skip"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "visible"},
        ],
        message_text="hello",
        visible_final_response="visible",
        resolve_gateway_model=lambda: "gpt-test",
        sync_visible_final_response=lambda messages, **_: messages,
        session_db_present=True,
        timestamp="2026-04-18T02:00:00",
    )

    assert result.wrote_session_meta is True
    assert result.used_fallback_transcript is False
    assert result.persisted_messages == 2
    assert session_store.append_to_transcript.call_args_list == [
        call(
            "sess-1",
            {
                "role": "session_meta",
                "tools": [{"name": "terminal"}],
                "model": "gpt-test",
                "platform": "qq",
                "timestamp": "2026-04-18T02:00:00",
            },
        ),
        call(
            "sess-1",
            {
                "role": "user",
                "content": "hello",
                "timestamp": "2026-04-18T02:00:00",
            },
            skip_db=True,
        ),
        call(
            "sess-1",
            {
                "role": "assistant",
                "content": "visible",
                "timestamp": "2026-04-18T02:00:00",
            },
            skip_db=True,
        ),
    ]
    session_store.update_session.assert_called_once_with(
        "key-1",
        last_prompt_tokens=42,
        mark_visible_reply=True,
    )


def test_persist_gateway_agent_transcript_falls_back_to_simple_user_assistant_entries():
    session_store = MagicMock()

    result = persist_gateway_agent_transcript(
        session_store=session_store,
        session_id="sess-1",
        session_key="key-1",
        platform="qq",
        history=[{"role": "user", "content": "older"}],
        agent_result={
            "history_offset": 2,
            "final_response": "visible",
            "last_prompt_tokens": 7,
        },
        agent_messages=[
            {"role": "user", "content": "older"},
            {"role": "assistant", "content": "older-reply"},
        ],
        message_text="new message",
        visible_final_response="visible",
        resolve_gateway_model=lambda: "gpt-test",
        sync_visible_final_response=lambda messages, **_: messages,
        session_db_present=False,
        timestamp="2026-04-18T03:00:00",
    )

    assert result.used_fallback_transcript is True
    assert result.persisted_messages == 2
    assert session_store.append_to_transcript.call_args_list == [
        call(
            "sess-1",
            {
                "role": "user",
                "content": "new message",
                "timestamp": "2026-04-18T03:00:00",
            },
        ),
        call(
            "sess-1",
            {
                "role": "assistant",
                "content": "visible",
                "timestamp": "2026-04-18T03:00:00",
            },
        ),
    ]
    session_store.update_session.assert_called_once_with(
        "key-1",
        last_prompt_tokens=7,
        mark_visible_reply=True,
    )


def test_persist_gateway_agent_transcript_does_not_mark_synthetic_fallback_as_visible_reply():
    session_store = MagicMock()

    persist_gateway_agent_transcript(
        session_store=session_store,
        session_id="sess-1",
        session_key="key-1",
        platform="qq",
        history=[{"role": "user", "content": "older"}],
        agent_result={
            "history_offset": 2,
            "final_response": "我在，你继续说。",
            "last_prompt_tokens": 9,
            "response_state": "qq_explicit_fallback",
            "synthetic_fallback": True,
        },
        agent_messages=[
            {"role": "user", "content": "older"},
            {"role": "assistant", "content": "older-reply"},
        ],
        message_text="new message",
        visible_final_response="我在，你继续说。",
        resolve_gateway_model=lambda: "gpt-test",
        sync_visible_final_response=lambda messages, **_: messages,
        session_db_present=False,
        timestamp="2026-04-18T03:30:00",
    )

    session_store.update_session.assert_called_once_with(
        "key-1",
        last_prompt_tokens=9,
        mark_visible_reply=False,
    )
