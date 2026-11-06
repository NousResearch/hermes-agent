"""Tests for task-local bound routing identity (never environment-derived)."""
import asyncio
import os

import pytest

import gateway.session_context as sc

FIELDS = {
    "platform": sc._SESSION_PLATFORM,
    "chat_id": sc._SESSION_CHAT_ID,
    "thread_id": sc._SESSION_THREAD_ID,
    "message_id": sc._SESSION_MESSAGE_ID,
    "user_id": sc._SESSION_USER_ID,
    "session_key": sc._SESSION_KEY,
    "session_id": sc._SESSION_ID,
}


@pytest.fixture(autouse=True)
def isolate_context_and_environment(monkeypatch):
    saved = {key: var.get() for key, var in FIELDS.items()}
    for var in FIELDS.values():
        var.set(sc._UNSET)
    for key in FIELDS:
        monkeypatch.delenv(f"HERMES_SESSION_{key.upper()}", raising=False)
    yield
    for key, var in FIELDS.items():
        var.set(saved[key])


def test_reads_only_valid_bound_identity_fields():
    expected = dict(platform="discord", chat_id="c", thread_id="t", message_id="m",
                    user_id="u", session_key="sk", session_id="sid")
    for key, value in expected.items():
        FIELDS[key].set(value)
    assert sc.get_bound_session_source() == expected


def test_unbound_empty_reset_and_environment_only_are_absent(monkeypatch):
    assert sc.get_bound_session_source() == {}
    FIELDS["chat_id"].set("")
    assert sc.get_bound_session_source() == {}
    FIELDS["chat_id"].reset(FIELDS["chat_id"].set("temporary"))
    assert sc.get_bound_session_source() == {}
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "environment")
    assert sc.get_bound_session_source() == {}


@pytest.mark.parametrize("value", [None, 1, False, [], {}, "   "])
def test_malformed_or_empty_values_are_excluded(value):
    FIELDS["chat_id"].set(value)
    assert sc.get_bound_session_source() == {}


def test_async_context_isolation_and_partial_tuple():
    async def run():
        FIELDS["chat_id"].set("parent")

        async def child():
            assert sc.get_bound_session_source() == {"chat_id": "parent"}
            FIELDS["chat_id"].set("")
            FIELDS["thread_id"].set("child-thread")
            assert sc.get_bound_session_source() == {"thread_id": "child-thread"}

        await asyncio.create_task(child())
        assert sc.get_bound_session_source() == {"chat_id": "parent"}

    asyncio.run(run())
    FIELDS["platform"].set("discord")
    FIELDS["chat_id"].set("only-chat")
    assert sc.get_bound_session_source() == {"platform": "discord", "chat_id": "only-chat"}
