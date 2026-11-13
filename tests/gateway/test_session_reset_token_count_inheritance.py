"""A compression-exhaustion auto-reset must not hand the successor session the exhausted
session's provider token count.

The reset rotates the route to a NEW session mid-turn, but ``agent_result`` still carries the
provider reading that belongs to the OLD transcript. Stamping that reading onto the new entry
gives the fresh session a token count it never had: gateway hygiene then reads it as "actual"
usage, sees it above threshold, and auto-compresses that fresh session on every turn until the
count is cleared.
"""
from types import SimpleNamespace

import pytest

from gateway.run_turn_persistence import GatewayTurnPersistenceMixin
from gateway.session import SessionEntry


class _FakeStore:
    """Minimal async_session_store double: records the last_prompt_tokens written per key."""

    def __init__(self):
        self.written = []
        self.rows = []

    async def update_session(self, session_key, last_prompt_tokens=None, touch_activity=True):
        self.written.append((session_key, last_prompt_tokens))

    async def _save(self):
        return None

    async def append_to_transcript(self, sid, row, skip_db=False):
        self.rows.append((sid, row.get("role")))


def _runner_with_store():
    """Bind the real ``_hmwa_persist_turn_transcript`` to a bare object.

    ``GatewayRunner.__init__`` touches the real Hermes home, so bind the method onto a plain
    namespace instead: the method under test only needs ``self.async_session_store``.
    """
    async def _refresh(*_a, **_k):
        return None

    runner = SimpleNamespace(async_session_store=_FakeStore(), _session_db=None,
                             _refresh_agent_cache_message_count=_refresh)
    runner._hmwa_persist_turn_transcript = (
        GatewayTurnPersistenceMixin._hmwa_persist_turn_transcript.__get__(runner)
    )
    runner._hmwa_user_transcript_entry = GatewayTurnPersistenceMixin._hmwa_user_transcript_entry
    return runner


def _entry(session_key: str, session_id: str) -> SessionEntry:
    return SessionEntry(
        session_key=session_key,
        session_id=session_id,
        created_at=SimpleNamespace(isoformat=lambda: "2026-01-01T00:00:00"),
        updated_at=SimpleNamespace(isoformat=lambda: "2026-01-01T00:00:00"),
    )


@pytest.mark.asyncio
async def test_reset_successor_does_not_inherit_exhausted_sessions_token_count():
    runner = _runner_with_store()
    old_entry = _entry("agent:main:discord:dm:1", "20260101_000000_aaaaaaaa")
    # The route rotated to a fresh session while the old turn was still unwinding.
    new_entry = _entry("agent:main:discord:dm:1", "20260101_000001_bbbbbbbb")

    agent_result = {"last_prompt_tokens": 907_317}

    await runner._hmwa_persist_turn_transcript(
        event=SimpleNamespace(internal=False, message_id="m1"),
        source=SimpleNamespace(platform=None),
        session_entry=new_entry,
        session_key="agent:main:discord:dm:1",
        agent_result=agent_result,
        agent_messages=[],
        prepared=SimpleNamespace(history=[], message_text="", persistence_owner=None, persist_user_display_kind=None, persist_user_message=False, persist_user_timestamp=None),
        response="",
        agent_failed_early=False,
        hidden_reasoning_incomplete=False,
        is_context_overflow_failure=False,
        turn_session_id=old_entry.session_id,
    )

    written = runner.async_session_store.written
    assert written, "the turn must still persist a token count"
    _key, value = written[-1]
    assert value == 0, (
        "the successor session inherited the exhausted session's token count "
        f"({value}); gateway hygiene would read it as real usage and re-compress every turn"
    )


@pytest.mark.asyncio
async def test_same_session_turn_still_persists_its_own_token_count():
    """The fix must not suppress the normal case: a turn that ran under the routed session
    keeps writing its own provider reading."""
    runner = _runner_with_store()
    entry = _entry("agent:main:discord:dm:2", "20260101_000002_cccccccc")

    await runner._hmwa_persist_turn_transcript(
        event=SimpleNamespace(internal=False, message_id="m2"),
        source=SimpleNamespace(platform=None),
        session_entry=entry,
        session_key="agent:main:discord:dm:2",
        agent_result={"last_prompt_tokens": 42_000},
        agent_messages=[],
        prepared=SimpleNamespace(history=[], message_text="", persistence_owner=None, persist_user_display_kind=None, persist_user_message=False, persist_user_timestamp=None),
        response="",
        agent_failed_early=False,
        hidden_reasoning_incomplete=False,
        is_context_overflow_failure=False,
        turn_session_id=entry.session_id,
    )

    _key, value = runner.async_session_store.written[-1]
    assert value == 42_000