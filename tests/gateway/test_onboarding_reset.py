"""``/new`` (a canonical reset) inside the onboarding chat, then Skip setup: the run ends and the install
stops being mid-onboarding. The reset child is its own conversation, so it carries no copy of the run
that Skip (which names the chat's logical id) could never reach.
"""
from __future__ import annotations

import pytest

from tests.gateway.test_onboarding_on_the_authority import _desktop, _request, _settled
from tests.waiting import async_wait_until


async def _mutate(desktop, rid, stored, operation):
    """One revision-fenced ``session.mutate``; resent on a revision conflict as the Desktop does."""
    for attempt in range(5):
        row = desktop.db.get_session(stored)
        reply = await _request(desktop, rid * 10 + attempt, "session.mutate", {
            "session_id": stored, "request_id": f"{operation}-{rid}-{attempt}", "operation": operation,
            "expected_revision": row["runtime_revision"], "expected_generation": row["runtime_generation"],
            "payload": {}})
        if "result" in reply or reply.get("error", {}).get("data", {}).get("reason") != "revision_conflict":
            return reply
    return reply


@pytest.mark.asyncio
async def test_skip_after_new_in_the_onboarding_chat_ends_the_run(tmp_path, monkeypatch):
    from agent.onboarding_record import open_session_ids
    from hermes_cli.config import read_user_config_raw
    from hermes_cli.onboarding_run import _root_config
    from hermes_state_runtime import list_session_admissions
    turns = []
    desktop = await _desktop(tmp_path, monkeypatch, turns)
    try:
        stored = (await _request(desktop, 1, "onboarding.start", {"rerun": True}))["result"]["stored_session_id"]
        await async_wait_until(lambda: turns, timeout=10)
        await _settled(desktop, stored)
        # The opening turn's admission is terminal: a reset refuses a started turn as session_busy.
        await async_wait_until(lambda: not list_session_admissions(desktop.db, session_id=stored), timeout=10)

        reset = await _mutate(desktop, 2, stored, "reset")
        assert "result" in reset, reset
        skipped = await _request(desktop, 3, "onboarding.skip", {"session_id": stored})

        assert skipped.get("result") == {"skipped": True}, skipped
        assert open_session_ids(desktop.db) == []
        assert read_user_config_raw(_root_config())["onboarding"]["run"] is False
    finally:
        await desktop.connection.close()
