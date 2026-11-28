"""A Desktop window that reconnects while a sudo prompt waits gets the card again and answers it.

The whole path on the session authority (``session.resume`` -> ``open_requests`` -> the response frame),
for the prompt that, before, was skipped at once because no viewer was attached when the turn asked
(``test_local_viewer_prompts_wait_for_viewer.py`` covers secret and vault prompts and the timeouts).
Its own file: after the sibling file's in-process sudo prompts ran, ``onboarding.start`` on a fresh
authority in the same process refused with "No model is set up" (an ordering leak this test does not
own; the runner isolates files in their own process).
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from tests.waiting import async_wait_until


@pytest.mark.asyncio
async def test_a_desktop_that_reconnects_mid_prompt_answers_the_sudo_card_it_is_replayed(tmp_path, monkeypatch):
    """The whole path on the session authority: the window closes, the turn asks, the window comes back,
    ``session.resume`` hands it the card and its response frame unblocks the turn."""
    from gateway.session_sudo_prompt import sudo_prompt_scope
    from tests.gateway.test_onboarding_on_the_authority import _desktop, _request
    from tests.gateway.test_server_request_resume_replay import _reload
    from tools.terminal_tool_sudo import _prompt_for_sudo_password
    desktop = await _desktop(tmp_path, monkeypatch, [])
    try:
        stored = (await _request(desktop, 1, "onboarding.start", {"rerun": True}))["result"]["stored_session_id"]
        assert "result" in await _request(desktop, 2, "session.resume", {"session_id": stored})
        live = desktop.authority.sessions[stored]
        await desktop.connection.close()
        assert not live.event_stream.fanout.transports()
        turn = SimpleNamespace(_approval_owner=(desktop.authority, stored, 1),
                               _ctx=SimpleNamespace(source=SimpleNamespace(platform=Platform.LOCAL)))

        def ask():
            with sudo_prompt_scope(turn, SimpleNamespace()):
                return _prompt_for_sudo_password(command="sudo true")
        answer = asyncio.get_running_loop().run_in_executor(None, ask)
        from gateway.session_server_requests import waiting
        await async_wait_until(lambda: waiting(live.event_stream, stored, "sudo"), timeout=5)

        await _reload(desktop)
        resumed = await _request(desktop, 3, "session.resume", {"session_id": stored})
        [entry] = [e for e in resumed["result"]["open_requests"] if e["method"] == "sudo"]
        assert await desktop.connection.dispatch({"jsonrpc": "2.0", "id": entry["id"],
                                                  "result": {"value": "pw"}}) is None
        assert await asyncio.wait_for(answer, 10) == "pw"
    finally:
        await desktop.connection.close()
