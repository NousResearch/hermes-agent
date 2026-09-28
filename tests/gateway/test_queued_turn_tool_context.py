"""Queued turns use their own tool identity without changing the cached prompt."""

import asyncio
import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, call

import pytest

from agent.runtime_cwd import scoped_session_cwd, set_session_cwd, reset_session_cwd
from agent.secret_scope import get_secret
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner, _profile_runtime_scope
from gateway.session import SessionSource
from gateway.session_identity import replace_source
from gateway.session_context import (
    clear_session_vars, get_session_env, get_session_transport, set_session_vars,
)
from hermes_constants import get_hermes_home
from plugins.platforms.matrix.adapter import MatrixAdapter
from tools.registry import registry


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True])
async def test_queued_tool_context_restores_outer_identity_and_profile(tmp_path, failure):
    importlib.import_module("tools.matrix_followup_tool")
    importlib.import_module("tools.matrix_reaction_tool")
    homes = {profile: tmp_path / profile for profile in ("a", "b")}
    adapters = {}
    sources = {}
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner.adapters = {}
    runner._draining = False
    runner._profile_adapters = {}
    runner._resolve_profile_home_for_source = lambda source: homes[source.profile]
    for profile, requester in (("a", "@alice:test"), ("b", "@bob:test")):
        homes[profile].mkdir()
        (homes[profile] / ".env").write_text(f"TEST_OWNER={profile}\n", encoding="utf-8")
        adapter = MatrixAdapter(PlatformConfig(enabled=True))
        adapter._reactions_enabled = False
        adapter.add_reaction = AsyncMock(return_value={"success": True})
        adapters[profile] = adapter
        runner._profile_adapters[profile] = {Platform.MATRIX: adapter}
        sources[profile] = SessionSource(
            platform=Platform.MATRIX, chat_id="!room:test", chat_type="group",
            thread_id="$thread", user_id=requester, profile=profile,
            message_id=f"${profile}-inbound",
        )
    runner._pending_event_audio_paths = lambda _event: []
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(return_value="Queued request")
    runner._is_goal_continuation_event = lambda _event: False
    runner._pinned_channel_inputs = lambda _key, prompt, source, **_kwargs: (prompt, source)
    runner._run_agent_deliver_first_response = AsyncMock()
    runner._refresh_agent_cache_message_count = AsyncMock()
    runner._clear_streaming_tts_turn_completed = lambda *_args: None
    async def execute(message, context_prompt, history, source, session_id, **kwargs):
        configured = json.loads(await asyncio.to_thread(
            registry.dispatch, "matrix_followup", {"enabled": True},
        ))
        reacted = json.loads(await asyncio.to_thread(
            registry.dispatch, "matrix_reaction", {"action": "react", "emoji": "👍"},
        ))
        bound_adapter, _ = get_session_transport()
        choice = bound_adapter._reaction_followup_actions[get_session_env("HERMES_SESSION_KEY")]
        observation = (
            get_hermes_home(), get_secret("TEST_OWNER"), context_prompt,
            choice.requester, choice.session_id, configured, reacted,
        )
        assert observation == (
            homes[source.profile], source.profile, "Pinned session prompt", source.user_id,
            session_id, {"success": True, "enabled": True, "emoji": []}, {"success": True},
        )
        if failure:
            raise RuntimeError("queued execution failed")
        return {"final_response": "Queued answer", "messages": history}

    runner._run_agent_inner = execute
    source = sources["a"]
    queued_sources = [replace_source(source, user_id="@bob:test", message_id="$bob-inbound"),
                      sources["b"], source]
    key = runner._session_key_for_source(source)
    turn = SimpleNamespace(
        source=source, session_key=key, session_id="sid", run_generation=1,
        _interrupt_depth=0, history=[], _status_thread_metadata={},
        context_prompt="Pinned session prompt",
    )
    tokens = set_session_vars(
        platform="matrix", chat_id=source.chat_id, user_id=source.user_id,
        thread_id=source.thread_id, profile="a", session_key=key,
        session_id="sid", message_id=source.message_id, transport_adapter=adapters["a"],
    )
    cwd_token = set_session_cwd("/outer/workspace")
    try:
        with _profile_runtime_scope(homes["a"]):
            outer = (get_session_env("HERMES_SESSION_USER_ID"), get_session_transport(),
                     scoped_session_cwd(), get_hermes_home(), get_secret("TEST_OWNER"))
            for queued_source in queued_sources:
                profile = queued_source.profile
                queued_key = runner._session_key_for_source(queued_source)
                adapter = adapters[profile]
                adapter._active_sessions[queued_key] = asyncio.Event()
                event = MessageEvent(text="Queued request", source=queued_source,
                                     message_id=queued_source.message_id)
                adapter._pending_messages[key] = event
                pending_event, pending = await runner._run_agent_drain_pending(
                    {"final_response": "First answer"}, adapter, source, key,
                )
                operation = runner._run_agent_queued_followup(
                    turn, adapter, pending, pending_event, "First answer",
                    {"final_response": "First answer", "messages": []}, None,
                )
                if failure:
                    with pytest.raises(RuntimeError, match="queued execution failed"):
                        await operation
                else:
                    await operation
                assert (get_session_env("HERMES_SESSION_USER_ID"), get_session_transport(),
                        scoped_session_cwd(), get_hermes_home(), get_secret("TEST_OWNER")) == outer
            assert adapters["a"].add_reaction.await_args_list == [
                call(chat_id="!room:test", emoji="👍", message_id="$bob-inbound"),
                call(chat_id="!room:test", emoji="👍", message_id="$a-inbound"),
            ]
            adapters["b"].add_reaction.assert_awaited_once_with(
                chat_id="!room:test", emoji="👍", message_id="$b-inbound",
            )
    finally:
        reset_session_cwd(cwd_token)
        clear_session_vars(tokens)
