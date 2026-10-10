"""Queued turns use their own tool identity without changing the cached prompt."""

import asyncio
import importlib
import json
from pathlib import Path
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
    clear_session_vars,
    get_session_env,
    get_session_transport,
    set_session_vars,
)
from gateway.turn_context import TurnContext
from hermes_constants import get_hermes_home
from plugins.platforms.matrix.adapter import MatrixAdapter
from tools.registry import registry


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True])
async def test_queued_tool_context_restores_outer_identity_and_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: bool
):
    importlib.import_module("tools.matrix_followup_tool")
    importlib.import_module("tools.matrix_reaction_tool")
    homes = {profile: tmp_path / profile for profile in ("a", "b")}
    adapters: dict[str, MatrixAdapter] = {}
    sources: dict[str, SessionSource] = {}
    reaction_mocks: dict[str, AsyncMock] = {}
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner.adapters = {}
    runner._draining = False
    runner._gateway_loop = asyncio.get_running_loop()
    runner._profile_adapters = {}

    def profile_home(source: SessionSource) -> Path:
        assert source.profile is not None
        return homes[source.profile]

    monkeypatch.setattr(runner, "_resolve_profile_home_for_source", profile_home)
    for profile, requester in (("a", "@alice:test"), ("b", "@bob:test")):
        homes[profile].mkdir()
        (homes[profile] / ".env").write_text(
            f"TEST_OWNER={profile}\n", encoding="utf-8"
        )
        adapter = MatrixAdapter(PlatformConfig(enabled=True))
        adapter._reactions_enabled = False
        adapter._joined_rooms.add("!room:test")
        adapter._is_dm_room = AsyncMock(return_value=False)
        adapter.set_authorization_check(
            lambda user, _chat_type, _chat_id: user in {"@alice:test", "@bob:test"}
        )
        reaction = AsyncMock(return_value={"success": True})
        monkeypatch.setattr(adapter, "add_reaction", reaction)
        reaction_mocks[profile] = reaction
        adapters[profile] = adapter
        runner._profile_adapters[profile] = {Platform.MATRIX: adapter}
        sources[profile] = SessionSource(
            platform=Platform.MATRIX,
            chat_id="!room:test",
            chat_type="group",
            thread_id="$thread",
            user_id=requester,
            profile=profile,
            message_id=f"${profile}-inbound",
        )

    def pending_audio_paths(event: MessageEvent) -> list[str]:
        return []

    monkeypatch.setattr(runner, "_pending_event_audio_paths", pending_audio_paths)
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(
        return_value="Queued request"
    )

    def is_goal_continuation(event: MessageEvent) -> bool:
        return False

    def pinned_channel_inputs(
        session_key: str | None,
        prompt: str | None,
        source: SessionSource,
        *,
        internal: bool,
    ) -> tuple[str | None, SessionSource]:
        return prompt, source

    monkeypatch.setattr(runner, "_is_goal_continuation_event", is_goal_continuation)
    monkeypatch.setattr(runner, "_pinned_channel_inputs", pinned_channel_inputs)
    runner._run_agent_deliver_first_response = AsyncMock()
    runner._refresh_agent_cache_message_count = AsyncMock()

    async def execute(
        message: str,
        context_prompt: str,
        history: list[dict[str, object]],
        source: SessionSource,
        session_id: str,
        **kwargs: object,
    ) -> dict[str, object]:
        configured_output = await asyncio.to_thread(
            registry.dispatch, "matrix_followup", {"enabled": True}
        )
        configured = (
            json.loads(configured_output)
            if isinstance(configured_output, str)
            else configured_output
        )
        reacted_output = await asyncio.to_thread(
            registry.dispatch, "matrix_reaction", {"action": "react", "emoji": "👍"}
        )
        reacted = (
            json.loads(reacted_output)
            if isinstance(reacted_output, str)
            else reacted_output
        )
        assert source.profile is not None
        bound_adapter, _ = get_session_transport()
        choice = bound_adapter._reaction_followup_actions[
            get_session_env("HERMES_SESSION_KEY")
        ]
        observation = (
            get_hermes_home(),
            get_secret("TEST_OWNER"),
            context_prompt,
            choice.requester,
            choice.session_id,
            configured,
            reacted,
        )
        assert observation == (
            homes[source.profile],
            source.profile,
            "Pinned session prompt",
            source.user_id,
            session_id,
            {"success": True, "enabled": True, "emoji": []},
            {"success": True},
        )
        if failure:
            raise RuntimeError("queued execution failed")
        return {"final_response": "Queued answer", "messages": history}

    monkeypatch.setattr(runner, "_run_agent_inner", execute)
    source = sources["a"]
    queued_sources = [
        replace_source(source, user_id="@bob:test", message_id="$bob-inbound"),
        sources["b"],
        source,
    ]
    key = runner._session_key_for_source(source)
    turn = TurnContext(
        source=source,
        session_key=key,
        session_id="sid",
        run_generation=1,
        _interrupt_depth=0,
        history=[],
        _status_thread_metadata={},
        context_prompt="Pinned session prompt",
    )
    assert source.user_id is not None
    assert source.thread_id is not None
    assert source.message_id is not None
    tokens = set_session_vars(
        platform="matrix",
        chat_id=source.chat_id,
        user_id=source.user_id,
        thread_id=source.thread_id,
        profile="a",
        session_key=key,
        session_id="sid",
        message_id=source.message_id,
        transport_adapter=adapters["a"],
        transport_loop=asyncio.get_running_loop(),
    )
    cwd_token = set_session_cwd("/outer/workspace")
    try:
        with _profile_runtime_scope(homes["a"]):
            outer = (
                get_session_env("HERMES_SESSION_USER_ID"),
                get_session_transport(),
                scoped_session_cwd(),
                get_hermes_home(),
                get_secret("TEST_OWNER"),
            )
            for queued_source in queued_sources:
                profile = queued_source.profile
                assert profile is not None
                queued_key = runner._session_key_for_source(queued_source)
                adapter = adapters[profile]
                adapter._active_sessions[queued_key] = asyncio.Event()
                event = MessageEvent(
                    text="Queued request",
                    source=queued_source,
                    message_id=queued_source.message_id,
                )
                adapter._pending_messages[key] = event
                pending_event, pending = await runner._run_agent_drain_pending(
                    {"final_response": "First answer"},
                    adapter,
                    source,
                    key,
                )
                operation = runner._run_agent_queued_followup(
                    turn,
                    adapter,
                    pending,
                    pending_event,
                    "First answer",
                    {"final_response": "First answer", "messages": []},
                    None,
                )
                if failure:
                    with pytest.raises(RuntimeError, match="queued execution failed"):
                        await operation
                else:
                    await operation
                assert (
                    get_session_env("HERMES_SESSION_USER_ID"),
                    get_session_transport(),
                    scoped_session_cwd(),
                    get_hermes_home(),
                    get_secret("TEST_OWNER"),
                ) == outer
            assert reaction_mocks["a"].await_args_list == [
                call(chat_id="!room:test", emoji="👍", message_id="$bob-inbound"),
                call(chat_id="!room:test", emoji="👍", message_id="$a-inbound"),
            ]
            reaction_mocks["b"].assert_awaited_once_with(
                chat_id="!room:test",
                emoji="👍",
                message_id="$b-inbound",
            )
    finally:
        reset_session_cwd(cwd_token)
        clear_session_vars(tokens)
