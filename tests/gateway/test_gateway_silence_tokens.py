"""Gateway intentional-silence token behavior."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionEntry, SessionSource
from gateway.response_filters import (
    is_intentional_silence_agent_result,
    is_intentional_silence_response,
)


def _source():
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        user_id="12345",
    )


def _event(*, internal: bool = False):
    return MessageEvent(
        text="side chatter",
        source=_source(),
        message_id="msg-42",
        internal=internal,
    )


def _runner(monkeypatch, tmp_path):
    runner = gateway_run.GatewayRunner(GatewayConfig())
    runner.adapters = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._handle_active_session_busy_message = AsyncMock(return_value=False)
    runner._session_db = MagicMock()
    runner._recover_telegram_topic_thread_id = lambda _source: None
    runner._cache_session_source = lambda _key, _source: None
    runner._is_session_run_current = lambda _key, _gen: True
    runner._reply_anchor_for_event = lambda _event: None
    runner._get_guild_id = lambda _event: None
    runner._should_send_voice_reply = lambda *_a, **_kw: False
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()

    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key="agent:main:telegram:group:-1001:12345",
        session_id="sess-silent",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="group",
    )
    runner.session_store.load_transcript.return_value = []
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"}
    )
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *_args, **_kwargs: 100_000,
    )
    return runner


def test_exact_silence_tokens_are_intentional_silence():
    for token in ("[SILENT]", " SILENT ", "NO_REPLY", "no reply"):
        assert is_intentional_silence_response(token)


def test_blank_and_prose_mentions_are_not_silence():
    assert not is_intentional_silence_response("")
    assert not is_intentional_silence_response("Use NO_REPLY when no answer is needed.")
    assert not is_intentional_silence_response("The reply was [SILENT], intentionally.")


def test_human_silence_requires_success_without_changing_legacy_marker_policy():
    assert is_intentional_silence_agent_result({"failed": False}, "NO_REPLY")
    for unsuccessful in (
        {"failed": True}, {"partial": True}, {"completed": False},
        {"interrupted": True}, {"error": "provider failed"},
    ):
        assert not is_intentional_silence_agent_result(
            unsuccessful, "NO_REPLY", human_silence_opt_in=True,
        )
        assert is_intentional_silence_agent_result(unsuccessful, "NO_REPLY") is not bool(unsuccessful.get("failed"))


@pytest.mark.asyncio
@pytest.mark.parametrize("text,status,expected", [
    ("[SILENT]", {}, ""),
    ("\u200b\ufeff", {}, ""),
    ("\u200b\ufeff", {"failed": True, "error": "provider failed"}, "Something went wrong"),
    ("\u200b\ufeff", {"partial": True}, "I had to stop"),
    ("\u200b\ufeff", {"completed": False}, "no response was generated"),
    ("", {}, "no response was generated"),
    ("Use [SILENT] when no answer is needed.", {}, "Use [SILENT] when no answer is needed."),
    ("Hello\u200b there", {}, "Hello\u200b there"),
])
async def test_opted_in_human_turn_only_suppresses_successful_non_content(
    monkeypatch, tmp_path, text, status, expected,
):
    runner = _runner(monkeypatch, tmp_path)
    runner.config.allow_human_silence_markers = True
    runner._run_agent = AsyncMock(return_value={
        "final_response": text,
        "messages": [
            {"role": "user", "content": "side chatter"},
            {"role": "assistant", "content": text},
        ],
        "tools": [], "history_offset": 0, "last_prompt_tokens": 0,
        "api_calls": 1, "failed": False, **status,
    })
    response = await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1,
    )
    if expected:
        assert expected in response
    else:
        assert response == ""
        appended = [call.args[1] for call in runner.session_store.append_to_transcript.call_args_list]
        assert {"role": "assistant", "content": text}.items() <= appended[-1].items()
        assert [msg["role"] for msg in appended if msg.get("role") in {"user", "assistant"}] == ["user", "assistant"]
        assert all(not msg.get("display_kind") for msg in appended if msg.get("role") == "user")


@pytest.mark.asyncio
@pytest.mark.parametrize("default_enabled,profile_enabled", [(False, True), (True, False), (True, None)])
async def test_silence_opt_in_belongs_to_the_routed_profile(
    monkeypatch, tmp_path, default_enabled, profile_enabled,
):
    from gateway.config import load_gateway_config
    from hermes_cli.config import set_config_value

    profile_home = tmp_path / "secondary"
    profile_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    if profile_enabled is not None:
        set_config_value("gateway.allow_human_silence_markers", str(profile_enabled).lower())
    assert load_gateway_config().allow_human_silence_markers is bool(profile_enabled)

    runner = _runner(monkeypatch, tmp_path)
    runner.config.allow_human_silence_markers = default_enabled
    runner.config.multiplex_profiles = True
    runner._resolve_profile_home_for_source = lambda source: profile_home
    runner._deliver_queued_first_response = AsyncMock()
    source = _source()
    source.profile = "secondary"
    result = {"final_response": "[SILENT]", "failed": False, "api_calls": 1}
    _, silent, _ = await runner._hmwa_shape_agent_response(
        result, source, [], SimpleNamespace(session_id="s"), None,
        None, 1, "s", "telegram", 0,
    )
    assert silent is bool(profile_enabled)

    turn_ctx = SimpleNamespace(
        session_key="secondary-key", stream_consumer_holder=[None],
        persist_user_display_kind=None, source=source, _status_thread_metadata=None,
        event_message_id=None, inbound_message_id="msg-42", run_generation=1,
    )
    await runner._run_agent_deliver_first_response(turn_ctx, None, result, result, None)
    assert runner._deliver_queued_first_response.await_count == (0 if profile_enabled else 1)


@pytest.mark.asyncio
async def test_silent_streamed_turn_does_not_send_a_standalone_footer(monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path)
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    response = await runner._hmwa_deliver_turn_response(
        _event(), _source(), SimpleNamespace(session_id="s"), "key", 1,
        {"already_sent": True, "failed": False}, [], "[SILENT]", "Tokens: 100", True,
    )
    adapter.send.assert_not_awaited()
    assert not response


@pytest.mark.asyncio
async def test_human_turn_gets_a_visible_fallback_for_a_silence_marker(monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(return_value={
        "final_response": "[SILENT]",
        "messages": [
            {"role": "user", "content": "side chatter"},
            {"role": "assistant", "content": "[SILENT]"},
        ],
        "tools": [],
        "history_offset": 0,
        "last_prompt_tokens": 0,
        "api_calls": 1,
        "failed": False,
    })

    response = await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    assert "silence marker" in response
    assert "Try again or rephrase" in response


@pytest.mark.asyncio
@pytest.mark.parametrize("failed,expected", [
    (False, "no response was generated"), (True, "Something went wrong"),
])
async def test_non_opted_in_invisible_output_is_normalized(monkeypatch, tmp_path, failed, expected):
    invisible = "\u200b\ufeff"
    runner = _runner(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(return_value={
        "final_response": invisible,
        "messages": [
            {"role": "user", "content": "side chatter"},
            {"role": "assistant", "content": invisible},
        ],
        "tools": [], "history_offset": 0, "last_prompt_tokens": 0,
        "api_calls": 1, "failed": failed,
        "error": "provider failed" if failed else None,
    })
    response = await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1,
    )
    assert expected in response
    assert invisible not in response


@pytest.mark.asyncio
@pytest.mark.parametrize("allow_silence", [False, True])
@pytest.mark.parametrize("status", [
    {}, {"partial": True}, {"interrupted": True},
    {"completed": False}, {"error": "provider detail"},
])
async def test_internal_silence_token_suppresses_delivery_but_preserves_transcript(
    monkeypatch, tmp_path, allow_silence, status,
):
    runner = _runner(monkeypatch, tmp_path)
    runner.config.allow_human_silence_markers = allow_silence
    runner._run_agent = AsyncMock(return_value={
        "final_response": "[SILENT]",
        "messages": [
            {"role": "user", "content": "side chatter"},
            {"role": "assistant", "content": "[SILENT]"},
        ],
        "tools": [],
        "history_offset": 0,
        "last_prompt_tokens": 0,
        "api_calls": 1,
        "failed": False,
        **status,
    })

    response = await runner._handle_message_with_agent(
        _event(internal=True), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    assert response == ""
    appended = [call.args[1] for call in runner.session_store.append_to_transcript.call_args_list]
    assert {"role": "assistant", "content": "[SILENT]"}.items() <= appended[-1].items()
    assert [msg["role"] for msg in appended if msg.get("role") in {"user", "assistant"}] == ["user", "assistant"]


@pytest.mark.asyncio
@pytest.mark.parametrize("internal", [False, True])
@pytest.mark.parametrize("allow_silence", [False, True])
@pytest.mark.parametrize("status", [
    {}, {"failed": True}, {"partial": True}, {"interrupted": True},
    {"completed": False}, {"error": "provider detail"},
])
async def test_queued_silence_policy_belongs_to_each_turn(
    monkeypatch, tmp_path, internal, allow_silence, status,
):
    runner = _runner(monkeypatch, tmp_path)
    runner.config.allow_human_silence_markers = allow_silence
    kind = "internal_notification" if internal else None
    result = {"final_response": "[SILENT]", "api_calls": 1, "failed": False, **status}
    expected_silence = not result["failed"] if internal else allow_silence and not status

    # The terminal kind overrides an opener with the opposite origin.
    terminal_result = {**result, "queued_terminal_display_kind": kind,
                       "queued_terminal_allow_human_silence": allow_silence}
    _, silent, _ = await runner._hmwa_shape_agent_response(
        terminal_result, _source(), [], SimpleNamespace(session_id="s"), None,
        None, 1, "s", "telegram", 0,
        persist_user_display_kind=None if internal else "internal_notification",
    )
    assert silent is expected_silence

    runner._deliver_queued_first_response = AsyncMock()
    turn_ctx = SimpleNamespace(
        session_key="key", stream_consumer_holder=[None],
        persist_user_display_kind=kind, source=_source(), _status_thread_metadata=None,
        event_message_id=None, inbound_message_id="msg-42", run_generation=1,
    )
    await runner._run_agent_deliver_first_response(turn_ctx, None, result, result, None)
    assert runner._deliver_queued_first_response.await_count == (0 if expected_silence else 1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "first_response,failed,expected",
    [
        ("NO_REPLY", False, "silence marker"),
        ("\u200b\ufeff", False, "no response was generated"),
        ("\u200b\ufeff", True, "Something went wrong"),
    ],
)
@pytest.mark.parametrize("allow_silence", [False, True])
async def test_queued_human_turn_only_suppresses_successful_non_content(first_response, failed, expected, allow_silence):
    runner = gateway_run.GatewayRunner(GatewayConfig())
    runner.config.allow_human_silence_markers = allow_silence
    runner._deliver_queued_first_response = AsyncMock()
    turn_ctx = SimpleNamespace(
        session_key="agent:main:telegram:group:-1001:12345",
        stream_consumer_holder=[None],
        persist_user_display_kind=None,
        source=_source(),
        _status_thread_metadata=None,
        event_message_id=None,
        inbound_message_id="msg-42",
        run_generation=1,
    )
    result = {
        "final_response": first_response,
        "failed": failed,
        "api_calls": 1,
        "error": "provider failed" if failed else None,
    }

    await runner._run_agent_deliver_first_response(
        turn_ctx, None, result, result, None,
    )

    queued_call = runner._deliver_queued_first_response.await_args
    if allow_silence and not failed:
        assert queued_call is None
    else:
        assert queued_call is not None
        assert expected in queued_call.args[0]


@pytest.mark.asyncio
async def test_queued_terminal_turn_owns_the_silence_verdict(monkeypatch, tmp_path):
    """The chain's LAST turn decides whether a bare marker may vanish, not the opener."""
    runner = _runner(monkeypatch, tmp_path)
    runner._MAX_INTERRUPT_DEPTH = 8
    runner._run_agent = AsyncMock(return_value={"final_response": "NO_REPLY", "messages": []})
    runner._is_goal_continuation_event = MagicMock(return_value=False)
    runner._session_key_for_source = MagicMock(return_value="agent:main:telegram:group:-1001:12345")
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(return_value="follow-up")
    runner._adapter_for_source = MagicMock(return_value=None)
    runner._refresh_agent_cache_message_count = AsyncMock()
    turn_ctx = SimpleNamespace(
        source=_source(), session_id="sid", session_key="agent:main:telegram:group:-1001:12345",
        run_generation=1, _interrupt_depth=0, history=[], _status_thread_metadata=None,
        context_prompt=None, result_holder=[None])
    pending_event = SimpleNamespace(source=_source(), message_id="43", channel_prompt=None,
                                    message_type=None, internal=True)

    merged = await gateway_run.GatewayRunner._run_agent_queued_followup(
        runner, turn_ctx, adapter=None, pending="hi again", pending_event=pending_event,
        response="resp", result={"interrupted": True, "messages": []}, stream_task=None)

    assert runner._run_agent.await_args.kwargs["persist_user_display_kind"] == "internal_notification"
    assert merged["queued_terminal_display_kind"] == "internal_notification"

    def _result(terminal_kind):
        return {
            "final_response": "[SILENT]", "tools": [], "history_offset": 0, "last_prompt_tokens": 0,
            "api_calls": 1, "failed": False, "queued_terminal_inbound_id": "43",
            "queued_terminal_display_kind": terminal_kind,
            "messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "[SILENT]"}],
        }

    # Human opener, internal terminal turn: silent.
    runner = _runner(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(return_value=_result("internal_notification"))
    assert await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1) == ""
    # Internal opener, human terminal turn: visible fallback.
    runner = _runner(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(return_value=_result(None))
    response = await runner._handle_message_with_agent(
        _event(internal=True), _source(), "agent:main:telegram:group:-1001:12345", 1)
    assert "silence marker" in response


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_enabled,nested", [(True, False), (False, False), (True, True), (False, True)])
async def test_queued_terminal_profile_owns_the_silence_opt_in(
    monkeypatch, tmp_path, terminal_enabled, nested,
):
    from hermes_cli.config import set_config_value

    homes = {}
    for profile, enabled in (("opener", not terminal_enabled), ("terminal", terminal_enabled)):
        homes[profile] = tmp_path / profile
        homes[profile].mkdir()
        monkeypatch.setenv("HERMES_HOME", str(homes[profile]))
        set_config_value("gateway.allow_human_silence_markers", str(enabled).lower())

    runner = _runner(monkeypatch, tmp_path)
    runner.config.multiplex_profiles = True
    runner._resolve_profile_home_for_source = lambda source: homes[source.profile]
    runner._MAX_INTERRUPT_DEPTH = 8
    runner._is_goal_continuation_event = MagicMock(return_value=False)
    runner._session_key_for_source = MagicMock(return_value="key")
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(return_value="follow-up")
    runner._adapter_for_source = MagicMock(return_value=None)
    runner._refresh_agent_cache_message_count = AsyncMock()
    opener, terminal = _source(), _source()
    opener.profile, terminal.profile = "opener", "terminal"
    turn_ctx = SimpleNamespace(
        source=opener, session_id="sid", session_key="key", run_generation=1,
        _interrupt_depth=0, history=[], _status_thread_metadata=None,
        context_prompt=None, result_holder=[None],
    )
    result = {"final_response": "[SILENT]", "messages": []}
    # First produce the terminal result through the real queue handoff. If nested,
    # unwind it through another handoff whose profile has the opposite setting.
    for next_source in ([terminal, opener] if nested else [terminal]):
        runner._run_agent = AsyncMock(return_value=result)
        pending_event = SimpleNamespace(
            source=next_source, message_id="43", channel_prompt=None,
            message_type=None, internal=False,
        )
        result = await runner._run_agent_queued_followup(
            turn_ctx, adapter=None, pending="follow-up", pending_event=pending_event,
            response="", result={"interrupted": True, "messages": []}, stream_task=None,
        )
    _, silent, _ = await runner._hmwa_shape_agent_response(
        result, opener, [], SimpleNamespace(session_id="sid"), None,
        None, 1, "sid", "telegram", 0,
    )
    assert silent is terminal_enabled


@pytest.mark.asyncio
async def test_empty_success_still_gets_empty_response_warning(monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(return_value={
        "final_response": "",
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": ""},
        ],
        "tools": [],
        "history_offset": 0,
        "last_prompt_tokens": 0,
        "api_calls": 1,
        "failed": False,
    })

    response = await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    assert "no response was generated" in response


@pytest.mark.asyncio
async def test_prose_mentioning_silence_token_is_delivered(monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path)
    text = "Use [SILENT] when no answer is needed."
    runner._run_agent = AsyncMock(return_value={
        "final_response": text,
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": text},
        ],
        "tools": [],
        "history_offset": 0,
        "last_prompt_tokens": 0,
        "api_calls": 1,
        "failed": False,
    })

    response = await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    assert response == text


@pytest.mark.asyncio
async def test_agent_end_hook_includes_model_and_provider(monkeypatch, tmp_path):
    """Gateway hooks receive the actual model/provider for post-turn routing."""
    runner = _runner(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(return_value={
        "final_response": "done",
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "done"},
        ],
        "tools": [],
        "history_offset": 0,
        "last_prompt_tokens": 0,
        "api_calls": 1,
        "failed": False,
        "model": "gpt-5.6-terra",
        "provider": "openai-codex",
    })

    await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    end_context = next(
        call.args[1]
        for call in runner.hooks.emit.await_args_list
        if call.args[0] == "agent:end"
    )
    assert end_context["model"] == "gpt-5.6-terra"
    assert end_context["provider"] == "openai-codex"
