"""Tests for /bg gateway slash command.

Tests the _handle_background_command handler (run a prompt in a separate
background session) across gateway messenger platforms.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource


def _make_event(text="/bg", platform=Platform.TELEGRAM,
                user_id="12345", chat_id="67890"):
    """Build a MessageEvent for testing."""
    source = SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        user_name="testuser",
    )
    return MessageEvent(text=text, source=source)


def _make_runner():
    """Create a bare GatewayRunner with minimal mocks."""
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner.config = GatewayConfig()
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._running_agents = {}
    runner._background_tasks = set()

    mock_store = MagicMock()
    # A real SessionStore returns None when no persisted /model override exists.
    # MagicMock's default truthy return would otherwise rehydrate a fake model
    # and make the session-scoped reasoning resolver receive a MagicMock.
    mock_store.get_model_override.return_value = None
    runner.session_store = mock_store

    from gateway.hooks import HookRegistry
    runner.hooks = HookRegistry()

    return runner


# ---------------------------------------------------------------------------
# _handle_background_command
# ---------------------------------------------------------------------------


class TestHandleBackgroundCommand:
    """Tests for GatewayRunner._handle_background_command."""

    @pytest.mark.asyncio
    async def test_no_prompt_shows_usage(self):
        """Running /bg with no prompt shows usage."""
        runner = _make_runner()
        event = _make_event(text="/bg")
        result = await runner._handle_background_command(event)
        assert "Usage:" in result
        assert "/bg" in result

    @pytest.mark.asyncio
    async def test_empty_prompt_shows_usage(self):
        """Running /bg with only whitespace shows usage."""
        runner = _make_runner()
        event = _make_event(text="/bg   ")
        result = await runner._handle_background_command(event)
        assert "Usage:" in result

    @pytest.mark.asyncio
    async def test_passes_parent_and_complete_reply_context(self):
        runner = _make_runner()
        event = _make_event(text="/bg inspect this")
        parent_key = runner._session_key_for_source(event.source)
        parent = MagicMock(session_id="parent-session", session_key=parent_key)
        async_store = AsyncMock()
        async_store._store = runner.session_store
        async_store.get_or_create_session.return_value = parent
        async_store.load_transcript.return_value = [
            {"role": "user", "content": "Original parent request"},
            {"role": "assistant", "content": "Original parent response"},
        ]
        runner._async_session_store = async_store
        runner._run_background_task = AsyncMock(return_value=None)
        decision_after_old_limit = "APPROVE_THE_REPORT_AFTER_CHARACTER_500"
        event.reply_to_text = "q" * 700 + decision_after_old_limit
        event.reply_to_is_own_message = True
        event.auto_skill = ["arbitrary-topic-skill"]
        event.channel_prompt = "ARBITRARY TOPIC RULE"
        event.channel_context = "SECONDARY TOPIC CONTEXT"

        result = await runner._handle_background_command(event)
        await asyncio.sleep(0)

        assert "Background" in result
        kwargs = runner._run_background_task.await_args.kwargs
        assert kwargs["parent_session_id"] == "parent-session"
        assert kwargs["parent_session_key"] == runner._session_key_for_source(event.source)
        assert kwargs["parent_conversation_history"] == [
            {"role": "user", "content": "Original parent request"},
            {"role": "assistant", "content": "Original parent response"},
        ]
        assert kwargs["reply_to_text"] == event.reply_to_text
        assert decision_after_old_limit in kwargs["reply_to_text"]
        assert kwargs["reply_to_is_own_message"] is True
        assert kwargs["auto_skill"] == ["arbitrary-topic-skill"]
        assert kwargs["channel_prompt"] == "ARBITRARY TOPIC RULE"
        assert kwargs["internal_context"] == {
            "channel_context": "SECONDARY TOPIC CONTEXT",
        }
        assert kwargs["origin"]["execution_kind"] == "user_explicit_background"
        assert kwargs["origin"]["user_initiated"] is True
        assert kwargs["origin"]["command"] == "/bg"

    @pytest.mark.asyncio
    async def test_refuses_parent_from_another_topic(self):
        runner = _make_runner()
        event = _make_event(text="/bg inspect this")
        event.source.thread_id = "expected-topic"
        async_store = AsyncMock()
        async_store._store = runner.session_store
        async_store.get_or_create_session.return_value = MagicMock(
            session_id="other-topic-session",
            session_key="telegram:67890:other-topic",
        )
        runner._async_session_store = async_store
        runner._run_background_task = AsyncMock(return_value=None)

        result = await runner._handle_background_command(event)

        assert "not started" in result.lower()
        async_store.load_transcript.assert_not_called()
        runner._run_background_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_parent_read_failure_is_explicit(self):
        runner = _make_runner()
        event = _make_event(text="/bg inspect this")
        parent_key = runner._session_key_for_source(event.source)
        async_store = AsyncMock()
        async_store._store = runner.session_store
        async_store.get_or_create_session.return_value = MagicMock(
            session_id="parent-session", session_key=parent_key
        )
        async_store.load_transcript.side_effect = RuntimeError("database unreadable")
        runner._async_session_store = async_store
        runner._run_background_task = AsyncMock(return_value=None)

        result = await runner._handle_background_command(event)

        assert "not started" in result.lower()
        assert "could not be read" in result.lower()
        runner._run_background_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_long_parent_snapshots_only_native_compression_tip(self, tmp_path):
        from gateway.session import AsyncSessionStore, SessionStore
        from hermes_state import SessionDB

        runner = _make_runner()
        store = SessionStore(tmp_path / "sessions", GatewayConfig())
        if store._db is not None:
            store._db.close()
        store._db = SessionDB(db_path=tmp_path / "state.db")
        runner.session_store = store
        runner._async_session_store = AsyncSessionStore(store)
        runner._run_background_task = AsyncMock(return_value=None)
        event = _make_event(text="/bg current instruction")
        try:
            entry = store.get_or_create_session(event.source)
            old_history = [
                {"role": "user", "content": f"old request {i}"}
                if i % 2 == 0
                else {"role": "assistant", "content": f"old answer {i}"}
                for i in range(240)
            ]
            store._db.append_messages_batch(entry.session_id, old_history)
            store._db.end_session(entry.session_id, "compression")
            tip_id = "compressed-parent-tip"
            store._db.create_session(
                tip_id,
                source="telegram",
                parent_session_id=entry.session_id,
            )
            store._db.append_messages_batch(
                tip_id,
                [
                    {"role": "user", "content": "compressed summary"},
                    {"role": "assistant", "content": "recent context"},
                ],
            )

            result = await runner._handle_background_command(event)
            await asyncio.sleep(0)

            assert "Background" in result
            snapshot = runner._run_background_task.await_args.kwargs[
                "parent_conversation_history"
            ]
            assert [message["content"] for message in snapshot] == [
                "compressed summary",
                "recent context",
            ]
        finally:
            store._db.close()

    @pytest.mark.asyncio
    async def test_parent_snapshot_isolated_between_real_topics(self, tmp_path):
        from gateway.session import AsyncSessionStore, SessionStore
        from hermes_state import SessionDB

        runner = _make_runner()
        store = SessionStore(tmp_path / "sessions", GatewayConfig())
        if store._db is not None:
            store._db.close()
        store._db = SessionDB(db_path=tmp_path / "state.db")
        runner.session_store = store
        runner._async_session_store = AsyncSessionStore(store)
        runner._run_background_task = AsyncMock(return_value=None)

        event_a = _make_event(text="/bg inspect this")
        event_a.source.thread_id = "topic-a"
        event_b = _make_event(text="ordinary message")
        event_b.source.thread_id = "topic-b"
        try:
            entry_a = store.get_or_create_session(event_a.source)
            entry_b = store.get_or_create_session(event_b.source)
            store._db.append_messages_batch(
                entry_a.session_id,
                [
                    {"role": "user", "content": "TOPIC_A_SENTINEL"},
                    {"role": "assistant", "content": "topic A answer"},
                ],
            )
            store._db.append_messages_batch(
                entry_b.session_id,
                [
                    {"role": "user", "content": "TOPIC_B_SENTINEL"},
                    {"role": "assistant", "content": "topic B answer"},
                ],
            )

            result = await runner._handle_background_command(event_a)
            await asyncio.sleep(0)

            assert "Background" in result
            snapshot = runner._run_background_task.await_args.kwargs[
                "parent_conversation_history"
            ]
            snapshot_text = json.dumps(snapshot)
            assert "TOPIC_A_SENTINEL" in snapshot_text
            assert "TOPIC_B_SENTINEL" not in snapshot_text
        finally:
            store._db.close()

    def test_bg_resolves_to_gateway_command(self):
        from hermes_cli.commands import resolve_command

        command = resolve_command("bg")
        assert command is not None
        assert command.name == "bg"
        assert command.busy_policy == "dispatch"


def test_auto_skill_loader_is_shared_and_generic(tmp_path):
    from gateway.run import _build_auto_skill_context

    payload = ({"name": "arbitrary-topic-skill"}, tmp_path, "arbitrary-topic-skill")
    with patch("agent.skill_commands._load_skill_payload", return_value=payload) as load, \
         patch("agent.skill_commands._build_skill_message", return_value="ARBITRARY SKILL PAYLOAD"):
        context, names = _build_auto_skill_context(
            ["arbitrary-topic-skill"],
            task_id="bg-test",
        )

    load.assert_called_once_with("arbitrary-topic-skill", task_id="bg-test")
    assert context == "ARBITRARY SKILL PAYLOAD"
    assert names == ["arbitrary-topic-skill"]


def test_background_parent_snapshot_respects_model_budget_and_priority_boundary():
    from agent.model_metadata import estimate_messages_tokens_rough
    from gateway.run import _bounded_background_parent_history

    recent_sentinel = "RECENT_PARENT_SENTINEL"
    history = [
        {"role": "user", "content": "u" * 30_000 + recent_sentinel},
        {"role": "assistant", "content": "a" * 30_000 + "RECENT_ANSWER_SENTINEL"},
        {"role": "user", "content": "stale unfinished instruction"},
    ]

    snapshot = _bounded_background_parent_history(history, context_length=4_096)

    assert estimate_messages_tokens_rough(snapshot) <= 1_024
    assert [message["role"] for message in snapshot] == ["user", "assistant"]
    assert recent_sentinel in snapshot[0]["content"]
    assert "RECENT_ANSWER_SENTINEL" in snapshot[1]["content"]
    assert "stale unfinished instruction" not in json.dumps(snapshot)


@pytest.mark.parametrize("oversized_index", range(4))
def test_background_parent_snapshot_never_splits_tool_turn(oversized_index):
    from gateway.run import _bounded_background_parent_history

    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"key":"value"}'},
        }
    ]
    turn = [
        {"role": "user", "content": "look this up", "api_content": "USER_SIDECAR"},
        {"role": "assistant", "content": "", "tool_calls": tool_calls},
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "lookup",
            "content": "TOOL_RESULT",
            "api_content": "TOOL_SIDECAR",
        },
        {"role": "assistant", "content": "final answer", "api_content": "ASSISTANT_SIDECAR"},
    ]
    if oversized_index == 0:
        turn[0]["content"] += "x" * 30_000
    elif oversized_index == 1:
        turn[1]["content"] = "x" * 30_000
    elif oversized_index == 2:
        turn[2]["content"] += "x" * 30_000
    else:
        turn[3]["content"] += "x" * 30_000

    snapshot = _bounded_background_parent_history(turn, context_length=4_096)
    roles = [message["role"] for message in snapshot]
    has_tool_call = any(message.get("tool_calls") for message in snapshot)
    has_tool_result = "tool" in roles

    assert roles[0] == "user"
    assert roles[-1] == "assistant"
    assert has_tool_call is has_tool_result
    if has_tool_call:
        assert snapshot == turn
        assert snapshot is not turn
        assert snapshot[0] is not turn[0]


# ---------------------------------------------------------------------------
# _run_background_task
# ---------------------------------------------------------------------------


class TestRunBackgroundTask:
    """Tests for GatewayRunner._run_background_task (the actual execution)."""


    @pytest.mark.asyncio
    async def test_no_credentials_sends_error(self):
        """When provider credentials are missing, an error is sent."""
        runner = _make_runner()
        mock_adapter = AsyncMock()
        mock_adapter.send = AsyncMock()
        runner.adapters[Platform.TELEGRAM] = mock_adapter

        source = SessionSource(
            platform=Platform.TELEGRAM,
            user_id="12345",
            chat_id="67890",
            user_name="testuser",
        )

        with patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": None}):
            await runner._run_background_task("test prompt", source, "bg_test")

        # Should have sent an error message
        mock_adapter.send.assert_called_once()
        call_args = mock_adapter.send.call_args
        assert "failed" in call_args[1].get("content", call_args[0][1] if len(call_args[0]) > 1 else "").lower()

    @pytest.mark.asyncio
    async def test_successful_task_sends_result(self):
        """When the agent completes successfully, the result is sent."""
        runner = _make_runner()
        mock_adapter = MagicMock()
        mock_adapter.send = AsyncMock()
        mock_adapter.extract_media = MagicMock(return_value=([], "Hello from background!"))
        mock_adapter.extract_images = MagicMock(return_value=([], "Hello from background!"))
        runner.adapters[Platform.TELEGRAM] = mock_adapter
        parent_history = [
            {"role": "user", "content": "Original parent request"},
            {"role": "assistant", "content": "Original parent response"},
        ]
        child_history = [dict(message) for message in parent_history]

        source = SessionSource(
            platform=Platform.TELEGRAM,
            user_id="12345",
            chat_id="67890",
            user_name="testuser",
        )

        mock_result = {"final_response": "Hello from background!", "messages": []}

        checkpoint_config = {
            "checkpoints": {
                "enabled": True,
                "max_snapshots": 8,
                "max_total_size_mb": 222,
                "max_file_size_mb": 3,
            }
        }
        with patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "test-key"}), \
             patch("gateway.run._load_gateway_config", return_value=checkpoint_config), \
             patch(
                 "gateway.run._build_auto_skill_context",
                 return_value=("ARBITRARY SKILL PAYLOAD", ["arbitrary-topic-skill"]),
             ), \
             patch(
                 "gateway.run.build_session_context",
                 return_value=MagicMock(),
             ), \
             patch(
                 "gateway.run.build_session_context_prompt",
                 return_value="ARBITRARY SOURCE CONTEXT",
             ), \
             patch.object(
                 runner,
                 "_get_system_prompt_for_channel",
                 return_value="ARBITRARY CONFIGURED CHANNEL OVERRIDE",
             ), \
             patch("run_agent.AIAgent") as MockAgent:
            mock_agent_instance = MagicMock()
            mock_agent_instance.shutdown_memory_provider = MagicMock()
            mock_agent_instance.close = MagicMock()
            mock_agent_instance._session_db = MagicMock()
            mock_agent_instance._session_db.get_messages_as_conversation.side_effect = [
                [],
                child_history,
            ]
            mock_agent_instance.run_conversation.return_value = mock_result
            MockAgent.return_value = mock_agent_instance

            decision_after_old_limit = "APPROVE_THE_REPORT_AFTER_CHARACTER_500"
            complete_reply = "q" * 700 + decision_after_old_limit
            await runner._run_background_task(
                "Ok, je valide pour Amandine",
                source,
                "bg_test",
                parent_session_id="parent-session",
                parent_session_key="telegram:67890",
                parent_conversation_history=parent_history,
                reply_to_text=complete_reply,
                reply_to_is_own_message=True,
                auto_skill=["arbitrary-topic-skill"],
                channel_prompt="ARBITRARY TOPIC RULE",
                internal_context={"channel_context": "SECONDARY TOPIC CONTEXT"},
                origin={
                    "execution_kind": "user_explicit_background",
                    "user_initiated": True,
                    "command": "/bg",
                },
            )

        # Should have sent the result
        mock_adapter.send.assert_called_once()
        call_args = mock_adapter.send.call_args
        content = call_args[1].get("content", call_args[0][1] if len(call_args[0]) > 1 else "")
        assert "Background task complete" in content
        assert "Hello from background!" in content
        agent_kwargs = MockAgent.call_args.kwargs
        assert agent_kwargs["checkpoints_enabled"] is True
        assert agent_kwargs["checkpoint_max_snapshots"] == 8
        assert agent_kwargs["checkpoint_max_total_size_mb"] == 222
        assert agent_kwargs["checkpoint_max_file_size_mb"] == 3
        assert agent_kwargs["parent_session_id"] == "parent-session"
        assert agent_kwargs.get("gateway_session_key") is None
        run_kwargs = mock_agent_instance.run_conversation.call_args.kwargs
        assert run_kwargs["conversation_history"] == parent_history
        assert run_kwargs["conversation_history"] is not parent_history
        assert run_kwargs["conversation_history"][0] is not parent_history[0]
        assert run_kwargs["current_user_text"] == "Ok, je valide pour Amandine"
        assert run_kwargs["reply_to_text"] == complete_reply
        assert run_kwargs["internal_context"] == {
            "auto_skill": ["arbitrary-topic-skill"],
            "channel_context": "SECONDARY TOPIC CONTEXT",
        }
        assert "ARBITRARY TOPIC RULE" in run_kwargs["system_message"]
        assert run_kwargs["system_message"].endswith(
            "ARBITRARY CONFIGURED CHANNEL OVERRIDE"
        )
        assert run_kwargs["user_message"].endswith("Ok, je valide pour Amandine")
        assert all(
            message.get("content") != "Ok, je valide pour Amandine"
            for message in run_kwargs["conversation_history"]
        )
        assert "SECONDARY TOPIC CONTEXT\n\n[New message]" in run_kwargs["user_message"]
        assert "ARBITRARY SKILL PAYLOAD" in run_kwargs["user_message"]
        assert f'[Replying to your previous message: "{complete_reply}"]' in run_kwargs["user_message"]
        assert decision_after_old_limit in run_kwargs["user_message"]
        record_kwargs = mock_agent_instance.record_gateway_session_peer.call_args.kwargs
        assert record_kwargs["routable"] is False
        origin = record_kwargs["origin"]
        assert origin["execution_kind"] == "user_explicit_background"
        assert origin["user_initiated"] is True
        assert origin["command"] == "/bg"
        mock_agent_instance._session_db.append_messages_batch.assert_called_once_with(
            "bg_test", parent_history
        )
        mock_agent_instance.shutdown_memory_provider.assert_called_once()
        mock_agent_instance.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_parent_tool_pair_survives_snapshot_reload_until_first_run(self, tmp_path):
        from hermes_state import SessionDB

        runner = _make_runner()
        adapter = MagicMock()
        adapter.send = AsyncMock()
        adapter.extract_media = MagicMock(return_value=([], "done"))
        adapter.extract_images = MagicMock(return_value=([], "done"))
        runner.adapters[Platform.TELEGRAM] = adapter
        db = SessionDB(db_path=tmp_path / "state.db")
        runner._session_db = db
        task_id = "bg_tool_pair"
        db.create_session("parent-session", source="telegram")
        db.create_session(task_id, source="telegram", parent_session_id="parent-session")
        tool_calls = [
            {
                "id": "call_weather_1",
                "type": "function",
                "function": {
                    "name": "web_search",
                    "arguments": '{"query":"weather Paris"}',
                },
            }
        ]
        parent_history = [
            {"role": "user", "content": "Quel temps fait-il à Paris ?"},
            {"role": "assistant", "content": "", "tool_calls": tool_calls},
            {
                "role": "tool",
                "tool_call_id": "call_weather_1",
                "name": "web_search",
                "content": '{"temperature_c":21}',
            },
            {"role": "assistant", "content": "Il fait 21 °C à Paris."},
        ]
        agent = MagicMock()
        agent._session_db = db
        agent.run_conversation.return_value = {
            "final_response": "done",
            "messages": [],
        }

        try:
            with patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "test-key"}), \
                 patch("gateway.run._load_gateway_config", return_value={}), \
                 patch("run_agent.AIAgent", return_value=agent):
                await runner._run_background_task(
                    "Résume la réponse précédente.",
                    _make_event().source,
                    task_id,
                    parent_session_id="parent-session",
                    parent_session_key="telegram:67890",
                    parent_conversation_history=parent_history,
                )

            reloaded = agent.run_conversation.call_args.kwargs["conversation_history"]
            assert [message["role"] for message in reloaded] == [
                "user",
                "assistant",
                "tool",
                "assistant",
            ]
            assert reloaded[1]["tool_calls"] == tool_calls
            assert reloaded[2]["tool_call_id"] == tool_calls[0]["id"]
            assert reloaded[2]["content"] == '{"temperature_c":21}'
            assert agent.run_conversation.call_args.kwargs["user_message"].endswith(
                "Résume la réponse précédente."
            )
        finally:
            db.close()

    @pytest.mark.asyncio
    async def test_parent_context_available_without_session_search_and_new_instruction_wins(
        self, tmp_path
    ):
        from hermes_state import SessionDB

        runner = _make_runner()
        adapter = MagicMock()
        adapter.send = AsyncMock()
        adapter.extract_media = MagicMock(return_value=([], "done"))
        adapter.extract_images = MagicMock(return_value=([], "done"))
        runner.adapters[Platform.TELEGRAM] = adapter
        db = SessionDB(db_path=tmp_path / "state.db")
        runner._session_db = db
        task_id = "bg_without_session_search"
        db.create_session("parent-session", source="telegram")
        db.create_session(task_id, source="telegram", parent_session_id="parent-session")
        parent_sentinel = "PARENT_FACT_AVAILABLE_WITHOUT_SEARCH"
        current_instruction = "CURRENT_CHILD_INSTRUCTION_WINS"
        parent_history = [
            {"role": "user", "content": "OLD_CONTRADICTORY_INSTRUCTION"},
            {"role": "assistant", "content": parent_sentinel},
            {"role": "user", "content": "unfinished parent input must be dropped"},
        ]
        agent = MagicMock()
        agent._session_db = db
        agent.context_compressor = MagicMock(context_length=8_192)
        agent.run_conversation.return_value = {
            "final_response": "done",
            "messages": [],
        }

        try:
            with patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "test-key"}), \
                 patch("gateway.run._load_gateway_config", return_value={}), \
                 patch.object(runner, "_resolve_enabled_toolsets_for_source", return_value=[]), \
                 patch("run_agent.AIAgent", return_value=agent) as MockAgent:
                await runner._run_background_task(
                    current_instruction,
                    _make_event().source,
                    task_id,
                    parent_session_id="parent-session",
                    parent_session_key="telegram:67890",
                    parent_conversation_history=parent_history,
                )

            run_kwargs = agent.run_conversation.call_args.kwargs
            history = run_kwargs["conversation_history"]
            assert MockAgent.call_args.kwargs["enabled_toolsets"] == []
            assert parent_sentinel in json.dumps(history)
            assert "unfinished parent input must be dropped" not in json.dumps(history)
            assert history[-1]["role"] == "assistant"
            assert current_instruction not in json.dumps(history)
            assert run_kwargs["user_message"].count(current_instruction) == 1
            assert run_kwargs["user_message"].endswith(current_instruction)
            assert run_kwargs["current_user_text"] == current_instruction
        finally:
            db.close()

    @pytest.mark.asyncio
    async def test_parent_snapshot_persistence_is_idempotent_for_same_task_id(self, tmp_path):
        from hermes_state import SessionDB

        runner = _make_runner()
        adapter = MagicMock()
        adapter.send = AsyncMock()
        adapter.extract_media = MagicMock(return_value=([], "done"))
        adapter.extract_images = MagicMock(return_value=([], "done"))
        runner.adapters[Platform.TELEGRAM] = adapter
        db = SessionDB(db_path=tmp_path / "state.db")
        runner._session_db = db
        task_id = "bg_redelivery"
        db.create_session("parent-session", source="telegram")
        db.create_session(task_id, source="telegram", parent_session_id="parent-session")
        parent_history = [
            {"role": "user", "content": "Parent request"},
            {"role": "assistant", "content": "Parent response"},
        ]
        agent = MagicMock()
        agent._session_db = db

        def _persist_child_turn(**kwargs):
            db.append_messages_batch(
                task_id,
                [
                    {"role": "user", "content": kwargs["user_message"]},
                    {"role": "assistant", "content": "done"},
                ],
            )
            return {"final_response": "done", "messages": []}

        agent.run_conversation.side_effect = _persist_child_turn

        try:
            with patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "test-key"}), \
                 patch("gateway.run._load_gateway_config", return_value={}), \
                 patch("run_agent.AIAgent", return_value=agent):
                for _ in range(2):
                    await runner._run_background_task(
                        "Do the detached work",
                        _make_event().source,
                        task_id,
                        parent_session_id="parent-session",
                        parent_session_key="telegram:67890",
                        parent_conversation_history=parent_history,
                    )

            persisted = db.get_messages_as_conversation(task_id)
            persisted_contents = [message["content"] for message in persisted]
            assert persisted_contents.count("Parent request") == 1
            assert persisted_contents.count("Parent response") == 1
            assert len(agent.run_conversation.call_args_list) == 2
            second_history_contents = [
                message["content"]
                for message in agent.run_conversation.call_args_list[1].kwargs[
                    "conversation_history"
                ]
            ]
            assert second_history_contents.count("Parent request") == 1
            assert second_history_contents.count("Parent response") == 1
            assert second_history_contents.count("Do the detached work") == 1
        finally:
            db.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("message_type", "media_type", "path", "prepared_marker"),
        [
            (MessageType.AUDIO, "audio/mpeg", "/cache/user-song.mp3", "audio file attachment"),
            (MessageType.DOCUMENT, "application/pdf", "/cache/report.pdf", "user sent a document"),
        ],
    )
    async def test_background_non_image_media_reuses_normal_inbound_preprocessor(
        self, message_type, media_type, path, prepared_marker
    ):
        runner = _make_runner()
        adapter = MagicMock()
        adapter.send = AsyncMock()
        runner.adapters[Platform.TELEGRAM] = adapter
        source = _make_event().source
        normal_preprocessor = runner._prepare_profile_scoped_inbound_message_text
        runner._prepare_profile_scoped_inbound_message_text = AsyncMock(
            wraps=normal_preprocessor
        )

        with patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "test-key"}), \
             patch("gateway.run._load_gateway_config", return_value={}), \
             patch("run_agent.AIAgent") as MockAgent:
            agent = MockAgent.return_value
            agent.run_conversation.return_value = {"final_response": "done", "messages": []}
            await runner._run_background_task(
                "inspect attachment",
                source,
                "bg_media",
                media_urls=[path],
                media_types=[media_type],
                message_type=message_type,
            )

        prepared_event = runner._prepare_profile_scoped_inbound_message_text.await_args.kwargs["event"]
        assert prepared_event.media_urls == [path]
        assert prepared_event.media_types == [media_type]
        assert prepared_event.message_type == message_type
        user_message = agent.run_conversation.call_args.kwargs["user_message"]
        assert agent.run_conversation.call_args.kwargs["conversation_history"] is None
        assert prepared_marker in user_message.lower()
        assert path in user_message

    @pytest.mark.asyncio
    async def test_missing_declared_parent_fails_instead_of_starting_fresh(self):
        runner = _make_runner()
        adapter = MagicMock()
        adapter.send = AsyncMock()
        runner.adapters[Platform.TELEGRAM] = adapter
        runner._session_db = MagicMock()
        runner._session_db._db = runner._session_db
        runner._session_db.get_session.return_value = None

        with patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "test-key"}), \
             patch("gateway.run._load_gateway_config", return_value={}), \
             patch("run_agent.AIAgent") as MockAgent:
            await runner._run_background_task(
                "inspect parent context",
                _make_event().source,
                "bg_missing_parent",
                parent_session_id="missing-parent",
                parent_session_key="telegram:67890",
            )

        MockAgent.assert_not_called()
        assert "parent context snapshot" in adapter.send.call_args.kwargs["content"].lower()

    @pytest.mark.asyncio
    async def test_generic_caller_supplies_its_own_provenance(self):
        runner = _make_runner()
        mock_adapter = MagicMock()
        mock_adapter.send = AsyncMock()
        mock_adapter.extract_media = MagicMock(return_value=([], "done"))
        mock_adapter.extract_images = MagicMock(return_value=([], "done"))
        runner.adapters[Platform.TELEGRAM] = mock_adapter
        source = _make_event().source
        caller_origin = {"execution_kind": "scheduled_background"}

        with patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "test-key"}), \
             patch("gateway.run._load_gateway_config", return_value={}), \
             patch("run_agent.AIAgent") as MockAgent:
            agent = MockAgent.return_value
            agent._session_db = MagicMock()
            agent._session_db.get_messages_as_conversation.return_value = []
            agent.run_conversation.return_value = {"final_response": "done", "messages": []}
            await runner._run_background_task(
                "scheduled work",
                source,
                "bg_other",
                parent_session_id="parent-session",
                parent_session_key="telegram:67890",
                parent_conversation_history=[],
                origin=caller_origin,
            )

        assert agent.record_gateway_session_peer.call_args.kwargs == {
            "origin": caller_origin,
            "routable": False,
        }
        assert "command" not in caller_origin

    @pytest.mark.asyncio
    async def test_peer_record_failure_still_cleans_up_agent(self):
        runner = _make_runner()
        mock_adapter = MagicMock()
        mock_adapter.send = AsyncMock()
        runner.adapters[Platform.TELEGRAM] = mock_adapter
        source = _make_event().source

        with patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "test-key"}), \
             patch("gateway.run._load_gateway_config", return_value={}), \
             patch("run_agent.AIAgent") as MockAgent:
            agent = MockAgent.return_value
            agent.record_gateway_session_peer.side_effect = RuntimeError(
                "Session DB unavailable; gateway peer was not recorded"
            )

            await runner._run_background_task(
                "scheduled work",
                source,
                "bg_failure",
                parent_session_id="parent-session",
                parent_session_key="telegram:67890",
                parent_conversation_history=[],
                origin={"execution_kind": "scheduled_background"},
            )

        agent.run_conversation.assert_not_called()
        agent.shutdown_memory_provider.assert_called_once()
        agent.close.assert_called_once()
        assert "Session DB unavailable" in mock_adapter.send.call_args.kwargs["content"]


class TestAIAgentGatewayPeerContract:
    def test_records_peer_through_public_contract(self):
        from run_agent import AIAgent

        agent = object.__new__(AIAgent)
        agent.session_id = "bg_test"
        agent.platform = "telegram"
        agent._session_db = MagicMock()
        agent._ensure_db_session = MagicMock()
        agent._user_id = "user"
        agent._gateway_session_key = "telegram:chat"
        agent._chat_id = "chat"
        agent._chat_type = "group"
        agent._thread_id = "topic"
        agent._chat_name = "Test chat"
        origin = {"execution_kind": "user_explicit_background"}

        agent.record_gateway_session_peer(origin=origin)

        agent._ensure_db_session.assert_called_once_with()
        record_kwargs = agent._session_db.record_gateway_session_peer.call_args.kwargs
        assert record_kwargs == {
            "source": "telegram",
            "user_id": "user",
            "session_key": "telegram:chat",
            "chat_id": "chat",
            "chat_type": "group",
            "thread_id": "topic",
            "display_name": "Test chat",
            "origin_json": json.dumps(origin),
            "routable": True,
        }
        record_args = agent._session_db.record_gateway_session_peer.call_args.args
        assert record_args == ("bg_test",)

    def test_missing_session_db_fails_explicitly(self):
        from run_agent import AIAgent

        agent = object.__new__(AIAgent)
        agent.session_id = "bg_test"
        agent._session_db = None
        agent._ensure_db_session = MagicMock()

        with pytest.raises(RuntimeError, match="Session DB unavailable"):
            agent.record_gateway_session_peer(origin={})

    def test_missing_gateway_session_key_fails_explicitly(self):
        from run_agent import AIAgent

        agent = object.__new__(AIAgent)
        agent.session_id = "bg_test"
        agent._session_db = MagicMock()
        agent._ensure_db_session = MagicMock()
        agent._gateway_session_key = ""

        with pytest.raises(RuntimeError, match="Gateway session key missing"):
            agent.record_gateway_session_peer(origin={})

        agent._session_db.record_gateway_session_peer.assert_not_called()

    def test_unrouted_child_records_peer_without_a_routing_key(self):
        from run_agent import AIAgent

        agent = object.__new__(AIAgent)
        agent.session_id = "bg_test"
        agent.platform = "telegram"
        agent._session_db = MagicMock()
        agent._ensure_db_session = MagicMock()
        agent._user_id = "user"
        agent._gateway_session_key = None
        agent._chat_id = "chat"
        agent._chat_type = "private"
        agent._thread_id = "topic"
        agent._chat_name = "Test chat"

        agent.record_gateway_session_peer(
            origin={"execution_kind": "user_explicit_background"},
            routable=False,
        )

        kwargs = agent._session_db.record_gateway_session_peer.call_args.kwargs
        assert kwargs["session_key"] is None
        assert kwargs["routable"] is False
        assert kwargs["chat_id"] == "chat"
        assert kwargs["thread_id"] == "topic"

    def test_unrouted_child_cannot_replace_parent_in_gateway_recovery(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        try:
            peer = {
                "source": "telegram",
                "user_id": "user",
                "chat_id": "chat",
                "chat_type": "private",
                "thread_id": "topic",
                "display_name": "Test chat",
            }
            db.create_session("parent", source="telegram")
            db.record_gateway_session_peer(
                "parent",
                session_key="telegram:chat:topic",
                origin_json="{}",
                **peer,
            )
            db.create_session(
                "bg_child",
                source="telegram",
                parent_session_id="parent",
            )
            db.record_gateway_session_peer(
                "bg_child",
                session_key=None,
                routable=False,
                origin_json=json.dumps(
                    {"execution_kind": "user_explicit_background"}
                ),
                **peer,
            )

            child = db.get_session("bg_child")
            assert child is not None
            assert child["parent_session_id"] == "parent"
            assert child["chat_id"] == "chat"
            assert child["thread_id"] == "topic"
            assert json.loads(child["origin_json"])["execution_kind"] == "user_explicit_background"
            db.append_message("parent", role="user", content="parent turn")
            db.append_message("bg_child", role="user", content="newer detached turn")
            db.end_session("bg_child", "session_reset")

            recovered = db.find_latest_gateway_session_for_peer(
                source="telegram",
                user_id="user",
                session_key="telegram:chat:topic",
                chat_id="chat",
                chat_type="private",
                thread_id="topic",
            )
            assert recovered is not None
            assert recovered["id"] == "parent"

            fallback_recovered = db.find_latest_gateway_session_for_peer(
                source="telegram",
                user_id="user",
                session_key="missing:exact:key",
                chat_id="chat",
                chat_type="private",
                thread_id="topic",
            )
            assert fallback_recovered is not None
            assert fallback_recovered["id"] == "parent"
        finally:
            db.close()


@pytest.mark.asyncio
async def test_immediate_voice_sets_normalized_current_user_text_from_spoken_words():
    runner = _make_runner()
    source = _make_event().source
    event = MessageEvent(
        text="",
        source=source,
        message_type=MessageType.VOICE,
        media_urls=["/cache/voice.ogg"],
        media_types=["audio/ogg"],
        reply_to_message_id="42",
        reply_to_text="INTERNAL REPLY CONTEXT",
    )
    runner._enrich_message_with_transcription = AsyncMock(
        return_value=("[Voice transcription]: ship the report", ["ship the report"])
    )

    await runner._prepare_inbound_message_text(event=event, source=source, history=[])

    assert event._gateway_current_user_text == "ship the report"
    assert "INTERNAL REPLY CONTEXT" not in event._gateway_current_user_text


@pytest.mark.asyncio
async def test_queued_voice_uses_cached_spoken_words_as_current_user_text():
    runner = _make_runner()
    source = _make_event().source
    event = MessageEvent(
        text="",
        source=source,
        message_type=MessageType.VOICE,
        media_urls=["/cache/voice.ogg"],
        media_types=["audio/ogg"],
    )
    event._gateway_pending_stt_text = "[Voice transcription]: queue this report"
    event._gateway_pending_stt_transcripts = ["queue this report"]

    await runner._prepare_inbound_message_text(event=event, source=source, history=[])

    assert event._gateway_current_user_text == "queue this report"


# ---------------------------------------------------------------------------
# /bg in help and known_commands
# ---------------------------------------------------------------------------


class TestBackgroundInHelp:
    """Verify /bg and /btw appear in help text and known commands."""

    @pytest.mark.asyncio
    async def test_bg_and_btw_in_help_output(self):
        """The /help output includes /bg and /btw."""
        runner = _make_runner()
        event = _make_event(text="/help")
        result = await runner._handle_help_command(event)
        assert "/bg" in result
        assert "/btw" in result


# ---------------------------------------------------------------------------
# CLI /bg command definition
# ---------------------------------------------------------------------------


class TestBackgroundInCLICommands:
    """Verify /bg and /btw are registered in the CLI command system."""


    def test_bg_autocompletes(self):
        """The /bg and /btw commands appear in autocomplete results."""
        pytest.importorskip("prompt_toolkit")
        from hermes_cli.commands import SlashCommandCompleter
        from prompt_toolkit.document import Document

        completer = SlashCommandCompleter()
        doc = Document("bg")  # Partial match
        completions = list(completer.get_completions(doc, None))
        # Text doesn't start with / so no completions
        assert len(completions) == 0

        doc = Document("/bg")  # With slash prefix
        completions = list(completer.get_completions(doc, None))
        cmd_displays = [str(c.display) for c in completions]
        assert any("/bg" in d for d in cmd_displays)

        doc = Document("/btw")
        completions = list(completer.get_completions(doc, None))
        cmd_displays = [str(c.display) for c in completions]
        assert any("/btw" in d for d in cmd_displays)


# ---------------------------------------------------------------------------
# _handle_btw_command
# ---------------------------------------------------------------------------


class TestHandleBtwCommand:
    """Tests for GatewayRunner._handle_btw_command (context-aware side question)."""

    @pytest.mark.asyncio
    async def test_no_question_shows_usage(self):
        runner = _make_runner()
        event = _make_event(text="/btw")
        result = await runner._handle_btw_command(event)
        assert "Usage:" in result
        assert "/btw" in result

    @pytest.mark.asyncio
    async def test_no_history_reports_no_conversation(self):
        runner = _make_runner()
        store = AsyncMock()
        store.get_or_create_session.return_value = MagicMock(session_id="s1")
        store.load_transcript.return_value = []
        store._store = runner.session_store
        runner._async_session_store = store
        event = _make_event(text="/btw what did we do?")
        result = await runner._handle_btw_command(event)
        assert "conversation" in result.lower()

    @pytest.mark.asyncio
    async def test_dispatches_side_question_and_sends_answer(self):
        runner = _make_runner()
        store = AsyncMock()
        store.get_or_create_session.return_value = MagicMock(session_id="s1")
        store.load_transcript.return_value = [
            {"role": "user", "content": "fix foo.py"},
            {"role": "assistant", "content": "done"},
        ]
        store._store = runner.session_store
        runner._async_session_store = store
        runner._resolve_session_agent_runtime = MagicMock(
            return_value=("test-model", {"api_key": "k", "provider": "p",
                                         "base_url": "u", "api_mode": "chat_completions"})
        )
        runner._reply_anchor_for_event = MagicMock(return_value=None)
        runner._thread_metadata_for_source = MagicMock(return_value=None)
        mock_adapter = AsyncMock()
        runner._adapter_for_source = MagicMock(return_value=mock_adapter)

        event = _make_event(text="/btw which file was that?")

        with patch("agent.side_question.answer_side_question",
                   return_value="it was foo.py") as mock_answer:
            result = await runner._handle_btw_command(event)
            # Ack returned immediately, worker task registered.
            assert "which file was that?" in result
            # Drain the fire-and-forget task.
            for task in list(runner._background_tasks):
                await task

        # Snapshot + question reached the engine; live history untouched.
        args, kwargs = mock_answer.call_args
        assert args[0] == "which file was that?"
        assert args[1][0]["content"] == "fix foo.py"
        assert kwargs["main_runtime"]["model"] == "test-model"

        # The answer was delivered to the chat.
        mock_adapter.send.assert_called_once()
        sent_text = mock_adapter.send.call_args[0][1]
        assert "it was foo.py" in sent_text

    @pytest.mark.asyncio
    async def test_no_credentials_reports_error(self):
        runner = _make_runner()
        store = AsyncMock()
        store.get_or_create_session.return_value = MagicMock(session_id="s1")
        store.load_transcript.return_value = [{"role": "user", "content": "hi"}]
        store._store = runner.session_store
        runner._async_session_store = store
        runner._resolve_session_agent_runtime = MagicMock(
            return_value=(None, {"api_key": None})
        )
        event = _make_event(text="/btw what?")
        result = await runner._handle_btw_command(event)
        assert "❌" in result
