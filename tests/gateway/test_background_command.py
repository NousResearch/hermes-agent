"""Tests for /background gateway slash command.

Tests the _handle_background_command handler (run a prompt in a separate
background session) across gateway messenger platforms.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource


def _make_event(text="/background", platform=Platform.TELEGRAM,
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
        """Running /background with no prompt shows usage."""
        runner = _make_runner()
        event = _make_event(text="/background")
        result = await runner._handle_background_command(event)
        assert "Usage:" in result
        assert "/background" in result

    @pytest.mark.asyncio
    async def test_bg_alias_no_prompt_shows_usage(self):
        """Running /bg with no prompt shows usage."""
        runner = _make_runner()
        event = _make_event(text="/bg")
        result = await runner._handle_background_command(event)
        assert "Usage:" in result

    @pytest.mark.asyncio
    async def test_empty_prompt_shows_usage(self):
        """Running /background with only whitespace shows usage."""
        runner = _make_runner()
        event = _make_event(text="/background   ")
        result = await runner._handle_background_command(event)
        assert "Usage:" in result

    @pytest.mark.asyncio
    async def test_passes_parent_and_complete_reply_context(self):
        runner = _make_runner()
        parent = MagicMock(session_id="parent-session")
        runner.session_store.get_or_create_session.return_value = parent
        runner._run_background_task = AsyncMock(return_value=None)
        event = _make_event(text="/background inspect this")
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
        assert kwargs["origin"]["command"] == "/background"


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


# ---------------------------------------------------------------------------
# _run_background_task
# ---------------------------------------------------------------------------


class TestRunBackgroundTask:
    """Tests for GatewayRunner._run_background_task (the actual execution)."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("platform", "chat_id", "thread_id", "prompt_key"),
        [
            (Platform.TELEGRAM, "-100123456", "42", "-100123456:42"),
            (Platform.DISCORD, "channel-123", None, "channel-123"),
            (Platform.SLACK, "C123", "1712345.0001", "C123:1712345.0001"),
        ],
    )
    async def test_inherits_configured_channel_prompt_with_auto_skill(
        self, platform, chat_id, thread_id, prompt_key
    ):
        runner = _make_runner()
        adapter = MagicMock()
        adapter.send = AsyncMock()
        adapter.extract_media = MagicMock(return_value=([], "done"))
        adapter.extract_images = MagicMock(return_value=([], "done"))
        runner.adapters[platform] = adapter
        runner.config.platforms[platform] = PlatformConfig(
            extra={
                "channel_prompts": {
                    prompt_key: "CHANNEL_SENTINEL_ONLY_IN_CONFIG",
                }
            }
        )
        source = SessionSource(
            platform=platform,
            user_id="12345",
            chat_id=chat_id,
            thread_id=thread_id,
        )

        with patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "test-key"}), \
             patch("gateway.run._load_gateway_config", return_value={}), \
             patch(
                 "gateway.run._build_auto_skill_context",
                 return_value=("GENERIC AUTO SKILL SENTINEL", ["generic-topic-skill"]),
             ), \
             patch("run_agent.AIAgent") as MockAgent:
            agent = MockAgent.return_value
            agent.run_conversation.return_value = {"final_response": "done", "messages": []}
            await runner._run_background_task(
                "return the configured sentinel",
                source,
                "bg_channel_prompt",
                auto_skill=["generic-topic-skill"],
            )

        run_kwargs = agent.run_conversation.call_args.kwargs
        assert "CHANNEL_SENTINEL_ONLY_IN_CONFIG" in run_kwargs["system_message"]
        assert "GENERIC AUTO SKILL SENTINEL" in run_kwargs["user_message"]
        assert run_kwargs["current_user_text"] == "return the configured sentinel"
        assert run_kwargs["internal_context"] == {
            "auto_skill": ["generic-topic-skill"],
        }


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
                reply_to_text=complete_reply,
                reply_to_is_own_message=True,
                auto_skill=["arbitrary-topic-skill"],
                channel_prompt="ARBITRARY TOPIC RULE",
                internal_context={"channel_context": "SECONDARY TOPIC CONTEXT"},
                origin={
                    "execution_kind": "user_explicit_background",
                    "user_initiated": True,
                    "command": "/background",
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
        assert "SECONDARY TOPIC CONTEXT\n\n[New message]" in run_kwargs["user_message"]
        assert "ARBITRARY SKILL PAYLOAD" in run_kwargs["user_message"]
        assert f'[Replying to your previous message: "{complete_reply}"]' in run_kwargs["user_message"]
        assert decision_after_old_limit in run_kwargs["user_message"]
        record_kwargs = mock_agent_instance.record_gateway_session_peer.call_args.kwargs
        assert record_kwargs["routable"] is False
        origin = record_kwargs["origin"]
        assert origin["execution_kind"] == "user_explicit_background"
        assert origin["user_initiated"] is True
        assert origin["command"] == "/background"
        mock_agent_instance.shutdown_memory_provider.assert_called_once()
        mock_agent_instance.close.assert_called_once()

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
        assert prepared_marker in user_message.lower()
        assert path in user_message

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
            agent.run_conversation.return_value = {"final_response": "done", "messages": []}
            await runner._run_background_task(
                "scheduled work",
                source,
                "bg_other",
                parent_session_id="parent-session",
                parent_session_key="telegram:67890",
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
# /background in help and known_commands
# ---------------------------------------------------------------------------


class TestBackgroundInHelp:
    """Verify /background appears in help text and known commands."""

    @pytest.mark.asyncio
    async def test_background_in_help_output(self):
        """The /help output includes /background."""
        runner = _make_runner()
        event = _make_event(text="/help")
        result = await runner._handle_help_command(event)
        assert "/background" in result


# ---------------------------------------------------------------------------
# CLI /background command definition
# ---------------------------------------------------------------------------


class TestBackgroundInCLICommands:
    """Verify /background is registered in the CLI command system."""


    def test_background_autocompletes(self):
        """The /background command appears in autocomplete results."""
        pytest.importorskip("prompt_toolkit")
        from hermes_cli.commands import SlashCommandCompleter
        from prompt_toolkit.document import Document

        completer = SlashCommandCompleter()
        doc = Document("backgro")  # Partial match
        completions = list(completer.get_completions(doc, None))
        # Text doesn't start with / so no completions
        assert len(completions) == 0

        doc = Document("/backgro")  # With slash prefix
        completions = list(completer.get_completions(doc, None))
        cmd_displays = [str(c.display) for c in completions]
        assert any("/background" in d for d in cmd_displays)
