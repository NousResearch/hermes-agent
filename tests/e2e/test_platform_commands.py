"""E2E tests for gateway slash commands (Telegram, Discord).

Each test drives a message through the full async pipeline:
    adapter.handle_message(event)
        → BasePlatformAdapter._process_message_background()
        → GatewayRunner._handle_message() (command dispatch)
        → adapter.send() (captured for assertions)

No LLM involved — only gateway-level commands are tested.
Tests are parametrized over platforms via the ``platform`` fixture in conftest.
"""

import asyncio
import hashlib
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import hermes_cli.telegram_canary as telegram_canary
from gateway.config import Platform
from gateway.platforms.base import SendResult
from gateway.session import build_session_key
from hermes_cli.telegram_canary import build_live_payload
from plugins.platforms.telegram.adapter import prepare_legacy_text_chunks
from tests.e2e.conftest import make_event, send_and_capture


class TestSlashCommands:
    """Gateway slash commands dispatched through the full adapter pipeline."""

    @pytest.mark.asyncio
    async def test_help_returns_command_list(self, adapter, platform):
        send = await send_and_capture(adapter, "/help", platform)

        send.assert_called_once()
        response_text = send.call_args[1].get("content") or send.call_args[0][1]
        assert "/new" in response_text
        assert "/status" in response_text

    @pytest.mark.asyncio
    async def test_status_shows_session_info(self, adapter, platform):
        send = await send_and_capture(adapter, "/status", platform)

        send.assert_called_once()
        response_text = send.call_args[1].get("content") or send.call_args[0][1]
        assert "session" in response_text.lower() or "Session" in response_text

    @pytest.mark.asyncio
    async def test_new_resets_session(self, adapter, runner, platform):
        send = await send_and_capture(adapter, "/new", platform)

        send.assert_called_once()
        runner.session_store.reset_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_when_no_agent_running(self, adapter, platform):
        send = await send_and_capture(adapter, "/stop", platform)

        send.assert_called_once()
        response_text = send.call_args[1].get("content") or send.call_args[0][1]
        response_lower = response_text.lower()
        assert "no" in response_lower or "stop" in response_lower or "not running" in response_lower

    @pytest.mark.asyncio
    async def test_commands_shows_listing(self, adapter, platform):
        send = await send_and_capture(adapter, "/commands", platform)

        send.assert_called_once()
        response_text = send.call_args[1].get("content") or send.call_args[0][1]
        # Should list at least some commands
        assert "/" in response_text

    @pytest.mark.asyncio
    async def test_sequential_commands_share_session(self, adapter, platform):
        """Two commands from the same chat_id should both succeed."""
        send_help = await send_and_capture(adapter, "/help", platform)
        send_help.assert_called_once()

        send_status = await send_and_capture(adapter, "/status", platform)
        send_status.assert_called_once()

    @pytest.mark.asyncio
    async def test_verbose_responds(self, adapter, platform):
        send = await send_and_capture(adapter, "/verbose", platform)

        send.assert_called_once()
        response_text = send.call_args[1].get("content") or send.call_args[0][1]
        # Either shows the mode cycle or tells user to enable it in config
        assert "verbose" in response_text.lower() or "tool_progress" in response_text

    @pytest.mark.asyncio
    async def test_plaintext_restart_gateway_routes_to_safe_restart_command(self, adapter, runner, platform, monkeypatch):
        if platform != Platform.TELEGRAM:
            pytest.skip("Plaintext restart shortcut is intentionally DM/Telegram-focused")

        monkeypatch.setenv("INVOCATION_ID", "e2e-systemd")
        runner.request_restart = MagicMock(return_value=True)

        send = await send_and_capture(adapter, "restart gateway", platform)

        send.assert_called_once()
        response_text = send.call_args[1].get("content") or send.call_args[0][1]
        assert "restart" in response_text.lower() or "draining" in response_text.lower()
        runner.request_restart.assert_called_once_with(detached=False, via_service=True)

    @pytest.mark.asyncio
    async def test_plaintext_restart_gateway_in_group_stays_plain_text(self, adapter, runner, platform, monkeypatch):
        if platform != Platform.TELEGRAM:
            pytest.skip("Shortcut scope is only verified for Telegram here")

        monkeypatch.setenv("INVOCATION_ID", "e2e-systemd")
        runner.request_restart = MagicMock(return_value=True)
        runner._handle_message_with_agent = AsyncMock(return_value="agent-handled")

        send = await send_and_capture(adapter, "restart gateway", platform, chat_id="group-chat-1", user_id="u1", chat_type="group")

        send.assert_called_once()
        response_text = send.call_args[1].get("content") or send.call_args[0][1]
        assert response_text == "agent-handled"
        runner.request_restart.assert_not_called()

    @pytest.mark.asyncio
    async def test_personality_lists_options(self, adapter, platform):
        send = await send_and_capture(adapter, "/personality", platform)

        send.assert_called_once()
        response_text = send.call_args[1].get("content") or send.call_args[0][1]
        assert "personalit" in response_text.lower()  # matches "personality" or "personalities"

    @pytest.mark.asyncio
    async def test_yolo_toggles_mode(self, adapter, platform):
        send = await send_and_capture(adapter, "/yolo", platform)

        send.assert_called_once()
        response_text = send.call_args[1].get("content") or send.call_args[0][1]
        assert "yolo" in response_text.lower()

    @pytest.mark.asyncio
    async def test_compress_command(self, adapter, platform):
        send = await send_and_capture(adapter, "/compress", platform)

        send.assert_called_once()
        response_text = send.call_args[1].get("content") or send.call_args[0][1]
        assert "compress" in response_text.lower() or "context" in response_text.lower()

    @pytest.mark.asyncio
    async def test_quick_command_alias_targets_builtin_command_with_args(
        self, adapter, runner, platform
    ):
        """Alias targets with args must reach the built-in command handler."""
        runner.config.quick_commands = {
            "s": {"type": "alias", "target": "/status extra-arg"}
        }
        async def _handle_status(event):
            assert event.get_command_args() == "extra-arg"
            return "status via alias"

        runner._handle_status_command = AsyncMock(side_effect=_handle_status)

        send = await send_and_capture(adapter, "/s", platform)

        send.assert_called_once()
        response_text = send.call_args[1].get("content") or send.call_args[0][1]
        assert response_text == "status via alias"
        runner._handle_status_command.assert_awaited_once()
        runner._handle_message_with_agent.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_quick_command_argv_runs_through_full_platform_pipeline(
        self, adapter, runner, platform
    ):
        runner.config.quick_commands = {
            "remember": {
                "type": "argv",
                "command": ["remember-spool-append", "--type", "fact"],
                "argument_mode": "text",
                "destination_alias": "owner",
            }
        }
        proc = MagicMock(returncode=0)

        with (
            patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as spawn,
            patch(
                "hermes_cli.quick_commands.communicate_bounded_async",
                AsyncMock(return_value=(b"saved", b"")),
            ),
        ):
            send = await send_and_capture(
                adapter, "/remember hello; echo not-a-shell", platform
            )

        assert spawn.await_args.args == (
            "remember-spool-append",
            "--type",
            "fact",
            "hello; echo not-a-shell",
        )
        send.assert_called_once()
        response_text = send.call_args[1].get("content") or send.call_args[0][1]
        assert response_text == "saved"
        runner._handle_message_with_agent.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_live_telegram_canary_binds_existing_delivery_and_suppresses_duplicate(
        self, adapter, runner, platform, monkeypatch, tmp_path
    ):
        if platform != Platform.TELEGRAM:
            pytest.skip("The synthetic delivery receipt is Telegram-specific")

        runtime_sha = "e" * 40
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "e2e-user-1")
        for key in (
            "GATEWAY_ALLOWED_USERS",
            "TELEGRAM_ALLOW_ALL_USERS",
            "GATEWAY_ALLOW_ALL_USERS",
        ):
            monkeypatch.delenv(key, raising=False)
        runner.config.quick_commands = {
            "gateway-canary": {
                "type": "argv",
                "command": [
                    "python",
                    "-m",
                    "hermes_cli.telegram_canary",
                    "payload",
                    "--runtime-sha",
                    runtime_sha,
                ],
                "argument_mode": "none",
                "destination_alias": "owner",
                "delivery_receipt": "telegram_canary",
                "runtime_sha": runtime_sha,
            }
        }
        payload = build_live_payload(runtime_sha)
        chunks = prepare_legacy_text_chunks(payload)
        delivery = SendResult(
            success=True,
            message_id="private-live-message-1",
            raw_response={
                "message_ids": [
                    f"private-live-message-{index}" for index in range(len(chunks))
                ],
                "attempt_counts": [2, *([1] * (len(chunks) - 1))],
                "synthetic_pre_send_failures": 1,
                "chunk_count": len(chunks),
                "chunk_sha256": [
                    "sha256:" + hashlib.sha256(chunk.encode("utf-8")).hexdigest()
                    for chunk in chunks
                ],
                "chunk_utf16_units": [
                    len(chunk.encode("utf-16-le")) // 2 for chunk in chunks
                ],
                "thread_fallback": False,
            },
        )
        adapter.send = AsyncMock(return_value=delivery)
        proc = MagicMock(returncode=0)
        event = make_event(platform, "/gateway-canary")
        event.message_id = "private-trigger-message"
        event.platform_update_id = 424242
        session_key = build_session_key(event.source)
        receipt_path = tmp_path / "hermes" / "receipts" / "telegram-canary.jsonl"
        state_path = (
            tmp_path / "hermes" / "receipts" / "telegram-canary-state.json"
        )
        register_callback = adapter.register_post_delivery_callback
        original_write_state = telegram_canary._write_state
        crashed_after_append = False

        def crash_after_receipt_append(path, state):
            nonlocal crashed_after_append
            if not crashed_after_append and any(
                isinstance(entry, dict)
                and entry.get("status") == "receipt_written"
                for entry in state.get("runs", {}).values()
            ):
                crashed_after_append = True
                raise OSError("synthetic callback crash after receipt append")
            original_write_state(path, state)

        monkeypatch.setattr(
            telegram_canary,
            "_write_state",
            crash_after_receipt_append,
        )

        with (
            patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as spawn,
            patch(
                "hermes_cli.quick_commands.communicate_bounded_async",
                AsyncMock(return_value=(payload.encode(), b"")),
            ),
            patch(
                "hermes_cli.telegram_canary.verify_running_runtime_sha",
                return_value=True,
            ),
            patch.object(
                adapter,
                "register_post_delivery_callback",
                wraps=register_callback,
            ) as register,
            patch(
                "hermes_cli.telegram_canary.recover_sealed_live_canary",
                wraps=telegram_canary.recover_sealed_live_canary,
            ) as recover,
        ):
            await adapter.handle_message(event)
            for _ in range(60):
                if receipt_path.exists() and session_key not in adapter._active_sessions:
                    break
                await asyncio.sleep(0.05)

            assert receipt_path.exists()
            receipt = json.loads(receipt_path.read_text(encoding="utf-8").strip())
            assert receipt["result"] == "pass"
            assert receipt["mode"] == "live_gateway_pipeline"
            assert receipt["qualifies_for_external_acceptance"] is False
            assert receipt["checks"]["idempotency"]["duplicate_probe_suppressed"] is True
            assert receipt["checks"]["retry"]["safe_retry_verified"] is True
            assert receipt["checks"]["length"]["all_chunks_acknowledged"] is True
            assert crashed_after_append is True
            state = json.loads(state_path.read_text(encoding="utf-8"))
            assert next(iter(state["runs"].values()))["status"] == "receipt_written"
            assert adapter.send.await_count == 1
            metadata = adapter.send.await_args.kwargs["metadata"]
            assert metadata["hermes_synthetic_pre_send_connect_failure"] is True
            callback_entry = adapter._post_delivery_callbacks.get(session_key)
            assert callback_entry is None
            canary_registrations = [
                call
                for call in register.call_args_list
                if call.args and call.args[0] == session_key
            ]
            assert canary_registrations
            assert all(
                isinstance(call.kwargs.get("generation"), int)
                for call in canary_registrations
            )

            # Replay the exact same Telegram update. The canary producer's
            # pre-send claim suppresses both the subprocess and delivery.
            await adapter.handle_message(event)
            for _ in range(40):
                if session_key not in adapter._active_sessions:
                    break
                await asyncio.sleep(0.05)

            assert recover.call_count == 2

        assert spawn.await_count == 1
        assert adapter.send.await_count == 1
        persisted = receipt_path.read_text(encoding="utf-8")
        assert "private-trigger-message" not in persisted
        assert "private-live-message-1" not in persisted

    @pytest.mark.asyncio
    async def test_live_telegram_canary_rejects_allowed_group_member_before_spawn(
        self, adapter, runner, platform, monkeypatch
    ):
        if platform != Platform.TELEGRAM:
            pytest.skip("The synthetic delivery receipt is Telegram-specific")

        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "e2e-user-1")
        for key in (
            "GATEWAY_ALLOWED_USERS",
            "TELEGRAM_ALLOW_ALL_USERS",
            "GATEWAY_ALLOW_ALL_USERS",
        ):
            monkeypatch.delenv(key, raising=False)
        runner.config.quick_commands = {
            "gateway-canary": {
                "type": "argv",
                "command": ["fixture"],
                "argument_mode": "none",
                "destination_alias": "owner",
                "delivery_receipt": "telegram_canary",
                "runtime_sha": "a" * 40,
            }
        }

        with patch("asyncio.create_subprocess_exec", AsyncMock()) as spawn:
            send = await send_and_capture(
                adapter,
                "/gateway-canary",
                platform,
                chat_id="allowed-group",
                user_id="e2e-user-1",
                chat_type="group",
            )

        spawn.assert_not_awaited()
        response = send.call_args.kwargs.get("content") or send.call_args.args[1]
        assert "exact owner DM authorization" in response

    @pytest.mark.asyncio
    async def test_canary_fault_flag_is_consumed_before_reused_guard_followup(
        self, adapter, platform
    ):
        if platform != Platform.TELEGRAM:
            pytest.skip("The synthetic fault flag is Telegram-specific")

        event = make_event(platform, "/help")
        session_key = build_session_key(event.source)
        reused_guard = asyncio.Event()
        setattr(
            reused_guard,
            "_hermes_synthetic_pre_send_connect_failure",
            True,
        )
        adapter._active_sessions[session_key] = reused_guard
        adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="1"))

        await adapter._process_message_background(event, session_key)
        first_metadata = adapter.send.await_args.kwargs["metadata"]
        assert first_metadata["hermes_synthetic_pre_send_connect_failure"] is True
        assert not hasattr(
            reused_guard,
            "_hermes_synthetic_pre_send_connect_failure",
        )

        adapter._active_sessions[session_key] = reused_guard
        await adapter._process_message_background(make_event(platform, "/help"), session_key)
        second_metadata = adapter.send.await_args.kwargs["metadata"]
        assert "hermes_synthetic_pre_send_connect_failure" not in second_metadata



class TestSessionLifecycle:
    """Verify session state changes across command sequences."""

    @pytest.mark.asyncio
    async def test_new_then_status_reflects_reset(self, adapter, runner, session_entry, platform):
        """After /new, /status should report the fresh session."""
        await send_and_capture(adapter, "/new", platform)
        runner.session_store.reset_session.assert_called_once()

        send = await send_and_capture(adapter, "/status", platform)
        send.assert_called_once()
        response_text = send.call_args[1].get("content") or send.call_args[0][1]
        # Session ID from the entry should appear in the status output
        assert session_entry.session_id[:8] in response_text

    @pytest.mark.asyncio
    async def test_new_is_idempotent(self, adapter, runner, platform):
        """/new called twice should not crash."""
        await send_and_capture(adapter, "/new", platform)
        await send_and_capture(adapter, "/new", platform)
        assert runner.session_store.reset_session.call_count == 2


class TestAuthorization:
    """Verify the pipeline handles unauthorized users."""

    @pytest.mark.asyncio
    async def test_unauthorized_user_gets_pairing_response(self, adapter, runner, platform):
        """Unauthorized DM should trigger pairing code, not a command response."""
        runner._is_user_authorized = lambda _source: False

        event = make_event(platform, "/help")
        adapter.send.reset_mock()
        await adapter.handle_message(event)
        await asyncio.sleep(0.3)

        # The adapter.send is called directly by the authorization path
        # (not via _send_with_retry), so check it was called with a pairing message
        adapter.send.assert_called()
        response_text = adapter.send.call_args[0][1] if len(adapter.send.call_args[0]) > 1 else ""
        assert "recognize" in response_text.lower() or "pair" in response_text.lower() or "ABC123" in response_text

    @pytest.mark.asyncio
    async def test_unauthorized_user_does_not_get_help(self, adapter, runner, platform):
        """Unauthorized user should NOT see the help command output."""
        runner._is_user_authorized = lambda _source: False

        event = make_event(platform, "/help")
        adapter.send.reset_mock()
        await adapter.handle_message(event)
        await asyncio.sleep(0.3)

        # If send was called, it should NOT contain the help text
        if adapter.send.called:
            response_text = adapter.send.call_args[0][1] if len(adapter.send.call_args[0]) > 1 else ""
            assert "/new" not in response_text


class TestSendFailureResilience:
    """Verify the pipeline handles send failures gracefully."""

    @pytest.mark.asyncio
    async def test_send_failure_does_not_crash_pipeline(self, adapter, platform):
        """If send() returns failure, the pipeline should not raise."""
        adapter.send = AsyncMock(return_value=SendResult(success=False, error="network timeout"))
        adapter.set_message_handler(adapter._message_handler) # re-wire with same handler

        event = make_event(platform, "/help")
        # Should not raise — pipeline handles send failures internally
        await adapter.handle_message(event)
        await asyncio.sleep(0.3)

        adapter.send.assert_called()
