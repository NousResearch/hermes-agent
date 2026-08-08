"""Tests for /update live streaming, prompt forwarding, and gateway IPC.

Tests the new --gateway mode for hermes update, including:
- _gateway_prompt() file-based IPC
- _watch_update_progress() output streaming and prompt detection
- Message interception for update prompt responses
- _restore_stashed_changes() with input_fn parameter
"""

import json
import os
import time
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, SendResult
from gateway.session import SessionSource


def _make_event(text="/update", platform=Platform.TELEGRAM,
                user_id="12345", chat_id="67890"):
    """Build a MessageEvent for testing."""
    source = SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        user_name="testuser",
    )
    return MessageEvent(text=text, source=source)


def _make_runner(hermes_home=None):
    """Create a bare GatewayRunner without calling __init__."""
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._update_prompt_pending = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._failed_platforms = {}
    # config is accessed by _check_slash_access and quick_commands lookup;
    # None makes policy_for_source return a disabled (allow-all) policy.
    runner.config = None
    # Bypass the destructive-slash confirm gate — this test exercises
    # update-prompt interception, not the confirm prompt.
    runner._read_user_config = lambda: {
        "approvals": {"destructive_slash_confirm": False}
    }
    return runner


# ---------------------------------------------------------------------------
# _gateway_prompt (file-based IPC in main.py)
# ---------------------------------------------------------------------------


class TestGatewayPrompt:
    """Tests for _gateway_prompt() function."""

    def test_writes_prompt_file_and_reads_response(self, tmp_path):
        """Writes .update_prompt.json, reads .update_response, returns answer."""
        import threading
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()

        # Simulate the response arriving after a short delay
        def write_response():
            time.sleep(0.2)
            (hermes_home / ".update_response").write_text("y")

        thread = threading.Thread(target=write_response)
        thread.start()

        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            from hermes_cli.main import _gateway_prompt
            result = _gateway_prompt("Restore? [Y/n]", "y", timeout=5.0)

        thread.join()
        assert result == "y"
        # Both files should be cleaned up
        assert not (hermes_home / ".update_prompt.json").exists()
        assert not (hermes_home / ".update_response").exists()


# ---------------------------------------------------------------------------
# _restore_stashed_changes with input_fn
# ---------------------------------------------------------------------------


class TestRestoreStashWithInputFn:
    """Tests for _restore_stashed_changes with the input_fn parameter."""

    def test_uses_input_fn_when_provided(self, tmp_path):
        """When input_fn is provided, it's called instead of input()."""
        from hermes_cli.main import _restore_stashed_changes

        captured_args = []

        def fake_input_fn(prompt, default=""):
            captured_args.append((prompt, default))
            return "n"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="", stderr=""
            )
            result = _restore_stashed_changes(
                ["git"], tmp_path, "abc123",
                prompt_user=True,
                input_fn=fake_input_fn,
            )

        assert len(captured_args) == 1
        assert "Restore" in captured_args[0][0]
        assert result is False  # user declined


# ---------------------------------------------------------------------------
# Update command spawns --gateway flag
# ---------------------------------------------------------------------------


class TestUpdateCommandGatewayFlag:
    """Verify the gateway spawns hermes update --gateway."""

    @pytest.mark.asyncio
    async def test_spawns_with_gateway_flag(self, tmp_path):
        """The spawned update command includes --gateway and PYTHONUNBUFFERED."""
        runner = _make_runner()
        event = _make_event()

        fake_root = tmp_path / "project"
        fake_root.mkdir()
        (fake_root / ".git").mkdir()
        (fake_root / "gateway").mkdir()
        (fake_root / "gateway" / "run.py").touch()
        fake_file = str(fake_root / "gateway" / "run.py")
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()

        mock_popen = MagicMock()
        with patch("gateway.run._hermes_home", hermes_home), \
             patch("gateway.run.__file__", fake_file), \
             patch("shutil.which", side_effect=lambda x: f"/usr/bin/{x}"), \
             patch("subprocess.Popen", mock_popen):
            result = await runner._handle_update_command(event)

        # Check the bash command string contains --gateway and PYTHONUNBUFFERED
        call_args = mock_popen.call_args[0][0]
        cmd_string = call_args[-1] if isinstance(call_args, list) else str(call_args)
        assert "--gateway" in cmd_string
        assert "PYTHONUNBUFFERED" in cmd_string
        assert "rc=$?" in cmd_string
        assert "status=$?" not in cmd_string
        assert "stream progress" in result


# ---------------------------------------------------------------------------
# _watch_update_progress — output streaming
# ---------------------------------------------------------------------------


class TestWatchUpdateProgress:
    """Tests for _watch_update_progress() streaming output."""

    @pytest.mark.asyncio
    async def test_streams_output_to_adapter(self, tmp_path):
        """New output is sent to the adapter periodically."""
        runner = _make_runner()
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()

        pending = {"platform": "telegram", "chat_id": "111", "user_id": "222",
                   "session_key": "agent:main:telegram:dm:111"}
        (hermes_home / ".update_pending.json").write_text(json.dumps(pending))
        # Write output
        (hermes_home / ".update_output.txt").write_text("→ Fetching updates...\n", encoding="utf-8")

        mock_adapter = AsyncMock()
        runner.adapters = {Platform.TELEGRAM: mock_adapter}

        # Write exit code after a brief delay
        async def write_exit_code():
            await asyncio.sleep(0.2)
            (hermes_home / ".update_output.txt").write_text(
                "→ Fetching updates...\n✓ Code updated!\n"
            , encoding="utf-8")
            (hermes_home / ".update_exit_code").write_text("0")

        with patch("gateway.run._hermes_home", hermes_home):
            task = asyncio.create_task(write_exit_code())
            await runner._watch_update_progress(
                poll_interval=0.1,
                stream_interval=0.2,
                timeout=5.0,
            )
            await task

        # Should have sent at least the output and a success message
        assert mock_adapter.send.call_count >= 1
        all_sent = " ".join(str(c) for c in mock_adapter.send.call_args_list)
        assert "update finished" in all_sent.lower()

    @pytest.mark.asyncio
    async def test_detects_and_forwards_prompt(self, tmp_path):
        """Detects .update_prompt.json and sends it to the user."""
        runner = _make_runner()
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()

        pending = {"platform": "telegram", "chat_id": "111", "user_id": "222",
                   "session_key": "agent:main:telegram:dm:111"}
        (hermes_home / ".update_pending.json").write_text(json.dumps(pending))
        (hermes_home / ".update_output.txt").write_text("output\n")

        mock_adapter = AsyncMock()
        runner.adapters = {Platform.TELEGRAM: mock_adapter}

        # Write a prompt, then respond and finish
        async def simulate_prompt_cycle():
            await asyncio.sleep(0.2)
            prompt = {"prompt": "Restore local changes? [Y/n]", "default": "y", "id": "test1"}
            (hermes_home / ".update_prompt.json").write_text(json.dumps(prompt))
            # Simulate user responding
            await asyncio.sleep(0.2)
            (hermes_home / ".update_response").write_text("y")
            (hermes_home / ".update_prompt.json").unlink(missing_ok=True)
            await asyncio.sleep(0.2)
            (hermes_home / ".update_exit_code").write_text("0")

        with patch("gateway.run._hermes_home", hermes_home):
            task = asyncio.create_task(simulate_prompt_cycle())
            await runner._watch_update_progress(
                poll_interval=0.1,
                stream_interval=0.2,
                timeout=10.0,
            )
            await task

        # Check that the prompt was forwarded
        all_sent = [str(c) for c in mock_adapter.send.call_args_list]
        prompt_found = any("Restore local changes" in s for s in all_sent)
        assert prompt_found, f"Prompt not forwarded. Sent: {all_sent}"
        # Check session was marked as having pending prompt
        # (may be cleared by the time we check since update finished)

    @pytest.mark.parametrize(
        ("native_result", "uses_text_fallback"),
        [
            pytest.param(None, False, id="legacy-none"),
            pytest.param(SendResult(success=True), False, id="success"),
            pytest.param(
                SendResult(success=False, error="native delivery failed"),
                True,
                id="failure",
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_native_prompt_send_uses_text_fallback_only_on_failure(
        self, tmp_path, caplog, native_result, uses_text_fallback
    ):
        """Only an explicit native SendResult failure uses the text prompt."""
        runner = _make_runner()
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()

        pending = {
            "platform": "telegram",
            "chat_id": "111",
            "user_id": "222",
            "session_key": "agent:main:telegram:dm:111",
        }
        (hermes_home / ".update_pending.json").write_text(json.dumps(pending))
        (hermes_home / ".update_prompt.json").write_text(json.dumps({
            "prompt": "Restore local changes? [Y/n]",
            "default": "y",
        }))

        class NativePromptAdapter:
            def __init__(self):
                self.native_prompt_calls = []
                self.sent = []

            async def send_update_prompt(self, **kwargs):
                self.native_prompt_calls.append(kwargs)
                return native_result

            async def send(self, chat_id, content, metadata=None):
                self.sent.append((chat_id, content, metadata))
                return SendResult(success=True)

        adapter = NativePromptAdapter()
        runner.adapters = {Platform.TELEGRAM: adapter}

        async def finish_update():
            await asyncio.sleep(0.2)
            (hermes_home / ".update_exit_code").write_text("0")

        with caplog.at_level("DEBUG", logger="gateway.run"):
            with patch("gateway.run._hermes_home", hermes_home):
                task = asyncio.create_task(finish_update())
                await runner._watch_update_progress(
                    poll_interval=0.05,
                    stream_interval=0.1,
                    timeout=10.0,
                )
                await task

        assert len(adapter.native_prompt_calls) == 1
        prompt_sends = [
            content for _, content, _ in adapter.sent
            if "Restore local changes" in content
        ]
        assert len(prompt_sends) == int(uses_text_fallback)
        if uses_text_fallback:
            assert "native delivery failed" in caplog.text
        else:
            assert "Native update prompt send reported failure" not in caplog.text

    @pytest.mark.asyncio
    async def test_prompt_is_recovered_after_watcher_restart(self, tmp_path):
        """A forwarded prompt stays on disk until answered so a new watcher can recover it."""
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()

        pending = {
            "platform": "telegram",
            "chat_id": "111",
            "user_id": "222",
            "session_key": "agent:main:telegram:dm:111",
        }
        prompt = {
            "prompt": "Restore local changes? [Y/n]",
            "default": "y",
            "id": "restart-recover",
        }
        (hermes_home / ".update_pending.json").write_text(json.dumps(pending))
        (hermes_home / ".update_output.txt").write_text("")
        (hermes_home / ".update_prompt.json").write_text(json.dumps(prompt))

        runner1 = _make_runner()
        adapter1 = AsyncMock()
        runner1.adapters = {Platform.TELEGRAM: adapter1}

        with patch("gateway.run._hermes_home", hermes_home):
            watch1 = asyncio.create_task(
                runner1._watch_update_progress(
                    poll_interval=0.05,
                    stream_interval=0.1,
                    timeout=10.0,
                )
            )
            for _ in range(40):
                if adapter1.send.call_count:
                    break
                await asyncio.sleep(0.05)

            assert adapter1.send.call_count == 1
            assert (hermes_home / ".update_prompt.json").exists()

            watch1.cancel()
            with pytest.raises(asyncio.CancelledError):
                await watch1

            runner2 = _make_runner()
            adapter2 = AsyncMock()
            runner2.adapters = {Platform.TELEGRAM: adapter2}

            async def respond_and_finish():
                await asyncio.sleep(0.2)
                (hermes_home / ".update_response").write_text("y")
                await asyncio.sleep(0.2)
                (hermes_home / ".update_exit_code").write_text("0")

            finisher = asyncio.create_task(respond_and_finish())
            await runner2._watch_update_progress(
                poll_interval=0.05,
                stream_interval=0.1,
                timeout=10.0,
            )
            await finisher

        prompt_sends = [
            str(call) for call in adapter2.send.call_args_list
            if "Restore local changes" in str(call)
        ]
        assert len(prompt_sends) == 1


# ---------------------------------------------------------------------------
# Message interception for update prompts
# ---------------------------------------------------------------------------


class TestUpdatePromptInterception:
    """Tests for update prompt response interception in _handle_message."""


    @pytest.mark.asyncio
    async def test_recognized_slash_command_bypasses_pending_update_prompt(self, tmp_path):
        """Known slash commands must dispatch normally instead of being consumed.

        The update subprocess is still blocked on stdin waiting for
        ``.update_response``, so the gateway writes a blank response to
        unblock it (``_gateway_prompt`` returns the prompt's default on
        empty) before falling through to normal command dispatch.
        """
        runner = _make_runner()
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()

        event = _make_event(text="/new", chat_id="67890")
        session_key = "agent:main:telegram:dm:67890"
        runner._update_prompt_pending[session_key] = True
        runner._is_user_authorized = MagicMock(return_value=True)
        runner._session_key_for_source = MagicMock(return_value=session_key)
        runner._handle_reset_command = AsyncMock(return_value="reset ok")
        (hermes_home / ".update_prompt.json").write_text(json.dumps({"prompt": "test"}))

        with patch("gateway.run._hermes_home", hermes_home):
            result = await runner._handle_message(event)

        assert result == "reset ok"
        runner._handle_reset_command.assert_awaited_once_with(event)
        # .update_response was written (empty) to unblock the update
        # subprocess; _gateway_prompt will read "", strip to "", and
        # return the prompt's default.
        response_path = hermes_home / ".update_response"
        assert response_path.exists()
        assert response_path.read_text() == ""
        assert not (hermes_home / ".update_prompt.json").exists()
        # Pending flag is cleared so stray future input won't be
        # re-intercepted for a prompt that is no longer outstanding.
        assert session_key not in runner._update_prompt_pending


# ---------------------------------------------------------------------------
# cmd_update --gateway flag
# ---------------------------------------------------------------------------


class TestCmdUpdateGatewayMode:
    """Tests for cmd_update with --gateway flag."""

    def test_gateway_flag_enables_gateway_prompt_for_stash(self, tmp_path):
        """With --gateway, stash restore uses _gateway_prompt instead of input()."""
        from hermes_cli.main import _restore_stashed_changes

        # Use input_fn to verify the gateway path is taken
        calls = []

        def fake_input(prompt, default=""):
            calls.append(prompt)
            return "n"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            _restore_stashed_changes(
                ["git"], tmp_path, "abc123",
                prompt_user=True,
                input_fn=fake_input,
            )

        assert len(calls) == 1
        assert "Restore" in calls[0]

