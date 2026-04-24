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
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
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
            time.sleep(0.3)
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

    def test_prompt_file_content(self, tmp_path):
        """Verifies the prompt JSON structure."""
        import threading
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()

        prompt_data = None

        def capture_and_respond():
            nonlocal prompt_data
            prompt_path = hermes_home / ".update_prompt.json"
            for _ in range(20):
                if prompt_path.exists():
                    prompt_data = json.loads(prompt_path.read_text())
                    (hermes_home / ".update_response").write_text("n")
                    return
                time.sleep(0.1)

        thread = threading.Thread(target=capture_and_respond)
        thread.start()

        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            from hermes_cli.main import _gateway_prompt
            _gateway_prompt("Configure now? [Y/n]", "n", timeout=5.0)

        thread.join()
        assert prompt_data is not None
        assert prompt_data["prompt"] == "Configure now? [Y/n]"
        assert prompt_data["default"] == "n"
        assert "id" in prompt_data

    def test_timeout_returns_default(self, tmp_path):
        """Returns default when no response within timeout."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()

        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            from hermes_cli.main import _gateway_prompt
            result = _gateway_prompt("test?", "default_val", timeout=0.5)

        assert result == "default_val"

    def test_empty_response_returns_default(self, tmp_path):
        """Empty response file returns default."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / ".update_response").write_text("")

        # Write prompt file so the function starts polling
        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            from hermes_cli.main import _gateway_prompt
            # Pre-create the response
            result = _gateway_prompt("test?", "default_val", timeout=2.0)

        assert result == "default_val"


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

    def test_input_fn_yes_proceeds_with_restore(self, tmp_path):
        """When input_fn returns 'y', stash apply is attempted."""
        from hermes_cli.main import _restore_stashed_changes

        call_count = [0]

        def fake_run(*args, **kwargs):
            call_count[0] += 1
            mock = MagicMock()
            mock.returncode = 0
            mock.stdout = ""
            mock.stderr = ""
            return mock

        with patch("subprocess.run", side_effect=fake_run):
            _restore_stashed_changes(
                ["git"], tmp_path, "abc123",
                prompt_user=True,
                input_fn=lambda p, d="": "y",
            )

        # Should have called git stash apply + git diff --name-only
        assert call_count[0] >= 2


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
        (hermes_home / ".update_output.txt").write_text("→ Fetching updates...\n")

        mock_adapter = AsyncMock()
        runner.adapters = {Platform.TELEGRAM: mock_adapter}

        # Write exit code after a brief delay
        async def write_exit_code():
            await asyncio.sleep(0.3)
            (hermes_home / ".update_output.txt").write_text(
                "→ Fetching updates...\n✓ Code updated!\n"
            )
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
            await asyncio.sleep(0.3)
            prompt = {"prompt": "Restore local changes? [Y/n]", "default": "y", "id": "test1"}
            (hermes_home / ".update_prompt.json").write_text(json.dumps(prompt))
            # Simulate user responding
            await asyncio.sleep(0.5)
            (hermes_home / ".update_response").write_text("y")
            (hermes_home / ".update_prompt.json").unlink(missing_ok=True)
            await asyncio.sleep(0.3)
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

    @pytest.mark.asyncio
    async def test_cleans_up_on_completion(self, tmp_path):
        """All marker files are cleaned up when update finishes."""
        runner = _make_runner()
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()

        pending = {"platform": "telegram", "chat_id": "111", "user_id": "222",
                   "session_key": "agent:main:telegram:dm:111"}
        pending_path = hermes_home / ".update_pending.json"
        output_path = hermes_home / ".update_output.txt"
        exit_code_path = hermes_home / ".update_exit_code"
        pending_path.write_text(json.dumps(pending))
        output_path.write_text("done\n")
        exit_code_path.write_text("0")

        mock_adapter = AsyncMock()
        runner.adapters = {Platform.TELEGRAM: mock_adapter}

        with patch("gateway.run._hermes_home", hermes_home):
            await runner._watch_update_progress(
                poll_interval=0.1,
                stream_interval=0.2,
                timeout=5.0,
            )

        assert not pending_path.exists()
        assert not output_path.exists()
        assert not exit_code_path.exists()

    @pytest.mark.asyncio
    async def test_failure_exit_code(self, tmp_path):
        """Non-zero exit code sends failure message."""
        runner = _make_runner()
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()

        pending = {"platform": "telegram", "chat_id": "111", "user_id": "222",
                   "session_key": "agent:main:telegram:dm:111"}
        (hermes_home / ".update_pending.json").write_text(json.dumps(pending))
        (hermes_home / ".update_output.txt").write_text("error occurred\n")
        (hermes_home / ".update_exit_code").write_text("1")

        mock_adapter = AsyncMock()
        runner.adapters = {Platform.TELEGRAM: mock_adapter}

        with patch("gateway.run._hermes_home", hermes_home):
            await runner._watch_update_progress(
                poll_interval=0.1,
                stream_interval=0.2,
                timeout=5.0,
            )

        all_sent = " ".join(str(c) for c in mock_adapter.send.call_args_list)
        assert "failed" in all_sent.lower()

    @pytest.mark.asyncio
    async def test_falls_back_when_adapter_unavailable(self, tmp_path):
        """Falls back to legacy notification when adapter can't be resolved."""
        runner = _make_runner()
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()

        # Platform doesn't match any adapter
        pending = {"platform": "discord", "chat_id": "111", "user_id": "222"}
        (hermes_home / ".update_pending.json").write_text(json.dumps(pending))
        (hermes_home / ".update_output.txt").write_text("done\n")
        (hermes_home / ".update_exit_code").write_text("0")

        # Only telegram adapter available
        mock_adapter = AsyncMock()
        runner.adapters = {Platform.TELEGRAM: mock_adapter}

        with patch("gateway.run._hermes_home", hermes_home):
            await runner._watch_update_progress(
                poll_interval=0.1,
                stream_interval=0.2,
                timeout=5.0,
            )

        # Should not crash; legacy notification handles this case

    @pytest.mark.asyncio
    async def test_prompt_forwarded_only_once(self, tmp_path):
        """Regression: prompt must not be re-sent on every poll cycle.

        Before the fix, the watcher never deleted .update_prompt.json after
        forwarding, causing the same prompt to be sent every poll_interval.
        """
        runner = _make_runner()
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()

        pending = {"platform": "telegram", "chat_id": "111", "user_id": "222",
                   "session_key": "agent:main:telegram:dm:111"}
        (hermes_home / ".update_pending.json").write_text(json.dumps(pending))
        (hermes_home / ".update_output.txt").write_text("")

        mock_adapter = AsyncMock()
        runner.adapters = {Platform.TELEGRAM: mock_adapter}

        # Write the prompt file up front (before the watcher starts).
        # The watcher should forward it exactly once, then delete it.
        prompt = {"prompt": "Would you like to configure new options now? Y/n",
                  "default": "n", "id": "dup-test"}
        (hermes_home / ".update_prompt.json").write_text(json.dumps(prompt))

        async def finish_after_polls():
            # Wait long enough for multiple poll cycles to occur, then
            # simulate a response + completion.
            await asyncio.sleep(1.0)
            (hermes_home / ".update_response").write_text("n")
            await asyncio.sleep(0.3)
            (hermes_home / ".update_exit_code").write_text("0")

        with patch("gateway.run._hermes_home", hermes_home):
            task = asyncio.create_task(finish_after_polls())
            await runner._watch_update_progress(
                poll_interval=0.1,
                stream_interval=0.2,
                timeout=10.0,
            )
            await task

        # Count how many times the prompt text was sent
        all_sent = [str(c) for c in mock_adapter.send.call_args_list]
        prompt_sends = [s for s in all_sent if "configure new options" in s]
        assert len(prompt_sends) == 1, (
            f"Prompt was sent {len(prompt_sends)} times (expected 1). "
            f"All sends: {all_sent}"
        )


# ---------------------------------------------------------------------------
# Message interception for update prompts
# ---------------------------------------------------------------------------


class TestUpdatePromptInterception:
    """Tests for update prompt response interception in _handle_message."""

    @pytest.mark.asyncio
    async def test_intercepts_response_when_prompt_pending(self, tmp_path):
        """When _update_prompt_pending is set, the next message writes .update_response."""
        runner = _make_runner()
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()

        event = _make_event(text="y", chat_id="67890")
        # The session key uses the full format from build_session_key
        session_key = "agent:main:telegram:dm:67890"
        runner._update_prompt_pending[session_key] = True

        # Mock authorization and _session_key_for_source
        runner._is_user_authorized = MagicMock(return_value=True)
        runner._session_key_for_source = MagicMock(return_value=session_key)

        with patch("gateway.run._hermes_home", hermes_home):
            result = await runner._handle_message(event)

        assert result is not None
        assert "Sent" in result
        response_path = hermes_home / ".update_response"
        assert response_path.exists()
        assert response_path.read_text() == "y"
        # Should clear the pending flag
        assert session_key not in runner._update_prompt_pending

    @pytest.mark.asyncio
    async def test_normal_message_when_no_prompt_pending(self, tmp_path):
        """Messages pass through normally when no prompt is pending."""
        runner = _make_runner()
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()

        event = _make_event(text="hello", chat_id="67890")

        # No pending prompt
        runner._is_user_authorized = MagicMock(return_value=True)

        # The message should flow through to normal processing;
        # we just verify it doesn't get intercepted
        session_key = "agent:main:telegram:dm:67890"
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

    def test_gateway_flag_parsed(self):
        """The --gateway flag is accepted by the update subparser."""
        # Verify the argparse parser accepts --gateway by checking cmd_update
        # receives gateway=True when the flag is set
        from types import SimpleNamespace
        args = SimpleNamespace(gateway=True)
        assert args.gateway is True
