"""Tests for /update gateway slash command.

Tests both the _handle_update_command handler (spawns update process) and
the _send_update_notification startup hook (sends results after restart).
"""

import json
import os
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


def _make_runner():
    """Create a bare GatewayRunner without calling __init__."""
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    return runner


# ---------------------------------------------------------------------------
# _handle_update_command
# ---------------------------------------------------------------------------


class TestHandleUpdateCommand:
    """Tests for GatewayRunner._handle_update_command."""

    @pytest.mark.asyncio
    async def test_no_git_directory(self, tmp_path):
        """Returns an error when .git does not exist."""
        runner = _make_runner()
        event = _make_event()
        # Point _hermes_home to tmp_path and project_root to a dir without .git
        fake_root = tmp_path / "project"
        fake_root.mkdir()
        with patch("gateway.run._hermes_home", tmp_path), \
             patch("gateway.run.Path") as MockPath:
            # Path(__file__).parent.parent.resolve() -> fake_root
            MockPath.return_value = MagicMock()
            MockPath.__truediv__ = Path.__truediv__
            # Easier: just patch the __file__ resolution in the method
            pass

        # Simpler approach — mock at method level using a wrapper
        from gateway.run import GatewayRunner
        runner = _make_runner()

        with patch("gateway.run._hermes_home", tmp_path):
            # The handler does Path(__file__).parent.parent.resolve()
            # We need to make project_root / '.git' not exist.
            # Since Path(__file__) resolves to the real gateway/run.py,
            # project_root will be the real hermes-agent dir (which HAS .git).
            # Patch Path to control this.
            original_path = Path

            class FakePath(type(Path())):
                pass

            # Actually, simplest: just patch the specific file attr
            fake_file = str(fake_root / "gateway" / "run.py")
            (fake_root / "gateway").mkdir(parents=True)
            (fake_root / "gateway" / "run.py").touch()

            with patch("gateway.run.__file__", fake_file):
                result = await runner._handle_update_command(event)

        assert "Not a git repository" in result

    @pytest.mark.asyncio
    async def test_no_hermes_binary(self, tmp_path):
        """Returns error when hermes is not on PATH."""
        runner = _make_runner()
        event = _make_event()

        # Create project dir WITH .git
        fake_root = tmp_path / "project"
        fake_root.mkdir()
        (fake_root / ".git").mkdir()
        (fake_root / "gateway").mkdir()
        (fake_root / "gateway" / "run.py").touch()
        fake_file = str(fake_root / "gateway" / "run.py")

        with patch("gateway.run._hermes_home", tmp_path), \
             patch("gateway.run.__file__", fake_file), \
             patch("shutil.which", return_value=None):
            result = await runner._handle_update_command(event)

        assert "not found on PATH" in result

    @pytest.mark.asyncio
    async def test_writes_pending_marker(self, tmp_path):
        """Writes .update_pending.json with correct platform and chat info."""
        runner = _make_runner()
        event = _make_event(platform=Platform.TELEGRAM, chat_id="99999")

        fake_root = tmp_path / "project"
        fake_root.mkdir()
        (fake_root / ".git").mkdir()
        (fake_root / "gateway").mkdir()
        (fake_root / "gateway" / "run.py").touch()
        fake_file = str(fake_root / "gateway" / "run.py")
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()

        with patch("gateway.run._hermes_home", hermes_home), \
             patch("gateway.run.__file__", fake_file), \
             patch("shutil.which", side_effect=lambda x: "/usr/bin/hermes" if x == "hermes" else "/usr/bin/systemd-run"), \
             patch("subprocess.Popen"):
            result = await runner._handle_update_command(event)

        pending_path = hermes_home / ".update_pending.json"
        assert pending_path.exists()
        data = json.loads(pending_path.read_text())
        assert data["platform"] == "telegram"
        assert data["chat_id"] == "99999"
        assert "timestamp" in data

    @pytest.mark.asyncio
    async def test_spawns_systemd_run(self, tmp_path):
        """Uses systemd-run when available."""
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

        # Verify systemd-run was used
        call_args = mock_popen.call_args[0][0]
        assert call_args[0] == "/usr/bin/systemd-run"
        assert "--scope" in call_args
        assert "Starting Hermes update" in result

    @pytest.mark.asyncio
    async def test_fallback_nohup_when_no_systemd_run(self, tmp_path):
        """Falls back to nohup when systemd-run is not available."""
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

        def which_no_systemd(x):
            if x == "hermes":
                return "/usr/bin/hermes"
            if x == "systemd-run":
                return None
            return None

        with patch("gateway.run._hermes_home", hermes_home), \
             patch("gateway.run.__file__", fake_file), \
             patch("shutil.which", side_effect=which_no_systemd), \
             patch("subprocess.Popen", mock_popen):
            result = await runner._handle_update_command(event)

        # Verify bash -c nohup fallback was used
        call_args = mock_popen.call_args[0][0]
        assert call_args[0] == "bash"
        assert "nohup" in call_args[2]
        assert "Starting Hermes update" in result

    @pytest.mark.asyncio
    async def test_popen_failure_cleans_up(self, tmp_path):
        """Cleans up pending file and returns error on Popen failure."""
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

        with patch("gateway.run._hermes_home", hermes_home), \
             patch("gateway.run.__file__", fake_file), \
             patch("shutil.which", side_effect=lambda x: f"/usr/bin/{x}"), \
             patch("subprocess.Popen", side_effect=OSError("spawn failed")):
            result = await runner._handle_update_command(event)

        assert "Failed to start update" in result
        # Pending file should be cleaned up
        assert not (hermes_home / ".update_pending.json").exists()

    @pytest.mark.asyncio
    async def test_returns_user_friendly_message(self, tmp_path):
        """The success response is user-friendly."""
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

        with patch("gateway.run._hermes_home", hermes_home), \
             patch("gateway.run.__file__", fake_file), \
             patch("shutil.which", side_effect=lambda x: f"/usr/bin/{x}"), \
             patch("subprocess.Popen"):
            result = await runner._handle_update_command(event)

        assert "notify you when it's done" in result


# ---------------------------------------------------------------------------
# _send_update_notification
# ---------------------------------------------------------------------------


class TestSendUpdateNotification:
    """Tests for GatewayRunner._send_update_notification."""

    @pytest.mark.asyncio
    async def test_no_pending_file_is_noop(self, tmp_path):
        """Does nothing when no pending file exists."""
        runner = _make_runner()
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()

        with patch("gateway.run._hermes_home", hermes_home):
            # Should not raise
            await runner._send_update_notification()

    @pytest.mark.asyncio
    async def test_sends_notification_with_output(self, tmp_path):
        """Sends update output to the correct platform and chat."""
        runner = _make_runner()
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()

        # Write pending marker
        pending = {
            "platform": "telegram",
            "chat_id": "67890",
            "user_id": "12345",
            "timestamp": "2026-03-04T21:00:00",
        }
        (hermes_home / ".update_pending.json").write_text(json.dumps(pending))
        (hermes_home / ".update_output.txt").write_text(
            "→ Found 3 new commit(s)\n✓ Code updated!\n✓ Update complete!"
        )

        # Mock the adapter
        mock_adapter = AsyncMock()
        mock_adapter.send = AsyncMock()
        runner.adapters = {Platform.TELEGRAM: mock_adapter}

        with patch("gateway.run._hermes_home", hermes_home):
            await runner._send_update_notification()

        mock_adapter.send.assert_called_once()
        call_args = mock_adapter.send.call_args
        assert call_args[0][0] == "67890"  # chat_id
        assert "Update complete" in call_args[0][1] or "update finished" in call_args[0][1].lower()

    @pytest.mark.asyncio
    async def test_strips_ansi_codes(self, tmp_path):
        """ANSI escape codes are removed from output."""
        runner = _make_runner()
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()

        pending = {"platform": "telegram", "chat_id": "111", "user_id": "222"}
        (hermes_home / ".update_pending.json").write_text(json.dumps(pending))
        (hermes_home / ".update_output.txt").write_text(
            "\x1b[32m✓ Code updated!\x1b[0m\n\x1b[1mDone\x1b[0m"
        )

        mock_adapter = AsyncMock()
        runner.adapters = {Platform.TELEGRAM: mock_adapter}

        with patch("gateway.run._hermes_home", hermes_home):
            await runner._send_update_notification()

        sent_text = mock_adapter.send.call_args[0][1]
        assert "\x1b[" not in sent_text
        assert "Code updated" in sent_text

    @pytest.mark.asyncio
    async def test_truncates_long_output(self, tmp_path):
        """Output longer than 3500 chars is truncated."""
        runner = _make_runner()
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()

        pending = {"platform": "telegram", "chat_id": "111", "user_id": "222"}
        (hermes_home / ".update_pending.json").write_text(json.dumps(pending))
        (hermes_home / ".update_output.txt").write_text("x" * 5000)

        mock_adapter = AsyncMock()
        runner.adapters = {Platform.TELEGRAM: mock_adapter}

        with patch("gateway.run._hermes_home", hermes_home):
            await runner._send_update_notification()

        sent_text = mock_adapter.send.call_args[0][1]
        # Should start with truncation marker
        assert "…" in sent_text
        # Total message should not be absurdly long
        assert len(sent_text) < 4500

    @pytest.mark.asyncio
    async def test_sends_generic_message_when_no_output(self, tmp_path):
        """Sends a success message even if the output file is missing."""
        runner = _make_runner()
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()

        pending = {"platform": "telegram", "chat_id": "111", "user_id": "222"}
        (hermes_home / ".update_pending.json").write_text(json.dumps(pending))
        # No .update_output.txt created

        mock_adapter = AsyncMock()
        runner.adapters = {Platform.TELEGRAM: mock_adapter}

        with patch("gateway.run._hermes_home", hermes_home):
            await runner._send_update_notification()

        sent_text = mock_adapter.send.call_args[0][1]
        assert "restarted successfully" in sent_text

    @pytest.mark.asyncio
    async def test_cleans_up_files_after_notification(self, tmp_path):
        """Both marker and output files are deleted after notification."""
        runner = _make_runner()
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()

        pending_path = hermes_home / ".update_pending.json"
        output_path = hermes_home / ".update_output.txt"
        pending_path.write_text(json.dumps({
            "platform": "telegram", "chat_id": "111", "user_id": "222",
        }))
        output_path.write_text("✓ Done")

        mock_adapter = AsyncMock()
        runner.adapters = {Platform.TELEGRAM: mock_adapter}

        with patch("gateway.run._hermes_home", hermes_home):
            await runner._send_update_notification()

        assert not pending_path.exists()
        assert not output_path.exists()

    @pytest.mark.asyncio
    async def test_cleans_up_on_error(self, tmp_path):
        """Files are cleaned up even if notification fails."""
        runner = _make_runner()
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()

        pending_path = hermes_home / ".update_pending.json"
        output_path = hermes_home / ".update_output.txt"
        pending_path.write_text(json.dumps({
            "platform": "telegram", "chat_id": "111", "user_id": "222",
        }))
        output_path.write_text("✓ Done")

        # Adapter send raises
        mock_adapter = AsyncMock()
        mock_adapter.send.side_effect = RuntimeError("network error")
        runner.adapters = {Platform.TELEGRAM: mock_adapter}

        with patch("gateway.run._hermes_home", hermes_home):
            await runner._send_update_notification()

        # Files should still be cleaned up (finally block)
        assert not pending_path.exists()
        assert not output_path.exists()

    @pytest.mark.asyncio
    async def test_handles_corrupt_pending_file(self, tmp_path):
        """Gracefully handles a malformed pending JSON file."""
        runner = _make_runner()
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()

        pending_path = hermes_home / ".update_pending.json"
        pending_path.write_text("{corrupt json!!")

        with patch("gateway.run._hermes_home", hermes_home):
            # Should not raise
            await runner._send_update_notification()

        # File should be cleaned up
        assert not pending_path.exists()

    @pytest.mark.asyncio
    async def test_no_adapter_for_platform(self, tmp_path):
        """Does not crash if the platform adapter is not connected."""
        runner = _make_runner()
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()

        pending = {"platform": "discord", "chat_id": "111", "user_id": "222"}
        pending_path = hermes_home / ".update_pending.json"
        output_path = hermes_home / ".update_output.txt"
        pending_path.write_text(json.dumps(pending))
        output_path.write_text("Done")

        # Only telegram adapter available, but pending says discord
        mock_adapter = AsyncMock()
        runner.adapters = {Platform.TELEGRAM: mock_adapter}

        with patch("gateway.run._hermes_home", hermes_home):
            await runner._send_update_notification()

        # send should not have been called (wrong platform)
        mock_adapter.send.assert_not_called()
        # Files should still be cleaned up
        assert not pending_path.exists()


# ---------------------------------------------------------------------------
# /update in help and known_commands
# ---------------------------------------------------------------------------


class TestUpdateInHelp:
    """Verify /update appears in help text and known commands set."""

    @pytest.mark.asyncio
    async def test_update_in_help_output(self):
        """The /help output includes /update."""
        runner = _make_runner()
        event = _make_event(text="/help")
        result = await runner._handle_help_command(event)
        assert "/update" in result

    def test_update_is_known_command(self):
        """The /update command is in the help text (proxy for _known_commands)."""
        # _known_commands is local to _handle_message, so we verify by
        # checking the help output includes it.
        from gateway.run import GatewayRunner
        import inspect
        source = inspect.getsource(GatewayRunner._handle_message)
        assert '"update"' in source
