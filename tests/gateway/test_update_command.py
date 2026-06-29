"""Tests for /update gateway slash command.

Tests both the _handle_update_command handler (spawns update process) and
the _send_update_notification startup hook (sends results after restart).
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_event(text="/update", platform=Platform.TELEGRAM,
                user_id="12345", chat_id="67890", thread_id=None):
    """Build a MessageEvent for testing."""
    source = SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        user_name="testuser",
        thread_id=thread_id,
    )
    return MessageEvent(text=text, source=source)


def _make_runner():
    """Create a bare GatewayRunner without calling __init__."""
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    return runner


# ---------------------------------------------------------------------------
# _handle_update_command
# ---------------------------------------------------------------------------


class TestHandleUpdateCommand:
    """Tests for GatewayRunner._handle_update_command."""

    @pytest.mark.asyncio
    async def test_managed_install_returns_package_manager_guidance(self, monkeypatch):
        runner = _make_runner()
        event = _make_event()
        monkeypatch.setenv("HERMES_MANAGED", "homebrew")

        # Guard: prevent any accidental fall-through from spawning a real
        # `hermes update --gateway` against the CI checkout. The managed-install
        # guard should return before Popen is ever reached, but mock it as
        # belt-and-suspenders so a premature return doesn't corrupt the repo.
        with patch("subprocess.Popen") as mock_popen:
            result = await runner._handle_update_command(event)

        assert "managed by Homebrew" in result
        assert "brew upgrade hermes-agent" in result
        mock_popen.assert_not_called()  # must return before reaching Popen

    @pytest.mark.asyncio
    async def test_no_git_directory_unknown_method(self, tmp_path):
        """Returns an error when .git is absent and install method is not pip."""
        runner = _make_runner()
        event = _make_event()
        fake_root = tmp_path / "project"
        fake_root.mkdir()
        (fake_root / "gateway").mkdir(parents=True)
        (fake_root / "gateway" / "slash_commands.py").touch()
        fake_file = str(fake_root / "gateway" / "slash_commands.py")

        with patch("gateway.run._hermes_home", tmp_path), \
             patch("gateway.slash_commands.__file__", fake_file), \
             patch("gateway.slash_commands.detect_install_method",
                   return_value="git"):
            result = await runner._handle_update_command(event)

        assert "Not a git repository" in result

    @pytest.mark.asyncio
    async def test_no_git_directory_pip_install_proceeds(self, tmp_path):
        """pip/pipx installs without .git are allowed to proceed."""
        runner = _make_runner()
        event = _make_event()
        fake_root = tmp_path / "project"
        fake_root.mkdir()
        (fake_root / "gateway").mkdir(parents=True)
        (fake_root / "gateway" / "slash_commands.py").touch()
        fake_file = str(fake_root / "gateway" / "slash_commands.py")

        with patch("gateway.run._hermes_home", tmp_path), \
             patch("gateway.slash_commands.__file__", fake_file), \
             patch("gateway.slash_commands.detect_install_method",
                   return_value="pip"), \
             patch("shutil.which", return_value="/usr/bin/hermes"), \
             patch("subprocess.Popen") as mock_popen:
            result = await runner._handle_update_command(event)

        # Should NOT return the "Not a git repository" error
        assert "Not a git repository" not in result

    @pytest.mark.asyncio
    async def test_no_hermes_binary(self, tmp_path):
        """Returns error when hermes is not on PATH and hermes_cli is not importable."""
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
             patch("shutil.which", return_value=None), \
             patch("importlib.util.find_spec", return_value=None):
            result = await runner._handle_update_command(event)

        assert "Could not locate" in result
        assert "hermes update" in result

    @pytest.mark.asyncio
    async def test_fallback_to_sys_executable(self, tmp_path):
        """Falls back to sys.executable -m hermes_cli.main when hermes not on PATH."""
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
        fake_spec = MagicMock()

        with patch("gateway.run._hermes_home", hermes_home), \
             patch("gateway.run.__file__", fake_file), \
             patch("shutil.which", return_value=None), \
             patch("importlib.util.find_spec", return_value=fake_spec), \
             patch("subprocess.Popen", mock_popen):
            result = await runner._handle_update_command(event)

        assert "Starting Hermes update" in result
        call_args = mock_popen.call_args[0][0]
        # The update_cmd uses sys.executable -m hermes_cli.main
        joined = " ".join(call_args) if isinstance(call_args, list) else call_args
        assert "hermes_cli.main" in joined or "bash" in call_args[0]

    @pytest.mark.asyncio
    async def test_resolve_hermes_bin_prefers_which(self, tmp_path):
        """_resolve_hermes_bin returns argv parts from shutil.which when available."""
        from gateway.run import _resolve_hermes_bin

        with patch("shutil.which", return_value="/custom/path/hermes"):
            result = _resolve_hermes_bin()

        assert result == ["/custom/path/hermes"]

    @pytest.mark.asyncio
    async def test_resolve_hermes_bin_fallback(self):
        """_resolve_hermes_bin falls back to sys.executable argv when which fails."""
        import sys
        from gateway.run import _resolve_hermes_bin

        fake_spec = MagicMock()
        with patch("shutil.which", return_value=None), \
             patch("importlib.util.find_spec", return_value=fake_spec):
            result = _resolve_hermes_bin()

        assert result == [sys.executable, "-m", "hermes_cli.main"]

    @pytest.mark.asyncio
    async def test_resolve_hermes_bin_returns_none_when_both_fail(self):
        """_resolve_hermes_bin returns None when both strategies fail."""
        from gateway.run import _resolve_hermes_bin

        with patch("shutil.which", return_value=None), \
             patch("importlib.util.find_spec", return_value=None):
            result = _resolve_hermes_bin()

        assert result is None

    @pytest.mark.asyncio
    async def test_writes_pending_marker(self, tmp_path):
        """Writes .update_pending.json with correct platform and chat info."""
        runner = _make_runner()
        event = _make_event(platform=Platform.TELEGRAM, chat_id="99999")
        event.message_id = "m-update"

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
             patch("shutil.which", side_effect=lambda x: "/usr/bin/hermes" if x == "hermes" else "/usr/bin/setsid"), \
             patch("subprocess.Popen"):
            result = await runner._handle_update_command(event)

        pending_path = hermes_home / ".update_pending.json"
        assert pending_path.exists()
        data = json.loads(pending_path.read_text())
        assert data["platform"] == "telegram"
        assert data["chat_id"] == "99999"
        assert data["chat_type"] == "dm"
        assert data["message_id"] == "m-update"
        assert "timestamp" in data
        assert not (hermes_home / ".update_exit_code").exists()

    @pytest.mark.asyncio
    async def test_writes_pending_marker_with_thread_id(self, tmp_path):
        """Persists thread_id so update notifications can route back to the thread."""
        runner = _make_runner()
        event = _make_event(
            platform=Platform.TELEGRAM,
            chat_id="99999",
            thread_id="777",
        )
        event.message_id = "m-update-thread"

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
             patch("shutil.which", side_effect=lambda x: "/usr/bin/hermes" if x == "hermes" else "/usr/bin/setsid"), \
             patch("subprocess.Popen"):
            await runner._handle_update_command(event)

        data = json.loads((hermes_home / ".update_pending.json").read_text())
        assert data["thread_id"] == "777"
        assert data["message_id"] == "m-update-thread"

    @pytest.mark.asyncio
    async def test_spawns_setsid(self, tmp_path):
        """Uses setsid when available."""
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

        # Verify setsid was used
        call_args = mock_popen.call_args[0][0]
        assert call_args[0] == "/usr/bin/setsid"
        assert call_args[1] == "bash"
        assert ".update_exit_code" in call_args[-1]
        assert "Starting Hermes update" in result

    @pytest.mark.asyncio
    async def test_fallback_when_no_setsid(self, tmp_path):
        """Falls back to start_new_session=True when setsid is not available."""
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

        def which_no_setsid(x):
            if x == "hermes":
                return "/usr/bin/hermes"
            if x == "setsid":
                return None
            return None

        with patch("gateway.run._hermes_home", hermes_home), \
             patch("gateway.run.__file__", fake_file), \
             patch("shutil.which", side_effect=which_no_setsid), \
             patch("subprocess.Popen", mock_popen):
            result = await runner._handle_update_command(event)

        # Verify plain bash -c fallback (no nohup, no setsid)
        call_args = mock_popen.call_args[0][0]
        assert call_args[0] == "bash"
        assert "nohup" not in call_args[2]
        assert ".update_exit_code" in call_args[2]
        # start_new_session=True should be in kwargs
        call_kwargs = mock_popen.call_args[1]
        assert call_kwargs.get("start_new_session") is True
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
        assert not (hermes_home / ".update_exit_code").exists()

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

        assert "stream progress" in result


# ---------------------------------------------------------------------------
# Platform allowlist gate
# ---------------------------------------------------------------------------


class TestUpdateCommandPlatformGate:
    """Tests for the platform-allowlist gate at the top of
    ``_handle_update_command``.  Built-in messaging platforms are listed in
    ``_UPDATE_ALLOWED_PLATFORMS``; plugin-migrated platforms (discord,
    mattermost, teams, …) are NOT in the frozenset and rely on the
    registry's ``allow_update_command=True`` fallback.  Programmatic
    interfaces (ACP, API server, webhooks) must be blocked.
    """

    @pytest.mark.asyncio
    async def test_blocks_programmatic_interface(self, monkeypatch):
        """``Platform.WEBHOOK`` is not a messaging platform and must be
        blocked by the allowlist gate before any side effects fire."""
        runner = _make_runner()
        event = _make_event(platform=Platform.WEBHOOK)
        monkeypatch.setenv("HERMES_MANAGED", "")

        # Guard: platform gate must fire before any real subprocess spawn.
        with patch("subprocess.Popen") as mock_popen:
            result = await runner._handle_update_command(event)

        # The exact rejection message comes from
        # ``gateway.update.platform_not_messaging`` translation key.
        assert "only available from messaging platforms" in result
        mock_popen.assert_not_called()

    @pytest.mark.asyncio
    async def test_blocks_api_server_platform(self, monkeypatch):
        """``Platform.API_SERVER`` (programmatic, not messaging) must be
        blocked by the allowlist gate.
        """
        runner = _make_runner()
        event = _make_event(platform=Platform.API_SERVER)
        monkeypatch.setenv("HERMES_MANAGED", "")

        with patch("subprocess.Popen") as mock_popen:
            result = await runner._handle_update_command(event)

        assert "only available from messaging platforms" in result
        mock_popen.assert_not_called()

    @pytest.mark.asyncio
    async def test_allows_plugin_platform_via_registry_fallback(self, monkeypatch):
        """A plugin-migrated platform (DISCORD) is no longer in
        ``_UPDATE_ALLOWED_PLATFORMS`` but must still pass the gate via
        the registry's ``allow_update_command=True`` flag.

        This test is the empirical guarantee that removing DISCORD from
        the hardcoded frozenset does not regress the /update command for
        Discord users.
        """
        from gateway.run import GatewayRunner

        # Precondition: DISCORD is NOT in the hardcoded set anymore.
        assert Platform.DISCORD not in GatewayRunner._UPDATE_ALLOWED_PLATFORMS

        # Make sure the plugin registry is populated so the fallback fires.
        from hermes_cli.plugins import PluginManager
        PluginManager().discover_and_load(force=True)
        from gateway.platform_registry import platform_registry
        discord_entry = platform_registry.get("discord")
        assert discord_entry is not None
        assert discord_entry.allow_update_command is True

        runner = _make_runner()
        event = _make_event(platform=Platform.DISCORD)
        monkeypatch.setenv("HERMES_MANAGED", "")

        with patch("subprocess.Popen"):
            result = await runner._handle_update_command(event)

        # The gate must NOT have rejected us — anything other than the
        # ``platform_not_messaging`` rejection string is acceptable here.
        # Later steps may legitimately return success ("Starting Hermes
        # update…") or fail for environment reasons.
        assert "only available from messaging platforms" not in result

    @pytest.mark.asyncio
    async def test_allows_mattermost_via_registry_fallback(self, monkeypatch):
        """Same as DISCORD: MATTERMOST is now plugin-migrated and not in
        the hardcoded frozenset; the registry must keep /update working.
        """
        from gateway.run import GatewayRunner

        assert Platform.MATTERMOST not in GatewayRunner._UPDATE_ALLOWED_PLATFORMS

        from hermes_cli.plugins import PluginManager
        PluginManager().discover_and_load(force=True)
        from gateway.platform_registry import platform_registry
        mm_entry = platform_registry.get("mattermost")
        assert mm_entry is not None
        assert mm_entry.allow_update_command is True

        runner = _make_runner()
        event = _make_event(platform=Platform.MATTERMOST)
        monkeypatch.setenv("HERMES_MANAGED", "")

        with patch("subprocess.Popen"):
            result = await runner._handle_update_command(event)

        assert "only available from messaging platforms" not in result

    @pytest.mark.asyncio
    async def test_allows_homeassistant_via_registry_fallback(self, monkeypatch):
        """Same as DISCORD/MATTERMOST: HOMEASSISTANT is now plugin-migrated
        (PR #40709) and not in the hardcoded frozenset; the registry must
        keep /update working via ``allow_update_command=True``.
        """
        from gateway.run import GatewayRunner

        assert Platform.HOMEASSISTANT not in GatewayRunner._UPDATE_ALLOWED_PLATFORMS

        from hermes_cli.plugins import PluginManager
        PluginManager().discover_and_load(force=True)
        from gateway.platform_registry import platform_registry
        ha_entry = platform_registry.get("homeassistant")
        assert ha_entry is not None
        assert ha_entry.allow_update_command is True

        runner = _make_runner()
        event = _make_event(platform=Platform.HOMEASSISTANT)
        monkeypatch.setenv("HERMES_MANAGED", "")

        with patch("subprocess.Popen"):
            result = await runner._handle_update_command(event)

        assert "only available from messaging platforms" not in result

    @pytest.mark.asyncio
    async def test_allows_builtin_platform_in_allowlist(self, monkeypatch):
        """``Platform.TELEGRAM`` is in the hardcoded allowlist — gate
        must pass without consulting the registry.
        """
        from gateway.run import GatewayRunner

        assert Platform.TELEGRAM in GatewayRunner._UPDATE_ALLOWED_PLATFORMS

        runner = _make_runner()
        event = _make_event(platform=Platform.TELEGRAM)
        monkeypatch.setenv("HERMES_MANAGED", "")

        with patch("subprocess.Popen"):
            result = await runner._handle_update_command(event)

        assert "only available from messaging platforms" not in result


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
    async def test_defers_notification_while_update_still_running(self, tmp_path):
        """Returns False and keeps marker files when the update has not exited yet."""
        runner = _make_runner()
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()

        pending_path = hermes_home / ".update_pending.json"
        pending_path.write_text(json.dumps({
            "platform": "telegram", "chat_id": "67890", "user_id": "12345",
        }))
        (hermes_home / ".update_output.txt").write_text("still running")

        mock_adapter = AsyncMock()
        runner.adapters = {Platform.TELEGRAM: mock_adapter}

        with patch("gateway.run._hermes_home", hermes_home):
            result = await runner._send_update_notification()

        assert result is False
        mock_adapter.send.assert_not_called()
        assert pending_path.exists()

    @pytest.mark.asyncio
    async def test_recovers_from_claimed_pending_file(self, tmp_path):
        """A claimed pending file from a crashed notifier is still deliverable."""
        runner = _make_runner()
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()

        claimed_path = hermes_home / ".update_pending.claimed.json"
        claimed_path.write_text(json.dumps({
            "platform": "telegram", "chat_id": "67890", "user_id": "12345",
        }))
        (hermes_home / ".update_output.txt").write_text("done")
        (hermes_home / ".update_exit_code").write_text("0")

        mock_adapter = AsyncMock()
        runner.adapters = {Platform.TELEGRAM: mock_adapter}

        with patch("gateway.run._hermes_home", hermes_home):
            result = await runner._send_update_notification()

        assert result is True
        mock_adapter.send.assert_called_once()
        assert not claimed_path.exists()

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
        (hermes_home / ".update_exit_code").write_text("0")

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
    async def test_sends_notification_with_thread_metadata(self, tmp_path):
        """Final update notification preserves thread metadata when present."""
        runner = _make_runner()
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()

        pending = {
            "platform": "telegram",
            "chat_id": "67890",
            "chat_type": "dm",
            "thread_id": "777",
            "message_id": "m-update-thread",
            "user_id": "12345",
        }
        (hermes_home / ".update_pending.json").write_text(json.dumps(pending))
        (hermes_home / ".update_output.txt").write_text("done")
        (hermes_home / ".update_exit_code").write_text("0")

        mock_adapter = AsyncMock()
        runner.adapters = {Platform.TELEGRAM: mock_adapter}

        with patch("gateway.run._hermes_home", hermes_home):
            await runner._send_update_notification()

        assert mock_adapter.send.call_args.kwargs["metadata"] == {
            "thread_id": "777",
            "telegram_dm_topic_reply_fallback": True,
            "direct_messages_topic_id": "777",
            "telegram_reply_to_message_id": "m-update-thread",
        }

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
        (hermes_home / ".update_exit_code").write_text("0")

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
        (hermes_home / ".update_exit_code").write_text("0")

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
    async def test_sends_failure_message_when_update_fails(self, tmp_path):
        """Non-zero exit codes produce a failure notification with captured output."""
        runner = _make_runner()
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()

        pending = {"platform": "telegram", "chat_id": "111", "user_id": "222"}
        (hermes_home / ".update_pending.json").write_text(json.dumps(pending))
        (hermes_home / ".update_output.txt").write_text("Traceback: boom")
        (hermes_home / ".update_exit_code").write_text("1")

        mock_adapter = AsyncMock()
        runner.adapters = {Platform.TELEGRAM: mock_adapter}

        with patch("gateway.run._hermes_home", hermes_home):
            result = await runner._send_update_notification()

        assert result is True
        sent_text = mock_adapter.send.call_args[0][1]
        assert "update failed" in sent_text.lower()
        assert "Traceback: boom" in sent_text

    @pytest.mark.asyncio
    async def test_sends_generic_message_when_no_output(self, tmp_path):
        """Sends a success message even if the output file is missing."""
        runner = _make_runner()
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()

        pending = {"platform": "telegram", "chat_id": "111", "user_id": "222"}
        (hermes_home / ".update_pending.json").write_text(json.dumps(pending))
        # No .update_output.txt created
        (hermes_home / ".update_exit_code").write_text("0")

        mock_adapter = AsyncMock()
        runner.adapters = {Platform.TELEGRAM: mock_adapter}

        with patch("gateway.run._hermes_home", hermes_home):
            await runner._send_update_notification()

        sent_text = mock_adapter.send.call_args[0][1]
        assert "finished successfully" in sent_text

    @pytest.mark.asyncio
    async def test_cleans_up_files_after_notification(self, tmp_path):
        """Both marker and output files are deleted after notification."""
        runner = _make_runner()
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()

        pending_path = hermes_home / ".update_pending.json"
        output_path = hermes_home / ".update_output.txt"
        exit_code_path = hermes_home / ".update_exit_code"
        pending_path.write_text(json.dumps({
            "platform": "telegram", "chat_id": "111", "user_id": "222",
        }))
        output_path.write_text("✓ Done")
        exit_code_path.write_text("0")

        mock_adapter = AsyncMock()
        runner.adapters = {Platform.TELEGRAM: mock_adapter}

        with patch("gateway.run._hermes_home", hermes_home):
            await runner._send_update_notification()

        assert not pending_path.exists()
        assert not output_path.exists()
        assert not exit_code_path.exists()

    @pytest.mark.asyncio
    async def test_cleans_up_on_error(self, tmp_path):
        """Files are cleaned up even if notification fails."""
        runner = _make_runner()
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()

        pending_path = hermes_home / ".update_pending.json"
        output_path = hermes_home / ".update_output.txt"
        exit_code_path = hermes_home / ".update_exit_code"
        pending_path.write_text(json.dumps({
            "platform": "telegram", "chat_id": "111", "user_id": "222",
        }))
        output_path.write_text("✓ Done")
        exit_code_path.write_text("0")

        # Adapter send raises
        mock_adapter = AsyncMock()
        mock_adapter.send.side_effect = RuntimeError("network error")
        runner.adapters = {Platform.TELEGRAM: mock_adapter}

        with patch("gateway.run._hermes_home", hermes_home):
            await runner._send_update_notification()

        # Files should still be cleaned up (finally block)
        assert not pending_path.exists()
        assert not output_path.exists()
        assert not exit_code_path.exists()

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
    async def test_no_adapter_for_platform_preserves_markers(self, tmp_path):
        """A finished update whose platform is offline keeps its markers.

        When the target platform's adapter has not reconnected yet, dropping
        the completion markers would silently lose the notification. Instead the
        call defers (returns False) and leaves every marker on disk so a later
        retry can deliver once the platform is back.
        """
        runner = _make_runner()
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()

        pending = {"platform": "discord", "chat_id": "111", "user_id": "222"}
        pending_path = hermes_home / ".update_pending.json"
        output_path = hermes_home / ".update_output.txt"
        exit_code_path = hermes_home / ".update_exit_code"
        pending_path.write_text(json.dumps(pending))
        output_path.write_text("Done")
        exit_code_path.write_text("0")

        # Only telegram adapter available, but pending says discord
        mock_adapter = AsyncMock()
        runner.adapters = {Platform.TELEGRAM: mock_adapter}

        with patch("gateway.run._hermes_home", hermes_home):
            result = await runner._send_update_notification()

        # No send (wrong platform offline) and the result is deferred.
        assert result is False
        mock_adapter.send.assert_not_called()
        # Markers are preserved for a later retry — NOT cleaned up.
        assert pending_path.exists()
        assert output_path.exists()
        assert exit_code_path.exists()
        # The marker stays in its canonical pending location (claim restored).
        assert not (hermes_home / ".update_pending.claimed.json").exists()

    @pytest.mark.asyncio
    async def test_deferred_notification_delivers_after_reconnect(self, tmp_path):
        """A deferred completion is delivered once the platform reconnects.

        Regression for the late-reconnect /update bug: the update finishes while
        the target platform is offline, the markers survive the deferral, and
        the next call (after the adapter is registered) delivers the result and
        cleans up — exactly once.
        """
        runner = _make_runner()
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()

        pending = {"platform": "discord", "chat_id": "111", "user_id": "222"}
        pending_path = hermes_home / ".update_pending.json"
        output_path = hermes_home / ".update_output.txt"
        exit_code_path = hermes_home / ".update_exit_code"
        pending_path.write_text(json.dumps(pending))
        output_path.write_text("✓ Update complete!")
        exit_code_path.write_text("0")

        # First pass: target platform (discord) is still offline → defer.
        with patch("gateway.run._hermes_home", hermes_home):
            first = await runner._send_update_notification()

        assert first is False
        assert pending_path.exists()

        # Platform reconnects: the reconnect watcher adds the adapter back.
        mock_adapter = AsyncMock()
        runner.adapters = {Platform.DISCORD: mock_adapter}

        with patch("gateway.run._hermes_home", hermes_home):
            second = await runner._send_update_notification()

        assert second is True
        mock_adapter.send.assert_called_once()
        sent_text = mock_adapter.send.call_args[0][1]
        assert "Update complete" in sent_text
        # Now everything is cleaned up — no duplicate deliveries possible.
        assert not pending_path.exists()
        assert not output_path.exists()
        assert not exit_code_path.exists()
        assert not (hermes_home / ".update_pending.claimed.json").exists()


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
