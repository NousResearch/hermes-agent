"""Tests for /update gateway slash command.

Tests both the _handle_update_command handler (spawns update process) and
the _send_update_notification startup hook (sends results after restart).
"""

import asyncio
import json
import os
import threading
import time
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
    runner._update_prompt_pending = {}
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

            # Actually, simplest: just patch the specific file attr.
            # The _handle_update_command handler lives in gateway/slash_commands.py
            # (extracted from run.py in the god-file decomposition); it resolves
            # project_root via Path(__file__).parent.parent, so fake that file.
            fake_file = str(fake_root / "gateway" / "slash_commands.py")
            (fake_root / "gateway").mkdir(parents=True)
            (fake_root / "gateway" / "slash_commands.py").touch()

            with patch("gateway.slash_commands.__file__", fake_file):
                result = await runner._handle_update_command(event)

        assert "Not a git repository" in result


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


# ---------------------------------------------------------------------------
# Concurrent /update admission control
# ---------------------------------------------------------------------------


def _fake_checkout(tmp_path):
    """Build a fake checkout + profile home for the /update pre-flight.

    ``_handle_update_command`` resolves ``project_root`` from
    ``gateway/slash_commands.py``'s own ``__file__``, so the fake tree mirrors
    that layout and carries a ``.git`` directory. Returns
    ``(slash_commands_file, hermes_home)``.
    """
    root = tmp_path / "project"
    (root / "gateway").mkdir(parents=True)
    (root / ".git").mkdir()
    slash_commands_file = root / "gateway" / "slash_commands.py"
    slash_commands_file.touch()
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    return str(slash_commands_file), hermes_home


class TestUpdateAdmissionControl:
    """``/update`` must admit exactly one updater per profile.

    ``hermes update`` rewrites the checkout and the virtualenv every session on
    the host shares, and the ``.update_*`` markers are a single-slot mailbox
    holding one requester's routing metadata. Two updaters racing the same
    checkout is a real failure mode: the user double-taps ``/update`` because
    the update takes minutes with no acknowledgement, or two platforms on one
    multiplexed gateway both trigger it.

    None of these tests pre-seed a marker file. A test that writes
    ``.update_pending.json`` itself only proves the handler reads a file the
    test created — it never exercises the window between observing the slot is
    free and taking it, which is where the collision actually happens.
    """

    def test_claim_update_slot_admits_exactly_one_caller(self, tmp_path):
        """The reservation primitive is exclusive, and creates its own marker."""
        from gateway.slash_commands import _claim_update_slot

        pending_path = tmp_path / ".update_pending.json"
        assert not pending_path.exists()

        assert _claim_update_slot(pending_path, 3600.0) is True
        assert pending_path.exists()
        assert _claim_update_slot(pending_path, 3600.0) is False
        assert _claim_update_slot(pending_path, 3600.0) is False

    def test_claim_update_slot_is_exclusive_under_parallel_callers(self, tmp_path):
        """16 threads released together still yield exactly one winner.

        This is the check-to-create gap in isolation. A
        ``if path.exists(): return`` guard fails here because every thread can
        observe the marker missing before any of them creates it.
        """
        from gateway.slash_commands import _claim_update_slot

        pending_path = tmp_path / ".update_pending.json"
        assert not pending_path.exists()

        workers = 16
        barrier = threading.Barrier(workers)
        results = []
        results_lock = threading.Lock()

        def _claim():
            barrier.wait(timeout=15)
            won = _claim_update_slot(pending_path, 3600.0)
            with results_lock:
                results.append(won)

        threads = [threading.Thread(target=_claim) for _ in range(workers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=20)

        assert not any(thread.is_alive() for thread in threads)
        assert len(results) == workers
        assert results.count(True) == 1

    def test_claim_update_slot_reclaims_an_abandoned_reservation(self, tmp_path):
        """A reservation past its TTL is taken over, so /update never wedges.

        An updater killed before it writes its exit code (host reboot, OOM
        kill) leaves a marker nothing will clean up. Without the TTL the guard
        would block /update for the lifetime of the profile.
        """
        from gateway.slash_commands import _claim_update_slot

        pending_path = tmp_path / ".update_pending.json"
        assert _claim_update_slot(pending_path, 3600.0) is True
        assert _claim_update_slot(pending_path, 3600.0) is False

        abandoned = time.time() - 7200
        os.utime(pending_path, (abandoned, abandoned))
        assert _claim_update_slot(pending_path, 3600.0) is True

    @pytest.mark.asyncio
    async def test_second_update_is_rejected_without_preseeding_a_marker(self, tmp_path):
        """The first /update creates the marker; the second is refused.

        Nothing is written to the profile home before the first call, so the
        reject path is driven entirely by state the handler itself produced.
        """
        runner = _make_runner()
        slash_commands_file, hermes_home = _fake_checkout(tmp_path)
        pending_path = hermes_home / ".update_pending.json"
        assert not pending_path.exists()

        first = _make_event(platform=Platform.TELEGRAM,
                            chat_id="first-chat", user_id="first-user")
        second = _make_event(platform=Platform.TELEGRAM,
                             chat_id="second-chat", user_id="second-user")

        mock_watch = MagicMock()
        with patch("gateway.run._hermes_home", hermes_home), \
             patch("gateway.slash_commands.__file__", slash_commands_file), \
             patch.object(runner, "_schedule_update_notification_watch", mock_watch), \
             patch("shutil.which", side_effect=lambda x: f"/usr/bin/{x}"), \
             patch("subprocess.Popen") as mock_popen:
            first_reply = await runner._handle_update_command(first)
            second_reply = await runner._handle_update_command(second)

        # Exactly one updater was launched.
        assert mock_popen.call_count == 1
        assert "Starting Hermes update" in first_reply
        assert "already running" in second_reply.lower()

        # The winner's routing metadata survived intact — the loser did not
        # overwrite it, so the completion notice still reaches the first chat.
        data = json.loads(pending_path.read_text())
        assert data["chat_id"] == "first-chat"
        assert data["user_id"] == "first-user"

        # Both callers get the notification watcher, so the loser still learns
        # how the update turned out.
        assert mock_watch.call_count == 2

    def test_concurrent_update_commands_spawn_exactly_one_updater(self, tmp_path):
        """Two callers racing into the handler produce one updater, not two.

        Both are released from a barrier inside ``_resolve_hermes_bin`` — the
        last step before the slot is claimed — so they enter the claim window
        together, in real threads, against the real filesystem, with no marker
        pre-seeded.
        """
        runner = _make_runner()
        slash_commands_file, hermes_home = _fake_checkout(tmp_path)
        pending_path = hermes_home / ".update_pending.json"
        assert not pending_path.exists()

        callers = 2
        barrier = threading.Barrier(callers)

        def _resolve_hermes_bin_at_barrier():
            barrier.wait(timeout=15)
            return ["/usr/bin/hermes"]

        spawns = []
        spawn_lock = threading.Lock()

        def _record_spawn(*args, **kwargs):
            with spawn_lock:
                spawns.append(args[0])
            return MagicMock()

        replies = []
        replies_lock = threading.Lock()

        def _invoke(chat_id):
            event = _make_event(platform=Platform.TELEGRAM,
                                chat_id=chat_id, user_id=f"user-{chat_id}")
            reply = asyncio.run(runner._handle_update_command(event))
            with replies_lock:
                replies.append(reply)

        with patch("gateway.run._hermes_home", hermes_home), \
             patch("gateway.slash_commands.__file__", slash_commands_file), \
             patch("gateway.run._resolve_hermes_bin", _resolve_hermes_bin_at_barrier), \
             patch.object(runner, "_schedule_update_notification_watch", MagicMock()), \
             patch("shutil.which", side_effect=lambda x: f"/usr/bin/{x}"), \
             patch("subprocess.Popen", side_effect=_record_spawn):
            threads = [
                threading.Thread(target=_invoke, args=(f"chat-{i}",))
                for i in range(callers)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=20)
            assert not any(thread.is_alive() for thread in threads)

        assert len(spawns) == 1, f"expected one updater, got {len(spawns)}"
        assert len(replies) == callers
        assert sum("already running" in reply.lower() for reply in replies) == 1

        # The surviving marker is one caller's complete, parseable metadata —
        # not a half-written or interleaved document.
        data = json.loads(pending_path.read_text())
        assert data["chat_id"] in {"chat-0", "chat-1"}
        assert data["user_id"] == f"user-{data['chat_id']}"

    @pytest.mark.asyncio
    async def test_rejects_when_notifier_claims_a_live_update_mid_admission(self, tmp_path):
        """A marker renamed to `.claimed` mid-admission still blocks the caller.

        ``_send_update_notification`` renames pending -> claimed while an update
        is in flight. Landing that between the ``claimed`` check and the
        exclusive create leaves ``pending`` momentarily absent, so the create
        would succeed and admit a second updater against the live one — and the
        notifier's later ``claimed.replace(pending)`` would clobber the new
        reservation.

        The claimed marker here is written from *inside* the admission window
        to stand in for that concurrent notifier — it is not pre-seeded state
        the guard is handed up front.
        """
        from gateway import slash_commands as slash_commands_module

        runner = _make_runner()
        slash_commands_file, hermes_home = _fake_checkout(tmp_path)
        pending_path = hermes_home / ".update_pending.json"
        claimed_path = hermes_home / ".update_pending.claimed.json"
        event = _make_event(platform=Platform.TELEGRAM, chat_id="second-chat")

        real_claim = slash_commands_module._claim_update_slot

        def _claim_after_notifier_rename(path, ttl_seconds):
            # The notifier claims a live update's marker right here, after this
            # handler already saw claimed_path missing.
            claimed_path.write_text(
                json.dumps({
                    "platform": "telegram",
                    "chat_id": "live-chat",
                    "user_id": "live-user",
                }),
                encoding="utf-8",
            )
            return real_claim(path, ttl_seconds)

        mock_watch = MagicMock()
        with patch("gateway.run._hermes_home", hermes_home), \
             patch("gateway.slash_commands.__file__", slash_commands_file), \
             patch.object(slash_commands_module, "_claim_update_slot",
                          _claim_after_notifier_rename), \
             patch.object(runner, "_schedule_update_notification_watch", mock_watch), \
             patch("shutil.which", side_effect=lambda x: f"/usr/bin/{x}"), \
             patch("subprocess.Popen") as mock_popen:
            reply = await runner._handle_update_command(event)

        mock_popen.assert_not_called()
        assert "already running" in reply.lower()
        mock_watch.assert_called_once()

        # The reservation was handed back, and the live update's claimed marker
        # is untouched so its owner still gets the completion notice.
        assert not pending_path.exists()
        assert json.loads(claimed_path.read_text())["chat_id"] == "live-chat"

    def test_release_update_slot_keeps_a_marker_that_holds_metadata(self, tmp_path):
        """Releasing never deletes someone else's routing metadata.

        If the notifier restores a live update's marker with
        ``claimed_path.replace(pending_path)`` before the release runs, the
        marker is no longer this caller's empty reservation and must survive.
        """
        from gateway.slash_commands import _claim_update_slot, _release_update_slot

        pending_path = tmp_path / ".update_pending.json"

        assert _claim_update_slot(pending_path, 3600.0) is True
        _release_update_slot(pending_path)
        assert not pending_path.exists()

        pending_path.write_text(json.dumps({"chat_id": "live-chat"}), encoding="utf-8")
        _release_update_slot(pending_path)
        assert pending_path.exists()
        assert json.loads(pending_path.read_text())["chat_id"] == "live-chat"

    @pytest.mark.asyncio
    async def test_failed_spawn_releases_the_update_slot(self, tmp_path):
        """A spawn that never started must not block the next /update.

        The slot is reserved before the spawn, so the failure path has to give
        it back — otherwise one transient OSError would wedge /update until the
        reservation TTL expired.
        """
        runner = _make_runner()
        slash_commands_file, hermes_home = _fake_checkout(tmp_path)
        pending_path = hermes_home / ".update_pending.json"
        event = _make_event(platform=Platform.TELEGRAM, chat_id="chat-1")

        with patch("gateway.run._hermes_home", hermes_home), \
             patch("gateway.slash_commands.__file__", slash_commands_file), \
             patch.object(runner, "_schedule_update_notification_watch", MagicMock()), \
             patch("shutil.which", side_effect=lambda x: f"/usr/bin/{x}"), \
             patch("subprocess.Popen", side_effect=OSError("cannot fork")):
            failed_reply = await runner._handle_update_command(event)

        assert "Failed to start update" in failed_reply
        assert not pending_path.exists()
        assert not (hermes_home / ".update_pending.tmp").exists()

        with patch("gateway.run._hermes_home", hermes_home), \
             patch("gateway.slash_commands.__file__", slash_commands_file), \
             patch.object(runner, "_schedule_update_notification_watch", MagicMock()), \
             patch("shutil.which", side_effect=lambda x: f"/usr/bin/{x}"), \
             patch("subprocess.Popen") as mock_popen:
            retry_reply = await runner._handle_update_command(event)

        assert mock_popen.call_count == 1
        assert "Starting Hermes update" in retry_reply


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


# ---------------------------------------------------------------------------
# _send_update_notification
# ---------------------------------------------------------------------------


class TestSendUpdateNotification:
    """Tests for GatewayRunner._send_update_notification."""


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

    @pytest.mark.asyncio
    async def test_completion_notification_tolerates_invalid_utf8_output(self, tmp_path):
        """Completion-only update notifications must not crash on bad bytes."""
        runner = _make_runner()
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()

        pending = {"platform": "discord", "chat_id": "111", "user_id": "222"}
        pending_path = hermes_home / ".update_pending.json"
        output_path = hermes_home / ".update_output.txt"
        exit_code_path = hermes_home / ".update_exit_code"
        pending_path.write_text(json.dumps(pending))
        output_path.write_bytes(b"ok before\ninvalid byte: \x96\ncontinued after\n")
        exit_code_path.write_text("0")

        mock_adapter = AsyncMock()
        runner.adapters = {Platform.DISCORD: mock_adapter}

        with patch("gateway.run._hermes_home", hermes_home):
            delivered = await runner._send_update_notification()

        assert delivered is True
        mock_adapter.send.assert_called_once()
        sent_text = mock_adapter.send.call_args[0][1]
        assert "ok before" in sent_text
        assert "invalid byte" in sent_text
        assert "continued after" in sent_text
        assert "Hermes update finished" in sent_text
        assert not pending_path.exists()
        assert not output_path.exists()
        assert not exit_code_path.exists()


# ---------------------------------------------------------------------------
# /update in help and known_commands
# ---------------------------------------------------------------------------


class TestUpdateInHelp:
    """Verify /update appears in help text and known commands set."""


    def test_update_is_known_command(self):
        """/update dispatches through the gateway's plain-command handler table.

        (Was an inspect.getsource() check for the literal '"update"' in
        _handle_message — a banned source-reading test. The if-chain was
        replaced by _gateway_plain_command_handlers(), so assert the real
        dispatch contract: the table maps "update" to the update handler.)
        """
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        handlers = runner._gateway_plain_command_handlers()
        assert handlers.get("update") == runner._handle_update_command

class TestWatchUpdateProgress:
    @pytest.mark.asyncio
    async def test_invalid_utf8_update_output_does_not_crash_watcher(self, tmp_path):
        runner = _make_runner()
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()

        (hermes_home / ".update_pending.json").write_text(json.dumps({
            "platform": "telegram",
            "chat_id": "67890",
            "user_id": "12345",
        }))
        (hermes_home / ".update_output.txt").write_bytes(
            b"ok before\n\xe2\x9c invalid-continuation: \x96\ncontinued after\n"
        )
        (hermes_home / ".update_exit_code").write_text("0")

        mock_adapter = AsyncMock()
        runner.adapters = {Platform.TELEGRAM: mock_adapter}

        with patch("gateway.run._hermes_home", hermes_home):
            await runner._watch_update_progress(poll_interval=0.01, stream_interval=0.01, timeout=1.0)

        sent = "\n".join(call.args[1] for call in mock_adapter.send.call_args_list)
        assert "ok before" in sent
        assert "continued after" in sent
        assert "Hermes update finished" in sent
        assert not (hermes_home / ".update_pending.json").exists()
