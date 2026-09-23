"""Tests for cron.delivery_fallback.

Covers the definitive-vs-uncertain failure decision, the parent/home fallback target resolution
across both thread models, the redirect notice text, and the redirect driven end-to-end through
``_deliver_result`` against a real send path.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cron.scheduler_delivery import _deliver_result

from cron.delivery_fallback import (
    build_fallback_targets,
    format_fallback_notice,
    is_definitive_delivery_failure,
)


class TestIsDefinitiveDeliveryFailure:
    @pytest.mark.parametrize(
        "error",
        [
            "404 Not Found (error code: 10003): Unknown Channel",  # Discord
            "403 Forbidden (error code: 50001): Missing Access",   # Discord
            "channel_not_found",                                    # Slack
            "is_archived",                                          # Slack
            "Bad Request: thread not found",                        # Telegram
            "Bad Request: topic_closed",                            # Telegram
            "This thread is locked",                                # generic
        ],
    )
    def test_definitive_failures(self, error):
        assert is_definitive_delivery_failure(error) is True

    @pytest.mark.parametrize(
        "error",
        [
            None,
            "",
            "Timeout sending message",
            "503 Service Unavailable",
            "Too Many Requests: retry after 30",
            "some entirely novel provider message",
            "media file not found on disk",
        ],
    )
    def test_uncertain_failures_are_not_definitive(self, error):
        assert is_definitive_delivery_failure(error) is False

    def test_accepts_exception_objects(self):
        assert is_definitive_delivery_failure(Exception("Unknown Channel")) is True
        assert is_definitive_delivery_failure(TimeoutError("timed out")) is False


class TestBuildFallbackTargets:
    def test_discord_model_a_parent_then_home(self):
        """Discord: chat_id IS the thread; redirect to the separate parent id, then home."""
        target = {"platform": "discord", "chat_id": "thread-1", "thread_id": "thread-1"}
        out = build_fallback_targets(
            target,
            parent_chat_id="parent-chan",
            home_chat_id="home-chan",
            home_thread_id=None,
        )
        assert out == [
            {"platform": "discord", "chat_id": "parent-chan", "thread_id": None, "fallback_kind": "parent"},
            {"platform": "discord", "chat_id": "home-chan", "thread_id": None, "fallback_kind": "home"},
        ]

    def test_telegram_model_b_drops_topic_to_channel(self):
        """Telegram: chat_id is the channel, thread_id the topic; drop the topic."""
        target = {"platform": "telegram", "chat_id": "-100chan", "thread_id": "topic-9"}
        out = build_fallback_targets(target, home_chat_id="home-chan")
        assert out[0] == {
            "platform": "telegram",
            "chat_id": "-100chan",
            "thread_id": None,
            "fallback_kind": "parent",
        }
        assert out[1]["fallback_kind"] == "home"
        assert out[1]["chat_id"] == "home-chan"

    def test_home_thread_id_is_preserved(self):
        target = {"platform": "telegram", "chat_id": "c1", "thread_id": None}
        out = build_fallback_targets(target, home_chat_id="home", home_thread_id="home-topic")
        assert out == [
            {"platform": "telegram", "chat_id": "home", "thread_id": "home-topic", "fallback_kind": "home"},
        ]

    def test_flat_dm_with_no_home_has_no_fallbacks(self):
        target = {"platform": "sms", "chat_id": "+15550001111", "thread_id": None}
        assert build_fallback_targets(target) == []

    def test_parent_equal_to_failed_target_is_skipped(self):
        target = {"platform": "discord", "chat_id": "same", "thread_id": "same"}
        out = build_fallback_targets(target, parent_chat_id="same", home_chat_id="home")
        # parent == current chat -> skipped; only home remains.
        assert [t["fallback_kind"] for t in out] == ["home"]

    def test_home_equal_to_failed_target_is_skipped(self):
        target = {"platform": "telegram", "chat_id": "home", "thread_id": None}
        out = build_fallback_targets(target, home_chat_id="home")
        assert out == []

    def test_home_deduped_against_parent(self):
        """If the parent channel and home channel are the same place, list it once."""
        target = {"platform": "discord", "chat_id": "thread-1", "thread_id": "thread-1"}
        out = build_fallback_targets(target, parent_chat_id="shared", home_chat_id="shared")
        assert [t["fallback_kind"] for t in out] == ["parent"]

    def test_model_a_takes_precedence_over_model_b(self):
        """An explicit parent id wins over dropping a differing thread id."""
        target = {"platform": "matrix", "chat_id": "room-thread", "thread_id": "tid"}
        out = build_fallback_targets(target, parent_chat_id="real-parent")
        assert out == [
            {"platform": "matrix", "chat_id": "real-parent", "thread_id": None, "fallback_kind": "parent"},
        ]

    def test_dm_suppresses_home_fallback(self):
        """A 1:1 DM that fails must not escalate to a (shared) home channel."""
        target = {"platform": "telegram", "chat_id": "user-123", "thread_id": None}
        out = build_fallback_targets(target, home_chat_id="group-home", is_direct_message=True)
        assert out == []  # no leak to the shared home channel

    def test_dm_still_drops_topic_to_dm_root(self):
        """A deleted DM *topic* still falls back to the DM root (stays private)."""
        target = {"platform": "telegram", "chat_id": "user-123", "thread_id": "topic-5"}
        out = build_fallback_targets(target, home_chat_id="group-home", is_direct_message=True)
        assert out == [
            {"platform": "telegram", "chat_id": "user-123", "thread_id": None, "fallback_kind": "parent"},
        ]

    def test_non_dm_still_uses_home(self):
        target = {"platform": "slack", "chat_id": "C-archived", "thread_id": None}
        out = build_fallback_targets(target, home_chat_id="slack-home", is_direct_message=False)
        assert [t["fallback_kind"] for t in out] == ["home"]


class TestFormatFallbackNotice:
    def test_parent_notice_prepended(self):
        out = format_fallback_notice("Daily report ready.", "parent")
        assert out.startswith("⚠️")
        assert "parent channel" in out
        assert out.endswith("Daily report ready.")

    def test_home_notice_prepended(self):
        out = format_fallback_notice("Daily report ready.", "home")
        assert out.startswith("⚠️")
        assert "home channel" in out
        assert "Daily report ready." in out

    def test_unknown_kind_returns_content_unchanged(self):
        assert format_fallback_notice("body", "mystery") == "body"

    def test_empty_content_returns_bare_notice(self):
        out = format_fallback_notice("", "parent")
        assert out.startswith("⚠️")
        assert "\n\n" not in out


class TestDeliverResultStaleTargetFallback:
    """Definitive delivery failures redirect to parent/home; uncertain ones don't."""

    def _cfg(self, platform):
        pconfig = MagicMock()
        pconfig.enabled = True
        mock_cfg = MagicMock()
        mock_cfg.platforms = {platform: pconfig}
        return mock_cfg

    def _clear_home_envs(self, monkeypatch):
        for env in (
            "DISCORD_HOME_CHANNEL", "DISCORD_HOME_CHANNEL_THREAD_ID",
            "TELEGRAM_HOME_CHANNEL", "TELEGRAM_HOME_CHANNEL_THREAD_ID",
            "TELEGRAM_CRON_THREAD_ID", "SLACK_HOME_CHANNEL",
        ):
            monkeypatch.delenv(env, raising=False)

    def test_discord_unknown_channel_falls_back_parent_then_home(self, monkeypatch):
        from gateway.config import Platform

        self._clear_home_envs(monkeypatch)
        monkeypatch.setenv("DISCORD_HOME_CHANNEL", "home-chan")

        send_mock = AsyncMock(side_effect=[
            {"error": "404 Not Found (error code: 10003): Unknown Channel"},
            {"success": True},
        ])
        job = {
            "id": "discord-stale",
            "deliver": "origin",
            "origin": {
                "platform": "discord", "chat_id": "old-thread",
                "thread_id": "old-thread", "parent_chat_id": "parent-chan",
            },
        }
        with patch("gateway.config.load_gateway_config", return_value=self._cfg(Platform.DISCORD)), \
             patch("tools.send_message_tool._send_to_platform", new=send_mock), \
             patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}):
            result = _deliver_result(job, "reminder")

        assert result is None
        assert [c.args[2] for c in send_mock.call_args_list] == ["old-thread", "parent-chan"]
        assert send_mock.call_args_list[1].kwargs["thread_id"] is None
        assert "parent channel" in send_mock.call_args_list[1].args[3]

    def test_telegram_topic_deleted_drops_to_channel(self, monkeypatch):
        from gateway.config import Platform

        self._clear_home_envs(monkeypatch)
        send_mock = AsyncMock(side_effect=[
            {"error": "Bad Request: topic_deleted"},
            {"success": True},
        ])
        job = {
            "id": "tg-stale",
            "deliver": "origin",
            "origin": {"platform": "telegram", "chat_id": "-100chan", "thread_id": "topic-9"},
        }
        with patch("gateway.config.load_gateway_config", return_value=self._cfg(Platform.TELEGRAM)), \
             patch("tools.send_message_tool._send_to_platform", new=send_mock), \
             patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}):
            result = _deliver_result(job, "reminder")

        assert result is None
        # Model B: same channel, topic dropped.
        assert [c.args[2] for c in send_mock.call_args_list] == ["-100chan", "-100chan"]
        assert send_mock.call_args_list[1].kwargs["thread_id"] is None
        assert "parent channel" in send_mock.call_args_list[1].args[3]

    def test_slack_archived_channel_falls_back_to_home(self, monkeypatch):
        from gateway.config import Platform

        self._clear_home_envs(monkeypatch)
        monkeypatch.setenv("SLACK_HOME_CHANNEL", "slack-home")
        send_mock = AsyncMock(side_effect=[
            {"error": "is_archived"},
            {"success": True},
        ])
        job = {
            "id": "slack-stale",
            "deliver": "origin",
            # Explicit shared chat_type: a channel origin DOES escalate to home.
            "origin": {"platform": "slack", "chat_id": "C-old", "chat_type": "channel"},
        }
        with patch("gateway.config.load_gateway_config", return_value=self._cfg(Platform.SLACK)), \
             patch("tools.send_message_tool._send_to_platform", new=send_mock), \
             patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}):
            result = _deliver_result(job, "reminder")

        assert result is None
        assert [c.args[2] for c in send_mock.call_args_list] == ["C-old", "slack-home"]
        assert "home channel" in send_mock.call_args_list[1].args[3]

    def test_timeout_does_not_redirect(self, monkeypatch):
        """An uncertain failure (timeout) must not be duplicated to parent/home."""
        from gateway.config import Platform

        self._clear_home_envs(monkeypatch)
        monkeypatch.setenv("DISCORD_HOME_CHANNEL", "home-chan")
        send_mock = AsyncMock(side_effect=TimeoutError("timed out"))
        job = {
            "id": "discord-timeout",
            "deliver": "origin",
            "origin": {
                "platform": "discord", "chat_id": "old-thread",
                "thread_id": "old-thread", "parent_chat_id": "parent-chan",
            },
        }
        with patch("gateway.config.load_gateway_config", return_value=self._cfg(Platform.DISCORD)), \
             patch("tools.send_message_tool._send_to_platform", new=send_mock), \
             patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}):
            result = _deliver_result(job, "reminder")

        assert result is not None
        assert "timed out" in result
        assert send_mock.call_count == 1

    def test_parent_definitive_failure_advances_to_home(self, monkeypatch):
        from gateway.config import Platform

        self._clear_home_envs(monkeypatch)
        monkeypatch.setenv("DISCORD_HOME_CHANNEL", "home-chan")
        send_mock = AsyncMock(side_effect=[
            {"error": "404 Not Found (error code: 10003): Unknown Channel"},
            {"error": "403 Forbidden (error code: 50001): Missing Access"},
            {"success": True},
        ])
        job = {
            "id": "discord-cascade",
            "deliver": "origin",
            "origin": {
                "platform": "discord", "chat_id": "old-thread",
                "thread_id": "old-thread", "parent_chat_id": "parent-chan",
                # A Discord thread is always shared (under a guild channel), so
                # the cascade may reach the home channel when the parent also dies.
                "chat_type": "thread",
            },
        }
        with patch("gateway.config.load_gateway_config", return_value=self._cfg(Platform.DISCORD)), \
             patch("tools.send_message_tool._send_to_platform", new=send_mock), \
             patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}):
            result = _deliver_result(job, "reminder")

        assert result is None
        assert [c.args[2] for c in send_mock.call_args_list] == ["old-thread", "parent-chan", "home-chan"]

    def test_slack_thread_does_not_escalate_to_home(self, monkeypatch):
        """A Slack thread can sit inside a DM, so its chat_type='thread' origin
        must fail closed: parent (the channel) is tried, but home is NOT — unlike
        a Discord thread, which is always shared. Locks the platform distinction
        in _THREAD_ALWAYS_SHARED_PLATFORMS."""
        from gateway.config import Platform

        self._clear_home_envs(monkeypatch)
        monkeypatch.setenv("SLACK_HOME_CHANNEL", "slack-home")
        send_mock = AsyncMock(side_effect=[
            {"error": "is_archived"},  # origin thread's channel gone
            {"error": "is_archived"},  # parent (same channel, no thread) also gone
        ])
        job = {
            "id": "slack-thread-priv",
            "deliver": "origin",
            "origin": {
                "platform": "slack", "chat_id": "C-priv", "thread_id": "ts-1",
                "chat_type": "thread",
            },
        }
        with patch("gateway.config.load_gateway_config", return_value=self._cfg(Platform.SLACK)), \
             patch("tools.send_message_tool._send_to_platform", new=send_mock), \
             patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}):
            result = _deliver_result(job, "reminder")

        assert result is not None  # dropped after parent, not escalated to home
        # Only origin-thread and its parent channel tried; slack-home never used.
        assert [c.args[2] for c in send_mock.call_args_list] == ["C-priv", "C-priv"]

    def test_uncertain_fallback_failure_stops_chain(self, monkeypatch):
        """If a fallback fails for an uncertain reason, stop (don't try home too)."""
        from gateway.config import Platform

        self._clear_home_envs(monkeypatch)
        monkeypatch.setenv("DISCORD_HOME_CHANNEL", "home-chan")
        send_mock = AsyncMock(side_effect=[
            {"error": "404 Not Found (error code: 10003): Unknown Channel"},
            TimeoutError("timed out"),
        ])
        job = {
            "id": "discord-stop",
            "deliver": "origin",
            "origin": {
                "platform": "discord", "chat_id": "old-thread",
                "thread_id": "old-thread", "parent_chat_id": "parent-chan",
            },
        }
        with patch("gateway.config.load_gateway_config", return_value=self._cfg(Platform.DISCORD)), \
             patch("tools.send_message_tool._send_to_platform", new=send_mock), \
             patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}):
            result = _deliver_result(job, "reminder")

        assert result is not None
        assert "timed out" in result
        assert send_mock.call_count == 2  # original + parent; home NOT attempted

    def test_two_dead_targets_redirect_to_home_only_once(self, monkeypatch):
        """Several failing targets redirecting to the same home channel must not
        duplicate the message there (cross-target fallback dedup)."""
        from gateway.config import Platform

        self._clear_home_envs(monkeypatch)
        monkeypatch.setenv("DISCORD_HOME_CHANNEL", "home-chan")
        # chanA fails -> home (delivered); chanB fails -> home (skipped, dup).
        send_mock = AsyncMock(side_effect=[
            {"error": "404 Not Found (error code: 10003): Unknown Channel"},
            {"success": True},
            {"error": "404 Not Found (error code: 10003): Unknown Channel"},
        ])
        job = {
            "id": "discord-dup",
            "deliver": "discord:chanA,discord:chanB",
        }
        with patch("gateway.config.load_gateway_config", return_value=self._cfg(Platform.DISCORD)), \
             patch("tools.send_message_tool._send_to_platform", new=send_mock), \
             patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}):
            result = _deliver_result(job, "reminder")

        assert result is None
        sent_chat_ids = [c.args[2] for c in send_mock.call_args_list]
        # chanA original, home fallback, chanB original; home NOT sent a 2nd time.
        assert sent_chat_ids == ["chanA", "home-chan", "chanB"]
        assert sent_chat_ids.count("home-chan") == 1

    def test_dm_blocked_does_not_leak_to_home_channel(self, monkeypatch):
        """A blocked 1:1 DM must NOT redirect private content to the shared home
        channel — it has no broader-but-private target, so it fails quietly."""
        from gateway.config import Platform

        self._clear_home_envs(monkeypatch)
        monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "-100groupHome")
        send_mock = AsyncMock(side_effect=[
            {"error": "Forbidden: bot was blocked by the user"},
        ])
        job = {
            "id": "tg-dm-blocked",
            "deliver": "origin",
            "origin": {"platform": "telegram", "chat_id": "user-1", "chat_type": "dm"},
        }
        with patch("gateway.config.load_gateway_config", return_value=self._cfg(Platform.TELEGRAM)), \
             patch("tools.send_message_tool._send_to_platform", new=send_mock), \
             patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}):
            result = _deliver_result(job, "private reminder")

        assert result is not None  # recorded as a delivery error, not redirected
        assert send_mock.call_count == 1  # no fallback send to the home group
        assert [c.args[2] for c in send_mock.call_args_list] == ["user-1"]

    def test_legacy_origin_without_chat_type_does_not_leak_to_home(self, monkeypatch):
        """Fail closed: a job created before chat_type was captured has no
        chat_type in its origin. It may be a private DM, so a definitively
        undeliverable send must NOT escalate to the shared home channel — the
        old behavior (missing chat_type == non-DM) could leak private content."""
        from gateway.config import Platform

        self._clear_home_envs(monkeypatch)
        monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "-100groupHome")
        send_mock = AsyncMock(side_effect=[
            {"error": "Forbidden: bot was blocked by the user"},
        ])
        job = {
            "id": "tg-legacy-origin",
            "deliver": "origin",
            # No chat_type — a legacy origin captured before chat_type existed.
            "origin": {"platform": "telegram", "chat_id": "user-1"},
        }
        with patch("gateway.config.load_gateway_config", return_value=self._cfg(Platform.TELEGRAM)), \
             patch("tools.send_message_tool._send_to_platform", new=send_mock), \
             patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}):
            result = _deliver_result(job, "private reminder")

        assert result is not None  # recorded as a delivery error, not redirected
        assert send_mock.call_count == 1  # no fallback send to the home group
        assert [c.args[2] for c in send_mock.call_args_list] == ["user-1"]

    def test_dm_deleted_topic_falls_back_to_dm_root_not_home(self, monkeypatch):
        """A deleted DM topic drops to the DM root (still private); home is not used."""
        from gateway.config import Platform

        self._clear_home_envs(monkeypatch)
        monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "-100groupHome")
        send_mock = AsyncMock(side_effect=[
            {"error": "Bad Request: topic_deleted"},
            {"success": True},
        ])
        job = {
            "id": "tg-dm-topic",
            "deliver": "origin",
            "origin": {
                "platform": "telegram", "chat_id": "user-1",
                "thread_id": "topic-5", "chat_type": "dm",
            },
        }
        with patch("gateway.config.load_gateway_config", return_value=self._cfg(Platform.TELEGRAM)), \
             patch("tools.send_message_tool._send_to_platform", new=send_mock), \
             patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}):
            result = _deliver_result(job, "private reminder")

        assert result is None
        # DM root retried (same chat, no topic); home group never used.
        assert [c.args[2] for c in send_mock.call_args_list] == ["user-1", "user-1"]
        assert send_mock.call_args_list[1].kwargs["thread_id"] is None

    def test_matrix_stale_room_fallback_reuses_live_adapter(self, monkeypatch):
        """A stale Matrix room must redirect through the live gateway adapter,
        not a fresh/ephemeral send.

        This proves ROUTING, not wire-level encryption: unlike the sibling tests
        it does NOT mock ``_send_to_platform``; it drives the actual
        ``_send_to_platform -> _send_matrix_via_adapter -> live adapter`` chain.
        Both the failed origin send and the home redirect must land on the
        persistent live adapter — the one holding the gateway's E2EE session
        (#46310) — with no ephemeral connect/disconnect. That the adapter then
        encrypts on the wire is the Matrix adapter's own contract, covered by its
        adapter tests, not here. This closes the review concern that the redirect
        bypasses ``DeliveryRouter._deliver_to_platform``: ``_send_to_platform`` is
        itself adapter-aware for Matrix, so the redirect reuses the same session."""
        import sys
        from types import SimpleNamespace
        from gateway.config import Platform

        self._clear_home_envs(monkeypatch)
        monkeypatch.setenv("MATRIX_HOME_ROOM", "!bot-room:example.org")

        sends = []
        connect_calls = []

        class LiveMatrixAdapter:
            async def connect(self, *a, **k):
                connect_calls.append("connect")
                return True

            async def disconnect(self):
                connect_calls.append("disconnect")

            async def send(self, chat_id, message, metadata=None):
                sends.append(chat_id)
                if chat_id == "!old:example.org":
                    return SimpleNamespace(success=False, error="M_FORBIDDEN: not in room", message_id=None)
                return SimpleNamespace(success=True, error=None, message_id="$ok")

        fake_runner = SimpleNamespace(adapters={Platform.MATRIX: LiveMatrixAdapter()})
        job = {
            "id": "matrix-stale",
            "deliver": "origin",
            # Explicit shared room: a channel origin may escalate to the home room.
            "origin": {"platform": "matrix", "chat_id": "!old:example.org", "chat_type": "channel"},
        }
        with patch("gateway.config.load_gateway_config", return_value=self._cfg(Platform.MATRIX)), \
             patch("gateway.run._gateway_runner_ref", return_value=fake_runner), \
             patch.dict(sys.modules, {"plugins.platforms.matrix.adapter": SimpleNamespace()}), \
             patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}):
            result = _deliver_result(job, "reminder")

        assert result is None
        # Origin room tried (fails M_FORBIDDEN), then redirected to the home room —
        # BOTH through the live gateway adapter's persistent E2EE session.
        assert sends == ["!old:example.org", "!bot-room:example.org"]
        # No ephemeral connect/disconnect: the live E2EE session was reused (#46310),
        # i.e. the redirect did not fall back to a cleartext/ephemeral send.
        assert connect_calls == []
