"""Tests for cron/scheduler.py — origin resolution, delivery routing, and error logging."""

import contextlib
import itertools
import json
import logging
import os
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from cron.scheduler import _resolve_origin, _resolve_delivery_target, _deliver_result, _send_media_via_adapter, run_job, SILENT_MARKER, _build_job_prompt, _resolve_cron_enabled_toolsets, _merge_mcp_into_per_job_toolsets
from tools.env_passthrough import clear_env_passthrough
from tools.credential_files import clear_credential_files


class TestPerJobToolsetMcpMerge:
    """A per-job enabled_toolsets allowlist must not silently drop MCP servers."""

    CFG = {
        "mcp_servers": {
            "finnhub": {"enabled": True},
            "playwright": {"enabled": True},
            "disabled_one": {"enabled": False},
            "string_enabled": {"enabled": "true"},
            "not_a_dict": "ignored",
        }
    }

    def _enabled_names(self):
        return {"finnhub", "playwright", "string_enabled"}

    def test_native_only_list_gets_all_enabled_mcp_servers(self):
        result = _merge_mcp_into_per_job_toolsets(["web", "terminal"], self.CFG)
        assert result[:2] == ["web", "terminal"]
        assert set(result) == {"web", "terminal"} | self._enabled_names()

    def test_disabled_servers_are_not_added(self):
        result = _merge_mcp_into_per_job_toolsets(["web"], self.CFG)
        assert "disabled_one" not in result

    def test_explicit_mcp_name_is_treated_as_allowlist(self):
        # User named one server -> add nothing further.
        result = _merge_mcp_into_per_job_toolsets(["web", "finnhub"], self.CFG)
        assert result == ["web", "finnhub"]
        assert "playwright" not in result

    def test_no_mcp_sentinel_opts_out_and_is_stripped(self):
        result = _merge_mcp_into_per_job_toolsets(["web", "no_mcp"], self.CFG)
        assert result == ["web"]
        assert not (set(result) & self._enabled_names())

    def test_no_mcp_config_adds_nothing(self):
        result = _merge_mcp_into_per_job_toolsets(["web"], {})
        assert result == ["web"]

    def test_no_duplicate_when_listed_name_also_globally_enabled(self):
        result = _merge_mcp_into_per_job_toolsets(["finnhub", "finnhub"], self.CFG)
        assert result.count("finnhub") == 2  # input dups preserved, none added

    def test_resolver_uses_merge_for_per_job_lists(self):
        job = {"enabled_toolsets": ["web", "terminal"]}
        result = _resolve_cron_enabled_toolsets(job, self.CFG)
        assert set(result) == {"web", "terminal"} | self._enabled_names()

    def test_resolver_empty_per_job_falls_through_to_platform(self):
        # No per-job list -> must delegate to _get_platform_tools (the platform
        # fallback), NOT the per-job merge. Stub the platform resolver and assert
        # it is the path taken and its result is returned.
        job = {"enabled_toolsets": None}
        sentinel = ["web", "finnhub"]
        with patch("hermes_cli.tools_config._get_platform_tools",
                   return_value=set(sentinel)) as m_platform:
            result = _resolve_cron_enabled_toolsets(job, self.CFG)
        m_platform.assert_called_once()
        # _get_platform_tools args: (cfg, "cron")
        assert m_platform.call_args[0][1] == "cron"
        assert set(result) == set(sentinel)


class TestResolveOrigin:
    def test_full_origin(self):
        job = {
            "origin": {
                "platform": "telegram",
                "chat_id": "123456",
                "chat_name": "Test Chat",
                "thread_id": "42",
            }
        }
        result = _resolve_origin(job)
        assert isinstance(result, dict)
        assert result == job["origin"]
        assert result["platform"] == "telegram"
        assert result["chat_id"] == "123456"
        assert result["chat_name"] == "Test Chat"
        assert result["thread_id"] == "42"

    def test_no_origin(self):
        assert _resolve_origin({}) is None
        assert _resolve_origin({"origin": None}) is None

    def test_missing_platform(self):
        job = {"origin": {"chat_id": "123"}}
        assert _resolve_origin(job) is None

    def test_missing_chat_id(self):
        job = {"origin": {"platform": "telegram"}}
        assert _resolve_origin(job) is None

    def test_empty_origin(self):
        job = {"origin": {}}
        assert _resolve_origin(job) is None

    @pytest.mark.parametrize(
        "non_dict_origin",
        [
            "combined-digest-replaces-x-and-y-20260503",
            123,
            ["telegram", "12345"],
            ("platform", "chat_id"),
            42.0,
        ],
    )
    def test_non_dict_origin_returns_none_instead_of_crashing(self, non_dict_origin):
        """Non-dict origins (provenance strings from hand-edited or migrated
        jobs.json) must be treated as missing instead of crashing the
        scheduler tick on ``origin.get('platform')`` with
        ``'str' object has no attribute 'get'`` (#18722).

        Before this guard a job in this state crashed every fire attempt
        forever; ``mark_job_run`` recorded the error but the next tick
        re-loaded the poisoned origin and crashed identically.
        """
        job = {"origin": non_dict_origin}
        assert _resolve_origin(job) is None


class TestResolveDeliveryTarget:
    def test_origin_delivery_preserves_thread_id(self):
        job = {
            "deliver": "origin",
            "origin": {
                "platform": "telegram",
                "chat_id": "-1001",
                "thread_id": "17585",
            },
        }

        assert _resolve_delivery_target(job) == {
            "platform": "telegram",
            "chat_id": "-1001",
            "thread_id": "17585",
        }

    @pytest.mark.parametrize(
        ("platform", "env_var", "chat_id"),
        [
            ("matrix", "MATRIX_HOME_ROOM", "!bot-room:example.org"),
            ("signal", "SIGNAL_HOME_CHANNEL", "+15551234567"),
            ("mattermost", "MATTERMOST_HOME_CHANNEL", "team-town-square"),
            ("sms", "SMS_HOME_CHANNEL", "+15557654321"),
            ("email", "EMAIL_HOME_ADDRESS", "home@example.com"),
            ("dingtalk", "DINGTALK_HOME_CHANNEL", "cidNNN"),
            ("feishu", "FEISHU_HOME_CHANNEL", "oc_home"),
            ("wecom", "WECOM_HOME_CHANNEL", "wecom-home"),
            ("weixin", "WEIXIN_HOME_CHANNEL", "wxid_home"),
            ("qqbot", "QQ_HOME_CHANNEL", "group-openid-home"),
        ],
    )
    def test_origin_delivery_without_origin_falls_back_to_supported_home_channels(
        self, monkeypatch, platform, env_var, chat_id
    ):
        for fallback_env in (
            "MATRIX_HOME_ROOM",
            "MATRIX_HOME_CHANNEL",
            "TELEGRAM_HOME_CHANNEL",
            "DISCORD_HOME_CHANNEL",
            "SLACK_HOME_CHANNEL",
            "SIGNAL_HOME_CHANNEL",
            "MATTERMOST_HOME_CHANNEL",
            "SMS_HOME_CHANNEL",
            "EMAIL_HOME_ADDRESS",
            "DINGTALK_HOME_CHANNEL",
            "BLUEBUBBLES_HOME_CHANNEL",
            "FEISHU_HOME_CHANNEL",
            "WECOM_HOME_CHANNEL",
            "WEIXIN_HOME_CHANNEL",
            "QQ_HOME_CHANNEL",
        ):
            monkeypatch.delenv(fallback_env, raising=False)
        monkeypatch.setenv(env_var, chat_id)

        assert _resolve_delivery_target({"deliver": "origin"}) == {
            "platform": platform,
            "chat_id": chat_id,
            "thread_id": None,
        }

    def test_bare_matrix_delivery_uses_matrix_home_room(self, monkeypatch):
        monkeypatch.delenv("MATRIX_HOME_CHANNEL", raising=False)
        monkeypatch.setenv("MATRIX_HOME_ROOM", "!room123:example.org")

        assert _resolve_delivery_target({"deliver": "matrix"}) == {
            "platform": "matrix",
            "chat_id": "!room123:example.org",
            "thread_id": None,
        }

    def test_bare_platform_delivery_preserves_home_thread_id(self, monkeypatch):
        monkeypatch.setenv("DISCORD_HOME_CHANNEL", "parent-42")
        monkeypatch.setenv("DISCORD_HOME_CHANNEL_THREAD_ID", "topic-7")

        assert _resolve_delivery_target({"deliver": "discord"}) == {
            "platform": "discord",
            "chat_id": "parent-42",
            "thread_id": "topic-7",
        }

    def test_telegram_cron_thread_id_overrides_home_thread_id(self, monkeypatch):
        """TELEGRAM_CRON_THREAD_ID wins over TELEGRAM_HOME_CHANNEL_THREAD_ID for cron (#24409)."""
        monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "-1001234567890")
        monkeypatch.setenv("TELEGRAM_HOME_CHANNEL_THREAD_ID", "5")
        monkeypatch.setenv("TELEGRAM_CRON_THREAD_ID", "42")

        assert _resolve_delivery_target({"deliver": "telegram"}) == {
            "platform": "telegram",
            "chat_id": "-1001234567890",
            "thread_id": "42",
        }

    def test_telegram_cron_thread_id_sets_thread_when_home_thread_unset(self, monkeypatch):
        """TELEGRAM_CRON_THREAD_ID supplies a thread when no home thread is configured."""
        monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "-1001234567890")
        monkeypatch.delenv("TELEGRAM_HOME_CHANNEL_THREAD_ID", raising=False)
        monkeypatch.setenv("TELEGRAM_CRON_THREAD_ID", "42")

        assert _resolve_delivery_target({"deliver": "telegram"}) == {
            "platform": "telegram",
            "chat_id": "-1001234567890",
            "thread_id": "42",
        }

    def test_telegram_cron_thread_id_does_not_leak_to_other_platforms(self, monkeypatch):
        """TELEGRAM_CRON_THREAD_ID is Telegram-only; other platforms keep their own thread resolution."""
        monkeypatch.setenv("DISCORD_HOME_CHANNEL", "parent-42")
        monkeypatch.setenv("DISCORD_HOME_CHANNEL_THREAD_ID", "topic-7")
        monkeypatch.setenv("TELEGRAM_CRON_THREAD_ID", "42")

        assert _resolve_delivery_target({"deliver": "discord"}) == {
            "platform": "discord",
            "chat_id": "parent-42",
            "thread_id": "topic-7",
        }

    def test_explicit_telegram_topic_target_overrides_cron_thread_id(self, monkeypatch):
        """Explicit ``telegram:chat:thread`` targets bypass TELEGRAM_CRON_THREAD_ID."""
        monkeypatch.setenv("TELEGRAM_CRON_THREAD_ID", "999")

        job = {"deliver": "telegram:-1003724596514:17"}
        assert _resolve_delivery_target(job) == {
            "platform": "telegram",
            "chat_id": "-1003724596514",
            "thread_id": "17",
        }

    def test_explicit_telegram_topic_target_with_thread_id(self):
        """deliver: 'telegram:chat_id:thread_id' parses correctly."""
        job = {
            "deliver": "telegram:-1003724596514:17",
        }
        assert _resolve_delivery_target(job) == {
            "platform": "telegram",
            "chat_id": "-1003724596514",
            "thread_id": "17",
        }

    def test_explicit_telegram_topic_thread_survives_bare_directory_match(self):
        """Exact channel-directory matches must not erase an explicit topic id."""
        job = {
            "deliver": "telegram:-1003724596514:17",
        }
        with patch(
            "gateway.channel_directory.resolve_channel_name",
            return_value="-1003724596514",
        ):
            result = _resolve_delivery_target(job)
        assert result == {
            "platform": "telegram",
            "chat_id": "-1003724596514",
            "thread_id": "17",
        }

    def test_explicit_telegram_chat_id_without_thread_id(self):
        """deliver: 'telegram:chat_id' sets thread_id to None."""
        job = {
            "deliver": "telegram:-1003724596514",
        }
        assert _resolve_delivery_target(job) == {
            "platform": "telegram",
            "chat_id": "-1003724596514",
            "thread_id": None,
        }

    def test_human_friendly_label_resolved_via_channel_directory(self):
        """deliver: 'whatsapp:Alice (dm)' resolves to the real JID."""
        job = {"deliver": "whatsapp:Alice (dm)"}
        with patch(
            "gateway.channel_directory.resolve_channel_name",
            return_value="12345678901234@lid",
        ) as resolve_mock:
            result = _resolve_delivery_target(job)
        resolve_mock.assert_called_once_with("whatsapp", "Alice (dm)")
        assert result == {
            "platform": "whatsapp",
            "chat_id": "12345678901234@lid",
            "thread_id": None,
        }

    def test_human_friendly_label_without_suffix_resolved(self):
        """deliver: 'telegram:My Group' resolves without display suffix."""
        job = {"deliver": "telegram:My Group"}
        with patch(
            "gateway.channel_directory.resolve_channel_name",
            return_value="-1009999",
        ):
            result = _resolve_delivery_target(job)
        assert result == {
            "platform": "telegram",
            "chat_id": "-1009999",
            "thread_id": None,
        }

    def test_human_friendly_topic_label_preserves_thread_id(self):
        """Resolved Telegram topic labels should split chat_id and thread_id."""
        job = {"deliver": "telegram:Coaching Chat / topic 17585 (group)"}
        with patch(
            "gateway.channel_directory.resolve_channel_name",
            return_value="-1009999:17585",
        ):
            result = _resolve_delivery_target(job)
        assert result == {
            "platform": "telegram",
            "chat_id": "-1009999",
            "thread_id": "17585",
        }

    def test_raw_id_not_mangled_when_directory_returns_none(self):
        """deliver: 'whatsapp:12345@lid' passes through when directory has no match."""
        job = {"deliver": "whatsapp:12345@lid"}
        with patch(
            "gateway.channel_directory.resolve_channel_name",
            return_value=None,
        ):
            result = _resolve_delivery_target(job)
        assert result == {
            "platform": "whatsapp",
            "chat_id": "12345@lid",
            "thread_id": None,
        }

    def test_bare_platform_uses_matching_origin_chat(self):
        job = {
            "deliver": "telegram",
            "origin": {
                "platform": "telegram",
                "chat_id": "-1001",
                "thread_id": "17585",
            },
        }

        assert _resolve_delivery_target(job) == {
            "platform": "telegram",
            "chat_id": "-1001",
            "thread_id": "17585",
        }

    def test_bare_platform_falls_back_to_home_channel(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "-2002")
        job = {
            "deliver": "telegram",
            "origin": {
                "platform": "discord",
                "chat_id": "abc",
            },
        }

        assert _resolve_delivery_target(job) == {
            "platform": "telegram",
            "chat_id": "-2002",
            "thread_id": None,
        }

    def test_explicit_discord_topic_target_with_thread_id(self):
        """deliver: 'discord:chat_id:thread_id' parses correctly."""
        job = {
            "deliver": "discord:-1001234567890:17585",
        }
        assert _resolve_delivery_target(job) == {
            "platform": "discord",
            "chat_id": "-1001234567890",
            "thread_id": "17585",
        }

    def test_explicit_discord_chat_id_without_thread_id(self):
        """deliver: 'discord:chat_id' sets thread_id to None."""
        job = {
            "deliver": "discord:9876543210",
        }
        assert _resolve_delivery_target(job) == {
            "platform": "discord",
            "chat_id": "9876543210",
            "thread_id": None,
        }

    def test_explicit_discord_channel_without_thread(self):
        """deliver: 'discord:1001234567890' resolves via explicit platform:chat_id path."""
        job = {
            "deliver": "discord:1001234567890",
        }
        result = _resolve_delivery_target(job)
        assert result == {
            "platform": "discord",
            "chat_id": "1001234567890",
            "thread_id": None,
        }

    def test_list_form_deliver_is_normalized(self, monkeypatch):
        """deliver=['telegram'] (Python list) should resolve like 'telegram' string.

        Regression test for #17139: MCP clients / scripts that pass the deliver
        field as an array-shaped value used to fail with "no delivery target
        resolved for deliver=['telegram']" because ``str(['telegram'])`` was
        passed through to ``split(',')`` verbatim.
        """
        monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "-4004")
        job = {
            "deliver": ["telegram"],
            "origin": None,
        }

        assert _resolve_delivery_target(job) == {
            "platform": "telegram",
            "chat_id": "-4004",
            "thread_id": None,
        }

    def test_list_form_multiple_platforms_normalized(self, monkeypatch):
        """deliver=['telegram', 'discord'] resolves to multiple targets."""
        from cron.scheduler import _resolve_delivery_targets

        monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "-111")
        monkeypatch.setenv("DISCORD_HOME_CHANNEL", "-222")
        job = {"deliver": ["telegram", "discord"], "origin": None}

        targets = _resolve_delivery_targets(job)
        platforms = sorted(t["platform"] for t in targets)
        assert platforms == ["discord", "telegram"]

    def test_empty_list_form_deliver_resolves_to_local(self):
        """deliver=[] is treated as local (no delivery)."""
        from cron.scheduler import _resolve_delivery_targets

        assert _resolve_delivery_targets({"deliver": []}) == []


class TestRoutingIntents:
    """``all`` routing intent expands at fire time."""

    def test_all_expands_to_every_connected_home_channel(self, monkeypatch):
        """deliver='all' fans out to every platform with a configured home channel."""
        from cron.scheduler import _resolve_delivery_targets

        monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "-111")
        monkeypatch.setenv("DISCORD_HOME_CHANNEL", "-222")
        monkeypatch.setenv("SLACK_HOME_CHANNEL", "C333")
        # Sanity: platforms without the env var must NOT appear in the expansion.
        monkeypatch.delenv("SIGNAL_HOME_CHANNEL", raising=False)
        monkeypatch.delenv("MATRIX_HOME_ROOM", raising=False)

        targets = _resolve_delivery_targets({"deliver": "all", "origin": None})
        platforms = sorted(t["platform"] for t in targets)

        assert "telegram" in platforms
        assert "discord" in platforms
        assert "slack" in platforms
        assert "signal" not in platforms
        assert "matrix" not in platforms

    def test_all_combines_with_explicit_target_and_dedups(self, monkeypatch):
        """'telegram:-999,all' yields every home channel + the explicit target without dupes."""
        from cron.scheduler import _resolve_delivery_targets

        monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "-111")
        monkeypatch.setenv("DISCORD_HOME_CHANNEL", "-222")

        # Explicit telegram target precedes 'all'. Expansion adds discord;
        # the dedup pass collapses any (platform, chat_id, thread_id) repeats.
        job = {"deliver": "telegram:-999,all", "origin": None}
        targets = _resolve_delivery_targets(job)

        platforms = sorted(t["platform"].lower() for t in targets)
        assert "telegram" in platforms
        assert "discord" in platforms
        # Every target is unique on (platform, chat_id, thread_id).
        keys = [(t["platform"].lower(), str(t["chat_id"]), t.get("thread_id")) for t in targets]
        assert len(keys) == len(set(keys))

    def test_all_with_no_connected_channels_returns_empty(self, monkeypatch):
        """deliver='all' with nothing connected returns [] — delivery is recorded as failed upstream."""
        from cron.scheduler import _resolve_delivery_targets

        for var in ("TELEGRAM_HOME_CHANNEL", "DISCORD_HOME_CHANNEL", "SLACK_HOME_CHANNEL",
                    "SIGNAL_HOME_CHANNEL", "MATRIX_HOME_ROOM", "MATTERMOST_HOME_CHANNEL",
                    "SMS_HOME_CHANNEL", "EMAIL_HOME_ADDRESS", "DINGTALK_HOME_CHANNEL",
                    "FEISHU_HOME_CHANNEL", "WECOM_HOME_CHANNEL", "WEIXIN_HOME_CHANNEL",
                    "BLUEBUBBLES_HOME_CHANNEL", "QQBOT_HOME_CHANNEL", "QQ_HOME_CHANNEL"):
            monkeypatch.delenv(var, raising=False)

        assert _resolve_delivery_targets({"deliver": "all", "origin": None}) == []

    def test_origin_comma_all_preserves_origin_first(self, monkeypatch):
        """'origin,all' delivers to the origin platform plus every other home channel."""
        from cron.scheduler import _resolve_delivery_targets

        monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "-111")
        monkeypatch.setenv("DISCORD_HOME_CHANNEL", "-222")

        job = {
            "deliver": "origin,all",
            "origin": {"platform": "discord", "chat_id": "888"},
        }
        targets = _resolve_delivery_targets(job)
        platforms = sorted(t["platform"].lower() for t in targets)
        assert "telegram" in platforms
        assert "discord" in platforms

        # The origin's explicit chat_id (888) wins the dedup race over the
        # discord home channel (-222) because origin is resolved first.
        discord = next(t for t in targets if t["platform"].lower() == "discord")
        assert discord["chat_id"] == "888"

    def test_all_token_case_insensitive(self, monkeypatch):
        """'ALL' / 'All' / 'all' are all recognized."""
        from cron.scheduler import _resolve_delivery_targets

        monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "-111")
        monkeypatch.setenv("DISCORD_HOME_CHANNEL", "-222")

        for token in ("ALL", "All", "all"):
            targets = _resolve_delivery_targets({"deliver": token, "origin": None})
            platforms = sorted(t["platform"].lower() for t in targets)
            assert platforms == ["discord", "telegram"], f"token={token!r} -> {platforms}"


class TestDeliverResultWrapping:
    """Verify that cron deliveries are wrapped with header/footer and no longer mirrored."""

    def _safe_media_path(self, tmp_path, monkeypatch, name, data=b"media"):
        root = tmp_path / "media-cache"
        media_file = root / name
        media_file.parent.mkdir(parents=True, exist_ok=True)
        media_file.write_bytes(data)
        monkeypatch.setattr(
            "gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS",
            (root,),
        )
        return media_file.resolve()

    def test_delivery_wraps_content_with_header_and_footer(self):
        """Delivered content should include task name header and agent-invisible note."""
        from gateway.config import Platform

        pconfig = MagicMock()
        pconfig.enabled = True
        mock_cfg = MagicMock()
        mock_cfg.platforms = {Platform.TELEGRAM: pconfig}

        with patch("gateway.config.load_gateway_config", return_value=mock_cfg), \
             patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})) as send_mock:
            job = {
                "id": "test-job",
                "name": "daily-report",
                "deliver": "origin",
                "origin": {"platform": "telegram", "chat_id": "123"},
            }
            _deliver_result(job, "Here is today's summary.")

        send_mock.assert_called_once()
        sent_content = send_mock.call_args.kwargs.get("content") or send_mock.call_args[0][-1]
        assert "Cronjob Response: daily-report" in sent_content
        assert "(job_id: test-job)" in sent_content
        assert "-------------" in sent_content
        assert "Here is today's summary." in sent_content
        assert "To stop or manage this job" in sent_content

    def test_delivery_uses_job_id_when_no_name(self):
        """When a job has no name, the wrapper should fall back to job id."""
        from gateway.config import Platform

        pconfig = MagicMock()
        pconfig.enabled = True
        mock_cfg = MagicMock()
        mock_cfg.platforms = {Platform.TELEGRAM: pconfig}

        with patch("gateway.config.load_gateway_config", return_value=mock_cfg), \
             patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})) as send_mock:
            job = {
                "id": "abc-123",
                "deliver": "origin",
                "origin": {"platform": "telegram", "chat_id": "123"},
            }
            _deliver_result(job, "Output.")

        sent_content = send_mock.call_args.kwargs.get("content") or send_mock.call_args[0][-1]
        assert "Cronjob Response: abc-123" in sent_content

    def test_delivery_skips_wrapping_when_config_disabled(self):
        """When cron.wrap_response is false, deliver raw content without header/footer."""
        from gateway.config import Platform

        pconfig = MagicMock()
        pconfig.enabled = True
        mock_cfg = MagicMock()
        mock_cfg.platforms = {Platform.TELEGRAM: pconfig}

        with patch("gateway.config.load_gateway_config", return_value=mock_cfg), \
             patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})) as send_mock, \
             patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}):
            job = {
                "id": "test-job",
                "name": "daily-report",
                "deliver": "origin",
                "origin": {"platform": "telegram", "chat_id": "123"},
            }
            _deliver_result(job, "Clean output only.")

        send_mock.assert_called_once()
        sent_content = send_mock.call_args.kwargs.get("content") or send_mock.call_args[0][-1]
        assert sent_content == "Clean output only."
        assert "Cronjob Response" not in sent_content
        assert "The agent cannot see" not in sent_content

    def test_delivery_extracts_media_tags_before_send(self, tmp_path, monkeypatch):
        """Cron delivery should pass MEDIA attachments separately to the send helper."""
        from gateway.config import Platform
        media_path = self._safe_media_path(tmp_path, monkeypatch, "test-voice.ogg")

        pconfig = MagicMock()
        pconfig.enabled = True
        mock_cfg = MagicMock()
        mock_cfg.platforms = {Platform.TELEGRAM: pconfig}

        with patch("gateway.config.load_gateway_config", return_value=mock_cfg), \
             patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})) as send_mock, \
             patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}):
            job = {
                "id": "voice-job",
                "deliver": "origin",
                "origin": {"platform": "telegram", "chat_id": "123"},
            }
            _deliver_result(job, f"Title\nMEDIA:{media_path}")

        send_mock.assert_called_once()
        args, kwargs = send_mock.call_args
        # Text content should have MEDIA: tag stripped
        assert "MEDIA:" not in args[3]
        assert "Title" in args[3]
        # Media files should be forwarded separately
        assert kwargs["media_files"] == [(str(media_path), False)]

    def test_live_adapter_sends_media_as_attachments(self, tmp_path, monkeypatch):
        """When a live adapter is available, MEDIA files should be sent as native
        platform attachments (e.g., Discord voice, Telegram audio) rather than
        as literal 'MEDIA:/path' text."""
        from gateway.config import Platform
        from concurrent.futures import Future
        media_path = self._safe_media_path(tmp_path, monkeypatch, "cron-voice.mp3")

        adapter = AsyncMock()
        adapter.send.return_value = MagicMock(success=True)
        adapter.send_voice.return_value = MagicMock(success=True)

        pconfig = MagicMock()
        pconfig.enabled = True
        mock_cfg = MagicMock()
        mock_cfg.platforms = {Platform.DISCORD: pconfig}

        loop = MagicMock()
        loop.is_running.return_value = True

        # run_coroutine_threadsafe returns concurrent.futures.Future (has timeout kwarg)
        def fake_run_coro(coro, _loop):
            # Actually run the routed coroutine (router._deliver_to_platform)
            # so the underlying adapter.send is invoked, then wrap the real
            # result in a completed Future (matching run_coroutine_threadsafe).
            import asyncio as _asyncio
            future = Future()
            try:
                future.set_result(_asyncio.run(coro))
            except BaseException as _e:  # noqa: BLE001
                future.set_exception(_e)
            return future

        job = {
            "id": "tts-job",
            "deliver": "origin",
            "origin": {"platform": "discord", "chat_id": "9876"},
        }

        with patch("gateway.config.load_gateway_config", return_value=mock_cfg), \
             patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
             patch("asyncio.run_coroutine_threadsafe", side_effect=fake_run_coro):
            _deliver_result(
                job,
                f"Here is TTS\nMEDIA:{media_path}",
                adapters={Platform.DISCORD: adapter},
                loop=loop,
            )

        # Text should be sent without the MEDIA tag
        adapter.send.assert_called_once()
        text_sent = adapter.send.call_args[0][1]
        assert "MEDIA:" not in text_sent
        assert "Here is TTS" in text_sent

        # Audio file should be sent as a voice attachment
        adapter.send_voice.assert_called_once()
        voice_call = adapter.send_voice.call_args
        assert voice_call[1]["audio_path"] == str(media_path)

    def test_live_adapter_routes_image_to_send_image_file(self, tmp_path, monkeypatch):
        """Image MEDIA files should be routed to send_image_file, not send_voice."""
        from gateway.config import Platform
        from concurrent.futures import Future
        media_path = self._safe_media_path(tmp_path, monkeypatch, "chart.png")

        adapter = AsyncMock()
        adapter.send.return_value = MagicMock(success=True)
        adapter.send_image_file.return_value = MagicMock(success=True)

        pconfig = MagicMock()
        pconfig.enabled = True
        mock_cfg = MagicMock()
        mock_cfg.platforms = {Platform.DISCORD: pconfig}

        loop = MagicMock()
        loop.is_running.return_value = True

        def fake_run_coro(coro, _loop):
            # Actually run the routed coroutine (router._deliver_to_platform)
            # so the underlying adapter.send is invoked, then wrap the real
            # result in a completed Future (matching run_coroutine_threadsafe).
            import asyncio as _asyncio
            future = Future()
            try:
                future.set_result(_asyncio.run(coro))
            except BaseException as _e:  # noqa: BLE001
                future.set_exception(_e)
            return future

        job = {
            "id": "img-job",
            "deliver": "origin",
            "origin": {"platform": "discord", "chat_id": "1234"},
        }

        with patch("gateway.config.load_gateway_config", return_value=mock_cfg), \
             patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
             patch("asyncio.run_coroutine_threadsafe", side_effect=fake_run_coro):
            _deliver_result(
                job,
                f"Chart attached\nMEDIA:{media_path}",
                adapters={Platform.DISCORD: adapter},
                loop=loop,
            )

        adapter.send_image_file.assert_called_once()
        assert adapter.send_image_file.call_args[1]["image_path"] == str(media_path)
        adapter.send_voice.assert_not_called()

    def test_live_adapter_media_only_no_text(self, tmp_path, monkeypatch):
        """When content is ONLY a MEDIA tag with no text, media should still be sent."""
        from gateway.config import Platform
        from concurrent.futures import Future
        media_path = self._safe_media_path(tmp_path, monkeypatch, "voice.ogg")

        adapter = AsyncMock()
        adapter.send_voice.return_value = MagicMock(success=True)

        pconfig = MagicMock()
        pconfig.enabled = True
        mock_cfg = MagicMock()
        mock_cfg.platforms = {Platform.TELEGRAM: pconfig}

        loop = MagicMock()
        loop.is_running.return_value = True

        def fake_run_coro(coro, _loop):
            # Actually run the routed coroutine (router._deliver_to_platform)
            # so the underlying adapter.send is invoked, then wrap the real
            # result in a completed Future (matching run_coroutine_threadsafe).
            import asyncio as _asyncio
            future = Future()
            try:
                future.set_result(_asyncio.run(coro))
            except BaseException as _e:  # noqa: BLE001
                future.set_exception(_e)
            return future

        job = {
            "id": "voice-only",
            "deliver": "origin",
            "origin": {"platform": "telegram", "chat_id": "999"},
        }

        with patch("gateway.config.load_gateway_config", return_value=mock_cfg), \
             patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
             patch("asyncio.run_coroutine_threadsafe", side_effect=fake_run_coro):
            _deliver_result(
                job,
                f"[[audio_as_voice]]\nMEDIA:{media_path}",
                adapters={Platform.TELEGRAM: adapter},
                loop=loop,
            )

        # Text send should NOT be called (no text after stripping MEDIA tag)
        adapter.send.assert_not_called()
        # Audio should still be delivered as a voice bubble
        adapter.send_voice.assert_called_once()

    def test_live_adapter_sends_cleaned_text_not_raw(self):
        """The live adapter path must send cleaned text (MEDIA tags stripped),
        not the raw delivery_content with embedded MEDIA: tags."""
        from gateway.config import Platform
        from concurrent.futures import Future

        adapter = AsyncMock()
        adapter.send.return_value = MagicMock(success=True)

        pconfig = MagicMock()
        pconfig.enabled = True
        mock_cfg = MagicMock()
        mock_cfg.platforms = {Platform.TELEGRAM: pconfig}

        loop = MagicMock()
        loop.is_running.return_value = True

        def fake_run_coro(coro, _loop):
            # Actually run the routed coroutine (router._deliver_to_platform)
            # so the underlying adapter.send is invoked, then wrap the real
            # result in a completed Future (matching run_coroutine_threadsafe).
            import asyncio as _asyncio
            future = Future()
            try:
                future.set_result(_asyncio.run(coro))
            except BaseException as _e:  # noqa: BLE001
                future.set_exception(_e)
            return future

        job = {
            "id": "img-job",
            "deliver": "origin",
            "origin": {"platform": "telegram", "chat_id": "555"},
        }

        with patch("gateway.config.load_gateway_config", return_value=mock_cfg), \
             patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
             patch("asyncio.run_coroutine_threadsafe", side_effect=fake_run_coro):
            _deliver_result(
                job,
                "Report\nMEDIA:/tmp/chart.png",
                adapters={Platform.TELEGRAM: adapter},
                loop=loop,
            )

        text_sent = adapter.send.call_args[0][1]
        assert "MEDIA:" not in text_sent
        assert "Report" in text_sent

    def test_no_mirror_to_session_call(self):
        """Cron deliveries should NOT mirror into the gateway session."""
        from gateway.config import Platform

        pconfig = MagicMock()
        pconfig.enabled = True
        mock_cfg = MagicMock()
        mock_cfg.platforms = {Platform.TELEGRAM: pconfig}

        with patch("gateway.config.load_gateway_config", return_value=mock_cfg), \
             patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})), \
             patch("gateway.mirror.mirror_to_session") as mirror_mock:
            job = {
                "id": "test-job",
                "deliver": "origin",
                "origin": {"platform": "telegram", "chat_id": "123"},
            }
            _deliver_result(job, "Hello!")

        mirror_mock.assert_not_called()

    def test_origin_delivery_preserves_thread_id(self):
        """Origin delivery should forward thread_id to the send helper."""
        from gateway.config import Platform

        pconfig = MagicMock()
        pconfig.enabled = True
        mock_cfg = MagicMock()
        mock_cfg.platforms = {Platform.TELEGRAM: pconfig}

        job = {
            "id": "test-job",
            "name": "topic-job",
            "deliver": "origin",
            "origin": {
                "platform": "telegram",
                "chat_id": "-1001",
                "thread_id": "17585",
            },
        }

        with patch("gateway.config.load_gateway_config", return_value=mock_cfg), \
             patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})) as send_mock:
            _deliver_result(job, "hello")

        send_mock.assert_called_once()
        assert send_mock.call_args.kwargs["thread_id"] == "17585"


class TestDeliverResultErrorReturns:
    """Verify _deliver_result returns error strings on failure, None on success."""

    def test_returns_error_when_platform_disabled(self):
        from gateway.config import Platform

        pconfig = MagicMock()
        pconfig.enabled = False
        mock_cfg = MagicMock()
        mock_cfg.platforms = {Platform.TELEGRAM: pconfig}

        with patch("gateway.config.load_gateway_config", return_value=mock_cfg):
            job = {
                "id": "disabled",
                "deliver": "origin",
                "origin": {"platform": "telegram", "chat_id": "123"},
            }
            result = _deliver_result(job, "Output.")
        assert result is not None
        assert "not configured" in result

    def test_returns_error_for_unresolved_target(self, monkeypatch):
        """Non-local delivery with no resolvable target should return an error."""
        monkeypatch.delenv("TELEGRAM_HOME_CHANNEL", raising=False)
        job = {"id": "no-target", "deliver": "telegram"}
        result = _deliver_result(job, "Output.")
        assert result is not None
        assert "no delivery target" in result


class TestRunJobSessionPersistence:
    def test_run_job_passes_session_db_and_cron_platform(self, tmp_path):
        job = {
            "id": "test-job",
            "name": "test",
            "prompt": "hello",
        }
        fake_db = MagicMock()

        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("hermes_cli.env_loader.load_hermes_dotenv"), \
             patch("hermes_cli.env_loader.reset_secret_source_cache"), \
             patch("hermes_state.SessionDB", return_value=fake_db), \
             patch(
                 "hermes_cli.runtime_provider.resolve_runtime_provider",
                 return_value={
                     "api_key": "test-key",
                     "base_url": "https://example.invalid/v1",
                     "provider": "openrouter",
                     "api_mode": "chat_completions",
                 },
             ), \
             patch("run_agent.AIAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {"final_response": "ok"}
            mock_agent_cls.return_value = mock_agent

            success, output, final_response, error = run_job(job)

        assert success is True
        assert error is None
        assert final_response == "ok"
        assert "ok" in output

        kwargs = mock_agent_cls.call_args.kwargs
        assert kwargs["session_db"] is fake_db
        assert kwargs["platform"] == "cron"
        assert kwargs["session_id"].startswith("cron_test-job_")
        fake_db.end_session.assert_called_once()
        call_args = fake_db.end_session.call_args
        assert call_args[0][0].startswith("cron_test-job_")
        assert call_args[0][1] == "cron_complete"
        fake_db.close.assert_called_once()
        mock_agent.close.assert_called_once()

    def test_run_job_suppresses_empty_turn_explainer(self, tmp_path):
        """An empty model turn becomes the '⚠️ No reply…' explainer (#34452).
        For cron, that abnormal-empty explainer must be treated as empty so it
        is suppressed instead of delivered (Manfredi's Telegram symptom)."""
        from run_agent import AIAgent
        explainer = AIAgent._format_turn_completion_explanation("empty_response_exhausted")
        assert explainer  # sanity: the explainer text exists
        job = {"id": "test-job", "name": "test", "prompt": "hello"}
        fake_db = MagicMock()

        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("hermes_cli.env_loader.load_hermes_dotenv"), \
             patch("hermes_cli.env_loader.reset_secret_source_cache"), \
             patch("hermes_state.SessionDB", return_value=fake_db), \
             patch(
                 "hermes_cli.runtime_provider.resolve_runtime_provider",
                 return_value={
                     "api_key": "test-key",
                     "base_url": "https://example.invalid/v1",
                     "provider": "openrouter",
                     "api_mode": "chat_completions",
                 },
             ), \
             patch("run_agent.AIAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent._format_turn_completion_explanation = (
                AIAgent._format_turn_completion_explanation
            )
            mock_agent.run_conversation.return_value = {
                "final_response": explainer,
                "turn_exit_reason": "empty_response_exhausted",
            }
            mock_agent_cls.return_value = mock_agent
            # Patch the class staticmethod the scheduler calls.
            mock_agent_cls._format_turn_completion_explanation = (
                AIAgent._format_turn_completion_explanation
            )

            success, output, final_response, error = run_job(job)

        # The explainer is stripped to empty inside run_job; the downstream
        # firing body (process_job) then suppresses delivery and marks the run
        # a soft failure via its empty-response guard.  Here we assert the
        # load-bearing transform: the "⚠️ No reply…" text never reaches delivery.
        assert final_response == ""

    def test_run_job_real_report_on_empty_reason_still_delivers(self, tmp_path):
        """Defensive: a real report must NOT be suppressed even if the result
        carries an abnormal turn_exit_reason — only the exact explainer text is."""
        from run_agent import AIAgent
        job = {"id": "test-job", "name": "test", "prompt": "hello"}
        fake_db = MagicMock()

        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("hermes_cli.env_loader.load_hermes_dotenv"), \
             patch("hermes_cli.env_loader.reset_secret_source_cache"), \
             patch("hermes_state.SessionDB", return_value=fake_db), \
             patch(
                 "hermes_cli.runtime_provider.resolve_runtime_provider",
                 return_value={
                     "api_key": "test-key",
                     "base_url": "https://example.invalid/v1",
                     "provider": "openrouter",
                     "api_mode": "chat_completions",
                 },
             ), \
             patch("run_agent.AIAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {
                "final_response": "Daily report: 4 PRs merged.",
                "turn_exit_reason": "empty_response_exhausted",
            }
            mock_agent_cls.return_value = mock_agent
            mock_agent_cls._format_turn_completion_explanation = (
                AIAgent._format_turn_completion_explanation
            )

            success, output, final_response, error = run_job(job)

        assert final_response == "Daily report: 4 PRs merged."
        assert success is True

    def test_run_job_titles_cron_session_from_job_not_important_hint(self, tmp_path):
        # The cron session's first message is the injected "[IMPORTANT: …]"
        # hint, which used to surface as the sidebar/history row label. run_job
        # must title the session from the job (name → short prompt → id).
        job = {
            "id": "test-job",
            "name": "Morning digest",
            "prompt": "summarize my inbox",
        }
        fake_db = MagicMock()

        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("hermes_cli.env_loader.load_hermes_dotenv"), \
             patch("hermes_cli.env_loader.reset_secret_source_cache"), \
             patch("hermes_state.SessionDB", return_value=fake_db), \
             patch(
                 "hermes_cli.runtime_provider.resolve_runtime_provider",
                 return_value={
                     "api_key": "test-key",
                     "base_url": "https://example.invalid/v1",
                     "provider": "openrouter",
                     "api_mode": "chat_completions",
                 },
             ), \
             patch("run_agent.AIAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {"final_response": "ok"}
            mock_agent_cls.return_value = mock_agent

            run_job(job)

        fake_db.set_session_title.assert_called_once()
        sid, title = fake_db.set_session_title.call_args[0]
        assert sid.startswith("cron_test-job_")
        assert "IMPORTANT" not in title
        assert title.startswith("Morning digest")

    def test_run_job_closes_agent_on_failure_to_prevent_fd_leak(self, tmp_path):
        # Regression: if ``run_conversation`` raises, the ephemeral cron
        # agent was previously leaked — over days of ticks this accumulated
        # httpx transports and hit EMFILE / "too many open files".
        job = {
            "id": "failing-job",
            "name": "failing",
            "prompt": "hello",
        }
        fake_db = MagicMock()

        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("hermes_cli.env_loader.load_hermes_dotenv"), \
             patch("hermes_cli.env_loader.reset_secret_source_cache"), \
             patch("hermes_state.SessionDB", return_value=fake_db), \
             patch(
                 "hermes_cli.runtime_provider.resolve_runtime_provider",
                 return_value={
                     "api_key": "***",
                     "base_url": "https://example.invalid/v1",
                     "provider": "openrouter",
                     "api_mode": "chat_completions",
                 },
             ), \
             patch("run_agent.AIAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.run_conversation.side_effect = RuntimeError("boom")
            mock_agent_cls.return_value = mock_agent

            success, output, final_response, error = run_job(job)

        assert success is False
        assert final_response == ""
        assert "RuntimeError: boom" in error
        mock_agent.close.assert_called_once()

    def test_run_job_reaps_stale_auxiliary_clients_per_tick(self, tmp_path):
        # Regression: auxiliary clients bound to the cron worker's dead
        # event loop must be reaped each tick. Without this, ``_client_cache``
        # holds onto transports whose underlying sockets can no longer be
        # closed (their loop is gone), leaking one fd batch per cron run.
        job = {
            "id": "aux-clean-job",
            "name": "aux-clean",
            "prompt": "hello",
        }
        fake_db = MagicMock()

        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("hermes_cli.env_loader.load_hermes_dotenv"), \
             patch("hermes_cli.env_loader.reset_secret_source_cache"), \
             patch("hermes_state.SessionDB", return_value=fake_db), \
             patch(
                 "hermes_cli.runtime_provider.resolve_runtime_provider",
                 return_value={
                     "api_key": "***",
                     "base_url": "https://example.invalid/v1",
                     "provider": "openrouter",
                     "api_mode": "chat_completions",
                 },
             ), \
             patch("run_agent.AIAgent") as mock_agent_cls, \
             patch("agent.auxiliary_client.cleanup_stale_async_clients") as cleanup_mock:
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {"final_response": "ok"}
            mock_agent_cls.return_value = mock_agent

            success, _output, _final_response, _error = run_job(job)

        assert success is True
        cleanup_mock.assert_called_once()

    @contextlib.contextmanager
    def _run_job_patches(self, tmp_path, extra=()):
        """Apply every patch run_job tests need, as one bundle.

        Yields ``(fake_db, mock_agent_cls)``. Using an ExitStack that enters
        the whole list means a caller can never silently drop a patch by
        index — the previous positional-list form let a seam split shift
        ``resolve_runtime_provider`` off the end of the applied slice, so the
        real resolver ran and (only on a dev machine with ambient creds) hid
        an auth failure that CI then caught. Every test enters all patches.

        ``extra`` is an iterable of additional context managers (e.g. a
        per-test ``_get_platform_tools`` patch) entered alongside the base set.
        """
        fake_db = MagicMock()
        mock_agent = MagicMock()
        mock_agent.run_conversation.return_value = {"final_response": "ok"}
        base = [
            patch("cron.scheduler._hermes_home", tmp_path),
            patch("cron.scheduler._resolve_origin", return_value=None),
            patch("hermes_cli.env_loader.load_hermes_dotenv"),
            patch("hermes_cli.env_loader.reset_secret_source_cache"),
            patch("hermes_state.SessionDB", return_value=fake_db),
            patch(
                "hermes_cli.runtime_provider.resolve_runtime_provider",
                return_value={
                    "api_key": "test-key",
                    "base_url": "https://example.invalid/v1",
                    "provider": "openrouter",
                    "api_mode": "chat_completions",
                },
            ),
            patch("run_agent.AIAgent", return_value=mock_agent),
        ]
        with contextlib.ExitStack() as stack:
            entered = [stack.enter_context(cm) for cm in base]
            for cm in extra:
                stack.enter_context(cm)
            mock_agent_cls = entered[-1]  # the AIAgent patch
            yield fake_db, mock_agent_cls

    def test_run_job_passes_enabled_toolsets_to_agent(self, tmp_path):
        job = {
            "id": "toolset-job",
            "name": "test",
            "prompt": "hello",
            "enabled_toolsets": ["web", "terminal", "file"],
        }
        with self._run_job_patches(tmp_path) as (_fake_db, mock_agent_cls):
            run_job(job)

        kwargs = mock_agent_cls.call_args.kwargs
        assert kwargs["enabled_toolsets"] == ["web", "terminal", "file"]

    def test_run_job_disabled_toolsets_layer_user_config_on_baseline(self, tmp_path):
        """agent.disabled_toolsets must be honoured in cron — issue #25752.

        The bug: per-job enabled_toolsets was returned verbatim, letting an
        LLM-supplied cronjob() call re-enable tools the operator had globally
        disabled. The fix: ALWAYS include agent.disabled_toolsets in the
        disabled_toolsets passed to AIAgent, on top of the cron baseline
        (cronjob/messaging/clarify). AIAgent's disabled_toolsets takes
        precedence over enabled_toolsets, so this stops the bypass.
        """
        (tmp_path / "config.yaml").write_text(
            "agent:\n"
            "  disabled_toolsets:\n"
            "    - terminal\n"
            "    - file\n",
            encoding="utf-8",
        )
        job = {
            "id": "policy-job",
            "name": "test",
            "prompt": "hello",
            "enabled_toolsets": ["web", "terminal", "file"],
        }
        with self._run_job_patches(tmp_path) as (_fake_db, mock_agent_cls):
            run_job(job)

        kwargs = mock_agent_cls.call_args.kwargs
        assert set(kwargs["disabled_toolsets"]) >= {
            "cronjob", "messaging", "clarify", "terminal", "file",
        }

    def test_run_job_enabled_toolsets_resolves_from_platform_config_when_not_set(self, tmp_path):
        """When a job has no explicit enabled_toolsets, the scheduler now
        resolves them from ``hermes tools`` platform config for ``cron``
        (PR #14xxx — blanket fix for Norbert's surprise ``moa`` run).

        The legacy "pass None → AIAgent loads full default" path is still
        reachable, but only when ``_get_platform_tools`` raises (safety net
        for any unexpected config shape).
        """
        job = {
            "id": "no-toolset-job",
            "name": "test",
            "prompt": "hello",
        }
        with self._run_job_patches(tmp_path) as (_fake_db, mock_agent_cls):
            run_job(job)

        kwargs = mock_agent_cls.call_args.kwargs
        # Resolution happened — not None, is a list.
        assert isinstance(kwargs["enabled_toolsets"], list)
        # The cron default is _HERMES_CORE_TOOLS with _DEFAULT_OFF_TOOLSETS
        # (``moa``, ``homeassistant``, ``rl``) removed. The most important
        # invariant: ``moa`` is NOT in the default cron toolset, so a cron
        # run cannot accidentally spin up frontier models.
        assert "moa" not in kwargs["enabled_toolsets"]

    def test_run_job_per_job_toolsets_win_over_platform_config(self, tmp_path):
        """Per-job enabled_toolsets (via cronjob tool) always take precedence
        over the platform-level ``hermes tools`` config."""
        job = {
            "id": "override-job",
            "name": "test",
            "prompt": "hello",
            "enabled_toolsets": ["terminal"],
        }
        # Even if the user has ``hermes tools`` configured to enable web+file
        # for cron, the per-job override wins.
        extra = [patch("hermes_cli.tools_config._get_platform_tools", return_value={"web", "file"})]
        with self._run_job_patches(tmp_path, extra=extra) as (_fake_db, mock_agent_cls):
            run_job(job)

        kwargs = mock_agent_cls.call_args.kwargs
        assert kwargs["enabled_toolsets"] == ["terminal"]

    def test_run_job_empty_response_returns_empty_not_placeholder(self, tmp_path):
        """Empty final_response should stay empty for delivery logic (issue #2234).

        The placeholder '(No response generated)' should only appear in the
        output log, not in the returned final_response that's used for delivery.
        """
        job = {
            "id": "silent-job",
            "name": "silent test",
            "prompt": "do work via tools only",
        }
        fake_db = MagicMock()

        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("hermes_cli.env_loader.load_hermes_dotenv"), \
             patch("hermes_cli.env_loader.reset_secret_source_cache"), \
             patch("hermes_state.SessionDB", return_value=fake_db), \
             patch(
                 "hermes_cli.runtime_provider.resolve_runtime_provider",
                 return_value={
                     "api_key": "***",
                     "base_url": "https://example.invalid/v1",
                     "provider": "openrouter",
                     "api_mode": "chat_completions",
                 },
             ), \
             patch("run_agent.AIAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            # Agent did work via tools but returned no text
            mock_agent.run_conversation.return_value = {"final_response": ""}
            mock_agent_cls.return_value = mock_agent

            success, output, final_response, error = run_job(job)

        assert success is True
        assert error is None
        # final_response should be empty for delivery logic to skip
        assert final_response == ""
        # But the output log should show the placeholder
        assert "(No response generated)" in output

    @pytest.mark.parametrize(
        "agent_result,expected_err_substring",
        [
            (
                {
                    "final_response": "API call failed after 3 retries: Request timed out.",
                    "failed": True,
                    "completed": False,
                    "error": "API call failed after 3 retries: Request timed out.",
                },
                "API call failed",
            ),
            (
                {"final_response": None, "completed": False, "failed": True},
                "agent reported failure",
            ),
            (
                {"final_response": "", "completed": False},
                "agent reported failure",
            ),
            (
                {
                    "final_response": "partial reply before crash",
                    "failed": True,
                    "completed": False,
                    "error": "model abort: connection reset",
                },
                "model abort",
            ),
        ],
    )
    def test_run_job_treats_agent_failure_flag_as_failure(
        self, tmp_path, agent_result, expected_err_substring
    ):
        """Issue #17855: run_conversation returns ``failed=True``/``completed=False``
        when the agent's API call exhausts retries or aborts mid-run. run_job
        must surface this as success=False so cron's last_status reflects the
        failure and the user gets an error notification, instead of treating
        the (often non-empty) error string in final_response as a legitimate
        agent reply.
        """
        job = {
            "id": "failing-api-job",
            "name": "failing api",
            "prompt": "do something",
        }
        fake_db = MagicMock()

        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("hermes_cli.env_loader.load_hermes_dotenv"), \
             patch("hermes_cli.env_loader.reset_secret_source_cache"), \
             patch("hermes_state.SessionDB", return_value=fake_db), \
             patch(
                 "hermes_cli.runtime_provider.resolve_runtime_provider",
                 return_value={
                     "api_key": "***",
                     "base_url": "https://example.invalid/v1",
                     "provider": "openrouter",
                     "api_mode": "chat_completions",
                 },
             ), \
             patch("run_agent.AIAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = agent_result
            mock_agent_cls.return_value = mock_agent

            success, output, final_response, error = run_job(job)

        assert success is False
        assert final_response == ""
        assert error is not None and expected_err_substring in error
        # Output should be the FAILED template, not the success template.
        assert "(FAILED)" in output
        # Ephemeral cron agent must still be closed even on agent-flagged failure.
        mock_agent.close.assert_called_once()

    def test_run_job_completed_true_without_failed_flag_succeeds(self, tmp_path):
        """Regression guard: a normal success result (``completed=True``,
        ``failed`` absent) must not trip the failure-flag check.
        """
        job = {
            "id": "ok-job",
            "name": "ok",
            "prompt": "hello",
        }
        fake_db = MagicMock()

        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("hermes_cli.env_loader.load_hermes_dotenv"), \
             patch("hermes_cli.env_loader.reset_secret_source_cache"), \
             patch("hermes_state.SessionDB", return_value=fake_db), \
             patch(
                 "hermes_cli.runtime_provider.resolve_runtime_provider",
                 return_value={
                     "api_key": "***",
                     "base_url": "https://example.invalid/v1",
                     "provider": "openrouter",
                     "api_mode": "chat_completions",
                 },
             ), \
             patch("run_agent.AIAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {
                "final_response": "all good",
                "completed": True,
            }
            mock_agent_cls.return_value = mock_agent

            success, output, final_response, error = run_job(job)

        assert success is True
        assert error is None
        assert final_response == "all good"

    def test_run_job_delivers_max_iteration_fallback_summary(self, tmp_path):
        """Cron should deliver a usable max-iteration fallback summary.

        A cron run can exhaust the iteration budget, get a final text summary
        from the no-tools fallback call, and still have ``completed=False`` in
        the generic agent result. That should not make cron raise the report
        text as a RuntimeError.
        """
        job = {
            "id": "summary-job",
            "name": "summary",
            "prompt": "finish the report",
        }
        fake_db = MagicMock()

        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("hermes_cli.env_loader.load_hermes_dotenv"), \
             patch("hermes_cli.env_loader.reset_secret_source_cache"), \
             patch("hermes_state.SessionDB", return_value=fake_db), \
             patch(
                 "hermes_cli.runtime_provider.resolve_runtime_provider",
                 return_value={
                     "api_key": "***",
                     "base_url": "https://example.invalid/v1",
                     "provider": "openrouter",
                     "api_mode": "chat_completions",
                 },
             ), \
             patch("run_agent.AIAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {
                "final_response": "final fallback report",
                "completed": False,
                "failed": False,
                "turn_exit_reason": "max_iterations_reached(60/60)",
            }
            mock_agent_cls.return_value = mock_agent

            success, output, final_response, error = run_job(job)

        assert success is True
        assert error is None
        assert final_response == "final fallback report"
        assert "final fallback report" in output
        assert "(FAILED)" not in output

    def test_tick_skips_due_jobs_while_dispatch_is_paused(self, tmp_path):
        """The drain gate runs before advancing a due job's schedule."""
        from cron.scheduler import tick

        job = {
            "id": "paused-due-job",
            "name": "paused due job",
            "schedule": {"kind": "interval", "seconds": 60},
            "next_run_at": "2020-01-01T00:00:00+00:00",
            "enabled": True,
        }
        with patch("cron.scheduler.get_due_jobs", return_value=[job]), patch(
            "cron.scheduler.advance_next_run"
        ) as advance, patch("cron.scheduler.run_one_job") as run_one:
            assert tick(verbose=False, sync=True, can_dispatch=lambda: False) == 0

        advance.assert_not_called()
        run_one.assert_not_called()

    def test_tick_marks_empty_response_as_error(self, tmp_path):
        """When run_job returns success=True but final_response is empty,
        tick() should mark the job as error so last_status != 'ok'.
        (issue #8585)
        """
        from cron.scheduler import tick

        job = {
            "id": "empty-job",
            "name": "empty-test",
            "prompt": "do something",
            "schedule": "every 1h",
            "enabled": True,
            "next_run_at": "2020-01-01T00:00:00",
            "deliver": "local",
            "last_status": None,
        }

        fake_db = MagicMock()

        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("cron.scheduler.get_due_jobs", return_value=[job]), \
             patch("cron.scheduler.advance_next_run"), \
             patch("cron.scheduler.mark_job_run") as mock_mark, \
             patch("cron.scheduler.save_job_output", return_value="/tmp/out.md"), \
             patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("cron.scheduler.run_job", return_value=(True, "output", "", None)):
            tick(verbose=False)

        # Should be called with success=False because final_response is empty
        mock_mark.assert_called_once()
        call_args = mock_mark.call_args
        assert call_args[0][0] == "empty-job"
        assert call_args[0][1] is False  # success should be False
        assert "empty" in call_args[0][2].lower()  # error should mention empty

    def test_run_job_sets_auto_delivery_env_from_dotenv_home_channel(self, tmp_path, monkeypatch):
        job = {
            "id": "test-job",
            "name": "test",
            "prompt": "hello",
            "deliver": "telegram",
        }
        fake_db = MagicMock()
        seen = {}

        (tmp_path / ".env").write_text("TELEGRAM_HOME_CHANNEL=-2002\n")
        monkeypatch.delenv("TELEGRAM_HOME_CHANNEL", raising=False)
        monkeypatch.delenv("HERMES_CRON_AUTO_DELIVER_PLATFORM", raising=False)
        monkeypatch.delenv("HERMES_CRON_AUTO_DELIVER_CHAT_ID", raising=False)
        monkeypatch.delenv("HERMES_CRON_AUTO_DELIVER_THREAD_ID", raising=False)

        class FakeAgent:
            def __init__(self, *args, **kwargs):
                pass

            def run_conversation(self, *args, **kwargs):
                from gateway.session_context import get_session_env
                seen["platform"] = get_session_env("HERMES_CRON_AUTO_DELIVER_PLATFORM") or None
                seen["chat_id"] = get_session_env("HERMES_CRON_AUTO_DELIVER_CHAT_ID") or None
                seen["thread_id"] = get_session_env("HERMES_CRON_AUTO_DELIVER_THREAD_ID") or None
                return {"final_response": "ok"}

        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("hermes_state.SessionDB", return_value=fake_db), \
             patch(
                 "hermes_cli.runtime_provider.resolve_runtime_provider",
                 return_value={
                     "api_key": "***",
                     "base_url": "https://example.invalid/v1",
                     "provider": "openrouter",
                     "api_mode": "chat_completions",
                 },
             ), \
             patch("run_agent.AIAgent", FakeAgent):
            success, output, final_response, error = run_job(job)

        assert success is True
        assert error is None
        assert final_response == "ok"
        assert "ok" in output
        assert seen == {
            "platform": "telegram",
            "chat_id": "-2002",
            "thread_id": None,
        }
        assert os.getenv("HERMES_CRON_AUTO_DELIVER_PLATFORM") is None
        assert os.getenv("HERMES_CRON_AUTO_DELIVER_CHAT_ID") is None
        assert os.getenv("HERMES_CRON_AUTO_DELIVER_THREAD_ID") is None
        fake_db.close.assert_called_once()

    @pytest.mark.parametrize("timeout_value", ["600", "0"])
    def test_run_job_heartbeats_oneshot_claim_in_both_wait_modes(
        self, tmp_path, monkeypatch, timeout_value
    ):
        """Timed and unlimited one-shot monitors both refresh their owned claim."""
        job = {
            "id": "heartbeat-job",
            "name": "heartbeat",
            "prompt": "hello",
            "schedule": {"kind": "once", "run_at": "2026-07-10T12:00:00Z"},
            "run_claim": {"at": "2026-07-10T12:00:00Z", "by": "owner-token"},
        }
        fake_db = MagicMock()

        class FakeAgent:
            def __init__(self, *args, **kwargs):
                pass

            def run_conversation(self, *args, **kwargs):
                return {"final_response": "ok"}

        class FakeFuture:
            def result(self):
                return {"final_response": "ok"}

        fake_future = FakeFuture()
        fake_pool = MagicMock()
        fake_pool.submit.return_value = fake_future
        wait_results = [(set(), set()), ({fake_future}, set())]
        monotonic_ticks = itertools.count(step=61.0)
        monkeypatch.setenv("HERMES_CRON_TIMEOUT", timeout_value)

        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("hermes_state.SessionDB", return_value=fake_db), \
             patch(
                 "hermes_cli.runtime_provider.resolve_runtime_provider",
                 return_value={
                     "api_key": "***",
                     "base_url": "https://example.invalid/v1",
                     "provider": "openrouter",
                     "api_mode": "chat_completions",
                 },
             ), \
             patch("run_agent.AIAgent", FakeAgent), \
             patch("cron.scheduler.concurrent.futures.ThreadPoolExecutor", return_value=fake_pool), \
             patch("cron.scheduler.concurrent.futures.wait", side_effect=wait_results), \
             patch("cron.scheduler.time.monotonic", side_effect=monotonic_ticks.__next__), \
             patch("cron.scheduler.heartbeat_run_claim", return_value=True) as heartbeat:
            success, _output, final_response, error = run_job(job)

        assert success is True
        assert error is None
        assert final_response == "ok"
        heartbeat.assert_called_once_with(
            "heartbeat-job", expected_owner="owner-token"
        )

    def test_run_job_resets_secret_source_cache_before_reload(self, tmp_path, monkeypatch):
        """Each run must clear the secret-source cache before re-reading the
        env, so a long-running gateway re-resolves Bitwarden/BSM-backed secrets
        instead of leaving the startup .env placeholder in place (#33465).

        A bare ``load_dotenv`` re-load can't do this: startup already recorded
        this HERMES_HOME in ``_APPLIED_HOMES``, so the external-secret pull
        no-ops and only the placeholder is re-applied. The scheduler must call
        ``reset_secret_source_cache()`` (forcing the re-pull) and route through
        ``load_hermes_dotenv`` (which then re-applies external secret sources).
        """
        job = {"id": "bsm-job", "name": "bsm", "prompt": "hello"}
        fake_db = MagicMock()
        call_order = []

        def _record_reset():
            call_order.append("reset")

        def _record_load(*args, **kwargs):
            call_order.append("load")
            return []

        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("hermes_cli.env_loader.reset_secret_source_cache", _record_reset), \
             patch("hermes_cli.env_loader.load_hermes_dotenv", _record_load), \
             patch("hermes_state.SessionDB", return_value=fake_db), \
             patch(
                 "hermes_cli.runtime_provider.resolve_runtime_provider",
                 return_value={
                     "api_key": "***",
                     "base_url": "https://example.invalid/v1",
                     "provider": "openrouter",
                     "api_mode": "chat_completions",
                 },
             ), \
             patch("run_agent.AIAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {"final_response": "ok"}
            mock_agent_cls.return_value = mock_agent
            success, _output, _final, error = run_job(job)

        assert success is True
        assert error is None
        # reset MUST precede the reload, else _APPLIED_HOMES no-ops the re-pull.
        assert call_order[:2] == ["reset", "load"], call_order

    def test_run_job_clears_stale_auto_delivery_thread_id_between_jobs(self, tmp_path, monkeypatch):
        jobs = [
            {
                "id": "threaded-job",
                "name": "threaded",
                "prompt": "hello",
                "deliver": "telegram:-1001:42",
            },
            {
                "id": "threadless-job",
                "name": "threadless",
                "prompt": "hello again",
                "deliver": "telegram:-2002",
            },
        ]
        fake_db = MagicMock()
        seen = []

        monkeypatch.delenv("HERMES_CRON_AUTO_DELIVER_PLATFORM", raising=False)
        monkeypatch.delenv("HERMES_CRON_AUTO_DELIVER_CHAT_ID", raising=False)
        monkeypatch.delenv("HERMES_CRON_AUTO_DELIVER_THREAD_ID", raising=False)

        class FakeAgent:
            def __init__(self, *args, **kwargs):
                pass

            def run_conversation(self, *args, **kwargs):
                from gateway.session_context import get_session_env

                seen.append(
                    {
                        "platform": get_session_env("HERMES_CRON_AUTO_DELIVER_PLATFORM") or None,
                        "chat_id": get_session_env("HERMES_CRON_AUTO_DELIVER_CHAT_ID") or None,
                        "thread_id": get_session_env("HERMES_CRON_AUTO_DELIVER_THREAD_ID") or None,
                    }
                )
                return {"final_response": "ok"}

        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("hermes_state.SessionDB", return_value=fake_db), \
             patch(
                 "hermes_cli.runtime_provider.resolve_runtime_provider",
                 return_value={
                     "api_key": "***",
                     "base_url": "https://example.invalid/v1",
                     "provider": "openrouter",
                     "api_mode": "chat_completions",
                 },
             ), \
             patch("run_agent.AIAgent", FakeAgent):
            for job in jobs:
                success, output, final_response, error = run_job(job)
                assert success is True
                assert error is None
                assert final_response == "ok"
                assert "ok" in output

        assert seen == [
            {
                "platform": "telegram",
                "chat_id": "-1001",
                "thread_id": "42",
            },
            {
                "platform": "telegram",
                "chat_id": "-2002",
                "thread_id": None,
            },
        ]
        assert os.getenv("HERMES_CRON_AUTO_DELIVER_PLATFORM") is None
        assert os.getenv("HERMES_CRON_AUTO_DELIVER_CHAT_ID") is None
        assert os.getenv("HERMES_CRON_AUTO_DELIVER_THREAD_ID") is None
        assert fake_db.close.call_count == 2


class TestRunJobConfigLogging:
    """Verify that config.yaml parse failures are logged, not silently swallowed."""

    def test_bad_config_yaml_is_logged(self, caplog, tmp_path):
        """When config.yaml is malformed, a warning should be logged."""
        bad_yaml = tmp_path / "config.yaml"
        bad_yaml.write_text("invalid: yaml: [[[bad")

        job = {
            "id": "test-job",
            "name": "test",
            "prompt": "hello",
        }

        # Mock heavy post-yaml work so the test only exercises the warning
        # path. Without these mocks, run_job continues into provider
        # resolution and MCP discovery, both of which can spawn subprocesses
        # / hit the network and have caused this test to time out on CI
        # (>30s wall clock) under load. See PR #33661 follow-up.
        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("hermes_cli.env_loader.load_hermes_dotenv"), \
             patch("hermes_cli.env_loader.reset_secret_source_cache"), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider",
                   return_value={"provider": "openrouter", "api_key": "x",
                                 "base_url": "https://example.invalid",
                                 "api_mode": "chat_completions"}), \
             patch("tools.mcp_tool.discover_mcp_tools", return_value=[]), \
             patch("run_agent.AIAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {"final_response": "ok"}
            mock_agent_cls.return_value = mock_agent

            with caplog.at_level(logging.WARNING, logger="cron.scheduler"):
                run_job(job)

        assert any("failed to load config.yaml" in r.message for r in caplog.records), \
            f"Expected 'failed to load config.yaml' warning in logs, got: {[r.message for r in caplog.records]}"

    def test_bad_prefill_messages_is_logged(self, caplog, tmp_path):
        """When the prefill messages file contains invalid JSON, a warning should be logged."""
        # Valid config.yaml that points to a bad prefill file
        config_yaml = tmp_path / "config.yaml"
        config_yaml.write_text("prefill_messages_file: prefill.json\n")

        bad_prefill = tmp_path / "prefill.json"
        bad_prefill.write_text("{not valid json!!!")

        job = {
            "id": "test-job",
            "name": "test",
            "prompt": "hello",
        }

        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("hermes_cli.env_loader.load_hermes_dotenv"), \
             patch("hermes_cli.env_loader.reset_secret_source_cache"), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider",
                   return_value={"provider": "openrouter", "api_key": "x",
                                 "base_url": "https://example.invalid",
                                 "api_mode": "chat_completions"}), \
             patch("tools.mcp_tool.discover_mcp_tools", return_value=[]), \
             patch("run_agent.AIAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {"final_response": "ok"}
            mock_agent_cls.return_value = mock_agent

            with caplog.at_level(logging.WARNING, logger="cron.scheduler"):
                run_job(job)

        assert any("failed to parse prefill messages" in r.message for r in caplog.records), \
            f"Expected 'failed to parse prefill messages' warning in logs, got: {[r.message for r in caplog.records]}"


class TestRunJobConfigEnvVarExpansion:
    """Verify that ${VAR} references in config.yaml are expanded when running cron jobs."""

    _RUNTIME = {
        "api_key": "test-key",
        "base_url": "https://example.invalid/v1",
        "provider": "openrouter",
        "api_mode": "chat_completions",
    }

    def test_model_env_ref_in_config_yaml_is_expanded(self, tmp_path, monkeypatch):
        """${VAR} in config.yaml model: is expanded using env after .env is loaded."""
        (tmp_path / "config.yaml").write_text("model: ${_HERMES_TEST_CRON_MODEL}\n")
        monkeypatch.setenv("_HERMES_TEST_CRON_MODEL", "gpt-4o-mini-cron-test")

        job = {"id": "env-job", "name": "env test", "prompt": "hi"}
        fake_db = MagicMock()

        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("hermes_cli.env_loader.load_hermes_dotenv"), \
             patch("hermes_cli.env_loader.reset_secret_source_cache"), \
             patch("hermes_state.SessionDB", return_value=fake_db), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider",
                   return_value=self._RUNTIME), \
             patch("run_agent.AIAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {"final_response": "ok"}
            mock_agent_cls.return_value = mock_agent
            success, _, _, error = run_job(job)

        assert success is True
        assert error is None
        kwargs = mock_agent_cls.call_args.kwargs
        assert kwargs["model"] == "gpt-4o-mini-cron-test", (
            f"Expected model='gpt-4o-mini-cron-test', got {kwargs['model']!r}. "
            "config.yaml ${VAR} was not expanded in the cron execution path."
        )

    def test_legacy_agent_prefill_messages_file_is_loaded(self, tmp_path, monkeypatch):
        """Cron accepts the legacy agent.prefill_messages_file fallback."""
        prefill = [{"role": "system", "content": "legacy cron prefill"}]
        (tmp_path / "prefill.json").write_text(json.dumps(prefill), encoding="utf-8")
        (tmp_path / "config.yaml").write_text(
            "agent:\n"
            "  prefill_messages_file: prefill.json\n",
            encoding="utf-8",
        )

        job = {"id": "prefill-job", "name": "prefill test", "prompt": "hi"}
        fake_db = MagicMock()

        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("hermes_cli.env_loader.load_hermes_dotenv"), \
             patch("hermes_cli.env_loader.reset_secret_source_cache"), \
             patch("hermes_state.SessionDB", return_value=fake_db), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider",
                   return_value=self._RUNTIME), \
             patch("tools.mcp_tool.discover_mcp_tools", return_value=[]), \
             patch("run_agent.AIAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {"final_response": "ok"}
            mock_agent_cls.return_value = mock_agent
            success, _, _, error = run_job(job)

        assert success is True
        assert error is None
        assert mock_agent_cls.call_args.kwargs["prefill_messages"] == prefill

    def test_fallback_model_env_ref_in_config_yaml_is_expanded(self, tmp_path, monkeypatch):
        """${VAR} in config.yaml fallback_providers model: is expanded."""
        (tmp_path / "config.yaml").write_text(
            "model: primary-model\n"
            "fallback_providers:\n"
            "  - provider: openrouter\n"
            "    model: ${_HERMES_TEST_CRON_FALLBACK}\n"
        )
        monkeypatch.setenv("_HERMES_TEST_CRON_FALLBACK", "gpt-4o-fallback-test")

        job = {"id": "fb-job", "name": "fallback test", "prompt": "hi"}
        fake_db = MagicMock()

        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("hermes_cli.env_loader.load_hermes_dotenv"), \
             patch("hermes_cli.env_loader.reset_secret_source_cache"), \
             patch("hermes_state.SessionDB", return_value=fake_db), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider",
                   return_value=self._RUNTIME), \
             patch("run_agent.AIAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {"final_response": "ok"}
            mock_agent_cls.return_value = mock_agent
            run_job(job)

        kwargs = mock_agent_cls.call_args.kwargs
        fb = kwargs.get("fallback_model") or []
        fb_list = fb if isinstance(fb, list) else [fb]
        expanded = [e.get("model") for e in fb_list if isinstance(e, dict)]
        assert "gpt-4o-fallback-test" in expanded, (
            f"Expected expanded fallback model in {expanded!r}. "
            "config.yaml ${VAR} in fallback_providers was not expanded."
        )

    def test_fallback_chain_merges_providers_and_legacy_model(self, tmp_path, monkeypatch):
        """Cron uses get_fallback_chain so legacy fallback_model is not dropped."""
        (tmp_path / "config.yaml").write_text(
            "fallback_providers:\n"
            "  - provider: openrouter\n"
            "    model: gpt-4o-mini\n"
            "fallback_model:\n"
            "  provider: anthropic\n"
            "  model: claude-sonnet-4-6\n"
        )

        job = {"id": "fb-merge", "name": "fallback merge", "prompt": "hi"}
        fake_db = MagicMock()

        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("dotenv.load_dotenv"), \
             patch("hermes_state.SessionDB", return_value=fake_db), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider",
                   return_value=self._RUNTIME), \
             patch("run_agent.AIAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {"final_response": "ok"}
            mock_agent_cls.return_value = mock_agent
            run_job(job)

        fb = mock_agent_cls.call_args.kwargs.get("fallback_model") or []
        models = [e.get("model") for e in fb if isinstance(e, dict)]
        assert models == ["gpt-4o-mini", "claude-sonnet-4-6"]

    def test_unexpanded_ref_passthrough_when_var_unset(self, tmp_path, monkeypatch):
        """When the env var is not set, the literal ${VAR} is kept verbatim (not crashed)."""
        (tmp_path / "config.yaml").write_text("model: ${_HERMES_TEST_CRON_UNSET_VAR}\n")
        monkeypatch.delenv("_HERMES_TEST_CRON_UNSET_VAR", raising=False)

        job = {"id": "unset-job", "name": "unset var test", "prompt": "hi"}
        fake_db = MagicMock()

        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("hermes_cli.env_loader.load_hermes_dotenv"), \
             patch("hermes_cli.env_loader.reset_secret_source_cache"), \
             patch("hermes_state.SessionDB", return_value=fake_db), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider",
                   return_value=self._RUNTIME), \
             patch("run_agent.AIAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {"final_response": "ok"}
            mock_agent_cls.return_value = mock_agent
            success, _, _, error = run_job(job)

        assert success is True
        kwargs = mock_agent_cls.call_args.kwargs
        # Unresolved refs are kept verbatim — _expand_env_vars contract
        assert kwargs["model"] == "${_HERMES_TEST_CRON_UNSET_VAR}"


class TestRunJobModelResolution:
    """Verify defensive model resolution for jobs stored with ``model: null``.

    Issue #23979: a cron job created without an explicit model is stored as
    ``model: null``. At fire time the scheduler must:
      1. fall back to ``HERMES_MODEL`` env if set,
      2. else fall back to config.yaml ``model.default`` if set,
      3. else fail fast with an actionable error — never let an empty string
         reach the provider where it surfaces as an opaque 400.
    """

    _RUNTIME = {
        "api_key": "test-key",
        "base_url": "https://example.invalid/v1",
        "provider": "openrouter",
        "api_mode": "chat_completions",
    }

    def test_null_job_model_falls_back_to_env(self, tmp_path, monkeypatch):
        """``model: null`` on the job uses HERMES_MODEL when set."""
        (tmp_path / "config.yaml").write_text("")
        monkeypatch.setenv("HERMES_MODEL", "env-model")

        job = {"id": "null-model-job", "name": "null model", "prompt": "hi", "model": None}
        fake_db = MagicMock()

        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("hermes_cli.env_loader.load_hermes_dotenv"), \
             patch("hermes_cli.env_loader.reset_secret_source_cache"), \
             patch("hermes_state.SessionDB", return_value=fake_db), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider",
                   return_value=self._RUNTIME), \
             patch("run_agent.AIAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {"final_response": "ok"}
            mock_agent_cls.return_value = mock_agent
            success, _, _, error = run_job(job)

        assert success is True
        assert error is None
        assert mock_agent_cls.call_args.kwargs["model"] == "env-model"

    def test_null_job_model_falls_back_to_config_default(self, tmp_path, monkeypatch):
        """``model: null`` on the job uses config.yaml model.default when env is empty."""
        (tmp_path / "config.yaml").write_text("model:\n  default: config-default-model\n")
        monkeypatch.delenv("HERMES_MODEL", raising=False)

        job = {"id": "cfg-default-job", "name": "cfg default", "prompt": "hi", "model": None}
        fake_db = MagicMock()

        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("hermes_cli.env_loader.load_hermes_dotenv"), \
             patch("hermes_cli.env_loader.reset_secret_source_cache"), \
             patch("hermes_state.SessionDB", return_value=fake_db), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider",
                   return_value=self._RUNTIME), \
             patch("run_agent.AIAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {"final_response": "ok"}
            mock_agent_cls.return_value = mock_agent
            success, _, _, error = run_job(job)

        assert success is True
        assert error is None
        assert mock_agent_cls.call_args.kwargs["model"] == "config-default-model"

    def test_explicit_null_model_block_in_config_does_not_overwrite_env(self, tmp_path, monkeypatch):
        """``model: null`` in config.yaml must not overwrite a resolved HERMES_MODEL.

        Regression: before #23979 the resolver coerced ``model: null`` to
        ``{}`` only via the ``.get("model", {})`` default — which does not
        fire when the key is present with a None value. The resolver then
        skipped both branches and kept the env value, but a similar
        ``model: {default: null}`` shape would call ``.get("default", model)``
        which returns ``None`` and clobbered ``model``.
        """
        (tmp_path / "config.yaml").write_text("model:\n  default: null\n")
        monkeypatch.setenv("HERMES_MODEL", "env-model")

        job = {"id": "null-default-job", "name": "null default", "prompt": "hi", "model": None}
        fake_db = MagicMock()

        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("hermes_cli.env_loader.load_hermes_dotenv"), \
             patch("hermes_cli.env_loader.reset_secret_source_cache"), \
             patch("hermes_state.SessionDB", return_value=fake_db), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider",
                   return_value=self._RUNTIME), \
             patch("run_agent.AIAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {"final_response": "ok"}
            mock_agent_cls.return_value = mock_agent
            success, _, _, error = run_job(job)

        assert success is True
        assert mock_agent_cls.call_args.kwargs["model"] == "env-model"

    def test_no_model_anywhere_fails_with_actionable_error(self, tmp_path, monkeypatch):
        """All three sources empty → fail fast with a clear message, not an opaque 400."""
        (tmp_path / "config.yaml").write_text("")
        monkeypatch.delenv("HERMES_MODEL", raising=False)

        job = {"id": "no-model-job", "name": "no model anywhere", "prompt": "hi", "model": None}
        fake_db = MagicMock()

        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("hermes_cli.env_loader.load_hermes_dotenv"), \
             patch("hermes_cli.env_loader.reset_secret_source_cache"), \
             patch("hermes_state.SessionDB", return_value=fake_db), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider",
                   return_value=self._RUNTIME), \
             patch("run_agent.AIAgent") as mock_agent_cls:
            success, _, _, error = run_job(job)

        assert success is False
        assert error is not None
        assert "no model configured" in error
        # AIAgent must never be constructed with an empty model — that's
        # precisely the bug we're guarding against.
        mock_agent_cls.assert_not_called()

    def test_job_model_update_takes_effect_on_next_run(self, tmp_path, monkeypatch):
        """The per-job model is re-read every tick — no in-memory cache.

        This is the property the original bug report asked for. We verify
        it by calling run_job twice with the same job dict mutated between
        calls, simulating the storage update flow.
        """
        (tmp_path / "config.yaml").write_text("")
        monkeypatch.delenv("HERMES_MODEL", raising=False)

        job = {"id": "updated-model-job", "name": "updated", "prompt": "hi", "model": "first-model"}
        fake_db = MagicMock()

        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("hermes_cli.env_loader.load_hermes_dotenv"), \
             patch("hermes_cli.env_loader.reset_secret_source_cache"), \
             patch("hermes_state.SessionDB", return_value=fake_db), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider",
                   return_value=self._RUNTIME), \
             patch("run_agent.AIAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {"final_response": "ok"}
            mock_agent_cls.return_value = mock_agent

            run_job(job)
            assert mock_agent_cls.call_args.kwargs["model"] == "first-model"

            job["model"] = "second-model"  # simulates jobs.json being rewritten
            run_job(job)
            assert mock_agent_cls.call_args.kwargs["model"] == "second-model"

    def test_config_model_as_plain_string(self, tmp_path, monkeypatch):
        """config.yaml ``model:`` given as a bare string is used directly."""
        (tmp_path / "config.yaml").write_text("model: string-form-model\n")
        monkeypatch.delenv("HERMES_MODEL", raising=False)

        job = {"id": "string-cfg-job", "name": "string cfg", "prompt": "hi", "model": None}
        fake_db = MagicMock()

        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("hermes_cli.env_loader.load_hermes_dotenv"), \
             patch("hermes_cli.env_loader.reset_secret_source_cache"), \
             patch("hermes_state.SessionDB", return_value=fake_db), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider",
                   return_value=self._RUNTIME), \
             patch("run_agent.AIAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {"final_response": "ok"}
            mock_agent_cls.return_value = mock_agent
            success, _, _, error = run_job(job)

        assert success is True
        assert error is None
        assert mock_agent_cls.call_args.kwargs["model"] == "string-form-model"

    def test_config_model_alias_key_resolves(self, tmp_path, monkeypatch):
        """A ``model: {model: ...}`` alias key resolves like the CLI sibling.

        ``hermes_cli/oneshot.py``, ``fallback_cmd.py`` and ``prompt_size.py``
        all accept ``model.model`` as an alias for ``model.default``. The cron
        resolver mirrors that so a config that works in the CLI also works in
        cron.
        """
        (tmp_path / "config.yaml").write_text("model:\n  model: alias-key-model\n")
        monkeypatch.delenv("HERMES_MODEL", raising=False)

        job = {"id": "alias-job", "name": "alias", "prompt": "hi", "model": None}
        fake_db = MagicMock()

        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("hermes_cli.env_loader.load_hermes_dotenv"), \
             patch("hermes_cli.env_loader.reset_secret_source_cache"), \
             patch("hermes_state.SessionDB", return_value=fake_db), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider",
                   return_value=self._RUNTIME), \
             patch("run_agent.AIAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {"final_response": "ok"}
            mock_agent_cls.return_value = mock_agent
            success, _, _, error = run_job(job)

        assert success is True
        assert error is None
        assert mock_agent_cls.call_args.kwargs["model"] == "alias-key-model"

    def test_corrupt_config_yaml_does_not_crash_with_job_model(self, tmp_path, monkeypatch):
        """A malformed config.yaml degrades gracefully when the job has a model."""
        (tmp_path / "config.yaml").write_text("{{{invalid yaml!!!")
        monkeypatch.delenv("HERMES_MODEL", raising=False)

        job = {"id": "corrupt-job", "name": "corrupt", "prompt": "hi", "model": "explicit-model"}
        fake_db = MagicMock()

        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("hermes_cli.env_loader.load_hermes_dotenv"), \
             patch("hermes_cli.env_loader.reset_secret_source_cache"), \
             patch("hermes_state.SessionDB", return_value=fake_db), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider",
                   return_value=self._RUNTIME), \
             patch("run_agent.AIAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {"final_response": "ok"}
            mock_agent_cls.return_value = mock_agent
            success, _, _, error = run_job(job)

        # Explicit job model survives the corrupt-config fall-through.
        assert success is True
        assert error is None
        assert mock_agent_cls.call_args.kwargs["model"] == "explicit-model"


class TestRunJobSkillBacked:
    def test_run_job_preserves_skill_env_passthrough_into_worker_thread(self, tmp_path):
        job = {
            "id": "skill-env-job",
            "name": "skill env test",
            "prompt": "Use the skill.",
            "skill": "notion",
        }

        fake_db = MagicMock()

        def _skill_view(name):
            assert name == "notion"
            from tools.env_passthrough import register_env_passthrough

            register_env_passthrough(["NOTION_API_KEY"])
            return json.dumps({"success": True, "content": "# notion\nUse Notion."})

        def _run_conversation(prompt):
            from tools.env_passthrough import get_all_passthrough

            assert "NOTION_API_KEY" in get_all_passthrough()
            return {"final_response": "ok"}

        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("hermes_cli.env_loader.load_hermes_dotenv"), \
             patch("hermes_cli.env_loader.reset_secret_source_cache"), \
             patch("hermes_state.SessionDB", return_value=fake_db), \
             patch(
                 "hermes_cli.runtime_provider.resolve_runtime_provider",
                 return_value={
                     "api_key": "***",
                     "base_url": "https://example.invalid/v1",
                     "provider": "openrouter",
                     "api_mode": "chat_completions",
                 },
             ), \
             patch("tools.skills_tool.skill_view", side_effect=_skill_view), \
             patch("run_agent.AIAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.run_conversation.side_effect = _run_conversation
            mock_agent_cls.return_value = mock_agent

            try:
                success, output, final_response, error = run_job(job)
            finally:
                clear_env_passthrough()

        assert success is True
        assert error is None
        assert final_response == "ok"

    def test_run_job_preserves_credential_file_passthrough_into_worker_thread(self, tmp_path):
        """copy_context() also propagates credential_files ContextVar."""
        job = {
            "id": "cred-env-job",
            "name": "cred file test",
            "prompt": "Use the skill.",
            "skill": "google-workspace",
        }

        fake_db = MagicMock()

        # Create a credential file so register_credential_file succeeds
        cred_dir = tmp_path / "credentials"
        cred_dir.mkdir()
        (cred_dir / "google_token.json").write_text('{"token": "t"}')

        def _skill_view(name):
            assert name == "google-workspace"
            from tools.credential_files import register_credential_file

            register_credential_file("credentials/google_token.json")
            return json.dumps({"success": True, "content": "# google-workspace\nUse Google."})

        def _run_conversation(prompt):
            from tools.credential_files import _get_registered

            registered = _get_registered()
            assert registered, "credential files must be visible in worker thread"
            assert any("google_token.json" in v for v in registered.values())
            return {"final_response": "ok"}

        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("tools.credential_files._resolve_hermes_home", return_value=tmp_path), \
             patch("hermes_cli.env_loader.load_hermes_dotenv"), \
             patch("hermes_cli.env_loader.reset_secret_source_cache"), \
             patch("hermes_state.SessionDB", return_value=fake_db), \
             patch(
                 "hermes_cli.runtime_provider.resolve_runtime_provider",
                 return_value={
                     "api_key": "***",
                     "base_url": "https://example.invalid/v1",
                     "provider": "openrouter",
                     "api_mode": "chat_completions",
                 },
             ), \
             patch("tools.skills_tool.skill_view", side_effect=_skill_view), \
             patch("run_agent.AIAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.run_conversation.side_effect = _run_conversation
            mock_agent_cls.return_value = mock_agent

            try:
                success, output, final_response, error = run_job(job)
            finally:
                clear_credential_files()

        assert success is True
        assert error is None
        assert final_response == "ok"

    def test_run_job_loads_skill_and_disables_recursive_cron_tools(self, tmp_path):
        job = {
            "id": "skill-job",
            "name": "skill test",
            "prompt": "Check the feeds and summarize anything new.",
            "skill": "blogwatcher",
        }

        fake_db = MagicMock()

        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("hermes_cli.env_loader.load_hermes_dotenv"), \
             patch("hermes_cli.env_loader.reset_secret_source_cache"), \
             patch("hermes_state.SessionDB", return_value=fake_db), \
             patch(
                 "hermes_cli.runtime_provider.resolve_runtime_provider",
                 return_value={
                     "api_key": "***",
                     "base_url": "https://example.invalid/v1",
                     "provider": "openrouter",
                     "api_mode": "chat_completions",
                 },
             ), \
             patch("tools.skills_tool.skill_view", return_value=json.dumps({"success": True, "content": "# Blogwatcher\nFollow this skill."})), \
             patch("run_agent.AIAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {"final_response": "ok"}
            mock_agent_cls.return_value = mock_agent

            success, output, final_response, error = run_job(job)

        assert success is True
        assert error is None
        assert final_response == "ok"

        kwargs = mock_agent_cls.call_args.kwargs
        assert "cronjob" in (kwargs["disabled_toolsets"] or [])

        prompt_arg = mock_agent.run_conversation.call_args.args[0]
        assert "blogwatcher" in prompt_arg
        assert "Follow this skill" in prompt_arg
        assert "Check the feeds and summarize anything new." in prompt_arg

    def test_run_job_loads_multiple_skills_in_order(self, tmp_path):
        job = {
            "id": "multi-skill-job",
            "name": "multi skill test",
            "prompt": "Combine the results.",
            "skills": ["blogwatcher", "maps"],
        }

        fake_db = MagicMock()

        def _skill_view(name):
            return json.dumps({"success": True, "content": f"# {name}\nInstructions for {name}."})

        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("hermes_cli.env_loader.load_hermes_dotenv"), \
             patch("hermes_cli.env_loader.reset_secret_source_cache"), \
             patch("hermes_state.SessionDB", return_value=fake_db), \
             patch(
                 "hermes_cli.runtime_provider.resolve_runtime_provider",
                 return_value={
                     "api_key": "***",
                     "base_url": "https://example.invalid/v1",
                     "provider": "openrouter",
                     "api_mode": "chat_completions",
                 },
             ), \
             patch("tools.skills_tool.skill_view", side_effect=_skill_view) as skill_view_mock, \
             patch("run_agent.AIAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {"final_response": "ok"}
            mock_agent_cls.return_value = mock_agent

            success, output, final_response, error = run_job(job)

        assert success is True
        assert error is None
        assert final_response == "ok"
        assert skill_view_mock.call_count == 2
        assert [call.args[0] for call in skill_view_mock.call_args_list] == ["blogwatcher", "maps"]

        prompt_arg = mock_agent.run_conversation.call_args.args[0]
        assert prompt_arg.index("blogwatcher") < prompt_arg.index("maps")
        assert "Instructions for blogwatcher." in prompt_arg
        assert "Instructions for maps." in prompt_arg
        assert "Combine the results." in prompt_arg


class TestSilentDelivery:
    """Verify that [SILENT] responses suppress delivery while still saving output."""

    def _make_job(self):
        return {
            "id": "monitor-job",
            "name": "monitor",
            "deliver": "origin",
            "origin": {"platform": "telegram", "chat_id": "123"},
        }

    def test_silent_response_suppresses_delivery(self, caplog):
        with patch("cron.scheduler.get_due_jobs", return_value=[self._make_job()]), \
             patch("cron.scheduler.run_job", return_value=(True, "# output", "[SILENT]", None)), \
             patch("cron.scheduler.save_job_output", return_value="/tmp/out.md"), \
             patch("cron.scheduler._deliver_result") as deliver_mock, \
             patch("cron.scheduler.mark_job_run"):
            from cron.scheduler import tick
            with caplog.at_level(logging.INFO, logger="cron.scheduler"):
                tick(verbose=False)
        deliver_mock.assert_not_called()
        assert any(SILENT_MARKER in r.message for r in caplog.records)

    def test_silent_with_note_suppresses_delivery(self):
        with patch("cron.scheduler.get_due_jobs", return_value=[self._make_job()]), \
             patch("cron.scheduler.run_job", return_value=(True, "# output", "[SILENT] No changes detected", None)), \
             patch("cron.scheduler.save_job_output", return_value="/tmp/out.md"), \
             patch("cron.scheduler._deliver_result") as deliver_mock, \
             patch("cron.scheduler.mark_job_run"):
            from cron.scheduler import tick
            tick(verbose=False)
        deliver_mock.assert_not_called()

    def test_silent_trailing_suppresses_delivery(self):
        """Agent appended [SILENT] after explanation text — must still suppress."""
        response = "2 deals filtered out (like<10, reply<15).\n\n[SILENT]"
        with patch("cron.scheduler.get_due_jobs", return_value=[self._make_job()]), \
             patch("cron.scheduler.run_job", return_value=(True, "# output", response, None)), \
             patch("cron.scheduler.save_job_output", return_value="/tmp/out.md"), \
             patch("cron.scheduler._deliver_result") as deliver_mock, \
             patch("cron.scheduler.mark_job_run"):
            from cron.scheduler import tick
            tick(verbose=False)
        deliver_mock.assert_not_called()

    def test_silent_is_case_insensitive(self):
        with patch("cron.scheduler.get_due_jobs", return_value=[self._make_job()]), \
             patch("cron.scheduler.run_job", return_value=(True, "# output", "[silent] nothing new", None)), \
             patch("cron.scheduler.save_job_output", return_value="/tmp/out.md"), \
             patch("cron.scheduler._deliver_result") as deliver_mock, \
             patch("cron.scheduler.mark_job_run"):
            from cron.scheduler import tick
            tick(verbose=False)
        deliver_mock.assert_not_called()

    def test_bracketless_silent_variants_suppress(self):
        """Bracketless near-markers the model emits when it drops brackets
        must still suppress delivery (#51438, #46917)."""
        from cron.scheduler import tick
        for marker in ("SILENT", "NO_REPLY", "NO REPLY", "no_reply"):
            with patch("cron.scheduler.get_due_jobs", return_value=[self._make_job()]), \
                 patch("cron.scheduler.run_job", return_value=(True, "# output", marker, None)), \
                 patch("cron.scheduler.save_job_output", return_value="/tmp/out.md"), \
                 patch("cron.scheduler._deliver_result") as deliver_mock, \
                 patch("cron.scheduler.mark_job_run"):
                tick(verbose=False)
            deliver_mock.assert_not_called()

    def test_report_quoting_marker_mid_sentence_still_delivers(self):
        """A genuine report that merely mentions the token mid-sentence must
        be delivered — the old substring check wrongly swallowed it."""
        response = "I considered staying [SILENT] but here is the summary: 3 items merged."
        with patch("cron.scheduler.get_due_jobs", return_value=[self._make_job()]), \
             patch("cron.scheduler.run_job", return_value=(True, "# output", response, None)), \
             patch("cron.scheduler.save_job_output", return_value="/tmp/out.md"), \
             patch("cron.scheduler._deliver_result") as deliver_mock, \
             patch("cron.scheduler.mark_job_run"):
            from cron.scheduler import tick
            tick(verbose=False)
        deliver_mock.assert_called_once()

    def test_is_cron_silence_response_contract(self):
        """Direct behavior contract for the cron silence matcher."""
        from cron.scheduler import _is_cron_silence_response as sil
        # Suppress: bare/bracketed/bracketless tokens, prefix, trailing-line.
        assert sil("[SILENT]")
        assert sil("[silent] nothing new")
        assert sil("[SILENT] No changes detected")
        assert sil("2 deals filtered.\n\n[SILENT]")
        assert sil("SILENT")
        assert sil("NO_REPLY")
        assert sil("NO REPLY")
        assert sil("Summary.\nSILENT")
        # Deliver: real content, mid-sentence quotes, bare words, junk.
        assert not sil("Daily report: 4 PRs merged.")
        assert not sil("I stayed [SILENT] but here is the report: 3 items.")
        assert not sil("Silent retry succeeded after 2 attempts.")
        assert not sil("[SILENT")  # malformed open-bracket is not the sentinel
        assert not sil("")
        assert not sil("   \n\t ")

    def test_failed_job_always_delivers(self):
        """Failed jobs deliver regardless of [SILENT] in output."""
        with patch("cron.scheduler.get_due_jobs", return_value=[self._make_job()]), \
             patch("cron.scheduler.run_job", return_value=(False, "# output", "", "some error")), \
             patch("cron.scheduler.save_job_output", return_value="/tmp/out.md"), \
             patch("cron.scheduler._deliver_result") as deliver_mock, \
             patch("cron.scheduler.mark_job_run"):
            from cron.scheduler import tick
            tick(verbose=False)
        deliver_mock.assert_called_once()

    def test_output_saved_even_when_delivery_suppressed(self):
        with patch("cron.scheduler.get_due_jobs", return_value=[self._make_job()]), \
             patch("cron.scheduler.run_job", return_value=(True, "# full output", "[SILENT]", None)), \
             patch("cron.scheduler.save_job_output") as save_mock, \
             patch("cron.scheduler._deliver_result") as deliver_mock, \
             patch("cron.scheduler.mark_job_run"):
            save_mock.return_value = "/tmp/out.md"
            from cron.scheduler import tick
            tick(verbose=False)
        save_mock.assert_called_once_with("monitor-job", "# full output")
        deliver_mock.assert_not_called()

    def test_whitespace_only_response_is_marked_failed_not_delivered(self):
        """Whitespace-only final responses should behave like empty responses."""
        with patch("cron.scheduler.get_due_jobs", return_value=[self._make_job()]), \
             patch("cron.scheduler.run_job", return_value=(True, "# output", "   \n\t  ", None)), \
             patch("cron.scheduler.save_job_output", return_value="/tmp/out.md"), \
             patch("cron.scheduler._deliver_result") as deliver_mock, \
             patch("cron.scheduler.mark_job_run") as mark_mock:
            from cron.scheduler import tick
            tick(verbose=False)

        deliver_mock.assert_not_called()
        mark_mock.assert_called_once_with(
            "monitor-job",
            False,
            "Agent completed but produced empty response (model error, timeout, or misconfiguration)",
            delivery_error=None,
        )


class TestOneShotDispatchClaim:
    """run_one_job must claim a finite one-shot's dispatch BEFORE run_job so a
    tick that dies mid-execution can't re-fire it forever (issue #38758)."""

    def _oneshot(self):
        return {
            "id": "monitor-job",
            "name": "monitor",
            "deliver": "origin",
            "origin": {"platform": "telegram", "chat_id": "123"},
            "schedule": {"kind": "once", "run_at": "2026-01-01T00:00:00+00:00"},
            "repeat": {"times": 1, "completed": 0},
        }

    def test_claim_runs_before_run_job(self):
        order = []
        with patch("cron.scheduler.get_due_jobs", return_value=[self._oneshot()]), \
             patch("cron.scheduler.claim_dispatch", side_effect=lambda _id: order.append("claim") or True), \
             patch("cron.scheduler.run_job", side_effect=lambda _j, **_kw: order.append("run") or (True, "# out", "ok", None)), \
             patch("cron.scheduler.save_job_output", return_value="/tmp/out.md"), \
             patch("cron.scheduler._deliver_result"), \
             patch("cron.scheduler.mark_job_run"):
            from cron.scheduler import tick
            tick(verbose=False)
        assert order == ["claim", "run"]  # claim strictly before side effect

    def test_refused_claim_skips_run_job(self):
        with patch("cron.scheduler.get_due_jobs", return_value=[self._oneshot()]), \
             patch("cron.scheduler.claim_dispatch", return_value=False), \
             patch("cron.scheduler.run_job") as run_mock, \
             patch("cron.scheduler.save_job_output"), \
             patch("cron.scheduler._deliver_result") as deliver_mock, \
             patch("cron.scheduler.mark_job_run") as mark_mock:
            from cron.scheduler import tick
            tick(verbose=False)
        run_mock.assert_not_called()
        deliver_mock.assert_not_called()
        mark_mock.assert_not_called()


class TestBuildJobPromptSilentHint:
    """Verify _build_job_prompt always injects [SILENT] guidance."""

    def test_hint_always_present(self):
        job = {"prompt": "Check for updates"}
        result = _build_job_prompt(job)
        assert "[SILENT]" in result
        assert "Check for updates" in result

    def test_hint_present_even_without_prompt(self):
        job = {"prompt": ""}
        result = _build_job_prompt(job)
        assert "[SILENT]" in result

    def test_hint_present_when_legacy_prompt_is_null(self):
        job = {"id": "abc123deadbe", "name": None, "prompt": None}
        result = _build_job_prompt(job)
        assert "[SILENT]" in result

    def test_delivery_guidance_present(self):
        """Cron hint tells agents their final response is auto-delivered."""
        job = {"prompt": "Generate a report"}
        result = _build_job_prompt(job)
        assert "do NOT use send_message" in result
        assert "automatically delivered" in result

    def test_delivery_guidance_precedes_user_prompt(self):
        """System guidance appears before the user's prompt text."""
        job = {"prompt": "My custom prompt"}
        result = _build_job_prompt(job)
        system_pos = result.index("do NOT use send_message")
        prompt_pos = result.index("My custom prompt")
        assert system_pos < prompt_pos


class TestParseWakeGate:
    """Unit tests for _parse_wake_gate — pure function, no side effects."""

    def test_empty_output_wakes(self):
        from cron.scheduler import _parse_wake_gate
        assert _parse_wake_gate("") is True
        assert _parse_wake_gate(None) is True

    def test_whitespace_only_wakes(self):
        from cron.scheduler import _parse_wake_gate
        assert _parse_wake_gate("   \n\n  \t\n") is True

    def test_non_json_last_line_wakes(self):
        from cron.scheduler import _parse_wake_gate
        assert _parse_wake_gate("hello world") is True
        assert _parse_wake_gate("line 1\nline 2\nplain text") is True

    def test_json_non_dict_wakes(self):
        """Bare arrays, numbers, strings must not be interpreted as a gate."""
        from cron.scheduler import _parse_wake_gate
        assert _parse_wake_gate("[1, 2, 3]") is True
        assert _parse_wake_gate("42") is True
        assert _parse_wake_gate('"wakeAgent"') is True

    def test_wake_gate_false_skips(self):
        from cron.scheduler import _parse_wake_gate
        assert _parse_wake_gate('{"wakeAgent": false}') is False

    def test_wake_gate_true_wakes(self):
        from cron.scheduler import _parse_wake_gate
        assert _parse_wake_gate('{"wakeAgent": true}') is True

    def test_wake_gate_missing_wakes(self):
        """A JSON dict without a wakeAgent key defaults to waking."""
        from cron.scheduler import _parse_wake_gate
        assert _parse_wake_gate('{"data": {"foo": "bar"}}') is True

    def test_non_boolean_false_still_wakes(self):
        """Only strict ``False`` skips — truthy/falsy shortcuts are too risky."""
        from cron.scheduler import _parse_wake_gate
        assert _parse_wake_gate('{"wakeAgent": 0}') is True
        assert _parse_wake_gate('{"wakeAgent": null}') is True
        assert _parse_wake_gate('{"wakeAgent": ""}') is True

    def test_only_last_non_empty_line_parsed(self):
        from cron.scheduler import _parse_wake_gate
        multi = 'some log output\nmore output\n{"wakeAgent": false}'
        assert _parse_wake_gate(multi) is False

    def test_trailing_blank_lines_ignored(self):
        from cron.scheduler import _parse_wake_gate
        multi = '{"wakeAgent": false}\n\n\n'
        assert _parse_wake_gate(multi) is False

    def test_non_last_json_line_does_not_gate(self):
        """A JSON gate on an earlier line with plain text after it does NOT trigger."""
        from cron.scheduler import _parse_wake_gate
        multi = '{"wakeAgent": false}\nactually this is the real output'
        assert _parse_wake_gate(multi) is True


class TestRunJobWakeGate:
    """Integration tests for run_job wake-gate short-circuit."""

    @pytest.fixture(autouse=True)
    def _stub_runtime_provider(self):
        """Stub ``resolve_runtime_provider`` for wake-gate tests.

        ``run_job`` resolves the runtime provider BEFORE constructing
        ``AIAgent``, so these tests must mock ``resolve_runtime_provider``
        in addition to ``AIAgent`` — otherwise in a hermetic CI env (no
        API keys), the resolver raises and the test fails before the
        patched AIAgent is ever reached.
        """
        fake_runtime = {
            "provider": "openrouter",
            "api_mode": "chat_completions",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "test-key",
            "source": "stub",
            "requested_provider": None,
        }
        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value=fake_runtime,
        ):
            yield

    def _make_job(self, name="wake-gate-test", script="check.py"):
        """Minimal valid cron job dict for run_job."""
        return {
            "id": f"job_{name}",
            "name": name,
            "prompt": "Do a thing",
            "schedule": "*/5 * * * *",
            "script": script,
        }

    def test_wake_false_skips_agent_and_returns_silent(self, caplog):
        """When _run_job_script output ends with {wakeAgent: false}, the agent
        is not invoked and run_job returns the SILENT marker so delivery is
        suppressed."""
        from cron.scheduler import SILENT_MARKER
        import cron.scheduler as scheduler

        with patch.object(scheduler, "_run_job_script",
                          return_value=(True, '{"wakeAgent": false}')), \
             patch("run_agent.AIAgent") as agent_cls:
            success, doc, final, err = scheduler.run_job(self._make_job())

        assert success is True
        assert err is None
        assert final == SILENT_MARKER
        assert "Script gate returned `wakeAgent=false`" in doc
        agent_cls.assert_not_called()

    def test_wake_true_runs_agent_with_injected_output(self):
        """When the script returns {wakeAgent: true, data: ...}, the agent is
        invoked and the data line still shows up in the prompt."""
        import cron.scheduler as scheduler

        script_output = '{"wakeAgent": true, "data": {"new": 3}}'
        agent = MagicMock()
        agent.run_conversation = MagicMock(return_value={
            "final_response": "ok", "messages": []
        })
        with patch.object(scheduler, "_run_job_script",
                          return_value=(True, script_output)), \
             patch("run_agent.AIAgent", return_value=agent) as agent_cls:
            success, doc, final, err = scheduler.run_job(self._make_job())

        agent_cls.assert_called_once()
        # The script output should be visible in the prompt passed to
        # run_conversation.
        call_kwargs = agent.run_conversation.call_args
        prompt_arg = call_kwargs.args[0] if call_kwargs.args else call_kwargs.kwargs.get("user_message", "")
        assert script_output in prompt_arg
        assert success is True
        assert err is None

    def test_script_runs_only_once_on_wake(self):
        """Wake-true path must not re-run the script inside _build_job_prompt
        (script would execute twice otherwise, wasting work and risking
        double-side-effects)."""
        import cron.scheduler as scheduler

        call_count = 0
        def _script_stub(path, **_kwargs):
            nonlocal call_count
            call_count += 1
            return (True, "regular output")

        agent = MagicMock()
        agent.run_conversation = MagicMock(return_value={
            "final_response": "ok", "messages": []
        })
        with patch.object(scheduler, "_run_job_script", side_effect=_script_stub), \
             patch("run_agent.AIAgent", return_value=agent):
            scheduler.run_job(self._make_job())

        assert call_count == 1, f"script ran {call_count}x, expected exactly 1"

    def test_script_failure_does_not_trigger_gate(self):
        """If _run_job_script returns success=False, the gate is NOT evaluated
        and the agent still runs (the failure is reported as context)."""
        import cron.scheduler as scheduler

        # Malicious or broken script whose stderr happens to contain the
        # gate JSON — we must NOT honor it because ran_ok is False.
        agent = MagicMock()
        agent.run_conversation = MagicMock(return_value={
            "final_response": "ok", "messages": []
        })
        with patch.object(scheduler, "_run_job_script",
                          return_value=(False, '{"wakeAgent": false}')), \
             patch("run_agent.AIAgent", return_value=agent) as agent_cls:
            success, doc, final, err = scheduler.run_job(self._make_job())

        agent_cls.assert_called_once()  # Agent DID wake despite the gate-like text

    def test_no_script_path_runs_agent_normally(self):
        """Regression: jobs without a script still work."""
        import cron.scheduler as scheduler

        agent = MagicMock()
        agent.run_conversation = MagicMock(return_value={
            "final_response": "ok", "messages": []
        })
        job = self._make_job(script=None)
        job.pop("script", None)
        with patch.object(scheduler, "_run_job_script") as script_fn, \
             patch("run_agent.AIAgent", return_value=agent) as agent_cls:
            scheduler.run_job(job)

        script_fn.assert_not_called()
        agent_cls.assert_called_once()


class TestBuildJobPromptMissingSkill:
    """Verify that a missing skill logs a warning and does not crash the job."""

    def _missing_skill_view(self, name: str) -> str:
        return json.dumps({"success": False, "error": f"Skill '{name}' not found."})

    def test_missing_skill_does_not_raise(self):
        """Job should run even when a referenced skill is not installed."""
        with patch("tools.skills_tool.skill_view", side_effect=self._missing_skill_view):
            result = _build_job_prompt({"skills": ["ghost-skill"], "prompt": "do something"})
        # prompt is preserved even though skill was skipped
        assert "do something" in result

    def test_missing_skill_injects_user_notice_into_prompt(self):
        """A system notice about the missing skill is injected into the prompt."""
        with patch("tools.skills_tool.skill_view", side_effect=self._missing_skill_view):
            result = _build_job_prompt({"skills": ["ghost-skill"], "prompt": "do something"})
        assert "ghost-skill" in result
        assert "not found" in result.lower() or "skipped" in result.lower()

    def test_missing_skill_logs_warning(self, caplog):
        """A warning is logged when a skill cannot be found."""
        with caplog.at_level(logging.WARNING, logger="cron.scheduler"):
            with patch("tools.skills_tool.skill_view", side_effect=self._missing_skill_view):
                _build_job_prompt({"name": "My Job", "skills": ["ghost-skill"], "prompt": "do something"})
        assert any("ghost-skill" in record.message for record in caplog.records)

    def test_valid_skill_loaded_alongside_missing(self):
        """A valid skill is still loaded when another skill in the list is missing."""

        def _mixed_skill_view(name: str) -> str:
            if name == "real-skill":
                return json.dumps({"success": True, "content": "Real skill content."})
            return json.dumps({"success": False, "error": f"Skill '{name}' not found."})

        with patch("tools.skills_tool.skill_view", side_effect=_mixed_skill_view):
            result = _build_job_prompt({"skills": ["ghost-skill", "real-skill"], "prompt": "go"})
        assert "Real skill content." in result
        assert "go" in result


class TestBuildJobPromptAbsoluteSkillPath:
    """Cron jobs may store absolute skill paths; normalize before skill_view."""

    def test_absolute_skill_path_normalized_before_skill_view(self, tmp_path):
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "alpha-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("# Alpha\nDo alpha.")
        absolute_path = str(skill_dir)
        seen_names: list[str] = []

        def _skill_view(name: str) -> str:
            seen_names.append(name)
            if name == "alpha-skill":
                return json.dumps({"success": True, "content": "# Alpha\nDo alpha."})
            return json.dumps({"success": False, "error": f"Skill '{name}' not found."})

        with patch("tools.skills_tool.SKILLS_DIR", skills_dir), \
             patch("tools.skills_tool.skill_view", side_effect=_skill_view):
            result = _build_job_prompt({"skills": [absolute_path], "prompt": "go"})

        assert seen_names == ["alpha-skill"]
        assert "Do alpha." in result


class TestBuildJobPromptBumpUse:
    """Verify that cron jobs bump skill usage counters so the curator sees them as active."""

    def test_bump_use_called_for_loaded_skill(self):
        """bump_use is called for each successfully loaded skill."""

        def _skill_view(name: str) -> str:
            return json.dumps({"success": True, "content": f"Content for {name}."})

        with patch("tools.skills_tool.skill_view", side_effect=_skill_view), \
             patch("tools.skill_usage.bump_use") as mock_bump:
            _build_job_prompt({"skills": ["alpha", "beta"], "prompt": "go"})

        assert mock_bump.call_count == 2
        calls = [c[0][0] for c in mock_bump.call_args_list]
        assert "alpha" in calls
        assert "beta" in calls

    def test_bump_use_not_called_for_missing_skill(self):
        """bump_use is NOT called when a skill fails to load."""

        def _missing_view(name: str) -> str:
            return json.dumps({"success": False, "error": "not found"})

        with patch("tools.skills_tool.skill_view", side_effect=_missing_view), \
             patch("tools.skill_usage.bump_use") as mock_bump:
            _build_job_prompt({"skills": ["ghost"], "prompt": "go"})

        assert mock_bump.call_count == 0

    def test_bump_failure_does_not_break_prompt(self, caplog):
        """If bump_use raises, the prompt still builds — error is logged at DEBUG."""

        def _skill_view(name: str) -> str:
            return json.dumps({"success": True, "content": "Works."})

        with patch("tools.skills_tool.skill_view", side_effect=_skill_view), \
             patch("tools.skill_usage.bump_use", side_effect=RuntimeError("boom")), \
             caplog.at_level(logging.DEBUG, logger="cron.scheduler"):
            result = _build_job_prompt({"skills": ["good-skill"], "prompt": "go"})

        # Prompt should still contain the skill content and original instruction
        assert "Works." in result
        assert "go" in result
        # The error should be logged at DEBUG level, not crash
        assert any("failed to bump" in r.message for r in caplog.records)


class TestSendMediaViaAdapter:
    """Unit tests for _send_media_via_adapter — routes files to typed adapter methods."""

    def _safe_media_path(self, tmp_path, monkeypatch, name, data=b"media"):
        root = tmp_path / "media-cache"
        media_file = root / name
        media_file.parent.mkdir(parents=True, exist_ok=True)
        media_file.write_bytes(data)
        monkeypatch.setattr(
            "gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS",
            (root,),
        )
        return media_file.resolve()

    @staticmethod
    def _run_with_loop(adapter, chat_id, media_files, metadata, job):
        """Helper: run _send_media_via_adapter with immediate scheduling."""
        from concurrent.futures import Future

        def fake_run_coro(coro, _loop):
            coro.close()
            completed = Future()
            completed.set_result(MagicMock(success=True))
            return completed

        with patch("asyncio.run_coroutine_threadsafe", side_effect=fake_run_coro):
            _send_media_via_adapter(adapter, chat_id, media_files, metadata, MagicMock(), job)

    def test_video_dispatched_to_send_video(self, tmp_path, monkeypatch):
        adapter = MagicMock()
        adapter.send_video = AsyncMock()
        media_path = self._safe_media_path(tmp_path, monkeypatch, "clip.mp4")
        media_files = [(str(media_path), False)]
        self._run_with_loop(adapter, "123", media_files, None, {"id": "j1"})
        adapter.send_video.assert_called_once()
        assert adapter.send_video.call_args[1]["video_path"] == str(media_path)

    def test_unknown_ext_dispatched_to_send_document(self, tmp_path, monkeypatch):
        adapter = MagicMock()
        adapter.send_document = AsyncMock()
        media_path = self._safe_media_path(tmp_path, monkeypatch, "report.pdf")
        media_files = [(str(media_path), False)]
        self._run_with_loop(adapter, "123", media_files, None, {"id": "j2"})
        adapter.send_document.assert_called_once()
        assert adapter.send_document.call_args[1]["file_path"] == str(media_path)

    def test_multiple_media_files_all_delivered(self, tmp_path, monkeypatch):
        adapter = MagicMock()
        adapter.send_voice = AsyncMock()
        adapter.send_image_file = AsyncMock()
        voice_path = self._safe_media_path(tmp_path, monkeypatch, "voice.mp3")
        photo_path = self._safe_media_path(tmp_path, monkeypatch, "photo.jpg")
        media_files = [(str(voice_path), False), (str(photo_path), False)]
        self._run_with_loop(adapter, "123", media_files, None, {"id": "j3"})
        adapter.send_voice.assert_called_once()
        adapter.send_image_file.assert_called_once()


class TestParallelTick:
    """Verify that tick() runs due jobs concurrently and isolates ContextVars."""

    @pytest.fixture(autouse=True)
    def _isolate_tick_lock(self, tmp_path):
        """Point the tick file lock at a per-test temp dir to avoid xdist contention."""
        lock_dir = tmp_path / "cron"
        lock_dir.mkdir()
        lock_file = lock_dir / ".tick.lock"
        with patch("cron.scheduler._get_lock_paths", return_value=(lock_dir, lock_file)):
            yield

    def test_parallel_jobs_run_concurrently(self):
        """Two jobs launched in the same tick should overlap in time."""
        import threading

        barrier = threading.Barrier(2, timeout=5)
        call_order = []

        def mock_run_job(job, *, defer_agent_teardown=None):
            """Each job hits a barrier — both must be active simultaneously."""
            call_order.append(("start", job["id"]))
            barrier.wait()  # blocks until both threads reach here
            call_order.append(("end", job["id"]))
            return (True, "output", "response", None)

        jobs = [
            {"id": "job-a", "name": "a", "deliver": "local"},
            {"id": "job-b", "name": "b", "deliver": "local"},
        ]

        with patch("cron.scheduler.get_due_jobs", return_value=jobs), \
             patch("cron.scheduler.advance_next_run"), \
             patch("cron.scheduler.run_job", side_effect=mock_run_job), \
             patch("cron.scheduler.save_job_output", return_value="/tmp/out.md"), \
             patch("cron.scheduler._deliver_result", return_value=None), \
             patch("cron.scheduler.mark_job_run"):
            from cron.scheduler import tick
            result = tick(verbose=False)

        assert result == 2
        # Both starts happened before both ends — proof of concurrency
        starts = [i for i, (action, _) in enumerate(call_order) if action == "start"]
        ends = [i for i, (action, _) in enumerate(call_order) if action == "end"]
        assert len(starts) == 2
        assert len(ends) == 2
        assert max(starts) < min(ends), f"Jobs not concurrent: {call_order}"

    def test_parallel_jobs_isolated_contextvars(self):
        """Each job's ContextVars must be isolated — no cross-contamination."""
        from gateway.session_context import get_session_env
        seen = {}

        def mock_run_job(job, *, defer_agent_teardown=None):
            origin = job.get("origin", {})
            # run_job sets ContextVars — verify each job sees its own
            from gateway.session_context import set_session_vars, clear_session_vars
            tokens = set_session_vars(
                platform=origin.get("platform", ""),
                chat_id=str(origin.get("chat_id", "")),
            )
            import time
            time.sleep(0.05)  # give other thread time to set its vars
            platform = get_session_env("HERMES_SESSION_PLATFORM")
            chat_id = get_session_env("HERMES_SESSION_CHAT_ID")
            seen[job["id"]] = {"platform": platform, "chat_id": chat_id}
            clear_session_vars(tokens)
            return (True, "output", "response", None)

        jobs = [
            {"id": "tg-job", "name": "tg", "deliver": "local",
             "origin": {"platform": "telegram", "chat_id": "111"}},
            {"id": "dc-job", "name": "dc", "deliver": "local",
             "origin": {"platform": "discord", "chat_id": "222"}},
        ]

        with patch("cron.scheduler.get_due_jobs", return_value=jobs), \
             patch("cron.scheduler.advance_next_run"), \
             patch("cron.scheduler.run_job", side_effect=mock_run_job), \
             patch("cron.scheduler.save_job_output", return_value="/tmp/out.md"), \
             patch("cron.scheduler._deliver_result", return_value=None), \
             patch("cron.scheduler.mark_job_run"):
            from cron.scheduler import tick
            tick(verbose=False)

        assert seen["tg-job"] == {"platform": "telegram", "chat_id": "111"}
        assert seen["dc-job"] == {"platform": "discord", "chat_id": "222"}

    def test_max_parallel_env_var(self, monkeypatch):
        """HERMES_CRON_MAX_PARALLEL=1 should restore serial behaviour."""
        monkeypatch.setenv("HERMES_CRON_MAX_PARALLEL", "1")
        call_times = []

        def mock_run_job(job, *, defer_agent_teardown=None):
            import time
            call_times.append(("start", job["id"], time.monotonic()))
            time.sleep(0.05)
            call_times.append(("end", job["id"], time.monotonic()))
            return (True, "output", "response", None)

        jobs = [
            {"id": "s1", "name": "s1", "deliver": "local"},
            {"id": "s2", "name": "s2", "deliver": "local"},
        ]

        with patch("cron.scheduler.get_due_jobs", return_value=jobs), \
             patch("cron.scheduler.advance_next_run"), \
             patch("cron.scheduler.run_job", side_effect=mock_run_job), \
             patch("cron.scheduler.save_job_output", return_value="/tmp/out.md"), \
             patch("cron.scheduler._deliver_result", return_value=None), \
             patch("cron.scheduler.mark_job_run"):
            from cron.scheduler import tick
            result = tick(verbose=False)

        assert result == 2
        # With max_workers=1, second job starts after first ends
        end_s1 = [t for action, jid, t in call_times if action == "end" and jid == "s1"][0]
        start_s2 = [t for action, jid, t in call_times if action == "start" and jid == "s2"][0]
        assert start_s2 >= end_s1, "Jobs ran concurrently despite max_parallel=1"


class TestDeliverResultTimeoutCancelsFuture:
    """When future.result(timeout=60) raises TimeoutError in the live adapter
    delivery path, the outcome depends on whether the coroutine was already
    running.  future.cancel() returning False means it is in flight on the wire
    (cannot be un-sent) → treat as DELIVERED and skip the standalone fallback to
    avoid a duplicate (#38922).  future.cancel() returning True means it never
    started (wedged loop) → nothing was sent, so fall through to standalone or
    the message is silently dropped.  Regression for #38922.
    """

    def test_live_adapter_timeout_assumes_delivered_no_duplicate(self):
        """End-to-end: live adapter confirmation times out past the 60s budget.
        The fix (#38922) treats the send as already-dispatched/delivered and
        does NOT run the standalone fallback — otherwise the message is sent
        twice."""
        from gateway.config import Platform
        from concurrent.futures import Future

        # Live adapter whose send() coroutine never resolves within the budget
        adapter = AsyncMock()
        adapter.send.return_value = MagicMock(success=True)

        pconfig = MagicMock()
        pconfig.enabled = True
        mock_cfg = MagicMock()
        mock_cfg.platforms = {Platform.TELEGRAM: pconfig}

        loop = MagicMock()
        loop.is_running.return_value = True

        # A real concurrent.futures.Future, but we override .result() to raise
        # TimeoutError exactly like the 60s wait firing in production.  We make
        # .cancel() return False to simulate the coroutine being ALREADY RUNNING
        # on the gateway loop (in flight on the wire) — the case where the send
        # cannot be un-sent and a standalone resend would be a duplicate.
        captured_future = Future()
        cancel_calls = []

        def in_flight_cancel():
            cancel_calls.append(True)
            return False  # already running — cannot be cancelled

        captured_future.cancel = in_flight_cancel
        captured_future.result = MagicMock(side_effect=TimeoutError("timed out"))

        def fake_run_coro(coro, _loop):
            coro.close()
            return captured_future

        job = {
            "id": "timeout-job",
            "deliver": "origin",
            "origin": {"platform": "telegram", "chat_id": "123"},
        }

        standalone_send = AsyncMock(return_value={"success": True})

        with patch("gateway.config.load_gateway_config", return_value=mock_cfg), \
             patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
             patch("asyncio.run_coroutine_threadsafe", side_effect=fake_run_coro), \
             patch("tools.send_message_tool._send_to_platform", new=standalone_send):
            result = _deliver_result(
                job,
                "Hello world",
                adapters={Platform.TELEGRAM: adapter},
                loop=loop,
            )

        # 1. cancel() was attempted (returned False = in flight).
        assert cancel_calls == [True], "future.cancel() should be attempted on TimeoutError"
        # 2. Delivery is reported successful (no error string returned).
        assert result is None, f"expected successful delivery, got error: {result!r}"
        # 3. The standalone fallback must NOT run — that is the #38922 fix:
        #    an in-flight confirmation timeout is assume-delivered, not a resend.
        standalone_send.assert_not_awaited()

    def test_live_adapter_timeout_before_dispatch_falls_back_to_standalone(self):
        """When the coroutine never started (loop wedged) — future.cancel()
        returns True — nothing was sent, so _deliver_result MUST fall through
        to the standalone path rather than silently dropping the message.
        This is the inverse of the assume-delivered case and guards against the
        wedged-loop silent drop."""
        from gateway.config import Platform
        from concurrent.futures import Future

        adapter = AsyncMock()
        adapter.send.return_value = MagicMock(success=True)

        pconfig = MagicMock()
        pconfig.enabled = True
        mock_cfg = MagicMock()
        mock_cfg.platforms = {Platform.TELEGRAM: pconfig}

        loop = MagicMock()
        loop.is_running.return_value = True

        captured_future = Future()
        cancel_calls = []

        def never_dispatched_cancel():
            cancel_calls.append(True)
            return True  # callback never ran — successfully cancelled

        captured_future.cancel = never_dispatched_cancel
        captured_future.result = MagicMock(side_effect=TimeoutError("timed out"))

        def fake_run_coro(coro, _loop):
            coro.close()
            return captured_future

        job = {
            "id": "timeout-undispatched-job",
            "deliver": "origin",
            "origin": {"platform": "telegram", "chat_id": "123"},
        }

        standalone_send = AsyncMock(return_value={"success": True})

        with patch("gateway.config.load_gateway_config", return_value=mock_cfg), \
             patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
             patch("asyncio.run_coroutine_threadsafe", side_effect=fake_run_coro), \
             patch("tools.send_message_tool._send_to_platform", new=standalone_send):
            result = _deliver_result(
                job,
                "Hello world",
                adapters={Platform.TELEGRAM: adapter},
                loop=loop,
            )

        assert cancel_calls == [True], "future.cancel() should be attempted"
        # The standalone path MUST run — the message was never sent.
        standalone_send.assert_awaited_once()
        assert result is None, f"standalone should have delivered, got: {result!r}"

    def test_live_adapter_real_exception_falls_back_to_standalone(self):
        """A non-timeout send Exception (real failure, not a slow confirmation)
        must fall through to the standalone path so the message is still
        delivered.  Guards the `except Exception: raise` branch — the bug class
        where broadening the timeout handler to swallow all exceptions would
        silently drop messages."""
        from gateway.config import Platform
        from concurrent.futures import Future

        adapter = AsyncMock()
        adapter.send.return_value = MagicMock(success=True)

        pconfig = MagicMock()
        pconfig.enabled = True
        mock_cfg = MagicMock()
        mock_cfg.platforms = {Platform.TELEGRAM: pconfig}

        loop = MagicMock()
        loop.is_running.return_value = True

        captured_future = Future()
        captured_future.result = MagicMock(side_effect=RuntimeError("adapter exploded"))

        def fake_run_coro(coro, _loop):
            coro.close()
            return captured_future

        job = {
            "id": "send-error-job",
            "deliver": "origin",
            "origin": {"platform": "telegram", "chat_id": "123"},
        }

        standalone_send = AsyncMock(return_value={"success": True})

        with patch("gateway.config.load_gateway_config", return_value=mock_cfg), \
             patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
             patch("asyncio.run_coroutine_threadsafe", side_effect=fake_run_coro), \
             patch("tools.send_message_tool._send_to_platform", new=standalone_send):
            result = _deliver_result(
                job,
                "Hello world",
                adapters={Platform.TELEGRAM: adapter},
                loop=loop,
            )

        # A real exception must NOT be assume-delivered: standalone runs.
        standalone_send.assert_awaited_once()
        assert result is None, f"standalone should have delivered, got: {result!r}"

    def test_live_adapter_forum_topic_in_private_chat_routes_via_message_thread_id(self):
        """#52060: a cron target to a PRIVATE Telegram chat with a numeric topic
        id is a normal forum-style topic — it must route via ``message_thread_id``,
        NOT ``direct_messages_topic_id``.  The #22773 heuristic inferred a Bot API
        channel DM topic from positive chat_id + numeric thread and nulled
        ``message_thread_id``, so deliveries landed in General.  We now probe the
        live adapter's ``get_chat_info``; a non-channel chat routes via
        ``message_thread_id``.
        """
        from gateway.config import Platform
        from gateway.platforms.base import SendResult
        from concurrent.futures import Future

        send_result = SendResult(success=True, message_id="42")

        class _ForumAdapter(MagicMock):
            async def get_chat_info(self, chat_id):
                return {"name": "Proyectos", "type": "forum", "is_forum": True}

        adapter = _ForumAdapter()
        adapter.send = AsyncMock(return_value=send_result)

        pconfig = MagicMock()
        pconfig.enabled = True
        mock_cfg = MagicMock()
        mock_cfg.platforms = {Platform.TELEGRAM: pconfig}
        mock_cfg.filter_silence_narration = False

        loop = MagicMock()
        loop.is_running.return_value = True

        job = {
            "id": "forum-topic-job",
            "deliver": "telegram:226252250:7072",  # private chat + numeric forum topic
        }

        def fake_run_coro(coro, _loop):
            import asyncio as _asyncio
            future = Future()
            try:
                future.set_result(_asyncio.run(coro))
            except BaseException as _e:  # noqa: BLE001
                future.set_exception(_e)
            return future

        with patch("gateway.config.load_gateway_config", return_value=mock_cfg), \
             patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
             patch("asyncio.run_coroutine_threadsafe", side_effect=fake_run_coro):
            result = _deliver_result(
                job,
                "Hello world",
                adapters={Platform.TELEGRAM: adapter},
                loop=loop,
            )

        assert result is None, f"expected clean delivery, got: {result!r}"
        adapter.send.assert_called_once()
        sent_chat_id, sent_text = adapter.send.call_args[0][0], adapter.send.call_args[0][1]
        sent_metadata = adapter.send.call_args[1]["metadata"]
        assert sent_chat_id == "226252250"
        assert sent_text == "Hello world"
        # Forum topics route via message_thread_id (thread_id in metadata), NOT
        # direct_messages_topic_id.
        assert not sent_metadata.get("direct_messages_topic_id")
        assert str(sent_metadata.get("thread_id")) == "7072"

    def test_live_adapter_ambiguous_topic_probe_failure_falls_back_to_message_thread_id(self):
        """Fail SAFE: when the ``get_chat_info`` probe cannot resolve the chat
        type (adapter with no usable probe / raising probe), an ambiguous
        private-chat topic target defaults to ``message_thread_id`` — the common
        forum-topic case and pre-#22773 behaviour, never the DM-topic route.
        """
        from gateway.config import Platform
        from gateway.platforms.base import SendResult
        from concurrent.futures import Future

        send_result = SendResult(success=True, message_id="42")

        # Plain MagicMock: its auto-created get_chat_info returns a MagicMock,
        # not an awaitable, so the scheduled coroutine raises and the probe
        # fails closed to message_thread_id.
        adapter = MagicMock()
        adapter.send = AsyncMock(return_value=send_result)

        pconfig = MagicMock()
        pconfig.enabled = True
        mock_cfg = MagicMock()
        mock_cfg.platforms = {Platform.TELEGRAM: pconfig}
        mock_cfg.filter_silence_narration = False

        loop = MagicMock()
        loop.is_running.return_value = True

        job = {
            "id": "probe-fail-job",
            "deliver": "telegram:226252250:7072",
        }

        def fake_run_coro(coro, _loop):
            import asyncio as _asyncio
            future = Future()
            try:
                future.set_result(_asyncio.run(coro))
            except BaseException as _e:  # noqa: BLE001
                future.set_exception(_e)
            return future

        with patch("gateway.config.load_gateway_config", return_value=mock_cfg), \
             patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
             patch("asyncio.run_coroutine_threadsafe", side_effect=fake_run_coro):
            result = _deliver_result(
                job,
                "Hello world",
                adapters={Platform.TELEGRAM: adapter},
                loop=loop,
            )

        assert result is None, f"expected clean delivery, got: {result!r}"
        adapter.send.assert_called_once()
        sent_metadata = adapter.send.call_args[1]["metadata"]
        assert not sent_metadata.get("direct_messages_topic_id")
        assert str(sent_metadata.get("thread_id")) == "7072"

    def test_live_adapter_probe_returns_none_falls_back_to_message_thread_id(self):
        """Fail SAFE when the probe yields a non-dict result. A relay/proxy
        adapter (or a future ``get_chat_info`` variant) may return ``None``
        rather than a dict; the ``isinstance(info, dict)`` guard must still route
        via ``message_thread_id``, distinct from the raising-probe path.

        (The real Telegram adapter returns a dict on every path — a
        ``type="dm"`` dict with an ``error`` key on failure, covered separately
        by ``..._adapter_error_dict_falls_back...`` — never ``None``. This test
        locks the non-dict defensive branch for other adapters.)"""
        from gateway.config import Platform
        from gateway.platforms.base import SendResult
        from concurrent.futures import Future

        send_result = SendResult(success=True, message_id="42")

        class _NoneProbeAdapter(MagicMock):
            async def get_chat_info(self, chat_id):
                return None

        adapter = _NoneProbeAdapter()
        adapter.send = AsyncMock(return_value=send_result)

        pconfig = MagicMock()
        pconfig.enabled = True
        mock_cfg = MagicMock()
        mock_cfg.platforms = {Platform.TELEGRAM: pconfig}
        mock_cfg.filter_silence_narration = False

        loop = MagicMock()
        loop.is_running.return_value = True

        job = {
            "id": "none-probe-job",
            "deliver": "telegram:226252250:7072",
        }

        def fake_run_coro(coro, _loop):
            import asyncio as _asyncio
            future = Future()
            try:
                future.set_result(_asyncio.run(coro))
            except BaseException as _e:  # noqa: BLE001
                future.set_exception(_e)
            return future

        with patch("gateway.config.load_gateway_config", return_value=mock_cfg), \
             patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
             patch("asyncio.run_coroutine_threadsafe", side_effect=fake_run_coro):
            result = _deliver_result(
                job,
                "Hello world",
                adapters={Platform.TELEGRAM: adapter},
                loop=loop,
            )

        assert result is None, f"expected clean delivery, got: {result!r}"
        adapter.send.assert_called_once()
        sent_metadata = adapter.send.call_args[1]["metadata"]
        assert not sent_metadata.get("direct_messages_topic_id")
        assert str(sent_metadata.get("thread_id")) == "7072"

    def test_live_adapter_error_dict_falls_back_to_message_thread_id(self):
        """Fail SAFE on the REAL Telegram adapter error contract: on a failed
        ``get_chat.get_chat`` the adapter returns ``{"type": "dm", "error": ...}``
        (plugins/platforms/telegram/adapter.py::get_chat_info), NOT ``None`` and
        NOT a raise. A ``type="dm"`` (or bot-missing ``{"type": "dm"}``) result
        must route via ``message_thread_id`` — only a genuine ``type="channel"``
        gets ``direct_messages_topic_id``. This locks the exact dict shape
        production emits so a forum-topic cron never mis-routes to General."""
        from gateway.config import Platform
        from gateway.platforms.base import SendResult
        from concurrent.futures import Future

        send_result = SendResult(success=True, message_id="42")

        class _ErrorDictAdapter(MagicMock):
            async def get_chat_info(self, chat_id):
                # Mirrors the real adapter's except-branch return shape.
                return {"name": str(chat_id), "type": "dm", "error": "Chat not found"}

        adapter = _ErrorDictAdapter()
        adapter.send = AsyncMock(return_value=send_result)

        pconfig = MagicMock()
        pconfig.enabled = True
        mock_cfg = MagicMock()
        mock_cfg.platforms = {Platform.TELEGRAM: pconfig}
        mock_cfg.filter_silence_narration = False

        loop = MagicMock()
        loop.is_running.return_value = True

        job = {
            "id": "error-dict-job",
            "deliver": "telegram:226252250:7072",
        }

        def fake_run_coro(coro, _loop):
            import asyncio as _asyncio
            future = Future()
            try:
                future.set_result(_asyncio.run(coro))
            except BaseException as _e:  # noqa: BLE001
                future.set_exception(_e)
            return future

        with patch("gateway.config.load_gateway_config", return_value=mock_cfg), \
             patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
             patch("asyncio.run_coroutine_threadsafe", side_effect=fake_run_coro):
            result = _deliver_result(
                job,
                "Hello world",
                adapters={Platform.TELEGRAM: adapter},
                loop=loop,
            )

        assert result is None, f"expected clean delivery, got: {result!r}"
        adapter.send.assert_called_once()
        sent_metadata = adapter.send.call_args[1]["metadata"]
        assert not sent_metadata.get("direct_messages_topic_id")
        assert str(sent_metadata.get("thread_id")) == "7072"

    def test_live_adapter_channel_dm_topic_routes_via_direct_messages_topic_id(self):
        """#22773 (done right): a genuine Bot API 10.0 *channel* Direct-Messages
        topic must be routed via ``direct_messages_topic_id`` (a bare
        ``message_thread_id`` is rejected / mis-routed there).  We recognise it
        from the real runtime signal — ``get_chat_info`` reports the chat as a
        ``channel`` — not from a positive-chat-id + numeric-thread guess.
        """
        from gateway.config import Platform
        from gateway.platforms.base import SendResult
        from concurrent.futures import Future

        send_result = SendResult(success=True, message_id="42")

        class _ChannelAdapter(MagicMock):
            async def get_chat_info(self, chat_id):
                return {"name": "My Channel", "type": "channel", "is_forum": False}

        adapter = _ChannelAdapter()
        adapter.send = AsyncMock(return_value=send_result)

        pconfig = MagicMock()
        pconfig.enabled = True
        mock_cfg = MagicMock()
        mock_cfg.platforms = {Platform.TELEGRAM: pconfig}
        mock_cfg.filter_silence_narration = False

        loop = MagicMock()
        loop.is_running.return_value = True

        job = {
            "id": "channel-dm-topic-job",
            "deliver": "telegram:226252250:7072",
        }

        def fake_run_coro(coro, _loop):
            import asyncio as _asyncio
            future = Future()
            try:
                future.set_result(_asyncio.run(coro))
            except BaseException as _e:  # noqa: BLE001
                future.set_exception(_e)
            return future

        with patch("gateway.config.load_gateway_config", return_value=mock_cfg), \
             patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
             patch("asyncio.run_coroutine_threadsafe", side_effect=fake_run_coro):
            result = _deliver_result(
                job,
                "Hello world",
                adapters={Platform.TELEGRAM: adapter},
                loop=loop,
            )

        assert result is None, f"expected clean delivery, got: {result!r}"
        adapter.send.assert_called_once()
        sent_metadata = adapter.send.call_args[1]["metadata"]
        # Genuine channel DM topic routes via direct_messages_topic_id, no bare
        # message_thread_id.
        assert str(sent_metadata.get("direct_messages_topic_id")) == "7072"
        assert not sent_metadata.get("message_thread_id")

    def test_live_adapter_forum_topic_media_routes_via_message_thread_id(self, tmp_path, monkeypatch):
        """#52060 (media): MEDIA attachments to a forum-style topic in a private
        chat must also route via ``thread_id`` (message_thread_id), not
        ``direct_messages_topic_id``."""
        from gateway.config import Platform
        from gateway.platforms.base import SendResult
        from concurrent.futures import Future

        media_root = tmp_path / "media-cache"
        media_file = media_root / "chart.png"
        media_file.parent.mkdir(parents=True, exist_ok=True)
        media_file.write_bytes(b"media")
        monkeypatch.setattr(
            "gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS",
            (media_root,),
        )
        media_path = media_file.resolve()

        probe_calls = {"n": 0}

        class _ForumAdapter(AsyncMock):
            async def get_chat_info(self, chat_id):
                probe_calls["n"] += 1
                return {"name": "Proyectos", "type": "forum", "is_forum": True}

        adapter = _ForumAdapter()
        adapter.send.return_value = SendResult(success=True, message_id="1")
        adapter.send_image_file.return_value = SendResult(success=True, message_id="2")

        pconfig = MagicMock()
        pconfig.enabled = True
        mock_cfg = MagicMock()
        mock_cfg.platforms = {Platform.TELEGRAM: pconfig}
        mock_cfg.filter_silence_narration = False

        loop = MagicMock()
        loop.is_running.return_value = True

        job = {
            "id": "forum-topic-media-job",
            "deliver": "telegram:226252250:7072",  # private chat + numeric forum topic
        }

        def fake_run_coro(coro, _loop):
            import asyncio as _asyncio
            future = Future()
            try:
                future.set_result(_asyncio.run(coro))
            except BaseException as _e:  # noqa: BLE001
                future.set_exception(_e)
            return future

        with patch("gateway.config.load_gateway_config", return_value=mock_cfg), \
             patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
             patch("asyncio.run_coroutine_threadsafe", side_effect=fake_run_coro):
            _deliver_result(
                job,
                f"Chart attached\nMEDIA:{media_path}",
                adapters={Platform.TELEGRAM: adapter},
                loop=loop,
            )

        adapter.send_image_file.assert_called_once()
        media_metadata = adapter.send_image_file.call_args[1]["metadata"]
        assert str(media_metadata.get("thread_id")) == "7072"
        assert not media_metadata.get("direct_messages_topic_id")
        # Probe exactly once and reuse the result for BOTH the text and media
        # sends — never re-probe per send (the "compute ONCE" contract).
        assert probe_calls["n"] == 1

    def test_live_adapter_channel_dm_topic_media_routes_via_direct_messages_topic_id(self, tmp_path, monkeypatch):
        """#22773 (media, done right): MEDIA attachments to a genuine channel DM
        topic must route via ``direct_messages_topic_id``."""
        from gateway.config import Platform
        from gateway.platforms.base import SendResult
        from concurrent.futures import Future

        media_root = tmp_path / "media-cache"
        media_file = media_root / "chart.png"
        media_file.parent.mkdir(parents=True, exist_ok=True)
        media_file.write_bytes(b"media")
        monkeypatch.setattr(
            "gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS",
            (media_root,),
        )
        media_path = media_file.resolve()

        class _ChannelAdapter(AsyncMock):
            async def get_chat_info(self, chat_id):
                return {"name": "My Channel", "type": "channel", "is_forum": False}

        adapter = _ChannelAdapter()
        adapter.send.return_value = SendResult(success=True, message_id="1")
        adapter.send_image_file.return_value = SendResult(success=True, message_id="2")

        pconfig = MagicMock()
        pconfig.enabled = True
        mock_cfg = MagicMock()
        mock_cfg.platforms = {Platform.TELEGRAM: pconfig}
        mock_cfg.filter_silence_narration = False

        loop = MagicMock()
        loop.is_running.return_value = True

        job = {
            "id": "channel-dm-topic-media-job",
            "deliver": "telegram:226252250:7072",
        }

        def fake_run_coro(coro, _loop):
            import asyncio as _asyncio
            future = Future()
            try:
                future.set_result(_asyncio.run(coro))
            except BaseException as _e:  # noqa: BLE001
                future.set_exception(_e)
            return future

        with patch("gateway.config.load_gateway_config", return_value=mock_cfg), \
             patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
             patch("asyncio.run_coroutine_threadsafe", side_effect=fake_run_coro):
            _deliver_result(
                job,
                f"Chart attached\nMEDIA:{media_path}",
                adapters={Platform.TELEGRAM: adapter},
                loop=loop,
            )

        adapter.send_image_file.assert_called_once()
        media_metadata = adapter.send_image_file.call_args[1]["metadata"]
        assert str(media_metadata.get("direct_messages_topic_id")) == "7072"
        assert not media_metadata.get("message_thread_id")
        assert not media_metadata.get("thread_id")

    def test_live_adapter_forum_thread_fallback_records_delivery_error(self):
        """A forum/supergroup cron target whose configured topic is gone must
        NOT be reported as a clean delivery: when the Telegram adapter falls
        back to the base chat (raw_response thread_fallback), the scheduler must
        record the "delivered without thread_id" delivery error.  Regression
        coverage for the thread_fallback-recording branch (kept distinct from
        the #22773 routing fix)."""
        from gateway.config import Platform
        from gateway.platforms.base import SendResult
        from concurrent.futures import Future

        send_result = SendResult(
            success=True,
            message_id="42",
            raw_response={
                "requested_thread_id": 17,
                "thread_fallback": True,
            },
        )
        adapter = MagicMock()
        adapter.send = AsyncMock(return_value=send_result)

        pconfig = MagicMock()
        pconfig.enabled = True
        mock_cfg = MagicMock()
        mock_cfg.platforms = {Platform.TELEGRAM: pconfig}
        mock_cfg.filter_silence_narration = False

        loop = MagicMock()
        loop.is_running.return_value = True

        # Forum supergroup (negative chat_id) + numeric topic → mode 1
        # (message_thread_id); NOT a private DM topic.
        job = {
            "id": "forum-fallback-job",
            "deliver": "telegram:-1001234567890:17",
        }

        def fake_run_coro(coro, _loop):
            import asyncio as _asyncio
            future = Future()
            try:
                future.set_result(_asyncio.run(coro))
            except BaseException as _e:  # noqa: BLE001
                future.set_exception(_e)
            return future

        with patch("gateway.config.load_gateway_config", return_value=mock_cfg), \
             patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
             patch("asyncio.run_coroutine_threadsafe", side_effect=fake_run_coro):
            result = _deliver_result(
                job,
                "Hello world",
                adapters={Platform.TELEGRAM: adapter},
                loop=loop,
            )

        assert result is not None
        assert "was not found; delivered without thread_id" in result
        # Forum target routes via message_thread_id (mode 1), not DM-topic.
        sent_metadata = adapter.send.call_args[1]["metadata"]
        assert not sent_metadata.get("direct_messages_topic_id")


class TestDeliverResultLiveAdapterUnconfirmed:
    """Regression for #47056.

    When a live adapter's send() returns ``None`` (swallowed exception / busy
    platform) or a result object that lacks an explicit ``success`` attribute
    (bare dict / partial object), the scheduler must NOT log "delivered via
    live adapter" and silently drop the message.  Every unconfirmed shape must
    fall through to the standalone delivery path so the message actually
    arrives.  The pre-fix check ``send_result is None or not getattr(...,
    "success", True)`` let a ``.success``-less object default to True = silent
    success.
    """

    def _run(self, send_value):
        from gateway.config import Platform
        from concurrent.futures import Future

        adapter = AsyncMock()
        adapter.send.return_value = send_value

        pconfig = MagicMock()
        pconfig.enabled = True
        mock_cfg = MagicMock()
        mock_cfg.platforms = {Platform.TELEGRAM: pconfig}

        loop = MagicMock()
        loop.is_running.return_value = True

        completed_future = Future()
        completed_future.set_result(send_value)

        def fake_run_coro(coro, _loop):
            coro.close()
            return completed_future

        job = {
            "id": "unconfirmed-job",
            "deliver": "origin",
            "origin": {"platform": "telegram", "chat_id": "123"},
        }

        standalone_send = AsyncMock(return_value={"success": True})

        with patch("gateway.config.load_gateway_config", return_value=mock_cfg), \
             patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
             patch("asyncio.run_coroutine_threadsafe", side_effect=fake_run_coro), \
             patch("tools.send_message_tool._send_to_platform", new=standalone_send):
            result = _deliver_result(
                job,
                "Hello world",
                adapters={Platform.TELEGRAM: adapter},
                loop=loop,
            )
        return result, standalone_send

    def test_none_result_falls_through_to_standalone(self):
        """send() returning None must trigger the standalone fallback, not a
        silent "delivered" log."""
        result, standalone_send = self._run(None)
        assert result is None, f"standalone should have delivered, got: {result!r}"
        standalone_send.assert_awaited_once()

    def test_result_missing_success_attr_falls_through(self):
        """A result object with no ``success`` attribute is a contract
        violation and must NOT be counted as delivered (it defaulted to True
        before the fix)."""
        class _NoSuccess:
            pass

        result, standalone_send = self._run(_NoSuccess())
        assert result is None, f"standalone should have delivered, got: {result!r}"
        standalone_send.assert_awaited_once()

    def test_confirmed_success_does_not_fall_through(self):
        """A genuine SendResult(success=True) is confirmed — the standalone
        path must NOT run (no duplicate)."""
        result, standalone_send = self._run(MagicMock(success=True, raw_response=None))
        assert result is None
        standalone_send.assert_not_awaited()


class TestDeliverOriginUnresolvableIsLocal:
    """Regression for #43014.

    A cron job created in a CLI session has no {platform, chat_id} origin.
    With ``deliver=origin`` (or auto-detect / deliver=None) and no configured
    platform home channel, delivery is unresolvable — but that is the EXPECTED
    state for CLI jobs, not an error.  _deliver_result must return None (treat
    as local; output stays in last_output), not the "no delivery target
    resolved" error string that previously fired on every run.
    """

    def _deliver(self, job, monkeypatch):
        import cron.scheduler as sched
        # No home channel for any platform → origin is unresolvable.
        monkeypatch.setattr(sched, "_get_home_target_chat_id", lambda *_: "")
        return _deliver_result(job, "CLI bulletin")

    def test_origin_with_no_home_channels_returns_none(self, monkeypatch):
        job = {"id": "cli-job", "deliver": "origin", "origin": "cli-session-provenance"}
        assert self._deliver(job, monkeypatch) is None

    def test_omitted_deliver_autodetect_returns_none(self, monkeypatch):
        # deliver key present but None (auto-detect) previously errored with
        # "no delivery target resolved for deliver=None".
        job = {"id": "cli-job", "deliver": None, "origin": "cli-session-provenance"}
        assert self._deliver(job, monkeypatch) is None

    def test_explicit_platform_with_no_channel_still_errors(self, monkeypatch):
        # A concrete platform target that cannot resolve is still a real error
        # (this must NOT be silently swallowed by the origin→local fallback).
        job = {"id": "tg-job", "deliver": "telegram"}
        result = self._deliver(job, monkeypatch)
        assert result is not None
        assert "no delivery target resolved" in result


class TestSendMediaTimeoutCancelsFuture:
    """Same orphan-coroutine guarantee for _send_media_via_adapter's
    future.result(timeout=30) call. If this times out mid-batch, the
    in-flight coroutine must be cancelled before the next file is tried.
    """

    def test_media_send_timeout_cancels_future_and_continues(self, tmp_path, monkeypatch):
        """End-to-end: _send_media_via_adapter with a future whose .result()
        raises TimeoutError. Assert cancel() fires and the loop proceeds
        to the next file rather than hanging or crashing."""
        from concurrent.futures import Future

        adapter = MagicMock()
        adapter.send_image_file = AsyncMock()
        adapter.send_video = AsyncMock()

        # First file: future that times out. Second file: future that resolves OK.
        timeout_future = Future()
        timeout_cancel_calls = []
        original_cancel = timeout_future.cancel

        def tracking_cancel():
            timeout_cancel_calls.append(True)
            return original_cancel()

        timeout_future.cancel = tracking_cancel
        timeout_future.result = MagicMock(side_effect=TimeoutError("timed out"))

        ok_future = Future()
        ok_future.set_result(MagicMock(success=True))

        futures_iter = iter([timeout_future, ok_future])

        def fake_run_coro(coro, _loop):
            coro.close()
            return next(futures_iter)

        root = tmp_path / "media-cache"
        slow = root / "slow.png"
        fast = root / "fast.mp4"
        slow.parent.mkdir(parents=True)
        slow.write_bytes(b"slow")
        fast.write_bytes(b"fast")
        monkeypatch.setattr(
            "gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS",
            (root,),
        )
        media_files = [
            (str(slow), False),   # times out
            (str(fast), False),   # succeeds
        ]

        loop = MagicMock()
        job = {"id": "media-timeout"}

        with patch("asyncio.run_coroutine_threadsafe", side_effect=fake_run_coro):
            # Should not raise — the except Exception clause swallows the timeout
            _send_media_via_adapter(adapter, "chat-1", media_files, None, loop, job)

        # 1. The timed-out future was cancelled (the bug fix)
        assert timeout_cancel_calls == [True], "future.cancel() must fire on TimeoutError"
        # 2. Second file still got dispatched — one timeout doesn't abort the batch
        adapter.send_video.assert_called_once()
        assert adapter.send_video.call_args[1]["video_path"] == str(fast.resolve())


class TestCronDeliveryTargets:
    """``cron_delivery_targets`` powers the dashboard delivery dropdown.

    It must list every configured + cron-deliverable platform (no hardcoded
    set), flag whether each has its home channel set, and never include
    platforms whose gateway isn't configured.
    """

    def _patch_connected(self, monkeypatch, names):
        import gateway.config as gateway_config

        class _Platform:
            def __init__(self, value):
                self.value = value

        class _GatewayConfig:
            def get_connected_platforms(self_inner):
                return [_Platform(n) for n in names]

        monkeypatch.setattr(
            gateway_config, "load_gateway_config", lambda: _GatewayConfig()
        )

    def test_lists_configured_platforms_flagging_missing_home_channel(self, monkeypatch):
        from cron.scheduler import cron_delivery_targets

        self._patch_connected(monkeypatch, ["matrix", "telegram"])
        monkeypatch.delenv("MATRIX_HOME_ROOM", raising=False)
        monkeypatch.delenv("TELEGRAM_HOME_CHANNEL", raising=False)

        targets = {t["id"]: t for t in cron_delivery_targets()}

        assert set(targets) == {"matrix", "telegram"}
        # Configured but no home channel → surfaced, flagged for the UI.
        assert targets["matrix"]["home_target_set"] is False
        assert targets["matrix"]["home_env_var"] == "MATRIX_HOME_ROOM"
        assert targets["telegram"]["home_target_set"] is False

    def test_home_channel_set_marks_target_ready(self, monkeypatch):
        from cron.scheduler import cron_delivery_targets

        self._patch_connected(monkeypatch, ["matrix"])
        monkeypatch.setenv("MATRIX_HOME_ROOM", "!room:matrix.org")

        targets = {t["id"]: t for t in cron_delivery_targets()}

        assert targets["matrix"]["home_target_set"] is True

    def test_unconfigured_platforms_excluded(self, monkeypatch):
        from cron.scheduler import cron_delivery_targets

        # Only telegram is connected; matrix env var set but gateway not configured.
        self._patch_connected(monkeypatch, ["telegram"])
        monkeypatch.setenv("MATRIX_HOME_ROOM", "!room:matrix.org")

        ids = {t["id"] for t in cron_delivery_targets()}

        assert ids == {"telegram"}
        assert "matrix" not in ids

    def test_no_gateway_config_returns_empty(self, monkeypatch):
        import gateway.config as gateway_config
        from cron.scheduler import cron_delivery_targets

        def _boom():
            raise RuntimeError("no gateway config")

        monkeypatch.setattr(gateway_config, "load_gateway_config", _boom)

        assert cron_delivery_targets() == []


class TestHomeTargetEnvVarRegistry:
    """Regression: ``_HOME_TARGET_ENV_VARS`` must include every gateway
    platform that supports cron-driven outbound delivery. Missing an
    entry means ``hermes cron create --deliver=<platform>`` silently
    fails to route through the platform's home channel."""

    def test_whatsapp_cloud_registered(self):
        """``deliver=whatsapp_cloud`` routes through
        WHATSAPP_CLOUD_HOME_CHANNEL — added alongside the existing
        ``whatsapp`` Baileys entry."""
        from cron.scheduler import _HOME_TARGET_ENV_VARS

        assert "whatsapp_cloud" in _HOME_TARGET_ENV_VARS
        assert _HOME_TARGET_ENV_VARS["whatsapp_cloud"] == "WHATSAPP_CLOUD_HOME_CHANNEL"

    def test_baileys_whatsapp_still_registered(self):
        """Sanity guard: the Cloud addition didn't disturb Baileys
        whatsapp routing."""
        from cron.scheduler import _HOME_TARGET_ENV_VARS

        assert _HOME_TARGET_ENV_VARS.get("whatsapp") == "WHATSAPP_HOME_CHANNEL"


class TestCronDeliveryMirror:
    """cron.mirror_delivery / per-job attach_to_session: opt-in append of a
    cron delivery into the target chat's gateway session transcript.

    Default OFF preserves the historical isolation guarantee byte-for-byte.
    When enabled, delivery rides the existing gateway.mirror.mirror_to_session
    so cron uses exactly the same path interactive send_message mirroring uses.
    """

    def test_gate_default_off(self):
        from cron.scheduler import _cron_mirror_delivery_enabled

        # No per-job flag, no config -> off (historical behaviour).
        assert _cron_mirror_delivery_enabled({}, {}) is False
        assert _cron_mirror_delivery_enabled({"id": "x"}, {"cron": {}}) is False

    def test_gate_global_config_on(self):
        from cron.scheduler import _cron_mirror_delivery_enabled

        assert _cron_mirror_delivery_enabled({}, {"cron": {"mirror_delivery": True}}) is True

    def test_gate_per_job_overrides_global(self):
        from cron.scheduler import _cron_mirror_delivery_enabled

        # Per-job False wins even if global is on.
        assert _cron_mirror_delivery_enabled(
            {"attach_to_session": False}, {"cron": {"mirror_delivery": True}}
        ) is False
        # Per-job True wins even if global is off/absent.
        assert _cron_mirror_delivery_enabled(
            {"attach_to_session": True}, {"cron": {"mirror_delivery": False}}
        ) is True

    def test_mirror_calls_mirror_to_session_when_enabled(self):
        from cron.scheduler import _maybe_mirror_cron_delivery

        with patch("gateway.mirror.mirror_to_session", return_value=True) as m:
            _maybe_mirror_cron_delivery(
                {"id": "j1", "name": "Daily Brief"}, "telegram", "123",
                "Daily brief Task #2", thread_id=None, enabled=True,
            )
        m.assert_called_once()
        args, kwargs = m.call_args
        assert args[0] == "telegram"
        assert args[1] == "123"
        assert "Task #2" in args[2]
        assert kwargs.get("source_label") == "cron"

    def test_mirror_writes_user_role_with_label_not_assistant(self):
        """Regression for #2221 / #2313: the cron brief must mirror as a USER
        turn (with a [Cron delivery: ...] label), NOT assistant — an
        assistant-role mirror lands as assistant->assistant after the agent's
        last turn and breaks strict alternation on non-Anthropic providers."""
        from cron.scheduler import _maybe_mirror_cron_delivery

        with patch("gateway.mirror.mirror_to_session", return_value=True) as m:
            _maybe_mirror_cron_delivery(
                {"id": "j1", "name": "Morning Brief"}, "telegram", "123",
                "Market movers today", thread_id=None, enabled=True,
            )
        m.assert_called_once()
        args, kwargs = m.call_args
        assert kwargs.get("role") == "user", "cron mirror must be a user turn, not assistant"
        # The brief text is prefixed with a human-readable cron-delivery label
        # so replay (where the mirror metadata is dropped at the SQLite
        # boundary) still distinguishes it from a genuine user message.
        assert args[2].startswith("[Cron delivery: Morning Brief]")
        assert "Market movers today" in args[2]

    def test_mirror_noop_when_disabled(self):
        from cron.scheduler import _maybe_mirror_cron_delivery

        with patch("gateway.mirror.mirror_to_session", return_value=True) as m:
            _maybe_mirror_cron_delivery(
                {"id": "j1"}, "telegram", "123", "should not mirror",
                enabled=False,
            )
        m.assert_not_called()

    def test_mirror_noop_on_empty_text(self):
        from cron.scheduler import _maybe_mirror_cron_delivery

        with patch("gateway.mirror.mirror_to_session", return_value=True) as m:
            _maybe_mirror_cron_delivery({"id": "j1"}, "telegram", "123", "   ", enabled=True)
        m.assert_not_called()

    def test_mirror_swallows_cold_start_miss(self):
        """A missing target session (cold start) must NOT raise — delivery
        already succeeded; the mirror is best-effort."""
        from cron.scheduler import _maybe_mirror_cron_delivery

        with patch("gateway.mirror.mirror_to_session", return_value=False) as m:
            # Should not raise.
            _maybe_mirror_cron_delivery(
                {"id": "j1"}, "telegram", "123", "brief", enabled=True
            )
        m.assert_called_once()

    def test_mirror_swallows_exceptions(self):
        from cron.scheduler import _maybe_mirror_cron_delivery

        with patch("gateway.mirror.mirror_to_session", side_effect=RuntimeError("boom")):
            # Must not propagate — a delivery that succeeded is never failed by
            # a mirror error.
            _maybe_mirror_cron_delivery(
                {"id": "j1"}, "telegram", "123", "brief", enabled=True
            )

    def test_delivery_mirrors_clean_content_not_wrapped(self):
        """When enabled, the mirror receives the CLEAN agent output, not the
        cron header/footer-wrapped delivery text."""
        from gateway.config import Platform

        pconfig = MagicMock()
        pconfig.enabled = True
        mock_cfg = MagicMock()
        mock_cfg.platforms = {Platform.TELEGRAM: pconfig}

        with patch("gateway.config.load_gateway_config", return_value=mock_cfg), \
             patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})), \
             patch("gateway.mirror.mirror_to_session", return_value=True) as mirror_mock:
            job = {
                "id": "test-job",
                "name": "daily-report",
                "deliver": "origin",
                "origin": {"platform": "telegram", "chat_id": "123"},
                "attach_to_session": True,
            }
            _deliver_result(job, "Here is today's summary.")

        mirror_mock.assert_called_once()
        mirrored_text = mirror_mock.call_args[0][2]
        # Clean content, no cron wrapper.
        assert "Here is today's summary." in mirrored_text
        assert "Cronjob Response:" not in mirrored_text
        assert "To stop or manage this job" not in mirrored_text

    def test_delivery_does_not_mirror_when_gate_off(self):
        """Default path: a job with no opt-in must never touch the mirror."""
        from gateway.config import Platform

        pconfig = MagicMock()
        pconfig.enabled = True
        mock_cfg = MagicMock()
        mock_cfg.platforms = {Platform.TELEGRAM: pconfig}

        with patch("gateway.config.load_gateway_config", return_value=mock_cfg), \
             patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})), \
             patch("gateway.mirror.mirror_to_session", return_value=True) as mirror_mock:
            job = {
                "id": "test-job",
                "name": "daily-report",
                "deliver": "origin",
                "origin": {"platform": "telegram", "chat_id": "123"},
            }
            _deliver_result(job, "Here is today's summary.")

        mirror_mock.assert_not_called()

    # --- origin-scoping (mirror only into the conversation that created the job) ---

    def test_target_matches_origin_exact(self):
        from cron.scheduler import _target_matches_origin

        origin = {"platform": "telegram", "chat_id": "123"}
        assert _target_matches_origin(origin, "telegram", "123", None) is True
        # Case-insensitive platform match.
        assert _target_matches_origin(origin, "Telegram", "123", None) is True

    def test_target_matches_origin_rejects_other_chat(self):
        from cron.scheduler import _target_matches_origin

        origin = {"platform": "telegram", "chat_id": "123"}
        # Different chat (fan-out / explicit other target) -> not the origin.
        assert _target_matches_origin(origin, "telegram", "999", None) is False
        # Different platform (deliver=all broadcast) -> not the origin.
        assert _target_matches_origin(origin, "discord", "123", None) is False
        # No origin at all (API/script job, home-channel fallback) -> never.
        assert _target_matches_origin({}, "telegram", "123", None) is False

    def test_target_matches_origin_thread_scoped(self):
        from cron.scheduler import _target_matches_origin

        origin = {"platform": "telegram", "chat_id": "123", "thread_id": "17"}
        assert _target_matches_origin(origin, "telegram", "123", "17") is True
        # Same chat, wrong/lost thread lane -> not the same conversation.
        assert _target_matches_origin(origin, "telegram", "123", None) is False
        assert _target_matches_origin(origin, "telegram", "123", "99") is False

    def test_delivery_does_not_mirror_fanout_non_origin_target(self):
        """Even with the gate ON, a delivery to a chat that is NOT the job's
        origin (explicit fan-out target) must not be mirrored — the mirror is
        scoped to the origin conversation, and the fan-out chat may have no
        session at all."""
        from gateway.config import Platform

        pconfig = MagicMock()
        pconfig.enabled = True
        mock_cfg = MagicMock()
        mock_cfg.platforms = {Platform.TELEGRAM: pconfig}

        with patch("gateway.config.load_gateway_config", return_value=mock_cfg), \
             patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})), \
             patch("gateway.mirror.mirror_to_session", return_value=True) as mirror_mock:
            job = {
                "id": "test-job",
                "name": "daily-report",
                # Explicit delivery to a DIFFERENT chat than the origin.
                "deliver": "telegram:999",
                "origin": {"platform": "telegram", "chat_id": "123"},
                "attach_to_session": True,
            }
            _deliver_result(job, "Here is today's summary.")

        # Delivered to 999, but origin is 123 -> no mirror.
        mirror_mock.assert_not_called()

    def test_delivery_mirrors_only_origin_target_in_fanout(self):
        """deliver to BOTH origin and another chat: only the origin target is
        mirrored."""
        from gateway.config import Platform

        pconfig = MagicMock()
        pconfig.enabled = True
        mock_cfg = MagicMock()
        mock_cfg.platforms = {Platform.TELEGRAM: pconfig}

        with patch("gateway.config.load_gateway_config", return_value=mock_cfg), \
             patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})), \
             patch("gateway.mirror.mirror_to_session", return_value=True) as mirror_mock:
            job = {
                "id": "test-job",
                "name": "daily-report",
                # Fan out to the origin chat (123) AND another chat (999).
                "deliver": "telegram:123,telegram:999",
                "origin": {"platform": "telegram", "chat_id": "123"},
                "attach_to_session": True,
            }
            _deliver_result(job, "Here is today's summary.")

        # Exactly one mirror, and it is the origin chat (123) — not 999.
        mirror_mock.assert_called_once()
        assert mirror_mock.call_args[0][1] == "123"

    # --- multi-participant parity with send_message (user_id passthrough) ---

    def test_mirror_passes_user_id_through(self):
        """The helper forwards user_id to mirror_to_session so a per-user-
        isolated group resolves to the exact member who scheduled the job —
        parity with interactive send_message."""
        from cron.scheduler import _maybe_mirror_cron_delivery

        with patch("gateway.mirror.mirror_to_session", return_value=True) as m:
            _maybe_mirror_cron_delivery(
                {"id": "j1"}, "telegram", "123", "brief",
                thread_id=None, user_id="U999", enabled=True,
            )
        m.assert_called_once()
        assert m.call_args.kwargs.get("user_id") == "U999"

    def test_delivery_forwards_origin_user_id(self):
        """End-to-end: a job whose origin carries user_id mirrors with that
        user_id, so multi-participant resolution matches send_message."""
        from gateway.config import Platform

        pconfig = MagicMock()
        pconfig.enabled = True
        mock_cfg = MagicMock()
        mock_cfg.platforms = {Platform.TELEGRAM: pconfig}

        with patch("gateway.config.load_gateway_config", return_value=mock_cfg), \
             patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})), \
             patch("gateway.mirror.mirror_to_session", return_value=True) as mirror_mock:
            job = {
                "id": "test-job",
                "name": "daily-report",
                "deliver": "origin",
                "origin": {"platform": "telegram", "chat_id": "123", "user_id": "U42"},
                "attach_to_session": True,
            }
            _deliver_result(job, "Here is today's summary.")

        mirror_mock.assert_called_once()
        assert mirror_mock.call_args.kwargs.get("user_id") == "U42"

    # --- continuable cron: thread-preferred (Teknium's interface) ---

    def test_open_thread_returns_id_on_thread_platform(self):
        """On a thread-capable adapter, _open_continuable_cron_thread returns
        the new thread id from create_handoff_thread."""
        from cron.scheduler import _open_continuable_cron_thread

        adapter = MagicMock()
        adapter.create_handoff_thread = AsyncMock(return_value="9001")

        # safe_schedule_threadsafe hands the coro to the gateway loop and
        # returns a future. Patch it to close the coro and return a ready
        # future carrying the adapter's thread id.
        def _run_now(coro, _loop):
            coro.close()
            fut = MagicMock()
            fut.result.return_value = "9001"
            return fut

        with patch("agent.async_utils.safe_schedule_threadsafe", side_effect=_run_now):
            tid = _open_continuable_cron_thread(
                {"id": "j1", "name": "Brief"}, adapter, "123", loop=MagicMock(),
            )
        assert tid == "9001"

    def test_open_thread_returns_none_on_dm_platform(self):
        """A DM-only adapter (WhatsApp) inherits the base create_handoff_thread
        that returns None → _open_continuable_cron_thread returns None so the
        caller falls back to DM-session mirroring."""
        from cron.scheduler import _open_continuable_cron_thread

        adapter = MagicMock()
        adapter.create_handoff_thread = AsyncMock(return_value=None)

        def _run_now(coro, _loop):
            fut = MagicMock()
            fut.result.return_value = None
            coro.close()
            return fut

        with patch("agent.async_utils.safe_schedule_threadsafe", side_effect=_run_now):
            tid = _open_continuable_cron_thread(
                {"id": "j1", "name": "Brief"}, adapter, "123", loop=MagicMock(),
            )
        assert tid is None

    def test_open_thread_none_without_capability_or_loop(self):
        """No create_handoff_thread attr, or no loop → None (no crash)."""
        from cron.scheduler import _open_continuable_cron_thread

        adapter_no_cap = MagicMock(spec=[])  # no create_handoff_thread
        assert _open_continuable_cron_thread(
            {"id": "j1"}, adapter_no_cap, "123", loop=MagicMock(),
        ) is None

        adapter = MagicMock()
        adapter.create_handoff_thread = AsyncMock(return_value="9001")
        assert _open_continuable_cron_thread(
            {"id": "j1"}, adapter, "123", loop=None,
        ) is None

    def test_seed_thread_session_creates_session_and_mirrors(self):
        """Seeding a freshly-opened thread creates the thread-keyed session via
        the adapter's live store and appends the brief via mirror_to_session."""
        from cron.scheduler import _seed_cron_thread_session

        store = MagicMock()
        adapter = MagicMock()
        adapter._session_store = store

        with patch("gateway.mirror.mirror_to_session", return_value=True) as mirror_mock:
            _seed_cron_thread_session(
                {"id": "j1"}, adapter, "telegram", "123", "9001",
                "Daily brief Task #2", chat_name="Ops",
            )

        # Session row created for the thread, then brief mirrored into it.
        store.get_or_create_session.assert_called_once()
        seeded_source = store.get_or_create_session.call_args[0][0]
        assert seeded_source.chat_type == "thread"
        assert seeded_source.thread_id == "9001"
        mirror_mock.assert_called_once()
        assert mirror_mock.call_args.kwargs.get("thread_id") == "9001"

    def test_seed_thread_session_noop_on_empty_text(self):
        from cron.scheduler import _seed_cron_thread_session

        store = MagicMock()
        adapter = MagicMock()
        adapter._session_store = store
        with patch("gateway.mirror.mirror_to_session") as mirror_mock:
            _seed_cron_thread_session(
                {"id": "j1"}, adapter, "telegram", "123", "9001", "   ",
            )
        store.get_or_create_session.assert_not_called()
        mirror_mock.assert_not_called()


class TestCronContinuableSurfaceInChannel:
    """cron_continuable_surface: in_channel — deliver a continuable cron FLAT
    into a channel (no dedicated thread), so a plain channel reply continues the
    job via the shared-channel session (platform, chat_id, None).

    Design: decisions.md D1/D2/D6 + F5. The scheduler reads the per-platform key
    generically from pconfig.extra; the in_channel branch is gated on the
    adapter capability flag ``supports_inchannel_continuable`` (Slack=True,
    others fail SAFE to thread). In in_channel mode the thread-open branch is
    SKIPPED (thread_id stays None), then ``_seed_cron_channel_session`` CREATES
    the flat shared-channel session and mirrors the brief into it (the shipped
    mirror only APPENDS to an existing session, and the flat channel row is
    otherwise absent for a chat_postMessage delivery).
    """

    def _slack_cfg(self, extra):
        """A mock GatewayConfig with a Slack pconfig carrying ``extra``."""
        from gateway.config import Platform

        pconfig = MagicMock()
        pconfig.enabled = True
        pconfig.extra = extra
        mock_cfg = MagicMock()
        mock_cfg.platforms = {Platform.SLACK: pconfig}
        return mock_cfg

    def _run_inchannel_delivery(self, extra, adapter, *, mirror_ok=True, origin=None):
        """Drive _deliver_result down the live-adapter path for a Slack
        channel-origin job with the given ``extra`` config. Returns the
        _open_continuable_cron_thread mock and the mirror_to_session mock."""
        from gateway.config import Platform
        from concurrent.futures import Future

        mock_cfg = self._slack_cfg(extra)

        loop = MagicMock()
        loop.is_running.return_value = True

        def fake_run_coro(coro, _loop):
            future = Future()
            try:
                import asyncio as _asyncio
                future.set_result(_asyncio.run(coro))
            except BaseException as _e:  # noqa: BLE001
                future.set_exception(_e)
            return future

        job = {
            "id": "brief-job",
            "name": "Daily Brief",
            "deliver": "origin",
            # Channel origin: no thread_id (flat channel message scheduled it).
            # Carries the scheduling user's id — the in_channel seed must key
            # the flat channel session to THIS user (see build_session_key).
            "origin": origin or {"platform": "slack", "chat_id": "C123", "user_id": "U_HUMAN"},
            # Opt into the continuable mirror.
            "attach_to_session": True,
        }

        with patch("gateway.config.load_gateway_config", return_value=mock_cfg), \
             patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
             patch("cron.scheduler._open_continuable_cron_thread") as open_thread_mock, \
             patch("asyncio.run_coroutine_threadsafe", side_effect=fake_run_coro), \
             patch("gateway.mirror.mirror_to_session", return_value=mirror_ok) as mirror_mock:
            _deliver_result(
                job, "Here is today's brief.",
                adapters={Platform.SLACK: adapter}, loop=loop,
            )
        return open_thread_mock, mirror_mock

    def _slack_adapter(self, supports_inchannel=True, with_store=True):
        adapter = AsyncMock()
        adapter.send.return_value = MagicMock(success=True)
        # Capability flag read via getattr in the scheduler.
        adapter.supports_inchannel_continuable = supports_inchannel
        # A live session store so the in_channel seed can CREATE the flat row
        # (the real bug: without a create step the mirror no-ops on a missing
        # session and the brief is lost). Use a plain MagicMock store.
        if with_store:
            adapter._session_store = MagicMock()
        return adapter

    def test_in_channel_skips_thread_open(self):
        """G2: in_channel mode must NOT open a handoff thread."""
        adapter = self._slack_adapter(supports_inchannel=True)
        open_thread_mock, _ = self._run_inchannel_delivery(
            {"cron_continuable_surface": "in_channel"}, adapter,
        )
        open_thread_mock.assert_not_called()

    def test_in_channel_seeds_shared_channel_session_flat(self):
        """G3 (the real fix): in_channel CREATES the flat channel session row
        (thread_id=None) via the adapter's live store AND mirrors the brief into
        it. The prior implementation relied on the bare mirror, which no-ops
        when the flat row doesn't already exist — so the brief was silently lost
        (verified live). This asserts the create-then-mirror handoff."""
        adapter = self._slack_adapter(supports_inchannel=True)
        _, mirror_mock = self._run_inchannel_delivery(
            {"cron_continuable_surface": "in_channel"}, adapter,
        )
        # The flat session row must be CREATED (this is what was missing).
        adapter._session_store.get_or_create_session.assert_called_once()
        seeded = adapter._session_store.get_or_create_session.call_args[0][0]
        assert seeded.thread_id is None, "seed must be flat (thread_id=None)"
        assert seeded.chat_type == "group", "a channel (non-D) keys as group"
        assert str(seeded.chat_id) == "C123"
        assert str(seeded.user_id) == "U_HUMAN", (
            "channel session key embeds user_id — the seed MUST use the origin "
            "user's id or the inbound reply keys to a different session"
        )
        # Brief mirrored flat into that row.
        mirror_mock.assert_called_once()
        assert mirror_mock.call_args.kwargs.get("thread_id") is None
        assert mirror_mock.call_args[0][0] == "slack"
        assert mirror_mock.call_args[0][1] == "C123"
        assert "Here is today's brief." in mirror_mock.call_args[0][2]

    def test_in_channel_dm_seeds_dm_session(self):
        """1:1 DM (chat_id starts with 'D'): the flat session is created with
        chat_type='dm'. The DM session key does NOT embed user_id, so any
        user_id resolves to the same session — but chat_type must be 'dm' so the
        key prefix matches the inbound DM reply's key."""
        adapter = self._slack_adapter(supports_inchannel=True)
        _, mirror_mock = self._run_inchannel_delivery(
            {"cron_continuable_surface": "in_channel"}, adapter,
            origin={"platform": "slack", "chat_id": "D999", "user_id": "U_HUMAN"},
        )
        adapter._session_store.get_or_create_session.assert_called_once()
        seeded = adapter._session_store.get_or_create_session.call_args[0][0]
        assert seeded.chat_type == "dm", "a DM (chat_id starts with 'D') keys as dm"
        assert seeded.thread_id is None
        assert str(seeded.chat_id) == "D999"
        mirror_mock.assert_called_once()
        assert mirror_mock.call_args.kwargs.get("thread_id") is None

    def test_thread_mode_default_still_opens_thread(self):
        """G1 regression: the default (thread) mode is byte-identical — the
        thread-open branch still fires when no surface key is set."""
        adapter = self._slack_adapter(supports_inchannel=True)
        open_thread_mock, _ = self._run_inchannel_delivery({}, adapter)
        open_thread_mock.assert_called_once()

    def test_explicit_thread_value_opens_thread(self):
        """An explicit cron_continuable_surface: thread is the default path."""
        adapter = self._slack_adapter(supports_inchannel=True)
        open_thread_mock, _ = self._run_inchannel_delivery(
            {"cron_continuable_surface": "thread"}, adapter,
        )
        open_thread_mock.assert_called_once()

    def test_in_channel_on_unsupported_platform_fails_safe_to_thread(self):
        """D6 fail-safe: in_channel on an adapter WITHOUT the capability flag
        falls back to the thread path (a threaded continuation ≈ today), never
        a dropped continuation."""
        adapter = self._slack_adapter(supports_inchannel=False)
        open_thread_mock, _ = self._run_inchannel_delivery(
            {"cron_continuable_surface": "in_channel"}, adapter,
        )
        # Capability absent → treated as thread → thread-open still attempted.
        open_thread_mock.assert_called_once()

    def test_unrecognised_surface_value_coerces_to_thread(self):
        """Any non-'in_channel' value is the default thread path (fail safe)."""
        adapter = self._slack_adapter(supports_inchannel=True)
        open_thread_mock, _ = self._run_inchannel_delivery(
            {"cron_continuable_surface": "bogus"}, adapter,
        )
        open_thread_mock.assert_called_once()

    # --- _seed_cron_channel_session: the create-then-mirror unit + the
    #     KEY-MATCH invariant (seed key must equal the inbound reply's key) ---

    def test_seed_channel_session_key_matches_inbound_channel_reply(self):
        """The whole point: the flat session the seed CREATES must be keyed
        identically to what a plain inbound channel reply resolves to. Assert
        the invariant directly via build_session_key, not just call args."""
        from cron.scheduler import _seed_cron_channel_session
        from gateway.session import build_session_key, SessionSource
        from gateway.config import Platform

        store = MagicMock()
        adapter = MagicMock()
        adapter._session_store = store

        with patch("gateway.mirror.mirror_to_session", return_value=True) as mirror_mock:
            ok = _seed_cron_channel_session(
                {"id": "j1", "name": "Brief"}, adapter, "slack", "C123",
                "Daily brief", is_dm=False, user_id="U_HUMAN", chat_name="ops",
            )
        assert ok is True
        seeded_source = store.get_or_create_session.call_args[0][0]
        seed_key = build_session_key(seeded_source)

        # What a plain top-level channel reply (reply_in_thread:false → thread
        # None) from the same user resolves to:
        inbound = SessionSource(
            platform=Platform.SLACK, chat_id="C123", chat_type="group",
            user_id="U_HUMAN", thread_id=None,
        )
        assert seed_key == build_session_key(inbound), (
            f"seed key {seed_key} != inbound reply key {build_session_key(inbound)} "
            "— the reply would NOT continue the seeded session"
        )
        mirror_mock.assert_called_once()
        assert mirror_mock.call_args.kwargs.get("thread_id") is None
        assert mirror_mock.call_args.kwargs.get("user_id") == "U_HUMAN"

    def test_seed_channel_session_key_matches_inbound_dm_reply(self):
        """DM case: seeded key (chat_type=dm) equals the inbound DM reply key.
        The DM key ignores user_id, so a system id would also match — but
        chat_type MUST be 'dm' so the prefix aligns."""
        from cron.scheduler import _seed_cron_channel_session
        from gateway.session import build_session_key, SessionSource
        from gateway.config import Platform

        store = MagicMock()
        adapter = MagicMock()
        adapter._session_store = store

        with patch("gateway.mirror.mirror_to_session", return_value=True):
            _seed_cron_channel_session(
                {"id": "j1"}, adapter, "slack", "D999", "Daily brief",
                is_dm=True, user_id="U_HUMAN",
            )
        seeded_source = store.get_or_create_session.call_args[0][0]
        inbound = SessionSource(
            platform=Platform.SLACK, chat_id="D999", chat_type="dm",
            user_id="U_HUMAN", thread_id=None,
        )
        assert build_session_key(seeded_source) == build_session_key(inbound)
        assert seeded_source.chat_type == "dm"

    def test_seed_channel_session_noop_on_empty_text(self):
        from cron.scheduler import _seed_cron_channel_session

        store = MagicMock()
        adapter = MagicMock()
        adapter._session_store = store
        with patch("gateway.mirror.mirror_to_session") as mirror_mock:
            ok = _seed_cron_channel_session(
                {"id": "j1"}, adapter, "slack", "C123", "   ",
                is_dm=False, user_id="U_HUMAN",
            )
        assert ok is False
        store.get_or_create_session.assert_not_called()
        mirror_mock.assert_not_called()


class TestMultiTargetDeliveryContinuesOnFailure:
    """When delivery to one target fails inside the standalone thread-pool
    fallback, the loop must continue to the remaining targets (#47163).

    The fallback runs inside the `except RuntimeError` block of
    `_deliver_result`. Before the fix, an exception raised there (SMTP
    ConnectionError, future.result timeout) escaped the function entirely —
    it is NOT caught by the sibling `except Exception` — crashing the loop
    and silently dropping every subsequent target.
    """

    def _email_cfg(self):
        from gateway.config import Platform

        pconfig = MagicMock()
        pconfig.enabled = True
        mock_cfg = MagicMock()
        mock_cfg.platforms = {Platform.EMAIL: pconfig}
        return mock_cfg

    def test_first_target_failure_does_not_crash_loop(self):
        """First email target fails in the fallback; the second is still attempted."""
        job = {
            "id": "multi-email-job",
            "deliver": "email:a@example.com,email:b@example.com",
        }

        with patch("gateway.config.load_gateway_config", return_value=self._email_cfg()), \
             patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
             patch("asyncio.run", side_effect=RuntimeError("no running loop")), \
             patch("concurrent.futures.ThreadPoolExecutor") as mock_pool_cls:
            mock_pool = MagicMock()
            mock_pool_cls.return_value = mock_pool

            fail_future = MagicMock()
            fail_future.result.side_effect = ConnectionError("SMTP connection refused")
            ok_future = MagicMock()
            ok_future.result.return_value = {"success": True}
            mock_pool.submit.side_effect = [fail_future, ok_future]

            result = _deliver_result(job, "Report content")

        # Both targets attempted — the loop did not crash after the first failure.
        assert mock_pool.submit.call_count == 2, (
            f"expected 2 delivery attempts, got {mock_pool.submit.call_count}"
        )
        # First target's failure is surfaced in the returned error string.
        assert result is not None
        assert "a@example.com" in result
        assert "SMTP connection refused" in result

    def test_all_targets_fail_returns_combined_errors(self):
        """When every target fails, the result reports all of them."""
        job = {
            "id": "all-fail-job",
            "deliver": "email:a@example.com,email:b@example.com",
        }

        with patch("gateway.config.load_gateway_config", return_value=self._email_cfg()), \
             patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
             patch("asyncio.run", side_effect=RuntimeError("no running loop")), \
             patch("concurrent.futures.ThreadPoolExecutor") as mock_pool_cls:
            mock_pool = MagicMock()
            mock_pool_cls.return_value = mock_pool

            fail_future = MagicMock()
            fail_future.result.side_effect = ConnectionError("connection refused")
            mock_pool.submit.return_value = fail_future

            result = _deliver_result(job, "Report content")

        assert result is not None
        assert "a@example.com" in result
        assert "b@example.com" in result
        assert mock_pool.submit.call_count == 2
