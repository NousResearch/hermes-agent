"""Tests for the Slack adapter's channel-join auto-onboarding and the
mtime-based hot reload of ``channel_skill_bindings``.

Both features are targeted, defensive additions to the Slack adapter and
are exercised without a live Slack connection by using the mock-modules
harness in ``tests/gateway/test_slack.py``.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# Reuse the shared slack-bolt mock installation from test_slack.py so
# SlackAdapter can be imported in environments without slack-bolt.
from tests.gateway.test_slack import _ensure_slack_mock  # noqa: F401

import plugins.platforms.slack.adapter as _slack_mod

_slack_mod.SLACK_AVAILABLE = True

from plugins.platforms.slack.adapter import SlackAdapter  # noqa: E402


def _make_adapter(extra=None):
    """Instantiate a bare SlackAdapter without running __init__."""
    adapter = object.__new__(SlackAdapter)
    adapter.config = MagicMock()
    adapter.config.extra = dict(extra or {})
    adapter._bot_user_id = "UBOT"
    return adapter


class TestMemberJoinedChannelHandler:
    """The ``member_joined_channel`` handler must only fire onboarding when
    the joiner is the bot itself; other users joining are ignored."""

    def test_non_bot_joiner_is_ignored(self):
        adapter = _make_adapter()
        adapter._handle_bot_added_to_channel = AsyncMock()

        # Reproduce the handler body inline (the closure is registered on
        # a Bolt app instance which we don't build here). This mirrors
        # adapter.py's ``handle_member_joined_channel``.
        async def handler(event):
            joined_user = event.get("user")
            channel_id = event.get("channel")
            if not joined_user or not channel_id:
                return
            if not adapter._bot_user_id or joined_user != adapter._bot_user_id:
                return
            await adapter._handle_bot_added_to_channel(channel_id)

        asyncio.run(handler({"user": "UOTHER", "channel": "C123"}))
        adapter._handle_bot_added_to_channel.assert_not_awaited()

    def test_bot_joiner_triggers_onboarding(self):
        adapter = _make_adapter()
        adapter._handle_bot_added_to_channel = AsyncMock()

        async def handler(event):
            joined_user = event.get("user")
            channel_id = event.get("channel")
            if not joined_user or not channel_id:
                return
            if not adapter._bot_user_id or joined_user != adapter._bot_user_id:
                return
            await adapter._handle_bot_added_to_channel(channel_id)

        asyncio.run(handler({"user": "UBOT", "channel": "C123"}))
        adapter._handle_bot_added_to_channel.assert_awaited_once_with("C123")


class TestBotAddedToChannelOnboarding:
    """``_handle_bot_added_to_channel`` posts a generic onboarding message
    when no binding exists yet, and skips posting if one already does."""

    def test_onboarding_message_posted_when_no_binding(self):
        adapter = _make_adapter(extra={})
        adapter.get_chat_info = AsyncMock(return_value={"name": "some-channel", "type": "channel"})
        client = MagicMock()
        client.chat_postMessage = AsyncMock()
        adapter._get_client = MagicMock(return_value=client)

        asyncio.run(adapter._handle_bot_added_to_channel("C123"))

        client.chat_postMessage.assert_awaited_once()
        kwargs = client.chat_postMessage.await_args.kwargs
        assert kwargs["channel"] == "C123"
        text = kwargs["text"]
        assert "some-channel" in text
        assert "C123" in text
        # No non-ASCII sneaked into the upstream onboarding string.
        assert text.isascii()

    def test_onboarding_skipped_when_binding_exists(self):
        adapter = _make_adapter(extra={
            "channel_skill_bindings": [{"id": "C123", "skills": ["some-skill"]}],
        })
        adapter.get_chat_info = AsyncMock(return_value={"name": "some-channel", "type": "channel"})
        client = MagicMock()
        client.chat_postMessage = AsyncMock()
        adapter._get_client = MagicMock(return_value=client)

        asyncio.run(adapter._handle_bot_added_to_channel("C123"))

        client.chat_postMessage.assert_not_awaited()


class TestChannelSkillBindingsHotReload:
    """``_refresh_channel_skill_bindings_if_changed`` reloads bindings from
    config.yaml only when the file's mtime has changed."""

    def _write_config(self, path: Path, bindings):
        import yaml
        payload = {"platforms": {"slack": {"channel_skill_bindings": bindings}}}
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    def test_reload_on_mtime_change(self, tmp_path, monkeypatch):
        # Redirect get_hermes_home() to tmp_path (the shared conftest already
        # sets HERMES_HOME, but we set it explicitly to be robust).
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        cfg = tmp_path / "config.yaml"
        self._write_config(cfg, [{"id": "C111", "skills": ["skill-a"]}])

        adapter = _make_adapter(extra={})
        adapter._refresh_channel_skill_bindings_if_changed()

        assert adapter.config.extra["channel_skill_bindings"] == [
            {"id": "C111", "skills": ["skill-a"]}
        ]

        # Second call with unchanged mtime must NOT re-parse. Prove this by
        # stubbing yaml.safe_load and asserting it is never invoked.
        import yaml
        called = {"n": 0}
        real_safe_load = yaml.safe_load

        def spy(*args, **kwargs):
            called["n"] += 1
            return real_safe_load(*args, **kwargs)

        monkeypatch.setattr(yaml, "safe_load", spy)
        adapter._refresh_channel_skill_bindings_if_changed()
        assert called["n"] == 0

        # Bump mtime and change bindings — next call must reload.
        self._write_config(cfg, [{"id": "C222", "skills": ["skill-b"]}])
        # ensure a strictly-newer mtime even on coarse filesystems
        st = cfg.stat()
        os.utime(cfg, (st.st_atime, st.st_mtime + 2))

        adapter._refresh_channel_skill_bindings_if_changed()
        assert adapter.config.extra["channel_skill_bindings"] == [
            {"id": "C222", "skills": ["skill-b"]}
        ]
        assert called["n"] == 1

    def test_missing_config_is_a_noop(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        # No config.yaml at all — must not raise, must not populate extra.
        adapter = _make_adapter(extra={})
        adapter._refresh_channel_skill_bindings_if_changed()
        assert "channel_skill_bindings" not in adapter.config.extra

    def test_flat_slack_layout_supported(self, tmp_path, monkeypatch):
        """Some layouts use top-level ``slack:`` instead of ``platforms.slack``."""
        import yaml
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            yaml.safe_dump({"slack": {"channel_skill_bindings": [{"id": "C1", "skills": ["s"]}]}}),
            encoding="utf-8",
        )
        adapter = _make_adapter(extra={})
        adapter._refresh_channel_skill_bindings_if_changed()
        assert adapter.config.extra["channel_skill_bindings"] == [
            {"id": "C1", "skills": ["s"]}
        ]
