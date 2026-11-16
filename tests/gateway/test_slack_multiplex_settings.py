"""Slack adapter settings resolve per profile under multiplex (fix 2 / 4b).

Two adapters are built under two profiles' secret scopes, each with its own
``config.yaml`` ``slack:`` block (``_apply_yaml_config`` output merged into
``extra``) and ``.env`` values, while ``os.environ`` carries a third,
"default-profile" value for every ``SLACK_*`` key. Every reader must return
its own profile's value; nothing may come from ``os.environ``.

A second block pins multiplex OFF: ``_apply_yaml_config`` still bridges to
``os.environ`` exactly as before and ``.env`` beats ``config.yaml`` for the
keys that used to be read only from the environment.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.gateway.test_slack_wake_bot_authored_threads_optin import _ensure_slack_mock

_ensure_slack_mock()

import plugins.platforms.slack.adapter as slack_mod  # noqa: E402

slack_mod.SLACK_AVAILABLE = True

from agent import secret_scope as ss  # noqa: E402
from gateway.config import PlatformConfig  # noqa: E402
from plugins.platforms.slack.adapter import (  # noqa: E402
    PROFILE_SETTINGS_KEY,
    SlackAdapter,
    _apply_yaml_config,
)

# Values the DEFAULT profile leaked into os.environ (first-writer bridge).
DEFAULT_ENV = {
    "SLACK_REQUIRE_MENTION": "false",
    "SLACK_STRICT_MENTION": "true",
    "SLACK_IGNORE_OTHER_USER_MENTIONS": "true",
    "SLACK_THREAD_REQUIRE_MENTION": "true",
    "SLACK_WAKE_IN_BOT_AUTHORED_THREADS": "true",
    "SLACK_ALLOW_BOTS": "all",
    "SLACK_ALLOW_BOTS_CHANNELS": "CDEFAULTBOTS",
    "SLACK_FREE_RESPONSE_CHANNELS": "CDEFAULTFREE",
    "SLACK_QUIET_CHANNELS": "CDEFAULTQUIET",
    "SLACK_REQUIRE_MENTION_CHANNELS": "CDEFAULTRMC",
    "SLACK_MENTION_PATTERNS": "defaultbot",
    "SLACK_REACTIONS": "false",
    "SLACK_REACTION_TRIGGERS": "eyes",
    "SLACK_REACTION_TRIGGER_TARGET": "CDEFAULTTARGET",
    "SLACK_DISABLE_DMS": "true",
    "SLACK_ALLOWED_CHANNELS": "CDEFAULTALLOWED",
    "SLACK_IGNORED_CHANNELS": "CDEFAULTIGNORED",
    "SLACK_ALLOW_ALL_USERS": "true",
    "SLACK_ALLOWED_USERS": "UDEFAULT",
    "SLACK_DEDUP_TTL_SECONDS": "999",
}

EDUCATOR_YAML = {
    "require_mention": True,
    "strict_mention": False,
    "ignore_other_user_mentions": False,
    "thread_require_mention": False,
    "wake_in_bot_authored_threads": False,
    "allow_bots": "mentions",
    "allow_bots_channels": ["CEDUBOTS"],
    "free_response_channels": "CEDUFREE",
    "quiet_channels": ["CEDUQUIET"],
    "require_mention_channels": ["CEDURMC"],
    "mention_patterns": ["edubot"],
    "reactions": True,
    "reaction_triggers": ["star"],
    "reaction_trigger_target": "CEDUTARGET",
    "disable_dms": False,
    "allowed_channels": ["CEDUALLOWED"],
    "ignored_channels": ["CEDUIGNORED"],
}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k, v in DEFAULT_ENV.items():
        monkeypatch.setenv(k, v)
    ss.set_multiplex_active(False)
    yield
    ss.set_multiplex_active(False)


@contextmanager
def scope(mapping):
    tok = ss.set_secret_scope(mapping)
    try:
        yield
    finally:
        ss.reset_secret_scope(tok)


def _build(extra, env):
    """Build an adapter the way _start_one_profile_adapters does: in scope."""
    with scope(env):
        return SlackAdapter(PlatformConfig(enabled=True, token="xoxb-x", extra=dict(extra)))


def _educator(env=None):
    ss.set_multiplex_active(True)
    before = dict(os.environ)
    seeded = _apply_yaml_config({}, dict(EDUCATOR_YAML))
    assert dict(os.environ) == before, "multiplex: YAML must not reach os.environ"
    return _build(seeded, env or {"SLACK_BOT_TOKEN": "xoxb-edu"})


class TestMultiplexOn:
    def test_every_flag_comes_from_the_profile(self):
        a = _educator()
        assert a._slack_require_mention() is True
        assert a._slack_strict_mention() is False
        assert a._slack_ignore_other_user_mentions() is False
        assert a._slack_thread_require_mention() is False
        assert a._slack_wake_in_bot_authored_threads() is False
        assert a._slack_allow_bots() == "mentions"
        assert a._slack_allow_bots("CEDUBOTS") == "all"
        assert a._slack_allow_bots_channels() == {"CEDUBOTS"}
        assert a._slack_free_response_channels() == {"CEDUFREE"}
        assert a._slack_quiet_channels() == {"CEDUQUIET"}
        assert a._slack_require_mention_channels() == {"CEDURMC"}
        assert [p.pattern for p in a._slack_mention_patterns()] == ["edubot"]
        assert a._reactions_enabled() is True
        assert a._slack_reaction_triggers() == {"star"}
        assert a._slack_reaction_trigger_target() == ("CEDUTARGET", "")
        assert a._slack_disable_dms() is False
        assert a._slack_allowed_channels() == {"CEDUALLOWED"}
        assert a._slack_ignored_channels() == {"CEDUIGNORED"}

    def test_scoped_miss_is_default_not_os_environ(self):
        ss.set_multiplex_active(True)
        a = _build({}, {"SLACK_BOT_TOKEN": "xoxb-edu"})
        assert a._slack_require_mention() is True
        assert a._slack_strict_mention() is False
        assert a._slack_ignore_other_user_mentions() is False
        assert a._slack_thread_require_mention() is False
        assert a._slack_wake_in_bot_authored_threads() is False
        assert a._slack_allow_bots() == "none"
        assert a._slack_allow_bots_channels() == set()
        assert a._slack_free_response_channels() == set()
        assert a._slack_quiet_channels() == set()
        assert a._slack_require_mention_channels() == set()
        assert a._slack_mention_patterns() == []
        assert a._reactions_enabled() is True
        assert a._slack_reaction_triggers() is None
        assert a._slack_reaction_trigger_target() == ("", "")
        assert a._slack_disable_dms() is False
        assert a._slack_allowed_channels() == set()
        assert a._slack_ignored_channels() == set()
        with scope({}):
            assert slack_mod._slack_dedup_ttl_seconds() != 999

    def test_profile_env_is_used_when_yaml_absent(self):
        ss.set_multiplex_active(True)
        a = _build({}, {"SLACK_STRICT_MENTION": "true", "SLACK_QUIET_CHANNELS": "CENVQ"})
        assert a._slack_strict_mention() is True
        assert a._slack_quiet_channels() == {"CENVQ"}

    def test_yaml_beats_profile_env(self):
        ss.set_multiplex_active(True)
        seeded = _apply_yaml_config({}, {"strict_mention": False})
        a = _build(seeded, {"SLACK_STRICT_MENTION": "true"})
        assert a._slack_strict_mention() is False

    def test_reads_use_construction_scope_outside_it(self):
        """A method running outside the adapter's context (e.g. under the
        other profile's turn) still reads its own profile."""
        ss.set_multiplex_active(True)
        a = _build({}, {"SLACK_QUIET_CHANNELS": "COWN"})
        with scope({"SLACK_QUIET_CHANNELS": "COTHER"}):
            assert a._slack_quiet_channels() == {"COWN"}

    def test_interactive_auth_uses_profile_allowlist(self):
        ss.set_multiplex_active(True)
        a = _build({}, {"SLACK_ALLOWED_USERS": "UEDU"})
        a._message_handler = None
        assert a._is_interactive_user_authorized("UEDU", channel_id="C1") is True
        assert a._is_interactive_user_authorized("UDEFAULT", channel_id="C1") is False

    @pytest.mark.asyncio
    async def test_cron_wake_zone_follows_each_profile(self):
        ss.set_multiplex_active(True)
        on = _build(_apply_yaml_config({}, {"wake_in_bot_authored_threads": True}), {})
        off = _build(_apply_yaml_config({}, {"wake_in_bot_authored_threads": False}), {})
        for adapter in (on, off):
            adapter._app = MagicMock()
            adapter._app.client = AsyncMock()
            adapter._app.client.chat_postMessage = AsyncMock(
                return_value={"ok": True, "ts": "1790000000.000100"}
            )
            adapter._bot_user_id = "UBOT"
            adapter._running = True
            await adapter.send("C1", "item", metadata={"job_id": "j1"})
        assert any(m.endswith("1790000000.000100") for m in on._bot_message_ts)
        assert not any(m.endswith("1790000000.000100") for m in off._bot_message_ts)

    def test_unscoped_default_adapter_reads_its_own_env(self):
        """The default profile's adapter is built unscoped; its env IS
        os.environ (secondary YAML no longer writes there)."""
        ss.set_multiplex_active(True)
        a = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-default", extra={}))
        assert a._slack_quiet_channels() == {"CDEFAULTQUIET"}


class TestMultiplexOff:
    """Regression guard: single-profile behaviour is unchanged."""

    def test_yaml_bridge_still_writes_env(self, monkeypatch):
        for k in DEFAULT_ENV:
            monkeypatch.delenv(k, raising=False)
        _apply_yaml_config({}, dict(EDUCATOR_YAML))
        assert os.environ["SLACK_STRICT_MENTION"] == "false"
        assert os.environ["SLACK_ALLOWED_CHANNELS"] == "CEDUALLOWED"
        assert os.environ["SLACK_REACTIONS"] == "true"

    def test_off_env_beats_yaml_for_env_only_keys(self, monkeypatch):
        """A key set in BOTH .env and config.yaml: .env wins, as today."""
        monkeypatch.setenv("SLACK_STRICT_MENTION", "false")
        monkeypatch.setenv("SLACK_REACTIONS", "true")
        seeded = _apply_yaml_config({}, {"strict_mention": True, "reactions": False})
        a = SlackAdapter(PlatformConfig(enabled=True, token="xoxb", extra=dict(seeded or {})))
        assert a._slack_strict_mention() is False
        assert a._reactions_enabled() is True

    def test_readers_return_todays_env_values(self):
        a = SlackAdapter(PlatformConfig(enabled=True, token="xoxb", extra={}))
        assert a._slack_require_mention() is False
        assert a._slack_strict_mention() is True
        assert a._slack_wake_in_bot_authored_threads() is True
        assert a._slack_allow_bots() == "all"
        assert a._slack_quiet_channels() == {"CDEFAULTQUIET"}
        assert a._slack_ignored_channels() == {"CDEFAULTIGNORED"}
        assert a._reactions_enabled() is False
        assert a._slack_reaction_triggers() == {"eyes"}
        assert a._slack_disable_dms() is True
        assert slack_mod._slack_dedup_ttl_seconds() == 999

    def test_explicit_extra_still_wins_where_it_did(self):
        a = SlackAdapter(PlatformConfig(enabled=True, token="xoxb", extra={"quiet_channels": "CX"}))
        assert a._slack_quiet_channels() == {"CX"}

    def test_profile_settings_ignored_when_off(self):
        seeded = {PROFILE_SETTINGS_KEY: {"quiet_channels": ["CPROFILE"]}}
        a = SlackAdapter(PlatformConfig(enabled=True, token="xoxb", extra=seeded))
        assert a._slack_quiet_channels() == {"CDEFAULTQUIET"}
