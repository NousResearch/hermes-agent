"""Gateway-runner reads follow the source's profile under multiplex (fix 3 / 4b).

Real profile homes (default = HERMES root, ``educator`` under ``profiles/``)
with different slash admins, ignored channels, channel models and Slack
flags. Each runner-level read must use the profile that owns the inbound
source — and, with multiplex off, exactly what it used before.
"""
from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch

import pytest

from tests.gateway.test_slack_wake_bot_authored_threads_optin import _ensure_slack_mock

_ensure_slack_mock()

import plugins.platforms.slack.adapter as slack_mod  # noqa: E402

slack_mod.SLACK_AVAILABLE = True

from agent import secret_scope as ss  # noqa: E402
from gateway import tool_access  # noqa: E402
from gateway.config import GatewayConfig, Platform, PlatformConfig  # noqa: E402
from gateway.session import SessionSource  # noqa: E402
from plugins.platforms.slack.adapter import SlackAdapter, _apply_yaml_config  # noqa: E402

DEFAULT_CFG = """\
slack:
  allow_admin_from: [UDEF]
  ignored_channels: [CDEFIGN]
  allow_bots_channels: [CDEFBOTS]
  home_channel_prompt: true
  channel_models:
    CSHARED: default-model
"""
EDU_CFG = """\
slack:
  allow_admin_from: [UEDU]
  ignored_channels: [CEDUIGN]
  allow_bots_channels: [CEDUBOTS]
  home_channel_prompt: false
  channel_models:
    CSHARED: educator-model
"""


@pytest.fixture()
def homes(tmp_path, monkeypatch):
    root = tmp_path / "hermes"
    edu = root / "profiles" / "educator"
    edu.mkdir(parents=True)
    (root / "config.yaml").write_text(DEFAULT_CFG, encoding="utf-8")
    (edu / "config.yaml").write_text(EDU_CFG, encoding="utf-8")
    (root / ".env").write_text("SLACK_BOT_TOKEN=xoxb-default\n", encoding="utf-8")
    (edu / ".env").write_text("SLACK_BOT_TOKEN=xoxb-edu\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-default")
    # The default profile's YAML bridged into os.environ (first writer).
    monkeypatch.setenv("SLACK_IGNORED_CHANNELS", "CDEFIGN")
    monkeypatch.setenv("SLACK_ALLOW_BOTS_CHANNELS", "CDEFBOTS")
    monkeypatch.setenv("SLACK_HOME_CHANNEL_PROMPT", "true")
    tool_access._config_cache.clear()
    ss.set_multiplex_active(False)
    yield root, edu
    ss.set_multiplex_active(False)
    tool_access._config_cache.clear()


@contextmanager
def _scope(mapping):
    tok = ss.set_secret_scope(mapping)
    try:
        yield
    finally:
        ss.reset_secret_scope(tok)


def _adapter(yaml_block, env):
    with _scope(env):
        seeded = _apply_yaml_config({}, yaml_block) or {}
        return SlackAdapter(PlatformConfig(enabled=True, token=env.get("SLACK_BOT_TOKEN", ""), extra=seeded))


def _runner(multiplex: bool):
    from gateway.run import GatewayRunner

    ss.set_multiplex_active(multiplex)
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=multiplex)
    default = _adapter(
        {"allow_bots_channels": ["CDEFBOTS"], "home_channel_prompt": True},
        {"SLACK_BOT_TOKEN": "xoxb-default"},
    )
    default.config.extra["channel_models"] = {"CSHARED": "default-model"}
    edu = _adapter(
        {"allow_bots_channels": ["CEDUBOTS"], "home_channel_prompt": False},
        {"SLACK_BOT_TOKEN": "xoxb-edu"},
    )
    edu.config.extra["channel_models"] = {"CSHARED": "educator-model"}
    runner.adapters = {Platform.SLACK: default}
    runner._profile_adapters = {"educator": {Platform.SLACK: edu}}
    # Single-profile mode keeps using ``self.config`` for slash policy.
    runner.config.platforms[Platform.SLACK] = PlatformConfig(
        enabled=True, extra={"allow_admin_from": ["UDEF"], "ignored_channels": ["CDEFIGN"]}
    )
    return runner, default, edu


def _src(profile=None, user="U1", chat="C1", chat_type="group"):
    return SessionSource(
        platform=Platform.SLACK, chat_id=chat, chat_type=chat_type, user_id=user, profile=profile
    )


def _dm(profile=None, user="U1"):
    # allow_admin_from is the DM-scope admin list.
    return _src(profile, user=user, chat="D1", chat_type="dm")


def _admins(cfg):
    return set(cfg.platforms[Platform.SLACK].extra.get("allow_admin_from") or [])


class TestMultiplexOn:
    def test_gateway_config_is_the_source_profile(self, homes):
        runner, _, _ = _runner(True)
        assert _admins(runner._gateway_config_for_source(_src("educator"))) == {"UEDU"}
        assert _admins(runner._gateway_config_for_source(_src())) == {"UDEF"}

    def test_slash_admins_per_profile_both_directions(self, homes):
        runner, _, _ = _runner(True)
        assert runner._check_slash_access(_dm("educator", user="UEDU"), "restart") is None
        assert runner._check_slash_access(_dm("educator", user="UDEF"), "restart") is not None
        assert runner._check_slash_access(_dm(user="UDEF"), "restart") is None
        assert runner._check_slash_access(_dm(user="UEDU"), "restart") is not None

    def test_ignored_channels_per_profile(self, homes):
        runner, _, _ = _runner(True)
        assert runner._slack_source_ignored(_src("educator", chat="CEDUIGN")) is True
        assert runner._slack_source_ignored(_src("educator", chat="CDEFIGN")) is False
        assert runner._slack_source_ignored(_src(chat="CDEFIGN")) is True
        assert runner._slack_source_ignored(_src(chat="CEDUIGN")) is False

    def test_adapter_for_relevance_gate_and_dm_denial(self, homes):
        runner, default, edu = _runner(True)
        assert runner._multiplex_adapter_for(_src("educator"), Platform.SLACK) is edu
        assert runner._multiplex_adapter_for(_src(), Platform.SLACK) is default

    def test_channel_models_from_source_adapter(self, homes):
        runner, _, _ = _runner(True)
        seen = []

        def spy(extra, chat_id, parent_id):
            seen.append((extra or {}).get("channel_models", {}).get(chat_id))
            return None

        with patch("gateway.platforms.base.resolve_channel_model", spy):
            runner._apply_channel_model_override(_src("educator", chat="CSHARED"), {}, "m", {})
            runner._apply_channel_model_override(_src(chat="CSHARED"), {}, "m", {})
        assert seen[0] == "educator-model"
        assert "default-model" in seen[1:]
        assert "default-model" not in seen[:1]

    def test_home_channel_prompt_per_profile(self, homes):
        from gateway.run import _home_channel_prompt_enabled

        runner, default, edu = _runner(True)
        assert _home_channel_prompt_enabled("slack", edu) is False
        assert _home_channel_prompt_enabled("slack", default) is True

    def test_allow_bots_channels_gate_per_profile(self, homes):
        runner, _, _ = _runner(True)
        assert runner._slack_allow_bots_channels(_src("educator")) == {"CEDUBOTS"}
        assert runner._slack_allow_bots_channels(_src()) == {"CDEFBOTS"}


class TestMultiplexOff:
    def test_gateway_config_is_self_config(self, homes):
        runner, _, _ = _runner(False)
        assert runner._gateway_config_for_source(_src("educator")) is runner.config

    def test_adapter_lookup_unchanged(self, homes):
        runner, default, _ = _runner(False)
        assert runner._multiplex_adapter_for(_src("educator"), Platform.SLACK) is default

    def test_ignored_channels_use_runner_config(self, homes):
        runner, _, _ = _runner(False)
        assert runner._slack_source_ignored(_src(chat="CDEFIGN")) is True
        assert runner._slack_source_ignored(_src(chat="CEDUIGN")) is False

    def test_home_prompt_reads_env(self, homes, monkeypatch):
        from gateway.run import _home_channel_prompt_enabled

        runner, _, edu = _runner(False)
        assert _home_channel_prompt_enabled("slack", edu) is True
        monkeypatch.setenv("SLACK_HOME_CHANNEL_PROMPT", "false")
        assert _home_channel_prompt_enabled("slack") is False

    def test_slash_admins_from_runner_config(self, homes):
        runner, _, _ = _runner(False)
        assert runner._check_slash_access(_dm(user="UDEF"), "restart") is None
        assert runner._check_slash_access(_dm(user="UEDU"), "restart") is not None
