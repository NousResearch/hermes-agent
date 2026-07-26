"""Regression guard for Slack bot/workflow-sender authorization bypass.

Mirrors tests/gateway/test_feishu_bot_auth_bypass.py for Platform.SLACK.

Slack Workflow Builder posts (and other app/bot messages) arrive as
``subtype=bot_message`` with ``user=None``, so the SessionSource carries
``is_bot=True`` and ``user_id=None``. Without the #4466 bot bypass running
*before* the no-user-id guard, these senders are rejected at
``_is_user_authorized`` even when the operator enabled ``SLACK_ALLOW_BOTS`` --
the bug that makes @mentioning the bot from a Slack workflow do nothing.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig
from gateway.session import Platform, SessionSource
from plugins.platforms.slack.adapter import SlackAdapter, _apply_yaml_config


@pytest.fixture(autouse=True)
def _isolate_slack_env(monkeypatch):
    for var in (
        "SLACK_ALLOW_BOTS",
        "SLACK_ALLOWED_USERS",
        "SLACK_ALLOW_ALL_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
        "GATEWAY_ALLOWED_USERS",
    ):
        monkeypatch.delenv(var, raising=False)


def _make_bare_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.pairing_store = SimpleNamespace(is_approved=lambda *_a, **_kw: False)
    return runner


def _make_slack_bot_source():
    # Workflow Builder / app posts: subtype=bot_message, user=None.
    return SessionSource(
        platform=Platform.SLACK,
        chat_id="C0123",
        chat_type="group",
        user_id=None,
        user_name="",
        is_bot=True,
    )


def _make_slack_human_source(user_id="U_human"):
    return SessionSource(
        platform=Platform.SLACK,
        chat_id="C0123",
        chat_type="group",
        user_id=user_id,
        user_name="Human",
        is_bot=False,
    )


def test_slack_bot_profile_scope_overrides_process_policy(monkeypatch):
    from agent import secret_scope

    runner = _make_bare_runner()
    monkeypatch.setenv("SLACK_ALLOW_BOTS", "none")
    previous_multiplex = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({"SLACK_ALLOW_BOTS": "mentions"})
    try:
        assert runner._is_user_authorized(_make_slack_bot_source()) is True
    finally:
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(previous_multiplex)


def test_slack_bot_empty_profile_scope_does_not_inherit_process_policy(monkeypatch):
    from agent import secret_scope

    runner = _make_bare_runner()
    monkeypatch.setenv("SLACK_ALLOW_BOTS", "all")
    previous_multiplex = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({})
    try:
        assert runner._is_user_authorized(_make_slack_bot_source()) is False
    finally:
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(previous_multiplex)


def test_slack_adapter_without_profile_policy_does_not_inherit_process_policy(
    monkeypatch,
):
    """A profile adapter with no policy must not inherit another profile's bridge."""
    from agent import secret_scope

    monkeypatch.setenv("SLACK_ALLOW_BOTS", "all")
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="secondary", extra={}))
    runner = _make_bare_runner()
    runner.adapters = {}
    runner._profile_adapters = {"secondary": {Platform.SLACK: adapter}}
    source = adapter.build_source(chat_id="C-secondary", chat_type="group", is_bot=True)
    source.profile = "secondary"

    previous_multiplex = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({})
    try:
        assert adapter._slack_allow_bots() == "none"
        assert runner._is_user_authorized(source) is False
    finally:
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(previous_multiplex)


def test_slack_bot_authorized_when_allow_bots_all(monkeypatch):
    runner = _make_bare_runner()
    monkeypatch.setenv("SLACK_ALLOW_BOTS", "all")
    assert runner._is_user_authorized(_make_slack_bot_source()) is True


def test_slack_bot_authorized_when_allow_bots_mentions(monkeypatch):
    runner = _make_bare_runner()
    monkeypatch.setenv("SLACK_ALLOW_BOTS", "mentions")
    assert runner._is_user_authorized(_make_slack_bot_source()) is True


def test_slack_bot_denied_when_allow_bots_unset(monkeypatch):
    # No SLACK_ALLOW_BOTS + no user_id => denied (no bypass, hits guard).
    runner = _make_bare_runner()
    assert runner._is_user_authorized(_make_slack_bot_source()) is False


def test_slack_bot_denied_when_allow_bots_none(monkeypatch):
    runner = _make_bare_runner()
    monkeypatch.setenv("SLACK_ALLOW_BOTS", "none")
    assert runner._is_user_authorized(_make_slack_bot_source()) is False


def test_slack_human_unaffected_by_bot_bypass(monkeypatch):
    runner = _make_bare_runner()
    monkeypatch.setenv("SLACK_ALLOW_ALL_USERS", "true")
    assert runner._is_user_authorized(_make_slack_human_source()) is True


@pytest.mark.parametrize(
    "load_order",
    [
        (("primary", "all"), ("secondary", "none")),
        (("secondary", "none"), ("primary", "all")),
    ],
)
def test_multiplex_yaml_bot_policy_matches_adapter_and_final_auth(
    monkeypatch, load_order
):
    """Profile YAML bot policy survives either multiplex profile load order."""
    from agent import secret_scope

    extras = {}
    for profile, policy in load_order:
        extras[profile] = _apply_yaml_config({}, {"allow_bots": policy})

    assert extras["primary"]["allow_bots"] == "all"
    assert extras["secondary"]["allow_bots"] == "none"

    primary = SlackAdapter(
        PlatformConfig(enabled=True, token="primary", extra=extras["primary"])
    )
    secondary = SlackAdapter(
        PlatformConfig(enabled=True, token="secondary", extra=extras["secondary"])
    )
    runner = _make_bare_runner()
    runner.adapters = {Platform.SLACK: primary}
    runner._profile_adapters = {"secondary": {Platform.SLACK: secondary}}

    primary_source = primary.build_source(
        chat_id="C-primary",
        chat_type="group",
        is_bot=True,
    )
    primary_source.profile = "primary"
    secondary_source = secondary.build_source(
        chat_id="C-secondary",
        chat_type="group",
        is_bot=True,
    )
    secondary_source.profile = "secondary"

    previous_multiplex = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({})
    try:
        assert primary._slack_allow_bots() == "all"
        assert runner._is_user_authorized(primary_source) is True
        assert secondary._slack_allow_bots() == "none"
        assert runner._is_user_authorized(secondary_source) is False
    finally:
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(previous_multiplex)
