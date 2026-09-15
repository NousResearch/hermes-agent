"""Regression coverage for signed gateway identities exported to tool children.

Issue #112119: plain ``HERMES_SESSION_*`` values are advisory.  A tool that
needs an authorization principal must use the signed tuple instead.
"""

from gateway.session_context import clear_session_vars, set_session_vars
from gateway.identity_sig import verify_session_identity_from_env
from tools.environments.local import _make_run_env


def _signed_child_env():
    tokens = set_session_vars(platform="wecom", chat_id="team-chat", user_id="employee-42")
    try:
        return _make_run_env({})
    finally:
        clear_session_vars(tokens)


def test_child_receives_a_verifiable_platform_chat_user_identity():
    env = _signed_child_env()

    assert verify_session_identity_from_env(env) == {
        "platform": "wecom",
        "chat_id": "team-chat",
        "user_id": "employee-42",
    }


def test_plain_session_env_spoof_does_not_change_verified_identity():
    env = _signed_child_env()
    env["HERMES_SESSION_USER_ID"] = "owner"
    env["HERMES_SESSION_CHAT_ID"] = "owner-private-chat"

    assert verify_session_identity_from_env(env) == {
        "platform": "wecom",
        "chat_id": "team-chat",
        "user_id": "employee-42",
    }


def test_missing_or_malformed_identity_fails_closed():
    assert verify_session_identity_from_env({}) is None
    assert verify_session_identity_from_env({
        "HERMES_SESSION_IDENTITY": "not-a-signed-identity",
        "HERMES_SESSION_IDENTITY_PUBLIC_KEY": "also-not-a-key",
    }) is None

    env = _signed_child_env()
    env["HERMES_SESSION_IDENTITY"] = env["HERMES_SESSION_IDENTITY"][:-1] + "x"
    assert verify_session_identity_from_env(env) is None
