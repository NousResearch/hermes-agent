"""Regression coverage for signed gateway identities exported to tool children.

Issue #112119: plain ``HERMES_SESSION_*`` values are advisory.  A tool that
needs an authorization principal must use the signed tuple instead.
"""

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateway.session_context import (
    clear_session_vars,
    current_session_identity_authority,
    set_session_vars,
)
from gateway.identity_sig import verify_session_identity_from_env
from tools.environments.local import _make_run_env


def _signed_child_env():
    tokens = set_session_vars(platform="wecom", chat_id="team-chat", user_id="employee-42")
    try:
        return _make_run_env({}), current_session_identity_authority()
    finally:
        clear_session_vars(tokens)


def test_child_receives_a_verifiable_platform_chat_user_identity():
    env, authority = _signed_child_env()

    assert "HERMES_SESSION_IDENTITY_PUBLIC_KEY" not in env
    assert verify_session_identity_from_env(env, authority=authority) == {
        "platform": "wecom",
        "chat_id": "team-chat",
        "user_id": "employee-42",
    }


def test_plain_session_env_spoof_does_not_change_verified_identity():
    env, authority = _signed_child_env()
    env["HERMES_SESSION_USER_ID"] = "owner"
    env["HERMES_SESSION_CHAT_ID"] = "owner-private-chat"

    assert verify_session_identity_from_env(env, authority=authority) == {
        "platform": "wecom",
        "chat_id": "team-chat",
        "user_id": "employee-42",
    }


def test_child_supplied_token_and_public_key_cannot_replace_parent_authority():
    env, authority = _signed_child_env()
    forged_key = Ed25519PrivateKey.generate()
    forged_payload = b'{"audience":"tool-executor","generation":"forged","platform":"wecom","chat_id":"owner-private-chat","user_id":"owner"}'
    forged_signature = forged_key.sign(forged_payload)

    from gateway.identity_sig import _b64encode

    env["HERMES_SESSION_IDENTITY"] = f"v2.{_b64encode(forged_payload)}.{_b64encode(forged_signature)}"
    env["HERMES_SESSION_IDENTITY_PUBLIC_KEY"] = _b64encode(forged_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    ))

    assert verify_session_identity_from_env(env, authority=authority) is None


def test_captured_turn_a_identity_cannot_replay_under_turn_b_authority():
    tokens_a = set_session_vars(platform="wecom", chat_id="team-chat", user_id="employee-42")
    try:
        captured = _make_run_env({})
    finally:
        clear_session_vars(tokens_a)

    tokens_b = set_session_vars(platform="wecom", chat_id="other-chat", user_id="employee-99")
    try:
        assert verify_session_identity_from_env(captured, authority=current_session_identity_authority()) is None
    finally:
        clear_session_vars(tokens_b)


def test_missing_or_malformed_identity_fails_closed():
    _, authority = _signed_child_env()
    assert verify_session_identity_from_env({}, authority=authority) is None
    assert verify_session_identity_from_env({
        "HERMES_SESSION_IDENTITY": "not-a-signed-identity",
        "HERMES_SESSION_IDENTITY_PUBLIC_KEY": "also-not-a-key",
    }, authority=authority) is None

    env, authority = _signed_child_env()
    env["HERMES_SESSION_IDENTITY"] = env["HERMES_SESSION_IDENTITY"][:-1] + "x"
    assert verify_session_identity_from_env(env, authority=authority) is None
