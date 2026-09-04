"""Per-account session identity — #8287.

A gateway hosting multiple bot accounts on one platform must keep their
conversations apart: the same chat reached through two bots is two sessions.
The account rides in the session-key namespace slot (``agent:main@support``)
— the same mechanism profiles use — so every positional parser
(``parts[2] == platform`` etc.) keeps its layout, and single-bot gateways
produce byte-identical keys to before.
"""

from gateway.config import Platform
from gateway.run import _parse_session_key
from gateway.session import (
    SessionSource,
    build_session_key,
    split_key_namespace,
)


def _source(account=None, **kw):
    defaults = dict(
        platform=Platform.TELEGRAM, chat_id="777", chat_type="dm", user_id="777"
    )
    defaults.update(kw)
    return SessionSource(account=account, **defaults)


def test_same_chat_two_bots_two_sessions():
    """The #10455-review isolation requirement: identical chat + user via
    two different bot accounts must never share a session key."""
    key_default = build_session_key(_source(account=None))
    key_support = build_session_key(_source(account="support"))
    key_sales = build_session_key(_source(account="sales"))
    assert len({key_default, key_support, key_sales}) == 3


def test_default_account_key_is_byte_identical_to_legacy():
    """Single-bot gateways must keep every key they have ever generated."""
    assert build_session_key(_source(account=None)) == "agent:main:telegram:dm:777"
    assert build_session_key(_source(account="default")) == "agent:main:telegram:dm:777"


def test_account_key_keeps_positional_layout():
    """The account lives in the namespace slot — platform/chat_type/chat_id
    stay at parts[2:5], so positional parsers are unaffected."""
    key = build_session_key(_source(account="support"))
    parts = key.split(":")
    assert parts[0] == "agent"
    assert parts[1] == "main@support"
    assert parts[2] == "telegram"
    assert parts[3] == "dm"
    assert parts[4] == "777"


def test_profile_and_account_compose():
    key = build_session_key(_source(account="support"), profile="coder")
    assert key.startswith("agent:coder@support:telegram:")


def test_group_and_thread_keys_carry_account():
    group_a = build_session_key(
        _source(account="support", chat_type="group", chat_id="-100", user_id="9")
    )
    group_b = build_session_key(
        _source(account=None, chat_type="group", chat_id="-100", user_id="9")
    )
    assert group_a != group_b
    assert group_a.split(":")[1] == "main@support"


def test_source_account_round_trips_serialization():
    src = _source(account="support")
    rebuilt = SessionSource.from_dict(src.to_dict())
    assert rebuilt.account == "support"
    # Default account stays wire-invisible (no key emitted), like profile.
    assert "account" not in _source(account=None).to_dict()


def test_split_key_namespace():
    assert split_key_namespace("main") == ("main", None)
    assert split_key_namespace("main@support") == ("main", "support")
    assert split_key_namespace("coder@support") == ("coder", "support")
    assert split_key_namespace("") == ("", None)


def test_profile_resolution_ignores_account_suffix():
    from gateway.session import SessionStore

    resolve = SessionStore._profile_from_session_key
    assert resolve("agent:main:telegram:dm:1") == "default"
    assert resolve("agent:main@support:telegram:dm:1") == "default"
    assert resolve("agent:coder@support:telegram:dm:1") == "coder"


def test_parse_session_key_accepts_account_namespace():
    parsed = _parse_session_key("agent:main@support:telegram:dm:777:42")
    assert parsed == {
        "platform": "telegram",
        "chat_type": "dm",
        "chat_id": "777",
        "account": "support",
        "thread_id": "42",
    }
    # Default-namespace behavior unchanged.
    legacy = _parse_session_key("agent:main:telegram:dm:777")
    assert legacy == {"platform": "telegram", "chat_type": "dm", "chat_id": "777"}
    # Named-profile keys stay excluded, as before.
    assert _parse_session_key("agent:coder:telegram:dm:777") is None
