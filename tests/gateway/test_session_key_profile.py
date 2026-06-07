"""build_session_key profile prefix — default 'main' keeps byte-identical keys."""

from gateway.session import build_session_key, SessionSource
from gateway.platforms.base import Platform


def test_default_profile_keys_unchanged():
    s = SessionSource(platform=Platform.TELEGRAM, chat_id="100", chat_type="group", user_id="u1")
    assert build_session_key(s).startswith("agent:main:")


def test_dm_default_unchanged():
    s = SessionSource(platform=Platform.TELEGRAM, chat_id="55", chat_type="dm")
    assert build_session_key(s) == "agent:main:telegram:dm:55"


def test_named_profile_prefix_group():
    s = SessionSource(platform=Platform.TELEGRAM, chat_id="100", chat_type="group", user_id="u1")
    assert build_session_key(s, profile="research").startswith("agent:research:")


def test_named_profile_prefix_dm():
    s = SessionSource(platform=Platform.TELEGRAM, chat_id="55", chat_type="dm")
    assert build_session_key(s, profile="research") == "agent:research:telegram:dm:55"


def test_named_profile_only_changes_prefix():
    s = SessionSource(platform=Platform.TELEGRAM, chat_id="100", chat_type="group", user_id="u1")
    main = build_session_key(s)
    named = build_session_key(s, profile="coder")
    assert named == main.replace("agent:main:", "agent:coder:", 1)
