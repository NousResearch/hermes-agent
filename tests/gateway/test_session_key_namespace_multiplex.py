"""Regression tests: session-key namespace must honor named profiles under multiplexing (#105931).

Incident shape: with ``multiplex_profiles`` active, ``build_session_key`` (via
``_session_key_namespace``) emits ``agent:<profile>:...`` keys for named profiles, but two
hardcoded ``agent:main`` sites failed to match them:

1. ``gateway/run.py:_parse_session_key`` checked ``parts[1] == "main"`` and so returned
   ``None`` for any named-profile key. Consumers (``run_shutdown.py`` notification target,
   ``run_notifications.py``) then fell back to a stale LRU source cache and could route
   delivery/wake to the wrong place.

2. ``gateway/run_busy.py:_sibling_thread_run_keys`` built its sibling-match prefix from a
   hardcoded ``"agent:main"``, so under multiplexing the prefix never matched the actual
   ``agent:<profile>:...`` run keys and ``/stop``-style sibling discovery missed them.

Fix under test: the parser accepts any non-empty namespace (``agent:<profile>:`` parses the
same way ``agent:main:`` always has), and the busy-prefix uses
``_session_key_namespace(source.profile)`` instead of the literal. Default/``None`` profiles
remain byte-identical to every historical key, so single-profile installs are unaffected.
"""

import pytest

from gateway.config import Platform
from gateway.run import _parse_session_key
from gateway.run_busy import GatewayBusySessionMixin
from gateway.session import SessionSource, _session_key_namespace


# ---------------------------------------------------------------------------
# _session_key_namespace helper — byte-identical for default/None profiles
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("profile", [None, "default", ""])
def test_namespace_default_is_byte_identical_agent_main(profile):
    """Default/None/empty profile MUST keep producing ``agent:main`` so every historical
    single-profile key is unchanged (no silent re-keying of existing installs)."""
    assert _session_key_namespace(profile) == "agent:main"


@pytest.mark.parametrize("profile", ["medicina", "prod", "beta-bot"])
def test_namespace_named_profile_is_agent_profile(profile):
    assert _session_key_namespace(profile) == f"agent:{profile}"


# ---------------------------------------------------------------------------
# _parse_session_key — default-profile keys are UNCHANGED (regression guard)
# ---------------------------------------------------------------------------


def test_parse_default_dm_key_unchanged():
    """The canonical single-profile DM key parses exactly as before the fix."""
    result = _parse_session_key("agent:main:whatsapp:dm:123456")
    assert result == {"platform": "whatsapp", "chat_type": "dm", "chat_id": "123456"}
    # DM keys never carry thread_id unless a 6th part is present.
    assert "thread_id" not in result


def test_parse_default_dm_key_with_thread_unchanged():
    result = _parse_session_key("agent:main:telegram:dm:111:222")
    assert result == {
        "platform": "telegram",
        "chat_type": "dm",
        "chat_id": "111",
        "thread_id": "222",
    }


def test_parse_default_group_key_unchanged():
    """Group keys keep the 6th part as a user_id, not a thread_id (only dm/thread get it)."""
    result = _parse_session_key("agent:main:discord:group:chan:thread456")
    assert result == {"platform": "discord", "chat_type": "group", "chat_id": "chan"}
    assert "thread_id" not in result


# ---------------------------------------------------------------------------
# _parse_session_key — named-profile keys now parse (the fix)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key,expected",
    [
        # Named-profile DM (was rejected with parts[1] == "main" hardcode).
        (
            "agent:medicina:whatsapp:dm:123456",
            {"platform": "whatsapp", "chat_type": "dm", "chat_id": "123456"},
        ),
        # Named-profile DM with thread_id.
        (
            "agent:prod:telegram:dm:111:222",
            {
                "platform": "telegram",
                "chat_type": "dm",
                "chat_id": "111",
                "thread_id": "222",
            },
        ),
        # Named-profile group — 6th part stays a user_id.
        (
            "agent:beta:discord:group:chan:u42",
            {"platform": "discord", "chat_type": "group", "chat_id": "chan"},
        ),
    ],
)
def test_parse_named_profile_key_now_resolves(key, expected):
    """Pre-fix these returned None (parts[1] != "main"); now they parse like default keys."""
    assert _parse_session_key(key) == expected


def test_parse_named_profile_thread_key_has_thread_id():
    result = _parse_session_key("agent:medicina:discord:thread:chan:t1")
    assert result == {
        "platform": "discord",
        "chat_type": "thread",
        "chat_id": "chan",
        "thread_id": "t1",
    }


# ---------------------------------------------------------------------------
# _parse_session_key — invalid keys still return None
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        "",                                     # empty
        "garbage",                              # not a session key
        "session:main:whatsapp:dm:123",         # wrong prefix
        "agent:main:whatsapp:dm",               # too few parts
        "agent::whatsapp:dm:123:456",           # empty namespace (invalid)
        "agent:whatsapp:dm",                     # 3 parts only
    ],
)
def test_parse_invalid_key_returns_none(key):
    assert _parse_session_key(key) is None


# ---------------------------------------------------------------------------
# run_busy: _sibling_thread_run_keys prefix honors source.profile (the fix)
# ---------------------------------------------------------------------------


class _StubBusySession(GatewayBusySessionMixin):
    """Only the sibling-discovery seam is under test: stub _running_agent_items so the
    prefix match is exercised against real keys without spinning up GatewayRunner."""

    def __init__(self, running):
        # running: list[(key, agent)] — agent is truthy to mean "an active run".
        self._running = list(running)

    def _running_agent_items(self):
        return list(self._running)


def _thread_source(profile=None):
    return SessionSource(
        platform=Platform.WHATSAPP,
        chat_id="chat-1",
        chat_type="thread",
        thread_id="t1",
        profile=profile,
    )


def test_sibling_keys_match_named_profile_namespace():
    """Under multiplexing, a named-profile source must match siblings in its OWN namespace
    (``agent:medicina:...``) and NOT the default-namespace keys of other profiles."""
    source = _thread_source(profile="medicina")
    own_key = "agent:medicina:whatsapp:thread:chat-1:t1:me"
    same_namespace_sibling = "agent:medicina:whatsapp:thread:chat-1:t1:other_user"
    # A run under the DEFAULT profile in the same thread/chat — a different agent, must NOT match.
    default_namespace_key = "agent:main:whatsapp:thread:chat-1:t1:someone_else"
    # A run under ANOTHER named profile — also must not match.
    other_named_key = "agent:prod:whatsapp:thread:chat-1:t1:third"

    stub = _StubBusySession(
        [
            (same_namespace_sibling, object()),
            (default_namespace_key, object()),
            (other_named_key, object()),
        ]
    )
    result = stub._sibling_thread_run_keys(source, own_key)

    assert same_namespace_sibling in result
    assert default_namespace_key not in result
    assert other_named_key not in result
    assert own_key not in result  # own_key is always excluded


def test_sibling_keys_default_profile_unchanged():
    """Default-profile source still matches default-namespace siblings (no regression)."""
    source = _thread_source(profile=None)  # default -> agent:main
    own_key = "agent:main:whatsapp:thread:chat-1:t1:me"
    default_sibling = "agent:main:whatsapp:thread:chat-1:t1:other_user"
    named_key = "agent:medicina:whatsapp:thread:chat-1:t1:medicina_user"

    stub = _StubBusySession(
        [(default_sibling, object()), (named_key, object())]
    )
    result = stub._sibling_thread_run_keys(source, own_key)

    assert default_sibling in result
    assert named_key not in result  # named-profile key doesn't match default-namespace prefix


def test_sibling_keys_skip_missing_thread_or_chat():
    """Without thread_id/chat_id there is nothing to match — early empty list."""
    no_thread = SessionSource(
        platform=Platform.WHATSAPP, chat_id="c", chat_type="dm", profile="medicina"
    )
    stub = _StubBusySession([("agent:medicina:whatsapp:dm:c:anything", object())])
    assert stub._sibling_thread_run_keys(no_thread, "own") == []

    no_chat = SessionSource(
        platform=Platform.WHATSAPP, chat_id="", chat_type="thread", thread_id="t1",
        profile="medicina",
    )
    assert stub._sibling_thread_run_keys(no_chat, "own") == []
