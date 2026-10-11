"""``rotate-session``: rotating one channel's session on demand from outside the gateway process.

The contract under test is relational, not a snapshot: the channel that was named gets a NEW session
id and its PREVIOUS row is closed with ``session_reset``, while every other channel on the same
gateway keeps the session it had. An unknown channel rotates nothing and says so — a caller that
fires before the channel's first message must not have to special-case the answer.

Round trips go over a REAL unix socket into a REAL ``SessionStore`` (real ``state.db``): client →
socket → verb → in-memory routing index → store → DB row, which is the whole point of the verb
(``reset_session`` is in-memory, so nothing outside the process can do this itself).
"""

import asyncio
import json

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.control_socket import GatewayControlServer, rotate_gateway_session
from gateway.run_session_rotate import rotate_session_verb
from gateway.session import SessionSource, SessionStore

pytestmark = pytest.mark.platforms("posix")  # Unix-socket transport; the pipe half is the wine2e lane

CHAT = "-1004453690729"
TOPIC = "6"


def _make_store(tmp_path, *, multiplex: bool = False) -> SessionStore:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True)},
                           multiplex_profiles=multiplex)
    store = SessionStore(sessions_dir=sessions_dir, config=config)
    assert store._db is not None, "test requires a real SessionDB"
    return store


class _Runner:
    """The gateway surface the verb reads: the live session store."""

    def __init__(self, store: SessionStore) -> None:
        self.session_store = store


def _source(*, chat_id=CHAT, thread_id=TOPIC, user_id="229917144", chat_type="group",
            profile=None) -> SessionSource:
    return SessionSource(platform=Platform.TELEGRAM, chat_id=chat_id, chat_type=chat_type,
                         thread_id=thread_id, user_id=user_id, profile=profile)


def _ask(store, home, **params):
    """One real round trip: client → socket → verb. None when no gateway answers."""
    async def scenario():
        server = GatewayControlServer(
            home, verb_handlers={"rotate-session": rotate_session_verb(_Runner(store))})
        assert await server.start()
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, lambda: rotate_gateway_session(home, **params))
        finally:
            await server.stop()

    return asyncio.run(scenario())


# ---------------------------------------------------------------------------
# The rotation itself
# ---------------------------------------------------------------------------

def test_verb_rotates_the_named_channel_and_closes_its_row(tmp_path):
    store = _make_store(tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir()
    before = store.get_or_create_session(_source())

    answer = _ask(store, home, platform="telegram", chat_id=CHAT, thread_id=TOPIC)

    assert answer["rotated"] is True
    assert answer["session_key"] == before.session_key
    assert answer["old_session_id"] == before.session_id
    assert answer["new_session_id"] != before.session_id
    # The routing index moved to the new session id…
    assert store.lookup_by_session_key(before.session_key).session_id == answer["new_session_id"]
    # …and the previous row is closed, with the reason the answer claims (not a guessed one).
    closed = store._db.get_session(before.session_id)
    assert closed["end_reason"] == answer["end_reason"] == "session_reset"
    assert closed["ended_at"] is not None


def test_other_channels_keep_their_session(tmp_path):
    store = _make_store(tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir()
    topic = store.get_or_create_session(_source())
    sibling = store.get_or_create_session(_source(thread_id="7"))

    answer = _ask(store, home, platform="telegram", chat_id=CHAT, thread_id=TOPIC)

    assert answer["session_key"] == topic.session_key
    assert store.lookup_by_session_key(sibling.session_key).session_id == sibling.session_id
    assert store._db.get_session(sibling.session_id)["ended_at"] is None


def test_explicit_session_key_rotates_without_naming_the_channel(tmp_path):
    store = _make_store(tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir()
    before = store.get_or_create_session(_source())

    answer = _ask(store, home, session_key=before.session_key)

    assert answer["rotated"] is True
    assert answer["session_key"] == before.session_key
    assert answer["new_session_id"] != before.session_id


def test_channel_fields_match_case_insensitively(tmp_path):
    """A platform name typed by an operator, or a chat label from a config file, is still the same
    channel — ids and platform names Hermes routes on are never case-significant."""
    store = _make_store(tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir()
    before = store.get_or_create_session(_source(chat_id="Hermes-HQ"))

    answer = _ask(store, home, platform="Telegram", chat_id="hermes-hq", thread_id=TOPIC)

    assert answer["rotated"] is True
    assert answer["new_session_id"] != before.session_id


def test_explicit_session_key_is_matched_verbatim(tmp_path):
    """A session key is an exact identifier, so it is never case-folded into some other key: an
    unknown one answers ``no_session`` instead of rotating a session nobody named."""
    store = _make_store(tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir()
    before = store.get_or_create_session(_source(chat_id="Hermes-HQ"))

    answer = _ask(store, home, session_key=before.session_key.upper())

    assert answer == {"rotated": False, "reason": "no_session",
                      "session_key": before.session_key.upper()}
    assert store.lookup_by_session_key(before.session_key).session_id == before.session_id


# ---------------------------------------------------------------------------
# What a caller fires before the channel exists, and when the identity is not unique
# ---------------------------------------------------------------------------

def test_unknown_channel_rotates_nothing_and_says_so(tmp_path):
    store = _make_store(tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir()
    seen = store.get_or_create_session(_source())

    answer = _ask(store, home, platform="telegram", chat_id="-1009999999999", thread_id=TOPIC)

    assert answer == {"rotated": False, "reason": "no_session",
                      "channel": {"platform": "telegram", "chat_id": "-1009999999999",
                                  "thread_id": TOPIC}}
    # No session was invented for it, and the channel that did exist is untouched.
    assert [e.session_key for e in store.list_sessions()] == [seen.session_key]
    assert store.lookup_by_session_key(seen.session_key).session_id == seen.session_id


def test_participant_isolated_chat_is_ambiguous_until_a_participant_is_named(tmp_path):
    store = _make_store(tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir()
    # group_sessions_per_user is on by default: outside a thread one chat has one session PER user.
    alice = store.get_or_create_session(_source(thread_id=None, user_id="111"))
    bob = store.get_or_create_session(_source(thread_id=None, user_id="222"))
    assert alice.session_key != bob.session_key

    ambiguous = _ask(store, home, platform="telegram", chat_id=CHAT)
    assert ambiguous["rotated"] is False
    assert ambiguous["reason"] == "ambiguous"
    assert ambiguous["candidates"] == sorted([alice.session_key, bob.session_key])

    named = _ask(store, home, platform="telegram", chat_id=CHAT, user_id="222")
    assert named["rotated"] is True
    assert named["session_key"] == bob.session_key
    # The one that was NOT named still holds its session id.
    assert store.lookup_by_session_key(alice.session_key).session_id == alice.session_id


def test_profile_selects_between_two_profiles_serving_one_chat(tmp_path):
    store = _make_store(tmp_path, multiplex=True)
    home = tmp_path / ".hermes"
    home.mkdir()
    dev = store.get_or_create_session(_source(profile="dev"))
    ops = store.get_or_create_session(_source(profile="ops"))
    assert dev.session_key != ops.session_key

    answer = _ask(store, home, platform="telegram", chat_id=CHAT, thread_id=TOPIC, profile="dev")

    assert answer["rotated"] is True
    assert answer["session_key"] == dev.session_key
    assert store.lookup_by_session_key(ops.session_key).session_id == ops.session_id


def test_missing_channel_names_the_fix(tmp_path):
    store = _make_store(tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir()
    store.get_or_create_session(_source())

    answer = _ask(store, home, platform="telegram")

    assert answer["rotated"] is False
    assert "chat_id" in answer["error"] and "session_key" in answer["error"]


# ---------------------------------------------------------------------------
# Discovery + degradation (no gateway / gateway without a store)
# ---------------------------------------------------------------------------

def test_client_returns_none_without_a_gateway(tmp_path):
    """No socket answering → None, the caller's signal to fall back to asking for /new."""
    assert rotate_gateway_session(tmp_path / ".hermes", platform="telegram", chat_id=CHAT) is None


def test_unknown_verb_lists_rotate_session(tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    server = GatewayControlServer(
        home, verb_handlers={"rotate-session": lambda params: {"echo": params}})
    response = json.loads(server.handle_request_line(json.dumps({"verb": "nope"}).encode()).decode())
    assert response["ok"] is False
    assert "rotate-session" in response["supported_verbs"]


def test_gateway_without_a_session_store_says_so():
    handler = rotate_session_verb(object())
    assert handler({"platform": "telegram", "chat_id": CHAT}) == {
        "rotated": False, "error": "gateway has no session store"}
