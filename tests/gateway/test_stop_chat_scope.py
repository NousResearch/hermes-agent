"""Regression tests: /stop falls back to a running turn in the SAME chat when the
caller's exact session key and thread-sibling key both miss.

Semantics under test: "/stop" means "stop what's running in this chat". On an exact +
thread-sibling miss, an AUTHORIZED user's /stop interrupts the chat's running turns;
another chat, workspace, profile or thread is never touched.

Regression for #113738 (found via Slack's native stop button, gateway-gateway#286).
"""

import pytest

from agent.i18n import t
from gateway.run import GatewayRunner, _AGENT_PENDING_SENTINEL
from gateway.session import SessionSource, build_session_key
from gateway.platforms.base import Platform
from gateway.platforms.event import MessageEvent, MessageType


class _FakeAgent:
    pass


class _StoreEntry:
    def __init__(self, session_key):
        self.session_key = session_key


class _FakeStore:
    def __init__(self, session_key):
        self._key = session_key

    def get_or_create_session(self, source):
        return _StoreEntry(self._key)


def _slack_source(chat_type, chat_id, thread_id=None, user_id="U-alice", scope_id="T1"):
    return SessionSource(
        platform=Platform.SLACK, chat_type=chat_type, chat_id=chat_id,
        thread_id=thread_id, user_id=user_id, scope_id=scope_id,
    )


def _runner_with_run(running_keys, own_key, authorized=True):
    runner = object.__new__(GatewayRunner)
    if isinstance(running_keys, str):
        running_keys = [running_keys]
    runner._running_agents = (
        dict(running_keys) if isinstance(running_keys, dict)
        else dict.fromkeys(running_keys, _FakeAgent())
    )
    runner.session_store = _FakeStore(own_key)
    runner._is_user_authorized_for_source = lambda source, **kw: authorized
    runner.adapters = {}
    interrupted = []

    async def _fake_interrupt(session_key, source, *, interrupt_reason, invalidation_reason):
        interrupted.append((session_key, invalidation_reason))

    runner._interrupt_and_clear_session = _fake_interrupt
    return runner, interrupted


async def _stop(source, running_keys, authorized=True):
    """Drive one /stop through the real handler; return (interrupted, reply)."""
    own_key = build_session_key(source)
    runner, interrupted = _runner_with_run(running_keys, own_key, authorized=authorized)
    event = MessageEvent(text="/stop", message_type=MessageType.TEXT, source=source)
    result = await runner._handle_stop_command(event)
    return interrupted, result


@pytest.mark.asyncio
async def test_stop_from_thread_reaches_top_level_channel_run():
    # Running turn: triggered by a top-level channel message (the relay stamps the
    # message's own ts as thread_id; the chat_type slot stays "channel").
    running_key = build_session_key(_slack_source("channel", "C9", thread_id="170.100"))
    # The stop arrives from inside the reply thread → normalizes to "thread".
    stop_source = _slack_source("thread", "C9", thread_id="170.100")
    assert build_session_key(stop_source) != running_key  # the miss under test

    interrupted, result = await _stop(stop_source, running_key)

    assert interrupted == [(running_key, "stop_command_chat_scope")]
    assert result == t("gateway.stop.stopped")


@pytest.mark.asyncio
async def test_stop_with_thread_reaches_rolling_dm_run():
    # Rolling-DM config: the running session keys WITHOUT a thread slot.
    running_key = build_session_key(_slack_source("dm", "D1"))
    # The stop event carries the session thread → keys a different session.
    stop_source = _slack_source("dm", "D1", thread_id="170.100")
    assert build_session_key(stop_source) != running_key

    interrupted, result = await _stop(stop_source, running_key)

    assert interrupted == [(running_key, "stop_command_chat_scope")]
    assert result == t("gateway.stop.stopped")


@pytest.mark.asyncio
async def test_stop_reaches_peer_run_in_per_sender_group():
    # Intentional widening, same contract: with nothing running under the caller's own key,
    # an authorized /stop interrupts the chat's live turn even when a PEER started it — the
    # bot-triggered runaway of #113846, where a per-sender group key hid the executing turn.
    running_key = build_session_key(_slack_source("group", "C9", user_id="U-bob"))
    stop_source = _slack_source("group", "C9", user_id="U-alice")
    assert build_session_key(stop_source) != running_key

    interrupted, result = await _stop(stop_source, running_key)

    assert interrupted == [(running_key, "stop_command_chat_scope")]
    assert result == t("gateway.stop.stopped")


@pytest.mark.asyncio
async def test_stop_does_not_reach_a_different_thread_of_the_same_channel():
    # Another thread in the same channel is another conversation, while the caller's own thread
    # stays reachable — proven in the same call so a matcher that returns nothing cannot pass.
    same_thread = build_session_key(_slack_source("channel", "C9", thread_id="170.100"))
    other_thread = build_session_key(_slack_source("thread", "C9", thread_id="170.200"))
    stop_source = _slack_source("thread", "C9", thread_id="170.100")

    interrupted, result = await _stop(stop_source, [same_thread, other_thread])

    assert interrupted == [(same_thread, "stop_command_chat_scope")]
    assert result == t("gateway.stop.stopped")


@pytest.mark.asyncio
async def test_stop_does_not_reach_another_reply_thread_of_a_channel_keyed_run():
    # A top-level channel turn keeps chat_type "channel" with the relay-stamped reply-thread ts,
    # so the SAME boundary must apply to it: a stop inside thread .100 interrupts the run whose
    # reply thread is .100 and must not touch the one belonging to .200.
    same_reply_thread = build_session_key(_slack_source("channel", "C9", thread_id="170.100"))
    other_reply_thread = build_session_key(_slack_source("channel", "C9", thread_id="170.200"))
    stop_source = _slack_source("thread", "C9", thread_id="170.100")

    interrupted, result = await _stop(stop_source, [same_reply_thread, other_reply_thread])

    assert interrupted == [(same_reply_thread, "stop_command_chat_scope")]
    assert result == t("gateway.stop.stopped")


@pytest.mark.asyncio
async def test_stop_interrupts_every_run_of_the_callers_thread():
    # A per-user thread sibling AND the same-thread channel-keyed run (the #286 shape this fix
    # exists for) are both live: the reply must not claim "Stopped" while one of them keeps going.
    sibling = build_session_key(
        _slack_source("thread", "C9", thread_id="170.100", user_id="U-bob"),
        thread_sessions_per_user=True,
    )
    same_thread_channel_run = build_session_key(_slack_source("channel", "C9", thread_id="170.100"))
    stop_source = _slack_source("thread", "C9", thread_id="170.100")

    interrupted, result = await _stop(stop_source, [sibling, same_thread_channel_run])

    assert sorted(key for key, _ in interrupted) == sorted([sibling, same_thread_channel_run])
    assert result == t("gateway.stop.stopped")


@pytest.mark.asyncio
async def test_a_lone_thread_sibling_keeps_its_own_invalidation_reason():
    # With nothing else live in the chat, the stop IS a thread-sibling stop: hook consumers must
    # still see that label rather than the wider chat-scope one.
    sibling = build_session_key(
        _slack_source("thread", "C9", thread_id="170.100", user_id="U-bob"),
        thread_sessions_per_user=True,
    )
    stop_source = _slack_source("thread", "C9", thread_id="170.100")

    interrupted, result = await _stop(stop_source, sibling)

    assert interrupted == [(sibling, "stop_command_thread_sibling")]
    assert result == t("gateway.stop.stopped")


@pytest.mark.asyncio
async def test_stop_matches_a_canonicalised_whatsapp_dm_chat_id():
    # build_session_key canonicalises a WhatsApp DM chat id, so the fallback must match the same
    # text: a stop whose source carries the raw JID has to reach the canonicalised run.
    running_key = build_session_key(
        SessionSource(platform=Platform.WHATSAPP, chat_type="dm", chat_id="1234567890@s.whatsapp.net",
                      user_id="1234567890@s.whatsapp.net")
    )
    stop_source = SessionSource(platform=Platform.WHATSAPP, chat_type="dm",
                                chat_id="1234567890:7@s.whatsapp.net",
                                user_id="1234567890:7@s.whatsapp.net", thread_id="t1")
    assert build_session_key(stop_source) != running_key

    interrupted, result = await _stop(stop_source, running_key)

    assert interrupted == [(running_key, "stop_command_chat_scope")]
    assert result == t("gateway.stop.stopped")


@pytest.mark.asyncio
async def test_stop_in_a_dm_does_not_reach_a_group_run_that_ends_in_the_same_user_id():
    # Non-Slack DMs key chat_id as the USER id (Telegram), and a per-sender group key ends with
    # that same user id — the group run is a different chat and must stay untouched, while the
    # DM's own threaded run in the same chat is still reached.
    same_chat = build_session_key(
        SessionSource(platform=Platform.TELEGRAM, chat_type="dm", chat_id="777",
                      thread_id="42", user_id="777")
    )
    group_run = build_session_key(
        SessionSource(platform=Platform.TELEGRAM, chat_type="group", chat_id="-100123",
                      user_id="777")
    )
    dm_stop = SessionSource(platform=Platform.TELEGRAM, chat_type="dm", chat_id="777",
                            user_id="777")

    interrupted, result = await _stop(dm_stop, [same_chat, group_run])

    assert interrupted == [(same_chat, "stop_command_chat_scope")]
    assert result == t("gateway.stop.stopped")


@pytest.mark.asyncio
async def test_chat_scope_fallback_is_authorization_gated():
    running_key = build_session_key(_slack_source("channel", "C9", thread_id="170.100"))
    stop_source = _slack_source("thread", "C9", thread_id="170.100")

    interrupted, result = await _stop(stop_source, running_key, authorized=False)

    assert interrupted == []
    assert result == t("gateway.stop.no_active")


@pytest.mark.asyncio
async def test_pending_sentinel_is_never_interrupted():
    # A session still being set up has no agent turn yet; /stop must not claim it stopped.
    pending = build_session_key(_slack_source("channel", "C9", thread_id="170.100"))
    stop_source = _slack_source("thread", "C9", thread_id="170.100")

    interrupted, result = await _stop(stop_source, {pending: _AGENT_PENDING_SENTINEL})

    assert interrupted == []
    assert result == t("gateway.stop.no_active")


@pytest.mark.asyncio
async def test_stop_reaches_a_chat_whose_id_contains_a_colon():
    # Matrix ids carry ":" (`!room:example.org`), so the chat id must be matched as text after
    # the fixed-shape head — slot-splitting the key would silently disable both fallbacks.
    room = SessionSource(platform=Platform.MATRIX, chat_type="channel", chat_id="!room:example.org",
                         thread_id="$t1", user_id="@alice:example.org")
    running_key = build_session_key(room)
    stop_source = SessionSource(platform=Platform.MATRIX, chat_type="thread",
                                chat_id="!room:example.org", thread_id="$t1",
                                user_id="@alice:example.org")
    assert build_session_key(stop_source) != running_key

    interrupted, result = await _stop(stop_source, running_key)

    assert interrupted == [(running_key, "stop_command_chat_scope")]
    assert result == t("gateway.stop.stopped")


@pytest.mark.asyncio
@pytest.mark.parametrize("other_chat_id", ["C-other", "C90"])
async def test_chat_scope_fallback_interrupts_only_the_callers_chat(other_chat_id):
    # A same-chat run AND a foreign run are live: exactly the caller's chat is stopped
    # ("C90" also proves a chat id that merely starts with "C9" is not folded in).
    same_chat = build_session_key(_slack_source("channel", "C9", thread_id="170.100"))
    foreign = build_session_key(_slack_source("channel", other_chat_id, thread_id="170.100"))
    stop_source = _slack_source("thread", "C9", thread_id="170.100")

    interrupted, result = await _stop(stop_source, [same_chat, foreign])

    assert interrupted == [(same_chat, "stop_command_chat_scope")]
    assert result == t("gateway.stop.stopped")


@pytest.mark.asyncio
async def test_chat_scope_fallback_does_not_fold_in_a_prefix_of_the_chat_id():
    # A chat id that merely STARTS with the caller's ("C9" vs "C90") is a different chat. The run
    # ends at its chat id, so only the id boundary separates the two — the tail case cannot show it.
    foreign = build_session_key(_slack_source("channel", "C90"))
    stop_source = _slack_source("channel", "C9")

    interrupted, result = await _stop(stop_source, foreign)

    assert interrupted == []
    assert result == t("gateway.stop.no_active")


@pytest.mark.asyncio
async def test_chat_scope_fallback_does_not_cross_workspace_scope_or_profile():
    # A same-chat run stays live beside the foreign ones, so a parser that matched NOTHING
    # cannot pass this test.
    same_chat = build_session_key(_slack_source("channel", "C9", thread_id="170.100"))
    running_keys = [
        same_chat,
        # Same chat_id, different Slack workspace (scope_id).
        build_session_key(_slack_source("channel", "C9", thread_id="170.100", scope_id="T2")),
        # Same chat_id, different profile namespace.
        build_session_key(_slack_source("channel", "C9", thread_id="170.100"), profile="work"),
    ]
    stop_source = _slack_source("thread", "C9", thread_id="170.100")

    interrupted, result = await _stop(stop_source, running_keys)

    assert interrupted == [(same_chat, "stop_command_chat_scope")]
    assert result == t("gateway.stop.stopped")
