"""Per-bot stop: ``/stop @bot`` interrupts only that bot's run in the same chat.

Issue #123928 layer A: in a multi-bot group chat a bare ``/stop`` deliberately
stops EVERY run in the room (``_chat_scoped_run_keys``). ``/stop @botname``
must stop only the named bot's run and leave other bots' in-flight work alone;
targeting a bot with nothing running must stop nothing.
"""

import pytest

from agent.i18n import t
from gateway.run import GatewayRunner
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


def _slack_group(user_id, profile=None):
    source = SessionSource(
        platform=Platform.SLACK, chat_type="group", chat_id="C9",
        user_id=user_id, scope_id="T1",
    )
    key = build_session_key(source, profile=profile) if profile else build_session_key(source)
    return source, key


def _runner_with_runs(running_keys, own_key, authorized=True):
    runner = object.__new__(GatewayRunner)
    runner._running_agents = dict.fromkeys(running_keys, _FakeAgent())
    runner.session_store = _FakeStore(own_key)
    runner._is_user_authorized_for_source = lambda source, **kw: authorized
    runner.adapters = {}
    interrupted = []

    async def _fake_interrupt(session_key, source, *, interrupt_reason, invalidation_reason,
                              release_running_state=True):
        interrupted.append((session_key, invalidation_reason))

    runner._interrupt_and_clear_session = _fake_interrupt
    return runner, interrupted


async def _stop(text, stop_source, running_keys, authorized=True):
    own_key = build_session_key(stop_source)
    runner, interrupted = _runner_with_runs(running_keys, own_key, authorized=authorized)
    event = MessageEvent(text=text, message_type=MessageType.TEXT, source=stop_source)
    result = await runner._handle_stop_command(event)
    return interrupted, result


@pytest.mark.asyncio
async def test_stop_at_bot_only_stops_that_bot():
    _, main_key = _slack_group("U-bob")
    _, work_key = _slack_group("U-bob", profile="work")
    assert main_key != work_key
    stop_source, _ = _slack_group("U-alice")

    interrupted, result = await _stop("/stop @work", stop_source, [main_key, work_key])

    assert interrupted == [(work_key, "stop_command_targeted")]
    assert result == t("gateway.stop.stopped")


@pytest.mark.asyncio
async def test_stop_at_idle_bot_stops_nothing():
    _, main_key = _slack_group("U-bob")
    _, work_key = _slack_group("U-bob", profile="work")
    stop_source, _ = _slack_group("U-alice")

    interrupted, result = await _stop("/stop @idle", stop_source, [main_key, work_key])

    assert interrupted == []
    assert result == t("gateway.stop.no_active")


@pytest.mark.asyncio
async def test_bare_stop_keeps_own_profile_scope():
    # Untargeted /stop keeps its historical scope: the caller's own profile namespace only
    # (cross-profile isolation is the established contract — see
    # test_stop_thread_sibling.py::test_sibling_does_not_cross_profiles). Reaching another bot
    # takes an explicit, authorized `/stop @bot`.
    _, main_key = _slack_group("U-bob")
    _, work_key = _slack_group("U-bob", profile="work")
    stop_source, _ = _slack_group("U-alice")

    interrupted, result = await _stop("/stop", stop_source, [main_key, work_key])

    assert [k for k, _ in interrupted] == [main_key]
    assert result == t("gateway.stop.stopped")
