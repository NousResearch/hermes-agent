"""A human message steered into a machinery turn must not be answered with silence (#149679).

``_fold_into_running_turn`` has always carried a follow-up's ``reply_expected`` onto the running
turn, but ``silence_allowed`` answered True for every machinery kind whatever that flag said, and
DM adapters leave the flag unset — so a bare ``[SILENT]`` ended the turn and the user's message went
unanswered. An addressed follow-up (a DM is addressed by construction) outranks the turn's kind.
"""

from types import SimpleNamespace

import pytest

from gateway.platforms.base import MessageEvent
from gateway.response_filters import is_intentional_silence_response, silence_allowed
from gateway.run import GatewayRunner
from gateway.run_turn import GatewayTurnMixin
from gateway.session import Platform, SessionSource
from gateway.session_state import SessionState, TurnState
from gateway.turn_context import TurnContext

NOTICE_KIND = "internal_notification"


def _source(chat_type="dm", is_bot=False):
    return SessionSource(platform=Platform.TELEGRAM, chat_id="c1", chat_type=chat_type,
                         user_id="u1", is_bot=is_bot)


def _notice():
    return MessageEvent(text="[IMPORTANT: Background process proc_1 completed]", internal=True,
                        source=_source("group"))


def _follow_up(text, *, chat_type="dm", is_bot=False, reply_expected=None):
    return MessageEvent(text=text, source=_source(chat_type, is_bot), reply_expected=reply_expected)


def _fold(follow_up, turn_event):
    """Drive the real fold against one running turn; returns the turn's event and its context."""
    agent = SimpleNamespace()
    ctx = TurnContext(reply_expected=turn_event.reply_expected)
    runner = object.__new__(GatewayRunner)
    runner.__dict__["_sessions"] = {
        "k": SessionState(turn=TurnState(agent=agent, event=turn_event, ctx=ctx)),
    }
    GatewayRunner._fold_into_running_turn(runner, agent, "k", follow_up)
    return turn_event, ctx


class _Shaper(GatewayTurnMixin):
    """The smallest runner that can shape a turn's final response."""

    def _silence_allowed_for_source(self, source, display_kind, reply_expected, agent_result):
        return silence_allowed(display_kind, reply_expected)

    def __init__(self):
        self.async_session_store = SimpleNamespace(clear_resume_pending=self._noop)

    async def _noop(self, *_a, **_k):
        return None

    async def _clear_restart_failure_count(self, *_a, **_k):
        return None


async def _shape(reply_expected):
    return await _Shaper()._hmwa_shape_agent_response(
        {"final_response": "[SILENT]"}, _source(), history=[],
        session_entry=SimpleNamespace(session_id="s"), session_key="k", _quick_key="k",
        run_generation=0, _run_start_session_id="s", _platform_name="telegram",
        _msg_start_time=0.0, persist_user_display_kind=NOTICE_KIND, reply_expected=reply_expected)


@pytest.mark.asyncio
async def test_addressed_follow_up_gets_the_recovery_reply_on_a_machinery_turn():
    response, silent, _ = await _shape(True)
    assert silent is False
    assert response and not is_intentional_silence_response(response)


@pytest.mark.asyncio
async def test_unaddressed_follow_up_may_still_end_a_machinery_turn_silently():
    _, silent, _ = await _shape(None)
    assert silent is True


def test_dm_follow_up_is_addressed_though_its_adapter_leaves_the_flag_unset():
    turn_event, ctx = _fold(_follow_up("did the build pass?"), _notice())
    assert turn_event.reply_expected is True
    assert ctx.reply_expected is True


def test_group_follow_up_without_a_mention_stays_unaddressed():
    turn_event, _ = _fold(_follow_up("any news?", chat_type="group"), _notice())
    assert turn_event.reply_expected is None


def test_bot_peer_follow_up_never_creates_a_reply_obligation():
    turn_event, _ = _fold(_follow_up("[peer] ping", is_bot=True), _notice())
    assert turn_event.reply_expected is None


def test_addressed_bot_peer_does_not_claim_a_human_reply_obligation():
    # An adapter may stamp reply_expected=True on a peer's explicit mention; the machinery turn must
    # still end in silence instead of sending the human-only fallback (#137027).
    peer = _follow_up("status?", chat_type="group", is_bot=True, reply_expected=True)
    turn_event, ctx = _fold(peer, _notice())
    assert turn_event.reply_expected is None
    assert ctx.reply_expected is None


def test_internal_follow_up_never_creates_a_reply_obligation():
    internal = MessageEvent(text="[tick]", internal=True, source=_source("dm"))
    turn_event, _ = _fold(internal, _notice())
    assert turn_event.reply_expected is None


def test_a_stamped_internal_follow_up_does_not_claim_the_obligation_either():
    internal = MessageEvent(text="[tick]", internal=True, source=_source("dm"), reply_expected=True)
    turn_event, _ = _fold(internal, _notice())
    assert turn_event.reply_expected is None


def test_addressed_group_follow_up_is_still_carried_over():
    turn_event, _ = _fold(_follow_up("@bot did it pass?", chat_type="group", reply_expected=True), _notice())
    assert turn_event.reply_expected is True


def test_an_explicit_room_opt_in_still_excuses_silence_on_a_machinery_turn():
    assert silence_allowed(NOTICE_KIND, True, allow_silence=True, agent_result={"completed": True}) is True
