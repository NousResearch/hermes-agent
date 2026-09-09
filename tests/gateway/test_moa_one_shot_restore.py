"""One-shot model overrides (``/moa <prompt>``, ``/model --once``) are TURN-OWNED.

The snapshot to put back is adopted by the turn when it claims the slot
(``_claim_running_agent_state``) and applied exactly once when that slot is
released (``_release_running_agent_state``), whoever releases it: the turn's own
finalizer on every exit path (success, exception, interrupt) or an evictor.

Bugs guarded:

- the restore used to live in the ``try`` block, so a raising turn skipped it and
  the MoA override leaked permanently (every later message fanned out through MoA);
- the restore used to run from the evicted turn's late ``finally`` against SHARED
  conversation state: a replacement turn B that had already claimed the slot and
  installed its own override got it overwritten (and its cached agent evicted) by
  turn A's stale snapshot, so B silently ran on the wrong model (#106966 review).

These drive the real producers and lifecycle helpers, not a re-implementation.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource

KEY = "agent:main:telegram:dm:999"
OPENROUTER = {"provider": "openrouter", "model": "gpt-4"}


def _make_runner():
    """Bare GatewayRunner; cache evictions are recorded instead of executed."""
    runner = object.__new__(GatewayRunner)
    runner._session_model_overrides = {}
    runner._pending_one_turn_model_restores = {}
    runner.cache_evictions = []
    runner._evict_cached_agent = lambda key: runner.cache_evictions.append(key)
    return runner


def _moa_event(text: str) -> MessageEvent:
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="999", chat_type="dm", user_id="999")
    return MessageEvent(text=text, message_type=MessageType.TEXT, source=source)


async def _stage_moa(runner, text="/moa do this") -> None:
    """Run the real ``/moa <prompt>`` one-shot producer (idle dispatch, before the claim)."""
    event = _moa_event(text)
    handled, reply = await runner._hm_cmd_moa(event, event.source, KEY)
    assert (handled, reply) == (False, None), reply
    assert runner._session_model_overrides[KEY]["provider"] == "moa"


def _override(runner):
    return runner._session_model_overrides.get(KEY)


class TestOneShotRestoreOnRelease:
    @pytest.mark.asyncio
    async def test_restore_runs_when_the_slot_is_released_even_if_the_turn_raised(self):
        """Mirrors the finalizer: the release is what the ``finally`` runs, so a raising
        turn still reverts the override."""
        runner = _make_runner()
        runner._session_model_overrides[KEY] = dict(OPENROUTER)
        await _stage_moa(runner)
        gen = runner._claim_running_agent_state(KEY, MagicMock())

        with pytest.raises(RuntimeError):
            try:
                raise RuntimeError("provider error mid-turn")
            finally:
                runner._release_running_agent_state(KEY, owner_generation=gen)

        assert _override(runner) == OPENROUTER
        assert runner._pending_one_turn_model_restores.get(KEY) is None

    def test_model_once_snapshot_is_adopted_by_the_claiming_turn(self):
        """``/model --once`` stages its snapshot for the NEXT turn; that turn owns it."""
        runner = _make_runner()
        runner._session_model_overrides[KEY] = {"provider": "openai", "model": "gpt-5.5"}
        runner._pending_one_turn_model_restores[KEY] = {"had_override": False, "override": None}

        gen = runner._claim_running_agent_state(KEY, MagicMock())

        assert runner._peek_session_state(KEY).turn.model_restore == {
            "had_override": False, "override": None,
        }
        assert KEY not in runner._pending_one_turn_model_restores, "still pending: a second turn could adopt it too"
        assert runner._release_running_agent_state(KEY, owner_generation=gen) is True
        assert _override(runner) is None

    @pytest.mark.asyncio
    async def test_earliest_pending_snapshot_wins_when_moa_follows_model_once(self):
        """``/model --once X`` then ``/moa <prompt>``: both one-shots end with this turn, so
        the turn puts back the override from BEFORE ``--once``, not X."""
        runner = _make_runner()
        runner._session_model_overrides[KEY] = {"provider": "openai", "model": "gpt-5.5"}  # X
        runner._pending_one_turn_model_restores[KEY] = {"had_override": True, "override": dict(OPENROUTER)}
        await _stage_moa(runner)
        gen = runner._claim_running_agent_state(KEY, MagicMock())

        runner._release_running_agent_state(KEY, owner_generation=gen)

        assert _override(runner) == OPENROUTER


class TestOneShotRestoreVsConversationBoundary:
    """The prior a one-shot puts back belongs to the conversation it was taken in. A boundary
    reached MID-TURN (compression exhaustion auto-reset) closes that conversation while the
    turn still owns the slot: the snapshot must die with it, or the release writes the old
    conversation's override into the fresh session (#106966 review)."""

    @pytest.mark.asyncio
    async def test_model_once_restore_does_not_cross_a_compression_exhaustion_reset(self):
        runner = _make_runner()
        runner._session_model_overrides[KEY] = {"provider": "openai", "model": "gpt-5.5"}  # --once model
        runner._pending_one_turn_model_restores[KEY] = {"had_override": True, "override": dict(OPENROUTER)}
        gen = runner._claim_running_agent_state(KEY, MagicMock())
        assert runner._peek_session_state(KEY).turn.model_restore is not None

        # Mid-turn the context can no longer be compressed: the real auto-reset rotates the
        # durable session and clears the conversation scope (fresh session, no override).
        runner.session_store = SimpleNamespace()
        runner._async_session_store = SimpleNamespace(
            _store=runner.session_store, reset_session=AsyncMock(return_value=None),
        )
        response, _entry = await runner._hmwa_compression_exhaustion_reset(
            {"compression_exhausted": True}, "reply", SimpleNamespace(session_id="old-sid"), KEY, None,
        )
        assert "auto-reset" in response
        assert _override(runner) is None

        # The turn ends and releases its slot: nothing from the closed conversation comes back.
        assert runner._release_running_agent_state(KEY, owner_generation=gen) is True
        assert _override(runner) is None, "the old conversation's override was restored into the fresh session"

    def test_moa_restore_is_dropped_at_a_conversation_boundary(self):
        """Same contract through the boundary funnel every reset path shares (/new, /resume,
        auto-reset): the adopted snapshot does not survive it."""
        runner = _make_runner()
        runner._session_model_overrides[KEY] = dict(OPENROUTER)
        runner._pending_one_turn_model_restores[KEY] = {"had_override": True, "override": dict(OPENROUTER)}
        gen = runner._claim_running_agent_state(KEY, MagicMock())

        runner._clear_conversation_scope(KEY, reason="test_boundary")

        assert runner._peek_session_state(KEY).turn.model_restore is None
        assert runner._release_running_agent_state(KEY, owner_generation=gen) is True
        assert _override(runner) is None


class TestOneShotRestoreVsReplacementTurn:
    """#106966 review interleaving: turn A (``/moa``) is evicted, replacement turn B claims
    the slot and installs its own override, then A's finalizer runs."""

    @pytest.mark.asyncio
    async def test_evicted_turn_restores_before_the_replacement_and_never_clobbers_it(self):
        runner = _make_runner()
        runner._session_model_overrides[KEY] = dict(OPENROUTER)
        await _stage_moa(runner, "/moa turn A")
        gen_a = runner._claim_running_agent_state(KEY, MagicMock())
        runner._session_state(KEY).turn.agent = MagicMock(name="agent-a")
        assert _override(runner)["provider"] == "moa"

        # Message 2 finds A's durable row reaped: A's one-shot ends HERE, with its slot.
        runner._hm_evict_running_agent(KEY, "reaped_session_eviction")
        assert _override(runner) == OPENROUTER, "the replacement must start from the prior override"

        # B is a /moa one-shot too: it snapshots the (restored) prior and installs its own MoA.
        await _stage_moa(runner, "/moa message 2")
        gen_b = runner._claim_running_agent_state(KEY, MagicMock())
        assert runner._peek_session_state(KEY).turn.model_restore == {"had_override": True, "override": OPENROUTER}
        evictions_before_a_finalizer = len(runner.cache_evictions)

        # A unwinds late and runs its finalizer: nothing of B's may change.
        assert runner._release_running_agent_state(KEY, owner_generation=gen_a) is False
        assert _override(runner)["provider"] == "moa", "A's stale snapshot overwrote B's override"
        assert len(runner.cache_evictions) == evictions_before_a_finalizer, "A's finalizer evicted B's cached agent"

        # B's own finalizer puts the prior back exactly once.
        assert runner._release_running_agent_state(KEY, owner_generation=gen_b) is True
        assert _override(runner) == OPENROUTER

    def test_model_once_turn_evicted_then_replaced_keeps_the_replacement_override(self):
        runner = _make_runner()
        runner._session_model_overrides[KEY] = {"provider": "openai", "model": "gpt-5.5"}  # --once model
        runner._pending_one_turn_model_restores[KEY] = {"had_override": True, "override": dict(OPENROUTER)}
        gen_a = runner._claim_running_agent_state(KEY, MagicMock())
        runner._session_state(KEY).turn.agent = MagicMock(name="agent-a")

        runner._hm_evict_running_agent(KEY, "stale_running_agent_eviction")
        assert _override(runner) == OPENROUTER

        # Replacement B switches the session model for good (plain /model, no --once).
        runner._session_model_overrides[KEY] = {"provider": "anthropic", "model": "claude"}
        gen_b = runner._claim_running_agent_state(KEY, MagicMock())

        assert runner._release_running_agent_state(KEY, owner_generation=gen_a) is False
        assert _override(runner) == {"provider": "anthropic", "model": "claude"}
        assert runner._release_running_agent_state(KEY, owner_generation=gen_b) is True
        assert _override(runner) == {"provider": "anthropic", "model": "claude"}, "B had no one-shot: nothing to restore"
