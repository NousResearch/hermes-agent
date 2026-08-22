"""Tests for the post-generation ``ContextEngine.enforce_response()`` hook.

``enforce_response()`` is the *enforcement* verb — a different lifecycle point
from ``select_context()`` (pre-request selection) and ``on_turn_complete()``
(post-turn observation). After the tool loop produces a TEXT-final assistant
answer, the host lets the active engine audit it against the engine's verbatim
record and either accept it, swap in a safe replacement, or (when the engine
advertises the capability) ask for a bounded number of regenerations. It is
additive and no-op by default; the host call site (``_maybe_enforce_response``)
is fail-open: a missing hook, the base no-op, an exception, or an unrecognized
return value must ship the model's answer unchanged and never break a turn.

These drive the real host helper against a real opt-in ``ContextEngine``
subclass and assert on the OBSERVABLE OUTCOME (the returned/shipped text and the
protocol the engine actually sees), rather than mirroring the implementation.
Re-scopes PR #50053: the selection/observation surface already landed via
#70458; enforce_response()+capabilities() is the surviving, different-lifecycle
scope, now wired into turn finalization.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import MagicMock

from agent.context_engine import ContextEngine
from agent.conversation_loop import _maybe_enforce_response


class _MinimalEngine(ContextEngine):
    """Concrete engine implementing only the abstract methods (inherits the
    ABC ``enforce_response`` / ``capabilities`` no-ops unless a subclass
    overrides them)."""

    @property
    def name(self) -> str:
        return "minimal"

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        pass

    def should_compress(self, prompt_tokens: int = None) -> bool:
        return False

    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: int = None,
        focus_topic: str = None,
    ) -> List[Dict[str, Any]]:
        return messages


def _agent_with(engine) -> Any:
    """A real (non-Mock) agent-like object exposing exactly what the host
    helper reads: ``context_compressor``, ``session_id``, ``model``."""
    return SimpleNamespace(
        context_compressor=engine,
        session_id="test-session",
        model="test-model",
    )


ANSWER = "The capital of France is Paris."
HISTORY = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "capital of France?"},
    {"role": "assistant", "content": ANSWER},
]


# -- ABC default / built-in engines: pay nothing, ship unchanged -----------


def test_base_noop_engine_ships_answer_unchanged_and_is_not_invoked():
    """An engine that merely inherits the ABC default must skip the hook.

    ``hasattr`` alone cannot distinguish "inherits the no-op default" from
    "implements the hook" because the ABC defines ``enforce_response`` on every
    engine. The host identity-checks the bound method against
    ``ContextEngine.enforce_response`` and short-circuits WITHOUT calling it —
    pinned here by patching the base method to explode: it must never run.
    """
    from unittest.mock import patch as _patch

    def _explode(self, answer, messages, **kwargs):
        raise AssertionError("base enforce_response must not be invoked")

    engine = _MinimalEngine()  # inherits the ABC default
    agent = _agent_with(engine)
    logger = MagicMock()
    with _patch.object(ContextEngine, "enforce_response", _explode):
        out = _maybe_enforce_response(agent, ANSWER, HISTORY, logger=logger)
    assert out == ANSWER
    assert not logger.warning.called


# -- replace: the answer is actually swapped -------------------------------


def test_replace_actually_replaces_the_final_response():
    """A real opt-in engine returning ``replace`` swaps the shipped text.

    This is the core wiring proof: the host does not just call the hook, it
    HONORS the verdict — the returned text is the engine's replacement, not the
    model's original answer.
    """
    SAFE = "I can't verify that against the sources on hand."

    class _RefusingEngine(_MinimalEngine):
        def enforce_response(self, answer, messages, model="", **kwargs):
            # A grounding engine that judged the answer unsupported and refuses.
            return {"action": "replace", "text": SAFE}

    agent = _agent_with(_RefusingEngine())
    out = _maybe_enforce_response(agent, ANSWER, HISTORY, logger=MagicMock())
    assert out == SAFE
    assert out != ANSWER


def test_replace_with_empty_text_falls_open_to_original():
    """A malformed ``replace`` (missing/empty text) must not ship an empty
    answer; it falls open to the model's original text."""

    class _BadReplaceEngine(_MinimalEngine):
        def enforce_response(self, answer, messages, model="", **kwargs):
            return {"action": "replace", "text": ""}

    agent = _agent_with(_BadReplaceEngine())
    out = _maybe_enforce_response(agent, ANSWER, HISTORY, logger=MagicMock())
    assert out == ANSWER


# -- accept + unknown actions ----------------------------------------------


def test_accept_ships_the_answer_unchanged():
    class _AcceptEngine(_MinimalEngine):
        def enforce_response(self, answer, messages, model="", **kwargs):
            return {"action": "accept"}

    agent = _agent_with(_AcceptEngine())
    out = _maybe_enforce_response(agent, ANSWER, HISTORY, logger=MagicMock())
    assert out == ANSWER


def test_unknown_action_falls_open_to_the_answer():
    class _WeirdEngine(_MinimalEngine):
        def enforce_response(self, answer, messages, model="", **kwargs):
            return {"action": "quarantine"}  # not part of the contract

    agent = _agent_with(_WeirdEngine())
    out = _maybe_enforce_response(agent, ANSWER, HISTORY, logger=MagicMock())
    assert out == ANSWER


# -- fail-open on exception -------------------------------------------------


def test_engine_raising_never_breaks_the_turn():
    """Any exception inside the hook ships the model's answer unchanged and is
    swallowed with a warning (fail-open)."""

    class _BrokenEngine(_MinimalEngine):
        def enforce_response(self, answer, messages, model="", **kwargs):
            raise RuntimeError("engine blew up")

    logger = MagicMock()
    agent = _agent_with(_BrokenEngine())
    out = _maybe_enforce_response(agent, ANSWER, HISTORY, logger=logger)
    assert out == ANSWER
    assert logger.warning.called


# -- the engine cannot corrupt persisted history ---------------------------


def test_engine_mutating_transcript_cannot_corrupt_persisted_history():
    """The hook receives structural clones of the transcript, so an engine that
    writes into the messages it is handed cannot alter the persisted history."""
    history_snapshot = [dict(m) for m in HISTORY]

    class _MutatingEngine(_MinimalEngine):
        def enforce_response(self, answer, messages, model="", **kwargs):
            # Misbehaving engine: tamper with its copy in place.
            if messages:
                messages[0]["content"] = "TAMPERED"
                messages.append({"role": "user", "content": "INJECTED"})
            return {"action": "accept"}

    agent = _agent_with(_MutatingEngine())
    _maybe_enforce_response(agent, ANSWER, HISTORY, logger=MagicMock())
    assert HISTORY == history_snapshot


# -- regenerate: capability-gated + bounded + final=True on last attempt ----


def test_regenerate_without_capability_is_not_honored_single_final_call():
    """An engine that returns ``regenerate`` but does NOT advertise the
    capability gets exactly ONE call, made with ``final=True`` (no loop), and
    the current answer ships as-is."""
    seen = []

    class _WantsRegenNoCap(_MinimalEngine):
        def enforce_response(self, answer, messages, model="", *, final=False, **kwargs):
            seen.append(final)
            return {"action": "regenerate", "message": "add a citation"}

    agent = _agent_with(_WantsRegenNoCap())
    out = _maybe_enforce_response(agent, ANSWER, HISTORY, logger=MagicMock())
    assert out == ANSWER
    assert seen == [True]  # single attempt, and it is the final one


def test_regenerate_with_capability_is_bounded_and_ends_final_true():
    """A capability-advertising engine that always asks to regenerate is called
    a bounded number of times, the LAST call carries ``final=True`` so the
    engine is forced to a terminal decision, and the loop never runs away."""
    finals = []

    class _AlwaysRegen(_MinimalEngine):
        def capabilities(self) -> Dict[str, bool]:
            return {"enforce_response": True}

        def enforce_response(self, answer, messages, model="", *, final=False, **kwargs):
            finals.append(final)
            return {"action": "regenerate", "message": "cite a source"}

    agent = _agent_with(_AlwaysRegen())
    out = _maybe_enforce_response(agent, ANSWER, HISTORY, logger=MagicMock())
    # Bounded: exactly one call per allowed attempt, terminating.
    assert len(finals) >= 2
    # Every attempt but the last is final=False; the last is final=True so the
    # engine refuses instead of looping forever.
    assert finals[-1] is True
    assert all(f is False for f in finals[:-1])
    # A stubborn regenerate that never resolves ships the original answer.
    assert out == ANSWER


def test_regenerate_then_replace_on_final_ships_replacement():
    """A capable engine may regenerate a bounded number of times, then, when
    told ``final=True``, refuse with a safe replacement — which the host ships.
    This proves the ``final=True`` semantics are actually honored end-to-end."""
    SAFE = "I won't answer without a verifiable source."
    calls = {"n": 0}

    class _RegenThenRefuse(_MinimalEngine):
        def capabilities(self) -> Dict[str, bool]:
            return {"enforce_response": True}

        def enforce_response(self, answer, messages, model="", *, final=False, **kwargs):
            calls["n"] += 1
            if final:
                return {"action": "replace", "text": SAFE}
            return {"action": "regenerate", "message": "cite a source"}

    agent = _agent_with(_RegenThenRefuse())
    out = _maybe_enforce_response(agent, ANSWER, HISTORY, logger=MagicMock())
    assert out == SAFE
    assert calls["n"] >= 2  # regenerated at least once, then refused on final
