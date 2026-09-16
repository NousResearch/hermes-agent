"""Guard: a model that echoes the injected ``(empty)`` sentinel back into its
final answer must RECOVER, not terminate with the leak as its answer.

Observed leak: ``"(empty) again risk. Need progress + tool."``

A garbage-but-present answer is worse than an empty one, because empty recovers
and garbage does not. The sentinel is injected by the empty-response ladder in
``agent/turn_empty_response.py``, so a model repeating it leaked scratchpad
shorthand rather than answering.

These tests drive the REAL production predicate ``leaked_empty_sentinel`` and
the REAL constant, never a copy of their logic. Both directions are asserted:
the leak recovers, and legitimate short answers still end the turn.
"""

import pytest

from agent.turn_empty_response import EMPTY_RESPONSE_SENTINEL
from agent.turn_final_response import (
    _visible_text_for_sentinel_check,
    leaked_empty_sentinel,
)


class _Msg:
    """Minimal assistant-message stand-in: only ``content`` is read."""

    def __init__(self, content):
        self.content = content


# --- the constant is the single source of truth --------------------------------

def test_sentinel_constant_matches_the_injected_value():
    """The guard and the injection sites must share one value.

    ``turn_empty_response`` assigns this constant into the assistant row it
    appends. If the two drifted, the guard would stop firing silently.
    """
    assert EMPTY_RESPONSE_SENTINEL == "(empty)"


def test_injection_sites_use_the_constant_not_a_literal():
    """Pin the wiring: the ladder must inject the constant.

    A future edit that reintroduces a bare literal at an injection site can
    drift from the guard, so assert the constant is what gets assigned.
    """
    import inspect

    from agent import turn_empty_response as mod

    src = inspect.getsource(mod)
    # Executable assignments of the sentinel go through the constant.
    assert 'assistant_msg["content"] = EMPTY_RESPONSE_SENTINEL' in src
    assert '_nudge_msg["content"] = EMPTY_RESPONSE_SENTINEL' in src
    assert "return EMPTY_RESPONSE_SENTINEL" in src


# --- the leak recovers ---------------------------------------------------------

@pytest.mark.parametrize(
    "leaked",
    [
        "(empty) again risk. Need progress + tool.",   # the observed leak
        "(empty)",                                     # the bare sentinel
        "  (empty) leading whitespace",                # lstrip applies
        "(empty)\nsecond line",
    ],
)
def test_leaked_sentinel_is_detected(leaked):
    assert leaked_empty_sentinel(_Msg(leaked), leaked) is True


# --- legitimate answers still end the turn ------------------------------------

@pytest.mark.parametrize(
    "answer",
    [
        "Done.",
        "PASS",
        "OK",
        "42",
        "## Result\n\nThe build succeeded.",
        # Merely DISCUSSING the sentinel is not leaking it: prefix, not substring.
        'The model returned "(empty)" because the tool produced no output.',
        "Sometimes the answer is (empty) but not here.",
    ],
)
def test_real_answers_are_not_treated_as_leaks(answer):
    assert leaked_empty_sentinel(_Msg(answer), answer) is False


def test_empty_string_is_not_a_sentinel_leak():
    """An empty answer is the ORDINARY empty path, not the leak path.

    It already recovers through ``_has_content_after_think_block``, so the
    sentinel guard must not claim it.
    """
    assert leaked_empty_sentinel(_Msg(""), "") is False


# --- the guard judges VISIBLE text, not reasoning ------------------------------

def test_sentinel_inside_reasoning_does_not_trigger_when_answer_is_real():
    """A scratchpad that says ``(empty)`` while the visible answer is real must
    NOT recover. Judging the turn by its reasoning would discard a good answer."""
    content = [
        {"type": "thinking", "text": "(empty) again risk. Need progress + tool."},
        {"type": "text", "text": "The build succeeded."},
    ]
    assert leaked_empty_sentinel(_Msg(content), "The build succeeded.") is False


def test_sentinel_in_visible_block_triggers_despite_reasoning():
    content = [
        {"type": "thinking", "text": "Let me answer properly."},
        {"type": "text", "text": "(empty) again risk."},
    ]
    assert leaked_empty_sentinel(_Msg(content), "(empty) again risk.") is True


@pytest.mark.parametrize(
    "part_type",
    ["thinking", "reasoning", "redacted_thinking", "reasoning_content"],
)
def test_every_non_visible_part_type_is_excluded(part_type):
    content = [
        {"type": part_type, "text": "(empty) scratchpad note"},
        {"type": "text", "text": "Real answer."},
    ]
    assert _visible_text_for_sentinel_check(content) == "Real answer."
    assert leaked_empty_sentinel(_Msg(content), "Real answer.") is False


# --- type tolerance: never raise ----------------------------------------------

@pytest.mark.parametrize(
    "content",
    [None, 123, 4.5, True, [], {}, [None], [123], {"no": "type"}, object()],
)
def test_odd_content_shapes_never_raise(content):
    """A bad shape must yield ``False`` rather than crash the turn.

    Failing closed here means "terminate as before", which is the safe
    direction: a crash would lose the whole turn.
    """
    assert leaked_empty_sentinel(_Msg(content), "") is False


def test_missing_content_attribute_is_tolerated():
    class NoContent:
        pass

    assert leaked_empty_sentinel(NoContent(), "Done.") is False


def test_recovery_decision_consults_the_guard():
    """The guard must be WIRED into the recovery decision, not merely defined.

    Every other test here calls ``leaked_empty_sentinel`` directly, so all of
    them stay green if the decision stops consulting it. Measured: unwiring the
    call left 33 of 33 tests passing. This test closes that hole by asserting
    the call site exists inside the branch that routes to recovery.
    """
    import inspect

    from agent import turn_final_response as mod

    src = inspect.getsource(mod)
    definitions = src.count("def leaked_empty_sentinel(")
    call_sites = src.count("leaked_empty_sentinel(") - definitions
    assert call_sites >= 1, (
        "leaked_empty_sentinel is defined but never called: the recovery "
        "decision no longer consults it, so the guard is dead code"
    )
    # And the call must sit in the condition guarding recover_empty_response.
    decision = src[src.index("# Think-block-only / empty content"):]
    decision = decision[:decision.index("recover_empty_response(")]
    assert "leaked_empty_sentinel(" in decision, (
        "leaked_empty_sentinel is called somewhere, but not in the condition "
        "that routes to recover_empty_response"
    )


def test_partial_stream_recovery_is_fenced_against_the_sentinel():
    """The partial-stream path re-delivers streamed text verbatim.

    That text holds the leak, so a turn routed to recovery BECAUSE of a leaked
    sentinel must not receive the same shorthand back from this path.
    """
    import inspect

    from agent import turn_empty_response as mod

    src = inspect.getsource(mod)
    partial = src[src.index("# Partial stream recovery"):]
    partial = partial[:partial.index('_turn_exit_reason = "partial_stream_recovery"')]
    assert "EMPTY_RESPONSE_SENTINEL" in partial, (
        "the partial-stream recovery branch does not exclude the sentinel, so "
        "it can re-deliver the leaked text as the final answer"
    )


def test_none_message_falls_back_to_final_response():
    """With no assistant message, the guard judges ``final_response``."""
    assert leaked_empty_sentinel(None, "(empty) again risk.") is True
    assert leaked_empty_sentinel(None, "Done.") is False


def test_final_response_fallback_when_visible_text_is_empty():
    """A message whose visible text is empty falls back to ``final_response``.

    Some providers deliver the answer only in ``final_response``, so the guard
    must still see the leak.
    """
    assert leaked_empty_sentinel(_Msg(None), "(empty) again risk.") is True
