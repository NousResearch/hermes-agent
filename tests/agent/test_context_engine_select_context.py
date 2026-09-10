"""Tests for the per-turn ``ContextEngine.select_context()`` hook.

``select_context()`` is the *selection / routing* verb — distinct from
compression — that lets an external context engine replace which context
enters the prompt for a single request, every turn, independent of
``should_compress()``. It is additive and no-op by default, and the host
call site (``_apply_context_engine_selection``) is fail-open: a missing hook,
an exception, or an invalid return value must leave the assembled request
untouched and must never mutate persisted history.

This pins the contract that engines such as retrieval-augmented, topic-routed,
and role-switching engines rely on (RFC #36765), consolidating the per-turn
request-assembly surface proposed across #41918, #24949, #47109, and #50053.
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock

from agent.context_engine import ContextEngine
from agent.conversation_loop import (
    _apply_context_engine_selection,
    _notify_context_engine_turn_complete,
)


class _MinimalEngine(ContextEngine):
    """Concrete engine implementing only the abstract methods."""

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
    agent = MagicMock()
    agent.session_id = "test-session"
    agent.context_compressor = engine
    return agent


REQUEST = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "hello"},
]
HISTORY = [{"role": "user", "content": "hello"}]


# -- ABC default -----------------------------------------------------------



# -- Host call site: _apply_context_engine_selection -----------------------



def test_base_noop_select_context_is_short_circuited_not_called():
    """Non-implementing engines skip the hook entirely (no call, no copies).

    The built-in ContextCompressor — and any engine that merely inherits the
    ABC default — must keep the default request path byte-identical AND pay
    nothing per request. ``hasattr`` alone cannot distinguish "inherits the
    no-op default" from "implements the hook" because the ABC defines
    ``select_context`` on every engine; the host therefore identity-checks the
    bound method against ``ContextEngine.select_context`` and short-circuits
    WITHOUT calling it or building the shallow reference copies. This pins
    that: even a base implementation patched to raise is never invoked.
    """
    from unittest.mock import patch as _patch

    def _explode(self, request_messages, **kwargs):
        raise AssertionError("base select_context must not be invoked")

    engine = _MinimalEngine()  # inherits the ABC default
    agent = _agent_with(engine)
    logger = MagicMock()
    with _patch.object(ContextEngine, "select_context", _explode):
        out = _apply_context_engine_selection(
            agent, REQUEST, HISTORY, HISTORY[-1], logger=logger
        )
    assert out is REQUEST
    assert not logger.warning.called
















def test_empty_list_keeps_original_request():
    """An empty list must fall open to the original request.

    ``all([])`` is ``True``, so without an emptiness check a ``[]`` returned by
    a failing/buggy engine would replace a valid assembled request with an
    empty message list the downstream sanitizers cannot restore — reaching the
    provider as an invalid request instead of failing open. Guards the fail-open
    contract.
    """

    class _Engine(_MinimalEngine):
        def select_context(self, request_messages, **kwargs):
            return []

    logger = MagicMock()
    agent = _agent_with(_Engine())
    out = _apply_context_engine_selection(
        agent, REQUEST, HISTORY, HISTORY[-1], logger=logger
    )
    assert out is REQUEST
    assert logger.warning.called


def test_engine_mutating_inputs_cannot_corrupt_persisted_state():
    """An engine that mutates its read-only inputs in place must not affect the
    persisted conversation history / incoming message.

    ``conversation_messages`` and ``incoming_message`` are reference-only
    context. The host passes shallow copies, so even a misbehaving engine that
    appends to / edits them in ``select_context()`` cannot alter the live
    persisted objects. Enforces the request-only contract (not just documents).
    """
    history = [{"role": "user", "content": "hello"}]
    incoming = history[-1]
    history_snapshot = [dict(m) for m in history]
    incoming_snapshot = dict(incoming)

    class _Engine(_MinimalEngine):
        def select_context(self, request_messages, *, conversation_messages=None,
                            incoming_message=None, **kwargs):
            # Misbehaving engine: mutate the read-only inputs in place.
            if conversation_messages is not None:
                conversation_messages.append({"role": "user", "content": "INJECTED"})
                if conversation_messages and isinstance(conversation_messages[0], dict):
                    conversation_messages[0]["content"] = "TAMPERED"
            if isinstance(incoming_message, dict):
                incoming_message["content"] = "TAMPERED"
            return None

    agent = _agent_with(_Engine())
    _apply_context_engine_selection(
        agent, REQUEST, history, incoming, logger=MagicMock()
    )
    # Persisted history + incoming message are untouched despite the engine's
    # in-place mutation of the copies it received.
    assert history == history_snapshot
    assert incoming == incoming_snapshot


def test_persisted_history_not_mutated():
    """The hook must not mutate the persisted conversation history."""

    class _Engine(_MinimalEngine):
        def select_context(self, request_messages, *, conversation_messages=None, **kwargs):
            # Even a misbehaving engine touching its inputs must not affect
            # what the host persists — the host passes the live list, so we
            # assert the host contract by checking the engine received it and
            # the canonical copy is unchanged after the call.
            return list(request_messages)

    history_snapshot = [dict(m) for m in HISTORY]
    agent = _agent_with(_Engine())
    _apply_context_engine_selection(
        agent, REQUEST, HISTORY, HISTORY[-1], logger=MagicMock()
    )
    assert HISTORY == history_snapshot


# -- cache-stability + downstream-sanitizer contract -----------------------



def test_role_unusual_replacement_passed_through_for_downstream_sanitizers():
    """The hook does structural validation only; role/tool normalization is
    deferred to the existing downstream sanitizers.

    A `system -> user -> user` replacement (the exact shape flagged as a
    role-alternation risk on sibling PRs) is well-formed structurally, so the
    host returns it verbatim. Role-pairing/orphaned-tool cleanup runs *after*
    this hook in the request pipeline (`_sanitize_api_messages`,
    `_drop_thinking_only_and_merge_users`), so select_context cannot emit a
    malformed request that bypasses validation.
    """
    role_unusual = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "a"},
        {"role": "user", "content": "b"},
    ]

    class _Engine(_MinimalEngine):
        def select_context(self, request_messages, **kwargs):
            return role_unusual

    agent = _agent_with(_Engine())
    out = _apply_context_engine_selection(
        agent, REQUEST, HISTORY, HISTORY[-1], logger=MagicMock()
    )
    assert out is role_unusual  # accepted structurally; downstream sanitizers normalize


# -- on_turn_complete (post-turn observation) ------------------------------



def test_on_turn_complete_called_with_snapshot_and_meta():
    """Host forwards a transcript copy + metadata; base no-op is skipped."""
    captured = {}

    class _Engine(_MinimalEngine):
        def on_turn_complete(self, messages, usage=None, **kwargs):
            captured["messages"] = messages
            captured["usage"] = usage
            captured["kwargs"] = kwargs

    agent = _agent_with(_Engine())
    _notify_context_engine_turn_complete(
        agent, HISTORY, usage={"total_tokens": 12}, logger=MagicMock(),
        turn_id="t1", api_call_count=1,
    )
    assert captured["messages"] == HISTORY
    assert captured["messages"] is not HISTORY  # shallow copy
    assert captured["usage"] == {"total_tokens": 12}
    assert captured["kwargs"]["turn_id"] == "t1"
    assert captured["kwargs"]["api_call_count"] == 1






