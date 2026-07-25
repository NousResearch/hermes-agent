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

def test_default_select_context_is_noop():
    """The base implementation returns None (no replacement)."""
    engine = _MinimalEngine()
    assert (
        engine.select_context(
            REQUEST,
            conversation_messages=HISTORY,
            incoming_message=HISTORY[-1],
            budget_tokens=0,
        )
        is None
    )


# -- Host call site: _apply_context_engine_selection -----------------------

def test_none_return_leaves_request_unchanged():
    """An engine returning None falls through to the assembled request."""
    engine = _MinimalEngine()  # default select_context -> None
    agent = _agent_with(engine)
    out = _apply_context_engine_selection(
        agent, REQUEST, HISTORY, HISTORY[-1], logger=MagicMock()
    )
    assert out is REQUEST


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


def test_builtin_compressor_inherits_base_select_context():
    """The built-in ContextCompressor must NOT implement the new verbs.

    Guards the default-path byte-identity contract: if someone overrides
    ``select_context`` / ``on_turn_complete`` on ContextCompressor, the host
    short-circuits no longer skip it and the default request pipeline gains a
    per-request call — update this pin only together with that decision.
    """
    from agent.context_compressor import ContextCompressor

    assert "select_context" not in ContextCompressor.__dict__
    assert "on_turn_complete" not in ContextCompressor.__dict__


def test_missing_hook_leaves_request_unchanged():
    """An engine without select_context (older/stub base) is a no-op."""
    engine = object()  # no select_context attribute
    agent = _agent_with(engine)
    out = _apply_context_engine_selection(
        agent, REQUEST, HISTORY, HISTORY[-1], logger=MagicMock()
    )
    assert out is REQUEST


def test_no_engine_leaves_request_unchanged():
    agent = MagicMock()
    agent.session_id = "test-session"
    agent.context_compressor = None
    out = _apply_context_engine_selection(
        agent, REQUEST, HISTORY, HISTORY[-1], logger=MagicMock()
    )
    assert out is REQUEST


def test_valid_list_replaces_request():
    """A valid list of dicts replaces the request messages for this call."""
    replacement = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "routed-context"},
    ]

    class _Engine(_MinimalEngine):
        def select_context(self, request_messages, **kwargs):
            return replacement

    agent = _agent_with(_Engine())
    out = _apply_context_engine_selection(
        agent, REQUEST, HISTORY, HISTORY[-1], logger=MagicMock()
    )
    assert out is replacement


def test_exception_fails_open():
    """A raising hook is swallowed; the unmodified request is used."""

    class _Engine(_MinimalEngine):
        def select_context(self, request_messages, **kwargs):
            raise RuntimeError("backend offline")

    logger = MagicMock()
    agent = _agent_with(_Engine())
    out = _apply_context_engine_selection(
        agent, REQUEST, HISTORY, HISTORY[-1], logger=logger
    )
    assert out is REQUEST
    assert logger.warning.called


def test_non_list_return_is_ignored():
    """A non-list return value is rejected and logged, request unchanged."""

    class _Engine(_MinimalEngine):
        def select_context(self, request_messages, **kwargs):
            return {"role": "user", "content": "oops not a list"}

    logger = MagicMock()
    agent = _agent_with(_Engine())
    out = _apply_context_engine_selection(
        agent, REQUEST, HISTORY, HISTORY[-1], logger=logger
    )
    assert out is REQUEST
    assert logger.warning.called


def test_list_of_non_dicts_is_ignored():
    """A list that isn't all dicts is rejected, request unchanged."""

    class _Engine(_MinimalEngine):
        def select_context(self, request_messages, **kwargs):
            return ["not", "dicts"]

    agent = _agent_with(_Engine())
    out = _apply_context_engine_selection(
        agent, REQUEST, HISTORY, HISTORY[-1], logger=MagicMock()
    )
    assert out is REQUEST


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

def test_noop_preserves_request_byte_stable_for_cache():
    """No-op default must leave the request byte-identical.

    Prompt-cache stability is a host invariant (AGENTS.md): the hook runs
    before cache-control, so a no-op engine must not perturb the list —
    otherwise cache breakpoints would shift for every existing engine. The
    host returns the *same object*, so cache-control sees identical input.
    """
    snapshot = [dict(m) for m in REQUEST]
    agent = _agent_with(_MinimalEngine())  # default select_context -> None
    out = _apply_context_engine_selection(
        agent, REQUEST, HISTORY, HISTORY[-1], logger=MagicMock()
    )
    assert out is REQUEST          # same object -> byte-stable for cache-control
    assert REQUEST == snapshot     # unperturbed


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

def test_default_on_turn_complete_is_noop():
    """The base on_turn_complete returns None and does nothing."""
    assert _MinimalEngine().on_turn_complete(HISTORY, usage=None) is None


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


def test_on_turn_complete_base_noop_is_skipped():
    """An engine that only inherits the base no-op is handled safely.

    The helper short-circuits the base implementation (so non-implementing
    engines pay nothing), and in any case must not raise.
    """
    agent = _agent_with(_MinimalEngine())  # inherits base on_turn_complete
    _notify_context_engine_turn_complete(agent, HISTORY, logger=MagicMock())


def test_on_turn_complete_fails_open():
    """A raising observation hook is swallowed and logged."""

    class _Engine(_MinimalEngine):
        def on_turn_complete(self, messages, usage=None, **kwargs):
            raise RuntimeError("indexing backend down")

    logger = MagicMock()
    agent = _agent_with(_Engine())
    _notify_context_engine_turn_complete(agent, HISTORY, logger=logger)
    assert logger.warning.called


def test_on_turn_complete_missing_engine_is_safe():
    agent = MagicMock()
    agent.session_id = "s"
    agent.context_compressor = None
    # No engine -> silent return, no raise.
    _notify_context_engine_turn_complete(agent, HISTORY, logger=MagicMock())
