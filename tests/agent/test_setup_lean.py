"""Invariant tests for the lean setup flow (issue #115482).

Behaviour contracts only: setup-intent classification agrees across modules
(single source of truth, no duplicated regex), setup turns skip memory
prefetch, and the per-turn iteration budget clamps to 12 and restores.
"""

from types import SimpleNamespace

from agent.conversation_loop import is_setup_message, run_conversation
from agent.memory_provider import is_setup_intent
from agent.turn_context import _memory_turn_start_and_prefetch


def test_setup_intent_contract_agrees_across_modules(monkeypatch):
    """Setup intent matches setup/configure phrasing and rejects unrelated text.

    `is_setup_message` must agree with `is_setup_intent` (the single source of
    truth): forcing the classifier's answer flips the wrapper too, proving no
    duplicated regex lives in conversation_loop.
    """
    for text in ("how do I setup telegram?", "set API key", "hermes tools"):
        assert is_setup_intent(text) is True
        assert is_setup_message(text) is True
    assert is_setup_intent("write a poem") is False
    assert is_setup_message("write a poem") is False

    import agent.memory_provider as _mp

    monkeypatch.setattr(_mp, "is_setup_intent", lambda text: True)
    assert is_setup_message("write a poem") is True
    monkeypatch.setattr(_mp, "is_setup_intent", lambda text: False)
    assert is_setup_message("how do I setup telegram?") is False


def test_setup_turn_skips_prefetch_and_clamps_budget(monkeypatch):
    """Setup text skips memory prefetch and clamps max_iterations to 12 for one turn."""
    # --- Prefetch gate: prefetch_all must not be called for setup text. ---
    def _boom(*args, **kwargs):
        raise AssertionError("prefetch_all must not run on setup turns")

    memory_manager = SimpleNamespace(
        on_turn_start=lambda *a, **k: None,
        prefetch_all=_boom,
        describe_recall=lambda: "",
    )
    agent = SimpleNamespace(
        _memory_manager=memory_manager,
        _user_turn_count=1,
        session_id="test-session",
        _emit_status=lambda *a, **k: None,
    )
    assert _memory_turn_start_and_prefetch(agent, "how do I setup telegram?") == ""

    # --- Lean budget: run_conversation clamps to 12, then restores. ---
    import agent.conversation_loop as _loop
    import agent.turn_context as _turn_context

    seen = {}
    runner = SimpleNamespace(max_iterations=500)
    monkeypatch.setattr(
        _loop,
        "_run_conversation_turn",
        lambda agent, *a, **k: (seen.setdefault("max_iterations", agent.max_iterations), {"ok": True})[1],
    )
    monkeypatch.setattr(_turn_context, "export_current_turn_boundary", lambda agent, result, msg: result)
    monkeypatch.setattr(_loop, "_close_durable_failed_turn", lambda agent, result: None)
    result = run_conversation(runner, "how do I setup telegram?")
    assert result == {"ok": True}
    assert seen["max_iterations"] == 12
    assert runner.max_iterations == 500
    assert _loop.lean_setup_active is False
