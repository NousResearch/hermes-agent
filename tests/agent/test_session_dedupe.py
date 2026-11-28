"""Session-level duplicate delivery (agent.tool_guardrails, cross-turn).

Contracts, measured from the real sessions this fixes (10-session Ollama Cloud
analysis, 2026-11-26: 55 byte-identical re-reads re-injecting ~837K chars):
- an idempotent read-only call repeated in a LATER batch with a byte-identical
  result is not re-delivered in full;
- when the earlier copy is still in context (its tool_call_id is in the batch
  snapshot) the duplicate becomes a reference stub;
- when the earlier copy was compacted out, the controller asks the runtime to
  re-materialize the payload as a persisted-output archive instead;
- a same-args repeat whose result CHANGED passes whole, pollers and mutating
  tools are never touched, and a runtime that never opts in (no snapshot)
  keeps the old full-delivery behavior.

The per-call loop suffix/streak stub (same-turn, pre-existing) is unaffected:
tests below interleave a different call to reach the cross-turn path cleanly.
"""

from pathlib import Path
import sys

from agent.tool_guardrails import (
    SESSION_DEDUPE_MAX_ENTRIES,
    SESSION_DEDUPE_REMAT_PERSIST_CHARS,
    ToolCallGuardrailController,
    ToolCallGuardrailConfig,
    collect_live_tool_call_ids,
)

BIG = "x" * SESSION_DEDUPE_REMAT_PERSIST_CHARS  # exactly at the persist threshold
BIG_PLUS = BIG + "y"
SMALL = "skill body " * 40  # a few KB, below the threshold

WORKTREE_ROOT = str(Path(__file__).resolve().parents[2])


def _args(name="s", value=1):
    return {"name": name, "value": value}


def _import_worktree_run_agent(monkeypatch=None):
    """Import run_agent from THIS checkout, not a sys.path shadow (worktrees of an
    in-home install otherwise resolve hermes_bootstrap to the parent checkout).

    Two environment-only guards, no-ops on a real CI checkout:
    - the bootstrap update-marker stat trips home_io_guard when the git worktree's
      common dir sits inside the real home -> run_agent's own early recovery call is
      pre-stubbed instead (it is a fast-path no-op without a torn update anyway);
    - run_agent only needs hermes_bootstrap first-imported, not functional, so a
      no-op stub stands in if its import fails."""
    if not sys.path[:1] or sys.path[0] != WORKTREE_ROOT:
        sys.path.insert(0, WORKTREE_ROOT)
    try:
        import hermes_cli._early_recovery as _early_recovery

        if monkeypatch is not None:
            monkeypatch.setattr(_early_recovery, "restore_interrupted_pull", lambda *a, **k: False)
    except Exception:
        pass
    try:
        from run_agent import AIAgent

        return AIAgent
    except AssertionError:
        import types

        stub = types.ModuleType("hermes_bootstrap")
        stub.harden_import_path = staticmethod(lambda *a, **k: None)
        sys.modules["hermes_bootstrap"] = stub
        from run_agent import AIAgent

        return AIAgent


def _live(ids=("call-0",)):
    return frozenset(ids)


def _break_streak(c, call_id="mid"):
    """A different (tool, args) call so the next repeat starts a fresh streak."""
    c.observe_call("read_file", {"path": "/unrelated"}, "file body", tool_call_id=call_id)
    return c


def test_collect_live_tool_call_ids_extracts_tool_results_only():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "tool_calls": [{"id": "tc-1"}]},
        {"role": "tool", "tool_call_id": "tc-1", "content": "result"},
        {"role": "tool", "tool_call_id": "tc-2"},
        "not-a-dict",
    ]
    assert collect_live_tool_call_ids(messages) == frozenset({"tc-1", "tc-2"})
    assert collect_live_tool_call_ids(None) == frozenset()
    assert collect_live_tool_call_ids(["nope"]) == frozenset()


# ── opt-in gate: no snapshot -> zero behavior change (back-compat) ─────────


def test_without_runtime_snapshot_nothing_is_rewritten():
    c = ToolCallGuardrailController()
    first = c.observe_call("skill_view", _args(), BIG, tool_call_id="call-0")
    assert first.stub is None and first.session_rematerialize is None
    _break_streak(c)
    # Later batch, snapshot still None (runtime never opted in): pass whole.
    c.set_live_tool_calls(None)
    duplicate = c.observe_call("skill_view", _args(), BIG, tool_call_id="call-9")
    assert duplicate.stub is None
    assert duplicate.session_rematerialize is None


# ── prior result still live: reference stub ────────────────────────────────


def test_identical_repeat_with_prior_still_live_becomes_session_stub():
    c = ToolCallGuardrailController()
    c.observe_call("skill_view", _args(), BIG, tool_call_id="call-0")
    _break_streak(c)
    c.set_live_tool_calls(_live())  # call-0 is still in context
    obs = c.observe_call("skill_view", _args(), BIG, tool_call_id="call-1")
    assert obs.stub is not None
    assert "call-0" in obs.stub
    assert "earlier this session" in obs.stub
    assert obs.session_rematerialize is None


def test_session_stub_references_persisted_path_of_prior_call():
    c = ToolCallGuardrailController()
    c.observe_call("web_extract", _args(), BIG, tool_call_id="call-0")
    _break_streak(c)
    c.record_persisted_result("call-0", "/tmp/spill/call-0.txt")
    c.set_live_tool_calls(_live())
    obs = c.observe_call("web_extract", _args(), BIG, tool_call_id="call-1")
    assert obs.stub is not None
    assert "/tmp/spill/call-0.txt" in obs.stub


def test_small_identical_repeat_passes_whole():
    c = ToolCallGuardrailController()
    c.observe_call("skill_view", _args(), SMALL, tool_call_id="call-0")
    _break_streak(c)
    c.set_live_tool_calls(_live())
    obs = c.observe_call("skill_view", _args(), SMALL, tool_call_id="call-1")
    assert obs.stub is None and obs.session_rematerialize is None


# ── prior evicted by compaction: re-materialization ────────────────────────


def test_evicted_prior_big_result_asks_for_rematerialization():
    c = ToolCallGuardrailController()
    c.observe_call("skill_view", _args(), BIG, tool_call_id="call-0")
    _break_streak(c)
    c.set_live_tool_calls(frozenset({"other-call"}))  # call-0 no longer in context
    obs = c.observe_call("skill_view", _args(), BIG, tool_call_id="call-1")
    assert obs.stub is None
    assert obs.session_rematerialize == "call-0"


def test_evicted_prior_small_result_passes_whole():
    c = ToolCallGuardrailController()
    c.observe_call("skill_view", _args(), SMALL, tool_call_id="call-0")
    _break_streak(c)
    c.set_live_tool_calls(frozenset({"other-call"}))
    obs = c.observe_call("skill_view", _args(), SMALL, tool_call_id="call-1")
    assert obs.stub is None and obs.session_rematerialize is None


def test_read_file_evicted_repeat_passes_whole():
    """An unchanged file is still on disk: no archive copy, no stub."""
    c = ToolCallGuardrailController()
    c.observe_call("read_file", {"path": "/a/b"}, BIG, tool_call_id="call-0")
    c.observe_call("skill_view", {"name": "z"}, BIG, tool_call_id="mid")
    c.set_live_tool_calls(frozenset({"other-call"}))
    obs = c.observe_call("read_file", {"path": "/a/b"}, BIG, tool_call_id="call-1")
    assert obs.stub is None and obs.session_rematerialize is None


def test_changed_result_is_new_information_and_re_registers():
    c = ToolCallGuardrailController()
    c.observe_call("web_search", _args(), BIG, tool_call_id="call-0")
    c.set_live_tool_calls(_live(("call-0",)))
    obs = c.observe_call("web_search", _args(), BIG_PLUS, tool_call_id="call-1")
    assert obs.stub is None and obs.session_rematerialize is None
    # The updated content is now the referenced copy: a byte-identical repeat stubs to call-1.
    obs2 = c.observe_call("web_search", _args(), BIG_PLUS, tool_call_id="call-2")
    assert obs2.stub is not None
    assert "call-1" in obs2.stub


# ── exemptions and config gate ─────────────────────────────────────────────


def test_pollers_and_mutating_tools_untouched_by_session_dedupe():
    c = ToolCallGuardrailController()
    c.observe_call("process_manage", {"action": "list"}, BIG, tool_call_id="call-0")
    c.observe_call("terminal", {"command": "ls"}, BIG, tool_call_id="call-t0")
    _break_streak(c)
    c.set_live_tool_calls(_live())
    poll = c.observe_call("process_manage", {"action": "list"}, BIG, tool_call_id="call-1")
    term = c.observe_call("terminal", {"command": "ls"}, BIG, tool_call_id="term-1")
    assert poll.stub is None and poll.session_rematerialize is None
    assert term.stub is None and term.session_rematerialize is None


def test_config_gate_disables_session_dedupe():
    cfg = ToolCallGuardrailConfig(session_dedupe_enabled=False)
    c = ToolCallGuardrailController(cfg)
    c.observe_call("skill_view", _args(), BIG, tool_call_id="call-0")
    _break_streak(c)
    c.set_live_tool_calls(_live())
    obs = c.observe_call("skill_view", _args(), BIG, tool_call_id="call-1")
    assert obs.stub is None and obs.session_rematerialize is None


def test_config_gate_round_trips_from_mapping():
    cfg = ToolCallGuardrailConfig.from_mapping({"session_dedupe_enabled": False})
    assert cfg.session_dedupe_enabled is False
    assert ToolCallGuardrailConfig.from_mapping({}).session_dedupe_enabled is True


def test_polls_keep_their_legacy_same_turn_stub():
    """Pre-existing contract: an unchanged poll repeated CONSECUTIVELY still stubs."""
    c = ToolCallGuardrailController()
    c.observe_call("process_manage", {"action": "list"}, BIG, tool_call_id="p0")
    obs = c.observe_call("process_manage", {"action": "list"}, BIG, tool_call_id="p1")
    assert obs.stub is not None  # legacy same-turn deduction, unchanged
    assert obs.session_rematerialize is None


# ── failure semantics and registry bounds ──────────────────────────────────


def test_registry_is_fifo_capped():
    c = ToolCallGuardrailController()
    for i in range(SESSION_DEDUPE_MAX_ENTRIES + 50):
        c.observe_call("read_file", {"path": f"/p/{i}"}, BIG, tool_call_id=f"c{i}")
    assert len(c._session_dedupe_order) <= SESSION_DEDUPE_MAX_ENTRIES
    assert len(c._session_dedupe) == len(c._session_dedupe_order)


def test_reset_for_session_clears_cross_turn_registry():
    c = ToolCallGuardrailController()
    c.observe_call("skill_view", _args(), BIG, tool_call_id="call-0")
    _break_streak(c)
    c.reset_for_session()
    c.set_live_tool_calls(_live(("call-0",)))
    obs = c.observe_call("skill_view", _args(), BIG, tool_call_id="call-1")
    assert obs.stub is None and obs.session_rematerialize is None


# ── runtime re-materialization (run_agent.AIAgent) ─────────────────────────


class _FakeAgentForRemat:
    """Just enough of AIAgent for the unbound _session_rematerialize_result call."""

    _current_task_id = ""


def _run_remat(payload, monkeypatch=None):
    AIAgent = _import_worktree_run_agent(monkeypatch)
    return AIAgent._session_rematerialize_result(
        _FakeAgentForRemat(),  # type: ignore[arg-type]
        payload, "skill_view", "call-1", "call-0",
    )


def test_session_rematerialize_result_persists_and_notes(monkeypatch):
    noted = _run_remat(BIG_PLUS, monkeypatch)
    assert "<persisted-output>" in noted
    assert "call-0" in noted and "byte-identical" in noted
    # Full payload left the context: it's in the archive, and the preview is bounded.
    assert BIG_PLUS not in noted
    from tools.tool_result_storage import _PERSISTED_PATH_RE

    path = _PERSISTED_PATH_RE.search(noted).group(1)
    archived = Path(path).read_text(encoding="utf-8", errors="replace")
    assert archived.startswith("x" * 10)
    assert len(archived) == len(BIG_PLUS)


def test_session_rematerialize_result_fails_open_for_small_payload(monkeypatch):
    import tools.tool_result_storage as stor

    noted = _run_remat(SMALL, monkeypatch)
    assert noted == SMALL
    assert "<persisted-output>" not in noted


def test_session_rematerialize_result_fails_open_when_persist_fails(tmp_path, monkeypatch):
    import tools.tool_result_storage as stor

    def boom(*args, **kwargs):
        raise RuntimeError("no disk")

    monkeypatch.setattr(stor, "maybe_persist_tool_result", boom)
    noted = _run_remat(BIG_PLUS, monkeypatch)
    assert noted == BIG_PLUS  # never starve the model when the archive can't be written
