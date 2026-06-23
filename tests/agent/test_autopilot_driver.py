"""Unit tests for the autopilot driver (engine-enforced goal-chasing)."""

import types

from agent.autopilot import driver
from agent.autopilot.council_gate import CompletionVerdict


class FakeBudget:
    def __init__(self, max_total):
        self.max_total = max_total
        self.used = 0

    @property
    def remaining(self):
        return max(0, self.max_total - self.used)


def make_agent(**overrides):
    a = types.SimpleNamespace()
    a.autopilot_mode = True
    a.iteration_budget = FakeBudget(90)
    a.max_iterations = 90
    a._api_call_count = 5
    a._status = []
    a._emit_status = lambda msg: a._status.append(msg)
    for k, v in overrides.items():
        setattr(a, k, v)
    driver.reset_turn_state(a)
    return a


# --------------------------------------------------------------------------- #
# activation / goal resolution                                                 #
# --------------------------------------------------------------------------- #
def test_active_via_attr():
    assert driver.is_autopilot_active(types.SimpleNamespace(autopilot_mode=True))


def test_active_via_env(monkeypatch):
    monkeypatch.setenv("HERMES_AUTOPILOT", "yes")
    assert driver.is_autopilot_active(types.SimpleNamespace())


def test_inactive(monkeypatch):
    monkeypatch.delenv("HERMES_AUTOPILOT", raising=False)
    assert not driver.is_autopilot_active(types.SimpleNamespace(autopilot_mode=False))


def test_resolve_goal_prefers_explicit():
    a = types.SimpleNamespace(_autopilot_goal="ship the feature")
    assert driver.resolve_goal(a, "ignored user msg") == "ship the feature"


def test_resolve_goal_from_user_message():
    assert driver.resolve_goal(types.SimpleNamespace(), "do the thing") == "do the thing"


def test_resolve_goal_multimodal():
    msg = [{"type": "text", "text": "alpha"}, {"type": "image_url"}, {"type": "text", "text": "beta"}]
    out = driver.resolve_goal(types.SimpleNamespace(), msg)
    assert "alpha" in out and "beta" in out


# ── /goal integration: autopilot chases the active standing goal ───────────
def test_resolve_goal_reads_active_standing_goal(monkeypatch):
    from hermes_cli import goals as goals_mod

    state = goals_mod.GoalState(goal="ship the parser fix", status="active")
    monkeypatch.setattr(goals_mod, "load_goal", lambda sid: state)
    a = types.SimpleNamespace(session_id="sess-1")  # no explicit autopilot goal
    assert driver.resolve_goal(a, "current chatter") == "ship the parser fix"


def test_resolve_goal_explicit_beats_standing(monkeypatch):
    from hermes_cli import goals as goals_mod

    state = goals_mod.GoalState(goal="standing goal", status="active")
    monkeypatch.setattr(goals_mod, "load_goal", lambda sid: state)
    a = types.SimpleNamespace(session_id="sess-1", _autopilot_goal="explicit goal")
    assert driver.resolve_goal(a, "msg") == "explicit goal"


def test_resolve_goal_ignores_paused_standing_goal(monkeypatch):
    from hermes_cli import goals as goals_mod

    state = goals_mod.GoalState(goal="paused goal", status="paused")
    monkeypatch.setattr(goals_mod, "load_goal", lambda sid: state)
    a = types.SimpleNamespace(session_id="sess-1")
    # paused (not active) → fall through to the user message
    assert driver.resolve_goal(a, "fallback task") == "fallback task"


def test_resolve_goal_standing_goal_includes_subgoals(monkeypatch):
    from hermes_cli import goals as goals_mod

    state = goals_mod.GoalState(goal="ship X", status="active", subgoals=["tests pass", "docs updated"])
    monkeypatch.setattr(goals_mod, "load_goal", lambda sid: state)
    a = types.SimpleNamespace(session_id="sess-1")
    out = driver.resolve_goal(a, "msg")
    assert "ship X" in out and "tests pass" in out and "docs updated" in out


def test_resolve_goal_no_session_id_skips_standing(monkeypatch):
    from hermes_cli import goals as goals_mod

    def _boom(sid):
        raise AssertionError("load_goal must not run without a session_id")

    monkeypatch.setattr(goals_mod, "load_goal", _boom)
    a = types.SimpleNamespace()  # no session_id
    assert driver.resolve_goal(a, "just the task") == "just the task"


def test_resolve_goal_standing_load_failure_falls_back(monkeypatch):
    from hermes_cli import goals as goals_mod

    def _boom(sid):
        raise RuntimeError("db down")

    monkeypatch.setattr(goals_mod, "load_goal", _boom)
    a = types.SimpleNamespace(session_id="sess-1")
    # load failure must fail safe to the user message, never crash
    assert driver.resolve_goal(a, "the task") == "the task"


# --------------------------------------------------------------------------- #
# maybe_continue                                                                #
# --------------------------------------------------------------------------- #
def test_complete_returns_none(monkeypatch):
    a = make_agent()
    monkeypatch.setattr(driver, "judge_completion",
                        lambda *args, **kw: CompletionVerdict(complete=True, summary="done"))
    assert driver.maybe_continue(a, [{"role": "user", "content": "g"}], "answer", "g") is None


def test_incomplete_returns_directive_and_extends_budget(monkeypatch):
    a = make_agent(_api_call_count=10)
    monkeypatch.setattr(driver, "judge_completion",
                        lambda *args, **kw: CompletionVerdict(complete=False, directive="do step 2", verdict="deny"))
    out = driver.maybe_continue(a, [{"role": "user", "content": "g"}], "partial", "g")
    assert out is not None and "do step 2" in out
    assert a._autopilot_continuations == 1
    # budget extended beyond current api_call_count so the loop won't end on the cap
    assert a.max_iterations >= 10 + 1
    assert a.iteration_budget.max_total >= 10 + 1


def test_inactive_short_circuits(monkeypatch):
    monkeypatch.delenv("HERMES_AUTOPILOT", raising=False)
    a = make_agent(autopilot_mode=False)
    assert driver.maybe_continue(a, [{"role": "user"}], "x", "g") is None


def test_empty_goal_returns_none(monkeypatch):
    a = make_agent()
    monkeypatch.setattr(driver, "judge_completion",
                        lambda *args, **kw: CompletionVerdict(complete=False, directive="x"))
    assert driver.maybe_continue(a, [{"role": "user"}], "x", "") is None


def test_user_cap_stops_after_limit(monkeypatch):
    a = make_agent(_autopilot_max_continuations=1)
    monkeypatch.setattr(driver, "judge_completion",
                        lambda *args, **kw: CompletionVerdict(complete=False, directive="x", verdict="deny"))
    assert driver.maybe_continue(a, [{"role": "user"}], "p1", "g") is not None  # #1
    assert driver.maybe_continue(a, [{"role": "user"}], "p2", "g") is None       # cap hit


def test_no_progress_stall_stops(monkeypatch):
    a = make_agent(_autopilot_no_progress_k=2)
    monkeypatch.setattr(driver, "judge_completion",
                        lambda *args, **kw: CompletionVerdict(complete=False, directive="x", verdict="deny"))
    msgs = [{"role": "user", "content": "g"}]
    assert driver.maybe_continue(a, msgs, "SAME", "g") is not None  # stall 0 -> continue
    assert driver.maybe_continue(a, msgs, "SAME", "g") is not None  # stall 1 -> continue
    assert driver.maybe_continue(a, msgs, "SAME", "g") is None      # stall 2 >= k -> stop


def test_progress_resets_stall(monkeypatch):
    a = make_agent(_autopilot_no_progress_k=2)
    monkeypatch.setattr(driver, "judge_completion",
                        lambda *args, **kw: CompletionVerdict(complete=False, directive="x", verdict="deny"))
    # growing transcript + changing final response => never stalls
    for i in range(5):
        msgs = [{"role": "user"}] * (i + 2)
        assert driver.maybe_continue(a, msgs, f"final-{i}", "g") is not None
    assert a._autopilot_continuations == 5


def test_judge_exception_delivers(monkeypatch):
    a = make_agent()

    def boom(*args, **kw):
        raise RuntimeError("judge down")

    monkeypatch.setattr(driver, "judge_completion", boom)
    assert driver.maybe_continue(a, [{"role": "user"}], "x", "g") is None


# --------------------------------------------------------------------------- #
# make_clarify_autoanswer (Seam A wiring)                                       #
# --------------------------------------------------------------------------- #
def test_clarify_autoanswer_uses_council(monkeypatch):
    from agent.autopilot import council_gate
    a = make_agent()
    monkeypatch.setattr(council_gate, "choose_answer_detailed",
                        lambda q, c=None, **k: council_gate.ClarifyDecision(answer="Option B", options=list(c or []), source="council"))
    cb = driver.make_clarify_autoanswer(a)
    assert cb("Which option?", ["Option A", "Option B"]) == "Option B"


def test_clarify_autoanswer_falls_back_on_error(monkeypatch):
    from agent.autopilot import council_gate
    a = make_agent()

    def boom(*args, **kw):
        raise RuntimeError("council down")

    monkeypatch.setattr(council_gate, "choose_answer_detailed", boom)
    seen = {}

    def fb(q, c):
        seen["called"] = (q, c)
        return "FALLBACK"

    cb = driver.make_clarify_autoanswer(a, fallback=fb)
    assert cb("q", ["a"]) == "FALLBACK"
    assert "called" in seen


def test_clarify_autoanswer_default_when_empty(monkeypatch):
    from agent.autopilot import council_gate
    a = make_agent()
    monkeypatch.setattr(council_gate, "choose_answer_detailed",
                        lambda q, c=None, **k: council_gate.ClarifyDecision(answer="", options=list(c or []), source="aux"))
    cb = driver.make_clarify_autoanswer(a)  # no fallback
    assert "default" in cb("q", None).lower()


# --------------------------------------------------------------------------- #
# _emit visibility (oneshot stderr fallback)                                   #
# --------------------------------------------------------------------------- #
def test_emit_uses_status_when_not_suppressed():
    a = make_agent()
    driver._emit(a, "hello status")
    assert "hello status" in a._status


def test_emit_falls_back_to_stderr_when_suppressed(capsys):
    a = make_agent()
    a.suppress_status_output = True
    driver._emit(a, "autopilot oneshot line")
    err = capsys.readouterr().err
    assert "autopilot oneshot line" in err
    # status callback must NOT be used when suppressed
    assert "autopilot oneshot line" not in a._status


def test_off_overrides_env(monkeypatch):
    # The reported bug: with HERMES_AUTOPILOT set, /autopilot off (autopilot_mode
    # = False) must still turn it OFF. The per-agent flag is authoritative.
    monkeypatch.setenv("HERMES_AUTOPILOT", "1")
    assert driver.is_autopilot_active(types.SimpleNamespace(autopilot_mode=False)) is False


def test_on_flag_beats_unset_env(monkeypatch):
    monkeypatch.delenv("HERMES_AUTOPILOT", raising=False)
    assert driver.is_autopilot_active(types.SimpleNamespace(autopilot_mode=True)) is True


def test_env_fallback_only_when_attr_missing(monkeypatch):
    monkeypatch.setenv("HERMES_AUTOPILOT", "1")
    # no autopilot_mode attribute at all -> env fallback applies
    assert driver.is_autopilot_active(types.SimpleNamespace()) is True


# ── keep_budget_ahead (long-run budget exhaustion fix) ──────────────────


def test_keep_budget_ahead_extends_when_active():
    a = make_agent(_api_call_count=200)
    a.iteration_budget = FakeBudget(90)
    a.iteration_budget.used = 88
    driver.keep_budget_ahead(a, headroom=50)
    # budget + max_iterations pushed ahead of current usage (200)
    assert a.iteration_budget.max_total >= 250
    assert a.max_iterations >= 250


def test_keep_budget_ahead_noop_when_inactive(monkeypatch):
    monkeypatch.delenv("HERMES_AUTOPILOT", raising=False)
    a = make_agent(autopilot_mode=False, _api_call_count=200)
    a.iteration_budget = FakeBudget(90)
    a.max_iterations = 90
    driver.keep_budget_ahead(a)
    assert a.iteration_budget.max_total == 90
    assert a.max_iterations == 90


def test_keep_budget_ahead_stops_at_user_cap():
    a = make_agent(_api_call_count=200, _autopilot_max_continuations=3)
    a._autopilot_continuations = 3  # cap reached
    a.iteration_budget = FakeBudget(90)
    a.max_iterations = 90
    driver.keep_budget_ahead(a)
    # at the cap, do not keep extending — let the run wind down
    assert a.max_iterations == 90


# --------------------------------------------------------------------------- #
# reenter_after_abnormal_exit (belt-and-suspenders for non-Seam-B loop exits)  #
# --------------------------------------------------------------------------- #
def make_reenter_agent(**overrides):
    return make_agent(**overrides)


def test_reenter_returns_directive_when_gate_continues(monkeypatch):
    a = make_reenter_agent(_api_call_count=10)
    monkeypatch.setattr(driver, "judge_completion",
                        lambda *args, **kw: CompletionVerdict(complete=False, directive="keep going", verdict="deny"))
    out = driver.reenter_after_abnormal_exit(
        a, [{"role": "user", "content": "g"}], "(empty)", "g", exit_kind="empty_response")
    assert out is not None and "keep going" in out
    # counts toward the SAME continuation bookkeeping as Seam B
    assert a._autopilot_continuations == 1


def test_reenter_does_not_mutate_messages(monkeypatch):
    # The driver only DECIDES; the loop owns injection (so it can keep role
    # alternation valid). The driver must not append/pop messages itself.
    a = make_reenter_agent(_api_call_count=4)
    monkeypatch.setattr(driver, "judge_completion",
                        lambda *args, **kw: CompletionVerdict(complete=False, directive="go", verdict="deny"))
    msgs = [{"role": "user", "content": "g"}]
    before = list(msgs)
    out = driver.reenter_after_abnormal_exit(a, msgs, "(empty)", "g", exit_kind="empty_response")
    assert out is not None
    assert msgs == before  # unchanged


def test_reenter_returns_none_when_complete(monkeypatch):
    a = make_reenter_agent()
    monkeypatch.setattr(driver, "judge_completion",
                        lambda *args, **kw: CompletionVerdict(complete=True, summary="done"))
    out = driver.reenter_after_abnormal_exit(
        a, [{"role": "user"}], "answer", "g", exit_kind="partial_stream_recovery")
    assert out is None


def test_reenter_blocked_by_interrupted_param(monkeypatch):
    a = make_reenter_agent()

    def boom(*args, **kw):
        raise AssertionError("judge must not run when interrupted")

    monkeypatch.setattr(driver, "judge_completion", boom)
    out = driver.reenter_after_abnormal_exit(
        a, [{"role": "user"}], "x", "g", exit_kind="empty_response", interrupted=True)
    assert out is None


def test_reenter_blocked_by_agent_interrupt_flag(monkeypatch):
    a = make_reenter_agent(_interrupt_requested=True)

    def boom(*args, **kw):
        raise AssertionError("judge must not run when agent interrupt is set")

    monkeypatch.setattr(driver, "judge_completion", boom)
    out = driver.reenter_after_abnormal_exit(
        a, [{"role": "user"}], "x", "g", exit_kind="empty_response")
    assert out is None


def test_reenter_inactive_returns_none(monkeypatch):
    monkeypatch.delenv("HERMES_AUTOPILOT", raising=False)
    a = make_reenter_agent(autopilot_mode=False)

    def boom(*args, **kw):
        raise AssertionError("judge must not run when autopilot is off")

    monkeypatch.setattr(driver, "judge_completion", boom)
    out = driver.reenter_after_abnormal_exit(
        a, [{"role": "user"}], "x", "g", exit_kind="empty_response")
    assert out is None


def test_reenter_judge_exception_delivers(monkeypatch):
    a = make_reenter_agent()

    def boom(*args, **kw):
        raise RuntimeError("judge down")

    monkeypatch.setattr(driver, "judge_completion", boom)
    out = driver.reenter_after_abnormal_exit(
        a, [{"role": "user"}], "x", "g", exit_kind="empty_response")
    assert out is None


# --------------------------------------------------------------------------- #
# give-up / handoff detection (engine hardening — never stop on a wrap-up)      #
# --------------------------------------------------------------------------- #
def test_looks_like_giveup_detects_handoff_phrases():
    for s in [
        "This session has reached its productive limit — handoff written.",
        "Context near exhaustion; next session should resume.",
        "I'll stop here and resume in a fresh session.",
        "Session summary (honest, gate-anchored): 2/7 GREEN.",
    ]:
        assert driver._looks_like_giveup(s), s


def test_looks_like_giveup_ignores_normal_text():
    for s in ["Fixed the search bug; all 305 tests pass.", "", "Continuing to lane 3."]:
        assert not driver._looks_like_giveup(s)


def test_giveup_fails_closed_on_judge_error(monkeypatch):
    # Judge unavailable + a handoff response => must CONTINUE (fail closed), not deliver.
    a = make_agent(_api_call_count=10)

    def boom(*args, **kw):
        raise RuntimeError("council down")

    monkeypatch.setattr(driver, "judge_completion", boom)
    out = driver.maybe_continue(
        a, [{"role": "user", "content": "g"}],
        "This session has reached its productive limit — handoff written for next session.",
        "ship the gate",
    )
    assert out is not None
    assert "do NOT stop" in out or "DIRECTIVE" in out
    assert a._autopilot_continuations == 1  # counted + budget extended


def test_normal_response_still_fails_open_on_judge_error(monkeypatch):
    # No give-up language + judge error => preserve fail-open (deliver).
    a = make_agent()

    def boom(*args, **kw):
        raise RuntimeError("council down")

    monkeypatch.setattr(driver, "judge_completion", boom)
    out = driver.maybe_continue(a, [{"role": "user"}], "All done, 305/305 pass.", "g")
    assert out is None


def test_giveup_strengthens_directive_when_incomplete(monkeypatch):
    a = make_agent(_api_call_count=5)
    monkeypatch.setattr(driver, "judge_completion",
                        lambda *args, **kw: CompletionVerdict(complete=False, directive="lane 3 still red", verdict="deny"))
    out = driver.maybe_continue(
        a, [{"role": "user", "content": "g"}],
        "Productive limit reached; writing handoff for the next session.",
        "ship the gate",
    )
    assert out is not None
    assert "do NOT stop" in out  # the give-up directive, not the plain one
    assert "lane 3 still red" in out  # still carries the council's finding


def test_build_directive_is_non_dismissible():
    d = driver._build_directive(CompletionVerdict(complete=False, directive="do X", verdict="deny"))
    assert "NOT a notification" in d and "do X" in d


# --------------------------------------------------------------------------- #
# ADR decision-log wiring (maybe_continue + clarify)                            #
# --------------------------------------------------------------------------- #
def test_adr_written_at_completion(monkeypatch, tmp_path):
    from agent.autopilot import council_gate
    target = tmp_path / "adr.md"
    a = make_agent(_autopilot_adr=True, _autopilot_adr_path=str(target), _autopilot_goal="fix lint")
    # Council says complete.
    monkeypatch.setattr(driver, "judge_completion",
                        lambda *args, **kw: council_gate.CompletionVerdict(
                            complete=True, verdict="allow", confidence=0.9, source="council",
                            summary="council verdict=allow", raw={"arbiter": {}}))
    out = driver.maybe_continue(a, [{"role": "user", "content": "go"}], "done", "fix lint")
    assert out is None                       # complete -> stop
    assert target.exists()
    body = target.read_text()
    assert "— completion" in body
    assert "stop — goal verified complete" in body


def test_adr_written_at_continue_with_gap(monkeypatch, tmp_path):
    from agent.autopilot import council_gate
    target = tmp_path / "adr.md"
    a = make_agent(_autopilot_adr=True, _autopilot_adr_path=str(target), _autopilot_goal="ship it")
    monkeypatch.setattr(driver, "judge_completion",
                        lambda *args, **kw: council_gate.CompletionVerdict(
                            complete=False, verdict="deny", confidence=0.6, directive="run the tests",
                            source="council", summary="council verdict=deny",
                            raw={"arbiter": {"most_likely_wrong_point": "no tests run",
                                             "required_checks": ["run pytest"]}}))
    out = driver.maybe_continue(a, [{"role": "user", "content": "go"}], "I think it's done", "ship it")
    assert out is not None                   # not complete -> continue directive
    body = target.read_text()
    assert "— continue" in body
    assert "no tests run" in body
    assert "run pytest" in body


def test_adr_not_written_when_disabled(monkeypatch, tmp_path):
    from agent.autopilot import council_gate
    target = tmp_path / "adr.md"
    a = make_agent(_autopilot_adr=False, _autopilot_adr_path=str(target), _autopilot_goal="x")
    monkeypatch.setattr(driver, "judge_completion",
                        lambda *args, **kw: council_gate.CompletionVerdict(
                            complete=True, verdict="allow", source="council", summary="ok", raw={}))
    driver.maybe_continue(a, [{"role": "user", "content": "go"}], "done", "x")
    assert not target.exists()               # ADR off -> no file


def test_adr_written_at_clarify(monkeypatch, tmp_path):
    from agent.autopilot import council_gate
    target = tmp_path / "adr.md"
    a = make_agent(_autopilot_adr=True, _autopilot_adr_path=str(target))
    monkeypatch.setattr(council_gate, "choose_answer_detailed",
                        lambda q, c=None, **k: council_gate.ClarifyDecision(
                            answer="SQLite", options=list(c or []), rationale="stdlib", source="council"))
    cb = driver.make_clarify_autoanswer(a)
    assert cb("Which DB?", ["Postgres", "SQLite"]) == "SQLite"
    body = target.read_text()
    assert "— clarify" in body
    assert "chosen path: SQLite" in body
    assert "Postgres" in body                # full option set recorded
