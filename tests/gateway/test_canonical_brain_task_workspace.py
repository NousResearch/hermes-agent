import json
import threading
import time

from tools.todo_tool import TodoStore

from gateway import canonical_brain_task_workspace as workspace


def _active_case(case_id="case:1", plan_id="plan:1"):
    return {
        "case_id": case_id,
        "next_action": {"kind": "task_resume", "next_step_id": "2"},
        "workspace": {
            "plan_event_id": "event-plan",
            "plan": {
                "plan_id": plan_id,
                "revision": 2,
                "objective": "Finish the approved complex task",
                "state": "active",
                "current_step_id": "2",
                "resume_cursor": {
                    "next_step_id": "2",
                    "summary": "Continue after the verified orientation step",
                },
                "steps": [
                    {"id": "1", "content": "Orient", "status": "completed"},
                    {"id": "2", "content": "Implement", "status": "in_progress"},
                    {"id": "3", "content": "Verify", "status": "pending"},
                ],
            },
            "remaining_step_ids": ["2", "3"],
            "verifications": [],
            "approvals": [],
            "capability_checks": [],
        },
    }


def _note_payload(note):
    return json.loads(note[note.index("{"):])


def _set_candidates(monkeypatch, case_ids, *, truncated=False, error=None):
    monkeypatch.setattr(
        workspace,
        "_candidate_case_ids",
        lambda thread_id, *, deadline: (list(case_ids), truncated, error),
    )


def _set_cases(monkeypatch, cases):
    monkeypatch.setattr(
        workspace,
        "_resume_case",
        lambda case_id, *, deadline: (cases[case_id], None),
    )


def test_exact_resume_hydrates_empty_todo_and_continues_plan(monkeypatch):
    store = TodoStore()
    _set_candidates(monkeypatch, ["case:1"])
    _set_cases(monkeypatch, {"case:1": _active_case()})

    result = workspace.prepare_task_workspace_resume(
        thread_id="thread-1",
        session_key="session-1",
        todo_store=store,
    )

    assert result["status"] == "exact"
    assert result["todo_hydrated"] is True
    assert [item["status"] for item in store.read()] == [
        "completed", "in_progress", "pending",
    ]
    assert "Continue from its resume_cursor" in result["note"]
    assert "do not repeat completed steps" in result["note"]
    assert len(result["note"]) <= workspace.MAX_RESUME_NOTE_CHARS
    assert _note_payload(result["note"])["local_todo_state"] == "hydrated_from_canonical"


def test_resume_reports_conflict_instead_of_preferring_nonmatching_local_todo(monkeypatch):
    store = TodoStore()
    store.write([{"id": "live", "content": "Current live plan", "status": "in_progress"}])
    _set_candidates(monkeypatch, ["case:1"])
    _set_cases(monkeypatch, {"case:1": _active_case()})

    result = workspace.prepare_task_workspace_resume(
        thread_id="thread-1",
        session_key="session-1",
        todo_store=store,
    )

    assert result["status"] == "incomplete"
    assert result["reason"] == "local_state_conflict"
    assert result["local_state_conflict"] is True
    assert result["todo_hydrated"] is False
    assert store.read() == [{"id": "live", "content": "Current live plan", "status": "in_progress"}]
    payload = _note_payload(result["note"])
    assert payload["reasons"] == ["local_state_conflict"]
    assert payload["local_todo"]["count"] == 1


def test_exact_matching_local_todo_remains_without_rewrite(monkeypatch):
    case = _active_case()
    store = TodoStore()
    store.write([
        {
            "id": step["id"],
            "content": step["content"],
            "status": step["status"],
        }
        for step in case["workspace"]["plan"]["steps"]
    ])
    before = store.read()
    _set_candidates(monkeypatch, ["case:1"])
    _set_cases(monkeypatch, {"case:1": case})

    result = workspace.prepare_task_workspace_resume(
        thread_id="thread-1",
        session_key="session-1",
        todo_store=store,
    )

    assert result["status"] == "exact"
    assert result["local_todo_state"] == "exact_match"
    assert result["todo_hydrated"] is False
    assert store.read() == before


def test_ambiguous_resume_leaves_selection_to_gpt(monkeypatch):
    store = TodoStore()
    cases = {
        "case:1": _active_case("case:1", "plan:1"),
        "case:2": _active_case("case:2", "plan:2"),
    }
    _set_candidates(monkeypatch, list(cases))
    _set_cases(monkeypatch, cases)

    result = workspace.prepare_task_workspace_resume(
        thread_id="thread-1",
        session_key="session-1",
        todo_store=store,
    )

    assert result["status"] == "ambiguous"
    assert result["candidate_case_ids"] == ["case:1", "case:2"]
    assert store.read() == []
    assert "Runtime has not selected or hydrated one" in result["note"]
    assert "Do not select by keywords" in result["note"]
    assert len(result["note"]) <= workspace.MAX_RESUME_NOTE_CHARS
    assert _note_payload(result["note"])["status"] == "ambiguous"


def test_completed_or_blocked_plan_is_not_auto_resumed(monkeypatch):
    case = _active_case()
    case["workspace"]["plan"]["state"] = "completed"
    _set_candidates(monkeypatch, ["case:1"])
    _set_cases(monkeypatch, {"case:1": case})

    result = workspace.prepare_task_workspace_resume(
        thread_id="thread-1",
        session_key="session-1",
        todo_store=TodoStore(),
    )

    assert result == {"status": "none", "note": ""}


def test_real_new_message_only_injects_bundle_without_hydrating_state(monkeypatch):
    store = TodoStore()
    _set_candidates(monkeypatch, ["case:1"])
    _set_cases(monkeypatch, {"case:1": _active_case()})

    result = workspace.prepare_task_workspace_resume(
        thread_id="thread-1",
        session_key="session-1",
        todo_store=store,
        hydrate_local_state=False,
    )

    assert result["status"] == "exact"
    assert result["todo_hydrated"] is False
    assert result["local_todo_state"] == "empty_not_hydrated"
    assert store.read() == []


def test_oversized_resume_note_is_bounded_and_valid_json(monkeypatch):
    case = _active_case()
    long_steps = []
    for index in range(64):
        step_id = f"step-{index}-" + "i" * 180
        long_steps.append({
            "id": step_id,
            "content": "y" * 4_000,
            "status": "in_progress" if index == 0 else "pending",
            "depends_on": [],
            "blocker": {"detail": "b" * 8_000},
        })
    plan = case["workspace"]["plan"]
    plan["plan_id"] = "plan-" + "p" * 400
    plan["objective"] = "x" * 20_000
    plan["steps"] = long_steps
    plan["current_step_id"] = long_steps[0]["id"]
    plan["resume_cursor"] = {
        "next_step_id": long_steps[0]["id"],
        "summary": "s" * 20_000,
        "extra": {"huge": "z" * 50_000},
    }
    case["case_id"] = "case:" + "c" * 20_000
    case["workspace"]["remaining_step_ids"] = [item["id"] for item in long_steps]
    case["next_action"] = {"summary": "n" * 50_000, "extra": ["q" * 20_000]}
    _set_candidates(monkeypatch, [case["case_id"]])
    _set_cases(monkeypatch, {case["case_id"]: case})

    result = workspace.prepare_task_workspace_resume(
        thread_id="thread-1",
        session_key="session-1",
        todo_store=TodoStore(),
    )

    assert len(result["note"]) <= workspace.MAX_RESUME_NOTE_CHARS
    payload = _note_payload(result["note"])
    assert payload["truncated"] is True
    assert payload["plan"]["current_step_id"].startswith("step-0-")
    assert len(payload["remaining_step_ids"]) <= 16


def test_oversized_ambiguous_and_local_conflict_notes_are_bounded_json(monkeypatch):
    cases = {}
    for index in range(3):
        case = _active_case(f"case:{index}:" + "c" * 10_000, f"plan:{index}")
        case["workspace"]["plan"]["objective"] = "o" * 50_000
        case["workspace"]["plan"]["resume_cursor"] = {
            "next_step_id": "2",
            "summary": "s" * 50_000,
            "extra": {"huge": "x" * 50_000},
        }
        cases[case["case_id"]] = case
    _set_candidates(monkeypatch, list(cases))
    _set_cases(monkeypatch, cases)

    result = workspace.prepare_task_workspace_resume(
        thread_id="thread-1",
        session_key="session-1",
        todo_store=TodoStore(),
    )

    assert result["status"] == "ambiguous"
    assert len(result["note"]) <= workspace.MAX_RESUME_NOTE_CHARS
    assert _note_payload(result["note"])["status"] == "ambiguous"


def test_truncated_discovery_never_silently_reports_no_workspace(monkeypatch):
    _set_candidates(monkeypatch, [], truncated=True)

    result = workspace.prepare_task_workspace_resume(
        thread_id="thread-1",
        session_key="session-1",
        todo_store=TodoStore(),
    )

    assert result["status"] == "incomplete"
    assert result["discovery_truncated"] is True
    assert result["reason"] == "candidate_discovery_truncated"
    assert _note_payload(result["note"])["reasons"] == [
        "candidate_discovery_truncated"
    ]


def test_discovery_query_failure_is_explicit_incomplete(monkeypatch):
    _set_candidates(
        monkeypatch,
        [],
        error="canonical_brain_query_failed",
    )

    result = workspace.prepare_task_workspace_resume(
        thread_id="thread-1",
        session_key="session-1",
        todo_store=TodoStore(),
    )

    assert result["status"] == "incomplete"
    assert result["reason"] == "canonical_brain_query_failed"
    assert result["todo_hydrated"] is False
    assert _note_payload(result["note"])["status"] == "incomplete"


def test_resume_case_propagates_incomplete_support(monkeypatch):
    monkeypatch.setattr(
        workspace,
        "_query",
        lambda **kwargs: ({
            "success": True,
            "support_incomplete": True,
            "support": {
                "reasons": ["completed_plan_verification_support_missing"],
            },
            "cases": [_active_case()],
        }, None),
    )

    case, error = workspace._resume_case("case:1", deadline=time.monotonic() + 1)

    assert case == {}
    assert error == (
        "candidate_resume_support_incomplete:"
        "completed_plan_verification_support_missing"
    )


def test_unresolved_candidate_prevents_false_exact_and_hydration(monkeypatch):
    store = TodoStore()
    _set_candidates(monkeypatch, ["case:loaded", "case:failed"])

    def _resume(case_id, *, deadline):
        if case_id == "case:failed":
            return {}, "canonical_brain_query_failed"
        return _active_case("case:loaded", "plan:loaded"), None

    monkeypatch.setattr(workspace, "_resume_case", _resume)

    result = workspace.prepare_task_workspace_resume(
        thread_id="thread-1",
        session_key="session-1",
        todo_store=store,
    )

    assert result["status"] == "incomplete"
    assert result["candidate_case_ids"] == ["case:loaded"]
    assert result["unresolved_case_ids"] == ["case:failed"]
    assert result["todo_hydrated"] is False
    assert store.read() == []
    assert _note_payload(result["note"])["unresolved_cases"][0]["case_id"] == "case:failed"


def test_global_deadline_marks_unchecked_candidates_incomplete(monkeypatch):
    _set_candidates(monkeypatch, ["case:first", "case:second"])
    monotonic_values = iter([0.0, 0.0, workspace.MAX_RECOVERY_SECONDS + 1.0])
    monkeypatch.setattr(workspace.time, "monotonic", lambda: next(monotonic_values))
    calls = []

    def _resume(case_id, *, deadline):
        calls.append(case_id)
        return _active_case(case_id, f"plan:{case_id}"), None

    monkeypatch.setattr(workspace, "_resume_case", _resume)

    result = workspace.prepare_task_workspace_resume(
        thread_id="thread-1",
        session_key="session-1",
        todo_store=TodoStore(),
    )

    assert calls == ["case:first"]
    assert result["status"] == "incomplete"
    assert result["reason"] == "recovery_deadline_exceeded"
    assert result["unresolved_case_ids"] == ["case:second"]
    assert result["todo_hydrated"] is False


def test_single_slow_query_cannot_overrun_shared_deadline(monkeypatch):
    from tools import canonical_brain_tool

    entered = threading.Event()
    release = threading.Event()
    monkeypatch.setattr(
        canonical_brain_tool,
        "check_canonical_brain_requirements",
        lambda: True,
    )

    def _slow_query(**kwargs):
        entered.set()
        release.wait(timeout=1)
        return json.dumps({"success": True, "cases": []})

    monkeypatch.setattr(
        canonical_brain_tool,
        "canonical_brain_query_tool",
        _slow_query,
    )
    started = time.monotonic()
    result, error = workspace._query(
        deadline=started + 0.03,
        thread_id="thread-1",
        view="summary",
        limit=80,
    )
    elapsed = time.monotonic() - started
    release.set()

    assert entered.is_set()
    assert result == {}
    assert error == "recovery_deadline_exceeded"
    assert elapsed < 0.5
