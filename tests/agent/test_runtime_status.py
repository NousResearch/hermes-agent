from agent import runtime_status


def test_snapshot_defaults_are_safe():
    runtime_status.clear_session("missing-session")

    snap = runtime_status.snapshot("missing-session")

    assert snap["phase"] == "idle"
    assert snap["run_mode"] == "idle"
    assert snap["target"] == ""
    assert snap["main_agent"] == "main"
    assert snap["recent_tool"] is None
    assert snap["recent_skill"] is None
    assert snap["task"] is None
    assert snap["wait"] == {"reason": "none", "since": None}


def test_tool_lifecycle_records_recent_status():
    runtime_status.clear_session("s1")

    runtime_status.record_tool_started("s1", "terminal", preview="pytest")
    running = runtime_status.snapshot("s1")["recent_tool"]
    assert running["name"] == "terminal"
    assert running["status"] == "running"
    assert running["preview"] == "pytest"
    snap = runtime_status.snapshot("s1")
    assert snap["phase"] == "tool"
    assert snap["run_mode"] == "agent"
    assert snap["wait"]["reason"] == "tool:terminal"

    runtime_status.record_tool_completed("s1", "terminal", status="ok", duration_ms=123)
    done = runtime_status.snapshot("s1")["recent_tool"]
    assert done["name"] == "terminal"
    assert done["status"] == "ok"
    assert done["duration_ms"] == 123
    snap = runtime_status.snapshot("s1")
    assert snap["phase"] == "running"
    assert snap["wait"] == {"reason": "none", "since": None}


def test_ring_buffer_keeps_latest_tool():
    runtime_status.clear_session("s2")

    for idx in range(20):
        runtime_status.record_tool_completed("s2", f"tool{idx}", status="ok")

    snap = runtime_status.snapshot("s2")
    assert snap["recent_tool"]["name"] == "tool19"
    assert len(snap["recent_tools"]) <= 8


def test_task_summary_is_normalized():
    runtime_status.clear_session("s3")

    runtime_status.record_task_summary(
        "s3",
        {"total": 7, "completed": 3, "in_progress": 1, "pending": 3},
    )

    assert runtime_status.snapshot("s3")["task"] == {
        "total": 7,
        "completed": 3,
        "in_progress": 1,
        "pending": 3,
        "cancelled": 0,
    }


def test_skill_and_wait_state_are_recorded():
    runtime_status.clear_session("s4")

    runtime_status.record_skill("s4", "hermes-agent", event="view")
    runtime_status.record_wait("s4", reason="thinking")

    snap = runtime_status.snapshot("s4")
    assert snap["recent_skill"]["name"] == "hermes-agent"
    assert snap["recent_skill"]["event"] == "view"
    assert snap["wait"]["reason"] == "thinking"
    assert snap["wait"]["since"] is not None


def test_record_wait_updates_phase_and_preserves_explicit_run_mode():
    runtime_status.clear_session("s-wait")
    runtime_status.record_phase("s-wait", run_mode="implement", phase="running")

    runtime_status.record_wait("s-wait", reason="approval")
    snap = runtime_status.snapshot("s-wait")
    assert snap["phase"] == "waiting"
    assert snap["run_mode"] == "implement"
    assert snap["wait"]["reason"] == "approval"

    runtime_status.record_wait("s-wait", reason="none")
    snap = runtime_status.snapshot("s-wait")
    assert snap["phase"] == "running"
    assert snap["run_mode"] == "implement"
    assert snap["wait"] == {"reason": "none", "since": None}

    runtime_status.record_wait("s4", reason="none")
    assert runtime_status.snapshot("s4")["wait"] == {"reason": "none", "since": None}


def test_summarize_active_subagents_prefers_running_with_last_tool():
    records = [
        {
            "subagent_id": "sa-old",
            "goal": "Older task",
            "status": "running",
            "started_at": 10,
            "tool_count": 0,
        },
        {
            "subagent_id": "sa-tool",
            "goal": "Verify statusbar implementation details",
            "status": "running",
            "started_at": 5,
            "tool_count": 2,
            "last_tool": "terminal",
        },
        {
            "subagent_id": "sa-done",
            "goal": "Done task",
            "status": "completed",
            "started_at": 20,
            "tool_count": 9,
        },
    ]

    summary = runtime_status.summarize_active_subagents(records)

    assert summary == {
        "id": "sa-tool",
        "label": "Verify statusbar implementation details",
        "status": "running",
        "last_tool": "terminal",
        "tool_count": 2,
    }


def test_record_active_subagents_adds_summary_to_snapshot():
    runtime_status.clear_session("s-sub")

    runtime_status.record_active_subagents(
        "s-sub",
        [
            {
                "subagent_id": "sa-1",
                "goal": "Review code quality",
                "status": "running",
                "started_at": 1,
                "last_tool": "read_file",
                "tool_count": 1,
            }
        ],
    )

    assert runtime_status.snapshot("s-sub")["active_subagent"] == {
        "id": "sa-1",
        "label": "Review code quality",
        "status": "running",
        "last_tool": "read_file",
        "tool_count": 1,
    }


def test_record_background_process_count_adds_runtime_summary():
    runtime_status.clear_session("s-bg")

    runtime_status.record_background_process_count("s-bg", 3)

    assert runtime_status.snapshot("s-bg")["background_tasks"] == {"running": 3}


def test_record_activity_summary_maps_current_tool_to_phase_and_wait():
    runtime_status.clear_session("s-activity-tool")

    runtime_status.record_activity_summary(
        "s-activity-tool",
        {
            "current_tool": "terminal",
            "last_activity_desc": "running tool terminal",
            "api_call_count": 2,
            "budget_used": 3,
            "budget_max": 9,
        },
    )

    snap = runtime_status.snapshot("s-activity-tool")
    assert snap["phase"] == "tool"
    assert snap["run_mode"] == "agent"
    assert snap["wait"]["reason"] == "tool:terminal"
    assert snap["task"] == {"total": 9, "completed": 3, "in_progress": 0, "pending": 6, "cancelled": 0}


def test_record_activity_summary_maps_model_and_backoff_waits():
    runtime_status.clear_session("s-activity-model")

    runtime_status.record_activity_summary(
        "s-activity-model",
        {"current_tool": None, "last_activity_desc": "starting API call #4"},
    )
    snap = runtime_status.snapshot("s-activity-model")
    assert snap["phase"] == "thinking"
    assert snap["wait"]["reason"] == "model"

    runtime_status.record_activity_summary(
        "s-activity-model",
        {"current_tool": None, "last_activity_desc": "error retry backoff (2/5), 30s remaining"},
    )
    snap = runtime_status.snapshot("s-activity-model")
    assert snap["phase"] == "waiting"
    assert snap["wait"]["reason"] == "retry_backoff"
