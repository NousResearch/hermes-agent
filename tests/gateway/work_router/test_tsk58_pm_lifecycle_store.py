from gateway.work_router.models import WorkExecution
from gateway.work_router.store import RouterStore


def test_pm_lifecycle_execution_roundtrip(tmp_path):
    db = tmp_path / "router.db"
    store = RouterStore(str(db))

    execution = WorkExecution(
        directive_id="tsk58-test",
        source_event_id="evt-1",
        slack_team_id="T1",
        channel_id="C1",
        thread_ts="1.000001",
        directive_ts="1.000001",
        owner_profile="Hans",
        owner_slack_user_id="UHANS",
        state="PM_REVIEW",
        review_generation=0,
        approval_required=False,
    )

    store.upsert_work_execution(execution)

    loaded = store.get_work_execution("evt-1")

    assert loaded is not None
    assert loaded.owner_profile == "Hans"
    assert loaded.state == "PM_REVIEW"
    assert loaded.review_generation == 0
    assert loaded.approval_required is False


def test_pm_rework_increments_generation(tmp_path):
    db = tmp_path / "router.db"
    store = RouterStore(str(db))

    execution = WorkExecution(
        directive_id="tsk58-test",
        source_event_id="evt-2",
        slack_team_id="T1",
        channel_id="C1",
        thread_ts="2.000001",
        directive_ts="2.000001",
        owner_profile="Hans",
        owner_slack_user_id="UHANS",
        state="PM_REVIEW",
        review_generation=0,
        approval_required=False,
    )

    store.upsert_work_execution(execution)
    updated = store.transition_work_execution(
        "evt-2",
        state="OWNER_REWORK",
        review_generation=1,
    )

    assert updated.state == "OWNER_REWORK"
    assert updated.review_generation == 1
    assert updated.owner_profile == "Hans"


def test_pm_decision_gate_marks_approval_required(tmp_path):
    db = tmp_path / "router.db"
    store = RouterStore(str(db))

    execution = WorkExecution(
        directive_id="tsk58-test",
        source_event_id="evt-3",
        slack_team_id="T1",
        channel_id="C1",
        thread_ts="3.000001",
        directive_ts="3.000001",
        owner_profile="Hans",
        owner_slack_user_id="UHANS",
        state="PM_REVIEW",
        review_generation=1,
        approval_required=False,
    )

    store.upsert_work_execution(execution)
    updated = store.transition_work_execution(
        "evt-3",
        state="WAIT_SINCLAIR_DECISION",
        approval_required=True,
    )

    assert updated.state == "WAIT_SINCLAIR_DECISION"
    assert updated.approval_required is True
    assert updated.review_generation == 1



def test_get_active_work_execution_for_thread(tmp_path):
    db = tmp_path / "router.db"
    store = RouterStore(str(db))

    execution = WorkExecution(
        directive_id="tsk58-thread-test",
        source_event_id="original-owner-event",
        slack_team_id="T1",
        channel_id="C1",
        thread_ts="10.000001",
        directive_ts="10.000001",
        owner_profile="Hans",
        owner_slack_user_id="UHANS",
        state="PM_REVIEW",
        review_generation=2,
        approval_required=False,
    )

    store.upsert_work_execution(execution)

    loaded = store.get_active_work_execution_for_thread(
        "C1",
        "10.000001",
    )

    assert loaded is not None
    assert loaded.source_event_id == "original-owner-event"
    assert loaded.owner_profile == "Hans"
    assert loaded.state == "PM_REVIEW"
    assert loaded.review_generation == 2


def test_completed_work_execution_is_not_active(tmp_path):
    db = tmp_path / "router.db"
    store = RouterStore(str(db))

    execution = WorkExecution(
        directive_id="tsk58-complete-test",
        source_event_id="completed-event",
        slack_team_id="T1",
        channel_id="C2",
        thread_ts="20.000001",
        directive_ts="20.000001",
        owner_profile="Hans",
        owner_slack_user_id="UHANS",
        state="FINAL_RESULT",
        review_generation=1,
        approval_required=False,
    )

    store.upsert_work_execution(execution)

    loaded = store.get_active_work_execution_for_thread(
        "C2",
        "20.000001",
    )

    assert loaded is None
