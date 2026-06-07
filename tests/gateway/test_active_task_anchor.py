from gateway.active_task import (
    LCM_ACTIVE_TASK,
    LCM_USER_CORRECTION,
    format_active_task_anchor_block,
    resolve_active_task_anchor,
)
from hermes_state import SessionDB


def test_reply_correction_keeps_watchdog_anchor_over_stale_pyr_lcm(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("watchdog-session", "telegram")

    watchdog_task_id = db.task_open(
        session_id="watchdog-session",
        content="Investigate the watchdog alert in Ops - Runtime Alerts and fix the failing probe.",
        platform="telegram",
        chat_id="-1003772049875",
        thread_id="19714",
        source_message_id="7001",
        lcm_label=LCM_ACTIVE_TASK,
    )
    db.task_open(
        session_id="watchdog-session",
        content="Draft the Plano Young Republicans speaker poster.",
        platform="telegram",
        chat_id="-1003715377988",
        thread_id="1150",
        source_message_id="100",
        lcm_label="PRIOR_INVESTIGATION",
    )

    history = [
        {
            "role": "user",
            "content": "## Active Task\nFinish the PYR speaker poster and use the old event details.",
            "lcm_label": "PRIOR_INVESTIGATION",
        }
    ]

    anchor = resolve_active_task_anchor(
        current_text="No, you bailed and repeated yourself. That is not what I asked.",
        history=history,
        open_tasks=db.get_open_tasks(
            platform="telegram",
            chat_id="-1003772049875",
            thread_id="19714",
        ),
        chat_id="-1003772049875",
        thread_id="19714",
        source_message_id="7002",
        reply_to_message_id="7001",
    )

    assert anchor.correction_mode is True
    assert anchor.source == "reply_to_task_anchor"
    assert anchor.task_id == watchdog_task_id
    assert "watchdog alert" in anchor.text
    assert "PYR" not in anchor.text

    block = format_active_task_anchor_block(anchor)
    assert block.index("Task:") < block.index("Instruction:")
    assert "LCM label: ACTIVE_TASK" in block
    assert "Correction mode: on" in block


def test_user_correction_updates_existing_task_label(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s1", "telegram")
    task_id = db.task_open(
        session_id="s1",
        content="Fix watchdog task routing.",
        platform="telegram",
        chat_id="c1",
        thread_id="t1",
        source_message_id="m1",
    )

    assert db.task_update(
        task_id,
        lcm_label=LCM_USER_CORRECTION,
        reply_to_message_id="m1",
    )

    [task] = db.get_open_tasks(platform="telegram", chat_id="c1", thread_id="t1")
    assert task["lcm_label"] == LCM_USER_CORRECTION
    assert task["reply_to_message_id"] == "m1"
    assert task["content"] == "Fix watchdog task routing."


def test_new_instruction_beats_existing_same_thread_task():
    anchor = resolve_active_task_anchor(
        current_text="Check the deployment logs and tell me what failed.",
        open_tasks=[
            {
                "task_id": "task_1",
                "content": "Finish the older unrelated investigation.",
                "chat_id": "c1",
                "thread_id": "t1",
                "source_message_id": "m1",
                "lcm_label": LCM_ACTIVE_TASK,
            }
        ],
        chat_id="c1",
        thread_id="t1",
        source_message_id="m2",
    )

    assert anchor.source == "explicit_latest_instruction"
    assert anchor.task_id is None
    assert "deployment logs" in anchor.text
