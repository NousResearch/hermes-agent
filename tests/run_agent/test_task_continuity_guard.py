"""Regression tests for post-compression task-vector drift protection."""

import inspect
from pathlib import Path
from types import SimpleNamespace

from agent.task_continuity import (
    CONTINUITY_CHECK_PREFIX,
    capture_current_task_frame,
    classify_message_type,
    detect_task_state_conflict,
    format_continuity_check_response,
    task_frame_to_dict,
)
from hermes_state import SessionDB
from run_agent import AIAgent
from tools.todo_tool import TodoStore


def test_classifies_preserved_task_list_as_not_real_user_prompt():
    msg = {
        "role": "user",
        "content": "Current todo list:\n- [in_progress] Policy edit",
        "message_type": "preserved_task_list",
    }

    assert classify_message_type(msg) == "preserved_task_list"


def test_session_db_persists_message_type(tmp_path: Path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s1", source="cli")

    db.append_message(
        "s1",
        role="user",
        content="[Your active task list was preserved across context compression]",
        message_type="preserved_task_list",
    )

    messages = db.get_messages("s1")
    assert messages[0]["message_type"] == "preserved_task_list"


def test_session_db_persists_continuity_frame(tmp_path: Path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s1", source="cli")
    frame = capture_current_task_frame(
        [{"role": "user", "content": "Open config.yaml", "message_type": "real_user_prompt"}],
        current_todo_items=[
            {"id": "policy-update", "content": "Edit config.yaml", "status": "in_progress"},
        ],
        compression_count=2,
    )

    db.append_message(
        "s1",
        role="user",
        content="Open config.yaml",
        continuity_frame=task_frame_to_dict(frame),
    )

    messages = db.get_messages("s1")
    assert messages[0]["continuity_frame"]["active_todo_ids"] == ["policy-update"]
    assert messages[0]["continuity_frame"]["latest_user_surfaces"] == ["config.yaml"]


def test_post_compression_task_vector_drift_blocks_tool_execution():
    """Task A from the real user must not be silently displaced by task B context.

    Scenario mirrors the observed failure:
    - latest real user prompt asks for task A
    - structured context injection and preserved todo state mention task B
    - after repeated compression the assistant appears poised to continue B
    - expected behavior is a continuity check, not tool execution
    """
    messages = [
        {
            "role": "user",
            "content": "Check Hermes gateway web search and web fetch.",
            "message_type": "real_user_prompt",
        },
        {
            "role": "user",
            "content": "[STRUCTURED CONTEXT] Resume project policy update.",
            "message_type": "structured_context_injection",
        },
        {
            "role": "assistant",
            "content": "I will edit config.yaml for the release checklist.",
        },
        {
            "role": "user",
            "content": "Current todo list:\n- policy-update: in_progress\n- gateway-check: pending",
            "message_type": "preserved_task_list",
        },
    ]
    todo_snapshot = [
        {"id": "policy-update", "content": "Edit config.yaml", "status": "in_progress"},
        {"id": "gateway-check", "content": "Check gateway tools", "status": "pending"},
    ]

    frame = capture_current_task_frame(
        messages,
        current_todo_items=todo_snapshot,
        compression_count=8,
    )
    conflict = detect_task_state_conflict(
        frame,
        latest_real_user_message="Check Hermes gateway web search and web fetch.",
        preserved_active_task_list=todo_snapshot,
        structured_context="Resume project policy update.",
        background_notifications=[],
    )

    assert conflict.should_block_tools is True
    assert conflict.code == "task_vector_drift"
    response = format_continuity_check_response(conflict)
    assert response.startswith(CONTINUITY_CHECK_PREFIX)
    assert "Choose A/B/C" in response
    assert "latest real user" in response
    assert "preserved active task" in response


def test_explicit_active_todo_id_resume_does_not_block_on_low_text_overlap():
    messages = [
        {
            "role": "user",
            "content": "Check Hermes gateway web search and web fetch.",
            "message_type": "real_user_prompt",
        },
        {
            "role": "assistant",
            "content": "I will edit config.yaml for the release checklist.",
        },
    ]
    todo_snapshot = [
        {"id": "policy-update", "content": "Edit config.yaml", "status": "in_progress"},
    ]

    frame = capture_current_task_frame(
        messages,
        current_todo_items=todo_snapshot,
        compression_count=8,
        latest_real_user_message="Continue policy-update.",
    )
    conflict = detect_task_state_conflict(
        frame,
        latest_real_user_message="Continue policy-update.",
        preserved_active_task_list=todo_snapshot,
        structured_context="Resume project policy update.",
        background_notifications=[],
    )

    assert conflict.should_block_tools is False


def test_latest_real_user_protected_surface_authorization_does_not_block():
    messages = [
        {
            "role": "user",
            "content": "Check Hermes gateway web search and web fetch.",
            "message_type": "real_user_prompt",
        },
        {
            "role": "assistant",
            "content": "I will edit config.yaml for the release checklist.",
        },
    ]
    todo_snapshot = [
        {"id": "policy-update", "content": "Edit config.yaml", "status": "in_progress"},
    ]

    frame = capture_current_task_frame(
        messages,
        current_todo_items=todo_snapshot,
        compression_count=8,
        latest_real_user_message="Open config.yaml and verify the setting.",
    )
    conflict = detect_task_state_conflict(
        frame,
        latest_real_user_message="Open config.yaml and verify the setting.",
        preserved_active_task_list=todo_snapshot,
        structured_context="Resume project policy update.",
        background_notifications=[],
    )

    assert conflict.should_block_tools is False


def test_run_agent_records_pending_conflict_after_compression():
    store = TodoStore()
    store.write(
        [
            {"id": "policy-update", "content": "Edit config.yaml", "status": "in_progress"},
            {"id": "gateway-check", "content": "Check gateway tools", "status": "pending"},
        ]
    )
    frame = capture_current_task_frame(
        [
            {"role": "user", "content": "Check Hermes gateway web search and web fetch.", "message_type": "real_user_prompt"},
            {"role": "assistant", "content": "I will edit config.yaml for the release checklist."},
        ],
        current_todo_items=store.read(),
        compression_count=8,
        latest_real_user_message="Check Hermes gateway web search and web fetch.",
    )
    agent = SimpleNamespace(
        _compressed_this_turn=True,
        _last_compression_task_frame=frame,
        _post_compression_task_conflict=None,
        _todo_store=store,
    )

    AIAgent._record_post_compression_task_conflict(
        agent,
        [
            {
                "role": "user",
                "content": "[STRUCTURED CONTEXT] Resume project policy update.",
                "message_type": "structured_context_injection",
            }
        ],
        latest_real_user_message="Check Hermes gateway web search and web fetch.",
    )

    assert agent._post_compression_task_conflict is not None
    assert agent._post_compression_task_conflict.should_block_tools is True


def test_run_conversation_sets_original_user_message_before_continuity_state():
    src = inspect.getsource(AIAgent.run_conversation)

    assert src.index("original_user_message =") < src.index("self._current_real_user_message =")


def test_run_conversation_strips_continuity_frame_before_provider_api():
    src = inspect.getsource(AIAgent.run_conversation)

    assert 'api_msg.pop("continuity_frame", None)' in src
