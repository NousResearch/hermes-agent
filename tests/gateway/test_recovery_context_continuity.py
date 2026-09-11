"""Recovery scaffolding must not replace the human task on replay or compaction."""

from copy import deepcopy
from datetime import datetime, timedelta
from types import SimpleNamespace
import time

import pytest

from agent.context_compressor import ContextCompressor
from gateway.run import _build_gateway_agent_history
from gateway.resume_recovery import build_resume_recovery_note
from gateway.run_turn_runner import TurnRunner
from gateway.turn_context import TurnContext


@pytest.mark.parametrize("reason", ["restart_timeout", "shutdown_timeout", "restart_interrupted"])
@pytest.mark.parametrize("timestamps", [False, True])
def test_restart_replay_keeps_human_task_as_compaction_anchor(reason, timestamps):
    task = "Implement the parser fix and validate its output."
    history = [
        {"role": "user", "content": task},
        {"role": "assistant", "content": "Implementation is underway."},
        {"role": "user", "content": build_resume_recovery_note(reason), "timestamp": time.time()},
        {"role": "assistant", "content": "Continuing validation."},
    ]
    original = deepcopy(history)

    replay, _ = _build_gateway_agent_history(history, inject_timestamps=timestamps)
    snapshot = ContextCompressor._latest_user_task_snapshot(replay)

    assert snapshot is not None and task in snapshot
    assert "gateway is now back online" not in snapshot
    assert history == original  # Clean the API replay, never the durable transcript.


def test_replay_strips_stacked_recovery_notes_but_keeps_real_followup():
    task = "Stop implementation; report what changed."
    wrapped = build_resume_recovery_note(
        "restart_timeout", build_resume_recovery_note("shutdown_timeout", task)
    )
    replay, _ = _build_gateway_agent_history([
        {"role": "user", "content": wrapped, "api_content": wrapped + "\nold context"},
    ])

    assert replay == [{"role": "user", "content": task}]


def test_replay_preserves_notification_provenance_before_compaction():
    task = "Finish the parser fix."
    notice = {
        "role": "user", "content": "Background validation completed.",
        "display_kind": "internal_notification",
        "display_metadata": {"process_id": "fixture-process"},
    }
    replay, _ = _build_gateway_agent_history([
        {"role": "user", "content": task}, notice,
    ])

    assert replay[-1] == notice
    assert ContextCompressor._find_inflight_user_task(replay)["content"] == task


@pytest.mark.parametrize("message", [
    "Continue the interrupted work.",
    "Stop; do not make further changes.",
    "What has finished? Report only.",
    "Start a new task: explain the parser API.",
])
def test_tool_tail_recovery_uses_same_policy_and_evidence_as_marked_restart(message):
    history = [
        {"role": "user", "content": "Implement and validate the parser."},
        {"role": "assistant", "tool_calls": [{
            "id": "verified", "function": {"name": "terminal", "arguments": "{}"},
        }]},
        {"role": "tool", "tool_call_id": "verified", "content": "Validation passed.",
         "timestamp": time.time()},
    ]
    replay, _ = _build_gateway_agent_history(history)
    runner = SimpleNamespace(
        session_store=SimpleNamespace(_entries={}),
        _adapter_for_source=lambda source: None,
    )
    ctx = TurnContext(message=message, history=history, session_key="fixture")
    persist, _ = TurnRunner(runner, ctx)._prepare_turn_message(replay)

    assert persist == message
    assert ctx.message == build_resume_recovery_note(None, message)
    assert "IGNORE those pending results" not in ctx.message
    assert replay[-1]["content"] == "Validation passed."


def test_restart_recovery_allows_verification_without_repeating_side_effects():
    note = build_resume_recovery_note("restart_timeout", "Verify the deployed revision.")

    assert "do NOT re-execute or verify it" not in note
    assert "read-only" in note
    assert "UNKNOWN" in note


@pytest.mark.parametrize("age_seconds", [0, 7200])
def test_pending_model_notice_does_not_turn_auto_resume_into_a_new_human_request(age_seconds):
    now = datetime.now() - timedelta(seconds=age_seconds)
    runner = SimpleNamespace(
        session_store=SimpleNamespace(_entries={"fixture": SimpleNamespace(
            resume_pending=True, resume_reason="restart_timeout", last_resume_marked_at=now,
        )}),
        _pending_model_notes={"fixture": "[The model changed.]"},
        _adapter_for_source=lambda source: None,
    )
    history = [{"role": "assistant", "content": "Working.", "timestamp": time.time() - age_seconds}]
    ctx = TurnContext(
        message="", history=history, session_key="fixture", persist_user_display_kind="internal_notification",
    )
    persist, _ = TurnRunner(runner, ctx)._prepare_turn_message([])

    assert "CONTINUE the interrupted task" in ctx.message
    assert "NEW message below" not in ctx.message
    assert "[The model changed.]" in ctx.message
    assert persist == build_resume_recovery_note("restart_timeout")


@pytest.mark.asyncio
@pytest.mark.parametrize("resume_pending", [False, True])
async def test_captionless_image_after_interruption_remains_new_input(tmp_path, resume_pending):
    """Native preprocessing keeps an empty caption; recovery must still honor the image."""
    import base64
    from gateway.config import Platform
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource

    image = tmp_path / "fixture.png"
    image.write_bytes(base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jF9kAAAAASUVORK5CYII="
    ))
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="fixture", user_id="fixture")
    event = MessageEvent(
        text="", source=source, message_type=MessageType.PHOTO,
        media_urls=[str(image)], media_types=["image/png"],
    )
    state = SimpleNamespace(persistent=SimpleNamespace(native_image_paths=[]))
    entry = SimpleNamespace(
        resume_pending=resume_pending, resume_reason="restart_timeout", last_resume_marked_at=datetime.now(),
    )
    runner = SimpleNamespace(
        session_store=SimpleNamespace(_entries={"fixture": entry}),
        _adapter_for_source=lambda source: None,
        _session_state=lambda key: state,
        _decide_image_input_mode=lambda **kwargs: "native",
        _consume_pending_native_image_paths=lambda key: state.persistent.native_image_paths,
    )
    caption = await GatewayRunner._enrich_inbound_images(runner, source, "fixture", event.text, event.media_urls)
    history = [{"role": "tool", "content": "Earlier validation completed.", "timestamp": time.time()}]
    ctx = TurnContext(
        message=caption, history=history, session_key="fixture",
        persist_user_display_kind="internal_notification" if event.internal else None,
    )
    turn = TurnRunner(runner, ctx)
    persist, _ = turn._prepare_turn_message(history)
    api_message = turn._native_image_run_message()

    assert caption == "" and ctx.message == ""
    assert persist is None  # Do not persist operational guidance as the human's caption.
    assert any(part["type"] == "image_url" for part in api_message)
    text = "\n".join(part.get("text", "") for part in api_message)
    assert "No new user message" not in text
    assert "CONTINUE the interrupted task" not in text
