from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agent.attachment_context import (
    bind_current_turn_attachments,
    get_current_turn_attachments,
)
from tools.thread_context import propagate_context_to_thread


def test_cli_single_attachment_is_exact_current_turn_snapshot(tmp_path):
    path = tmp_path / "cli.png"
    path.write_bytes(b"cli")

    with bind_current_turn_attachments(
        [path], session_id="cli-session", task_id="cli-task", turn_id="cli-turn", surface="cli"
    ):
        current = get_current_turn_attachments(session_id="cli-session", task_id="cli-task")

    assert len(current) == 1
    assert current[0]["path"] == str(path)
    assert current[0]["reference"] == "hermes:cli-session:cli-turn:attachment:0"
    assert current[0]["metadata"]["surface"] == "cli"
    assert current[0]["metadata"]["turn_id"] == "cli-turn"


def test_desktop_single_attachment_is_exact_materialized_snapshot(tmp_path):
    path = tmp_path / "desktop.png"
    path.write_bytes(b"desktop")

    with bind_current_turn_attachments(
        [{"path": str(path), "reference": "desktop-ref", "metadata": {"mime_type": "image/png"}}],
        session_id="desktop-session",
        task_id="desktop-task",
        turn_id="desktop-turn",
        surface="desktop",
    ):
        current = get_current_turn_attachments(
            session_id="desktop-session", task_id="desktop-task"
        )

    assert current[0]["path"] == str(path)
    assert current[0]["reference"] == "desktop-ref"
    assert current[0]["metadata"]["mime_type"] == "image/png"


def test_zero_attachments_is_empty():
    with bind_current_turn_attachments(
        [], session_id="s", task_id="t", turn_id="turn", surface="cli"
    ):
        assert get_current_turn_attachments(session_id="s", task_id="t") == ()


def test_multiple_attachments_remain_explicit_snapshot(tmp_path):
    one = tmp_path / "one.png"
    two = tmp_path / "two.png"
    one.write_bytes(b"1")
    two.write_bytes(b"2")

    with bind_current_turn_attachments(
        [one, two], session_id="s", task_id="t", turn_id="turn", surface="desktop"
    ):
        current = get_current_turn_attachments(session_id="s", task_id="t")

    assert [item["path"] for item in current] == [str(one), str(two)]


def test_previous_turn_cannot_leak_into_next_turn(tmp_path):
    path = tmp_path / "old.png"
    path.write_bytes(b"old")

    with bind_current_turn_attachments(
        [path], session_id="s", task_id="old-task", turn_id="old-turn", surface="cli"
    ):
        assert len(get_current_turn_attachments(session_id="s", task_id="old-task")) == 1

    with bind_current_turn_attachments(
        [], session_id="s", task_id="new-task", turn_id="new-turn", surface="cli"
    ):
        assert get_current_turn_attachments(session_id="s", task_id="new-task") == ()


def test_other_session_cannot_read_attachment(tmp_path):
    path = tmp_path / "private.png"
    path.write_bytes(b"private")

    with bind_current_turn_attachments(
        [path], session_id="session-a", task_id="task-a", turn_id="turn-a", surface="desktop"
    ):
        assert get_current_turn_attachments(session_id="session-b", task_id="task-a") == ()
        assert get_current_turn_attachments(session_id="session-a", task_id="task-b") == ()
        assert get_current_turn_attachments(
            session_id="session-a", task_id="task-a", turn_id="turn-b"
        ) == ()


def test_detached_attachment_is_not_selected_by_next_snapshot(tmp_path):
    path = tmp_path / "detached.png"
    path.write_bytes(b"detached")

    with bind_current_turn_attachments(
        [path], session_id="s", task_id="turn-a", turn_id="turn-a", surface="desktop"
    ):
        assert len(get_current_turn_attachments(session_id="s", task_id="turn-a")) == 1

    # The gateway detach operation removes the path before the next prompt
    # snapshot; an empty next-turn input is therefore not allowed to inherit it.
    with bind_current_turn_attachments(
        [], session_id="s", task_id="turn-b", turn_id="turn-b", surface="desktop"
    ):
        assert get_current_turn_attachments(session_id="s", task_id="turn-b") == ()


def test_snapshot_propagates_to_tool_worker_context(tmp_path):
    path = tmp_path / "worker.png"
    path.write_bytes(b"worker")

    with bind_current_turn_attachments(
        [path], session_id="s", task_id="t", turn_id="turn", surface="cli"
    ):
        with ThreadPoolExecutor(max_workers=1) as executor:
            current = executor.submit(
                propagate_context_to_thread(
                    lambda: get_current_turn_attachments(session_id="s", task_id="t")
                )
            ).result()

    assert current[0]["path"] == str(path)


def test_context_resets_after_failure(tmp_path):
    path = tmp_path / "failed.png"
    path.write_bytes(b"failed")

    try:
        with bind_current_turn_attachments(
            [path], session_id="s", task_id="t", turn_id="turn", surface="cli"
        ):
            raise RuntimeError("turn failed")
    except RuntimeError:
        pass

    assert get_current_turn_attachments() == ()
