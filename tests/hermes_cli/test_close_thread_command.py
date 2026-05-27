from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from hermes_cli import kanban_db as kb
from hermes_cli.close_thread import build_close_thread_packet
from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS, gateway_help_lines, resolve_command


FIXED_NOW = datetime(2026, 5, 24, 12, 0, 0, tzinfo=timezone.utc)


def _init_test_kanban(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    home = tmp_path / "hermes-home"
    office_root = tmp_path / "foundation-discord-office"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("FOUNDATION_DISCORD_OFFICE_ROOT", str(office_root))
    db_path = home / "kanban.db"
    kb.init_db(db_path)
    return home, db_path


def test_close_thread_command_is_registry_driven():
    assert resolve_command("close-thread").name == "close-thread"
    assert resolve_command("/close-thread").name == "close-thread"
    assert resolve_command("close-tread").name == "close-thread"
    assert "close-thread" in GATEWAY_KNOWN_COMMANDS

    help_lines = gateway_help_lines()
    assert any("/close-thread" in line for line in help_lines)


def test_close_thread_packet_writes_artifacts_and_redacts_secrets(tmp_path, monkeypatch):
    _, db_path = _init_test_kanban(tmp_path, monkeypatch)

    conn = kb.connect(db_path)
    task_id = kb.create_task(
        conn,
        title="Ship close-thread command",
        body="Decision: ship the command. token: super-secret-token",
        assignee="codeworker",
        workspace_kind="dir",
        workspace_path="/tmp/close-thread-work",
    )
    kb.add_comment(conn, task_id, "reviewer", "Decision: proceed with API_KEY=shhh-secret-value")
    kb.complete_task(
        conn,
        task_id,
        summary="Approved release with 4 tests passing",
        metadata={
            "changed_files": ["/tmp/close-thread-work/cli.py"],
            "tests_run": 4,
            "tests_passed": 4,
        },
    )
    conn.close()

    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    packet, response = build_close_thread_packet(
        "/close-thread",
        source={"requested_by": "HD", "channel_name": "human-review", "thread_name": "Foundation Office"},
        session={"profile": "codeworker", "command_surface": "gateway", "run_id": "r-123"},
        now=FIXED_NOW,
    )

    assert packet["status"] in {"GREEN", "YELLOW"}
    assert packet["missing_durable_anchor"] is False
    assert packet["human_review_packets"]["queued"] == []
    assert packet["closeout_artifacts"]["json_path"]
    assert packet["closeout_artifacts"]["markdown_path"]
    assert "Artifact: " in response
    assert f"Durable anchor: Kanban {task_id}" in response

    json_path = Path(packet["closeout_artifacts"]["json_path"])
    md_path = Path(packet["closeout_artifacts"]["markdown_path"])
    assert json_path.exists()
    assert md_path.exists()

    saved = json.loads(json_path.read_text())
    serialized = json.dumps(saved)
    assert "super-secret-token" not in serialized
    assert "shhh-secret-value" not in serialized
    assert "[REDACTED]" in serialized
    assert saved["kanban_refs"]["task"]["id"] == task_id
    assert saved["evidence"]["tests_or_smokes"][0]["tests_run"] == 4


def test_close_thread_packet_blocks_when_anchor_missing(tmp_path, monkeypatch):
    _init_test_kanban(tmp_path, monkeypatch)

    packet, response = build_close_thread_packet(
        "/close-thread --task-id t_missing_anchor",
        source={"requested_by": "HD"},
        session={"profile": "gateway"},
        now=FIXED_NOW,
    )

    assert packet["status"] == "BLOCKED"
    assert packet["missing_durable_anchor"] is True
    assert packet["human_review_packets"]["queued"]
    assert packet["human_review_packets"]["queued"][0]["packet_type"] == "BLOCKED_NEEDS_HUMAN"
    assert "Durable anchor: missing task t_missing_anchor" in response


def test_close_thread_packet_preserves_thread_labels_as_metadata(tmp_path, monkeypatch):
    _, db_path = _init_test_kanban(tmp_path, monkeypatch)
    conn = kb.connect(db_path)
    task_id = kb.create_task(
        conn,
        title="Close labelled worktree thread",
        assignee="codeworker",
        workspace_kind="dir",
        workspace_path="/tmp/labelled-thread-work",
    )
    kb.complete_task(conn, task_id, summary="Labelled thread ready for closeout", metadata={})
    conn.close()

    packet, _response = build_close_thread_packet(
        f"/close-thread --task-id {task_id}",
        source={
            "requested_by": "HD",
            "thread_id": "555",
            "thread_name": "[wtupdate] [wtsignoff] Routing status",
        },
        session={"profile": "gateway"},
        now=FIXED_NOW,
    )

    discord_ref = packet["source_refs"]["discord"]
    assert [label["id"] for label in discord_ref["thread_labels"]] == [
        "wtupdate",
        "wtsignoff",
    ]
    assert packet["machine_metadata"]["thread_label_ids"] == ["wtupdate", "wtsignoff"]
