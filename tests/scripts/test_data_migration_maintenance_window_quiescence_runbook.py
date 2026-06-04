from __future__ import annotations

import re
from pathlib import Path


RUNBOOK = (
    Path(__file__).resolve().parents[2]
    / "notes"
    / "data-migration"
    / "maintenance-window-writer-quiescence-runbook.md"
)


def _read_runbook() -> str:
    return RUNBOOK.read_text(encoding="utf-8")


def _section(markdown: str, heading: str) -> str:
    match = re.search(
        rf"^## {re.escape(heading)}\n(?P<body>.*?)(?=^## |\Z)",
        markdown,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"missing section: {heading}"
    return match.group("body")


def _assert_contains_all(text: str, expected: list[str]) -> None:
    missing = [item for item in expected if item not in text]
    assert not missing, f"missing runbook requirements: {missing}"


def _assert_in_order(text: str, expected: list[str]) -> None:
    cursor = 0
    for item in expected:
        next_index = text.find(item, cursor)
        assert next_index != -1, f"missing or out-of-order item: {item}"
        cursor = next_index + len(item)


def test_had_1150_runbook_names_scope_and_non_mutation_guards() -> None:
    text = _read_runbook()
    scope = _section(text, "1. Scope and Safety Invariants")

    _assert_contains_all(
        text,
        [
            "# Data Migration Maintenance Window and Writer Quiescence Runbook",
            "Issue: HAD-1150",
            "It does not execute the migration",
            "does not replace the migration command plan",
        ],
    )
    _assert_contains_all(
        scope,
        [
            "`/home/david/.hermes`",
            "`/home/david/stacks`",
            "`/home/david/.ctx-data`",
            "`/home/david/.codex`",
            "planned `/data/hermes` paths",
            "Do not copy SQLite databases, WAL files, SHM files, repo `.git` state",
            "Do not start copy, rename, mount, bind-mount, delete, ownership, or permission",
            "Treat missing inventory as blocking",
            "Treat active unknown writers as blocking",
        ],
    )


def test_had_1150_preflight_gate_fails_closed_before_copy() -> None:
    preflight = _section(_read_runbook(), "4. Preflight Gate")

    _assert_contains_all(
        preflight,
        [
            "before any copy, rename, mount",
            "It fails closed when unexpected writers are active.",
            "Run it from `/tmp`",
            ': "${MIGRATION_RUN_ID:?set MIGRATION_RUN_ID before the window}"',
            ': "${MIGRATION_EVIDENCE_DIR:?set MIGRATION_EVIDENCE_DIR before the window}"',
            "systemctl --user is-active --quiet hermes-gateway.service",
            "open-handles.txt",
            "docker-bind-writers.txt",
            "active-runtime-processes.txt",
            "sqlite-wal-shm.txt",
            "dirty-repos.txt",
            "PREFLIGHT PASS: writer quiescence gate passed",
        ],
    )
    _assert_in_order(
        preflight,
        [
            "systemctl --user is-active --quiet hermes-gateway.service",
            "open-handles.txt",
            "docker-bind-writers.txt",
            "active-runtime-processes.txt",
            "sqlite-wal-shm.txt",
            "dirty-repos.txt",
            "PREFLIGHT PASS: writer quiescence gate passed",
        ],
    )


def test_had_1150_orders_stop_copy_and_start_steps_safely() -> None:
    text = _read_runbook()
    stop_order = _section(text, "5. Stop Order and Per-Step Verification Probes")
    guardrails = _section(text, "6. Copy-Window Guardrails")
    start_order = _section(text, "7. Start Order and Per-Step Verification Probes")

    _assert_in_order(
        stop_order,
        [
            "### 5.1 Announce the Window",
            "### 5.2 Stop Hermes Gateway Intake",
            "### 5.3 Stop Hermes Cron and User Timers",
            "### 5.4 Drain or Cancel Tracked Codex, Hermes, and ctx Sessions",
            "### 5.5 Stop Docker Containers With Relevant Bind Mounts",
            "### 5.6 Checkpoint SQLite and Verify WAL/SHM Quiet",
            "### 5.7 Run the Preflight Gate",
        ],
    )
    _assert_contains_all(
        guardrails,
        [
            "The copy window is open only after section 4 passes.",
            "If any writer restarts unexpectedly, stop the copy",
            "Do not run those copy commands from this runbook unless",
        ],
    )
    _assert_in_order(
        start_order,
        [
            "### 7.1 Verify Mounts and Path Contracts First",
            "### 7.2 Start Docker Containers With Relevant Bind Mounts",
            "### 7.3 Start User Timers",
            "### 7.4 Start Hermes Gateway Last",
        ],
    )


def test_had_1150_bind_mount_ancestor_checks_cover_both_directions() -> None:
    text = _read_runbook()

    assert text.count("source_resolved in root.parents") == 2
    assert text.count("root in source_resolved.parents") == 2
    _assert_contains_all(
        text,
        [
            "Catch bind sources under a copied root and ancestor bind sources",
            "such as /home/david or / exposing /home/david/.hermes.",
        ],
    )


def test_had_1150_evidence_capture_and_operator_message_boundary() -> None:
    text = _read_runbook()
    messages = _section(text, "8. Slack-Facing Operator Messages")
    evidence = _section(text, "9. Post-Window Verification and Evidence Capture Checklist")

    _assert_contains_all(
        messages,
        [
            "message bodies for the human operator to paste",
            "This runbook does not send Slack messages.",
            "Blocked before copy. Writer-quiescence preflight found active writers",
            "Maintenance window complete. Writers were stopped before copy",
        ],
    )
    assert "curl -X POST" not in messages
    assert "webhook" not in messages.lower()

    _assert_contains_all(
        evidence,
        [
            "completed-at.txt",
            "hermes-agent-git-status.txt",
            "hermes-gateway-status.txt",
            "systemd-user-timers.txt",
            "docker-ps.txt",
            "findmnt-path-contracts.txt",
            "state-db-quick-check.txt",
            "post-window-wal-shm.txt",
            "hermes-gateway-journal.txt",
            "Codex, Hermes, and ctx process inventories were quiet before copy.",
            "SQLite `quick_check` passed before and after copy.",
        ],
    )
