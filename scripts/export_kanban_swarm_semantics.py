#!/usr/bin/env python3
"""Export the concrete SQLite Kanban gate truth table for TLA+ binding."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any
from unittest.mock import patch

from hermes_cli import kanban_db as kb
from tools import kanban_tools as kt

SOURCE_PATHS = (
    "scripts/export_kanban_swarm_semantics.py",
    "hermes_cli/kanban_db.py",
    "hermes_cli/kanban_swarm.py",
    "hermes_cli/kanban_decompose.py",
    "tools/kanban_tools.py",
)

CASES = (
    ("done_pass", "done", {"gate": "pass"}, "metadata_gate_pass", True),
    ("done_fail", "done", {"gate": "fail"}, "metadata_gate_pass", False),
    ("done_missing", "done", {}, "metadata_gate_pass", False),
    ("done_malformed", "done", {"gate": {"bad": True}}, "metadata_gate_pass", False),
    ("archived_pass", "archived", {"gate": "pass"}, "metadata_gate_pass", False),
    ("running_pass", "running", {"gate": "pass"}, "metadata_gate_pass", False),
    ("unknown_gate_done_pass", "done", {"gate": "pass"}, "unknown", False),
)

BOARD_CASES = (
    ("worker_missing_pin_default_route", None, {}, False),
    ("worker_missing_pin_explicit_route", "alpha", {}, False),
    ("worker_db_pin_default_route", None, {"HERMES_KANBAN_DB": "/tmp/pin.db"}, True),
    (
        "worker_db_pin_explicit_route",
        "alpha",
        {"HERMES_KANBAN_DB": "/tmp/pin.db"},
        False,
    ),
    ("worker_board_pin_default_route", None, {"HERMES_KANBAN_BOARD": "alpha"}, True),
    ("worker_matching_board_route", "alpha", {"HERMES_KANBAN_BOARD": "alpha"}, True),
    ("worker_mismatched_board_route", "beta", {"HERMES_KANBAN_BOARD": "alpha"}, False),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _exercise_case(
    db_path: Path,
    name: str,
    verifier_status: str,
    metadata: dict[str, Any],
    gate_kind: str,
    expected: bool,
) -> dict[str, Any]:
    conn = kb.connect(db_path)
    try:
        verifier = kb.create_task(conn, title=f"verifier:{name}")
        synthesizer = kb.create_task(
            conn,
            title=f"synthesizer:{name}",
            parents=[verifier],
            parent_gates={verifier: "metadata_gate_pass"},
        )
        run_id = kb.claim_task(conn, verifier)
        assert run_id is not None
        assert kb.complete_task(conn, verifier, summary=name, metadata=metadata)
        if verifier_status != "done":
            conn.execute(
                "UPDATE tasks SET status = ? WHERE id = ?", (verifier_status, verifier)
            )
            conn.commit()
        if gate_kind != "metadata_gate_pass":
            conn.execute(
                "UPDATE task_links SET gate_kind = ? WHERE parent_id = ? AND child_id = ?",
                (gate_kind, verifier, synthesizer),
            )
            conn.commit()

        # ``complete_task`` recomputes children immediately. Reset to todo so
        # every case is evaluated from the same authoritative transition edge.
        conn.execute("UPDATE tasks SET status = 'todo' WHERE id = ?", (synthesizer,))
        conn.commit()
        kb.recompute_ready(conn)
        synth_task = kb.get_task(conn, synthesizer)
        assert synth_task is not None
        promoted = synth_task.status == "ready"
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (synthesizer,))
        conn.commit()
        claimable = kb.claim_task(conn, synthesizer) is not None
        review_synth = kb.create_task(
            conn,
            title=f"review-synthesizer:{name}",
            parents=[verifier],
            parent_gates={verifier: "metadata_gate_pass"},
        )
        if gate_kind != "metadata_gate_pass":
            conn.execute(
                "UPDATE task_links SET gate_kind = ? WHERE parent_id = ? AND child_id = ?",
                (gate_kind, verifier, review_synth),
            )
        conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (review_synth,))
        conn.commit()
        review_claimable = kb.claim_review_task(conn, review_synth) is not None
        # Recovery/retry writers share one dependency-aware status chooser.
        # Exercise a concrete running->ready/todo recovery edge as part of the
        # exported production semantics, including fail-closed cases.
        conn.execute("UPDATE tasks SET status = 'running' WHERE id = ?", (synthesizer,))
        conn.commit()
        assert kb.reclaim_task(conn, synthesizer)
        recovered = kb.get_task(conn, synthesizer)
        assert recovered is not None
        recovery_ready = recovered.status == "ready"
        observed = promoted and claimable and review_claimable and recovery_ready
        return {
            "name": name,
            "verifier_status": verifier_status,
            "verifier_metadata": metadata,
            "gate_kind": gate_kind,
            "expected": expected,
            "promoted": promoted,
            "claimable_when_forced_ready": claimable,
            "review_claimable": review_claimable,
            "recovery_ready": recovery_ready,
            "observed": observed,
            "pass": all(
                value is expected
                for value in (promoted, claimable, review_claimable, recovery_ready)
            ),
        }
    finally:
        conn.close()


def _exercise_board_case(
    name: str, board: str | None, pins: dict[str, str], expected: bool
) -> dict[str, Any]:
    env = {"HERMES_KANBAN_TASK": "t_formal_binding", **pins}
    with patch.dict(os.environ, env, clear=False):
        for key in ("HERMES_KANBAN_BOARD", "HERMES_KANBAN_DB"):
            if key not in env:
                os.environ.pop(key, None)
        error = kt._pinned_worker_board_or_error(board, "formal_binding")
    allowed = error is None
    pin_kind = (
        "board"
        if "HERMES_KANBAN_BOARD" in pins
        else "db"
        if "HERMES_KANBAN_DB" in pins
        else "none"
    )
    return {
        "name": name,
        "requested_board": board or "none",
        "pin_kind": pin_kind,
        "pinned_board": pins.get("HERMES_KANBAN_BOARD", "none"),
        "expected": expected,
        "allowed": allowed,
        "pass": allowed is expected,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.source_root.resolve()

    with tempfile.TemporaryDirectory(prefix="kanban-formal-binding-") as temp:
        temp_root = Path(temp)
        cases = [
            _exercise_case(
                temp_root / f"{name}.db", name, status, metadata, gate, expected
            )
            for name, status, metadata, gate, expected in CASES
        ]
        board_cases = [
            _exercise_board_case(name, board, pins, expected)
            for name, board, pins, expected in BOARD_CASES
        ]

    payload = {
        "schema": "hermes.kanban-swarm.production-semantics.v1",
        "source_root": str(root),
        "sources": {path: _sha256(root / path) for path in SOURCE_PATHS},
        "cases": cases,
        "board_cases": board_cases,
        "success": all(case["pass"] for case in [*cases, *board_cases]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps({
            "board_cases": len(board_cases),
            "cases": len(cases),
            "output": str(args.output),
            "success": payload["success"],
        })
    )
    return 0 if payload["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
