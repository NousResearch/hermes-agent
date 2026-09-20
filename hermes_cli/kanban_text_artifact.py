"""Full-text handoffs declared at creation, checked under the completion lock.

The caller's completion metadata is evidence, never the source of the expected
identity or size bound. Legacy/local-only and PR contracts keep their semantics.
"""
from __future__ import annotations

import json
import re

PREFIX = "text-artifact-v1:"
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")
_RESERVED = {"task_id", "text", "complete", "truncated", "evidence_source"}


def parse_contract(value: str) -> dict:
    if not isinstance(value, str) or not value.startswith(PREFIX) or len(value) > 8192:
        raise ValueError("invalid text artifact completion contract")
    try:
        contract = json.loads(value[len(PREFIX):])
    except (ValueError, RecursionError) as exc:
        raise ValueError("invalid text artifact completion contract JSON") from exc
    if (not isinstance(contract, dict)
            or set(contract) != {"metadata_key", "identity", "assignee", "max_characters", "evidence_source"}
            or not isinstance(contract["metadata_key"], str)
            or not _NAME.fullmatch(contract["metadata_key"])
            or not isinstance(contract["assignee"], str) or not contract["assignee"].strip()
            or len(contract["assignee"]) > 128
            or not isinstance(contract["evidence_source"], str) or not contract["evidence_source"].strip()
            or len(contract["evidence_source"]) > 128
            or type(contract["max_characters"]) is not int
            or not 1 <= contract["max_characters"] <= 100000
            or not isinstance(contract["identity"], dict) or not contract["identity"]
            or len(contract["identity"]) > 32):
        raise ValueError("invalid text artifact completion contract fields")
    for key, expected in contract["identity"].items():
        if (not _NAME.fullmatch(key) or key in _RESERVED
                or type(expected) not in {str, int, bool}
                or isinstance(expected, str) and (not expected or len(expected) > 1024)):
            raise ValueError("invalid text artifact completion identity")
    return contract


def require_artifact(conn, task_id: str, metadata: dict | None, summary: str | None) -> None:
    """Run inside complete_task's write transaction, before any terminal mutation."""
    row = conn.execute(
        "SELECT completion_contract,assignee,current_run_id FROM tasks WHERE id=?", (task_id,)
    ).fetchone()
    if row is None or not (row["completion_contract"] or "").startswith(PREFIX):
        return
    contract = parse_contract(row["completion_contract"])
    key = contract["metadata_key"]

    def require(ok, reason):
        if not ok:
            raise ValueError(
                f"Completion requires metadata.{key}: {reason}. Task remains open; "
                "supply the complete matching artifact or block the task."
            )

    require(row["assignee"] == contract["assignee"], "task assignee differs from declared contract")
    if row["current_run_id"] is not None:
        run = conn.execute("SELECT task_id,profile FROM task_runs WHERE id=?", (row["current_run_id"],)).fetchone()
        require(run is not None and run["task_id"] == task_id and run["profile"] == contract["assignee"],
                "current run profile differs from declared contract")
    artifact = metadata.get(key) if isinstance(metadata, dict) else None
    require(isinstance(artifact, dict), "missing structured artifact (attachment paths alone do not count)")
    expected = {**contract["identity"], "task_id": task_id}
    require(all(type(artifact.get(k)) is type(v) and artifact.get(k) == v for k, v in expected.items()),
            "artifact identity does not match the creation contract")
    require(artifact.get("complete") is True and artifact.get("truncated") is False,
            "artifact must be complete and untruncated")
    require(artifact.get("evidence_source", contract["evidence_source"]) == contract["evidence_source"],
            "artifact evidence source differs from the contract")
    text = artifact.get("text")
    require(isinstance(text, str) and bool(text.strip()), "complete text is missing")
    require(len(text) <= contract["max_characters"], "complete text exceeds the declared character bound")
    require(text.strip() != (summary or "").strip(), "artifact only repeats the completion summary")
