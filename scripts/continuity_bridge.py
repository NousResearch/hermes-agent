#!/usr/bin/env python3
"""Bounded preflight for Basic Memory + Beads + Spec Kit continuity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from continuity_common import (
    ContinuityError,
    compact_json,
    external_input_digests,
    external_volume_available,
    git_state,
    load_json,
    minimal_child_env,
    read_markdown_frontmatter,
    receipt_errors,
    receipt_signature_errors,
    run_command,
    verify_pinned_executable,
)


ALLOWED_ACTIVE_STATES = {"open", "ready", "in_progress", "blocked"}
REQUIRED_CARD_FIELDS = {
    "project",
    "folder",
    "goal",
    "user_context",
    "state",
    "active_task",
    "spec_change",
    "evidence",
    "next_action",
    "blockers",
    "supersedes",
}
REQUIRED_SPEC_FIELDS = {
    "project",
    "repo_id",
    "goal",
    "state",
    "active_task",
    "change_id",
    "folder",
}


def _task_from_json(value: Any, task_id: str) -> dict[str, Any] | None:
    candidates = value if isinstance(value, list) else [value]
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate.get("id") == task_id:
            return candidate
    return None


def _read_beads(
    config: dict[str, Any], repo_root: Path
) -> tuple[dict[str, Any], bool, str]:
    beads = config["beads"]
    task_id = beads["active_task"]
    verify_pinned_executable(config, repo_root, "beads", Path(beads["binary"]))
    env = minimal_child_env({
        "BEADS_DIR": beads["data_dir"],
        "HOME": config["state_root"],
    })
    degraded_reason = ""
    try:
        result = run_command(
            [beads["binary"], "show", task_id, "--json"],
            cwd=repo_root,
            timeout=float(beads.get("timeout_seconds", 5)),
            env=env,
        )
        if result.returncode != 0:
            raise ContinuityError(f"bd show exited with status {result.returncode}")
        task = _task_from_json(json.loads(result.stdout), task_id)
        if task is not None:
            return task, False, ""
        degraded_reason = f"bd show did not return active task {task_id}"
    except (ContinuityError, json.JSONDecodeError) as exc:
        degraded_reason = str(exc)

    issues_path = Path(beads["issues_path"])
    try:
        lines = issues_path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            task = _task_from_json(json.loads(line), task_id)
            if task is not None:
                return task, True, degraded_reason or "Beads CLI unavailable"
    except (OSError, json.JSONDecodeError) as exc:
        raise ContinuityError(
            f"cannot read Beads task or JSONL fallback: {exc}"
        ) from exc
    raise ContinuityError(f"active Beads task {task_id} is absent")


def validate_tool_adjacency(events: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    index = 0
    while index < len(events):
        event = events[index]
        event_type = event.get("type")
        if event_type in {"server_tool_use", "tool_use"}:
            uses: list[str] = []
            while index < len(events) and events[index].get("type") in {
                "server_tool_use",
                "tool_use",
            }:
                uses.append(str(events[index].get("id") or index))
                index += 1
            results: list[str] = []
            while index < len(events) and events[index].get("type") in {
                "tool_result",
                "server_tool_result",
            }:
                results.append(
                    str(
                        events[index].get("tool_use_id")
                        or events[index].get("id")
                        or index
                    )
                )
                index += 1
            if not results:
                errors.append(
                    f"tool use batch {', '.join(uses)} is interrupted before its results"
                )
            elif results != uses:
                errors.append(
                    "tool result batch does not match tool use batch: expected "
                    + ", ".join(uses)
                    + "; got "
                    + ", ".join(results)
                )
            continue
        if event_type in {"tool_result", "server_tool_result"}:
            result_id = str(event.get("tool_use_id") or event.get("id") or "")
            errors.append(f"tool result {result_id or index} has no pending use")
        index += 1
    return errors


def tool_events_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Normalize explicit events or Hermes-native conversation history."""
    raw_events = payload.get("events")
    if isinstance(raw_events, list) and all(
        isinstance(item, dict) for item in raw_events
    ):
        return raw_events
    extra = payload.get("extra")
    history = extra.get("conversation_history") if isinstance(extra, dict) else None
    if not isinstance(history, list) or not all(
        isinstance(item, dict) for item in history
    ):
        return None
    normalized: list[dict[str, Any]] = []
    for message in history:
        if message.get("role") == "tool" and message.get("tool_call_id") is not None:
            normalized.append({
                "type": "tool_result",
                "tool_use_id": message.get("tool_call_id"),
            })
            continue
        emitted = False
        calls = message.get("tool_calls")
        if isinstance(calls, list):
            for call in calls:
                if isinstance(call, dict):
                    normalized.append({"type": "tool_use", "id": call.get("id")})
                    emitted = True
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    normalized.append({"type": "content"})
                    emitted = True
                    continue
                block_type = block.get("type")
                if block_type in {"tool_use", "server_tool_use"}:
                    normalized.append({"type": block_type, "id": block.get("id")})
                    emitted = True
                elif block_type in {"tool_result", "server_tool_result"}:
                    normalized.append({
                        "type": block_type,
                        "tool_use_id": block.get("tool_use_id") or block.get("id"),
                    })
                    emitted = True
                else:
                    normalized.append({"type": "content"})
                    emitted = True
        elif content not in (None, ""):
            normalized.append({"type": "content"})
            emitted = True
        if not emitted:
            normalized.append({"type": str(message.get("role") or "message")})
    return normalized


def _required(mapping: dict[str, Any], fields: set[str], authority: str) -> list[str]:
    return [
        f"{authority} missing {field}"
        for field in sorted(fields)
        if mapping.get(field) in (None, "")
    ]


def build_preflight(
    config_path: Path,
    *,
    cwd: Path,
    tool_envelope: Path | None = None,
    tool_events: list[dict[str, Any]] | None = None,
    require_mounted_volume: bool = True,
    allow_recovery_journal: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    config = load_json(config_path)
    repo_root = Path(config["expected_repo_root"]).resolve()
    external_volume = Path(config["external_volume"])

    if require_mounted_volume and not external_volume_available(external_volume):
        errors.append(f"external volume is not mounted: {external_volume}")

    try:
        state = git_state(cwd.resolve(), integration_ref=config.get("integration_ref"))
    except ContinuityError as exc:
        errors.append(str(exc))
        state = None
    if state and Path(state.root) != repo_root:
        errors.append(
            f"wrong repository folder: expected {repo_root}, got {state.root}"
        )

    try:
        card, card_body = read_markdown_frontmatter(
            Path(config["basic_memory"]["card_path"])
        )
        errors.extend(_required(card, REQUIRED_CARD_FIELDS, "Basic Memory card"))
        card_word_count = len((json.dumps(card, default=str) + " " + card_body).split())
        if card_word_count > 1200:
            errors.append(f"Basic Memory card exceeds 1,200 words: {card_word_count}")
    except ContinuityError as exc:
        errors.append(str(exc))
        card = {}
        card_word_count = None
    if state and card:
        evidence = card.get("evidence")
        evidence_commit = evidence.get("commit") if isinstance(evidence, dict) else None
        if evidence_commit != state.commit:
            errors.append(
                "Basic Memory evidence commit is stale: "
                f"expected {state.commit}, got {evidence_commit!r}"
            )

    try:
        spec_path = repo_root / config["spec"]["path"]
        spec, _ = read_markdown_frontmatter(spec_path)
        errors.extend(_required(spec, REQUIRED_SPEC_FIELDS, "Spec Kit change"))
    except ContinuityError as exc:
        errors.append(str(exc))
        spec = {}

    degraded = False
    try:
        task, degraded, reason = _read_beads(config, repo_root)
        if degraded:
            warnings.append(f"Beads CLI degraded to JSONL: {reason}")
    except ContinuityError as exc:
        errors.append(str(exc))
        task = {}

    expected = {
        "project": config.get("project"),
        "goal": config.get("goal"),
        "folder": str(repo_root),
        "active_task": config.get("beads", {}).get("active_task"),
        "change_id": config.get("spec", {}).get("change_id"),
    }
    comparisons = (
        ("card.project", card.get("project"), expected["project"]),
        ("spec.project", spec.get("project"), expected["project"]),
        ("card.goal", card.get("goal"), expected["goal"]),
        ("spec.goal", spec.get("goal"), expected["goal"]),
        ("task.title", task.get("title"), expected["goal"]),
        ("card.folder", card.get("folder"), expected["folder"]),
        ("spec.folder", spec.get("folder"), expected["folder"]),
        ("card.active_task", card.get("active_task"), expected["active_task"]),
        ("spec.active_task", spec.get("active_task"), expected["active_task"]),
        ("task.id", task.get("id"), expected["active_task"]),
        ("card.spec_change", card.get("spec_change"), expected["change_id"]),
        ("spec.change_id", spec.get("change_id"), expected["change_id"]),
        ("task.spec_id", task.get("spec_id"), config.get("spec", {}).get("path")),
    )
    for label, actual, wanted in comparisons:
        if actual != wanted:
            errors.append(
                f"authority conflict for {label}: expected {wanted!r}, got {actual!r}"
            )
    task_status = task.get("status") if task else None
    terminal_task_is_valid = card.get("state") == "ENFORCED" and task_status == "closed"
    if task and task_status not in ALLOWED_ACTIVE_STATES and not terminal_task_is_valid:
        errors.append(f"Beads task is stale or terminal: {task.get('status')!r}")
    if task_status == "blocked":
        errors.append("Beads task lifecycle is blocked")
    if (
        card.get("state") == "ENFORCED"
        and task_status != "closed"
        and not allow_recovery_journal
    ):
        errors.append("ENFORCED lifecycle requires a closed Beads task")
    terminal_states = {"TESTED", "ENFORCED"}
    card_state = card.get("state")
    spec_state = spec.get("state")
    if card_state in terminal_states or spec_state in terminal_states:
        if spec_state in terminal_states and card_state != spec_state:
            errors.append(
                f"terminal lifecycle conflict: card={card_state!r}, spec={spec_state!r}"
            )
        receipt_path = (
            (card.get("evidence") or {}).get("receipt")
            if isinstance(card.get("evidence"), dict)
            else None
        )
        if not receipt_path:
            errors.append("terminal lifecycle has no gate receipt")
        elif state:
            try:
                receipt = load_json(Path(receipt_path))
                errors.extend(receipt_signature_errors(config, receipt))
                errors.extend(receipt_errors(receipt, state))
                try:
                    current_inputs = external_input_digests(config, repo_root)
                    if receipt.get("external_inputs") != current_inputs:
                        errors.append(
                            "terminal receipt external instruction or executable identity is stale"
                        )
                except ContinuityError as exc:
                    errors.append(str(exc))
                if receipt.get("lifecycle_target") != card_state:
                    errors.append(
                        "receipt lifecycle target does not match authority state"
                    )
            except ContinuityError as exc:
                errors.append(str(exc))
    else:
        allowed = {"PROPOSED", "ACTIVE", "IMPLEMENTED", "BLOCKED"}
        if card and card_state not in allowed:
            errors.append(f"invalid card state: {card_state!r}")
        if spec and spec_state not in allowed:
            errors.append(f"invalid spec state: {spec_state!r}")
        if card_state == "BLOCKED":
            errors.append("Basic Memory card lifecycle is BLOCKED")
        if spec_state == "BLOCKED":
            errors.append("Spec Kit change lifecycle is BLOCKED")

    journal_path = Path(config["state_root"]) / "promotion.json"
    if journal_path.is_file():
        try:
            journal = load_json(journal_path)
            blocked_journals = {"PREPARED", "CARD_WRITTEN", "RECOVERY_REQUIRED"}
            if journal.get("status") in blocked_journals and not (
                allow_recovery_journal and journal.get("status") in blocked_journals
            ):
                errors.append(
                    f"promotion journal requires recovery: {journal.get('status')}"
                )
        except ContinuityError as exc:
            errors.append(str(exc))

    for relative in config.get("instructions", []):
        if not (repo_root / relative).is_file():
            errors.append(f"native instruction is missing: {relative}")
    try:
        external_digests = external_input_digests(config, repo_root)
    except ContinuityError as exc:
        errors.append(str(exc))
        external_digests = {}

    if tool_envelope is not None:
        try:
            raw = json.loads(tool_envelope.read_text(encoding="utf-8"))
            events = raw.get("events") if isinstance(raw, dict) else raw
            if not isinstance(events, list) or not all(
                isinstance(item, dict) for item in events
            ):
                raise ValueError("expected a list of event objects")
            errors.extend(validate_tool_adjacency(events))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"invalid tool envelope: {exc}")
    if tool_events is not None:
        errors.extend(validate_tool_adjacency(tool_events))

    if state is not None:
        if state.integration_ref and state.merge_base != state.integration_sha:
            errors.append(
                "current branch is not based on the configured integration SHA"
            )
        try:
            final_state = git_state(
                cwd.resolve(), integration_ref=config.get("integration_ref")
            )
            if final_state.as_dict() != state.as_dict():
                errors.append("repository state changed while preflight was running")
        except ContinuityError as exc:
            errors.append(f"cannot revalidate repository state: {exc}")

    status = "BLOCKED" if errors else ("DEGRADED" if degraded else "AVAILABLE")
    completion_allowed = not errors and not degraded
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "completion_allowed": completion_allowed,
        "project": config.get("project"),
        "goal": config.get("goal"),
        "active_task": task.get("id"),
        "task_status": task.get("status"),
        "spec_change": spec.get("change_id"),
        "lifecycle": card.get("state"),
        "card_words": card_word_count,
        "next_action": card.get("next_action"),
        "blockers": card.get("blockers", []),
        "git": state.as_dict() if state else None,
        "external_inputs": external_digests,
        "errors": errors,
        "warnings": warnings,
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(".continuity/config.json"))
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument("--tool-envelope", type=Path)
    parser.add_argument(
        "--allow-unmounted-fixture", action="store_true", help=argparse.SUPPRESS
    )
    args = parser.parse_args(argv)
    try:
        config = load_json(args.config)
        result = build_preflight(
            args.config,
            cwd=args.cwd,
            tool_envelope=args.tool_envelope,
            require_mounted_volume=not args.allow_unmounted_fixture,
        )
        print(compact_json(result, int(config.get("max_output_chars", 8000))))
        return 0 if result["status"] != "BLOCKED" else 2
    except (ContinuityError, KeyError, TypeError, ValueError) as exc:
        result = {
            "schema_version": 1,
            "status": "BLOCKED",
            "completion_allowed": False,
            "errors": [str(exc)],
        }
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 2


if __name__ == "__main__":
    sys.exit(main())
