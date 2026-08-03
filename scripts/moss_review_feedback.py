#!/usr/bin/env python3
"""Read-only Mossy PR-review feedback normalisation and prevention-loop helpers.

This module deliberately has no write-capable GitHub or git path.  It turns
already-fetched JSON into a locally persisted, auditable triage queue.
"""
from __future__ import annotations

import json
import os
import shlex
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Sequence

SELF_AUTHORS = {"sahil-ss9", "sahil", "mossy"}
MAINTAINER_AUTHORS = {"teknium", "teknium1"}
PROMOTABLE = {"routine_patch", "sahil_escalation"}


def feedback_paths() -> dict[str, Path]:
    """Return distinct state surfaces below the sandboxable HERMES_HOME."""
    home = Path(os.environ.get("HERMES_HOME", "~/.hermes")).expanduser()
    data = home / "data"
    return {
        "state": data / "moss-review-feedback-state.json",
        "queue": data / "moss-review-feedback-queue.json",
        "ledger": data / "moss-review-feedback-ledger.json",
        "rules": data / "moss-review-feedback-rules.json",
    }


def load_json(path: Path, default: Any) -> Any:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def atomic_write_json(path: Path, value: Any) -> None:
    """Write one JSON document atomically without sharing a temp path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _author(item: dict[str, Any]) -> str:
    source = item.get("author") or item.get("user") or {}
    return str(source.get("login") or source.get("name") or item.get("author_login") or "")


def _excluded_author(author: str) -> bool:
    normal = author.strip().lower()
    return not normal or normal in SELF_AUTHORS or "bot" in normal or "[bot]" in normal


def normalise_feedback(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalise formal reviews, inline comments, and PR conversations.

    Each source is namespaced so GitHub IDs from different API surfaces cannot
    suppress each other during deduplication.
    """
    repo = str(payload["repo"])
    pr = payload["pr"]
    number = int(pr["number"])
    rows: list[dict[str, Any]] = []
    for surface, source_name, key_prefix in (
        ("review", "reviews", "review"),
        ("inline", "review_comments", "inline"),
        ("conversation", "comments", "conversation"),
    ):
        for item in payload.get(source_name, []):
            author = _author(item)
            if _excluded_author(author):
                continue
            item_id = str(item.get("id") or item.get("node_id") or "")
            if not item_id:
                continue
            rows.append({
                "key": f"{key_prefix}:{repo}#{number}:{item_id}",
                "surface": surface,
                "repo": repo,
                "pr_number": number,
                "source_item_id": item_id,
                "pr_title": str(pr.get("title", "")),
                "pr_url": str(pr.get("url", "")),
                "author": author,
                "priority": "maintainer" if author.strip().lower() in MAINTAINER_AUTHORS else "normal",
                "body": str(item.get("body") or ""),
                "path": item.get("path"),
                "line": item.get("line") or item.get("original_line"),
                "review_state": item.get("state"),
            })
    return rows


def dedupe_records(records: list[dict[str, Any]], state: dict[str, Any]) -> list[dict[str, Any]]:
    seen = set(state.get("seen", []))
    return [record for record in records if record["key"] not in seen]


def classify_feedback(record: dict[str, Any]) -> str:
    """Conservative four-way matrix applied to every human reviewer.

    Maintainer identity changes queue priority, never the safety decision.
    """
    body = str(record.get("body", "")).lower()
    if any(term in body for term in (
        "security", "data loss", "breaking change", "architecture", "production outage",
        "unsafe", "privacy", "credential", "regression risk",
    )):
        return "sahil_escalation"
    if any(term in body for term in ("clarify", "could you explain", "why is", "what happens if")):
        return "clarification"
    if not body.strip() or any(term in body for term in ("looks good", "thanks", "thank you", "lgtm", "approved")):
        return "reply_only"
    return "routine_patch"


def classify_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**record, "classification": classify_feedback(record)} for record in records]


def promote_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Only implementation work and explicit Sahil decisions enter the queue."""
    return [record for record in records if record["classification"] in PROMOTABLE]


def build_pre_push_report(
    classified: list[dict[str, Any]], promoted: list[dict[str, Any]], rules: dict[str, str]
) -> dict[str, Any]:
    counts = Counter(record["classification"] for record in classified)
    return {
        "counts": dict(counts),
        "promoted_keys": [record["key"] for record in promoted],
        "rule_selection": {kind: rules[kind] for kind in sorted(counts) if kind in rules},
        # A Sahil decision must be resolved before any eventual push workflow.
        "blocked": any(record["classification"] == "sahil_escalation" for record in classified),
    }


def _is_mutating_command(command: Sequence[str]) -> bool:
    lowered = [str(part).lower() for part in command]
    if not lowered:
        return True
    tool = lowered[0]
    if tool == "git":
        return any(token in {"push", "commit", "merge", "rebase", "reset", "tag", "branch", "checkout", "switch", "cherry-pick"} for token in lowered[1:])
    if tool == "gh":
        mutating = {"comment", "close", "merge", "edit", "create", "delete", "reopen", "review"}
        return any(token in mutating for token in lowered[1:]) or any(token in {"post", "put", "patch", "delete"} for token in lowered)
    return True


def run_read_only(command: Sequence[str], *, runner: Callable[..., Any] | None = None, dry_run: bool = True) -> dict[str, Any]:
    """Deny mutation-shaped gh/git commands and never execute in dry-run mode."""
    command = [str(part) for part in command]
    if _is_mutating_command(command):
        return {"status": "denied", "command": command}
    if dry_run:
        return {"status": "dry_run", "command": command}
    if runner is None:
        raise RuntimeError("A runner must be injected; review feedback stays read-only")
    return {"status": "executed", "command": command, "result": runner(command)}


ACTION_REQUIREMENTS = {
    "routine_patch": ["root_cause", "changed_paths", "test_evidence", "prevention"],
    "reply_only": ["reply"],
    "clarification": ["question"],
    "sahil_escalation": ["sahil_decision"],
}
ACTIONABLE = {"routine_patch", "reply_only", "clarification"}
EXECUTOR_REJECTED_RISKS = {"security", "data", "dependency", "ci", "scope", "architecture"}


def build_action_plan(classified: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Produce stable, reviewable next steps without taking any action."""
    plan = []
    for record in classified:
        action = str(record["classification"])
        if action not in ACTION_REQUIREMENTS:
            raise ValueError(f"unknown feedback action: {action}")
        author = str(record.get("author", ""))
        priority = "maintainer" if author.lower() in MAINTAINER_AUTHORS else "normal"
        plan.append({
            "key": record.get("key", ""),
            "thread": record.get("thread") or record.get("key", ""),
            "action": action,
            "repo": record.get("repo", ""),
            "pr_number": record.get("pr_number"),
            "source_item_id": record.get("source_item_id", ""),
            "surface": record.get("surface", ""),
            "author": author,
            "priority": priority,
            "requires": list(ACTION_REQUIREMENTS[action]),
        })
    return plan


def record_resolution(record: dict[str, Any], resolution: dict[str, Any]) -> dict[str, Any]:
    """Create the complete audit record required when actionable work resolves."""
    if record.get("classification") not in ACTIONABLE:
        raise ValueError("only actionable feedback can be recorded as resolved")
    required = ("root_cause", "changed_paths", "test_evidence", "prevention")
    for field in required:
        value = resolution.get(field)
        if not value:
            raise ValueError(f"resolution requires {field}")
    if not isinstance(resolution["changed_paths"], list) or not isinstance(resolution["test_evidence"], list):
        raise ValueError("changed_paths and test_evidence must be lists")
    return {
        "source": str(record.get("surface", "")),
        "thread": str(record.get("thread") or record.get("key", "")),
        "reviewer": str(record.get("author", "")),
        "root_cause": str(resolution["root_cause"]),
        "changed_paths": [str(path) for path in resolution["changed_paths"]],
        "test_evidence": [str(test) for test in resolution["test_evidence"]],
        "prevention": str(resolution["prevention"]),
    }


def _feedback_theme(body: str) -> str:
    """Small deterministic theme key; it intentionally avoids LLM inference."""
    words = [word.strip(".,:;!?()[]") for word in body.lower().split()]
    stop = {"please", "this", "that", "the", "a", "an", "is", "use", "here", "for", "and"}
    return " ".join(word for word in words if word and word not in stop)


def learn_prevention_rule(record: dict[str, Any], prior_events: list[dict[str, Any]]) -> dict[str, Any]:
    """Turn only objective, scoped, or repeated feedback into a prevention rule."""
    body = str(record.get("body", ""))
    lowered = body.lower()
    theme = _feedback_theme(body)
    base = {"theme": theme, "evidence": str(record.get("key", ""))}
    if any(term in lowered for term in ("security", "vulnerability", "credential", "privacy")):
        return {**base, "kind": "regression_test", "rule": "security regression test"}
    if any(term in lowered for term in ("bug", "regress", "invalid", "crash", "error")):
        return {**base, "kind": "regression_test", "rule": "regression test"}
    if any(term in lowered for term in ("ci", "lint", "build fail", "test fail")):
        return {**base, "kind": "pre_check", "rule": "named pre-push check"}
    if "convention" in lowered or "project" in lowered:
        return {**base, "kind": "project_convention", "scope": str(record.get("repo", "")), "rule": "project-scoped convention"}
    prior_themes = {str(event.get("theme", "")) for event in prior_events}
    if theme and theme in prior_themes:
        return {**base, "kind": "repeated_theme", "rule": "repeated-review pre-check"}
    return {**base, "kind": "ledger_only", "rule": "one-off preference"}


def persist_resolution(record: dict[str, Any], resolution: dict[str, Any]) -> dict[str, Any]:
    """Atomically append a resolved actionable event and its allowed prevention rule."""
    event = record_resolution(record, resolution)
    paths = feedback_paths()
    ledger = load_json(paths["ledger"], {"events": []})
    rules = load_json(paths["rules"], {"rules": []})
    events = list(ledger.get("events", []))
    events.append(event)
    promoted_rules = list(rules.get("rules", []))
    rule = learn_prevention_rule(record, promoted_rules)
    # Ledger-only entries are retained as non-executable history so a genuine
    # repeated theme can later be promoted deterministically.
    promoted_rules.append(rule)
    persisted_ledger = {"events": events}
    persisted_rules = {"rules": promoted_rules}
    atomic_write_json(paths["ledger"], persisted_ledger)
    atomic_write_json(paths["rules"], persisted_rules)
    return {"record": event, "rule": rule, "ledger": persisted_ledger, "rules": persisted_rules}


def _forbidden_command(command: Sequence[str]) -> bool:
    lowered = [str(part).lower() for part in command]
    if not lowered:
        return True
    joined = " ".join(lowered)
    return (
        any(token in lowered for token in ("--force", "-f", "delete", "merge"))
        or "force-with-lease" in joined
        or (lowered[0] == "git" and any(token in lowered for token in ("reset", "rebase")))
    )


def _named_test_commands(evidence: list[Any]) -> list[list[str]]:
    commands: list[list[str]] = []
    for item in evidence:
        command = shlex.split(str(item))
        if not command or "pytest" not in command:
            raise ValueError("test_evidence must name a pytest command")
        commands.append(command)
    if not commands:
        raise ValueError("routine patch requires named test_evidence")
    return commands


def _reply_command(action: dict[str, Any], content: str) -> list[str]:
    """Build the GitHub endpoint matching the source feedback surface."""
    repo = str(action["repo"])
    pr_number = int(action["pr_number"])
    source_item_id = str(action["source_item_id"])
    surface = str(action["surface"])
    if surface == "inline":
        return [
            "gh", "api", f"/repos/{repo}/pulls/{pr_number}/comments/{source_item_id}/replies",
            "-f", f"body={content}",
        ]
    # Formal reviews do not provide a reply endpoint. Conversation comments
    # likewise fall back to the PR's issue thread, where the reviewer mention
    # preserves the original conversational target.
    return [
        "gh", "api", f"/repos/{repo}/issues/{pr_number}/comments",
        "-f", f"body=@{action['author']} {content}",
    ]


def _valid_reply_target(action: dict[str, Any]) -> bool:
    return (
        bool(str(action.get("repo", "")))
        and isinstance(action.get("pr_number"), int)
        and action["pr_number"] > 0
        and bool(str(action.get("source_item_id", "")))
        and action.get("surface") in {"inline", "conversation", "review"}
        and bool(str(action.get("author", "")))
    )


def execute_action(action: dict[str, Any], *, runner: Callable[..., Any] | None = None, allow_mutation: bool = False) -> dict[str, Any]:
    """Guarded, injected execution path; it has no default subprocess runner.

    The caller must explicitly provide both a runner and ``allow_mutation``.
    Unsafe classes and destructive commands are rejected before the injected
    runner observes anything. This makes fixture tests incapable of live side
    effects while retaining an auditable eventual execution seam.
    """
    if not allow_mutation or runner is None:
        return {"status": "denied", "reason": "explicit runner and allow_mutation are required"}
    action_name = action.get("action")
    if action_name == "sahil_escalation":
        return {"status": "rejected", "reason": "Sahil escalation is never executor-eligible"}
    if action_name not in ACTIONABLE:
        return {"status": "rejected", "reason": "unknown executor action"}
    if not _valid_reply_target(action):
        return {"status": "rejected", "reason": "stable reply target is required"}
    content_field = "question" if action_name == "clarification" else "reply"
    content = action.get(content_field)
    if not isinstance(content, str) or not content:
        return {"status": "rejected", "reason": f"{content_field} is required"}
    if action_name in {"reply_only", "clarification"}:
        return {"status": "executed", "results": [runner(_reply_command(action, content))]}
    risk_text = " ".join(
        str(action.get(field, "")) for field in ("risk", "body", "scope")
    ).lower()
    if any(risk in risk_text for risk in EXECUTOR_REJECTED_RISKS):
        return {"status": "rejected", "reason": "unsafe feedback requires Sahil review"}
    try:
        tests = _named_test_commands(list(action.get("test_evidence", [])))
    except ValueError as exc:
        return {"status": "rejected", "reason": str(exc)}
    commands = [[str(part) for part in command] for command in action.get("commands", [])]
    if not commands or any(_forbidden_command(command) for command in commands):
        return {"status": "rejected", "reason": "missing or forbidden mutation command"}
    if any(len(command) < 2 or command[0] != "git" or command[1] != "push" for command in commands):
        return {"status": "rejected", "reason": "only ordinary git push is allowed"}
    results = []
    for command in tests + commands + [_reply_command(action, content)]:
        results.append(runner(command))
    return {"status": "executed", "results": results}
