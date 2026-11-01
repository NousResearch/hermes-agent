"""Canonical task/run binding for local-model admission.

The worker receives this digest as a lookup hint only. A trusted admission controller must
read the pinned Kanban database itself, recompute the digest, and compare the current run and
claim lock before granting a GPU lease.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


CAPABILITY_MARKER = "Local-GPU-Capability:"
CONTRACT_SCHEMA = "hermes.local-model-task-contract.v1"


def capability_id_from_body(body: Any) -> str:
    """Return exactly one declared capability id; malformed or duplicate markers fail closed."""
    if not isinstance(body, str):
        return ""
    values = [
        line.strip()[len(CAPABILITY_MARKER):].strip()
        for line in body.splitlines()
        if line.strip().startswith(CAPABILITY_MARKER)
    ]
    if len(values) != 1 or not values[0] or any(ch.isspace() for ch in values[0]):
        return ""
    return values[0]


def task_run_contract_sha256(task: Any) -> str:
    """Hash all task fields that materially define a local worker's assignment."""
    if isinstance(task, Mapping):
        get = task.get
    else:
        get = lambda key, default=None: getattr(task, key, default)

    skills = get("skills")
    if skills is not None and not isinstance(skills, (list, tuple)):
        raise ValueError("task skills must be a list or null")
    payload = {
        "schema": CONTRACT_SCHEMA,
        "task_id": str(get("id") or ""),
        "title": str(get("title") or ""),
        "body": str(get("body") or ""),
        "assignee": str(get("assignee") or ""),
        "workspace_kind": str(get("workspace_kind") or ""),
        "workspace_path": str(get("workspace_path") or ""),
        "branch_name": str(get("branch_name") or ""),
        "model_override": str(get("model_override") or ""),
        "provider_override": str(get("provider_override") or ""),
        "reasoning_effort": str(get("reasoning_effort") or ""),
        "completion_contract": str(get("completion_contract") or ""),
        "skills": None if skills is None else [str(value) for value in skills],
        "max_runtime_seconds": get("max_runtime_seconds"),
        "max_retries": get("max_retries"),
        "goal_mode": bool(get("goal_mode", False)),
        "goal_max_turns": get("goal_max_turns"),
        "run_id": str(get("current_run_id") or ""),
        "claim_lock": str(get("claim_lock") or ""),
    }
    if not payload["task_id"] or not payload["run_id"] or not payload["claim_lock"]:
        raise ValueError("task, run, and claim-lock identities are required")
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
