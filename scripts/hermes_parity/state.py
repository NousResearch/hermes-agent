"""Atomic state file handling for parity merge worktrees."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import gitops


STATE_FILE = ".parity-state.json"


@dataclass
class ParityState:
    tree_sha: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        payload = dict(self.data)
        payload["tree_sha"] = self.tree_sha
        return payload

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "ParityState":
        if "data" in payload and isinstance(payload.get("data"), dict):
            return cls(
                tree_sha=str(payload.get("tree_sha", "")),
                data=dict(payload.get("data") or {}),
            )
        data = dict(payload)
        tree_sha = str(data.pop("tree_sha", ""))
        return cls(
            tree_sha=tree_sha,
            data=data,
        )


def state_path(worktree: Path) -> Path:
    return worktree / STATE_FILE


def load_state(worktree: Path, *, invalidate_on_tree_change: bool = True) -> ParityState | None:
    path = state_path(worktree)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        state = ParityState.from_json(json.load(fh))
    if invalidate_on_tree_change:
        current = gitops.tree_sha(worktree)
        if state.tree_sha != current:
            return None
    return state


def save_state(worktree: Path, state: ParityState) -> None:
    path = state_path(worktree)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{STATE_FILE}.", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(state.to_json(), fh, sort_keys=True, indent=2)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def update_state(worktree: Path, **values: Any) -> ParityState:
    current = load_state(worktree, invalidate_on_tree_change=False)
    data = dict(current.data) if current else {}
    data.update(values)
    updated = ParityState(tree_sha=gitops.tree_sha(worktree), data=data)
    save_state(worktree, updated)
    return updated


def record_gate(worktree: Path, name: str, *, ok: bool, extra: dict[str, Any] | None = None) -> ParityState:
    current = load_state(worktree, invalidate_on_tree_change=False)
    data = dict(current.data) if current else {}
    gates = dict(data.get("gates") or {})
    record = {
        "ok": ok,
        "at": datetime.now(timezone.utc).isoformat(),
        "tree_sha": gitops.tree_sha(worktree),
    }
    if extra:
        record.update(extra)
    gates[name] = record
    data["gates"] = gates
    updated = ParityState(tree_sha=gitops.tree_sha(worktree), data=data)
    save_state(worktree, updated)
    return updated


def valid_gate_names_for_current_tree(worktree: Path, ordered_names: list[str]) -> set[str]:
    current = load_state(worktree, invalidate_on_tree_change=False)
    if not current:
        return set()
    tree = gitops.tree_sha(worktree)
    gates = current.data.get("gates") or {}
    valid: set[str] = set()
    for name in ordered_names:
        item = gates.get(name) or {}
        if item.get("ok") is True and item.get("tree_sha") == tree:
            valid.add(name)
        else:
            break
    return valid


def acked_paths(worktree: Path) -> set[str]:
    """Paths the operator has explicitly acknowledged as intentionally
    dropped/renamed fork files (RC2): first-class clearance for the
    fork-delta gate, distinct from ``finish --force``."""
    current = load_state(worktree, invalidate_on_tree_change=False)
    if not current:
        return set()
    acks = current.data.get("forkdelta_acks") or []
    return {str(item.get("path")) for item in acks if item.get("path")}


def record_ack(worktree: Path, path: str, reason: str) -> ParityState:
    """Record a reviewed-and-intentional fork-delta exception with an audit
    trail (path, reason, timestamp). Idempotent per path (reason updated)."""
    current = load_state(worktree, invalidate_on_tree_change=False)
    data = dict(current.data) if current else {}
    acks = [dict(item) for item in (data.get("forkdelta_acks") or [])]
    acks = [item for item in acks if item.get("path") != path]
    acks.append(
        {
            "path": path,
            "reason": reason,
            "at": datetime.now(timezone.utc).isoformat(),
        }
    )
    data["forkdelta_acks"] = acks
    updated = ParityState(tree_sha=gitops.tree_sha(worktree), data=data)
    save_state(worktree, updated)
    return updated
