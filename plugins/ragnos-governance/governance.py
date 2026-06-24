"""RAGnos governance policy: a deterministic pre-tool-call gate + telemetry.

Pure, dependency-free policy so it is hermetically testable. The plugin
(``__init__.py``) wires these functions into the gateway's ``pre_tool_call`` /
``post_tool_call`` / ``on_session_end`` hooks.

OBSERVE by default: every tool call is recorded to a JSONL governance ledger.
Set ``RAGNOS_GOVERNANCE_ENFORCE=1`` (plus ``RAGNOS_GOVERNANCE_FORBIDDEN_TOOLS``)
to BLOCK gated tools so they must route through the Hermes Hub (preview +
approval). This file edits no upstream code; it lives only under
``plugins/ragnos-governance/`` (a RAGnos-owned surface).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional

ENFORCE_ENV = "RAGNOS_GOVERNANCE_ENFORCE"
FORBIDDEN_ENV = "RAGNOS_GOVERNANCE_FORBIDDEN_TOOLS"
LEDGER_ENV = "RAGNOS_GOVERNANCE_LEDGER"

_TRUTHY = {"1", "true", "yes", "on"}


def is_enforcing(env: Optional[Mapping[str, str]] = None) -> bool:
    env = os.environ if env is None else env
    return str(env.get(ENFORCE_ENV, "")).strip().lower() in _TRUTHY


def forbidden_tools(env: Optional[Mapping[str, str]] = None) -> frozenset[str]:
    env = os.environ if env is None else env
    raw = str(env.get(FORBIDDEN_ENV, "")).strip()
    if not raw:
        return frozenset()
    return frozenset(part.strip() for part in raw.replace("\n", ",").split(",") if part.strip())


def evaluate_tool(
    tool_name: str,
    args: Optional[Mapping[str, Any]] = None,
    *,
    enforce: bool = False,
    forbidden: frozenset[str] = frozenset(),
) -> Optional[dict[str, str]]:
    """Return a block directive when a gated tool runs under enforcement, else None.

    The return shape matches the gateway's ``get_pre_tool_call_block_message``
    contract: ``{"action": "block", "message": "..."}``.
    """
    if enforce and tool_name in forbidden:
        return {
            "action": "block",
            "message": (
                f"RAGnos governance: tool '{tool_name}' is gated. Route it through "
                "the Hermes Hub (preview + approval) instead of calling it directly."
            ),
        }
    return None


def record_event(event_type: str, payload: Mapping[str, Any], *, env: Optional[Mapping[str, str]] = None) -> None:
    """Append a governance telemetry event to the JSONL ledger (best effort)."""
    env = os.environ if env is None else env
    path = ledger_path(env)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {"event_type": event_type, **dict(payload)}
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, default=str) + "\n")
    except Exception:  # noqa: BLE001 - telemetry must never break a tool call.
        pass


def ledger_path(env: Optional[Mapping[str, str]] = None) -> Path:
    env = os.environ if env is None else env
    raw = env.get(LEDGER_ENV)
    if raw:
        return Path(str(raw)).expanduser()
    home = env.get("HERMES_HOME") or str(Path.home() / ".hermes")
    return Path(home).expanduser() / "ragnos-governance" / "governance-ledger.jsonl"
