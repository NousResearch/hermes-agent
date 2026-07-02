"""
A2A protocol helpers — Agent Card construction, JSON-RPC framing, and
disk-backed conversation persistence.

Wire shape follows the A2A spec (JSON-RPC 2.0 over HTTP):
  - Agent Card served at GET /.well-known/agent.json
  - Tasks via POST {jsonrpc:"2.0", method:"message/send", params:{...}}
  - Methods handled inbound: message/send, tasks/get

We deliberately implement the subset of A2A needed for text task exchange with
stdlib only (no a2a-sdk). If a2a-sdk is later added as an optional extra, the
client can upgrade transparently — the wire format is identical.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Optional

# A2A task lifecycle states (subset we use).
STATE_SUBMITTED = "submitted"
STATE_WORKING = "working"
STATE_INPUT_REQUIRED = "input-required"
STATE_COMPLETED = "completed"
STATE_FAILED = "failed"
STATE_CANCELED = "canceled"


# --------------------------------------------------------------------------
# Agent Card
# --------------------------------------------------------------------------

def build_agent_card(
    *,
    name: str,
    url: str,
    description: str,
    skills: Optional[list[dict]] = None,
    streaming: bool = False,
    auth_required: bool = False,
) -> dict:
    """Construct an A2A Agent Card document (the /.well-known/agent.json body)."""
    card: dict[str, Any] = {
        "name": name,
        "description": description,
        "url": url,
        "version": "0.1.0",
        "protocolVersion": "0.3",
        "capabilities": {
            "streaming": streaming,
            "pushNotifications": False,
            "stateTransitionHistory": False,
        },
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "skills": skills or [],
    }
    if auth_required:
        card["securitySchemes"] = {
            "bearer": {"type": "http", "scheme": "bearer"}
        }
        card["security"] = [{"bearer": []}]
    return card


def skills_from_toolsets(toolset_names: list[str]) -> list[dict]:
    """Derive A2A skill descriptors from the agent's enabled toolsets.

    A2A 'skills' are coarse capability advertisements, not tool schemas. We map
    each enabled toolset to one skill entry so peers can match tasks to us.
    """
    skills = []
    for ts in sorted(set(toolset_names or [])):
        skills.append({
            "id": f"toolset.{ts}",
            "name": ts,
            "description": f"Hermes '{ts}' capabilities",
            "tags": [ts],
        })
    if not skills:
        skills.append({
            "id": "general",
            "name": "general",
            "description": "General-purpose conversational agent",
            "tags": ["general"],
        })
    return skills


# --------------------------------------------------------------------------
# JSON-RPC framing
# --------------------------------------------------------------------------

def jsonrpc_result(req_id: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def jsonrpc_error(req_id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def new_task_id() -> str:
    return "task-" + uuid.uuid4().hex[:16]


def new_context_id() -> str:
    return "ctx-" + uuid.uuid4().hex[:16]


def text_message(role: str, text: str) -> dict:
    """Build an A2A Message with a single text Part."""
    return {
        "role": role,  # "user" | "agent"
        "parts": [{"kind": "text", "text": text}],
        "messageId": uuid.uuid4().hex,
    }


def extract_text(message_or_params: dict) -> str:
    """Pull concatenated text from an A2A Message / params payload.

    Tolerant of both ``{"message": {...}}`` params and a bare message dict, and
    of both ``kind`` and legacy ``type`` part discriminators.
    """
    msg = message_or_params.get("message", message_or_params)
    parts = msg.get("parts", []) if isinstance(msg, dict) else []
    chunks = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        if part.get("kind") in (None, "text") or part.get("type") == "text":
            txt = part.get("text")
            if isinstance(txt, str):
                chunks.append(txt)
    return "\n".join(chunks).strip()


def build_task(task_id: str, context_id: str, state: str, agent_text: str = "") -> dict:
    """Build an A2A Task object for a message/send result."""
    status: dict[str, Any] = {"state": state, "timestamp": _now_iso()}
    task: dict[str, Any] = {
        "id": task_id,
        "contextId": context_id,
        "status": status,
        "kind": "task",
    }
    if agent_text:
        # ty warning fix (cleaner than dict() cast): build status as
        # dict[str, Any] from the start. The original code typed
        # task["status"] as a literal dict which ty inferred too narrowly.
        status["message"] = text_message("agent", agent_text)
        task["artifacts"] = [{
            "artifactId": uuid.uuid4().hex,
            "parts": [{"kind": "text", "text": agent_text}],
        }]
    return task


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# --------------------------------------------------------------------------
# Conversation persistence (outside the context-compaction pipeline)
# --------------------------------------------------------------------------
#
# A2A exchanges are stored on disk per context-id so they survive context
# compaction and agent restarts (the #11025 requirement). One JSONL file per
# context; each line is one message {role, text, ts, task_id}.

def _conv_dir() -> Path:
    try:
        from hermes_constants import get_hermes_home
        base = Path(get_hermes_home())
    except Exception:
        base = Path(os.path.expanduser("~/.hermes"))
    return base / "a2a_conversations"


def _safe_name(context_id: str) -> str:
    return "".join(c for c in (context_id or "default") if c.isalnum() or c in "-_") or "default"


def persist_message(context_id: str, role: str, text: str, task_id: str = "") -> None:
    """Append one message to the context's on-disk conversation log."""
    try:
        d = _conv_dir()
        d.mkdir(parents=True, exist_ok=True)
        rec = {"ts": time.time(), "role": role, "text": text, "task_id": task_id}
        with (d / f"{_safe_name(context_id)}.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def load_conversation(context_id: str, limit: int = 50) -> list[dict]:
    """Load the last *limit* messages for a context (empty list if none)."""
    path = _conv_dir() / f"{_safe_name(context_id)}.jsonl"
    if not path.exists():
        return []
    out: list[dict] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception:
        return []
    return out[-limit:]


def list_conversations() -> list[str]:
    """Return known context-ids that have persisted conversations."""
    d = _conv_dir()
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.jsonl"))
