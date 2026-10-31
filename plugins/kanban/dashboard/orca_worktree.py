"""Orca worktree status + reveal for the kanban dashboard badge (ORCA-06).

The dashboard badge needs two facts about a task's linked Orca worktree:
its current state (active / idle / complete) and a way to focus it in the
Orca UI. Both come from Orca's own runtime, reached over the same local RPC
the ``orca`` CLI uses (``src/cli/runtime/transport.ts`` in stablyai/orca):

* Orca writes ``orca-runtime.json`` into its userData dir
  (``~/Library/Application Support/orca`` on macOS) containing the runtime
  pid, a unix-socket (or named-pipe) endpoint, and an auth token.
* Clients speak newline-delimited JSON envelopes over that socket:
  request ``{"id", "authToken", "method", "params"}``, response
  ``{"id", "ok": true, "result", "_meta": {"runtimeId"}}`` (or
  ``ok: false`` with ``error: {code, message}``). Keepalive frames
  (``{"_keepalive": true}``) must be skipped.

Verified against Orca source (stablyai/orca@main, 2026-10-26):

* ``worktree.ps`` returns ``RuntimeWorktreePsResult`` whose rows carry
  ``worktreeId`` and ``status`` — Orca's ``RuntimeWorktreeStatus`` is
  ``active | working | permission | done | inactive`` (see
  ``src/shared/runtime-worktree-contracts.ts``). The badge collapses that
  5-state vocabulary into the three states the card spec mandates:
  active/working/permission -> **active**, inactive -> **idle**,
  done -> **complete**.
* ``worktree.activate`` with ``{"worktree": "id:<worktreeId>",
  "navigation": "caller", "notifyClients": false}`` is exactly what Orca's
  mobile companion "open" action issues to focus a worktree in the desktop
  UI (mobile RPC golden ``host-worktree-actions-pin-open-delete.json``).
  That is the badge's click action — there is no ``orca://worktree/...``
  URL route (Orca's ``open-url`` handler only routes skill-share links,
  ``src/main/index.ts``), so the RPC is the honest deep link.

Resolution layers, in order (each bounded; failures degrade, never raise):

1. Live RPC against the running Orca runtime (truth when Orca is up).
2. The orca-status skill's ``state.json`` (the /orca-status command's
   journal of worktree session statuses) when the runtime is unreachable.
3. ``unknown`` — Orca not installed/running and no journal entry; the
   badge still renders (the link exists) with a muted style and an
   explanatory tooltip.

The module is import-safe without fastapi/hermes runtime state so the
dashboard tests can exercise it standalone.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Optional

# --- Badge state vocabulary (card spec: exactly these three + honest unknown)

BADGE_ACTIVE = "active"
BADGE_IDLE = "idle"
BADGE_COMPLETE = "complete"
BADGE_UNKNOWN = "unknown"

# Orca RuntimeWorktreeStatus (src/shared/runtime-worktree-contracts.ts).
_RUNTIME_STATUS_ACTIVE = {"active", "working", "permission"}
_RUNTIME_STATUS_DONE = {"done"}
_RUNTIME_STATUS_INACTIVE = {"inactive"}

# Orca AgentStatusState values that mean "a session is live on this worktree"
# (orca CLI's own CLI_STATE_MAP treats working/blocked/waiting as running).
_AGENT_LIVE_STATES = {"working", "blocked", "waiting"}
_AGENT_DONE_STATES = {"done"}

# --- Error codes (structured, UI-facing; never a bare traceback)

ERR_RUNTIME_UNAVAILABLE = "runtime_unavailable"
ERR_RUNTIME_TIMEOUT = "runtime_timeout"
ERR_INVALID_WORKTREE_ID = "invalid_worktree_id"
ERR_UNKNOWN_WORKTREE = "unknown_worktree"
ERR_RUNTIME_RESPONSE = "invalid_runtime_response"
ERR_LAUNCH_FAILED = "orca_launch_failed"
ERR_INTERNAL = "internal_error"

# RPC bounds — the dashboard polls per badge; keep every call short.
_RPC_TIMEOUT_S = 5.0
_LAUNCH_TIMEOUT_S = 12.0

# Newline-delimited JSON frames can exceed socket buffer sizes on big ps
# snapshots; read until the envelope for our request id lands.
_MAX_FRAME_BYTES = 4 * 1024 * 1024


class OrcaRuntimeError(Exception):
    """Structured failure talking to the Orca runtime (code + message)."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


# --- Input validation --------------------------------------------------------

def _worktree_id_pattern():
    # Same identifier rule the kanban schema enforces (t_f15ce610); reuse the
    # canonical validator when importable so the two surfaces cannot drift.
    try:
        from hermes_cli import kanban_db
        return kanban_db
    except Exception:
        return None


def validate_worktree_id(worktree_id: Any) -> str:
    """Return the normalised link or raise OrcaRuntimeError(invalid)."""
    kb = _worktree_id_pattern()
    if kb is not None and hasattr(kb, "validate_worktree_id"):
        try:
            out = kb.validate_worktree_id(worktree_id)
        except ValueError as exc:
            raise OrcaRuntimeError(ERR_INVALID_WORKTREE_ID, str(exc)) from exc
        if out is None:
            raise OrcaRuntimeError(ERR_INVALID_WORKTREE_ID, "worktree_id must be a non-empty string identifier")
        return out
    # Local fallback mirroring kanban_db._WORKTREE_ID_PATTERN.
    import re
    if worktree_id is None:
        raise OrcaRuntimeError(ERR_INVALID_WORKTREE_ID, "worktree_id is required")
    if not isinstance(worktree_id, str):
        raise OrcaRuntimeError(ERR_INVALID_WORKTREE_ID, "worktree_id must be a string identifier")
    wid = worktree_id.strip()
    if not wid or len(wid) > 255 or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/:-]{0,254}", wid):
        raise OrcaRuntimeError(
            ERR_INVALID_WORKTREE_ID,
            "worktree_id must be a non-empty string identifier: start with a letter or digit, "
            "then letters/digits/./_/-//:, at most 255 characters")
    return wid


# --- Runtime metadata discovery ----------------------------------------------

_METADATA_FILENAME = "orca-runtime.json"

def _default_user_data_dir() -> Optional[Path]:
    """Platform default for Orca's Electron userData dir (app name "Orca")."""
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif sys.platform.startswith("win"):
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "orca"


def resolve_user_data_dir() -> Optional[Path]:
    """ORCA_USER_DATA_DIR override wins (tests + nonstandard installs)."""
    override = os.environ.get("ORCA_USER_DATA_DIR")
    if override:
        return Path(override)
    return _default_user_data_dir()


def read_runtime_metadata(user_data_dir: Optional[Path] = None) -> Optional[dict]:
    """Read orca-runtime.json; None when absent/incomplete (Orca not running)."""
    explicit = os.environ.get("ORCA_RUNTIME_METADATA")
    if explicit:
        path = Path(explicit)
    else:
        base = user_data_dir if user_data_dir is not None else resolve_user_data_dir()
        if base is None:
            return None
        path = Path(base) / _METADATA_FILENAME
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    transports = raw.get("transports")
    if not isinstance(transports, list) or not transports:
        # Legacy singular "transport" field (pre-transports-array hosts).
        legacy = raw.get("transport")
        transports = [legacy] if isinstance(legacy, dict) else []
    raw["transports"] = transports
    return raw


def _pick_unix_transport(metadata: dict) -> Optional[str]:
    for transport in metadata.get("transports") or []:
        if isinstance(transport, dict) and transport.get("kind") == "unix":
            endpoint = transport.get("endpoint")
            if isinstance(endpoint, str) and endpoint:
                return endpoint
    return None


# --- RPC client (mirrors src/cli/runtime/transport.ts framing) ---------------

def rpc_call(metadata: dict, method: str, params: Any = None,
             timeout_s: float = _RPC_TIMEOUT_S) -> Any:
    """One newline-delimited JSON-RPC call over the runtime's unix socket.

    Returns the response ``result``; raises OrcaRuntimeError with a stable
    code for every failure mode (unavailable / timeout / bad frame / RPC
    error envelope / runtime identity change mid-flight).
    """
    if sys.platform.startswith("win"):
        # Named-pipe transports need pywin32; the gateway never runs there.
        raise OrcaRuntimeError(ERR_RUNTIME_UNAVAILABLE, "Orca runtime RPC is unsupported on Windows")
    endpoint = _pick_unix_transport(metadata)
    if not endpoint:
        raise OrcaRuntimeError(ERR_RUNTIME_UNAVAILABLE, "No unix transport in Orca runtime metadata")
    runtime_id = metadata.get("runtimeId")
    auth_token = metadata.get("authToken")
    request_id = str(uuid.uuid4())
    frame = {"id": request_id, "authToken": auth_token, "method": method}
    if params is not None:
        frame["params"] = params
    payload = (json.dumps(frame) + "\n").encode("utf-8")

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout_s)
    try:
        sock.connect(endpoint)
        sock.sendall(payload)
        buf = b""
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise OrcaRuntimeError(ERR_RUNTIME_TIMEOUT, "Timed out waiting for the Orca runtime to respond")
            sock.settimeout(max(0.05, remaining))
            try:
                chunk = sock.recv(65536)
            except TimeoutError:
                raise OrcaRuntimeError(ERR_RUNTIME_TIMEOUT, "Timed out waiting for the Orca runtime to respond")
            if not chunk:
                raise OrcaRuntimeError(ERR_RUNTIME_UNAVAILABLE, "The Orca runtime closed the connection before responding")
            buf += chunk
            if len(buf) > _MAX_FRAME_BYTES:
                raise OrcaRuntimeError(ERR_RUNTIME_RESPONSE, "Orca runtime response exceeded the frame size limit")
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line.decode("utf-8", "replace"))
                except ValueError:
                    raise OrcaRuntimeError(ERR_RUNTIME_RESPONSE, "The Orca runtime returned an invalid response frame")
                if not isinstance(raw, dict):
                    continue
                if raw.get("_keepalive") is True:
                    continue  # refresh idle timeout; no terminal frame
                if raw.get("id") != request_id:
                    continue  # not ours (e.g. server-initiated frames)
                if raw.get("_meta") and runtime_id is not None:
                    resp_runtime = raw["_meta"].get("runtimeId")
                    if resp_runtime is not None and resp_runtime != runtime_id:
                        raise OrcaRuntimeError(ERR_RUNTIME_UNAVAILABLE, "The Orca runtime changed while the request was in flight")
                if raw.get("ok") is True:
                    return raw.get("result")
                err = raw.get("error") or {}
                code = err.get("code") or ERR_RUNTIME_RESPONSE
                raise OrcaRuntimeError(str(code), str(err.get("message") or "Orca runtime RPC failed"))
    except OSError as exc:
        raise OrcaRuntimeError(ERR_RUNTIME_UNAVAILABLE,
                               "Could not connect to the running Orca app. Start Orca and retry.") from exc
    finally:
        try:
            sock.close()
        except OSError:
            pass


# --- Layer 1: live worktree.ps ------------------------------------------------

def _ps_rows(metadata: dict) -> list:
    result = rpc_call(metadata, "worktree.ps", {})
    if isinstance(result, dict):
        rows = result.get("worktrees")
    else:
        rows = result
    if isinstance(rows, list):
        return [r for r in rows if isinstance(r, dict)]
    return []


def _find_row(rows: list, worktree_id: str) -> Optional[dict]:
    for row in rows:
        if row.get("worktreeId") == worktree_id:
            return row
    return None


# --- State derivation ----------------------------------------------------------

def derive_badge_state(row: dict) -> tuple[str, dict]:
    """Collapse one worktree.ps row onto the badge's 3-state vocabulary.

    Primary signal is the runtime's own per-worktree projection
    (``RuntimeWorktreeStatus``); agent rows corroborate when the status
    field is absent (older hosts). Returns (state, detail).
    """
    status = row.get("status")
    agents = row.get("agents")
    agent_states = [a.get("state") for a in agents if isinstance(a, dict)] if isinstance(agents, list) else []
    live = [s for s in agent_states if s in _AGENT_LIVE_STATES]
    done = [s for s in agent_states if s in _AGENT_DONE_STATES]

    if isinstance(status, str):
        s = status.lower()
        if s in _RUNTIME_STATUS_ACTIVE:
            return BADGE_ACTIVE, {"orca_status": status, "agent_states": agent_states}
        if s in _RUNTIME_STATUS_DONE:
            return BADGE_COMPLETE, {"orca_status": status, "agent_states": agent_states}
        if s in _RUNTIME_STATUS_INACTIVE:
            return BADGE_IDLE, {"orca_status": status, "agent_states": agent_states}
    # No/unknown runtime status: fall back to agent rows.
    if live:
        return BADGE_ACTIVE, {"orca_status": None, "agent_states": agent_states}
    if done:
        return BADGE_COMPLETE, {"orca_status": None, "agent_states": agent_states}
    return BADGE_IDLE, {"orca_status": None, "agent_states": agent_states}


# --- Layer 2: orca-status skill state.json --------------------------------------

def _default_skill_state_dirs() -> list:
    """Where the orca-status skill keeps state.json.

    Resolved through hermes_constants (same root the gateway itself uses),
    never a hardcoded Path.home() — profiled gateways and sandboxed test
    harnesses must agree on the skill location.
    """
    try:
        from hermes_constants import get_default_hermes_root
        roots = [Path(get_default_hermes_root())]
    except Exception:  # standalone use without the hermes package importable
        roots = [Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")]
    return [r / "skills" / "orca_orchestrator" / ".orca_orchestrator" for r in roots]


def _skill_state_entries() -> dict:
    """worktree_id -> entry from the orca-status skill's state.json.

    Precedence mirrors orca_status.resolve_config: ORCA_STATE_DIR env, then
    the skill's colocated .orca_orchestrator dir. Malformed files are treated
    as absent — this layer is a fallback, never a crash source.
    """
    candidates = []
    env_dir = os.environ.get("ORCA_STATE_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.extend(_default_skill_state_dirs())
    for base in candidates:
        path = Path(base) / "state.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict):
            return data
    return {}


_SKILL_STATUS_TO_BADGE = {
    "running": BADGE_ACTIVE,
    "done": BADGE_COMPLETE,
    "failed": BADGE_COMPLETE,
    "pending": BADGE_IDLE,
}


def _badge_from_skill_entry(entry: Any) -> Optional[str]:
    if isinstance(entry, dict):
        status = entry.get("status")
    else:
        status = entry
    if isinstance(status, str):
        return _SKILL_STATUS_TO_BADGE.get(status.lower())
    return None


# --- Public API ------------------------------------------------------------------

def get_worktree_status(worktree_id: Any) -> dict:
    """Badge payload for one worktree link. Never raises; degrades to unknown."""
    response = {
        "worktree_id": None,
        "state": BADGE_UNKNOWN,
        "source": "none",
        "orca_status": None,
        "agent_states": None,
        "model": None,
        "is_active_view": None,
        "workspace_status": None,
        "path": None,
        "last_activity_at": None,
        "updated_at": int(time.time()),
        "error": None,
        "error_code": None,
    }
    try:
        wid = validate_worktree_id(worktree_id)
    except OrcaRuntimeError as exc:
        response["error"] = exc.message
        response["error_code"] = exc.code
        return response
    response["worktree_id"] = wid

    metadata = read_runtime_metadata()
    if metadata:
        try:
            row = _find_row(_ps_rows(metadata), wid)
        except OrcaRuntimeError as exc:
            row = None
            ps_error = exc
        else:
            ps_error = None
        if row is not None:
            state, detail = derive_badge_state(row)
            response.update({
                "state": state,
                "source": "orca-runtime-rpc",
                "orca_status": detail["orca_status"],
                "agent_states": detail["agent_states"],
                "model": _first_agent_field(row, "agentType"),
                "is_active_view": row.get("isActive"),
                "workspace_status": row.get("workspaceStatus"),
                "path": row.get("path"),
                "last_activity_at": _epoch(row.get("lastActivityAt")),
            })
            return response
        # Runtime reachable but this worktree unknown to it — still better
        # evidence than the journal; report idle-with-no-session only if the
        # journal agrees the worktree exists, else unknown.
        if ps_error is None:
            entry_state = _badge_from_skill_entry(_skill_state_entries().get(wid))
            if entry_state is not None:
                response.update({"state": entry_state, "source": "orca-status-state"})
                return response
            response.update({
                "state": BADGE_UNKNOWN,
                "source": "orca-runtime-rpc",
                "error": f"worktree {wid} is not listed by the Orca runtime",
                "error_code": ERR_UNKNOWN_WORKTREE,
            })
            return response
        # ps itself failed (timeout etc.) — fall through to the journal.

    entries = _skill_state_entries()
    if wid in entries:
        state = _badge_from_skill_entry(entries[wid]) or BADGE_UNKNOWN
        response.update({"state": state, "source": "orca-status-state"})
        return response
    response["error"] = "Orca runtime unreachable and no orca-status journal entry for this worktree"
    response["error_code"] = ERR_RUNTIME_UNAVAILABLE
    return response


def _first_agent_field(row: dict, field: str) -> Optional[str]:
    agents = row.get("agents")
    if isinstance(agents, list):
        for agent in agents:
            if isinstance(agent, dict):
                value = agent.get(field)
                if isinstance(value, str) and value:
                    return value
    return None


def _epoch(value: Any) -> Optional[int]:
    if isinstance(value, (int, float)) and value > 0:
        return int(value)
    return None


# --- Reveal (badge click -> focus worktree in Orca UI) ---------------------------

def reveal_worktree(worktree_id: Any) -> dict:
    """Focus the worktree in Orca via ``worktree.activate``.

    Same RPC Orca's own mobile companion issues for its worktree "open"
    action. When the runtime is unreachable, launch Orca once (CLI
    ``orca open``, or ``open -a Orca`` on macOS when the CLI is absent) and
    retry the activation exactly once.
    """
    response = {
        "worktree_id": None,
        "ok": False,
        "method": "worktree.activate",
        "launched": False,
        "error": None,
        "error_code": None,
    }
    try:
        wid = validate_worktree_id(worktree_id)
    except OrcaRuntimeError as exc:
        response["error"] = exc.message
        response["error_code"] = exc.code
        return response
    response["worktree_id"] = wid

    params = {"worktree": f"id:{wid}", "navigation": "caller", "notifyClients": False}

    def attempt() -> Optional[OrcaRuntimeError]:
        metadata = read_runtime_metadata()
        if not metadata:
            return OrcaRuntimeError(ERR_RUNTIME_UNAVAILABLE, "Orca runtime metadata not found — is Orca running?")
        try:
            rpc_call(metadata, "worktree.activate", params)
        except OrcaRuntimeError as exc:
            return exc
        return None

    err = attempt()
    if err is None:
        response["ok"] = True
        return response

    # Runtime down: launch Orca once, then retry the activation.
    launch_err = _launch_orca()
    if launch_err is not None:
        response["error"] = f"{err.message} (launch: {launch_err.message})"
        response["error_code"] = err.code
        return response
    response["launched"] = True
    time.sleep(0.5)  # let the runtime bind its socket before retrying
    err = attempt()
    if err is None:
        response["ok"] = True
        return response
    response["error"] = err.message
    response["error_code"] = err.code
    return response


def _launch_orca() -> Optional[OrcaRuntimeError]:
    """Best-effort Orca launch. Returns None on success, error otherwise."""
    cli = os.environ.get("ORCA_CLI_COMMAND") or "orca"
    try:
        completed = subprocess.run(
            [cli, "open", "--json"], capture_output=True, text=True, timeout=_LAUNCH_TIMEOUT_S, check=False)
        if completed.returncode == 0:
            return None
        cli_failed = completed.returncode != 0
    except (OSError, subprocess.TimeoutExpired):
        cli_failed = True
    if not cli_failed:
        return None
    if sys.platform == "darwin":
        # `open -a Orca` raises the app (launching it when installed).
        try:
            completed = subprocess.run(["open", "-a", "Orca"], capture_output=True, text=True,
                                       timeout=_LAUNCH_TIMEOUT_S, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return OrcaRuntimeError(ERR_LAUNCH_FAILED, f"Could not launch Orca: {exc}")
        if completed.returncode != 0:
            return OrcaRuntimeError(ERR_LAUNCH_FAILED,
                                    "Could not launch Orca (open -a Orca failed — is Orca installed?)")
        return None
    return OrcaRuntimeError(ERR_LAUNCH_FAILED, "Orca CLI launch failed and no platform launch fallback exists")
