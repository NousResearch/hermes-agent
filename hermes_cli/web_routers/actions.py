"""Gateway restart/drain, Hermes update and background-action status dashboard routes.

Extracted from ``hermes_cli.web_server``; helpers/state that tests monkeypatch on
``web_server`` stay there and are late-bound (cycle-safe).
"""

import asyncio
import contextlib
import logging
import re
import secrets
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request

from hermes_cli import __version__
from hermes_cli.config import format_docker_update_message, recommended_update_command_for_method
from hermes_cli.web_deps import LateState, late
from hermes_cli.web_server_gateway import _ACTION_LOG_FILES
from hermes_cli.web_routers._common import http_failure

_log = logging.getLogger("hermes_cli.web_server")
router = APIRouter()
status_router = APIRouter()

# Late-bound so a test's monkeypatch on the owning module wins at call time.
_dashboard_local_update_managed_externally = late("_dashboard_local_update_managed_externally", "hermes_cli.web_server_files")
_spawn_gateway_restart = late("_spawn_gateway_restart")
_spawn_hermes_action = late("_spawn_hermes_action", "hermes_cli.web_server_gateway")
detect_install_method = late("detect_install_method", "hermes_cli.config")
get_hermes_home = late("get_hermes_home", "hermes_cli.config")
_ACTION_COMMANDS = LateState("_ACTION_COMMANDS", "hermes_cli.web_server_gateway")
_ACTION_IDS = LateState("_ACTION_IDS", "hermes_cli.web_server_gateway")
_ACTION_PROCS = LateState("_ACTION_PROCS", "hermes_cli.web_server_gateway")
_ACTION_RESULTS = LateState("_ACTION_RESULTS", "hermes_cli.web_server_gateway")


def _server_path(name: str) -> Path:
    """Live Path value (``_ACTION_LOG_DIR`` on web_server_gateway, ``PROJECT_ROOT`` on
    web_server; plain values are not proxied by LateState)."""
    import hermes_cli.web_server as ws
    import hermes_cli.web_server_gateway as gw
    return getattr(gw if name == "_ACTION_LOG_DIR" else ws, name)


_ACTION_LOG_TAIL_MAX_BYTES = 256 * 1024
_ACTION_LOG_TAIL_INITIAL_CHUNK_BYTES = 8 * 1024
_ACTION_LOG_TAIL_MAX_CHUNK_BYTES = 64 * 1024

_UPDATE_ACTION_COMPLETED_RE = re.compile(r"^=== hermes-update completed ([0-9a-f]{32}) ===$")

_MANAGED_EXTERNALLY_MESSAGE = "Hermes updates are managed outside this dashboard in containerized environments."

# Per-kind dashboard error codes the UI keys on, by admission-refusal code.
_UPDATE_REFUSAL_ERROR_CODES = {
    "docker": "docker_update_unsupported", "image-marker": "docker_update_unsupported",
    "image-marker-invalid": "docker_update_unsupported", "apt": "apt_update_required",
    "nix": "nix_update_unsupported",
}


def _finish_action(name: str, exit_code: Optional[int], pid: Optional[int]) -> None:
    """Record a terminal result and drop the live-process registries for ``name``."""
    _ACTION_RESULTS[name] = {"exit_code": exit_code, "pid": pid}
    for registry in (_ACTION_PROCS, _ACTION_COMMANDS, _ACTION_IDS):
        registry.pop(name, None)


def _record_completed_action(name: str, message: str, exit_code: int = 1) -> None:
    """Record a non-spawned action result and write it to the action log."""
    log_dir = _server_path("_ACTION_LOG_DIR")
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / _ACTION_LOG_FILES[name], "ab", buffering=0) as log_file:
        log_file.write(f"\n=== {name} completed {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n".encode())
        log_file.write(message.encode("utf-8", errors="replace"))
        if not message.endswith("\n"):
            log_file.write(b"\n")
    _finish_action(name, exit_code, None)


def _tail_lines(path: Path, n: int) -> List[str]:
    """Return the last ``n`` lines of ``path`` without loading huge logs."""
    try:
        size = path.stat().st_size
    except OSError:
        return []
    if n <= 0 or size <= 0:
        return []

    min_offset = max(0, size - _ACTION_LOG_TAIL_MAX_BYTES)
    offset = size
    chunk_size = _ACTION_LOG_TAIL_INITIAL_CHUNK_BYTES
    newline_count = 0
    chunks: List[bytes] = []
    drop_partial_first_line = False
    try:
        with path.open("rb") as handle:
            while offset > min_offset and newline_count <= n:
                read_size = min(chunk_size, offset - min_offset)
                offset -= read_size
                handle.seek(offset)
                chunk = handle.read(read_size)
                chunks.append(chunk)
                newline_count += chunk.count(b"\n")
                chunk_size = min(chunk_size * 2, _ACTION_LOG_TAIL_MAX_CHUNK_BYTES)
            if offset > 0:
                handle.seek(offset - 1)
                drop_partial_first_line = handle.read(1) != b"\n"
    except OSError:
        return []

    lines = b"".join(reversed(chunks)).decode("utf-8", errors="replace").splitlines()
    if drop_partial_first_line and lines:
        lines = lines[1:]
    return lines[-n:]


def _durable_completed_update_action_id(lines: List[str]) -> Optional[str]:
    """Latest successful update id from ``update.log`` — the durable record that survives
    the update restarting the dashboard (losing the in-memory ``Popen``/result registries).
    Only a completion marker after the latest start marker counts, so a stale success
    cannot mask a newer failed attempt."""
    last_start = last_completed = -1
    completed_action_id: Optional[str] = None
    for index, line in enumerate(lines):
        if line.startswith("=== hermes update started "):
            last_start = index
        match = _UPDATE_ACTION_COMPLETED_RE.fullmatch(line.strip())
        if match:
            last_completed = index
            completed_action_id = match.group(1)
    return completed_action_id if completed_action_id and last_completed > last_start else None


@router.post("/api/gateway/restart")
async def restart_gateway(profile: Optional[str] = None):
    """Kick off a ``hermes gateway restart`` in the background."""
    with http_failure("Failed to spawn gateway restart", 500, "Failed to restart gateway"):
        proc, _reused = _spawn_gateway_restart(profile)
    return {"ok": True, "pid": proc.pid, "name": "gateway-restart"}


@router.post("/api/gateway/drain")
async def gateway_drain(request: Request):
    """Begin or cancel an external (NAS-driven) gateway drain.

    Authenticated by the non-interactive token-auth seam (the ``dashboard_auth/drain``
    plugin registers this path as a token route and verifies the bearer secret);
    without that plugin the cookie gate covers a gated bind and the legacy
    session-token gate a loopback bind — never unauthenticated on a network bind.

    Body ``{"action": "drain"|"cancel"}``. Only the ``.drain_request.json`` marker is
    written/removed here — the gateway's ``_drain_control_watcher`` owns the state
    transition (the marker IS the control channel). Idempotent on both sides;
    ``POST /api/gateway/restart`` is the force-override that supersedes a drain.
    """
    from gateway.drain_control import clear_drain_request, drain_requested, write_drain_request

    try:
        body = await request.json()
    except Exception:
        body = {}
    body = body or {}
    action = str(body.get("action", "drain")).strip().lower()
    # Attribute to the verified token principal when the token-auth seam attached one.
    principal = getattr(getattr(request.state, "token_principal", None), "principal", None) or "dashboard"

    if action == "cancel":
        existed = clear_drain_request()
        _log.info("Gateway drain CANCEL requested by %s (existed=%s)", principal, existed)
        return {"ok": True, "action": "cancel", "was_draining": existed}
    if action != "drain":
        raise HTTPException(status_code=400, detail=f"Unknown drain action {action!r}; expected 'drain' or 'cancel'")

    payload = write_drain_request(
        principal=str(principal), suppress_notification=bool(body.get("suppress_notification", False)),
    )
    _log.info(
        "Gateway drain BEGIN requested by %s (suppress_notification=%s)", principal, payload["suppress_notification"],
    )
    return {
        "ok": True, "action": "drain", "requested_at": payload["requested_at"],
        # Echo so a caller polling /api/status knows the marker is now set;
        # the gateway watcher flips gateway_state -> draining within ~1s.
        "draining": drain_requested(), "suppress_notification": payload["suppress_notification"],
    }


def _update_refused(error: str, message: str, update_command: str) -> Dict[str, Any]:
    _record_completed_action("hermes-update", message, exit_code=1)
    return {
        "ok": False, "pid": None, "name": "hermes-update", "error": error, "message": message,
        "update_command": update_command,
    }


@router.post("/api/hermes/update")
async def update_hermes():
    """Kick off ``hermes update`` in the background."""
    if _dashboard_local_update_managed_externally():
        message = _MANAGED_EXTERNALLY_MESSAGE + " The built-in local updater is disabled here."
        return _update_refused("dashboard_update_managed_externally", message, "managed outside dashboard")

    # Shared admission gate: marker-first, then the docker/nix/apt heuristics —
    # one decision with the CLI paths.
    from hermes_cli.update_contract import evaluate_update_admission, record_refusal_receipt

    refusal = evaluate_update_admission(_server_path("PROJECT_ROOT"))
    if refusal is not None:
        response = _update_refused(
            _UPDATE_REFUSAL_ERROR_CODES.get(refusal.code, "update_not_in_place"), refusal.message, refusal.update_command,
        )
        record_refusal_receipt(refusal)
        return response

    existing = _ACTION_PROCS.get("hermes-update")
    if existing is not None and existing.poll() is None:
        response = {"ok": True, "pid": existing.pid, "name": "hermes-update", "already_running": True}
        action_id = _ACTION_IDS.get("hermes-update")
        if action_id:
            response["action_id"] = action_id
        return response

    action_id = secrets.token_hex(16)
    with http_failure("Failed to spawn hermes update", 500, "Failed to start update"):
        proc = _spawn_hermes_action(["update"], "hermes-update", env_overrides={"HERMES_ACTION_ID": action_id})
    return {"ok": True, "pid": proc.pid, "name": "hermes-update", "action_id": action_id}


def _recent_upstream_commits(n: int = 20) -> List[Dict[str, Any]]:
    """Commits the local checkout is behind ``origin/main`` by, newest first; [] on any failure.

    Logs the SAME range the behind-count uses (``HEAD..origin/main``, see
    ``banner._check_via_local_git``), NOT ``@{upstream}``: on a feature branch that is
    the branch's own tip (zero commits), leaving the changelog empty while the count is non-zero.
    """
    try:
        # git log emits UTF-8 (emoji/CJK subjects). On Windows text=True defaults to
        # the ANSI code page; an undefined cp1252 byte crashed the stdlib
        # _readerthread and killed the desktop backend — hence encoding="utf-8".
        out = subprocess.run(
            [
                "git", "-C", str(_server_path("PROJECT_ROOT")), "log", "--format=%H%x1f%s%x1f%an%x1f%ct",
                "HEAD..origin/main", f"-n{int(n)}",
            ],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5,
        )
        if out.returncode != 0:
            return []
        rows: List[Dict[str, Any]] = []
        for line in out.stdout.splitlines():
            if line.strip():
                sha, summary, author, at = (line.split("\x1f") + ["", "", "", "0"])[:4]
                rows.append({"sha": sha[:7], "summary": summary, "author": author, "at": int(at or 0)})
        return rows
    except Exception:
        return []


_NON_APPLYABLE_MESSAGES = {
    "docker": format_docker_update_message,
    "apt": lambda: "Hermes is managed by Termux APT; run `pkg upgrade hermes-agent`.",
}


@router.get("/api/hermes/update/check")
async def check_hermes_update(force: bool = False):
    """Report whether a Hermes update is available, without applying it.

    Returns install_method ('apt'|'git'|'docker'|'nix'|'nixos'|'unknown'),
    current_version, behind (commits behind, 0 = up to date, -1 = unknown count,
    null = check could not run), update_available, can_apply (git only — the
    dashboard button can apply in place), update_command, message (guidance for
    non-applyable methods) and, for git installs that are behind, commits
    [{sha, summary, author, at}] (additive; existing consumers ignore it).
    """
    if _dashboard_local_update_managed_externally():
        return {
            "install_method": "managed-runtime", "current_version": __version__, "behind": None,
            "update_available": False, "can_apply": False,
            "update_command": "managed outside dashboard", "message": _MANAGED_EXTERNALLY_MESSAGE,
        }

    install_method = detect_install_method(_server_path("PROJECT_ROOT"))
    payload: Dict[str, Any] = {
        "install_method": install_method, "current_version": __version__, "behind": None,
        "update_available": False, "can_apply": install_method == "git",
        "update_command": recommended_update_command_for_method(install_method), "message": None,
    }
    non_applyable = _NON_APPLYABLE_MESSAGES.get(install_method)
    if non_applyable is not None:
        payload["message"] = non_applyable()
        return payload

    # banner.check_for_updates() handles git / nix-revision paths and caches
    # the result for 6h. ``force`` busts the cache so "Check now" reflects reality.
    try:
        from hermes_cli.banner import check_for_updates

        if force:
            with contextlib.suppress(OSError):
                (get_hermes_home() / ".update_check").unlink()
        behind = await asyncio.to_thread(check_for_updates)
    except Exception:
        _log.exception("Update check failed")
        behind = None

    payload["behind"] = behind
    if behind is None:
        payload["message"] = "Couldn't reach the update source — try again later."
    elif behind == 0:
        payload["message"] = "You're on the latest version."
    else:
        payload["update_available"] = True
        # "What's changed" for the desktop's remote update overlay; git only,
        # best-effort (empty list on any failure).
        if install_method == "git":
            payload["commits"] = await asyncio.to_thread(_recent_upstream_commits)
    return payload


def _completed_exit_code(
    result: Optional[Dict[str, Any]], durable_action_id: Optional[str], receipt: Optional[Dict[str, Any]],
) -> Optional[int]:
    """Exit code for an action with no live process: in-memory result, else durable evidence."""
    if result is not None:
        return result.get("exit_code")
    if durable_action_id:
        return 0
    if receipt is not None and receipt.get("outcome") in ("success", "partial"):
        # No in-memory result and no log marker (e.g. log rotated), but the
        # receipt proves a completed run: report its outcome rather than a
        # null clients time out on. ``partial`` maps to exit 1 like the CLI.
        return 0 if receipt["outcome"] == "success" else 1
    return None


@status_router.get("/api/actions/{name}/status")
async def get_action_status(name: str, lines: int = 200):
    """Tail an action log and report whether the process is still running."""
    log_file_name = _ACTION_LOG_FILES.get(name)
    if log_file_name is None:
        raise HTTPException(status_code=404, detail=f"Unknown action: {name}")

    log_dir = _server_path("_ACTION_LOG_DIR")
    requested_lines = min(max(lines, 1), 2000)
    tail = _tail_lines(log_dir / log_file_name, requested_lines)

    durable_update_action_id = None
    update_receipt_summary = None
    if name == "hermes-update":
        durable_update_action_id = _durable_completed_update_action_id(_tail_lines(log_dir / "update.log", 2000))
        if durable_update_action_id:
            marker = f"=== hermes-update completed {durable_update_action_id} ==="
            if marker not in tail:
                tail = [*tail, marker][-requested_lines:]
        # The update receipt is the durable, structured truth about the last
        # update (written by every run, incl. refused/failed; survives the
        # dashboard restarting itself mid-action). Surface it so clients READ
        # the outcome instead of inferring it from liveness probes.
        # See #81193, #87359, #91277.
        update_receipt_summary = _latest_update_receipt_summary()

    proc = _ACTION_PROCS.get(name)
    if proc is None:
        result = _ACTION_RESULTS.get(name)
        running = False
        pid = result.get("pid") if result else None
        exit_code = _completed_exit_code(result, durable_update_action_id, update_receipt_summary)
    else:
        exit_code = proc.poll()
        running = exit_code is None
        pid = proc.pid
        if exit_code is not None:
            with contextlib.suppress(Exception):
                proc.wait(timeout=1)
            _finish_action(name, exit_code, pid)

    response = {"name": name, "running": running, "exit_code": exit_code, "pid": pid, "lines": tail}
    if durable_update_action_id:
        response["action_id"] = durable_update_action_id
    if update_receipt_summary is not None:
        response["receipt"] = update_receipt_summary
    return response


def _read_latest_receipt() -> Optional[Dict[str, Any]]:
    """Latest update receipt, or None on any failure (never raises)."""
    try:
        from hermes_cli.update_receipt import read_latest_receipt
        return read_latest_receipt() or None
    except Exception:
        return None


def _latest_update_receipt_summary() -> Optional[Dict[str, Any]]:
    """Compact summary of the latest receipt (written by EVERY ``hermes update`` run,
    incl. refused/failed), or None; never raises. Steps/skips stay in the full endpoint.

    Phase-1 bullet 3 (#91277): the receipt (written by EVERY ``hermes update`` run since #91283, including
    refused and failed ones, with a ``latest.json`` pointer) is the durable success signal the Desktop and
    dashboard should read instead of inferring outcomes from liveness probes across the update's stop/start
    gap (#81193, #87359).
    """
    receipt = _read_latest_receipt()
    if not receipt:
        return None
    try:
        post = receipt.get("post_update") or {}
        return {
            **{k: receipt.get(k) for k in ("outcome", "started_at", "finished_at")},
            "pre_sha": (receipt.get("pre_update") or {}).get("sha"),
            "post_sha": post.get("sha"), "post_version": post.get("version"),
            "fleet_states": sorted({str(e.get("state")) for e in receipt.get("fleet") or [] if isinstance(e, dict)}),
        }
    except Exception:
        return None


@status_router.get("/api/hermes/update/receipt")
async def get_update_receipt():
    """The FULL latest update receipt (steps, skips, gateway restart outcome, fleet
    matrix) plus a compact ``summary``; 404 when no update has run since receipts landed.
    Clients read this instead of inferring success from backend liveness, which misread
    the update's own restart gap as a failed update/boot.

    See #81193, #87359, #91277.
    """
    receipt = _read_latest_receipt()
    if not receipt:
        raise HTTPException(status_code=404, detail="No update receipt found (no `hermes update` run recorded).")
    return {"receipt": receipt, "summary": _latest_update_receipt_summary()}
