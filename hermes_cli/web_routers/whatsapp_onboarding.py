"""WhatsApp onboarding routes for the dashboard.

Extracted from hermes_cli/web_server.py (god-file slice R4-C3, epic #78791):
the WhatsApp bridge onboarding cluster (spawn/watch/apply/cancel + session
lifecycle).  Cross-cluster helpers resolve through web_deps.late().
"""

from datetime import datetime, timezone
import json
import logging
import re
import secrets
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException

from hermes_cli.web_deps import late
from hermes_cli.web_models import WhatsAppOnboardingApply, WhatsAppOnboardingStart

save_env_value = late("save_env_value")
remove_env_value = late("remove_env_value")

_log = logging.getLogger("hermes_cli.web_server")

router = APIRouter()

_config_profile_scope = late("_config_profile_scope")
_spawn_gateway_restart = late("_spawn_gateway_restart")
_write_platform_enabled = late("_write_platform_enabled")

_WHATSAPP_ONBOARDING_TTL_SECONDS = 600
_WHATSAPP_ONBOARDING_TERMINAL_STATUSES = {"connected", "error", "expired", "cancelled"}


@dataclass
class _WhatsAppOnboardingSession:
    proc: subprocess.Popen | None
    mode: str
    allowed_users: str
    session_path: str
    expires_at: str
    expires_at_ts: float
    profile: str | None = None
    status: str = "starting"
    qr_payload: str | None = None
    account_id: str | None = None
    account_name: str | None = None
    account_phone: str | None = None
    error: str | None = None


_whatsapp_onboarding_sessions: dict[str, _WhatsAppOnboardingSession] = {}
_whatsapp_onboarding_lock = threading.RLock()


def _utc_iso_from_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_whatsapp_onboarding_mode(value: Any) -> str:
    mode = str(value or "bot").strip().lower()
    if mode not in {"bot", "self-chat"}:
        raise HTTPException(status_code=400, detail="WhatsApp mode must be 'bot' or 'self-chat'.")
    return mode


def _normalize_whatsapp_allowed_users(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    return ",".join(part.replace(" ", "") for part in raw.split(",") if part.strip())


def _whatsapp_session_path() -> Path:
    from hermes_constants import get_hermes_dir

    return get_hermes_dir("platforms/whatsapp/session", "whatsapp/session")


def _whatsapp_phone_from_identifier(value: Any) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    candidate = raw.split("@", 1)[0].split(":", 1)[0]
    digits = re.sub(r"\D+", "", candidate)
    return digits or None


def _whatsapp_linked_account_from_session(session_path: Path) -> tuple[str | None, str | None, str | None]:
    creds_path = session_path / "creds.json"
    try:
        payload = json.loads(creds_path.read_text(encoding="utf-8"))
    except Exception:
        return None, None, None

    account_id: str | None = None
    account_name: str | None = None

    def collect(candidate: Any) -> None:
        nonlocal account_id, account_name
        if not isinstance(candidate, dict):
            return
        if account_id is None:
            for key in ("id", "jid", "lid"):
                value = str(candidate.get(key) or "").strip()
                if value:
                    account_id = value
                    break
        if account_name is None:
            for key in ("name", "verifiedName", "notify", "pushName"):
                value = str(candidate.get(key) or "").strip()
                if value:
                    account_name = value
                    break

    collect(payload.get("me"))
    collect(payload.get("account"))
    collect(payload)
    return account_id, account_name, _whatsapp_phone_from_identifier(account_id)


def _ensure_whatsapp_bridge_dependencies(bridge_dir: Path) -> None:
    """Install bridge dependencies when the dashboard is the setup surface."""
    if (bridge_dir / "node_modules").exists():
        return

    from hermes_constants import find_node_executable, with_hermes_node_path
    from utils import env_int

    npm = find_node_executable("npm")
    if not npm:
        raise HTTPException(
            status_code=500,
            detail="npm was not found. WhatsApp setup needs Node.js and npm.",
        )

    timeout = env_int("WHATSAPP_NPM_INSTALL_TIMEOUT", 300)
    try:
        result = subprocess.run(
            [npm, "install", "--silent"],
            cwd=str(bridge_dir),
            capture_output=True,
            text=True,
            # npm output is UTF-8; guard the Windows ANSI-code-page default
            # against undefined bytes crashing the reader thread (#52649).
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=with_hermes_node_path(),
            creationflags=windows_hide_flags(),
        )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(
            status_code=500,
            detail="Installing WhatsApp bridge dependencies timed out.",
        ) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to install WhatsApp bridge dependencies: {exc}",
        ) from exc

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        if detail:
            detail = "\n".join(detail.splitlines()[-10:])
        raise HTTPException(
            status_code=500,
            detail=f"npm install failed for WhatsApp bridge: {detail or 'no output'}",
        )


def _spawn_whatsapp_pairing_process(session_path: Path, mode: str) -> subprocess.Popen:
    from gateway.platforms.whatsapp_common import resolve_whatsapp_bridge_dir
    from hermes_constants import find_node_executable, with_hermes_node_path

    bridge_dir = resolve_whatsapp_bridge_dir()
    bridge_script = bridge_dir / "bridge.js"
    if not bridge_script.exists():
        raise HTTPException(
            status_code=500,
            detail=f"WhatsApp bridge script was not found at {bridge_script}.",
        )
    node = find_node_executable("node")
    if not node:
        raise HTTPException(
            status_code=500,
            detail="Node.js was not found. WhatsApp setup needs Node.js.",
        )

    _ensure_whatsapp_bridge_dependencies(bridge_dir)
    session_path.mkdir(parents=True, exist_ok=True)

    env = with_hermes_node_path()
    env["WHATSAPP_MODE"] = mode
    env["WHATSAPP_DM_POLICY"] = "pairing"
    return subprocess.Popen(
        [
            node,
            str(bridge_script),
            "--pair-only",
            "--pair-json",
            "--session",
            str(session_path),
        ],
        cwd=str(bridge_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        start_new_session=True,
        env=env,
        creationflags=windows_hide_flags(),
    )


def _terminate_whatsapp_pairing(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=3)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _watch_whatsapp_pairing(pairing_id: str, proc: subprocess.Popen) -> None:
    try:
        stream = proc.stdout
        if stream is not None:
            for line in stream:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                event = str(payload.get("event") or "").strip()
                with _whatsapp_onboarding_lock:
                    record = _whatsapp_onboarding_sessions.get(pairing_id)
                    if not record or record.proc is not proc:
                        return
                    if event == "qr":
                        qr = str(payload.get("qr") or "").strip()
                        if qr:
                            record.qr_payload = qr
                            record.status = "waiting"
                            record.error = None
                    elif event == "connected":
                        user = payload.get("user")
                        if isinstance(user, dict):
                            account_id = str(user.get("id") or "").strip()
                            account_name = str(user.get("name") or "").strip()
                            record.account_id = account_id or None
                            record.account_name = account_name or None
                            record.account_phone = _whatsapp_phone_from_identifier(account_id)
                        record.status = "connected"
                        record.error = None
                    elif event == "error":
                        record.status = "error"
                        record.error = str(payload.get("error") or "WhatsApp pairing failed.")
                    elif event == "disconnected" and record.status == "starting":
                        record.status = "waiting"
        returncode = proc.wait()
    except Exception as exc:
        with _whatsapp_onboarding_lock:
            record = _whatsapp_onboarding_sessions.get(pairing_id)
            if record and record.proc is proc and record.status not in _WHATSAPP_ONBOARDING_TERMINAL_STATUSES:
                record.status = "error"
                record.error = str(exc)
        return

    with _whatsapp_onboarding_lock:
        record = _whatsapp_onboarding_sessions.get(pairing_id)
        if not record or record.proc is not proc:
            return
        if record.status in {"connected", "cancelled", "expired"}:
            return
        record.status = "error"
        record.error = (
            "WhatsApp pairing process exited before pairing completed."
            if returncode == 0
            else f"WhatsApp pairing process exited with code {returncode}."
        )


def _run_whatsapp_pairing(pairing_id: str, session_path: Path, mode: str) -> None:
    with _whatsapp_onboarding_lock:
        record = _whatsapp_onboarding_sessions.get(pairing_id)
        if not record or record.status in _WHATSAPP_ONBOARDING_TERMINAL_STATUSES:
            return
        record.status = "installing"

    try:
        proc = _spawn_whatsapp_pairing_process(session_path, mode)
    except Exception as exc:
        with _whatsapp_onboarding_lock:
            record = _whatsapp_onboarding_sessions.get(pairing_id)
            if record and record.status not in _WHATSAPP_ONBOARDING_TERMINAL_STATUSES:
                record.status = "error"
                record.error = str(exc)
        return

    with _whatsapp_onboarding_lock:
        record = _whatsapp_onboarding_sessions.get(pairing_id)
        if not record or record.status in _WHATSAPP_ONBOARDING_TERMINAL_STATUSES:
            _terminate_whatsapp_pairing(proc)
            return
        record.proc = proc
        record.status = "starting"

    _watch_whatsapp_pairing(pairing_id, proc)


def _prune_whatsapp_onboarding_sessions() -> None:
    now = time.time()
    remove_ids: list[str] = []
    for pairing_id, record in _whatsapp_onboarding_sessions.items():
        if (
            record.proc is not None
            and record.status not in _WHATSAPP_ONBOARDING_TERMINAL_STATUSES
            and record.proc.poll() is not None
        ):
            record.status = "error"
            record.error = "WhatsApp pairing process exited before pairing completed."
        if record.expires_at_ts <= now and record.status not in _WHATSAPP_ONBOARDING_TERMINAL_STATUSES:
            _terminate_whatsapp_pairing(record.proc)
            record.status = "expired"
            record.error = "WhatsApp QR setup expired. Start a new setup."
        if record.status in _WHATSAPP_ONBOARDING_TERMINAL_STATUSES and record.expires_at_ts + 300 <= now:
            remove_ids.append(pairing_id)
    for pairing_id in remove_ids:
        _whatsapp_onboarding_sessions.pop(pairing_id, None)


def _supersede_whatsapp_onboarding_sessions(session_path: Path) -> None:
    for existing in _whatsapp_onboarding_sessions.values():
        if existing.session_path == str(session_path) and existing.status not in _WHATSAPP_ONBOARDING_TERMINAL_STATUSES:
            existing.status = "cancelled"
            existing.error = "Superseded by a newer WhatsApp setup session."
            _terminate_whatsapp_pairing(existing.proc)


def _whatsapp_onboarding_payload(pairing_id: str, record: _WhatsAppOnboardingSession) -> dict[str, Any]:
    return {
        "pairing_id": pairing_id,
        "status": record.status,
        "qr_payload": record.qr_payload,
        "expires_at": record.expires_at,
        "mode": record.mode,
        "allowed_users": record.allowed_users,
        "account_id": record.account_id,
        "account_name": record.account_name,
        "account_phone": record.account_phone,
        "error": record.error,
    }


def _restart_gateway_after_whatsapp_onboarding(profile: Optional[str] = None) -> dict[str, Any]:
    try:
        proc, reused = _spawn_gateway_restart(profile)
    except Exception as exc:
        _log.exception("Failed to auto-restart gateway after WhatsApp onboarding")
        return {
            "restart_started": False,
            "restart_error": str(exc),
        }
    if reused:
        _log.info(
            "WhatsApp onboarding: reusing in-flight gateway restart (pid %s)",
            proc.pid,
        )
    return {
        "restart_started": True,
        "restart_action": "gateway-restart",
        "restart_pid": proc.pid,
    }


@router.post("/api/messaging/whatsapp/onboarding/start")
async def start_whatsapp_onboarding(body: WhatsAppOnboardingStart):
    mode = _normalize_whatsapp_onboarding_mode(body.mode)
    allowed_users = _normalize_whatsapp_allowed_users(body.allowed_users)
    effective_profile = body.profile

    with _config_profile_scope(effective_profile):
        session_path = late("_whatsapp_session_path")()
        expires_at_ts = time.time() + _WHATSAPP_ONBOARDING_TTL_SECONDS
        expires_at = _utc_iso_from_ts(expires_at_ts)
        if (session_path / "creds.json").exists():
            import hermes_cli.web_server as _ws
            pairing_id = _ws.secrets.token_urlsafe(16)
            account_id, account_name, account_phone = _whatsapp_linked_account_from_session(session_path)
            record = _WhatsAppOnboardingSession(
                proc=None,
                mode=mode,
                allowed_users=allowed_users,
                session_path=str(session_path),
                expires_at=expires_at,
                expires_at_ts=expires_at_ts,
                profile=effective_profile,
                status="connected",
                account_id=account_id,
                account_name=account_name,
                account_phone=account_phone,
            )
            with _whatsapp_onboarding_lock:
                _prune_whatsapp_onboarding_sessions()
                _supersede_whatsapp_onboarding_sessions(session_path)
                _whatsapp_onboarding_sessions[pairing_id] = record
            return _whatsapp_onboarding_payload(pairing_id, record)

        import hermes_cli.web_server as _ws
        pairing_id = _ws.secrets.token_urlsafe(16)
    record = _WhatsAppOnboardingSession(
        proc=None,
        mode=mode,
        allowed_users=allowed_users,
        session_path=str(session_path),
        expires_at=expires_at,
        expires_at_ts=expires_at_ts,
        profile=effective_profile,
    )

    with _whatsapp_onboarding_lock:
        _prune_whatsapp_onboarding_sessions()
        _supersede_whatsapp_onboarding_sessions(session_path)
        _whatsapp_onboarding_sessions[pairing_id] = record

    threading.Thread(
        target=_run_whatsapp_pairing,
        args=(pairing_id, session_path, mode),
        daemon=True,
    ).start()

    return _whatsapp_onboarding_payload(pairing_id, record)


@router.get("/api/messaging/whatsapp/onboarding/{pairing_id}")
async def get_whatsapp_onboarding_status(pairing_id: str):
    with _whatsapp_onboarding_lock:
        _prune_whatsapp_onboarding_sessions()
        record = _whatsapp_onboarding_sessions.get(pairing_id)
        if not record:
            raise HTTPException(
                status_code=404,
                detail="WhatsApp setup session was not found. Start a new setup.",
            )
        if record.status == "expired":
            raise HTTPException(status_code=410, detail=record.error or "WhatsApp setup expired.")
        return _whatsapp_onboarding_payload(pairing_id, record)


@router.post("/api/messaging/whatsapp/onboarding/{pairing_id}/apply")
async def apply_whatsapp_onboarding(
    pairing_id: str, body: WhatsAppOnboardingApply, profile: Optional[str] = None
):
    with _whatsapp_onboarding_lock:
        _prune_whatsapp_onboarding_sessions()
        record = _whatsapp_onboarding_sessions.get(pairing_id)
        if not record:
            raise HTTPException(
                status_code=404,
                detail="WhatsApp setup session was not found. Start a new setup.",
            )
        if record.status != "connected":
            raise HTTPException(status_code=409, detail="WhatsApp setup is not connected yet.")
        mode = _normalize_whatsapp_onboarding_mode(body.mode or record.mode)
        allowed_users = _normalize_whatsapp_allowed_users(
            record.allowed_users if body.allowed_users is None else body.allowed_users
        )
        if mode == "self-chat" and not allowed_users:
            allowed_users = record.account_phone or record.account_id or ""
        record_profile = record.profile

    effective_profile = body.profile or profile or record_profile
    try:
        with _config_profile_scope(effective_profile):
            save_env_value("WHATSAPP_MODE", mode)
            save_env_value("WHATSAPP_DM_POLICY", "pairing")
            if allowed_users:
                save_env_value("WHATSAPP_ALLOWED_USERS", allowed_users)
            # Blank means "keep the existing allowlist"; explicit clearing
            # still lives in the normal config editor where the field is visible.
            save_env_value("WHATSAPP_ENABLED", "true")
            _write_platform_enabled("whatsapp", True)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        _log.exception("WhatsApp onboarding apply failed")
        raise HTTPException(
            status_code=500,
            detail="Failed to save WhatsApp setup.",
        ) from exc

    with _whatsapp_onboarding_lock:
        _whatsapp_onboarding_sessions.pop(pairing_id, None)

    restart_result = _restart_gateway_after_whatsapp_onboarding(effective_profile)
    return {
        "ok": True,
        "platform": "whatsapp",
        "needs_restart": not restart_result["restart_started"],
        **restart_result,
    }


@router.delete("/api/messaging/whatsapp/onboarding/{pairing_id}")
async def cancel_whatsapp_onboarding(pairing_id: str):
    with _whatsapp_onboarding_lock:
        record = _whatsapp_onboarding_sessions.pop(pairing_id, None)
    if record:
        record.status = "cancelled"
        _terminate_whatsapp_pairing(record.proc)
    return {"ok": True}
