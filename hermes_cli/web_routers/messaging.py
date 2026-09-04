"""Messaging dashboard routes: WhatsApp/Telegram onboarding and per-platform enable/config/test.

Extracted from ``hermes_cli.web_server``; helpers/state that tests monkeypatch on
``web_server`` stay there and are resolved late at call time (cycle-safe).
"""

import asyncio
import contextlib
import json
import logging
import os
import re
import secrets
import subprocess
import threading
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException

from gateway.status import resolve_gateway_liveness
from hermes_cli._subprocess_compat import windows_hide_flags
from hermes_cli.config import OPTIONAL_ENV_VARS, get_env_path, redact_key
from hermes_cli.web_deps import LateState, late
from hermes_cli.web_server_gateway import _restart_gateway_after
from hermes_cli.web_server_messaging import (
    _TelegramOnboardingPairing, _WhatsAppOnboardingSession, _managed_platform_record, _managed_platforms, _messaging_platform_catalog, _refuse_if_platform_managed, _telegram_onboarding_error_message, _telegram_onboarding_lock, _telegram_onboarding_pairings, _whatsapp_onboarding_payload, _whatsapp_onboarding_sessions,
)
from hermes_cli.web_routers._common import http_failure
from hermes_cli.web_models import (
    MessagingPlatformUpdate, TelegramOnboardingApply, TelegramOnboardingStart,
    WhatsAppOnboardingApply, WhatsAppOnboardingStart,
)

_log = logging.getLogger("hermes_cli.web_server")
router = APIRouter()

# Late-bound so a test's monkeypatch on the owning module wins at call time.
_config_profile_scope = late("_config_profile_scope", "hermes_cli.web_server_profiles")
_profile_scope = late("_profile_scope", "hermes_cli.web_server_profiles")
_resolve_profile_dir = late("_resolve_profile_dir", "hermes_cli.web_server_profiles")
_restart_gateway_after_whatsapp_onboarding = late("_restart_gateway_after_whatsapp_onboarding", "hermes_cli.web_server_messaging")
_telegram_onboarding_request_sync = late("_telegram_onboarding_request_sync", "hermes_cli.web_server_messaging")
_whatsapp_session_path = late("_whatsapp_session_path", "hermes_cli.web_server_messaging")
_write_platform_enabled = late("_write_platform_enabled", "hermes_cli.web_server_messaging")
load_env = late("load_env", "hermes_cli.config")
load_config = late("load_config", "hermes_cli.config")
read_runtime_status = late("read_runtime_status", "gateway.status")
remove_env_value = late("remove_env_value", "hermes_cli.config")
save_env_value = late("save_env_value", "hermes_cli.config")
_gateway_subcommand = late("_gateway_subcommand", "hermes_cli.web_server_gateway")
_probe_gateway_health = late("_probe_gateway_health", "hermes_cli.web_server_gateway")
get_running_pid_cached = late("get_running_pid_cached", "gateway.status")
get_runtime_status_running_pid = late("get_runtime_status_running_pid", "gateway.status")
_GATEWAY_HEALTH_URL = LateState("_GATEWAY_HEALTH_URL")
# Display labels for env vars not in OPTIONAL_ENV_VARS (bridge toggles, Twilio, HASS, Email, ...)
# so the UI can still render a friendly label. Rows: (key, description, prompt, extra flags).
_MESSAGING_ENV_FALLBACKS: dict[str, dict[str, Any]] = {
    key: {"description": description, "prompt": prompt, **extra}
    for key, description, prompt, extra in (
        ("SIGNAL_HTTP_URL", "signal-cli REST API base URL, e.g. http://127.0.0.1:8080", "Signal bridge URL", {"url": "https://github.com/bbernhard/signal-cli-rest-api"}),
        ("SIGNAL_ACCOUNT", "Signal account phone number registered with the bridge", "Signal account", {}),
        ("SIGNAL_ALLOWED_USERS", "Comma-separated Signal users allowed to use the bot", "Allowed Signal users", {}),
        ("WHATSAPP_ENABLED", "Enable the WhatsApp gateway adapter", "Enable WhatsApp", {"advanced": True}),
        ("WHATSAPP_MODE", "WhatsApp bridge mode", "WhatsApp mode", {"advanced": True}),
        ("WHATSAPP_DM_POLICY", "How WhatsApp direct messages are authorized", "WhatsApp DM policy", {"advanced": True}),
        ("WHATSAPP_ALLOWED_USERS", "Comma-separated WhatsApp users allowed to use the bot", "Allowed WhatsApp users", {}),
        ("HASS_URL", "Home Assistant base URL, e.g. https://homeassistant.local:8123", "Home Assistant URL", {}),
        ("HASS_TOKEN", "Long-lived access token from Home Assistant (Profile → Security)", "Home Assistant access token", {"password": True}),
        ("EMAIL_ADDRESS", "Email address to send and receive from", "Email address", {}),
        ("EMAIL_PASSWORD", "Email account password or app password", "Email password", {"password": True}),
        ("EMAIL_IMAP_HOST", "IMAP server host (e.g. imap.gmail.com)", "IMAP host", {}),
        ("EMAIL_SMTP_HOST", "SMTP server host (e.g. smtp.gmail.com)", "SMTP host", {}),
        ("TWILIO_ACCOUNT_SID", "Twilio Account SID", "Twilio Account SID", {"url": "https://www.twilio.com/console"}),
        ("TWILIO_AUTH_TOKEN", "Twilio Auth Token", "Twilio Auth Token", {"password": True}),
        ("WECOM_BOT_ID", "WeCom group bot ID", "WeCom Bot ID", {}),
        ("WECOM_SECRET", "WeCom group bot secret", "WeCom Secret", {"password": True}),
        ("WECOM_CALLBACK_CORP_ID", "WeCom corp ID", "WeCom Corp ID", {}),
        ("WECOM_CALLBACK_CORP_SECRET", "WeCom app corp secret", "WeCom Corp Secret", {"password": True}),
        ("WECOM_CALLBACK_AGENT_ID", "WeCom app agent ID", "WeCom Agent ID", {}),
        ("WECOM_CALLBACK_TOKEN", "WeCom callback verification token", "WeCom Token", {}),
        ("WECOM_CALLBACK_ENCODING_AES_KEY", "WeCom callback AES encoding key", "WeCom AES Key", {"password": True}),
        ("WEIXIN_ACCOUNT_ID", "iLink Bot account ID obtained through QR login in hermes gateway setup", "iLink Bot account ID", {}),
        ("WEIXIN_TOKEN", "iLink Bot token obtained through QR login in hermes gateway setup", "iLink Bot token", {"password": True}),
        ("WEIXIN_BASE_URL", "iLink API base URL saved by QR login (default: https://ilinkai.weixin.qq.com)", "iLink API base URL", {}),
        ("FEISHU_APP_ID", "Feishu / Lark app ID", "App ID", {}),
        ("FEISHU_APP_SECRET", "Feishu / Lark app secret", "App secret", {"password": True}),
        ("FEISHU_ENCRYPT_KEY", "Feishu / Lark encrypt key", "Encrypt key", {"password": True}),
        ("FEISHU_VERIFICATION_TOKEN", "Feishu / Lark verification token", "Verification token", {"password": True}),
        ("DINGTALK_CLIENT_ID", "DingTalk client ID (App key)", "Client ID", {}),
        ("DINGTALK_CLIENT_SECRET", "DingTalk client secret (App secret)", "Client secret", {"password": True}),
    )
}


# Kept in sync with the corresponding frontend validation in ChannelsPage.tsx.
_TELEGRAM_BOT_TOKEN_RE = re.compile(r"\d+:[A-Za-z0-9_-]{30,}")
_TELEGRAM_USER_ID_RE = re.compile(r"\d+")
_SLACK_MEMBER_ID_RE = re.compile(r"[UW][A-Z0-9]{2,}")


def _csv_ids(value: str) -> list[str]:
    """Split like the gateway parsers (gateway/platforms/*.py): comma, strip, drop
    empties — so a trailing/interior comma isn't rejected when the runtime accepts it."""
    return [part.strip() for part in value.split(",") if part.strip()]


# (platform, env key) -> (accepts(value), 400 detail). Rejects credentials that
# are clearly in the wrong field. "*" is Slack's allow-all wildcard.
_ENV_VALUE_RULES: dict[tuple[str, str], tuple[Any, str]] = {
    ("telegram", "TELEGRAM_BOT_TOKEN"): (
        _TELEGRAM_BOT_TOKEN_RE.fullmatch,
        "Telegram bot token must be the complete token from @BotFather, such as 123456789:ABC…"),
    ("telegram", "TELEGRAM_ALLOWED_USERS"): (
        lambda v: all(_TELEGRAM_USER_ID_RE.fullmatch(u) for u in _csv_ids(v)),
        "Telegram allowed users must be comma-separated numeric user IDs."),
    ("slack", "SLACK_BOT_TOKEN"): (
        lambda v: v.startswith("xoxb-"),
        "Slack Bot Token must start with xoxb-. Paste the bot token from OAuth & Permissions."),
    ("slack", "SLACK_APP_TOKEN"): (
        lambda v: v.startswith("xapp-"),
        "Slack App Token must start with xapp-. Paste the app-level token from Basic Information > App-Level Tokens."),
    ("slack", "SLACK_ALLOWED_USERS"): (
        lambda v: all(u == "*" or _SLACK_MEMBER_ID_RE.fullmatch(u) for u in _csv_ids(v)),
        "Slack allowed user IDs must be comma-separated member IDs like U01ABC2DEF3."),
}


def _validate_messaging_env_value(platform_id: str, key: str, value: str) -> None:
    rule = _ENV_VALUE_RULES.get((platform_id, key))
    if value and rule and not rule[0](value):
        raise HTTPException(status_code=400, detail=rule[1])


def _messaging_env_info(key: str) -> dict[str, Any]:
    info = OPTIONAL_ENV_VARS.get(key) or _MESSAGING_ENV_FALLBACKS.get(key) or {}
    return {
        "description": info.get("description", ""),
        "prompt": info.get("prompt", key),
        "help": info.get("help", ""),
        "url": info.get("url"),
        "is_password": info.get("password", False),
        "advanced": info.get("advanced", False),
    }


def _catalog_lookup(platform_id: str) -> dict[str, Any] | None:
    return next((e for e in _messaging_platform_catalog() if e["id"] == platform_id), None)


def _require_platform(platform_id: str) -> dict[str, Any]:
    entry = _catalog_lookup(platform_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Unknown messaging platform: {platform_id}")
    return entry


def _platform_enablement(
    platform_id: str, entry: dict[str, Any], env_on_disk: dict[str, str], scoped: bool
) -> tuple[bool, bool, dict | None]:
    """(enabled, configured, home_channel). Profile-scoped: derive from the profile's
    config.yaml + .env only — load_gateway_config()'s env-override layer reads
    os.environ and would leak the root install's tokens into the profile's state."""
    required = entry["required_env"]
    if scoped:
        try:
            plat_cfg = (load_config().get("platforms") or {}).get(platform_id)
            plat_cfg = plat_cfg if isinstance(plat_cfg, dict) else {}
            hc = plat_cfg.get("home_channel")
            enabled, home_channel = bool(plat_cfg.get("enabled")), (hc if isinstance(hc, dict) else None)
        except Exception:
            enabled, home_channel = False, None
        return enabled, all(env_on_disk.get(key) for key in required), home_channel
    try:
        from gateway.config import Platform, load_gateway_config

        gateway_config = load_gateway_config()
        platform = Platform(platform_id)
        platform_config = gateway_config.platforms.get(platform)
        enabled = bool(platform_config and platform_config.enabled)
        configured = bool(platform_config and gateway_config._is_platform_connected(platform, platform_config))
        home_channel = platform_config.home_channel.to_dict() if platform_config and platform_config.home_channel else None
    except Exception:
        enabled, home_channel = False, None
        configured = all(env_on_disk.get(key) or os.getenv(key, "") for key in required)
    return enabled, configured, home_channel


def _messaging_platform_payload(
    entry: dict[str, Any], env_on_disk: dict[str, str], runtime: dict | None,
    scoped: bool = False, profile_home: Optional[Path] = None,
) -> dict[str, Any]:
    platform_id = entry["id"]
    rt = runtime if isinstance(runtime, dict) else {}
    runtime_platforms = rt.get("platforms")
    runtime_platform = runtime_platforms.get(platform_id, {}) if isinstance(runtime_platforms, dict) else {}
    if not isinstance(runtime_platform, dict):
        runtime_platform = {}
    # Same shared liveness ladder /api/status uses, so the sidebar strip and the
    # Channels page can never disagree on one page load. profile_home is passed when
    # scoped to a named profile: gateway/status readers resolve process-level paths
    # and do NOT follow the HERMES_HOME contextvar override, so without it messaging
    # silently reports another profile's gateway.
    gateway_running = resolve_gateway_liveness(
        profile_dir=profile_home, runtime=runtime,
        health_probe=_probe_gateway_health if _GATEWAY_HEALTH_URL else None,
        pid_probe=get_running_pid_cached, runtime_reader=read_runtime_status,
        runtime_pid_probe=get_runtime_status_running_pid,
    ).running

    def env_value(key: str) -> str:
        # Profile-scoped: judge only the profile's own .env — the dashboard process's
        # os.environ carries the ROOT install's .env and would report root credentials as the profile's.
        return env_on_disk.get(key) or ("" if scoped else os.getenv(key, ""))

    env_vars = [
        {
            "key": key, "required": key in entry["required_env"], "is_set": bool(value),
            "redacted_value": redact_key(value) if value else None, **_messaging_env_info(key),
        }
        for key, value in ((key, env_value(key)) for key in entry["env_vars"])
    ]

    enabled, configured, home_channel = _platform_enablement(platform_id, entry, env_on_disk, scoped)

    state = runtime_platform.get("state")
    if not enabled:
        state = "disabled"
    elif not configured:
        state = "not_configured"
    elif gateway_running and not state:
        state = "pending_restart"
    elif not gateway_running and not state:
        state = "startup_failed" if rt.get("gateway_state") == "startup_failed" else "gateway_stopped"

    error_code = runtime_platform.get("error_code")
    error_message = runtime_platform.get("error_message")
    if state == "startup_failed":
        error_code = error_code or "startup_failed"
        error_message = error_message or rt.get("exit_reason")

    payload = {
        "id": platform_id, "name": entry["name"], "description": entry["description"],
        "docs_url": entry["docs_url"], "enabled": enabled, "configured": configured,
        "gateway_running": gateway_running, "state": state, "error_code": error_code,
        "error_message": error_message, "updated_at": runtime_platform.get("updated_at"),
        "home_channel": home_channel, "env_vars": env_vars,
    }
    if platform_id == "whatsapp":
        whatsapp_mode = env_value("WHATSAPP_MODE").strip()
        payload["whatsapp_setup"] = {
            "mode": whatsapp_mode if whatsapp_mode in {"bot", "self-chat"} else "",
            "allowed_users_set": bool(env_value("WHATSAPP_ALLOWED_USERS").strip()),
            "home_channel_set": bool(home_channel),
        }
    managed = _managed_platforms()
    if managed:
        payload["managed"] = managed.record_for(platform_id)
    return payload


def _platform_payloads(scoped_dir: Optional[Path], entries) -> list[dict[str, Any]]:
    """Payloads for ``entries``; call inside ``_profile_scope`` (load_env honors the
    HERMES_HOME contextvar; the gateway status readers do not, hence the explicit path)."""
    env_on_disk = load_env()
    runtime = read_runtime_status(path=scoped_dir / "gateway_state.json") if scoped_dir is not None else read_runtime_status()
    return [_messaging_platform_payload(entry, env_on_disk, runtime, scoped=scoped_dir is not None, profile_home=scoped_dir)
            for entry in entries]


@contextlib.contextmanager
def _onboarding_save_errors(log_msg: str, detail: str):
    """Map onboarding env/config write failures: ValueError -> 400 (its text),
    anything else -> logged + fixed 500 ``detail``; HTTPException passes through."""
    try:
        yield
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        _log.exception(log_msg)
        raise HTTPException(status_code=500, detail=detail) from exc


# ── WhatsApp QR onboarding ─────────────────────────────────────

_WHATSAPP_ONBOARDING_TTL_SECONDS = 600
_WHATSAPP_ONBOARDING_TERMINAL_STATUSES = {"connected", "error", "expired", "cancelled"}
_WHATSAPP_SESSION_NOT_FOUND = "WhatsApp setup session was not found. Start a new setup."
_whatsapp_onboarding_lock = threading.RLock()


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


def _whatsapp_phone_from_identifier(value: Any) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    digits = re.sub(r"\D+", "", raw.split("@", 1)[0].split(":", 1)[0])
    return digits or None


def _first_str(candidate: Any, keys: tuple[str, ...]) -> str | None:
    if not isinstance(candidate, dict):
        return None
    return next((v for v in (str(candidate.get(k) or "").strip() for k in keys) if v), None)


def _whatsapp_linked_account_from_session(session_path: Path) -> tuple[str | None, str | None, str | None]:
    try:
        payload = json.loads((session_path / "creds.json").read_text(encoding="utf-8"))
    except Exception:
        return None, None, None
    candidates = (payload.get("me"), payload.get("account"), payload)
    account_id = next((v for v in (_first_str(c, ("id", "jid", "lid")) for c in candidates) if v), None)
    account_name = next((v for v in (_first_str(c, ("name", "verifiedName", "notify", "pushName")) for c in candidates) if v), None)
    return account_id, account_name, _whatsapp_phone_from_identifier(account_id)


def _ensure_whatsapp_bridge_dependencies(bridge_dir: Path) -> None:
    """Install bridge dependencies when the dashboard is the setup surface."""
    if (bridge_dir / "node_modules").exists():
        return

    from hermes_constants import find_node_executable, with_hermes_node_path
    from utils import env_int

    npm = find_node_executable("npm")
    if not npm:
        raise HTTPException(status_code=500, detail="npm was not found. WhatsApp setup needs Node.js and npm.")

    try:
        # npm output is UTF-8; encoding= guards the Windows ANSI-code-page
        # default against undefined bytes crashing the reader thread.
        result = subprocess.run(
            [npm, "install", "--silent"], cwd=str(bridge_dir), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=env_int("WHATSAPP_NPM_INSTALL_TIMEOUT", 300),
            env=with_hermes_node_path(), creationflags=windows_hide_flags(),
        )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=500, detail="Installing WhatsApp bridge dependencies timed out.") from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to install WhatsApp bridge dependencies: {exc}") from exc

    if result.returncode != 0:
        detail = "\n".join((result.stderr or result.stdout or "").strip().splitlines()[-10:])
        raise HTTPException(status_code=500, detail=f"npm install failed for WhatsApp bridge: {detail or 'no output'}")


def _spawn_whatsapp_pairing_process(session_path: Path, mode: str) -> subprocess.Popen:
    from gateway.platforms.whatsapp_common import resolve_whatsapp_bridge_dir
    from hermes_constants import find_node_executable, with_hermes_node_path

    bridge_dir = resolve_whatsapp_bridge_dir()
    bridge_script = bridge_dir / "bridge.js"
    if not bridge_script.exists():
        raise HTTPException(status_code=500, detail=f"WhatsApp bridge script was not found at {bridge_script}.")
    node = find_node_executable("node")
    if not node:
        raise HTTPException(status_code=500, detail="Node.js was not found. WhatsApp setup needs Node.js.")

    _ensure_whatsapp_bridge_dependencies(bridge_dir)
    session_path.mkdir(parents=True, exist_ok=True)

    env = with_hermes_node_path()
    env["WHATSAPP_MODE"] = mode
    env["WHATSAPP_DM_POLICY"] = "pairing"
    return subprocess.Popen(
        [node, str(bridge_script), "--pair-only", "--pair-json", "--session", str(session_path)],
        cwd=str(bridge_dir), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="utf-8", errors="replace", start_new_session=True, env=env,
        creationflags=windows_hide_flags(),
    )


def _terminate_whatsapp_pairing(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=3)
    except Exception:
        with contextlib.suppress(Exception):
            proc.kill()


def _fail_whatsapp_pairing(pairing_id: str, error: str, *, proc=None, unless=_WHATSAPP_ONBOARDING_TERMINAL_STATUSES) -> None:
    """Mark the session errored unless its status is in ``unless`` (or, when
    ``proc`` is given, it has since been superseded by another process)."""
    with _whatsapp_onboarding_lock:
        record = _whatsapp_onboarding_sessions.get(pairing_id)
        if record and (proc is None or record.proc is proc) and record.status not in unless:
            record.status = "error"
            record.error = error


def _apply_pairing_event(record, payload: dict) -> None:
    event = str(payload.get("event") or "").strip()
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
            record.account_id = account_id or None
            record.account_name = str(user.get("name") or "").strip() or None
            record.account_phone = _whatsapp_phone_from_identifier(account_id)
        record.status = "connected"
        record.error = None
    elif event == "error":
        record.status = "error"
        record.error = str(payload.get("error") or "WhatsApp pairing failed.")
    elif event == "disconnected" and record.status == "starting":
        record.status = "waiting"


def _watch_whatsapp_pairing(pairing_id: str, proc: subprocess.Popen) -> None:
    try:
        for line in proc.stdout or ():
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            with _whatsapp_onboarding_lock:
                record = _whatsapp_onboarding_sessions.get(pairing_id)
                if not record or record.proc is not proc:
                    return
                _apply_pairing_event(record, payload)
        returncode = proc.wait()
    except Exception as exc:
        _fail_whatsapp_pairing(pairing_id, str(exc), proc=proc)
        return
    # An "error" status from the stream may be overwritten by the exit reason.
    _fail_whatsapp_pairing(
        pairing_id, "WhatsApp pairing process exited before pairing completed." if returncode == 0
        else f"WhatsApp pairing process exited with code {returncode}.",
        proc=proc, unless={"connected", "cancelled", "expired"})


def _run_whatsapp_pairing(pairing_id: str, session_path: Path, mode: str) -> None:
    with _whatsapp_onboarding_lock:
        record = _whatsapp_onboarding_sessions.get(pairing_id)
        if not record or record.status in _WHATSAPP_ONBOARDING_TERMINAL_STATUSES:
            return
        record.status = "installing"

    try:
        proc = _spawn_whatsapp_pairing_process(session_path, mode)
    except Exception as exc:
        _fail_whatsapp_pairing(pairing_id, str(exc))
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
        live = record.status not in _WHATSAPP_ONBOARDING_TERMINAL_STATUSES
        if live and record.proc is not None and record.proc.poll() is not None:
            record.status = "error"
            record.error = "WhatsApp pairing process exited before pairing completed."
            live = False
        if live and record.expires_at_ts <= now:
            _terminate_whatsapp_pairing(record.proc)
            record.status = "expired"
            record.error = "WhatsApp QR setup expired. Start a new setup."
        if record.status in _WHATSAPP_ONBOARDING_TERMINAL_STATUSES and record.expires_at_ts + 300 <= now:
            remove_ids.append(pairing_id)
    for pairing_id in remove_ids:
        _whatsapp_onboarding_sessions.pop(pairing_id, None)


def _register_whatsapp_session(session_path: Path, record) -> str:
    """Store ``record`` under a fresh pairing id, cancelling any live session on
    the same session dir (superseded by the newer setup)."""
    pairing_id = secrets.token_urlsafe(16)
    with _whatsapp_onboarding_lock:
        _prune_whatsapp_onboarding_sessions()
        for existing in _whatsapp_onboarding_sessions.values():
            if existing.session_path == str(session_path) and existing.status not in _WHATSAPP_ONBOARDING_TERMINAL_STATUSES:
                existing.status = "cancelled"
                existing.error = "Superseded by a newer WhatsApp setup session."
                _terminate_whatsapp_pairing(existing.proc)
        _whatsapp_onboarding_sessions[pairing_id] = record
    return pairing_id


@router.post("/api/messaging/whatsapp/onboarding/start")
async def start_whatsapp_onboarding(body: WhatsAppOnboardingStart):
    _refuse_if_platform_managed("whatsapp")
    mode = _normalize_whatsapp_onboarding_mode(body.mode)
    allowed_users = _normalize_whatsapp_allowed_users(body.allowed_users)

    with _config_profile_scope(body.profile):
        session_path = _whatsapp_session_path()
        expires_at_ts = time.time() + _WHATSAPP_ONBOARDING_TTL_SECONDS
        fields = dict(
            proc=None, mode=mode, allowed_users=allowed_users, session_path=str(session_path),
            expires_at=datetime.fromtimestamp(expires_at_ts, timezone.utc).isoformat().replace("+00:00", "Z"),
            expires_at_ts=expires_at_ts, profile=body.profile,
        )
        already_linked = (session_path / "creds.json").exists()
        if already_linked:  # creds on disk: report connected without pairing
            account_id, account_name, account_phone = _whatsapp_linked_account_from_session(session_path)
            fields.update(status="connected", account_id=account_id, account_name=account_name, account_phone=account_phone)

    record = _WhatsAppOnboardingSession(**fields)
    pairing_id = _register_whatsapp_session(session_path, record)
    if not already_linked:
        threading.Thread(target=_run_whatsapp_pairing, args=(pairing_id, session_path, mode), daemon=True).start()
    return _whatsapp_onboarding_payload(pairing_id, record)


def _whatsapp_record_or_404(pairing_id: str):
    """Call with ``_whatsapp_onboarding_lock`` held."""
    _prune_whatsapp_onboarding_sessions()
    record = _whatsapp_onboarding_sessions.get(pairing_id)
    if not record:
        raise HTTPException(status_code=404, detail=_WHATSAPP_SESSION_NOT_FOUND)
    return record


@router.get("/api/messaging/whatsapp/onboarding/{pairing_id}")
async def get_whatsapp_onboarding_status(pairing_id: str):
    with _whatsapp_onboarding_lock:
        record = _whatsapp_record_or_404(pairing_id)
        if record.status == "expired":
            raise HTTPException(status_code=410, detail=record.error or "WhatsApp setup expired.")
        return _whatsapp_onboarding_payload(pairing_id, record)


@router.post("/api/messaging/whatsapp/onboarding/{pairing_id}/apply")
async def apply_whatsapp_onboarding(pairing_id: str, body: WhatsAppOnboardingApply, profile: Optional[str] = None):
    _refuse_if_platform_managed("whatsapp")
    with _whatsapp_onboarding_lock:
        record = _whatsapp_record_or_404(pairing_id)
        if record.status != "connected":
            raise HTTPException(status_code=409, detail="WhatsApp setup is not connected yet.")
        mode = _normalize_whatsapp_onboarding_mode(body.mode or record.mode)
        allowed_users = _normalize_whatsapp_allowed_users(record.allowed_users if body.allowed_users is None else body.allowed_users)
        if mode == "self-chat" and not allowed_users:
            allowed_users = record.account_phone or record.account_id or ""
        record_profile = record.profile

    effective_profile = body.profile or profile or record_profile
    with _onboarding_save_errors("WhatsApp onboarding apply failed", "Failed to save WhatsApp setup."):
        with _config_profile_scope(effective_profile):
            save_env_value("WHATSAPP_MODE", mode)
            save_env_value("WHATSAPP_DM_POLICY", "pairing")
            # Blank means "keep the existing allowlist"; explicit clearing
            # still lives in the normal config editor where the field is visible.
            if allowed_users:
                save_env_value("WHATSAPP_ALLOWED_USERS", allowed_users)
            save_env_value("WHATSAPP_ENABLED", "true")
            _write_platform_enabled("whatsapp", True)

    with _whatsapp_onboarding_lock:
        _whatsapp_onboarding_sessions.pop(pairing_id, None)

    restart_result = _restart_gateway_after_whatsapp_onboarding(effective_profile)
    return {"ok": True, "platform": "whatsapp", "needs_restart": not restart_result["restart_started"], **restart_result}


@router.delete("/api/messaging/whatsapp/onboarding/{pairing_id}")
async def cancel_whatsapp_onboarding(pairing_id: str):
    with _whatsapp_onboarding_lock:
        record = _whatsapp_onboarding_sessions.pop(pairing_id, None)
    if record:
        record.status = "cancelled"
        _terminate_whatsapp_pairing(record.proc)
    return {"ok": True}


# ── Telegram QR onboarding ─────────────────────────────────────

_TELEGRAM_SESSION_NOT_FOUND = "Telegram setup session was not found. Start a new setup."
_TELEGRAM_INCOMPLETE_RESPONSE = "Telegram setup service returned an incomplete response."


def _parse_expiry_ts(value: str) -> float:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except Exception:
        return time.time() + 600


def _prune_telegram_onboarding_pairings() -> None:
    now = time.time()
    for pairing_id in [pid for pid, record in _telegram_onboarding_pairings.items() if record.expires_at_ts <= now]:
        _telegram_onboarding_pairings.pop(pairing_id, None)


def _normalize_telegram_user_id(value: Any) -> str | None:
    normalized = str(value or "").strip()
    return normalized if _TELEGRAM_USER_ID_RE.fullmatch(normalized) else None


def _telegram_record_or_404(pairing_id: str):
    """Call with ``_telegram_onboarding_lock`` held."""
    _prune_telegram_onboarding_pairings()
    record = _telegram_onboarding_pairings.get(pairing_id)
    if not record:
        raise HTTPException(status_code=404, detail=_TELEGRAM_SESSION_NOT_FOUND)
    return record


def _telegram_ready_payload(record) -> dict[str, Any]:
    return {
        "status": "ready", "bot_username": record.bot_username,
        "owner_user_id": record.owner_user_id, "expires_at": record.expires_at,
    }


async def _telegram_onboarding_request(method: str, path: str, *, body=None, bearer_token=None) -> dict[str, Any]:
    return await asyncio.to_thread(_telegram_onboarding_request_sync, method, path, body=body, bearer_token=bearer_token)


@router.post("/api/messaging/telegram/onboarding/start")
async def start_telegram_onboarding(body: TelegramOnboardingStart):
    _refuse_if_platform_managed("telegram")
    bot_name = (body.bot_name or "Hermes Agent").strip() or "Hermes Agent"
    payload = await _telegram_onboarding_request("POST", "/v1/telegram/pairings", body={"bot_name": bot_name})

    def field(key: str) -> str:
        return str(payload.get(key) or "").strip()

    pairing_id, poll_token, expires_at, deep_link = map(field, ("pairing_id", "poll_token", "expires_at", "deep_link"))
    if not pairing_id or not poll_token or not expires_at or not deep_link:
        raise HTTPException(status_code=502, detail=_TELEGRAM_INCOMPLETE_RESPONSE)

    with _telegram_onboarding_lock:
        _prune_telegram_onboarding_pairings()
        _telegram_onboarding_pairings[pairing_id] = _TelegramOnboardingPairing(
            poll_token=poll_token, expires_at=expires_at, expires_at_ts=_parse_expiry_ts(expires_at))

    return {
        "pairing_id": pairing_id, "suggested_username": field("suggested_username"), "deep_link": deep_link,
        "qr_payload": str(payload.get("qr_payload") or deep_link).strip(), "expires_at": expires_at,
    }


@router.get("/api/messaging/telegram/onboarding/{pairing_id}")
async def get_telegram_onboarding_status(pairing_id: str):
    with _telegram_onboarding_lock:
        record = _telegram_record_or_404(pairing_id)
        if record.bot_token:
            return _telegram_ready_payload(record)
        poll_token = record.poll_token

    payload = await _telegram_onboarding_request(
        "GET", f"/v1/telegram/pairings/{urllib.parse.quote(pairing_id, safe='')}", bearer_token=poll_token)
    status = str(payload.get("status") or "").strip()
    if status == "waiting":
        with _telegram_onboarding_lock:
            current = _telegram_onboarding_pairings.get(pairing_id)
            expires_at = current.expires_at if current else ""
        return {"status": "waiting", "expires_at": expires_at}

    if status == "ready":
        bot_token = str(payload.get("token") or "").strip()
        if not bot_token:
            raise HTTPException(status_code=502, detail=_TELEGRAM_INCOMPLETE_RESPONSE)
        with _telegram_onboarding_lock:
            record = _telegram_onboarding_pairings.get(pairing_id)
            if not record:
                raise HTTPException(status_code=404, detail=_TELEGRAM_SESSION_NOT_FOUND)
            record.bot_token = bot_token
            record.bot_username = str(payload.get("bot_username") or "").strip() or None
            record.owner_user_id = _normalize_telegram_user_id(payload.get("owner_user_id"))
            return _telegram_ready_payload(record)

    if status in {"expired", "claimed"}:
        with _telegram_onboarding_lock:
            _telegram_onboarding_pairings.pop(pairing_id, None)
        raise HTTPException(status_code=410, detail=_telegram_onboarding_error_message(
            status, "Telegram setup is no longer available. Start a new setup."))

    raise HTTPException(status_code=502, detail="Telegram setup service returned an unknown status.")


@router.post("/api/messaging/telegram/onboarding/{pairing_id}/apply")
async def apply_telegram_onboarding(pairing_id: str, body: TelegramOnboardingApply, profile: Optional[str] = None):
    _refuse_if_platform_managed("telegram")
    normalized_ids = [_normalize_telegram_user_id(raw_id) for raw_id in body.allowed_user_ids]
    if not all(normalized_ids):
        raise HTTPException(status_code=400, detail="Allowed Telegram user IDs must be numeric.")
    allowed_user_ids = list(dict.fromkeys(normalized_ids))
    if not allowed_user_ids:
        raise HTTPException(status_code=400, detail="Add at least one allowed Telegram user ID.")

    with _telegram_onboarding_lock:
        record = _telegram_record_or_404(pairing_id)
        bot_token = record.bot_token
        bot_username = record.bot_username
        if not bot_token:
            raise HTTPException(status_code=409, detail="Telegram setup is not ready yet.")

    effective_profile = body.profile or profile

    def _apply():
        with _profile_scope(effective_profile):
            save_env_value("TELEGRAM_BOT_TOKEN", bot_token)
            save_env_value("TELEGRAM_ALLOWED_USERS", ",".join(allowed_user_ids))
            _write_platform_enabled("telegram", True)

    with _onboarding_save_errors("Telegram onboarding apply failed", "Failed to save Telegram setup."):
        await asyncio.to_thread(_apply)

    with _telegram_onboarding_lock:
        _telegram_onboarding_pairings.pop(pairing_id, None)

    # Best-effort restart: the QR flow pulls users into Telegram on another device, so a
    # saved token waiting on a manual restart click reads as "Hermes is broken" from the
    # chat side. The save stays authoritative; a failed restart is reported for the UI banner.
    restart_result = _restart_gateway_after(effective_profile, what="Telegram onboarding", label="Telegram onboarding")
    return {
        "ok": True, "platform": "telegram", "bot_username": bot_username,
        "needs_restart": not restart_result["restart_started"], **restart_result,
    }


@router.delete("/api/messaging/telegram/onboarding/{pairing_id}")
async def cancel_telegram_onboarding(pairing_id: str):
    with _telegram_onboarding_lock:
        _telegram_onboarding_pairings.pop(pairing_id, None)
    return {"ok": True}


# ── platform list / update / test ──────────────────────────────


@router.get("/api/messaging/platforms")
async def get_messaging_platforms(profile: Optional[str] = None):
    # Profile-scoped so the global profile switcher shows the TARGET profile's channel state.
    def _run():
        # Profile-scoped so the dashboard's global profile switcher shows the TARGET profile's channel
        # credentials/state, not the root install's. load_env() honors the HERMES_HOME contextvar override;
        # the gateway status readers do NOT (they resolve process-level paths), so the profile directory is
        # passed explicitly for those (#71211).
        with _profile_scope(profile) as scoped_dir:
            result = {
                "env_path": str(get_env_path()),
                "gateway_start_command": " ".join(["hermes", *_gateway_subcommand(profile, "start")]),
                "platforms": _platform_payloads(scoped_dir, _messaging_platform_catalog()),
            }
            managed = _managed_platforms()
            if managed:
                result["managed_by"] = {"label": managed.label, "url": managed.url}
            return result

    return await asyncio.to_thread(_run)


def _multiplex_port_binding_conflict(platform_id: str, requested_profile: Optional[str]) -> Optional[str]:
    """Reason enabling ``platform_id`` on the target profile would break a
    multiplexed gateway, or ``None`` when allowed.

    Mirrors ``_start_one_profile_adapters`` (gateway/run.py): with
    ``gateway.multiplex_profiles`` on, the default profile owns the single shared
    HTTP listener (``/p/<profile>/``), so a SECONDARY profile must never enable a
    port-binding platform or the shared gateway dies with ``MultiplexConfigError``
    for ALL profiles. Only *enabling* is blocked; disabling/clearing stays allowed
    so users can repair an invalid profile.
    """
    from gateway.config import PORT_BINDING_PLATFORM_VALUES, load_gateway_config

    if platform_id not in PORT_BINDING_PLATFORM_VALUES:
        return None

    requested = (requested_profile or "").strip()
    if not requested or requested.lower() == "current":
        from hermes_cli.profiles import get_active_profile_name

        # The dashboard's own profile. "custom" (unrecognized HERMES_HOME) is outside
        # the profiles tree, so a multiplexed gateway never serves it.
        target = get_active_profile_name()
    else:
        _resolve_profile_dir(requested)  # same 400/404 as _profile_scope
        target = requested
    if target in ("default", "custom"):
        return None

    # The flag that matters is the one the shared gateway reads at startup: the DEFAULT
    # profile's config (plus the process-wide GATEWAY_MULTIPLEX_PROFILES override).
    with _config_profile_scope("default"):
        if not load_gateway_config().multiplex_profiles:
            return None

    return (
        f"Cannot enable '{platform_id}' on profile '{target}': it binds its own listener port, "
        "and gateway.multiplex_profiles is on, so the default profile owns the single shared HTTP "
        "listener for every profile. Configure this channel on the default profile instead "
        "(disabling or clearing it here is still allowed)."
    )


@router.put("/api/messaging/platforms/{platform_id}")
async def update_messaging_platform(platform_id: str, body: MessagingPlatformUpdate, profile: Optional[str] = None):
    entry = _require_platform(platform_id)
    _refuse_if_platform_managed(platform_id)

    target_profile = body.profile or profile
    if body.enabled:
        conflict = _multiplex_port_binding_conflict(platform_id, target_profile)
        if conflict:
            # Reject BEFORE any .env/config.yaml write so the profile stays
            # loadable by the multiplexed gateway.
            _log.info(
                "Rejected messaging platform update: platform=%s profile=%s "
                "(multiplex port-binding conflict)", platform_id, target_profile or "current",
            )
            raise HTTPException(status_code=409, detail=conflict)

    allowed_env = set(entry["env_vars"])

    def _check_allowed(key: str) -> None:
        if key not in allowed_env:
            raise HTTPException(status_code=400, detail=f"{key} is not configurable for {entry['name']}")

    def _apply():
        with _profile_scope(target_profile):
            for key in body.clear_env:
                _check_allowed(key)
                remove_env_value(key)

            for key, value in body.env.items():
                _check_allowed(key)
                trimmed = value.strip()
                if trimmed:
                    _validate_messaging_env_value(platform_id, key, trimmed)
                    save_env_value(key, trimmed)

            if body.enabled is not None:
                _write_platform_enabled(platform_id, body.enabled)

    with http_failure(f"PUT /api/messaging/platforms/{platform_id} failed", 500, detail="Internal server error"):
        await asyncio.to_thread(_apply)

        # Audit trail for channel config mutations: names only, never values.
        _log.info(
            "Messaging platform updated: platform=%s profile=%s enabled=%s "
            "env_keys=%s cleared_keys=%s",
            platform_id, target_profile or "current", body.enabled, sorted(body.env), sorted(body.clear_env),
        )
        return {"ok": True, "platform": platform_id}


@router.post("/api/messaging/platforms/{platform_id}/test")
async def test_messaging_platform(platform_id: str, profile: Optional[str] = None):
    entry = _require_platform(platform_id)
    record = _managed_platform_record(platform_id)
    if record is not None and record["kind"] == "relay":
        raise HTTPException(
            status_code=409,
            detail=f"{entry['name']} runs through {record['label']}; the native connection test does not apply.",
        )

    def _run():
        with _profile_scope(profile) as scoped_dir:
            return _platform_payloads(scoped_dir, [entry])[0]

    payload = await asyncio.to_thread(_run)

    def result(ok: bool, message: str) -> dict[str, Any]:
        return {"ok": ok, "state": payload["state"], "message": message}

    if not payload["enabled"]:
        return result(False, f"{entry['name']} is disabled. Enable it, then restart the gateway.")
    if not payload["configured"]:
        missing = [field["key"] for field in payload["env_vars"] if field["required"] and not field["is_set"]]
        return result(False, f"Missing required setup: {', '.join(missing)}" if missing else "Platform setup is incomplete.")
    if not payload["gateway_running"]:
        return result(False, "Gateway is not running. Restart the gateway to connect this platform.")
    if payload["state"] == "connected":
        return result(True, f"{entry['name']} is connected.")
    if payload.get("error_message"):
        return result(False, payload["error_message"])
    return result(False, "Setup looks complete, but the gateway has not reported a connection yet. Restart the gateway.")
