"""
Inbound Webhook Trigger System

Exposes HTTP endpoints that let external services trigger Hermes agent turns.
Ported from OpenClaw's src/gateway/server-http.ts and src/gateway/server/hooks.ts.

Endpoints (all POST, base path configurable, default /hooks):

  POST /hooks/agent  -- trigger a full agent turn
  POST /hooks/wake   -- inject a system event (lighter weight)
  POST /hooks/<name> -- mapped paths (configured per-webhook)

Authentication:
  Authorization: Bearer <token>   (preferred)
  X-Hermes-Token: <token>         (alternative)
  Query params are REJECTED to prevent log leakage.

Request (agent):
  {
    "message":        string   (required) prompt for the agent
    "name":           string   run label
    "session_key":    string   conversation session key
    "deliver":        bool     push reply to a messaging channel
    "channel":        string   "telegram" | "discord" | "slack" | ...
    "to":             string   chat_id for delivery
    "model":          string   model override
    "idempotency_key": string  dedup key
  }

Response (immediate):
  { "ok": true, "run_id": "<uuid>" }

Agent runs async — result delivered to messaging channel, not HTTP response.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Max request body size (256 KB)
MAX_BODY_BYTES = 256 * 1024

# Auth failure rate limiting
AUTH_FAILURE_LIMIT = 10
AUTH_FAILURE_WINDOW = 300  # 5 minutes

# Idempotency dedup TTL
DEDUPE_TTL = 300  # seconds

# In-memory stores
_auth_failures: Dict[str, list] = {}     # ip -> [timestamp, ...]
_dedup_cache: Dict[str, dict] = {}       # idempotency_key -> {run_id, ts}
_running_jobs: Dict[str, asyncio.Task] = {}


def _get_webhook_token() -> Optional[str]:
    """Read the webhook token from env or config."""
    token = os.getenv("HERMES_WEBHOOK_TOKEN", "")
    if token:
        return token
    # Fall back to config.yaml
    try:
        import yaml
        hermes_home = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))
        cfg_path = hermes_home / "config.yaml"
        if cfg_path.exists():
            cfg = yaml.safe_load(cfg_path.read_text()) or {}
            return cfg.get("webhooks", {}).get("token", "")
    except Exception:
        pass
    return None


def _safe_equal(a: str, b: str) -> bool:
    """Constant-time string comparison to prevent timing attacks."""
    return hmac.compare_digest(a.encode(), b.encode())


def _check_auth(headers: Dict[str, str], client_ip: str) -> tuple[bool, str]:
    """
    Validate webhook auth token.
    Returns (authorized, error_message).
    """
    token = _get_webhook_token()
    if not token:
        return False, "Webhook token not configured (set HERMES_WEBHOOK_TOKEN)"

    # Rate limit auth failures
    now = time.time()
    failures = _auth_failures.get(client_ip, [])
    failures = [ts for ts in failures if now - ts < AUTH_FAILURE_WINDOW]
    if len(failures) >= AUTH_FAILURE_LIMIT:
        return False, f"Too many auth failures — try again in {AUTH_FAILURE_WINDOW}s"

    # Extract token from headers (never query params)
    provided = (
        headers.get("authorization", "").removeprefix("Bearer ").strip()
        or headers.get("x-hermes-token", "").strip()
    )

    if not provided or not _safe_equal(provided, token):
        failures.append(now)
        _auth_failures[client_ip] = failures
        return False, "Unauthorized"

    # Clear failures on success
    _auth_failures.pop(client_ip, None)
    return True, ""


def _check_idempotency(key: str) -> Optional[str]:
    """Return cached run_id if this key was seen recently, else None."""
    now = time.time()
    # Purge stale entries
    stale = [k for k, v in _dedup_cache.items() if now - v["ts"] > DEDUPE_TTL]
    for k in stale:
        _dedup_cache.pop(k, None)

    entry = _dedup_cache.get(key)
    if entry:
        return entry["run_id"]
    return None


def _store_idempotency(key: str, run_id: str) -> None:
    _dedup_cache[key] = {"run_id": run_id, "ts": time.time()}


async def _run_agent_turn(
    run_id: str,
    message: str,
    session_key: Optional[str] = None,
    model: Optional[str] = None,
    deliver: bool = False,
    channel: Optional[str] = None,
    to: Optional[str] = None,
    name: str = "Webhook",
) -> None:
    """
    Execute an agent turn asynchronously.
    Runs in background — HTTP response already sent before this completes.
    """
    logger.info("[webhooks] Starting agent turn run_id=%s name=%s", run_id, name)
    try:
        from run_agent import AIAgent
        from hermes_time import now as _hermes_now

        hermes_home = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))

        # Load config
        cfg = {}
        try:
            import yaml
            cfg_path = hermes_home / "config.yaml"
            if cfg_path.exists():
                cfg = yaml.safe_load(cfg_path.read_text()) or {}
        except Exception as e:
            logger.warning("[webhooks] Could not load config.yaml: %s", e)

        resolved_model = model or os.getenv("HERMES_MODEL") or "anthropic/claude-opus-4.6"
        if not model:
            model_cfg = cfg.get("model", {})
            if isinstance(model_cfg, str):
                resolved_model = model_cfg
            elif isinstance(model_cfg, dict):
                resolved_model = model_cfg.get("default", resolved_model)

        agent = AIAgent(
            model=resolved_model,
            quiet_mode=True,
            platform="webhook",
            session_id=session_key or f"webhook_{run_id}",
            disabled_toolsets=["cronjob"],
        )

        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: agent.run_conversation(message)
        )

        final_response = result.get("final_response", "")
        logger.info("[webhooks] Agent turn complete run_id=%s", run_id)

        # Deliver result to messaging channel if requested
        if deliver and channel and to and final_response:
            try:
                from tools.send_message_tool import _send_to_platform
                from gateway.config import load_gateway_config, Platform

                platform_map = {
                    "telegram": Platform.TELEGRAM,
                    "discord": Platform.DISCORD,
                    "slack": Platform.SLACK,
                    "signal": Platform.SIGNAL,
                }
                platform = platform_map.get(channel.lower())
                if platform:
                    config = load_gateway_config()
                    pconfig = config.platforms.get(platform)
                    if pconfig and pconfig.enabled:
                        await _send_to_platform(platform, pconfig, to, final_response)
                        logger.info("[webhooks] Delivered to %s:%s", channel, to)
            except Exception as e:
                logger.error("[webhooks] Delivery failed run_id=%s: %s", run_id, e)

    except Exception as e:
        logger.error("[webhooks] Agent turn failed run_id=%s: %s", run_id, e)
    finally:
        _running_jobs.pop(run_id, None)


async def handle_webhook_request(
    method: str,
    path: str,
    headers: Dict[str, str],
    body: bytes,
    client_ip: str = "unknown",
    base_path: str = "/hooks",
) -> tuple[int, Dict[str, Any]]:
    """
    Main webhook request handler. Called by the gateway HTTP server.

    Returns (http_status_code, response_dict).
    """
    # Only POST
    if method.upper() != "POST":
        return 405, {"ok": False, "error": "Method not allowed"}

    # Auth
    ok, err = _check_auth(headers, client_ip)
    if not ok:
        status = 429 if "Too many" in err else 401
        return status, {"ok": False, "error": err}

    # Body size limit
    if len(body) > MAX_BODY_BYTES:
        return 413, {"ok": False, "error": f"Request body too large (max {MAX_BODY_BYTES} bytes)"}

    # Parse body
    try:
        payload = json.loads(body) if body else {}
    except json.JSONDecodeError as e:
        return 400, {"ok": False, "error": f"Invalid JSON: {e}"}

    # Resolve sub-path
    sub_path = path.removeprefix(base_path).strip("/") or "agent"

    # /hooks/wake — lightweight system event injection
    if sub_path == "wake":
        text = payload.get("text", "Wake signal received")
        logger.info("[webhooks] Wake signal: %s", text[:100])
        return 200, {"ok": True, "mode": "now"}

    # /hooks/agent or mapped paths — full agent turn
    message = payload.get("message", "")
    if not message:
        return 400, {"ok": False, "error": "Missing required field: message"}

    # Idempotency check
    idem_key = (
        headers.get("idempotency-key")
        or headers.get("x-hermes-idempotency-key")
        or payload.get("idempotency_key", "")
    )
    if idem_key:
        cached_run_id = _check_idempotency(idem_key)
        if cached_run_id:
            logger.info("[webhooks] Idempotent request — returning cached run_id=%s", cached_run_id)
            return 200, {"ok": True, "run_id": cached_run_id, "cached": True}

    run_id = uuid.uuid4().hex

    if idem_key:
        _store_idempotency(idem_key, run_id)

    # Fire agent turn async (fire-and-forget)
    task = asyncio.create_task(_run_agent_turn(
        run_id=run_id,
        message=message,
        session_key=payload.get("session_key"),
        model=payload.get("model"),
        deliver=bool(payload.get("deliver", False)),
        channel=payload.get("channel"),
        to=payload.get("to"),
        name=payload.get("name", sub_path or "Webhook"),
    ))
    _running_jobs[run_id] = task

    logger.info("[webhooks] Dispatched agent turn run_id=%s path=/%s", run_id, sub_path)
    return 200, {"ok": True, "run_id": run_id}
