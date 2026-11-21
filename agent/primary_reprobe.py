"""Optional mid-turn recovery when a primary route failed over to a fallback.

The turn loop checks at a configured interval whether a fallback is active. For
supported chat-completions and Anthropic-message routes with a static API key,
it sends a tiny health request. A successful response restores the primary and
re-runs the iteration so preflight uses the restored route.

This is opt-in (``fallback.reprobe_per_call: 0`` by default). Unsupported auth
or API modes fail closed: no probe is sent, and the turn remains on fallback.
"""
from __future__ import annotations

import contextlib
import logging
import time
from typing import Any

logger = logging.getLogger("agent.conversation_loop")

def _reprobe_cfg() -> tuple[int, float]:
    """Return (interval_calls, cooldown_seconds); interval 0 disables probing."""
    try:
        from hermes_cli.config import load_config

        fb = (load_config() or {}).get("fallback") or {}
        return (
            max(0, int(fb.get("reprobe_per_call", 0) or 0)),
            max(1.0, float(fb.get("reprobe_cooldown_sec", 300) or 300)),
        )
    except Exception:
        logger.debug("Could not read fallback reprobe settings; leaving the feature off", exc_info=True)
        return 0, 300.0


def _primary_probe(agent: Any, rt: dict[str, Any]) -> bool:
    """Send one small authenticated probe. Only HTTP 200 proves the route healthy.

    Deliberately limit this helper to wire modes whose basic health request is
    well-defined. Dynamic OAuth/callable credentials and other API modes are not
    guessed at; they stay on fallback until the next normal turn-start restore.
    """
    import httpx

    base = str(rt.get("base_url") or "")
    model = str(rt.get("model") or "")
    api_mode = str(rt.get("api_mode") or "chat_completions").strip().lower()
    if not base or not model or api_mode not in {"chat_completions", "anthropic_messages"}:
        return False

    headers = {"Content-Type": "application/json"}
    if api_mode == "anthropic_messages":
        api_key = rt.get("anthropic_api_key") or rt.get("api_key")
        if not isinstance(api_key, str) or not api_key.strip():
            return False
        base = str(rt.get("anthropic_base_url") or base)
        url = base.rstrip("/") + "/messages"
        headers["x-api-key"] = api_key
        headers["anthropic-version"] = "2023-06-01"
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "ping"}],
        }
    else:
        api_key = rt.get("api_key")
        if not isinstance(api_key, str) or not api_key.strip():
            return False
        url = base.rstrip("/") + "/chat/completions"
        headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": model,
            "max_tokens": 2,
            "messages": [{"role": "user", "content": "ping"}],
        }

    try:
        # Preserve configured TLS behavior for custom/self-hosted endpoints.
        client_kwargs = rt.get("client_kwargs") or {}
        from agent.ssl_verify import resolve_httpx_verify

        verify = resolve_httpx_verify(
            ca_bundle=client_kwargs.get("ssl_ca_cert"),
            ssl_verify=client_kwargs.get("ssl_verify"),
            base_url=base,
        )
        # Preserve explicit per-request headers without logging their values.
        overrides = rt.get("request_overrides") or {}
        extra_headers = overrides.get("extra_headers") if isinstance(overrides, dict) else None
        if isinstance(extra_headers, dict):
            headers.update({str(k): str(v) for k, v in extra_headers.items()})
        with httpx.Client(timeout=12.0, verify=verify) as client:
            resp = client.post(url, headers=headers, json=payload)
    except Exception:
        logger.debug("Primary re-probe request failed", exc_info=True)
        return False
    if resp.status_code == 200:
        logger.info("Primary re-probe OK: %s (%s)", model, rt.get("provider"))
        return True
    logger.debug("Primary re-probe got HTTP %s; staying on fallback", resp.status_code)
    return False


def maybe_reprobe_primary(agent: Any, api_call_count: int) -> bool:
    """Re-probe the primary periodically during a real provider failover.

    Returns True only when a healthy primary was restored mid-turn; the caller
    then re-issues the iteration so the preflight gate runs against that route.
    """
    # `_fallback_activated` is also used by `/model --once`; only a real provider
    # failover sets `_provider_fallback_active` and is eligible for recovery probes.
    if not getattr(agent, "_provider_fallback_active", False):
        return False
    if not getattr(agent, "_fallback_activated", False):
        return False
    rt = getattr(agent, "_primary_runtime", None)
    if not isinstance(rt, dict) or not rt:
        return False
    interval, cooldown = _reprobe_cfg()
    if interval <= 0:
        return False

    activated_at_call = getattr(agent, "_fallback_activated_at_call", None)
    if activated_at_call is None:
        agent._fallback_activated_at_call = api_call_count
        return False
    if (api_call_count - activated_at_call) <= 0 or (api_call_count - activated_at_call) % interval != 0:
        return False
    now = time.monotonic()
    if now - float(getattr(agent, "_last_primary_reprobe_at", 0.0)) < cooldown:
        return False
    agent._last_primary_reprobe_at = now
    if not _primary_probe(agent, rt):
        return False
    try:
        from agent.agent_runtime_helpers import restore_primary_runtime

        if restore_primary_runtime(agent):
            agent._fallback_activated_at_call = None
            logger.info(
                "Primary restored mid-turn after %d fallback calls: %s (%s)",
                api_call_count - (activated_at_call or 0), agent.model, agent.provider,
            )
            with contextlib.suppress(Exception):
                agent._emit_diagnostic_status(
                    f"✅ Primary model restored mid-turn: {agent.model} via {agent.provider}"
                )
            return True
    except Exception:
        logger.warning("Mid-turn primary restore failed; staying on fallback", exc_info=True)
    return False
