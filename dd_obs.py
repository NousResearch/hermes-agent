"""DecisionData observability client for Hermes.

Posts spawn / generation / keepalive / complete events to the OpenClaw
obs-ingest service so Hermes runs appear in the MC live agents tab.

Most HTTP calls run on a daemon thread and never raise into the caller —
instrumentation must NEVER block or fail Hermes responses. The single
exception is :func:`log_generation_sync`, used by run_agent.py's per-API-
call emit path: it MUST return a real success/failure signal so the
gateway can decide whether to skip the synthetic completion fallback
without risking a "no rows at all" outcome.

Service discovery happens once at import time via the dd-registry-service
on :8500 with a fallback to localhost:8511 (current obs-ingest port).
This must resolve to the SAME obs-ingest as gateway/run.py's
``_dd_observability_post`` (which defaults to ``127.0.0.1:8511`` and
honors HERMES_DECISIONDATA_OBSERVABILITY_URL). If those drift in
production, the per-call success → "skip synthetic" decision becomes
unsafe; both should be pinned to the canonical obs-ingest port.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Dict, Optional

try:
    import urllib.request
    import urllib.error
    import json as _json
except Exception:  # pragma: no cover
    urllib = None  # type: ignore

logger = logging.getLogger("hermes.dd_obs")

_REGISTRY_URL = os.getenv("DD_REGISTRY_URL", "http://localhost:8500")
_OBS_INGEST_BASE: Optional[str] = None
_OBS_DISCOVERY_LOCK = threading.Lock()
_HTTP_TIMEOUT = float(os.getenv("DD_OBS_HTTP_TIMEOUT", "2.0"))


def _resolve_obs_ingest_base() -> Optional[str]:
    """Resolve obs-ingest base URL via dd-registry. Cache after first success."""
    global _OBS_INGEST_BASE
    if _OBS_INGEST_BASE:
        return _OBS_INGEST_BASE
    with _OBS_DISCOVERY_LOCK:
        if _OBS_INGEST_BASE:
            return _OBS_INGEST_BASE
        # Try registry first
        try:
            req = urllib.request.Request(f"{_REGISTRY_URL}/services/discover/obs-ingest")
            with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
                data = _json.loads(resp.read().decode("utf-8"))
                base = data.get("base_url")
                if base:
                    _OBS_INGEST_BASE = base
                    return _OBS_INGEST_BASE
        except Exception:
            pass
        # Fallback to env var or localhost:8511
        env_base = os.getenv("DD_OBS_INGEST_URL")
        if env_base:
            _OBS_INGEST_BASE = env_base.rstrip("/")
            return _OBS_INGEST_BASE
        _OBS_INGEST_BASE = "http://localhost:8511"
        return _OBS_INGEST_BASE


def _post_async(path: str, payload: Dict[str, Any]) -> None:
    """Fire-and-forget POST to obs-ingest."""
    base = _resolve_obs_ingest_base()
    if not base:
        return

    def _send() -> None:
        try:
            url = f"{base}{path}"
            body = _json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url, data=body, method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
                _ = resp.read()
        except Exception as e:
            logger.debug("dd_obs POST %s failed: %s", path, e)

    t = threading.Thread(target=_send, daemon=True, name=f"dd_obs:{path.lstrip('/')}")
    t.start()


def _post_sync(path: str, payload: Dict[str, Any]) -> bool:
    """Synchronous POST to obs-ingest. Returns True on 2xx, False on any failure.

    Used by per-call generation emit where the caller must know whether
    the row landed before flipping ``per_call_generation_emitted`` (so a
    failed POST does not silently suppress the synthetic fallback row).
    """
    base = _resolve_obs_ingest_base()
    if not base:
        return False
    try:
        url = f"{base}{path}"
        body = _json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
            status = getattr(resp, "status", None) or resp.getcode()
            _ = resp.read()
            return 200 <= int(status) < 300
    except Exception as e:
        logger.debug("dd_obs sync POST %s failed: %s", path, e)
        return False


def log_spawn(
    *,
    run_id: str,
    session_key: str,
    parent_run_id: Optional[str] = None,
    task_prompt: str = "",
    label: str = "Hermes Worker",
    model: str = "",
    provider: str = "",
    channel: str = "hermes",
    spawn_context: str = "",
) -> None:
    payload = {
        "run_id": run_id,
        "parent_run_id": parent_run_id,
        "session_key": session_key,
        "task_prompt": (task_prompt or "")[:500],
        "label": label,
        "model": model,
        "provider": provider,
        "spawned_at": _iso_now(),
        "channel": channel,
        "spawn_context": spawn_context,
    }
    _post_async("/log_spawn", payload)


def log_keepalive(*, run_id: str) -> None:
    _post_async("/log_keepalive", {"run_id": run_id})


def log_complete(
    *,
    run_id: str,
    status: str = "done",
    result_summary: str = "",
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cost_usd: float = 0.0,
) -> None:
    payload = {
        "run_id": run_id,
        "status": status,
        "result_summary": (result_summary or "")[:200],
        "input_tokens": int(input_tokens or 0),
        "output_tokens": int(output_tokens or 0),
        "cache_read_tokens": int(cache_read_tokens or 0),
        "cost_usd": float(cost_usd or 0),
        "completed_at": _iso_now(),
    }
    _post_async("/log_complete", payload)


def log_generation(
    *,
    generation_id: str,
    session_id: str,
    model: str,
    requested_at: str,
    run_id: Optional[str] = None,
    provider: Optional[str] = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
    cost_total_usd: float = 0.0,
    completed_at: Optional[str] = None,
    latency_ms: Optional[int] = None,
    stop_reason: Optional[str] = None,
    cwd: Optional[str] = None,
) -> None:
    payload: Dict[str, Any] = {
        "generation_id": generation_id,
        "session_id": session_id,
        "model": model or "",
        "requested_at": requested_at,
        "input_tokens": int(input_tokens or 0),
        "output_tokens": int(output_tokens or 0),
        "cache_read_tokens": int(cache_read_tokens or 0),
        "cache_creation_tokens": int(cache_creation_tokens or 0),
        "cost_total_usd": float(cost_total_usd or 0),
        "ingest_source": "hermes-gateway",
        "agent_class": "Hermes",
    }
    if run_id:
        payload["run_id"] = run_id
    if provider:
        # provider is not a top-level GenerationRequest field; encode in cwd-adjacent
        # fields the schema accepts. The agent_class + ingest_source is enough to
        # identify Hermes; provider lives on agent_runs (set by /log_spawn).
        pass
    if completed_at:
        payload["completed_at"] = completed_at
    if latency_ms is not None:
        payload["latency_ms"] = int(latency_ms)
    if stop_reason:
        payload["stop_reason"] = stop_reason
    if cwd:
        payload["cwd"] = cwd
    _post_async("/log_generation", payload)


def log_generation_sync(
    *,
    generation_id: str,
    session_id: str,
    model: str,
    requested_at: str,
    run_id: Optional[str] = None,
    provider: Optional[str] = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
    cost_total_usd: float = 0.0,
    completed_at: Optional[str] = None,
    latency_ms: Optional[int] = None,
    stop_reason: Optional[str] = None,
    cwd: Optional[str] = None,
) -> bool:
    """Synchronous /log_generation. Returns True iff obs-ingest accepted (2xx).

    Same payload contract as :func:`log_generation`, but the caller can
    branch on the return value — used by the per-call emit path where a
    real success signal is required before suppressing the synthetic
    fallback row.
    """
    payload: Dict[str, Any] = {
        "generation_id": generation_id,
        "session_id": session_id,
        "model": model or "",
        "requested_at": requested_at,
        "input_tokens": int(input_tokens or 0),
        "output_tokens": int(output_tokens or 0),
        "cache_read_tokens": int(cache_read_tokens or 0),
        "cache_creation_tokens": int(cache_creation_tokens or 0),
        "cost_total_usd": float(cost_total_usd or 0),
        "ingest_source": "hermes-gateway",
        "agent_class": "Hermes",
    }
    if run_id:
        payload["run_id"] = run_id
    if completed_at:
        payload["completed_at"] = completed_at
    if latency_ms is not None:
        payload["latency_ms"] = int(latency_ms)
    if stop_reason:
        payload["stop_reason"] = stop_reason
    if cwd:
        payload["cwd"] = cwd
    return _post_sync("/log_generation", payload)


def _iso_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class KeepaliveLoop:
    """Periodic background keepalive — every 20 seconds while Hermes is running."""

    def __init__(self, run_id: str, interval_s: float = 20.0) -> None:
        self.run_id = run_id
        self.interval = interval_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"dd_obs:keepalive:{self.run_id[:8]}")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                log_keepalive(run_id=self.run_id)
            except Exception:
                pass
