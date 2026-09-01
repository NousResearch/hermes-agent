"""Live model-load progress from the managed llama-server router.

llama-server's child processes emit per-tensor load progress
({stages, current, value}, throttled upstream to ~200ms) which the
router relays ONLY over its /models/sse stream — GET /models carries
just the coarse status string. This module owns one lazy background
watcher on that stream and keeps an in-memory snapshot other code can
poll cheaply:

    get_loading_progress() -> {model_id: {"stage", "value", "percent"}}

"percent" is a composite across stages so a bar doesn't sprint 0->100
once per stage: the text model dominates load time (its weights dwarf
the mmproj/spec extras), so it gets the lion's share of the range and
the extras split the remainder.

The watcher starts on first call, reconnects with backoff (the router
bounces on model download/eject), and never raises into callers — no
router, no state file, or no SSE support (older engines) all read as
"nothing loading". Safe from any process on the machine: the endpoint
comes from the supervisor's machine-scoped state file.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request

logger = logging.getLogger(__name__)

_TEXT_STAGE_SHARE = 0.85     # composite range share for the text model
_RECONNECT_DELAY_S = 3.0
_STALE_ENTRY_TTL_S = 120.0   # a loading entry with no events this long is dead

_lock = threading.Lock()
_watcher: threading.Thread | None = None
_snapshot: dict[str, dict] = {}


def _composite_percent(stages: list[str], current: str, value: float) -> int:
    """Map (stage, in-stage value) onto one 0-100 range, text-heavy."""
    if not stages or current not in stages or len(stages) == 1:
        return max(0, min(100, round(value * 100)))
    extras = [s for s in stages if s != "text_model"]
    extra_share = (1.0 - _TEXT_STAGE_SHARE) / len(extras) if extras else 0.0
    offset = 0.0
    for stage in stages:
        share = _TEXT_STAGE_SHARE if stage == "text_model" else extra_share
        if stage == current:
            return max(0, min(100, round((offset + share * value) * 100)))
        offset += share
    return max(0, min(100, round(value * 100)))


def _endpoint() -> "tuple[str, str] | None":
    """(base_root, api_key) of the managed router, or None.

    Resolved through the endpoint module's ownership-guarded reader, not
    a raw state-file read: on the shared stable port, a foreign install's
    server answers /health for anyone, and a raw read would attach this
    watcher to someone else's SSE stream (or spin on 401s against it).
    The guard's dead-pid check is the ownership proof."""
    try:
        from hermes_cli.local_runtime.endpoint import _state_endpoint

        state = _state_endpoint()
        if state is None:
            return None
        base = str(state.get("base_url", "")).rsplit("/v1", 1)[0]
        return (base, str(state.get("api_key", ""))) if base else None
    except Exception:  # noqa: BLE001
        return None


def _apply_event(model: str, event: str, data: dict) -> None:
    with _lock:
        status = str(data.get("status", ""))
        if event in ("status_change", "model_status") and status == "loading":
            progress = data.get("progress") or {}
            stages = [str(s) for s in (progress.get("stages") or [])]
            current = str(progress.get("current", ""))
            value = progress.get("value")
            entry = _snapshot.setdefault(model, {"stage": "", "value": 0.0,
                                                 "percent": 0, "ts": 0.0})
            entry["ts"] = time.monotonic()
            if current and isinstance(value, (int, float)):
                entry["stage"] = current
                entry["value"] = float(value)
                entry["percent"] = _composite_percent(stages, current, float(value))
        elif event in ("status_change", "model_status", "model_remove"):
            # Any terminal status (loaded/unloaded/failed) ends the load.
            if status != "loading":
                _snapshot.pop(model, None)


def _watch() -> None:
    while True:
        endpoint = _endpoint()
        if endpoint is None:
            with _lock:
                _snapshot.clear()
            time.sleep(_RECONNECT_DELAY_S)
            continue
        base, key = endpoint
        try:
            req = urllib.request.Request(
                f"{base}/models/sse",
                headers={"Authorization": f"Bearer {key}",
                         "Accept": "text/event-stream"})
            with urllib.request.urlopen(req, timeout=60) as r:
                buf = b""
                while True:
                    chunk = r.read1(4096) if hasattr(r, "read1") else r.read(4096)
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        text = line.decode("utf-8", "replace").strip()
                        if not text.startswith("data:"):
                            continue
                        try:
                            msg = json.loads(text[5:].strip())
                            _apply_event(str(msg.get("model", "")),
                                         str(msg.get("event", "")),
                                         msg.get("data") or {})
                        except (json.JSONDecodeError, TypeError):
                            continue
        except Exception as exc:  # noqa: BLE001 — watcher must never die loud
            logger.debug("load-progress SSE reconnecting: %s", exc)
        # Stream ended (router bounce, timeout, error): loading entries from
        # the dead connection are unverifiable — drop rather than freeze.
        with _lock:
            _snapshot.clear()
        time.sleep(_RECONNECT_DELAY_S)


def _ensure_watcher() -> None:
    global _watcher
    with _lock:
        if _watcher is None or not _watcher.is_alive():
            _watcher = threading.Thread(target=_watch, daemon=True,
                                        name="llamacpp-load-progress")
            _watcher.start()


def get_loading_progress() -> dict[str, dict]:
    """{model_id: {"stage", "value", "percent"}} for models loading right
    now. Empty when nothing is loading (or nothing is knowable)."""
    _ensure_watcher()
    now = time.monotonic()
    with _lock:
        return {m: {"stage": e["stage"], "value": e["value"],
                    "percent": e["percent"]}
                for m, e in _snapshot.items()
                if now - e["ts"] < _STALE_ENTRY_TTL_S}


def get_prefill_progress(model: str) -> "dict | None":
    """{"processed": tokens} while the managed server is prompt-processing
    for ``model``, or None (idle, decoding, unreachable, or foreign server).

    llama-server's /slots reports ``n_prompt_tokens_processed`` climbing in
    real time during prefill, but exposes no total — callers supply their
    own denominator (the request's estimated token count). Busiest
    processing slot wins when several are active: a parallel small request
    (title generation) freezes its counter during decode while a live
    prefill keeps climbing past it. One authenticated HTTP call per poll;
    every failure reads as "no prefill" — this is garnish, never load-
    bearing.
    """
    ep = _endpoint()
    if ep is None:
        return None
    base, key = ep
    try:
        from urllib.parse import quote

        req = urllib.request.Request(
            f"{base}/slots?model={quote(model)}",
            headers={"Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req, timeout=2) as r:
            slots = json.loads(r.read())
    except Exception:  # noqa: BLE001
        return None
    best = 0
    for slot in slots if isinstance(slots, list) else []:
        if not slot.get("is_processing"):
            continue
        try:
            processed = int(slot.get("n_prompt_tokens_processed") or 0)
        except (TypeError, ValueError):
            continue
        best = max(best, processed)
    return {"processed": best} if best > 0 else None
