"""Live model-load progress from the managed llama-server router.

Children emit per-tensor progress ({stages, current, value}) which the router relays ONLY over its
/models/sse stream — GET /models carries just the coarse status. The watcher starts on first call,
reconnects with backoff (the router bounces on download/eject), and never raises into callers: no
router, no state file, or no SSE support (older engines) all read as "nothing loading".
"""

from __future__ import annotations

from contextlib import suppress
import json
import logging
import threading
import time
import urllib.request

logger = logging.getLogger(__name__)

_TEXT_STAGE_SHARE = 0.85     # composite range share for the text model
_RECONNECT_DELAY_S = 3.0
_STALE_ENTRY_TTL_S = 120.0   # a loading entry with no events this long is dead
_LOAD_EVENTS = ("status_change", "model_status")

_lock = threading.Lock()
_watcher: threading.Thread | None = None
_snapshot: dict[str, dict] = {}


def _pct(x: float) -> int:
    return max(0, min(100, round(x * 100)))


def _composite_percent(stages: list[str], current: str, value: float) -> int:
    """Map (stage, in-stage value) onto one 0-100 range, text-heavy."""
    if not stages or current not in stages or len(stages) == 1:
        return _pct(value)
    extras = [s for s in stages if s != "text_model"]
    extra_share = (1.0 - _TEXT_STAGE_SHARE) / len(extras) if extras else 0.0
    offset = 0.0
    for stage in stages:
        share = _TEXT_STAGE_SHARE if stage == "text_model" else extra_share
        if stage == current:
            return _pct(offset + share * value)
        offset += share
    return _pct(value)


def _endpoint() -> "tuple[str, str] | None":
    """(base_root, api_key) of the managed router via the ownership-guarded reader, or None."""
    from hermes_cli.local_runtime.endpoint import managed_root

    return managed_root()


def _apply_event(model: str, event: str, data: dict) -> None:
    with _lock:
        status = str(data.get("status", ""))
        if event in _LOAD_EVENTS and status == "loading":
            progress = data.get("progress") or {}
            stages = [str(s) for s in (progress.get("stages") or [])]
            current = str(progress.get("current", ""))
            value = progress.get("value")
            entry = _snapshot.setdefault(model, {"stage": "", "value": 0.0, "percent": 0, "ts": 0.0})
            entry["ts"] = time.monotonic()
            if current and isinstance(value, (int, float)):
                entry["stage"] = current
                entry["value"] = float(value)
                entry["percent"] = _composite_percent(stages, current, float(value))
        elif event in (*_LOAD_EVENTS, "model_remove") and status != "loading":
            # Any terminal status (loaded/unloaded/failed) ends the load.
            _snapshot.pop(model, None)


def _clear_snapshot() -> None:
    with _lock:
        _snapshot.clear()


def _watch() -> None:
    while True:
        endpoint = _endpoint()
        if endpoint is None:
            _clear_snapshot()
            time.sleep(_RECONNECT_DELAY_S)
            continue
        base, key = endpoint
        try:
            req = urllib.request.Request(f"{base}/models/sse", headers={
                "Authorization": f"Bearer {key}", "Accept": "text/event-stream"})
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
                        with suppress(json.JSONDecodeError, TypeError):
                            msg = json.loads(text[5:].strip())
                            _apply_event(str(msg.get("model", "")), str(msg.get("event", "")),
                                         msg.get("data") or {})
        except Exception as exc:  # noqa: BLE001 — watcher must never die loud
            logger.debug("load-progress SSE reconnecting: %s", exc)
        # Stream ended (router bounce, timeout, error): loading entries from the dead connection
        # are unverifiable — drop rather than freeze.
        _clear_snapshot()
        time.sleep(_RECONNECT_DELAY_S)


def _ensure_watcher() -> None:
    global _watcher
    with _lock:
        if _watcher is None or not _watcher.is_alive():
            _watcher = threading.Thread(target=_watch, daemon=True, name="llamacpp-load-progress")
            _watcher.start()


def get_loading_progress() -> dict[str, dict]:
    """{model_id: {"stage", "value", "percent"}} for models loading right now. Empty when
    nothing is loading (or nothing is knowable)."""
    _ensure_watcher()
    now = time.monotonic()
    with _lock:
        return {m: {"stage": e["stage"], "value": e["value"], "percent": e["percent"]}
                for m, e in _snapshot.items() if now - e["ts"] < _STALE_ENTRY_TTL_S}


def get_prefill_progress(model: str) -> "dict | None":
    """{"processed": tokens} while the managed server is prompt-processing for ``model``, or None
    (idle, decoding, unreachable, or foreign server).

    /slots exposes ``n_prompt_tokens_processed`` but no total, so callers supply the denominator.
    The busiest processing slot wins: a parallel small request freezes its counter during decode
    while a live prefill keeps climbing. One HTTP call per poll; every failure reads as "no
    prefill" — garnish, never load-bearing.
    """
    ep = _endpoint()
    if ep is None:
        return None
    try:
        from urllib.parse import quote

        from hermes_cli.local_runtime.endpoint import managed_get_json

        slots = managed_get_json(*ep, f"/slots?model={quote(model)}", timeout_s=2)
    except Exception:  # noqa: BLE001
        return None
    best = 0
    for slot in slots if isinstance(slots, list) else []:
        with suppress(TypeError, ValueError):
            if slot.get("is_processing"):
                best = max(best, int(slot.get("n_prompt_tokens_processed") or 0))
    return {"processed": best} if best > 0 else None
