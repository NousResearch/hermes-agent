"""macs_dump — Hermes observability plugin: conditional Decision-Chain Dump.

CICS DUMP/SLIP, for agents. Observer hooks are an always-on telemetry stream;
when a SLIP-style trigger fires (a tool fails N times, latency over budget, an
abnormal finish), this plugin freezes the *whole decision chain* of that turn —
system prompt / input, the full LLM response, the tool-call sequence, timings
and resources — into a self-contained ``macs.dump.v0`` JSON artifact. Logs say
what happened; a dump preserves what the world looked like at the instant it did.

Self-contained: standard library only, **zero external dependencies**. Read-only
observer hooks (``hermes.observer.v1``); no middleware; **zero core changes**;
**fail-open** — a bug in here can never break the agent.

Enable:  ``hermes plugins enable observability/macs_dump``
Output:  ``~/.hermes/macs-dump/<date>/`` (+ ``index.jsonl``)

Canonical multi-runtime version + spec + schema: github.com/DeepArchi/macs
Relates to Hermes issues #6741 (structured tracing) and #6642 (unified telemetry).
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "macs.dump.v0"
SOURCE_SCHEMA = "hermes.observer.v1"
Key = Tuple[Optional[str], Optional[str]]
Event = Tuple[str, Dict[str, Any]]

# --------------------------------------------------------------------------- #
# SLIP-style triggers (evaluate on a completed step; first match wins).
# Defaults capture *escalated* conditions only — not every transient hiccup, so
# production stays quiet until something is genuinely worth a snapshot. The
# `tool_error` predicate exists for users who want a dump on the first failure;
# it is intentionally NOT in the defaults.
# --------------------------------------------------------------------------- #
DEFAULT_TRIGGERS: List[Dict[str, Any]] = [
    {"predicate": "tool_repeat_fail", "threshold": 3},
    {"predicate": "latency", "threshold_ms": 30000},
    {"predicate": "finish_anomaly", "reasons": ["length", "content_filter"]},
]


def _evaluate(triggers, events: List[Event], latest: Event) -> Optional[Dict[str, Any]]:
    kind, data = latest
    for t in triggers:
        p = t.get("predicate")
        if p == "tool_error" and kind == "tool_post" and data.get("status") == "error":
            return {"predicate": p, "detail": f"tool={data.get('tool_name')}",
                    "matched_value": data.get("error_message")}
        if p == "tool_repeat_fail" and kind == "tool_post" and data.get("status") == "error":
            thr = int(t.get("threshold", 3))
            name = data.get("tool_name")
            n = sum(1 for k, d in events
                    if k == "tool_post" and d.get("tool_name") == name and d.get("status") == "error")
            if n >= thr:
                return {"predicate": p, "detail": f"tool={name}", "matched_value": n, "threshold": thr}
        if p == "api_error" and kind == "api_error":
            return {"predicate": p, "detail": str(data.get("reason") or ""),
                    "matched_value": data.get("status_code")}
        if p == "latency" and kind == "response":
            thr = int(t.get("threshold_ms", 30000))
            dur = data.get("duration_ms")
            if isinstance(dur, (int, float)) and dur > thr:
                return {"predicate": p, "detail": "api_duration", "matched_value": dur, "threshold": thr}
        if p == "finish_anomaly" and kind == "response":
            if data.get("finish_reason") in set(t.get("reasons", ["length", "content_filter"])):
                return {"predicate": p, "matched_value": data.get("finish_reason")}
    return None


# --------------------------------------------------------------------------- #
# Per-turn ring buffer + dump assembly
# --------------------------------------------------------------------------- #
class _Collector:
    def __init__(self, ttl_s: float = 900.0, maxlen: int = 200):
        self.ttl_s, self.maxlen = ttl_s, maxlen
        self._buffers: Dict[Key, Dict[str, Any]] = {}
        self._meta: Dict[str, Dict[str, Any]] = {}

    def open(self, session_id, meta=None):
        self._meta[session_id] = meta or {}

    def evict(self, session_id):
        for k in [k for k in self._buffers if k[0] == session_id]:
            self._buffers.pop(k, None)
        self._meta.pop(session_id, None)

    def record(self, key: Key, kind: str, data: Dict[str, Any], now=None):
        now = time.time() if now is None else now
        for k in [k for k, b in self._buffers.items() if now - b["t"] > self.ttl_s]:
            self._buffers.pop(k, None)
        b = self._buffers.get(key)
        if b is None:
            b = {"events": deque(maxlen=self.maxlen), "t": now, "meta": self._meta.get(key[0], {})}
            self._buffers[key] = b
        b["events"].append((kind, dict(data)))
        return b

    def view(self, key: Key) -> List[Event]:
        b = self._buffers.get(key)
        return list(b["events"]) if b else []

    def assemble(self, key: Key, trigger: Dict[str, Any]) -> Dict[str, Any]:
        b = self._buffers.get(key, {"events": deque(), "meta": {}})
        events: List[Event] = list(b["events"])
        meta = b.get("meta", {})
        req = next((d for k, d in events if k == "request"), {})
        resp = next((d for k, d in reversed(events) if k == "response"), {})
        pre = {d.get("tool_call_id"): d for k, d in events if k == "tool_pre"}
        tool_calls = []
        for k, d in events:
            if k != "tool_post":
                continue
            a = pre.get(d.get("tool_call_id"), {})
            tool_calls.append({
                "tool_call_id": d.get("tool_call_id"),
                "tool_name": d.get("tool_name") or a.get("tool_name"),
                "args": a.get("args"), "status": d.get("status"),
                "duration_ms": d.get("duration_ms"), "result": d.get("result"),
                "error_type": d.get("error_type"), "error_message": d.get("error_message"),
            })
        return {
            "dump_id": str(uuid.uuid4()),
            "schema": SCHEMA_VERSION,
            "source_telemetry_schema": meta.get("source_schema", SOURCE_SCHEMA),
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "trigger": trigger,
            "correlation": {
                "session_id": key[0], "turn_id": key[1],
                "api_request_id": resp.get("api_request_id") or req.get("api_request_id"),
                "parent_session_id": meta.get("parent_session_id"), "subagent_id": meta.get("subagent_id"),
            },
            "context": {
                "model": req.get("model") or resp.get("model") or meta.get("model"),
                "provider": req.get("provider"),
                "system_prompt": req.get("system_prompt") or meta.get("system_prompt"),
                "input_messages": req.get("input_messages"),
                "approx_input_tokens": req.get("approx_input_tokens"),
            },
            "llm_response": {
                "assistant_message": resp.get("assistant_message"),
                "finish_reason": resp.get("finish_reason"), "usage": resp.get("usage"),
                "reasoning": resp.get("reasoning"),
            },
            "tool_calls": tool_calls,
            "subagents": [d for k, d in events if k == "subagent"],
            "control_block": {
                "started_at": req.get("started_at") or meta.get("started_at"),
                "api_call_count": sum(1 for k, _ in events if k == "request"),
                "tool_call_count": len(tool_calls),
                "duration_ms": resp.get("duration_ms"), "tokens": resp.get("usage") or {},
            },
            "env": {"hermes_version": meta.get("hermes_version"),
                    "platform": meta.get("platform") or req.get("provider"), "cwd": meta.get("cwd")},
        }


def _write_dump(base_dir: str, dump: Dict[str, Any]) -> Optional[str]:
    try:
        date = (dump.get("captured_at", "") or "")[:10] or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        day = os.path.join(os.path.expanduser(base_dir), date)
        os.makedirs(day, exist_ok=True)
        c = dump.get("correlation", {})
        path = os.path.join(day, f"{str(c.get('session_id') or 'ns')[:24]}__"
                                 f"{str(c.get('turn_id') or 'nt')[:24]}__{dump['dump_id']}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dump, f, ensure_ascii=False, indent=2)
        with open(os.path.join(day, "index.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({"dump_id": dump["dump_id"], "captured_at": dump["captured_at"],
                                "trigger": dump["trigger"].get("predicate"),
                                "session_id": c.get("session_id"), "turn_id": c.get("turn_id"),
                                "path": path}, ensure_ascii=False) + "\n")
        return path
    except Exception as e:  # fail-open
        logger.warning("macs_dump: sink write failed: %r", e)
        return None


# --------------------------------------------------------------------------- #
# Hermes hook mapping (verified against Hermes v0.16.0, commit 5d3be89)
# --------------------------------------------------------------------------- #
def _key(kw: Dict[str, Any]) -> Key:
    return (kw.get("session_id"), kw.get("task_id") or kw.get("turn_id"))


def _ms(v: Any) -> Optional[float]:
    if not isinstance(v, (int, float)):
        return None
    return v * 1000.0 if v < 1000 else float(v)


def _tool_status(result: Any) -> Tuple[str, Optional[str]]:
    if isinstance(result, dict):
        err = result.get("error") or result.get("is_error") or result.get("isError")
        if err:
            return "error", str(result.get("error") or result.get("error_message") or err)
    if isinstance(result, str) and result.strip().lower().startswith("error"):
        return "error", result[:300]
    return "ok", None


def register(ctx) -> None:
    """Hermes plugin entry point."""
    cfg = getattr(ctx, "config", None) or {}
    get = cfg.get if hasattr(cfg, "get") else (lambda *a: None)
    base_dir = get("dir") or "~/.hermes/macs-dump"
    triggers = get("triggers") or DEFAULT_TRIGGERS
    collector = _Collector(ttl_s=get("ttl_s") or 900.0, maxlen=get("maxlen") or 200)

    def _feed(key: Key, kind: str, data: Dict[str, Any]) -> None:
        try:
            collector.record(key, kind, data)
            if kind not in ("response", "api_error", "tool_post"):
                return
            hit = _evaluate(triggers, collector.view(key), (kind, data))
            if hit:
                _write_dump(base_dir, collector.assemble(key, hit))
        except Exception as e:  # fail-open: never break the agent
            logger.warning("macs_dump: feed failed: %r", e)

    def pre_api_request(**kw):
        collector.open(kw.get("session_id"), {"model": kw.get("model"),
                                               "platform": kw.get("platform")})
        _feed(_key(kw), "request", {
            "api_request_id": str(kw.get("api_call_count", "")),
            "model": kw.get("model"), "provider": kw.get("provider"),
            "system_prompt": kw.get("system_prompt"),
            "input_messages": kw.get("request_messages") or kw.get("messages")
                              or kw.get("conversation_history") or kw.get("request"),
            "approx_input_tokens": kw.get("approx_input_tokens"),
        })

    def post_api_request(**kw):
        _feed(_key(kw), "response", {
            "api_request_id": str(kw.get("api_call_count", "")),
            "assistant_message": kw.get("assistant_message") or kw.get("response")
                                 or kw.get("assistant_response"),
            "finish_reason": kw.get("finish_reason"), "usage": kw.get("usage"),
            "duration_ms": _ms(kw.get("api_duration") or kw.get("duration_ms")),
            "model": kw.get("model"),
        })

    def pre_tool_call(**kw):
        _feed(_key(kw), "tool_pre", {"tool_call_id": kw.get("tool_call_id"),
                                     "tool_name": kw.get("tool_name"), "args": kw.get("args")})

    def post_tool_call(**kw):
        status = kw.get("status")
        err = kw.get("error_message")
        if status is None:
            status, err = _tool_status(kw.get("result"))
        _feed(_key(kw), "tool_post", {
            "tool_call_id": kw.get("tool_call_id"), "tool_name": kw.get("tool_name"),
            "args": kw.get("args"), "status": status, "result": kw.get("result"),
            "duration_ms": kw.get("duration_ms"), "error_type": kw.get("error_type"),
            "error_message": err,
        })

    def on_session_finalize(**kw):
        collector.evict(kw.get("session_id"))

    for event, cb in (("pre_api_request", pre_api_request),
                      ("post_api_request", post_api_request),
                      ("pre_tool_call", pre_tool_call),
                      ("post_tool_call", post_tool_call),
                      ("on_session_finalize", on_session_finalize)):
        try:
            ctx.register_hook(event, cb)
        except Exception:  # fail-open
            pass
