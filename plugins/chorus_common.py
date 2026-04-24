"""Shared Chorus JSON-RPC client helpers for Hermes plugins.

This module is intentionally dependency-light: stdlib HTTP, profile-aware
config lookup, and fail-open behavior for agent lifecycle hooks.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional


_SECRET_KEYS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "AUTHORIZATION")


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: ("[REDACTED]" if any(s in k.upper() for s in _SECRET_KEYS) else _redact(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


def _config_chorus_env() -> Dict[str, str]:
    try:
        from hermes_cli.config import load_config
        cfg = load_config() or {}
        server = ((cfg.get("mcp_servers") or {}).get("chorus") or {})
        env = server.get("env") or {}
        return {str(k): str(v) for k, v in env.items() if v is not None}
    except Exception:
        return {}


def get_chorus_config() -> Dict[str, str]:
    env_cfg = _config_chorus_env()
    url = os.getenv("CHORUS_URL") or env_cfg.get("CHORUS_URL") or "http://localhost:3001"
    api_key = os.getenv("CHORUS_API_KEY") or env_cfg.get("CHORUS_API_KEY") or ""
    return {"url": url.rstrip("/"), "api_key": api_key}


@dataclass
class ChorusClient:
    url: str
    api_key: str
    timeout: float = 20.0

    @classmethod
    def from_env(cls, timeout: float = 20.0) -> "ChorusClient":
        cfg = get_chorus_config()
        return cls(url=cfg["url"], api_key=cfg["api_key"], timeout=timeout)

    def is_configured(self) -> bool:
        return bool(self.url and self.api_key)

    def rpc(self, method: str, params: Optional[Dict[str, Any]] = None) -> Any:
        if not self.is_configured():
            raise RuntimeError("Chorus URL/API key not configured")
        payload = {
            "jsonrpc": "2.0",
            "id": f"hermes-{int(time.time() * 1000)}",
            "method": method,
            "params": params or {},
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.url}/rpc",
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Chorus RPC {method} HTTP {exc.code}: {body[:500]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Chorus RPC {method} failed: {exc.reason}") from exc
        parsed = json.loads(body)
        if parsed.get("error"):
            raise RuntimeError(f"Chorus RPC {method} error: {_redact(parsed['error'])}")
        return parsed.get("result")

    def safe_rpc(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            return {"ok": True, "result": self.rpc(method, params)}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}


def compact_resume(result: Any, *, max_items: int = 5) -> str:
    if not isinstance(result, dict):
        return ""
    identity = result.get("identity") or {}
    project = result.get("project") or {}
    workstream = result.get("workstream") or {}
    parts = [
        "Chorus resume context:",
        f"- identity: {identity.get('name') or 'unknown'}",
        f"- project: {project.get('tag') or 'unknown'}",
    ]
    if workstream:
        parts.append(f"- workstream: {workstream.get('title')} ({workstream.get('status')})")
    for key in ("inbox_now", "active_tasks", "blocked_items", "inbox_recent"):
        items = result.get(key) or []
        if items:
            parts.append(f"- {key}: {len(items)}")
            for item in items[:max_items]:
                preview = item.get("preview") or item.get("content") or str(item)[:160]
                parts.append(f"  - {preview[:220]}")
    suggested = result.get("suggested_next_action")
    if suggested:
        parts.append(f"- suggested_next_action: {suggested[:400]}")
    return "\n".join(parts)


def emit_signal(client: ChorusClient, *, content: str, signal_type: str = "sense", to_ring: str = "agents-of-proto", urgency: float = 0.4, tags: Optional[list[str]] = None, from_role: str = "ops") -> Any:
    signal = {
        "signal_type": signal_type,
        "content": content,
        "from_role": from_role,
        "to_ring": to_ring,
        "urgency": urgency,
        "tags": tags or ["hermes", "vesta", "chorus"],
        "resources": [],
        "attachments": [],
    }
    return client.rpc("signals/batch_emit", {"signals": [signal]})
