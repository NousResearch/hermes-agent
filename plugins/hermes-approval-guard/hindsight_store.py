"""Approval memory — supports Hindsight / Honcho / none backends.

All memory config is explicitly declared in config.yaml:
  plugin_guard:
    memory:
      backend: hindsight    # hindsight | honcho | none (required)
      bank: approval        # Hindsight bank or Honcho user_id (required)
      hindsight_url: ...    # Hindsight server address (default: localhost:8888)
      honcho_url: ...       # Honcho server address (default: localhost:1819)

Or override via environment variables:
  HINDSIGHT_URL — Hindsight server address
  HONCHO_URL    — Honcho server address

Provides two query dimensions:
  1. Session-level — query approval history for the current session (understand operation chain)
  2. Pattern-level — query cross-session similar operations (trust escalation)
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.request
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_BACKEND = "hindsight"
_DEFAULT_HINDSIGHT_URL = "http://localhost:8888"
_DEFAULT_HONCHO_URL = "http://localhost:1819"
TIMEOUT = 3


def _resolve_hindsight_url(cfg: Dict[str, Any]) -> str:
    """Resolve Hindsight URL: explicit config > env var > default."""
    mem_cfg = cfg.get("memory", {}) if isinstance(cfg.get("memory"), dict) else {}

    explicit = mem_cfg.get("hindsight_url", "")
    if explicit:
        return explicit

    env_val = os.getenv("HINDSIGHT_URL", "")
    if env_val:
        return env_val

    return _DEFAULT_HINDSIGHT_URL


def _resolve_hindsight_bank(cfg: Dict[str, Any]) -> str:
    """Resolve Hindsight bank ID: explicit config > default."""
    mem_cfg = cfg.get("memory", {}) if isinstance(cfg.get("memory"), dict) else {}

    bank = mem_cfg.get("bank", "")
    if bank:
        return bank

    return "approval"


def _get_honcho_url(cfg: Dict[str, Any]) -> str:
    """Resolve Honcho URL: config > env var > default."""
    mem_cfg = cfg.get("memory", {}) if isinstance(cfg.get("memory"), dict) else {}
    return (
        mem_cfg.get("honcho_url")
        or os.getenv("HONCHO_URL")
        or _DEFAULT_HONCHO_URL
    )


# ── Pattern key generation: cross-session matching of similar ops ──


def _build_pattern_key(tool_name: str, args: Dict[str, Any]) -> str:
    """Generate pattern key for cross-session matching of similar operations.

    Examples:
      write_file /etc/nginx/nginx.conf → "write_file/etc/nginx/"
      terminal rm -rf node_modules     → "terminal/rm"
    """
    if tool_name in ("write_file", "patch"):
        path = str(args.get("path", ""))
        dirs = path.rsplit("/", 1)
        if len(dirs) > 1:
            return f"{tool_name}{dirs[0]}/"
        return f"{tool_name}{path}"

    if tool_name == "terminal":
        cmd = str(args.get("command", "")).strip()
        first_word = re.split(r"[;\s|&]", cmd)[0].strip()
        if first_word:
            return f"terminal/{first_word}"
        return "terminal/other"

    if tool_name == "delegate_task":
        goal = str(args.get("goal", ""))
        words = goal.lower().split()
        key_words = [w for w in words if w not in ("the", "a", "an", "is", "in", "to", "of")][:3]
        if key_words:
            return f"delegate_task/ {' '.join(key_words)}"
        return "delegate_task/other"

    return tool_name


# ── Backend selection ──────────────────────────────────────────────


def _get_backend(cfg: Dict[str, Any]) -> str:
    mem_cfg = cfg.get("memory", {})
    if not isinstance(mem_cfg, dict):
        return DEFAULT_BACKEND
    backend = mem_cfg.get("backend", DEFAULT_BACKEND)
    if backend == "none":
        return "none"
    if backend in ("hindsight", "honcho"):
        return backend
    return DEFAULT_BACKEND


# ══════════════════════════════════════════════════════════════════
# Hindsight HTTP API (explicit config, urllib direct)
# ══════════════════════════════════════════════════════════════════


def _hindsight_api(endpoint: str, payload: Dict[str, Any], cfg: Dict[str, Any]) -> Optional[Dict]:
    """Call Hindsight REST API. Uses urllib direct (no extra dependencies)."""
    try:
        url = f"{_resolve_hindsight_url(cfg)}/{endpoint}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.debug("Hindsight API failed (%s): %s", endpoint, exc)
        return None


def _hindsight_retain(content: str, tags: list, cfg: Dict[str, Any]) -> None:
    bank = _resolve_hindsight_bank(cfg)
    _hindsight_api(f"v1/default/banks/{bank}/memories", {
        "items": [{
            "content": content,
            "context": "approval_decision",
            "tags": tags,
        }],
    }, cfg)


def _hindsight_recall_extended(query_parts: list, cfg: Dict[str, Any], limit: int = 5) -> list:
    """Recall and return full memory list (with content + tags)."""
    bank = _resolve_hindsight_bank(cfg)
    result = _hindsight_api(f"v1/default/banks/{bank}/memories/recall", {
        "query": " ".join(query_parts),
    }, cfg)
    if result:
        return result.get("results", [])[:limit]
    return []


# ══════════════════════════════════════════════════════════════════
# Honcho API
# ══════════════════════════════════════════════════════════════════


def _honcho_api(endpoint: str, payload: Dict[str, Any], cfg: Dict[str, Any]) -> Optional[Dict]:
    try:
        url = f"{_get_honcho_url(cfg)}/{endpoint}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.debug("Honcho API failed (%s): %s", endpoint, exc)
        return None


def _honcho_retain(content: str, tags: list, cfg: Dict[str, Any]) -> None:
    user_id = cfg.get("memory", {}).get("bank", "approval") if isinstance(cfg.get("memory"), dict) else "approval"
    _honcho_api("memories", {
        "user_id": user_id,
        "content": content,
        "metadata": {"type": "approval_decision", "tags": tags},
    }, cfg)


def _honcho_recall_extended(query_parts: list, cfg: Dict[str, Any], limit: int = 5) -> list:
    user_id = cfg.get("memory", {}).get("bank", "approval") if isinstance(cfg.get("memory"), dict) else "approval"
    result = _honcho_api(f"memories/{user_id}", {}, cfg)
    if result and isinstance(result, list):
        query = " ".join(query_parts).lower()
        matches = []
        for m in result:
            content = m.get("content", "").lower()
            if any(q in content for q in query_parts):
                matches.append(m)
        return matches[:limit]
    return []


# ══════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════


def record_decision(
    tool_name: str,
    args: Dict[str, Any],
    verdict: str,
    reason: str,
    session_id: str = "",
    cfg: Dict[str, Any] = None,
) -> None:
    """Record approval decision to memory backend.

    Content includes session_id for later session-scoped queries.
    Tags include pattern_key for cross-session pattern matching.
    """
    if cfg is None:
        cfg = {}
    backend = _get_backend(cfg)
    if backend == "none":
        return

    pattern_key = _build_pattern_key(tool_name, args)
    args_summary = _summarize_args(args)
    content = (
        f"[{session_id or 'no_session'}] "
        f"approval:{verdict}:{tool_name}:{args_summary} "
        f"reason={reason[:80]} pk={pattern_key}"
    )

    tags = _build_tags(tool_name, args, session_id, pattern_key)

    try:
        if backend == "hindsight":
            _hindsight_retain(content, tags, cfg)
        elif backend == "honcho":
            _honcho_retain(content, tags, cfg)
    except Exception as exc:
        logger.debug("Failed to record decision (%s): %s", backend, exc)


def query_session_history(
    session_id: str,
    cfg: Dict[str, Any] = None,
    limit: int = 5,
) -> str:
    """Query approval history for the current session.

    Used for ACP prompt injection: lets ACP know what was approved/denied
    earlier in this session. Returns formatted text (empty if no history).
    """
    if not session_id:
        return ""
    cfg = cfg or {}
    backend = _get_backend(cfg)
    if backend == "none":
        return ""

    try:
        if backend == "hindsight":
            memories = _hindsight_recall_extended([session_id, "approval"], cfg, limit)
        elif backend == "honcho":
            memories = _honcho_recall_extended([session_id, "approval"], cfg, limit)
        else:
            return ""

        if not memories:
            return ""

        lines = []
        for m in memories[:limit]:
            content = m.get("content", "")
            lines.append(f"  - {content}")
        return "\n".join(lines)
    except Exception as exc:
        logger.debug("Failed to query session history: %s", exc)
        return ""


def query_pattern_history(
    tool_name: str,
    args: Dict[str, Any],
    cfg: Dict[str, Any] = None,
    limit: int = 5,
) -> str:
    """Query cross-session approval history for similar operations.

    Used for ACP prompt injection: shows ALLOW/DENY stats for historically
    similar operations. Returns formatted text (empty if no history).
    """
    if cfg is None:
        cfg = {}
    backend = _get_backend(cfg)
    if backend == "none":
        return ""

    pattern_key = _build_pattern_key(tool_name, args)
    query_parts = [pattern_key, "approval"]

    try:
        if backend == "hindsight":
            memories = _hindsight_recall_extended(query_parts, cfg, limit)
        elif backend == "honcho":
            memories = _honcho_recall_extended(query_parts, cfg, limit)
        else:
            return ""

        if not memories:
            return ""

        # Count ALLOW/DENY stats
        allows = 0
        denies = 0
        latest = ""
        for m in memories:
            content = m.get("content", "")
            if ":ALLOW:" in content:
                allows += 1
            elif ":DENY:" in content:
                denies += 1
            if not latest:
                latest = content[:120]

        summary = f"{pattern_key}: {allows}x ALLOW, {denies}x DENY"
        if latest:
            summary += f"\n    Latest: {latest}"
        return summary
    except Exception as exc:
        logger.debug("Failed to query pattern history: %s", exc)
        return ""


# ── Helpers ───────────────────────────────────────────────────────


def _build_tags(
    tool_name: str, args: Dict[str, Any],
    session_id: str, pattern_key: str,
) -> list:
    tags = ["approval", tool_name, pattern_key]
    if session_id:
        tags.append(f"sid:{session_id}")

    if tool_name == "terminal":
        cmd = str(args.get("command", ""))
        for kw in ("rm", "mv", "cp", "chmod", "chown", "sudo", "git"):
            if kw in cmd.lower():
                tags.append(kw)
    elif tool_name in ("write_file", "patch"):
        path = str(args.get("path", ""))
        if "/etc/" in path:
            tags.append("system_path")
        elif "/home/" in path:
            tags.append("home_path")
    return tags


def _summarize_args(args: Dict[str, Any]) -> str:
    if not args:
        return "()"
    parts = []
    for k, v in sorted(args.items()):
        v_str = str(v)
        if len(v_str) > 50:
            v_str = v_str[:50] + "..."
        parts.append(f"{k}={v_str}")
    return "(" + ", ".join(parts[:3]) + ")"
