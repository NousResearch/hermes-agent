"""Approval memory — optional backend: Hindsight (default) or Honcho.

Configure via config.yaml:
    plugin_guard:
      memory:
        backend: hindsight  # hindsight | honcho | none
        bank: approval      # Hindsight bank or Honcho user_id
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# Default backend config
HINDSIGHT_URL = "http://localhost:8420"
HONCHO_URL = ""  # Honcho endpoint from Hermes Honcho plugin


def _get_backend(cfg: Dict[str, Any]) -> str:
    """Read configured backend type."""
    mem_cfg = cfg.get("memory", {})
    return str(mem_cfg.get("backend", "hindsight")).lower()


# ── Hindsight REST API ─────────────────────────────────────────────────


def _hindsight_retain(
    content: str,
    bank: str = "approval",
    tags: Optional[List[str]] = None,
) -> bool:
    """Call Hindsight REST API."""
    try:
        resp = requests.post(
            f"{HINDSIGHT_URL}/retain",
            json={
                "content": content,
                "bank": bank,
                "tags": tags or [],
            },
            timeout=5,
        )
        if resp.status_code == 200:
            return True
        logger.warning("Hindsight retain returned %d: %s", resp.status_code, resp.text[:100])
        return False
    except requests.RequestException as exc:
        logger.warning("Hindsight retain failed: %s", exc)
        return False


def _hindsight_recall(
    query: str,
    bank: str = "approval",
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """Query Hindsight for related approval history."""
    try:
        resp = requests.post(
            f"{HINDSIGHT_URL}/recall",
            json={
                "query": query,
                "bank": bank,
                "limit": limit,
            },
            timeout=5,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("results", []) if isinstance(data, dict) else []
        return []
    except requests.RequestException as exc:
        logger.warning("Hindsight recall failed: %s", exc)
        return []


# ── Honcho API (via Hermes Honcho plugin HTTP endpoint) ──────────────


def _honcho_store(user_id: str, content: str) -> bool:
    """Call Honcho API."""
    if not HONCHO_URL:
        return False
    try:
        resp = requests.post(
            f"{HONCHO_URL}/sessions/approval/messages",
            json={"content": content, "user_id": user_id, "is_user": False},
            timeout=5,
        )
        return resp.status_code == 200
    except requests.RequestException as exc:
        logger.warning("Honcho store failed: %s", exc)
        return False


# ── Public API ──────────────────────────────────────────────────────


def store_decision(
    verdict: str,
    tool_name: str,
    args: Dict[str, Any],
    reason: str,
    stage: str,
    cfg: Dict[str, Any],
) -> bool:
    """Record approval decision to memory backend."""
    backend = _get_backend(cfg)
    if backend == "none":
        return True

    bank = cfg.get("memory", {}).get("bank", "approval")
    content = json.dumps({
        "verdict": verdict,
        "tool": tool_name,
        "args_summary": {k: str(v)[:200] for k, v in (args or {}).items()},
        "reason": reason,
        "stage": stage,
        "timestamp": time.time(),
    }, ensure_ascii=False)

    if backend == "honcho":
        return _honcho_store(bank, content)
    else:
        return _hindsight_retain(content, bank=bank, tags=[f"tool:{tool_name}", f"verdict:{verdict}"])


def recall_history(
    tool_name: str,
    cfg: Dict[str, Any],
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """Query approval history for similar operations."""
    backend = _get_backend(cfg)
    if backend == "none":
        return []

    bank = cfg.get("memory", {}).get("bank", "approval")
    if backend == "honcho":
        return []  # Honcho recall not implemented for simplicity
    return _hindsight_recall(f"tool:{tool_name}", bank=bank, limit=limit)
