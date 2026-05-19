"""Busy-queue configuration and priority helpers.

Standalone module so GatewayRunner logic can stay thin while keeping file-based
queue behavior.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict


def default_busy_queue_config(hermes_home: Path) -> Dict[str, Any]:
    return {
        "enabled": True,
        "storage_path": str(hermes_home / "queues" / "busy_queue.json"),
        "dedupe_exact": True,
        "default_priority": "P1",
        "ack_template": "✅ Queued. I'll process this right after the current task.",
    }


def load_busy_queue_config(hermes_home: Path) -> Dict[str, Any]:
    cfg = default_busy_queue_config(hermes_home)
    try:
        import yaml as _y

        cfg_path = hermes_home / "config.yaml"
        if cfg_path.exists():
            with open(cfg_path, encoding="utf-8") as f:
                user_cfg = (_y.safe_load(f) or {}).get("busy_queue") or {}
            if isinstance(user_cfg, dict):
                cfg.update(user_cfg)
    except Exception:
        pass
    return cfg


def extract_priority(text: str, default: str = "P1") -> str:
    t = (text or "").upper()
    if "P0" in t:
        return "P0"
    if "P2" in t:
        return "P2"
    if "P1" in t:
        return "P1"
    return default if default in {"P0", "P1", "P2"} else "P1"


def priority_rank(priority: str) -> int:
    return {"P0": 0, "P1": 1, "P2": 2}.get((priority or "P1").upper(), 1)


def event_fingerprint(event: Any) -> str:
    source = getattr(event, "source", None)
    src = (
        f"{getattr(getattr(source, 'platform', None), 'value', '')}|"
        f"{getattr(source, 'chat_id', '')}|{getattr(source, 'thread_id', '')}|"
        f"{getattr(event, 'text', '') or ''}|"
        f"{','.join(getattr(event, 'media_urls', []) or [])}|"
        f"{','.join(getattr(event, 'media_types', []) or [])}"
    )
    return hashlib.sha256(src.encode("utf-8", errors="ignore")).hexdigest()
