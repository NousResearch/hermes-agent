#!/usr/bin/env python3
"""Resolve Chinese title for Feishu online doc name (paper-nexus)."""

from __future__ import annotations

import json
import re
from pathlib import Path


def _has_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def resolve_title_zh(
    meta: dict,
    *,
    handoff: dict | None = None,
    title_zh: str | None = None,
    registry_entry: dict | None = None,
) -> str:
    """Pick Chinese paper name for Feishu doc --title (not English arXiv title)."""
    for src in (
        title_zh,
        (handoff or {}).get("title_zh"),
        (handoff or {}).get("paper_title_zh"),
        (registry_entry or {}).get("title_zh"),
    ):
        if src and str(src).strip():
            t = str(src).strip()
            if _has_cjk(t):
                return t[:120]
    raise ValueError(
        "title_zh required for Feishu doc name: set in T0 handoff (title_zh) "
        "or pass --title-zh / registry"
    )


def load_handoff(path: str | Path | None) -> dict | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def feishu_doc_title(canonical_id: str, title_zh: str) -> str:
    """Online doc display name: [id] Chinese name."""
    cid = (canonical_id or "").strip()
    zh = (title_zh or "").strip()
    return f"[{cid}] {zh}"[:200]
