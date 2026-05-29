"""AgentOS skills projection for Hermes skill surfaces."""

from __future__ import annotations

import json
import logging
import urllib.parse
from typing import Any, Dict, List, Optional

from .agentos_bridge import KynverAgentOSClient, KynverAgentOSError

logger = logging.getLogger(__name__)


def _coerce_skills(payload: Any) -> List[dict[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return [s for s in payload if isinstance(s, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("skills", "items", "manifest", "results"):
        val = payload.get(key)
        if isinstance(val, list):
            return [s for s in val if isinstance(s, dict)]
    return []


def list_agentos_skill_manifest(client: KynverAgentOSClient) -> List[dict[str, Any]]:
    try:
        payload = client.get("/skills?view=manifest")
        return _coerce_skills(payload)
    except KynverAgentOSError:
        logger.debug("AgentOS skills manifest fetch failed", exc_info=True)
        return []


def format_agentos_skills_index(skills: List[dict[str, Any]]) -> str:
    if not skills:
        return ""
    lines = [
        "# Kynver AgentOS skills (manifest)",
        "Runtime-eligible AgentOS skills. Use local ``skill_view`` for bundled Hermes skills; "
        "fetch AgentOS skill instructions on demand when a slug matches this list.",
    ]
    for skill in skills[:40]:
        slug = skill.get("slug") or skill.get("skillSlug") or skill.get("name") or "?"
        desc = (skill.get("description") or "").strip()
        if len(desc) > 80:
            desc = desc[:77] + "..."
        lines.append(f"- `{slug}`: {desc or '(no description)'}")
    if len(skills) > 40:
        lines.append(f"- … and {len(skills) - 40} more")
    return "\n".join(lines)
