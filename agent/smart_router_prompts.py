"""Prompt and JSON parsing helpers for the smart model router."""

from __future__ import annotations

import json
import re
from typing import Any


ROUTER_SYSTEM_PROMPT = """You are Hermes's pre-turn smart model router.

Choose the least expensive route that is likely to complete the user's next
turn well. This is an advisory decision only: do not solve the task.

Return strict JSON with these keys:
- route: one of cheap, default, strong, moa, no_change
- confidence: number from 0 to 1
- risk: one of low, medium, high
- expected_tool_calls: non-negative integer
- reason: short explanation
- should_use_moa: boolean

Prefer no_change when evidence is weak. Prefer strong for high-risk or
multi-file debugging tasks. Prefer moa only for complex architecture, review,
or multi-perspective reasoning tasks where extra latency is justified.
"""


def build_router_messages(payload: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, sort_keys=True),
        },
    ]


def parse_router_json(text: str) -> dict[str, Any] | None:
    """Parse a router JSON object, tolerating fenced output."""

    raw = (text or "").strip()
    if not raw:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL)
    if fenced:
        raw = fenced.group(1)
    else:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            raw = raw[start : end + 1]
    try:
        parsed = json.loads(raw)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None
