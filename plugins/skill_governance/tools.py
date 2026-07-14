"""Handlers for the skill-governance plugin.

All handlers are read-only and return JSON strings, matching the Hermes plugin
contract documented in ``website/docs/developer-guide/plugins/index.md``.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from .catalog import all_candidates, find_candidate

REGISTERED_TOOLS = [
    "skills_plugin_roadmap_summary",
    "skills_find_plugin_candidates",
    "skills_to_plugin_plan",
]

_PRIORITY_ORDER = {
    "very_high": 0,
    "high": 1,
    "medium_high": 2,
    "medium": 3,
}
_ROADMAP_ORDER = {
    candidate["id"]: index
    for index, candidate in enumerate(all_candidates())
}


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _normalize_args(args: Any) -> dict[str, Any]:
    return args if isinstance(args, dict) else {}


def _compact_candidate(candidate: dict[str, Any], *, include_plan: bool = False) -> dict[str, Any]:
    compact = {
        "id": candidate["id"],
        "title": candidate["title"],
        "priority": candidate["priority"],
        "wave": candidate["wave"],
        "areas": candidate["areas"],
        "sources": candidate["sources"],
        "tools": candidate["tools"],
        "business_value": candidate["business_value"],
        "requires_live_go": candidate["requires_live_go"],
    }
    if include_plan:
        compact["plan"] = _build_plan(candidate)
    return compact


def _safe_limit(raw_limit: Any, default: int, maximum: int) -> int:
    try:
        limit = int(raw_limit if raw_limit is not None else default)
    except (TypeError, ValueError):
        limit = default
    return max(1, min(limit, maximum))


def _matches_query(candidate: dict[str, Any], query: str) -> bool:
    if not query:
        return True
    haystack = " ".join(
        [
            candidate["id"],
            candidate["title"],
            candidate["business_value"],
            " ".join(candidate["areas"]),
            " ".join(candidate["sources"]),
            " ".join(candidate["tools"]),
        ]
    ).lower()
    return query.lower() in haystack


def _sort_key(candidate: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(candidate["wave"]),
        _PRIORITY_ORDER.get(str(candidate["priority"]), 99),
        _ROADMAP_ORDER.get(str(candidate["id"]), 999),
    )


def _build_plan(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract": {
            "goal": "Define typed schemas, safe defaults, required env/config, and JSON output shape.",
            "tools": candidate["tools"],
        },
        "dry_run_mvp": {
            "goal": "Implement read-only and preview/dry-run handlers before live writes.",
            "sources": candidate["sources"],
        },
        "guarded_live": {
            "goal": "Add live/write operations only behind explicit policy gates and tests.",
            "requires_live_go": candidate["requires_live_go"],
        },
        "rollout": {
            "goal": "Document enabling, run targeted tests, then release through normal PR gates.",
            "areas": candidate["areas"],
        },
    }


def _candidate_gates(candidate: dict[str, Any]) -> list[str]:
    gates = [
        "Unit tests for schemas, handlers, and error JSON.",
        "Plugin discovery/load test through PluginManager.",
        "No secrets, raw private IDs, cookies, tokens, or production DSNs in artifacts.",
    ]
    if candidate["requires_live_go"]:
        gates.append("Live/write operations require explicit live GO and read-back proof.")
    return gates


def skills_find_plugin_candidates(args: dict, **kwargs) -> str:
    """Return approved plugin candidates filtered by simple roadmap metadata."""

    args = _normalize_args(args)
    candidates = all_candidates()
    area = str(args.get("area") or "").strip().lower()
    priority = str(args.get("priority") or "").strip().lower()
    wave = args.get("wave")
    query = str(args.get("query") or "").strip()
    include_plan = bool(args.get("include_plan", False))

    if area:
        candidates = [c for c in candidates if area in {a.lower() for a in c["areas"]}]
    if priority:
        candidates = [c for c in candidates if c["priority"].lower() == priority]
    if wave not in (None, ""):
        try:
            wave_int = int(wave)
            candidates = [c for c in candidates if int(c["wave"]) == wave_int]
        except (TypeError, ValueError):
            return _json({"success": False, "error": f"invalid wave: {wave}"})
    if query:
        candidates = [c for c in candidates if _matches_query(c, query)]

    candidates = sorted(candidates, key=_sort_key)
    limit = _safe_limit(args.get("limit"), default=len(candidates) or 1, maximum=12)
    selected = candidates[:limit]
    return _json(
        {
            "success": True,
            "count": len(selected),
            "total_matches": len(candidates),
            "candidates": [
                _compact_candidate(candidate, include_plan=include_plan)
                for candidate in selected
            ],
        }
    )


def skills_to_plugin_plan(args: dict, **kwargs) -> str:
    """Return a detailed implementation plan for one approved candidate."""

    args = _normalize_args(args)
    candidate_id = str(args.get("candidate_id") or "").strip()
    if not candidate_id:
        return _json({"success": False, "error": "candidate_id is required"})

    candidate = find_candidate(candidate_id)
    if candidate is None:
        return _json({"success": False, "error": f"unknown candidate: {candidate_id}"})

    return _json(
        {
            "success": True,
            "candidate": _compact_candidate(candidate),
            "guardrails": candidate["guardrails"],
            "gates": _candidate_gates(candidate),
            "phases": [
                {"name": name, **details}
                for name, details in _build_plan(candidate).items()
            ],
        }
    )


def skills_plugin_roadmap_summary(args: dict, **kwargs) -> str:
    """Summarize the approved skill-to-plugin migration roadmap."""

    _args = _normalize_args(args)
    candidates = all_candidates()
    waves = Counter(str(candidate["wave"]) for candidate in candidates)
    priorities = Counter(str(candidate["priority"]) for candidate in candidates)
    areas = Counter(area for candidate in candidates for area in candidate["areas"])
    requires_live_go = [
        candidate["id"] for candidate in candidates if bool(candidate["requires_live_go"])
    ]
    return _json(
        {
            "success": True,
            "summary_type": "static_roadmap_catalog",
            "total_candidates": len(candidates),
            "waves": dict(sorted(waves.items())),
            "priorities": dict(sorted(priorities.items())),
            "areas": dict(sorted(areas.items())),
            "registered_tools": REGISTERED_TOOLS,
            "requires_live_go": requires_live_go,
            "recommended_first_slice": "skill_governance",
            "next_candidates": [
                "bitrix_ops",
                "telegram_thread_router",
                "management_digest",
                "watchdog_guardian",
            ],
        }
    )
