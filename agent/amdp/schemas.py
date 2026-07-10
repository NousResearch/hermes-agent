"""COA / review / audit schemas + tolerant JSON extraction for AMDP v0.

Kept as plain dict validators (not a schema library) so the prototype stays
dependency-free. Each ``coerce_*`` takes whatever the model returned and either
returns a normalized dict or raises ValueError with a specific reason, so the
planner can log exactly why a malformed COA/review was rejected.
"""

from __future__ import annotations

import json
import re
from typing import Any


def extract_json(text: str) -> Any:
    """Best-effort parse of a JSON value out of model text.

    Handles three common shapes: clean JSON, a ```json fenced block, and JSON
    embedded in prose (grabs the outermost {...} or [...] span). Raises
    ValueError if nothing parses.
    """
    if not text or not text.strip():
        raise ValueError("empty text")
    # 1. straight parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 2. fenced ```json ... ``` or ``` ... ```
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass
    # 3. outermost object or array span
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        end = text.rfind(close_ch)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("no parseable JSON found")


def _as_list(val: Any) -> list[Any]:
    if val is None:
        return []
    if isinstance(val, list):
        return val
    return [val]


def coerce_coa(obj: Any, fallback_id: str) -> dict[str, Any]:
    """Normalize one COA object to the v0 schema. Raises ValueError if it has
    no usable dispatches (a COA with nothing to do is not a course of action)."""
    if not isinstance(obj, dict):
        raise ValueError("COA is not an object")
    dispatches = []
    for d in _as_list(obj.get("dispatches")):
        if not isinstance(d, dict):
            continue
        task = str(d.get("task") or "").strip()
        if not task:
            continue
        kind = str(d.get("kind") or "act").strip().lower()
        if kind not in ("observe", "act"):
            kind = "act"
        dispatches.append(
            {
                "task": task,
                "constraints": [str(c) for c in _as_list(d.get("constraints"))],
                "success_criteria": [str(c) for c in _as_list(d.get("success_criteria"))],
                "kind": kind,
                "irreversible": bool(d.get("irreversible", False)),
            }
        )
    if not dispatches:
        raise ValueError("COA has no valid dispatches")
    branches = []
    for b in _as_list(obj.get("branches")):
        if isinstance(b, dict) and (b.get("if") or b.get("then")):
            branches.append({"if": str(b.get("if") or ""), "then": str(b.get("then") or "")})
    return {
        "coa_id": str(obj.get("coa_id") or fallback_id).strip() or fallback_id,
        "summary": str(obj.get("summary") or "").strip(),
        "dispatches": dispatches,
        "assumptions": [str(a) for a in _as_list(obj.get("assumptions"))],
        "branches": branches,
    }


def coerce_coas(parsed: Any) -> list[dict[str, Any]]:
    """Accept either a bare list of COAs or ``{"coas": [...]}`` and normalize
    each. Drops individually-invalid COAs but raises if NONE survive."""
    if isinstance(parsed, dict):
        if "coas" in parsed:
            raw = parsed["coas"]
        else:
            # Forced-JSON mode follows the prompt's "coas" key, but be defensive:
            # if the model wrapped the list under a different key, grab the first
            # value that is a list of dicts carrying dispatches/tasks.
            candidate = None
            for val in parsed.values():
                if isinstance(val, list) and any(
                    isinstance(x, dict) and ("dispatches" in x or "task" in x) for x in val
                ):
                    candidate = val
                    break
            raw = candidate if candidate is not None else parsed
    else:
        raw = parsed
    raw = _as_list(raw)
    out: list[dict[str, Any]] = []
    errors: list[str] = []
    for i, item in enumerate(raw):
        fid = chr(ord("A") + i) if i < 26 else f"COA{i}"
        try:
            out.append(coerce_coa(item, fid))
        except ValueError as exc:
            errors.append(f"#{i}: {exc}")
    if not out:
        raise ValueError(f"no valid COAs (tried {len(raw)}: {'; '.join(errors) or 'none'})")
    return out


def coerce_review(obj: Any, coa_id: str) -> dict[str, Any]:
    """Normalize one reviewer verdict to the v0 review schema. Clamps scalar
    ranges; missing/blank risks become an empty list (the planner treats an
    empty risk list as a rubber-stamp signal)."""
    if not isinstance(obj, dict):
        raise ValueError("review is not an object")

    def _num(key: str, lo: float, hi: float, default: float) -> float:
        try:
            v = float(obj.get(key))
        except (TypeError, ValueError):
            v = default
        return max(lo, min(hi, v))

    risks = []
    for r in _as_list(obj.get("risks")):
        if not isinstance(r, dict):
            continue
        desc = str(r.get("desc") or "").strip()
        if not desc:
            continue
        try:
            sev = int(float(r.get("severity_1to5", 1)))
        except (TypeError, ValueError):
            sev = 1
        risks.append({"desc": desc, "severity_1to5": max(1, min(5, sev))})
    return {
        "coa_id": str(obj.get("coa_id") or coa_id),
        "alignment_1to10": _num("alignment_1to10", 1, 10, 5),
        "risks": risks,
        "unstated_assumptions": [str(a) for a in _as_list(obj.get("unstated_assumptions"))],
        "fragility_0to1": _num("fragility_0to1", 0.0, 1.0, 0.5),
        "no_credible_failure_mode": bool(obj.get("no_credible_failure_mode", False)),
    }
