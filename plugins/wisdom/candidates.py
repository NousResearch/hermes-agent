"""Share candidates: deterministic, on-device qualification of local skills worth sharing.

Facts come from the host's ``on_skill_lifecycle`` hook (loaded / patched / edited), folded into
bounded per-skill day sets in plugin state. Two rules qualify a skill, both traceable to counters
the user can inspect with ``wisdom candidates``:

* ``high_usage`` — loaded on ``CONSECUTIVE_BUSINESS_DAYS`` consecutive business days (Mon–Fri,
  profile-local time) inside the last ``RECENT_DAYS``.
* ``refinement`` — at least ``REQUIRED_REFINEMENTS`` edits in the last ``RECENT_DAYS``, none in
  the last ``STABILITY_DAYS`` (it settled), and used since.

Model text can never qualify a candidate, and a candidate is only ever a suggestion: sharing
still goes through ``Wisdom.share`` with its two human confirmations. Presentation is capped
per ISO week and "not now" silences a skill for ``NOT_NOW_DAYS``.
"""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from typing import Any

RECENT_DAYS = 30
RETENTION_DAYS = 35
CONSECUTIVE_BUSINESS_DAYS = 7
REQUIRED_REFINEMENTS = 3
STABILITY_DAYS = 7
WEEKLY_QUOTA = 3
NOT_NOW_DAYS = 30
_ALLOWED_PROVENANCE = {"local", "agent_created"}  # host labels: installed/external/unknown are not ours to share
_EXCLUDED_PREFIXES = ("_wisdom", "_org", ".archive", ".hub")


def _today(now: float | None) -> date:
    return datetime.fromtimestamp(now if now is not None else time.time()).date()


def _eligible(skill_name: str, provenance: str | None) -> bool:
    if not skill_name or skill_name.startswith(_EXCLUDED_PREFIXES) or "/" in skill_name:
        return False
    if provenance is None:
        from tools.skill_usage import telemetry_provenance
        provenance = telemetry_provenance(skill_name)
    return provenance in _ALLOWED_PROVENANCE


def observe(state, *, action: str, skill_name: str, provenance: str | None = None,
            now: float | None = None, **_: Any) -> None:
    """Hook callback: fold one lifecycle fact into ``candidate_facts[skill] = {days, edits}``."""
    if action not in ("loaded", "patched", "edited") or not _eligible(skill_name, provenance):
        return
    today = _today(now)
    cutoff = (today - timedelta(days=RETENTION_DAYS)).isoformat()
    facts = dict(state.get("candidate_facts") or {})
    rec = dict(facts.get(skill_name) or {"days": [], "edits": []})
    key = "days" if action == "loaded" else "edits"
    values = [*rec.get(key, []), today.isoformat()]
    if key == "days":  # one entry per day; edits keep every occurrence
        values = list(set(values))
    rec[key] = sorted(d for d in values if d >= cutoff)[-64:]
    other = "edits" if key == "days" else "days"
    rec[other] = [d for d in rec.get(other, []) if d >= cutoff][-64:]
    facts[skill_name] = rec
    state.set("candidate_facts", facts)


def _next_business_day(day: date) -> date:
    day += timedelta(days=1)
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return day


def consecutive_business_days(days: list[str]) -> int:
    """Longest run where each day is the next business day after the previous (weekends skipped)."""
    best = run = 0
    prev: date | None = None
    for d in sorted({date.fromisoformat(x) for x in days if date.fromisoformat(x).weekday() < 5}):
        run = run + 1 if prev is not None and d == _next_business_day(prev) else 1
        best, prev = max(best, run), d
    return best


def qualify(state, *, now: float | None = None) -> list[dict]:
    """Candidates worth presenting right now, quota and deferrals applied, most evidence first."""
    today = _today(now)
    recent = (today - timedelta(days=RECENT_DAYS)).isoformat()
    stable = (today - timedelta(days=STABILITY_DAYS)).isoformat()
    shared = set(state.get("shared") or {})
    deferred = state.get("candidate_deferred") or {}
    presented = state.get("candidate_presented") or {}
    week = today.isocalendar()
    week_key = f"{week[0]}-W{week[1]:02d}"
    shown_this_week = sum(1 for v in presented.values() if v.get("week") == week_key)
    out = []
    for skill, rec in (state.get("candidate_facts") or {}).items():
        if skill in shared or deferred.get(skill, "") > today.isoformat():
            continue
        if presented.get(skill, {}).get("week") == week_key:
            continue
        days = [d for d in rec.get("days", []) if d >= recent]
        edits = [d for d in rec.get("edits", []) if d >= recent]
        streak = consecutive_business_days(days)
        if streak >= CONSECUTIVE_BUSINESS_DAYS:
            out.append({"skill": skill, "reason": "high_usage",
                        "evidence": {"consecutive_business_days": streak, "used_days": len(days)}})
        elif len(edits) >= REQUIRED_REFINEMENTS and days and max(edits) < stable and max(days) >= max(edits):
            out.append({"skill": skill, "reason": "refinement",
                        "evidence": {"edits": len(edits), "stable_days": STABILITY_DAYS, "used_days": len(days)}})
    out.sort(key=lambda c: (-c["evidence"].get("consecutive_business_days", 0), -c["evidence"].get("edits", 0), c["skill"]))
    return out[: max(0, WEEKLY_QUOTA - shown_this_week)]


def mark_presented(state, skill: str, *, now: float | None = None) -> None:
    """Count one presentation against this ISO week's quota; the same skill is not re-raised this week."""
    today = _today(now)
    week = today.isocalendar()
    presented = dict(state.get("candidate_presented") or {})
    presented[skill] = {"week": f"{week[0]}-W{week[1]:02d}", "at": today.isoformat()}
    state.set("candidate_presented", presented)


def defer(state, skill: str, *, days: int = NOT_NOW_DAYS, now: float | None = None) -> str:
    """"Not now": silence this skill until the returned date; nothing else changes."""
    until = (_today(now) + timedelta(days=days)).isoformat()
    deferred = dict(state.get("candidate_deferred") or {})
    deferred[skill] = until
    state.set("candidate_deferred", deferred)
    return until


def default_description(skill_name: str) -> str:
    """Share description a surface may prefill: the skill's own frontmatter description."""
    from agent.skill_utils import parse_frontmatter
    from tools.skill_usage import _find_skill_dir
    d = _find_skill_dir(skill_name)
    if d is None:
        raise ValueError(f"local skill {skill_name!r} not found")
    meta, _body = parse_frontmatter((d / "SKILL.md").read_text(encoding="utf-8"))
    desc = str((meta or {}).get("description") or "").strip()
    if not desc:
        raise ValueError(f"{skill_name} has no frontmatter description; pass one explicitly")
    return desc


def describe(c: dict) -> str:
    ev = c["evidence"]
    if c["reason"] == "high_usage":
        return f"{c['skill']}: used on {ev['consecutive_business_days']} consecutive business days"
    return f"{c['skill']}: refined {ev['edits']} times, stable for {ev['stable_days']} days and still in use"
