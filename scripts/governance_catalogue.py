#!/usr/bin/env python3
"""Governance catalogue — BOUNDED DESIGN GAP (P05 Batch 1).

This file is intentionally NOT an implementation.  It documents the bounded
design gap for the governance catalogue/tier framework so that a future
batch can resolve the missing schema decision before writing code.

═══════════════════════════════════════════════════════════════════════════
DESIGN GAP: governance_catalogue.py cannot be safely built yet
═══════════════════════════════════════════════════════════════════════════

The P05 Batch 1 handoff authorised ``governance_catalogue.py`` ONLY IF "a
minimal ownership/tier schema is clearly derivable from existing
skills/tool registry".  After audit of the current skill frontmatter
(``skills/*/SKILL.md``), the ownership/tier schema is NOT clearly
derivable.  Inventing a tier framework without a governing schema decision
would violate the AGENTS.md doctrine against speculative infrastructure
("hooks, callbacks, or extension points with no concrete consumer") and
the handoff's own constraint against inventing code.

── What was audited ──────────────────────────────────────────────────────

All ``skills/*/SKILL.md`` frontmatter fields across 10 skill directories.
Unique frontmatter keys found:

    adoption_status, author, description, license, metadata, name,
    platforms, tags, title, trigger, type, version

The ``metadata.hermes`` sub-table contains:

    tags, category, related_skills

── What is missing ────────────────────────────────────────────────────────

1. **No ``owner`` field.**  Skills do not declare which profile or human
   owns them.  ``adoption_status`` (permanent|unset) is binary, not an
   ownership claim.

2. **No tier framework.**  The handoff requires a "Tier 1/2/3 framework
   DISTINCT from profile runtime tiers".  The existing ``adoption_status``
   field is binary (permanent or absent) — it cannot model three tiers.
   The profile runtime tiers (Tier 1/2/3 from ``config.yaml: tier`` and
   the profile-tier-registry.md) are a DIFFERENT axis (profile liveness)
   and must not be conflated with a skill/tool governance tier (which
   would measure e.g. core/optional/community or maintained/experimental).

3. **No consumer.**  No existing script, cron, or dashboard reads a
   governance catalogue.  Building one now would be speculative
   infrastructure with no concrete consumer (AGENTS.md forbids this).

── What is needed before this can be built ───────────────────────────────

A governing schema decision that defines:

  a. **Tier semantics.**  What does Tier 1/2/3 mean for a skill/tool in the
     governance catalogue?  (e.g. Tier 1 = core-maintained, Tier 2 =
     community-maintained, Tier 3 = experimental/external?)  This must be
     distinct from the profile runtime tier (liveness/dispatch).

  b. **Ownership attribution.**  Where is the owner recorded?  Options:
     - A new ``owner:`` frontmatter key in SKILL.md (requires migration of
       existing skills).
     - A separate ``governance/catalogue.yaml`` registry file.
     - Derivation from the profile that maintains the skill directory.

  c. **Consumer.**  Which script/cron/dashboard will read this catalogue?
     Without a consumer, the catalogue is dead code.

── What this file provides now ───────────────────────────────────────────

A single function, ``audit_skill_metadata``, that extracts the EXISTING
machine-readable metadata from skill frontmatter WITHOUT inventing a tier
framework.  This gives a future batch a factual baseline of what is
available so the schema decision can be made against real data.

This function is test-covered by ``tests/scripts/test_governance_catalogue.py``
which asserts the bounded gap: it confirms the audit runs and that no
tier/owner field exists in the current skill set.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

# Bounded gap: no tier framework is defined.  These are placeholders for
# the future schema decision — they are deliberately empty so that any
# consumer attempting to use them gets an immediate, clear signal.
GOVERNANCE_TIERS: dict[str, set[str]] = {
    "1": set(),  # core-maintained (to be defined)
    "2": set(),  # community-maintained (to be defined)
    "3": set(),  # experimental/external (to be defined)
}

DESIGN_GAP_NOTE = (
    "governance_catalogue.py cannot build a tier/ownership framework: "
    "no owner or tier field exists in skill SKILL.md frontmatter, and "
    "no consumer requires the catalogue. See module docstring for the "
    "bounded design-gap note."
)


def _parse_frontmatter(text: str) -> dict[str, Any]:
    """Parse YAML frontmatter from a SKILL.md file (between --- markers).

    Returns an empty dict if no frontmatter or if YAML is unavailable.
    Does not raise — this is a best-effort audit.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    end = None
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end = i
            break
    if end is None:
        return {}
    fm_text = "\n".join(lines[1:end])
    try:
        import yaml

        return yaml.safe_load(fm_text) or {}
    except Exception:
        return {}


def audit_skill_metadata(skills_root: Path) -> list[dict[str, Any]]:
    """Extract existing machine-readable metadata from skill frontmatter.

    Returns a list of dicts, one per skill directory containing a SKILL.md.
    Each dict includes:
        - name: directory name
        - frontmatter: parsed frontmatter (may be empty)
        - has_owner: bool (always False in the current skill set)
        - has_tier: bool (always False in the current skill set)
        - adoption_status: str | None

    This is a factual audit — it does NOT assign tiers or owners.
    """
    results: list[dict[str, Any]] = []
    if not skills_root.is_dir():
        return results
    for skill_dir in sorted(skills_root.iterdir()):
        if not skill_dir.is_dir():
            continue
        skill_file = skill_dir / "SKILL.md"
        if not skill_file.exists():
            continue
        text = skill_file.read_text(encoding="utf-8", errors="ignore")
        fm = _parse_frontmatter(text)
        results.append(
            {
                "name": skill_dir.name,
                "frontmatter": fm,
                "has_owner": "owner" in fm,
                "has_tier": "tier" in fm,
                "adoption_status": fm.get("adoption_status"),
            }
        )
    return results


def catalogue_gap_report(skills_root: Path) -> dict[str, Any]:
    """Produce a bounded design-gap report for the governance catalogue.

    Returns a dict describing:
        - gap: the design gap string
        - audited_skills: count of skills audited
        - any_owner: whether any skill has an owner field
        - any_tier: whether any skill has a tier field
        - tiers_defined: whether GOVERNANCE_TIERS has any non-empty tier
    """
    audit = audit_skill_metadata(skills_root)
    return {
        "gap": DESIGN_GAP_NOTE,
        "audited_skills": len(audit),
        "any_owner": any(s["has_owner"] for s in audit),
        "any_tier": any(s["has_tier"] for s in audit),
        "tiers_defined": any(bool(v) for v in GOVERNANCE_TIERS.values()),
        "adoption_statuses": sorted(
            {str(s["adoption_status"]) for s in audit if s["adoption_status"]}
        ),
    }


if __name__ == "__main__":
    import json
    import sys

    skills = Path(__file__).resolve().parent.parent / "skills"
    if len(sys.argv) > 1:
        skills = Path(sys.argv[1])
    report = catalogue_gap_report(skills)
    print(json.dumps(report, indent=2, default=str))
