"""Skill routing table — compact intent→skill mapping for system prompt.

Replaces the full ``<available_skills>`` block with a routing-oriented format
that maps categories to skill names.  Skill names are self-describing
(e.g. ``github-code-review``, ``systematic-debugging``) so the model can pick
the closest match and load the full skill with ``skill_view(name)``.

Savings vs full listing: ~75 % fewer tokens (~800 vs ~3100 for a typical
60-skill installation).
"""

from __future__ import annotations

from typing import Dict, List, Tuple


def _derive_routing_hints(
    skills_by_category: Dict[str, List[Tuple[str, str]]],
    category_descriptions: Dict[str, str],
) -> Dict[str, str]:
    """Derive brief routing hints (3-6 words) from category descriptions.

    Truncates long descriptions and falls back to representative skill names
    when the description is missing or irrelevant."""
    hints: Dict[str, str] = {}
    for category, skills in sorted(skills_by_category.items()):
        if not skills:
            continue
        names = [name for name, _ in skills]

        cat_desc = (category_descriptions.get(category) or "").strip().strip("'\"")

        if cat_desc:
            # Try to extract the most meaningful fragment
            # Category descriptions follow pattern: "Skills for X — ..." or "X skills: ..."
            # We want just the core domain description
            for prefix in ("Skills for ", "Knowledge and Tools for "):
                if cat_desc.startswith(prefix):
                    cat_desc = cat_desc[len(prefix):]
                    break
            # Take first ~50 chars or up to first period/semicolon
            if len(cat_desc) > 50:
                cut = max(cat_desc[:50].rfind(" "), cat_desc[:50].rfind(","), 30)
                cat_desc = cat_desc[:cut] if cut > 0 else cat_desc[:50]
            hints[category] = cat_desc
        elif len(names) <= 3:
            hints[category] = ", ".join(names)
        else:
            sample = names[:2]
            hints[category] = f"{', '.join(sample)} +{len(names)-2} more"

    return hints


def format_skills_routing_table(
    skills_by_category: Dict[str, List[Tuple[str, str]]],
    category_descriptions: Dict[str, str] | None = None,
) -> str:
    """Build a compact routing table from the skill index.

    Format::

        ## Skill Router
        category: skill1, skill2, skill3  # routing hint

    The model scans for a matching category, picks a skill name, and calls
    ``skill_view(name)`` for the full instructions.

    Returns empty string if no skills.
    """
    if not skills_by_category:
        return ""

    category_descriptions = category_descriptions or {}
    hints = _derive_routing_hints(skills_by_category, category_descriptions)

    lines: List[str] = [
        "## Skill Router",
        "Scan for matching category, load with skill_view(name).",
        "Skill names are descriptive — pick the closest match.",
        "",
    ]

    for category in sorted(skills_by_category.keys()):
        skills = skills_by_category[category]
        if not skills:
            continue

        # Deduplicate and sort
        skill_names = sorted(set(name for name, _ in skills))
        names_str = ", ".join(skill_names)
        hint = hints.get(category, category)
        lines.append(f"  {category}: {names_str}  # {hint}")

    lines.append("")
    lines.append(
        "Only proceed without loading a skill if genuinely none are relevant."
    )

    return "\n".join(lines)
