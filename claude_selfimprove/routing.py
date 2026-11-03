"""Decide where an eligible candidate's text actually lands.

Three destinations, matching the pipeline's brief:

* the bounded managed section of ``~/.claude/CLAUDE.md`` — short, critical,
  always-loaded policy;
* a pipeline-owned file under ``~/.claude/rules/`` — detailed global rules;
* a pipeline-owned skill under ``~/.claude/skills/`` — reusable procedures.

Two hard rules enforced here, not left to the caller to remember:

1. **Repo-specific candidates never route anywhere.** ``scope: "repo"``
   means the evidence came from one particular project, so it belongs in
   that project's own instructions file, not in George's global Claude Code
   configuration. This pipeline does not touch per-repo files at all —
   :func:`resolve_target` simply returns ``None`` for these, and the caller
   records the candidate as excluded rather than applying it.
2. **The CLAUDE.md managed block has a strict size cap.** A block that grew
   without bound would defeat the reason CLAUDE.md is always loaded into
   every conversation — long tail detail belongs in a rule file instead.
   When a candidate would push the block over the cap, this downgrades it
   to a rule file rather than refusing it outright.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from . import paths

# Characters allowed in the CLAUDE.md managed section, body text only (not
# counting the marker comments themselves). Kept small on purpose — this
# file is loaded into every single Claude Code conversation.
MAX_MANAGED_BLOCK_CHARS = 1500


@dataclass(frozen=True)
class RoutingDecision:
    kind: str  # "claude_md_block" | "rule" | "skill"
    path: Path | None  # None for claude_md_block; a concrete file otherwise


def slugify(canonical_key: str) -> str:
    slug = re.sub(r"[^a-z0-9-]+", "-", str(canonical_key or "").lower()).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    return slug or "lesson"


def rule_path_for(canonical_key: str) -> Path:
    return paths.rules_dir() / f"{slugify(canonical_key)}.md"


def skill_path_for(canonical_key: str) -> Path:
    return paths.skills_dir() / slugify(canonical_key) / "SKILL.md"


def resolve_target(
    row: dict, *, current_claude_md_block_chars: int = 0
) -> RoutingDecision | None:
    """Decide the target for a candidate row (from ``store.CandidateStore``).

    Returns ``None`` when the candidate must not be routed anywhere globally
    (repo-scoped, or an unrecognized target kind).
    """
    if row.get("scope") != "global":
        return None

    target_kind = row.get("target_kind")
    canonical_key = row.get("canonical_key") or ""

    if target_kind == "claude_md_block":
        entry_chars = len(str(row.get("body") or "")) + len(str(row.get("title") or "")) + 10
        if current_claude_md_block_chars + entry_chars > MAX_MANAGED_BLOCK_CHARS:
            return RoutingDecision(kind="rule", path=rule_path_for(canonical_key))
        return RoutingDecision(kind="claude_md_block", path=None)

    if target_kind == "rule":
        return RoutingDecision(kind="rule", path=rule_path_for(canonical_key))

    if target_kind == "skill":
        return RoutingDecision(kind="skill", path=skill_path_for(canonical_key))

    return None
