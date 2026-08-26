"""Implicit skill prefetch for the turn prologue.

Ported semantic from OpenAI Codex's implicit skill invocation
(``codex-rs/skills/src/invocation.rs``): when the user's prompt mentions a
skill name, load that skill's full instructions into the turn's prefetch
cache so the model has them on the first turn instead of round-tripping
``skill_view()`` first.

Hermes differs from Codex in the detection surface. Codex parses shell
command tokens (running a skill's script / reading a skill's doc); Hermes
matches the natural-language prompt against the skill index. The matching
rules are borrowed from the desktop suggestion provider
(``apps/desktop/src/store/suggestion-providers/skill.ts``), which already
solved the false-positive problem:

- Unicode word boundaries (a bare ``codex`` cannot match inside
  ``codexified``; a trailing ``-`` is excluded so ``codex`` cannot match the
  prefix of ``codex-operations``).
- Hyphens/underscores in a skill name also match spaces — people write
  "pr ready", the skill is ``pr-ready``.
- A minimum name length guards against common-word false positives (``pdf``,
  ``git``, ``box``).

The result is appended to the same ``ext_prefetch_cache`` the memory manager
uses, so it rides the existing prompt-cache-safe channel (injected into the
API copy of the user message only; stored content stays clean). Zero cost
when nothing matches; bounded when several skills match.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable, List, Optional

logger = logging.getLogger(__name__)

# Names shorter than this are too likely to be ordinary English words.
MIN_NAME_LENGTH = 4
# Cap on how many skills a single turn prefetches (prompt rarely names more).
MAX_PREFETCH_SKILLS = 3
# Per-skill and total body budgets keep the injection bounded.
MAX_SKILL_CHARS = 8_000
MAX_TOTAL_CHARS = 16_000


def _skill_pattern(name: str) -> re.Pattern:
    """Whole-word pattern for a skill name, exported for tests.

    Hyphens and underscores in the name also match spaces — people type
    "pr ready", the skill is ``pr-ready`` — while the leading boundary and
    the trailing ``(?![A-Za-z0-9-])`` guarantee a bare ``codex`` can never
    match inside ``codexified`` or as the prefix of ``codex-operations``.
    """
    flexible = re.escape(name.lower()).replace(r"\-", "[-_ ]").replace(r"\_", "[-_ ]")
    return re.compile(rf"(?<![A-Za-z0-9]){flexible}(?![A-Za-z0-9-])", re.IGNORECASE)


def detect_mentioned_skill_names(
    prompt: "str | None",
    skill_names: Iterable[str],
) -> List[str]:
    """Return skill names the prompt mentions, most-specific first, capped.

    Names are matched whole-word against the prompt. When several patterns
    hit (e.g. both ``codex`` and ``codex-operations`` are skills and the
    prompt says "codex-operations"), longer names win so the most specific
    skill is prefetched.
    """
    if not prompt:
        return []
    hits = []
    for name in skill_names:
        if not name or len(name) < MIN_NAME_LENGTH:
            continue
        try:
            if _skill_pattern(name).search(prompt):
                hits.append(name)
        except re.error:
            continue
    hits.sort(key=len, reverse=True)
    return hits[:MAX_PREFETCH_SKILLS]


def _safe_skill_name(name: str) -> bool:
    """A skill name coming from the index is already trusted, but defense in
    depth: refuse anything that could escape the skills directory (path
    separators / traversal) before we join it onto a search dir."""
    if not name or name != name.strip():
        return False
    if any(ch in name for ch in ("/", "\\", "\x00")):
        return False
    if ".." in name:
        return False
    return True


def _find_skill_md(name: str) -> Optional[Path]:
    """Locate a SKILL.md by directory name or frontmatter ``name``.

    Mirrors ``skill_view``'s recursive lookup without its JSON output and
    side effects (env registration, read tracking) — prefetch must not
    trigger those.
    """
    if not _safe_skill_name(name):
        return None
    try:
        from agent.skill_utils import (
            get_scan_ordered_skills_dirs,
            iter_skill_index_files,
            parse_frontmatter,
        )
    except Exception:
        return None
    for skills_dir in get_scan_ordered_skills_dirs():
        if not skills_dir.exists():
            continue
        try:
            for skill_md in iter_skill_index_files(skills_dir, "SKILL.md"):
                if skill_md.parent.name == name:
                    return skill_md
                try:
                    raw = skill_md.read_text(encoding="utf-8-sig", errors="replace")
                    fm, _ = parse_frontmatter(raw)
                except Exception:
                    fm = {}
                if fm.get("name") == name:
                    return skill_md
        except Exception as e:
            logger.debug("Skill prefetch scan failed in %s: %s", skills_dir, e)
    return None


def _read_skill_body(name: str) -> str:
    """Read a skill's SKILL.md body (frontmatter stripped), bounded."""
    md = _find_skill_md(name)
    if md is None:
        return ""
    try:
        raw = md.read_text(encoding="utf-8-sig", errors="replace")
    except Exception as e:
        logger.debug("Skill prefetch read failed for %s: %s", name, e)
        return ""
    try:
        from agent.skill_utils import parse_frontmatter

        _, body = parse_frontmatter(raw)
    except Exception:
        body = raw
    return body[:MAX_SKILL_CHARS]


def build_skill_prefetch(prompt: str) -> str:
    """Return fenced skill bodies for skills the prompt mentions, or ``""``.

    Safe no-op when nothing matches, when the skill index is unavailable, or
    when a skill vanished between index and read.
    """
    if not prompt or not prompt.strip():
        return ""
    try:
        from tools.skills_tool import _find_all_skills

        names = [str(s["name"]) for s in _find_all_skills() if s.get("name")]
    except Exception as e:
        logger.debug("Skill prefetch index unavailable: %s", e)
        return ""
    hits = detect_mentioned_skill_names(prompt, names)
    if not hits:
        return ""
    parts: List[str] = []
    total = 0
    for name in hits:
        body = _read_skill_body(name)
        if not body:
            continue
        if total + len(body) > MAX_TOTAL_CHARS:
            body = body[: MAX_TOTAL_CHARS - total]
        parts.append(f"[Implicitly loaded skill: {name}]\n{body}")
        total += len(body)
        if total >= MAX_TOTAL_CHARS:
            break
    return "\n\n".join(parts)
