"""Implicit skill prefetch for the turn prologue.

Ported semantic from OpenAI Codex's implicit skill invocation
(``codex-rs/skills/src/invocation.rs``): when the user's prompt mentions a
skill name, load that skill's full instructions into the turn's prefetch
cache so the model has them on the first turn instead of round-tripping
``skill_view()`` first.

Hermes differs from Codex in the detection surface. Codex parses shell
command tokens (running a skill's script / reading a skill's doc); Hermes
matches the natural-language prompt against the skill index. Two
complementary detection channels are supported:

**Word-boundary auto-match** (the natural-language default). Rules are
borrowed from the desktop suggestion provider
(``apps/desktop/src/store/suggestion-providers/skill.ts``):

- Unicode word boundaries (a bare ``codex`` cannot match inside
  ``codexified``; a trailing ``-`` is excluded so ``codex`` cannot match the
  prefix of ``codex-operations``).
- Hyphens/underscores in a skill name also match spaces — people write
  "pr ready", the skill is ``pr-ready``.
- A minimum name length guards against common-word false positives (``pdf``,
  ``git``, ``box``).

**Explicit mention markers** (bypass the length guard). Three syntaxes let
users force a prefetch when they know they want a specific skill:

- ``@skill-name`` (most common — same as Discord/Slack mentions)
- ``skill: skill-name`` (natural-language intent)
- ``/skill skill-name`` (slash-command style)

Markers bypass ``MIN_NAME_LENGTH`` so short skills (``pdf``, ``git``, ``box``)
remain triggerable when the user is explicit. They still go through
``_safe_skill_name`` defense in depth.

Markers and word-boundary hits are merged, longest-name-first; if both
channels pick up the same skill it is deduped. The merged list is still
capped at ``MAX_PREFETCH_SKILLS`` and ``MAX_TOTAL_CHARS``.

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

# Explicit mention markers that bypass MIN_NAME_LENGTH. Three syntaxes:
#   @skill-name            (Discord/Slack-style mention)
#   skill: skill-name      (natural-language intent)
#   /skill skill-name      (slash-command style)
# The skill-name capture is the same character class used by skill names
# in the index (letters, digits, hyphen, underscore, dot); we validate
# the captured name through _safe_skill_name before accepting it.
_MENTION_MARKER_RE = re.compile(
    r"(?:^|[\s,(])(?:@|(?:skill\s*:\s*)|/(?:skill\s+))"
    r"([A-Za-z0-9][A-Za-z0-9._-]*)",
)


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

    Two detection channels contribute, then are merged and deduped:

    1. **Word-boundary auto-match** — ``codex`` matches ``"research the
       codex harness"``. Skips names shorter than ``MIN_NAME_LENGTH`` to
       avoid common-word false positives.
    2. **Explicit mention marker** — ``@pdf``, ``skill: git``,
       ``/skill obsidian`` force a match regardless of length. The captured
       name still has to pass ``_safe_skill_name`` (defense in depth).

    The merge sorts longest-name-first (most specific wins) and caps the
    result at ``MAX_PREFETCH_SKILLS``. Names that are not in the index are
    silently dropped — a marker for an uninstalled skill should not break
    the turn.
    """
    if not prompt:
        return []
    index = {str(n) for n in skill_names if n}
    if not index:
        return []
    # Channel 1: word-boundary auto-match (length-gated).
    auto_hits: List[str] = []
    for name in index:
        if len(name) < MIN_NAME_LENGTH:
            continue
        try:
            if _skill_pattern(name).search(prompt):
                auto_hits.append(name)
        except re.error:
            continue
    # Channel 2: explicit mention markers (bypass length guard).
    marker_hits: List[str] = []
    for match in _MENTION_MARKER_RE.finditer(prompt):
        candidate = match.group(1)
        # Match is case-insensitive against the index; preserve the
        # index's casing in the result so callers see canonical names.
        resolved = next(
            (n for n in index if n.lower() == candidate.lower()), None
        )
        if resolved and _safe_skill_name(resolved) and resolved not in marker_hits:
            marker_hits.append(resolved)
    # Merge: marker hits first (explicit intent wins), then auto hits.
    # Dedup keeps first occurrence. Longest-first sort applied at the end.
    merged: List[str] = []
    for n in marker_hits + auto_hits:
        if n not in merged:
            merged.append(n)
    merged.sort(key=len, reverse=True)
    return merged[:MAX_PREFETCH_SKILLS]


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

    Observability: every call that produces a non-empty result logs a debug
    line with the resolved skill names and total injected size. The injection
    rides ``ext_prefetch_cache`` which is part of the user-message API
    payload — variable content there can degrade prompt-cache hit rate, so
    it is worth being able to grep for "how much did this turn add?".

    Truncation rule: hits are processed longest-first (most specific wins).
    Each skill is included in full up to ``MAX_SKILL_CHARS``. The accumulator
    stops the moment adding the next skill would exceed ``MAX_TOTAL_CHARS``,
    so a long single skill cannot squeeze out shorter-but-higher-priority
    ones already in the cache.
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
    truncated = False
    for name in hits:
        body = _read_skill_body(name)
        if not body:
            continue
        if total + len(body) > MAX_TOTAL_CHARS:
            body = body[: MAX_TOTAL_CHARS - total]
            truncated = True
        parts.append(f"[Implicitly loaded skill: {name}]\n{body}")
        total += len(body)
        if total >= MAX_TOTAL_CHARS:
            break
    if parts:
        logger.debug(
            "Skill prefetch injected %d skill(s) %s (%d chars, truncated=%s)",
            len(parts),
            [p.split("\n", 1)[0].removeprefix("[Implicitly loaded skill: ").removesuffix("]") for p in parts],
            total,
            truncated,
        )
    return "\n\n".join(parts)
