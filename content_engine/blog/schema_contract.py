"""Schema contract — enforce the SahilBlog Astro schema at MDX write time.

The Astro content collection (`src/content.config.ts`) validates EVERY entry at
build time, so a single draft with an out-of-enum `source`, `tier`, `format`,
or `tags` value fails the whole production build — even unapproved drafts.

This module reads the allowed values directly from `content.config.ts` (single
source of truth, no drift) and normalises a frontmatter dict so the assembler
can never emit a schema-invalid post again. Unknown values are alias-mapped
where we know the intent, otherwise clamped to the enum default (scalars) or
dropped (tags) — always favouring a buildable post over a broken one.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

from config import SAHILBLOG_REPO
from blog.blog_slug import slugify

# ── Fallback contract (used only if content.config.ts can't be parsed) ──
_FALLBACK_SOURCES = {"research-paper", "gitradar", "manual"}
_FALLBACK_TIERS = {"ai", "pm", "builder"}
_FALLBACK_FORMATS = {"essay", "note", "review", "brief", "blueprint"}

# Known bad-value → valid-value aliases the generator/router have emitted.
# Scalars:
_SOURCE_ALIASES = {
    "manual_queue": "manual",
    "manual-queue": "manual",
    "manualqueue": "manual",
    "adhoc": "manual",
    "manual_adhoc": "manual",
    "research": "research-paper",
    "research_paper": "research-paper",
    "paper": "research-paper",
    "git-radar": "gitradar",
    "git_radar": "gitradar",
    "pr": "gitradar",
    # Research-roundup stream (Approach A) uses two NEW enum values that
    # SahilBlog's Astro schema (src/content.config.ts) does not yet enumerate:
    #   tier="research", source="curated-roundup". Until that downstream schema
    #   is updated, normalise_frontmatter clamps them to the safe defaults
    #   (tier=pm, source=manual) so the production build is never broken.
    # See blog/blog_streams.py and docs/RESEARCH_ROUNDUP_LANE.md.
    "curated-roundup": "manual",
    "curated_roundup": "manual",
    "curatedroundup": "manual",
    "roundup": "manual",
}
# Tier aliases — only the research-stream tier needs an explicit alias because
# the other stream tiers (ai/pm/builder) are already valid enum values. The
# alias is keyed on the literal new value so we never accidentally swallow an
# out-of-vocab tier emitted by a different stream in the future.
_TIER_ALIASES = {
    "research": "pm",
}
# Format aliases — "roundup" is a distinct shape (numbered entries, recurring
# container, takeaways) introduced by the research-roundup stream. The Astro
# schema (src/content.config.ts) only enumerates essay/note/review/brief/
# blueprint, so until the schema is updated, normalise_frontmatter clamps
# format="roundup" → "essay". See blog/blog_streams.py and
# docs/RESEARCH_ROUNDUP_LANE.md.
_FORMAT_ALIASES = {
    "roundup": "essay",
}
# Tags (applied before the allowed-set check; slugified first):
_TAG_ALIASES = {
    "evaluation": "evals",
    "evaluations": "evals",
    "eval": "evals",
    "ai-agents": "agents",
    "agent": "agents",
    "llm": "ai",
    "llms": "ai",
    "cost": "cost-optimisation",
    "cost-optimization": "cost-optimisation",
    "optimisation": "cost-optimisation",
    "pm": "product-management",
    "product": "product-management",
    "prompt-engineer": "prompt-engineering",
    "prompts": "prompt",
    "context": "context-engineering",
    "human-in-loop": "human-in-the-loop",
    "reliability-engineering": "reliability",
    "infra": "infrastructure",
}


def _extract_ts_array(ts: str, marker: str, terminator: str) -> set[str]:
    """Extract quoted strings from a TS array literal starting at `marker`."""
    start = ts.find(marker)
    if start == -1:
        return set()
    end = ts.find(terminator, start)
    if end == -1:
        return set()
    block = ts[start:end]
    return set(re.findall(r"['\"]([^'\"]+)['\"]", block))


def _extract_enum(ts: str, field: str) -> set[str]:
    """Extract values from `field: z.enum([...])` in the schema."""
    m = re.search(rf"{field}:\s*z\.enum\(\[(.*?)\]\)", ts, re.DOTALL)
    if not m:
        return set()
    return set(re.findall(r"['\"]([^'\"]+)['\"]", m.group(1)))


@lru_cache(maxsize=4)
def load_contract(repo: Optional[str] = None) -> dict:
    """Load allowed enum/tag sets from content.config.ts (cached per repo).

    Falls back to the hardcoded sets if the file is missing/unparseable.
    Returns {tags, sources, tiers, formats} as frozensets.
    """
    repo_path = Path(repo) if repo else Path(SAHILBLOG_REPO)
    config_ts = repo_path / "src" / "content.config.ts"
    tags: set[str] = set()
    sources: set[str] = set()
    tiers: set[str] = set()
    formats: set[str] = set()
    try:
        ts = config_ts.read_text(encoding="utf-8", errors="replace")
        tags = _extract_ts_array(ts, "ALLOWED_TAGS = [", "] as const")
        sources = _extract_enum(ts, "source")
        tiers = _extract_enum(ts, "tier")
        formats = _extract_enum(ts, "format")
    except OSError:
        pass

    return {
        "tags": frozenset(tags),
        "sources": frozenset(sources or _FALLBACK_SOURCES),
        "tiers": frozenset(tiers or _FALLBACK_TIERS),
        "formats": frozenset(formats or _FALLBACK_FORMATS),
    }


def _normalise_scalar(value: str, allowed: frozenset, aliases: dict,
                      default: str) -> str:
    """Alias-map then clamp a scalar to the allowed set, else default."""
    v = (value or "").strip().strip('"').strip("'")
    if v in allowed:
        return v
    mapped = aliases.get(v.lower())
    if mapped and mapped in allowed:
        return mapped
    return default


def normalise_tags(tags, allowed: frozenset) -> tuple[list[str], list[str]]:
    """Slugify + alias-map tags, keep those in the allowed set, dedupe.

    Returns (kept_tags, dropped_raw) — dropped are logged by the caller.
    Empty allowed set means "schema unreadable": pass tags through untouched.
    """
    if not isinstance(tags, (list, tuple)):
        return [], []
    kept: list[str] = []
    dropped: list[str] = []
    seen: set[str] = set()
    for raw in tags:
        if not isinstance(raw, str) or not raw.strip():
            continue
        slug = slugify(raw.strip())
        mapped = _TAG_ALIASES.get(slug, slug)
        if not allowed:  # schema unreadable — don't silently drop everything
            if mapped not in seen:
                kept.append(mapped)
                seen.add(mapped)
            continue
        if mapped in allowed and mapped not in seen:
            kept.append(mapped)
            seen.add(mapped)
        elif mapped not in allowed:
            dropped.append(raw)
    return kept, dropped


def normalise_frontmatter(fm: dict, repo: Optional[str] = None) -> dict:
    """Return a copy of `fm` with source/tier/format/tags clamped to the schema.

    This is the single guarantee that the assembler never emits an entry that
    fails Astro schema validation. Logs any dropped tags.
    """
    contract = load_contract(repo)
    out = dict(fm)

    out["source"] = _normalise_scalar(
        str(fm.get("source", "")), contract["sources"], _SOURCE_ALIASES, "manual")
    # Tier aliasing is needed for the research stream (tier="research" →
    # "pm" until SahilBlog's schema is updated). Source aliasing already
    # covers "curated-roundup" → "manual". See docs/RESEARCH_ROUNDUP_LANE.md.
    out["tier"] = _normalise_scalar(
        str(fm.get("tier", "")), contract["tiers"], _TIER_ALIASES, "pm")
    out["format"] = _normalise_scalar(
        str(fm.get("format", "")), contract["formats"], _FORMAT_ALIASES, "essay")

    kept, dropped = normalise_tags(fm.get("tags", []), contract["tags"])
    out["tags"] = kept
    if dropped:
        print(f"[schema_contract] dropped out-of-vocab tags {dropped} "
              f"(kept {kept})")

    return out
