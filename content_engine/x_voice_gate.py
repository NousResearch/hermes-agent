"""Deterministic voice gate for Sahil's X drafts.

Enforces the corpus-calibrated voice skill without an LLM: mechanical
checks that run on every draft before it reaches the review HTML. The LLM
still carries the taste; this module catches the mechanical failures that
make output feel "entirely off": AI-isms, generic praise, analyst cadence,
engagement bait.

Sources:
  - skills/social-media/sahil-twitter-voice/SKILL.md
    ("Corpus-Calibrated Runtime Voice" — rejection test + banned phrases)
  - corpus shape: median 23 words, no hashtags, no emoji stacks
"""
from __future__ import annotations

import re

# Banned analyst-tell phrases (from the voice skill, verbatim).
ANALYST_TELLS = (
    "the real test is whether",
    "what does this mean in practice",
    "i'd like to see",
    "show me the workflow",
    "before you scale",
    "dive into",
    "let's dive in",
    "it's not just",
    "here's the thing",
    "game changer",
    "game-changer",
    "unlock the",
    "unleash",
    "delve",
    "delve into",
    "elevate",
    "seamless",
    "seamlessly",
    "leverage",
    "leverages",
    "leverage this",
    "in today's",
    "in the world of",
    "embark on",
    "embarks",
    "navigate the",
    "cutting-edge",
    "cutting edge",
    "revolutionize",
    "revolutionise",
    "paradigm",
    "synergy",
    "synergies",
    "harness the power",
    "in conclusion",
    "as we've seen",
    "take a deeper look",
    "understand the",
    "at the end of the day",
    "the bottom line",
)

# Generic praise / empty reactions.
GENERIC_PRAISE = (
    "great point",
    "great take",
    "love this",
    "this is huge",
    "well said",
    "couldn't agree more",
    "so true",
    "facts.",
    "no notes",
)

# Emojis are absent from the corpus (standard social-media emoji stacks).
_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U0001F900-\U0001F9FF\u2600-\u27BF\u2B00-\u2BFF]"
)
_HASHTAG_RE = re.compile(r"(^|\s)#\w+")
# Em-dashes: the corpus avoids them.
EMDASH_RE = re.compile(r"\u2014|\u2013")


def voice_gate_issues(draft: str) -> list[str]:
    """Return a list of human-readable issues (empty = clean)."""
    issues: list[str] = []
    text = (draft or "").strip()
    if not text:
        return ["empty draft"]
    low = text.lower()

    if len(text.split()) > 280:
        pass  # soft cap, reported by caller
    for tell in ANALYST_TELLS:
        if tell in low:
            issues.append(f"analyst tell: '{tell}'")
    for p in GENERIC_PRAISE:
        if re.search(rf"(?<![\w]){re.escape(p)}(?!\w)", low):
            issues.append(f"generic praise: '{p}'")
    if _HASHTAG_RE.search(text):
        issues.append("hashtag used (corpus uses none)")
    if _EMOJI_RE.search(text):
        issues.append("emoji used (corpus uses none)")
    if EMDASH_RE.search(text):
        issues.append("em/en dash (corpus uses commas and periods)")
    if text.count("?") >= 2:
        issues.append("more than one question mark")
    # Sentence-initial capitalised emphasis is fine; ALL-CAPS words are not.
    if re.search(r"\b[A-Z]{4,}\b", text.replace("LLM", "").replace("API", "")):
        issues.append("ALL-CAPS word (use normal casing)")
    # Consultant cadence: more than three 3+word clauses chained with 'and'
    clauses = [c.strip() for c in re.split(r"[.;!?\n]", text) if len(c.split()) >= 3]
    if len([c for c in clauses if c.count(" and ") >= 1]) >= 2:
        issues.append("chained clauses — fragment it, Sahil types short")
    # Word repetition — "and on and on" filler is a classic AI tell.
    words = [w for w in re.sub(r"[^a-z0-9 ]", " ", text.lower()).split() if w]
    if len(words) >= 12:
        from collections import Counter
        for word, count in Counter(words).most_common(3):
            if word in ("the", "and", "that", "with"):
                continue
            if count >= max(5, len(words) // 5):
                issues.append(f"repetition tell: '{word}' x{count}")
                break
    if len(text.split()) > 60:
        issues.append("over 60 words — median public post is 23; cut it")
    # First-person check for opinion posts is caller's job (context-dependent).
    return issues


def voice_gate_pass(draft: str, max_issues: int = 2) -> bool:
    """A draft passes when it has at most ``max_issues`` minor issues.

    Hashtags, praise and analyst tells are hard fails (0 tolerance each is
    overkill — the corpus has a few organic 'great' moments) so we allow a
    small budget but any draft with 3+ issues is structurally off-voice.
    """
    return len(voice_gate_issues(draft)) <= max_issues


# ── SahilBlog cross-reference (opinion posts only) ──────────────────────

BLOG_IDEA_PATHS = (
    "/home/kensei/.hermes/research/idea-backlog.jsonl",
)


def _load_blog_ideas(limit: int = 30) -> list[dict]:
    import json
    from pathlib import Path

    out: list[dict] = []
    for p in BLOG_IDEA_PATHS:
        try:
            for line in Path(p).read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
        except FileNotFoundError:
            continue
    return out[-limit:]


def _tokens(s: str) -> set[str]:
    return {t for t in re.sub(r"[^a-z0-9 ]", " ", s.lower()).split() if len(t) > 3}


def blog_cross_reference(draft: str, top_n: int = 3) -> list[dict]:
    """Return blog ideas whose content overlaps the draft's topic.

    Used on independent-opinion posts so the review can show "this feeds
    blog idea X" — compounding the two channels.
    """
    draft_tokens = _tokens(draft)
    if len(draft_tokens) < 4:
        return []
    scored: list[tuple[int, dict]] = []
    for idea in _load_blog_ideas():
        blob = f"{idea.get('title', '')} {idea.get('concept', '')}"
        idea_tokens = _tokens(blob)
        overlap = len(draft_tokens & idea_tokens)
        if overlap >= 3:
            scored.append((overlap, idea))
    scored.sort(key=lambda x: -x[0])
    return [
        {
            "id": i.get("id", ""),
            "title": i.get("title", "")[:120],
            "overlap": n,
        }
        for n, i in scored[:top_n]
    ]
