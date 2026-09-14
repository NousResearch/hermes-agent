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


def voice_gate_issues(draft: str, *, evidence: list[dict] | None = None) -> list[str]:
    """Return a list of human-readable issues (empty = clean)."""
    issues: list[str] = []
    text = (draft or "").strip()
    if not text:
        return ["empty draft"]
    low = text.lower().replace("’", "'")
    if re.search(r"\b(?:rt|like|repost|share|comment|follow)\s+(?:if|for|yes|below|me)|\bagree\?", low):
        issues.append("engagement bait")
    # Explicit autobiographical claims require approved evidence. Conversational
    # questions/proposals are not automatically claims of past experience.
    # This mechanical screen is not semantic fact verification; human approval
    # remains mandatory and generation must never invent personal history.
    fragments = re.split(r'(?<=[.!?])\s+|\n+', low)
    approved_sentences = {
        sentence.strip()
        for item in (evidence or []) if isinstance(item, dict)
        and item.get('approved') is True and item.get('url') and item.get('provenance')
        for sentence in re.split(r'(?<=[.!?])\s+|\n+', str(item.get('text', '')).lower().replace('’', "'"))
    }
    for claim in fragments:
        # Remove only the conversational subject, not the rest of the sentence.
        # 'I think I shipped...' must still expose its autobiographical claim.
        screened = re.sub(r"^(?:i|we)\s+(?:think|wonder|suggest|would|could|should)\b", "", claim.strip())
        if claim.rstrip().endswith('?'):
            screened = re.sub(r"^(?:can|could|should|would)\s+(?:i|we)\b", "", screened)
        # First person alone is not a claim of experience. Opinions, desires and
        # proposals are allowed; witnessed implementation/outcome claims need evidence.
        verbs = r'(?:built|shipped|tested|used|ran|run|tried|saw|found|learned|spent|saved|deployed|measured|worked|implemented|migrated|fixed|wrote|documented|doubled|tripled|rewritten|replaced)'
        personal = re.search(r"\b(?:i|we)(?:'ve| have| had)?(?:\s+\w+){0,3}\s+" + verbs + r'\b', screened)
        possessive = re.search(r"\b(?:my|our|mine)\b.{0,80}\b" + verbs + r'\b', screened)
        implied = re.match(r"(?:(?:just|recently)\s+)?(?:shipped|built|tested|used|spent|saved|fixed|deployed|tried|migrated)\b", claim)
        if (personal or possessive or implied) and claim.strip() not in approved_sentences:
            issues.append('unsupported first-hand experience; supply approved own evidence')

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
    # Selective caps and repeated punctuation are present in the actual corpus.
    # Catch witnessed rejected constructions, not their expressive punctuation.
    # This is a narrow regression screen, not semantic detection of all AI slop.
    compact = ' '.join(low.split())
    for rejected in (
        'interested in how they handle that part',
        'this setup just saved the wrong lesson and confidently repeated it',
    ):
        if rejected in compact:
            issues.append(f"structural slop: user-rejected construction '{rejected}'")
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


def voice_gate_pass(draft: str, max_issues: int = 0, *, evidence=None) -> bool:
    """Every detected issue rejects; legacy budgets cannot weaken the gate."""
    return not voice_gate_issues(draft, evidence=evidence)


def load_voice_corpus(path=None, *, account: str = "Sahil_Saghir", limit: int = 30,
                      approved_sha256: str | None = None, now=None) -> list[dict]:
    """Read a hash-approved local export, never substitute illustrative posts.

    Approval is a caller-supplied reviewed file digest, or a sibling
    `<file>.approval.json` containing approved:true and sha256. Missing approval
    returns no evidence. URL authorship must match, including reply permalinks.
    """
    import hashlib
    import json
    from pathlib import Path
    from urllib.parse import urlsplit
    from hermes_constants import get_hermes_home
    if __package__:
        from .x_ingest import normalize_source, _utc_time
    else:
        from x_ingest import normalize_source, _utc_time

    # Runtime references come from current observed OWN publications, not the
    # deleted legacy archive. Explicit paths retain the old import validator for
    # migration tooling only; runtime never falls back to that archive.
    living = path is None
    import os
    path = Path(path) if path is not None else Path(os.environ.get('X_OBSERVATIONS_PATH', str(get_hermes_home() / 'data/x-analytics/observations.json')))
    if not re.fullmatch(r"[A-Za-z0-9_]{1,15}", account) or limit <= 0:
        return []
    try:
        if path.stat().st_size > 10_000_000:
            return []
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if not living:
            if approved_sha256 is None:
                approval = json.loads(Path(str(path) + '.approval.json').read_text())
                if not isinstance(approval, dict) or approval.get('approved') is not True:
                    return []
                approved_sha256 = approval.get('sha256')
            if digest != approved_sha256:
                return []
        if path.suffix == ".jsonl":
            records = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
        else:
            records = json.loads(raw)
        if isinstance(records, dict):
            records = records.get("all", records.get("posts", []))
        if not isinstance(records, list):
            return []
    except (OSError, ValueError, UnicodeError):
        return []
    out, seen = [], set()
    if living:
        records = sorted((r for r in records if isinstance(r, dict)), key=lambda r: str(r.get('observed_at', '')), reverse=True)
        latest = {}
        for record in records:
            # A newer incomplete observation must not revive an older version.
            identity = str(record.get('id') or record.get('url') or '')
            if identity and identity not in latest:
                latest[identity] = record
        records = list(latest.values())
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("text"), str) or not record["text"].strip():
            continue
        observed = None
        if living:
            provenance = record.get('provenance')
            observed = _utc_time(record.get('observed_at'))
            import time
            if observed is not None and observed.timestamp() > (time.time() if now is None else now):
                continue
            if (not isinstance(provenance, dict) or provenance.get('kind') != 'browser'
                    or observed is None or str(record.get('author', '')).lstrip('@').lower() != account.lower()):
                continue
            # Text with explicit truncation cannot calibrate the author's ending.
            if record.get('truncated') is True:
                continue
        url = record.get("url")
        if not isinstance(url, str):
            continue
        if not re.fullmatch(rf"/{re.escape(account)}/status/[0-9]+", urlsplit(url).path, re.I):
            continue
        author = record.get("author", account)
        if not isinstance(author, str) or author.casefold() != account.casefold():
            continue
        timestamp = record.get("created_at") or record.get("timestamp")
        if _utc_time(timestamp) is None:
            continue
        source = normalize_source({**record, "created_at": timestamp}, origin="own", now=now)
        if source is None or source["age_hours"] < 0 or source["id"] in seen:
            continue
        if living:
            published = _utc_time(source['created_at'])
            if observed is None or published is None or observed < published:
                continue
        seen.add(source["id"])
        out.append({**source, "author": account, "approved": True,
                    "provenance": {"path": str(path), "sha256": digest,
                                   "kind": 'observed_published_post' if living else 'approved_local_export'}})
    out.sort(key=lambda item: item["created_at"], reverse=True)
    return out[:min(limit, 1000)]


# ── SahilBlog cross-reference (opinion posts only) ──────────────────────

# Optional injected paths for offline tests; normal runtime is profile-scoped.
BLOG_IDEA_PATHS = ()


def _load_blog_ideas(limit: int = 30) -> list[dict]:
    import json
    from pathlib import Path

    from hermes_constants import get_hermes_home
    out: list[dict] = []
    for p in BLOG_IDEA_PATHS or (get_hermes_home() / "research/idea-backlog.jsonl",):
        try:
            for line in Path(p).read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    if isinstance(item, dict):
                        out.append(item)
                except Exception:
                    continue
        except FileNotFoundError:
            continue
    return out[-limit:]


def _tokens(s: str) -> set[str]:
    stop = {"this", "that", "with", "from", "have", "will", "where", "what", "when", "about", "their", "there", "which", "into", "more", "than", "just", "your", "been", "would", "could", "should", "actually"}
    return {t for t in re.sub(r"[^a-z0-9 ]", " ", s.lower()).split() if len(t) > 3 and t not in stop}


def blog_cross_reference(draft: str, top_n: int = 3) -> list[dict]:
    """Return blog ideas whose content overlaps the draft's topic.

    Used on independent-opinion posts so the review can show "this feeds
    blog idea X" — compounding the two channels.
    """
    draft_tokens = _tokens(draft)
    if len(draft_tokens) < 4:
        return []
    scored: list[tuple[int, dict]] = []
    from urllib.parse import urlsplit
    for idea in _load_blog_ideas():
        url, evidence = idea.get("url", ""), idea.get("evidence", "")
        if not isinstance(url, str) or not isinstance(evidence, str) or not evidence.strip():
            continue
        parsed = urlsplit(url)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username:
            continue
        # Topic overlap alone is not a reference: the supplied supporting
        # excerpt must itself discuss the draft's subject.
        if len(draft_tokens & _tokens(evidence)) < 2:
            continue
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
            "url": i["url"],
            "evidence": i["evidence"],
        }
        for n, i in scored[:top_n]
    ]
