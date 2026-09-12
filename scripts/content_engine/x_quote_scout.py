#!/usr/bin/env python3
"""Approval-only X co-manage scout — reads Sahil's timeline, finds the moves.

Sources (registry-driven, no X API cost — logged-in Playwright scrape):
  1. Home timeline  (his actual feed: following graph as X curated it)
  2. Mentions       (reply-eligible, low-hanging)
  3. Registry extras (data/x_registry.json — Sahil-editable account list)

Freshness: registry-driven (default 6h hard window via snowflake age).
Voice: corpus-calibrated prompt + deterministic voice gate per draft.
Argument packs are reserved for standalone opinions (the blog-cross-ref
lane); replies and quotes are gated on the voice rejection test instead.
"""
import json
import os
import sys
import contextlib
import io
from datetime import datetime, timezone
from pathlib import Path

CE = Path('/home/kensei/repos/KenseiAgent/content_engine')
sys.path.insert(0, str(CE))
sys.path.insert(0, str(Path.home() / '.hermes' / 'scripts'))  # x_manager_report

import x_ingest
from x_ingest import load_registry, apply_freshness, dedupe_by_id, tweet_age
from llm_generate import _call_llm_chain
import x_manager as xm
from x_manager_report import render_report
from x_voice_gate import voice_gate_issues, blog_cross_reference

REGISTRY = load_registry()
FRESHNESS = REGISTRY["freshness_hours"]
MAX_CANDIDATES = int(REGISTRY.get("max_candidates_per_run", 40))
MIN_TWEET_CHARS = int(REGISTRY.get("min_tweet_chars", 40))
MAX_REPLIES_PER_RUN = 2   # replies are the etiquette-risky lane: keep them rare
MAX_QUOTES_PER_RUN = 4

CANDIDATES_JSON = CE / 'data' / 'x_scout_candidates.json'
STATE_FILE = CE / 'data' / 'x_scout_last_run.json'
VERDICTS = ("reply", "quote", "standalone", "discard")
VERDICT_FILE = CE / 'data' / 'x_scout_verdicts.json'
STANDALONE_SEEDS = CE / 'data' / 'x_standalone_seeds.json'
VOICE_SKILL = Path('/home/kensei/.hermes/skills/social-media/sahil-twitter-voice/SKILL.md')

LLM_SYSTEM = (
    "You are drafting for Sahil's own X account. He is an indie builder (AI "
    "agents, Hermes, UK groceries app Plenishd, football) who posts in a "
    "specific voice — you will be given his calibrated voice rules below. "
    "You are reading ONE post from his timeline or mentions. Decide the "
    "strongest move and draft it ONLY if it would sound like him typing it.\n"
    "Verdicts:\n"
    "  reply      — enter the conversation (add, challenge, reframe, react).\n"
    "  quote      — the source is raw material for a quote post in his voice.\n"
    "  standalone — the subject deserves its own broader post later; seed it.\n"
    "  discard    — nothing unique to add; a popular post is not a reason to engage.\n"
    "Rules: reply only when you can add, challenge or reframe — a popular post "
    "is not a reason to reply. Prefer discard when in doubt; the account gets "
    "stronger when selective. Do NOT write an analyst review: no 'the real "
    "test is whether…', no 'in practice…', no feature-spec language. One "
    "natural move: genuine excitement, playful observation, sharp "
    "disagreement, personal build comparison, or a curious challenge. A "
    "perfect one-liner beats a padded paragraph. Never invent first-hand "
    "experience he hasn't shown you. Never use hashtags, em-dashes, emoji, "
    "engagement bait or generic praise. Do not reveal credentials or private "
    "data.\n"
    "Reply with exactly one JSON object: "
    '{"verdict": "reply|quote|standalone|discard", "reason": "...", '
    '"post": "...", "stance": "..."}\n'
    "For standalone, also include claim/evidence/mechanism/position (the "
    "thesis, for the incubator). For reply/quote: post is the full draft "
    "(25-280 chars), stance is a one-line summary of the position the draft "
    "takes, claim/evidence/mechanism may be empty. For discard, post may be "
    "empty and reason explains why there is no unique angle."
)


def _runtime_voice() -> str:
    """Load only the corpus-calibrated section from the voice skill."""
    try:
        text = VOICE_SKILL.read_text()
        marker = '## Corpus-Calibrated Runtime Voice'
        start = text.index(marker)
        return text[start:start + 7000]
    except Exception:
        return ''


def _load_env():
    env = Path.home() / '.hermes' / '.env'
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            key, _, value = line.partition('=')
            os.environ.setdefault(key.strip(), value.strip())


def _collect() -> list[dict]:
    """Scrape home + mentions + registry extras, freshness-filter, dedupe."""
    rows = x_ingest.ingest(
        include_home=True,
        include_mentions=True,
        include_following=bool(os.environ.get('X_SCOUT_FOLLOWING')),  # phase 2 flag
    )
    # Mentions are reply-eligible: keep them even if slightly outside the
    # reply window, because a mention IS a conversation already started.
    rows = dedupe_by_id(rows)
    fresh = apply_freshness(rows, FRESHNESS.get("quote", 6))
    # Rank: mentions first (reply-eligible), then by recency.
    def sort_key(r):
        return (0 if r.get("source") == "mention" else 1,
                r.get("age_hours") if r.get("age_hours") is not None else 999)
    fresh.sort(key=sort_key)
    CANDIDATES_JSON.parent.mkdir(parents=True, exist_ok=True)
    CANDIDATES_JSON.write_text(json.dumps(fresh, ensure_ascii=False))
    return fresh[:MAX_CANDIDATES]


def _load_verdicts() -> dict:
    try:
        data = json.loads(VERDICT_FILE.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _record_verdict(tweet: dict, verdict: str, reason: str) -> None:
    data = _load_verdicts()
    entries = data.get("ids", [])
    entries.append({
        "id": tweet.get("id", ""),
        "author": tweet.get("author", ""),
        "source": tweet.get("source", ""),
        "verdict": verdict,
        "reason": (reason or "")[:300],
        "at": datetime.now(timezone.utc).isoformat(),
    })
    data["ids"] = entries[-200:]
    VERDICT_FILE.parent.mkdir(parents=True, exist_ok=True)
    VERDICT_FILE.write_text(json.dumps(data, ensure_ascii=False))


def _draft(tweet: dict) -> dict:
    """LLM verdict + draft for one tweet."""
    system = LLM_SYSTEM + "\n\n" + _runtime_voice()
    src = (tweet.get('text') or '')[:2000]
    head = f"Source: @{tweet.get('author','?')} ({tweet.get('source','?')})"
    if tweet.get("source") == "mention":
        head += " — this post mentions/mentions Sahil or is in his timeline thread"
    out = _call_llm_chain(system, head + "\n\nPOST:\n" + src,
                          timeout=90, max_tokens=4000)
    if not out:
        raise RuntimeError('empty LLM output')
    text = out.strip()
    start, end = text.find('{'), text.rfind('}')
    if start == -1 or end <= start:
        raise RuntimeError('no JSON object in LLM output')
    return json.loads(text[start:end + 1])


def _candidate_artifacts(rows):
    """Return (staged_artifacts, seeds, discards)."""
    artifacts: list = []
    seeds: list = []
    discards: list = []
    reply_count = quote_count = 0
    for tweet in rows:
        text = (tweet.get('text') or '').strip()
        if len(text) < MIN_TWEET_CHARS:
            continue
        try:
            data = _draft(tweet)
        except Exception as exc:
            print(f"[x-scout] draft skip: {exc}", file=sys.stderr)
            continue
        verdict = (data.get('verdict') or 'discard').strip().lower()
        if verdict not in VERDICTS:
            verdict = 'discard'
        reason = (data.get('reason') or '').strip()
        _record_verdict(tweet, verdict, reason)

        if verdict == 'standalone':
            seeds.append({
                "source_id": tweet.get("id", ""),
                "author": tweet.get("author", ""),
                "source_url": tweet.get("url", ""),
                "source_text": text[:600],
                "claim": (data.get('claim') or '').strip(),
                "evidence": (data.get('evidence') or '').strip(),
                "mechanism": (data.get('mechanism') or '').strip(),
                "position": (data.get('position') or '').strip(),
                "reason": reason,
                "blog_refs": blog_cross_reference(
                    (data.get('position') or '') + ' ' + (data.get('claim') or '')),
            })
            continue
        if verdict == 'discard':
            discards.append({"id": tweet.get('id', ''), "reason": reason})
            continue

        post = (data.get('post') or '').strip()
        if not post or post.casefold() == text.casefold():
            continue
        # ── deterministic voice gate (the corpus rejection test, mechanical) ──
        issues = voice_gate_issues(post)
        if len(issues) > 2:
            print(f"[x-scout] voice gate fail ({len(issues)}): {issues[:3]}",
                  file=sys.stderr)
            continue
        stance = (data.get('stance') or reason or post[:120]).strip()
        # Pack for the artifact record: stance carries the position; the
        # LLM is not forced to write pack prose for replies/quotes.
        pack = xm.ArgumentPack(
            claim=stance,
            evidence=tweet.get('url', ''),
            mechanism=(data.get('mechanism') or reason or stance).strip(),
            position=stance,
            context={
                'tweet_id': str(tweet.get('id', '')),
                'author': str(tweet.get('author', '')),
                'source': tweet.get('source', ''),
                'source_url': tweet.get('url', ''),
                'source_text': text[:500],
                'age_hours': tweet.get('age_hours'),
                'voice_issues': issues,  # minor issues kept for the review
                'blog_refs': [] if verdict != 'quote' else blog_cross_reference(post),
            },
        )
        if not pack.is_complete():
            continue
        try:
            if verdict == 'reply':
                if reply_count >= MAX_REPLIES_PER_RUN:
                    continue
                artifacts.append(xm.reply_draft_artifact(
                    tweet_id=str(tweet.get('id', '')),
                    author=str(tweet.get('author', '')),
                    source_text=text,
                    body=post[:280],
                    pack=pack,
                ))
                reply_count += 1
            else:
                if quote_count >= MAX_QUOTES_PER_RUN:
                    continue
                artifacts.append(xm.XArtifact(
                    id=xm._new_id(xm.LANE_QUOTE_SCAN),
                    lane=xm.LANE_QUOTE_SCAN,
                    brand='sahil_twitter',
                    body=post[:280],
                    pack=pack,
                ))
                quote_count += 1
        except Exception as exc:
            print(f"[x-scout] build skip: {exc}", file=sys.stderr)
    return artifacts, seeds, discards


def _merge_standalone_seeds(seeds: list) -> None:
    try:
        existing = json.loads(STANDALONE_SEEDS.read_text())
        if not isinstance(existing, list):
            existing = []
    except Exception:
        existing = []
    known = {s.get("source_id") for s in existing}
    fresh = [s for s in seeds if s.get("source_id") not in known]
    if fresh:
        STANDALONE_SEEDS.parent.mkdir(parents=True, exist_ok=True)
        STANDALONE_SEEDS.write_text(json.dumps(existing + fresh, ensure_ascii=False))


STATE_FILE2 = STATE_FILE


def _already_reported(artifacts) -> bool:
    try:
        state = json.loads(STATE_FILE2.read_text())
        return any(a.id in state.get('ids', []) for a in artifacts)
    except Exception:
        return False


def _record_reported(artifacts) -> None:
    ids = [a.id for a in artifacts]
    try:
        state = json.loads(STATE_FILE2.read_text())
    except Exception:
        state = {'ids': []}
    state.setdefault('ids', [])
    state['ids'] = list(dict.fromkeys(state['ids'] + ids))[-40:]
    STATE_FILE2.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE2.write_text(json.dumps(state))


def main():
    _load_env()
    rows = _collect()
    if not rows:
        return
    artifacts, seeds, discards = _candidate_artifacts(rows)
    _merge_standalone_seeds(seeds)
    if len(artifacts) < xm.QUOTE_SCAN_MIN:
        print(f"[x-scout] only {len(artifacts)} valid drafts (floor {xm.QUOTE_SCAN_MIN})",
              file=sys.stderr)
        return
    if _already_reported(artifacts):
        return
    staged = []
    for art in artifacts:
        try:
            xm.stage_for_approval(art)
            staged.append(art)
        except Exception as exc:
            print(f"[x-scout] stage failed: {exc}", file=sys.stderr)
    if not staged:
        return
    report = render_report(staged, lane='quote-scout',
                           title='Quote-post recommendations')
    _record_reported(staged)
    print(f"X Manager · {len(staged)} post recommendations (replies + quotes)")
    print("Original posts and recommended drafts are in the attached review.")
    print(f"MEDIA:{report}")


if __name__ == "__main__":
    main()
