#!/usr/bin/env python3
"""Approval-only X quote-post scout — deterministic filtering stage.

Stages pending artifacts through x_manager with LLM draft generation and
discord cards. This stage uses the already-live session cookies through the
local Playwright reader and never posts, likes, or writes to X.
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

from engagement_x_poster import fetch_tweets_batch
from llm_generate import _call_llm_chain
import x_manager as xm
from x_manager_report import render_report

TARGET_ACCOUNTS = [
    "NousResearch", "hermesagent", "teknium", "karpathy", "swyx",
    "rauchg", "levelsio", "tahseen_rahman", "AnthropicAI", "openai",
]
MAX_PER_ACCOUNT = 8
MIN_TWEET_CHARS = 60
BAIT_WORDS = ("rt if", "retweet if", "like if", "follow for", "tag someone", "vote if", "#ad", "#sponsored")
CANDIDATES_JSON = Path('/home/kensei/repos/KenseiAgent/content_engine/data/x_scout_candidates.json')
VOICE_SKILL = Path('/home/kensei/.hermes/skills/social-media/sahil-twitter-voice/SKILL.md')

VERDICTS = ("reply", "quote", "standalone", "discard")
VERDICT_FILE = Path('/home/kensei/repos/KenseiAgent/content_engine/data/x_scout_verdicts.json')
STANDALONE_SEEDS = Path('/home/kensei/repos/KenseiAgent/content_engine/data/x_standalone_seeds.json')

LLM_SYSTEM = (
    "You evaluate one X post for Sahil and decide the strongest move. "
    "Verdicts:\n"
    "  reply      — enter the author's conversation (add, challenge, reframe).\n"
    "  quote      — the source is good raw material for your own standalone-ish "
    "quote post.\n"
    "  standalone — the subject deserves a broader argument as its own post;\n"
    "               you will NOT draft it here.\n"
    "  discard    — nothing unique to add; a popular post is not a reason to "
    "reply.\n"
    "Never reply merely because the post is popular. Reply only when you can add, "
    "challenge or reframe. Prefer discard when in doubt — the account gets "
    "stronger when selective.\n"
    "For reply/quote, craft the post in Sahil's real voice, not a generic "
    "analyst or product manager voice. Choose the natural response mode: genuine "
    "excitement, playful observation, sharp disagreement, personal build "
    "comparison, or curious challenge. Do not force a lesson or critique when "
    "the source only earns a short reaction.\n"
    "Reply with exactly one JSON object: "
    '{"verdict": "reply|quote|standalone|discard", "reason": "...", '
    '"claim": "...", "evidence": "...", "mechanism": "...", "weak_practice": "...", '
    '"position": "...", "post": "..."} '
    "For discard: reason is why there is no unique angle; leave claim/evidence/"
    "mechanism/position/post empty or minimal. For standalone: fill claim/"
    "evidence/mechanism/position as a seed for the idea, post may be empty. "
    "The post may be 25-280 characters and must be distinct from the source. "
    "No em-dashes, hashtags, engagement bait, invented experience or generic "
    "praise. Weak practice may be 'none; constructive explanation' when nothing "
    "is being challenged. Do not reveal credentials or private data."
)


def _runtime_voice() -> str:
    """Load only the compact corpus-calibrated section from the voice skill."""
    try:
        text = VOICE_SKILL.read_text()
        marker = '## Corpus-Calibrated Runtime Voice'
        start = text.index(marker)
        return text[start:start + 7000]
    except Exception:
        return ''


def _load_env():
    env = Path('/home/kensei/.hermes/.env')
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            key, _, value = line.partition('=')
            os.environ.setdefault(key.strip(), value.strip())


def _bait(text: str) -> bool:
    low = (text or '').lower()
    return any(w in low for w in BAIT_WORDS)


def _collect() -> list:
    rows = []
    for account in TARGET_ACCOUNTS:
        try:
            # The browser collector is intentionally chatty. Keep those logs
            # local so no_agent stdout remains a clean Discord summary.
            with contextlib.redirect_stdout(io.StringIO()):
                batch = fetch_tweets_batch([account], limit=MAX_PER_ACCOUNT)
            rows.extend(batch.get(account, []))
        except Exception as exc:
            print(f"[x-quote-scout] fetch {account}: {exc}", file=sys.stderr)
    CANDIDATES_JSON.parent.mkdir(parents=True, exist_ok=True)
    CANDIDATES_JSON.write_text(json.dumps(rows, ensure_ascii=False))
    return rows


def _load_verdicts() -> dict:
    try:
        data = json.loads(VERDICT_FILE.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_verdicts(data: dict) -> None:
    VERDICT_FILE.parent.mkdir(parents=True, exist_ok=True)
    VERDICT_FILE.write_text(json.dumps(data, ensure_ascii=False))


def _load_standalone_seeds() -> list:
    try:
        data = json.loads(STANDALONE_SEEDS.read_text())
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_standalone_seeds(seeds: list) -> None:
    STANDALONE_SEEDS.parent.mkdir(parents=True, exist_ok=True)
    STANDALONE_SEEDS.write_text(json.dumps(seeds, ensure_ascii=False))


def _record_verdict(tweet: dict, verdict: str, reason: str) -> None:
    data = _load_verdicts()
    ids = data.setdefault("ids", [])
    entry = {
        "id": tweet.get("id", ""),
        "author": tweet.get("author", ""),
        "verdict": verdict,
        "reason": (reason or "")[:300],
        "at": datetime.now(timezone.utc).isoformat(),
    }
    ids.append(entry)
    # Keep only the most recent 200 verdicts for the health metric.
    data["ids"] = ids[-200:]
    _save_verdicts(data)


def _verdict_health() -> dict:
    data = _load_verdicts()
    ids = data.get("ids", [])
    counts: dict[str, int] = {}
    for entry in ids:
        v = entry.get("verdict", "discard")
        counts[v] = counts.get(v, 0) + 1
    total = sum(counts.values()) or 1
    return {
        "total": len(ids),
        "counts": counts,
        "discard_rate": round(counts.get("discard", 0) / total, 3),
    }


def _draft(tweet: dict) -> dict:
    """LLM draft; returns verdict fields plus post or raises."""
    system = LLM_SYSTEM + "\n\n" + _runtime_voice()
    out = _call_llm_chain(system, (tweet.get('text') or '')[:2000],
                          timeout=90, max_tokens=4000)
    if not out:
        raise RuntimeError('empty LLM output')
    text = out.strip()
    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1 or end <= start:
        raise RuntimeError('no JSON object in LLM output')
    data = json.loads(text[start:end + 1])
    return data


def _candidate_artifacts(rows) -> tuple[list, list, list]:
    """Return (staged_artifacts, standalone_seeds, verdict_summary).

    Verdict-aware: only reply/quote produce staged artifacts; standalone is
    recorded as a seed for the thesis incubator; discard is recorded for the
    health metric.
    """
    artifacts: list = []
    seeds: list = []
    discards: list = []
    for tweet in rows:
        text = (tweet.get('text') or '').strip()
        if len(text) < MIN_TWEET_CHARS or _bait(text):
            continue
        try:
            data = _draft(tweet)
        except Exception as exc:
            print(f"[x-quote-scout] draft skip: {exc}", file=sys.stderr)
            continue
        verdict = (data.get('verdict') or 'discard').strip().lower()
        if verdict not in VERDICTS:
            verdict = 'discard'
        reason = (data.get('reason') or '').strip()
        _record_verdict(tweet, verdict, reason)

        if verdict == 'standalone':
            seeds.append({
                "source_id": tweet.get('id', ''),
                "author": tweet.get('author', ''),
                "source_url": tweet.get('url', ''),
                "source_text": text[:600],
                "claim": (data.get('claim') or '').strip(),
                "evidence": (data.get('evidence') or '').strip(),
                "mechanism": (data.get('mechanism') or '').strip(),
                "position": (data.get('position') or '').strip(),
                "reason": reason,
            })
            continue
        if verdict == 'discard':
            discards.append({"id": tweet.get('id', ''), "reason": reason})
            continue

        post = (data.get('post') or '').strip()
        if not post:
            continue
        if post.casefold() == text.casefold():
            continue
        pack = xm.ArgumentPack(
            claim=(data.get('claim') or '').strip(),
            evidence=((data.get('evidence') or '').strip() or tweet.get('url', '')),
            mechanism=(data.get('mechanism') or '').strip(),
            position=(data.get('position') or '').strip(),
            context={'tweet_id': tweet.get('id', ''), 'author': tweet.get('author', ''),
                     'source_url': tweet.get('url', ''), 'source_text': text[:500]},
        )
        if not pack.is_complete():
            continue
        if verdict == 'reply':
            try:
                art = xm.reply_draft_artifact(
                    tweet_id=str(tweet.get('id', '')),
                    author=str(tweet.get('author', '')),
                    source_text=text,
                    body=post[:280],
                    pack=pack,
                )
                artifacts.append(art)
            except Exception as exc:
                print(f"[x-quote-scout] reply build skip: {exc}", file=sys.stderr)
        else:  # quote
            artifacts.append(xm.XArtifact(id=xm._new_id(xm.LANE_QUOTE_SCAN),
                                          lane=xm.LANE_QUOTE_SCAN, brand='sahil_twitter',
                                          body=post[:280], pack=pack))
        if len(artifacts) >= xm.QUOTE_SCAN_MAX + 2:
            break
    return artifacts[:xm.QUOTE_SCAN_MAX], seeds, discards


STATE_FILE = Path('/home/kensei/repos/KenseiAgent/content_engine/data/x_scout_last_run.json')


def _already_reported(artifacts) -> bool:
    try:
        state = json.loads(STATE_FILE.read_text())
    except Exception:
        return False
    return any(a.id in state.get('ids', []) for a in artifacts)


def _record_reported(artifacts) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    ids = [a.id for a in artifacts]
    try:
        state = json.loads(STATE_FILE.read_text())
    except Exception:
        state = {'ids': []}
    state.setdefault('ids', [])
    state['ids'] = list(dict.fromkeys(state['ids'] + ids))[-40:]
    STATE_FILE.write_text(json.dumps(state))


def _merge_standalone_seeds(seeds: list) -> None:
    """Append new standalone seeds, keeping the file bounded and deduped by source id."""
    if not seeds:
        return
    existing = _load_standalone_seeds()
    known = {s.get("source_id") for s in existing}
    fresh = [s for s in seeds if s.get("source_id") not in known]
    if not fresh:
        return
    merged = existing + fresh
    _save_standalone_seeds(merged[-40:])


def main():
    _load_env()
    rows = _collect()
    if not rows:
        return
    artifacts, seeds, discards = _candidate_artifacts(rows)
    _merge_standalone_seeds(seeds)
    health = _verdict_health()
    print(f"[x-quote-scout] verdicts: {health.get('counts')} discard_rate={health.get('discard_rate')}", file=sys.stderr)
    if len(artifacts) < xm.QUOTE_SCAN_MIN:
        print(f"[x-quote-scout] only {len(artifacts)} valid artifacts, below floor", file=sys.stderr)
        return
    if _already_reported(artifacts):
        return
    staged = []
    for art in artifacts:
        try:
            xm.stage_for_approval(art)
            staged.append(art)
        except Exception as exc:
            print(f"[x-quote-scout] stage failed: {exc}", file=sys.stderr)
    if not staged:
        return
    report = render_report(staged, lane='quote-scout', title='Quote-post recommendations')
    _record_reported(staged)
    print(f"X Manager · {len(staged)} post recommendations (quotes + replies)")
    print("Original posts and recommended drafts are in the attached review.")
    print(f"MEDIA:{report}")


if __name__ == "__main__":
    main()
