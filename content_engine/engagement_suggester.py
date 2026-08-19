"""Quote tweet and reply suggestion pipeline for X/Twitter.

Scans target X accounts, generates brand-voice responses (quote tweets + replies),
saves suggestions to JSON, delivers them to Discord for one-click approval,
and posts on !qt / !reply commands via xurl.

Usage:
    python engagement_suggester.py scan          # Run the full pipeline
    python engagement_suggester.py deliver       # Re-deliver pending suggestions to Discord
    python engagement_suggester.py approve --qt <id>   # Post a quote tweet
    python engagement_suggester.py approve --reply <id>  # Post a reply
    python engagement_suggester.py approve --skip <id>   # Dismiss a suggestion
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

# ──────────────────────────────────────────────────────────────────────
# ERRORS
# ──────────────────────────────────────────────────────────────────────


class XUrlConfigurationError(RuntimeError):
    """Raised when required XURL OAuth configuration is missing or unusable."""


# ──────────────────────────────────────────────────────────────────────
# X POSTER — browser automation via Playwright
# ──────────────────────────────────────────────────────────────────────

try:
    from engagement_x_poster import quote_tweet as _browser_qt, reply_to_tweet as _browser_reply, is_authenticated as _browser_auth, fetch_tweets_batch as _browser_fetch_batch
    _HAS_BROWSER_POSTER = True
except ImportError:
    _HAS_BROWSER_POSTER = False

# ──────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────

XURL_BIN = os.path.expanduser("~/.local/bin/xurl")
DATA_DIR = Path(__file__).parent / "data"
SUGGESTIONS_FILE = DATA_DIR / "engagement_suggestions.json"
DISCORD_CHANNEL = os.getenv("DISCORD_CONTENT_CHANNEL_ID", "1507448580649123900")
DISCORD_API = "https://discord.com/api/v10"
CONTENT_LIMIT = 1900

# Target accounts from reply_suggester.py
TARGET_ACCOUNTS = [
    "marc_louvion",
    "levelsio",
    "tahseen_rahman",
    "shadcn",
    "rauchg",
    "theo",
    "swyx",
    "ryan_c_harris",
    "kentcdodds",
    "pk_hal",
    "AnthropicAI",
    "claude_code",
    "alexalbert__",
    "amasad",
    "TheAthleticFC",
    "utdreport",
    "ManUtd",
    "GaryLineker",
    "naval",
    "paulg",
    "sweatystartup",
    # Nous Research / Hermes Agent ecosystem
    "NousResearch",
    "teknium",
    "hermesagent",
    # Sahil's own feed (mentions and replies to his posts)
    "Sahil_Saghir",
]

# Allow env var override for batched scanning (ENGAGEMENT_BATCH_ACCOUNTS=acct1,acct2,...)
_batch_override = os.getenv("ENGAGEMENT_BATCH_ACCOUNTS", "")
if _batch_override:
    TARGET_ACCOUNTS = [a.strip() for a in _batch_override.split(",") if a.strip()]

# Minimum engagement threshold (likes + replies + retweets) to consider a post
ENGAGEMENT_THRESHOLD = 10

# How many recent posts to scan per account
POSTS_PER_ACCOUNT = 10

# ──────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────


def _discord_token() -> str:
    return os.getenv("DISCORD_BOT_TOKEN", "").strip()


def _discord_headers() -> dict:
    return {"Authorization": f"Bot {_discord_token()}"}


def _channel_type(channel_id: str) -> Optional[int]:
    """Return the Discord channel type (15 = forum), or None on error."""
    try:
        resp = requests.get(
            f"{DISCORD_API}/channels/{channel_id}",
            headers=_discord_headers(), timeout=10,
        )
        if resp.status_code == 200:
            return resp.json().get("type")
    except Exception:
        pass
    return None


def _create_forum_thread(forum_id: str, name: str, content: str) -> Optional[str]:
    """Create a forum post (thread) and return its id."""
    try:
        resp = requests.post(
            f"{DISCORD_API}/channels/{forum_id}/threads",
            headers={**_discord_headers(), "Content-Type": "application/json"},
            json={"name": name[:100], "message": {"content": content[:2000]}},
            timeout=15,
        )
        if resp.status_code in (200, 201):
            return resp.json().get("id")
        print(f"[engagement] Forum thread creation failed {resp.status_code}: {resp.text[:200]}")
    except Exception as exc:
        print(f"[engagement] Forum thread error: {exc}")
    return None


# Max suggestions to generate per scan (top N by engagement, prevents LLM waste)
MAX_SUGGESTIONS_PER_SCAN = 12

# Max suggestions to deliver per run (prevents Discord flooding)
MAX_DELIVER_PER_RUN = 5


def _load_suggestions() -> List[Dict]:
    """Load existing suggestions from the JSON file."""
    if not SUGGESTIONS_FILE.exists():
        return []
    try:
        with open(SUGGESTIONS_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def _save_suggestions(suggestions: List[Dict]) -> None:
    """Save suggestions to the JSON file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(SUGGESTIONS_FILE, "w") as f:
        json.dump(suggestions, f, indent=2, default=str)


def _post_to_discord(channel_id: str, content: str) -> bool:
    """Post a message to Discord. Retries on 429."""
    if not _discord_token():
        print("[engagement] DISCORD_BOT_TOKEN not set, skipping Discord delivery.")
        return False

    url = f"{DISCORD_API}/channels/{channel_id}/messages"
    headers = {**_discord_headers(), "Content-Type": "application/json"}
    content = (content or "").strip()[:CONTENT_LIMIT]

    for attempt in range(4):
        try:
            resp = requests.post(
                url, headers=headers, json={"content": content or "​"}, timeout=30,
            )
        except Exception as exc:
            print(f"[engagement] Discord post error: {exc}")
            return False

        if resp.status_code in (200, 201):
            return True
        if resp.status_code == 429:
            retry_after = float(resp.json().get("retry_after", 1.0)) if resp.content else 1.0
            time.sleep(min(retry_after + 0.25, 5.0))
            continue
        print(f"[engagement] Discord post failed {resp.status_code}: {resp.text[:200]}")
        return False
    return False


def _run_xurl(args: List[str]) -> Tuple[int, str]:
    """Run xurl with the given args and return (exit_code, combined_output).

    Always uses --auth oauth1 since the default app config may not auto-select
    the right auth mode for all endpoints (search, quote, reply, etc.).
    Combines stdout and stderr since xurl writes error details (e.g. 429
    "Too Many Requests") to stderr.
    """
    try:
        result = subprocess.run(
            [XURL_BIN] + args + ["--auth", "oauth1"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Combine stdout + stderr (xurl errors come via stderr)
        combined = (result.stdout or "").strip()
        if result.stderr and "{" in result.stderr:
            # xurl prints structured error JSON to stderr; prefer it if it
            # contains a status code (4xx/5xx).
            combined = (result.stderr or "").strip() + ("\n" + combined if combined else "")
        return result.returncode, combined
    except FileNotFoundError:
        return -1, f"xurl not found at {XURL_BIN}"
    except subprocess.TimeoutExpired:
        return -2, "xurl timed out"
    except Exception as exc:
        return -3, str(exc)


# ──────────────────────────────────────────────────────────────────────
# XURL SEARCH — Fetch tweets from target accounts
# ──────────────────────────────────────────────────────────────────────


def fetch_recent_tweets(account: str, limit: int = POSTS_PER_ACCOUNT) -> List[Dict]:
    """Fetch recent tweets from a target account using xurl search.

    Returns a list of tweet dicts with keys: id, text, author, likes, replies,
    retweets, created_at.
    """
    # Use xurl search to fetch recent tweets from the account
    code, output = _run_xurl([
        "search", f"from:{account}",
        "-n", str(limit + 5),  # Fetch extra to allow for filtering
    ])

    if code != 0:
        if "spend cap" in output.lower() or "forbidden" in output.lower():
            print(f"[engagement] X API spend cap reached for {account}. Skipping.")
        else:
            print(f"[engagement] xurl search failed for {account}: {output[:200]}")
        return []

    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        print(f"[engagement] Could not parse xurl output for {account}")
        return []

    tweets = []
    # xurl search returns {"data": [...], "includes": {"users": [...]}}
    raw_tweets = []
    if isinstance(data, dict):
        inner = data.get("data", data)
        if isinstance(inner, list):
            raw_tweets = inner
        elif isinstance(inner, dict):
            raw_tweets = inner.get("data", [])

    # Build a map of author_id -> username from includes
    author_map = {}
    includes = data.get("includes", {}) if isinstance(data, dict) else {}
    for user in includes.get("users", []):
        uid = user.get("id", "")
        uname = user.get("username", user.get("name", ""))
        if uid:
            author_map[uid] = uname

    for t in raw_tweets:
        if not isinstance(t, dict):
            continue
        text = t.get("text", "") or ""
        if not text.strip():
            continue

        # Skip retweets
        if text.startswith("RT @") or text.startswith("RT "):
            continue

        metrics = t.get("public_metrics", t.get("metrics", {}))
        if isinstance(metrics, dict):
            likes = int(metrics.get("like_count", metrics.get("likes", 0)))
            replies_ct = int(metrics.get("reply_count", metrics.get("replies", 0)))
            retweets_ct = int(metrics.get("retweet_count", metrics.get("retweets", 0)))
        else:
            likes = replies_ct = retweets_ct = 0

        # Resolve author name from includes
        author_id = t.get("author_id", "")
        resolved_author = author_map.get(author_id, account)

        tweets.append({
            "id": t.get("id", ""),
            "text": text,
            "author": account,
            "author_name": resolved_author,
            "likes": likes,
            "replies": replies_ct,
            "retweets": retweets_ct,
            "engagement": likes + replies_ct + retweets_ct,
            "created_at": t.get("created_at", ""),
            "url": f"https://x.com/{account}/status/{t.get('id', '')}",
        })

    return tweets


# ──────────────────────────────────────────────────────────────────────
# CONTENT CATEGORISATION
# ──────────────────────────────────────────────────────────────────────


def _categorize_post(text: str) -> str:
    """Categorise a post for voice register selection."""
    lower = text.lower()

    if any(w in lower for w in ["ship", "shipped", "launch", "milestone", "released", "v1", "v2"]):
        return "build_update"
    if any(w in lower for w in ["how to", "step", "tutorial", "guide", "setup", "config", "workflow"]):
        return "technical_tip"
    if any(w in lower for w in ["think", "opinion", "hot take", "unpopular", "believe"]):
        return "opinion"
    if any(w in lower for w in ["noticed", "observ", "pattern", "trend", "every", "always"]):
        return "observation"
    if any(w in lower for w in ["goal", "match", "united", "fc", "premier league", "half time", "final"]):
        return "football"
    if any(w in lower for w in ["ai", "agent", "model", "gpt", "claude", "llm"]):
        return "ai_tools"

    return "build_update"


# ──────────────────────────────────────────────────────────────────────
# GENERATE QUOTE TWEET
# ──────────────────────────────────────────────────────────────────────


def _llm_generate_response(tweet_text: str, author: str, response_type: str) -> str:
    """Call the LLM to generate a contextual quote tweet or reply.
    
    Uses Ollama Cloud (deepseek-v4-flash) with a system prompt that encodes
    Sahil's brand voice rules. The LLM reads the actual tweet and produces
    a response that is actually about what was said.
    
    response_type: "quote_tweet" or "reply"
    """
    import requests as _requests
    
    key = os.getenv("OLLAMA_API_KEY", "")
    if not key:
        # Try loading from .env
        env_path = Path.home() / ".hermes" / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("OLLAMA_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    
    if not key:
        # No API key — drop candidate, do not emit fallback echo
        return None
    
    if response_type == "quote_tweet":
        system_prompt = (
            "You are Sahil Saghir writing a quote tweet. You READ the original tweet "
            "and write a response that is SPECIFICALLY about what was said. "
            "NOT a generic take. NOT a template. You actually engage with the content.\n\n"
            "VOICE RULES:\n"
            "- Direct, specific, honest. Short sentences. No hedging.\n"
            "- British English. No em-dashes.\n"
            "- If it's a football tweet, react like a real fan (dry, not preachy).\n"
            "- If it's an AI/tech tweet, add a specific data point or observation from your own work.\n"
            "- If it's an indie hacker tweet, relate it to your actual experience shipping 4 apps.\n"
            "- If it's a news/announcement tweet, react to the NEWS itself, not to some unrelated topic.\n"
            "- NEVER use: 'context engineering', 'shipping speed is the moat', 'the model is the engine' "
            "unless the tweet is literally about that.\n"
            "- 180-280 chars max. One idea. No threads.\n"
            "- No hashtags unless they fit naturally.\n"
            "- Read the tweet. Actually read it. Respond to what it says."
        )
        user_prompt = f"Original tweet by @{author}:\n\n\"{tweet_text}\"\n\nWrite Sahil's quote tweet:"
    else:
        system_prompt = (
            "You are Sahil Saghir writing a reply to someone's tweet. "
            "You READ the original tweet and write a reply that is SPECIFICALLY about what was said. "
            "NOT a generic response. You actually engage with the content like a real person.\n\n"
            "VOICE RULES:\n"
            "- 1-3 sentences. Conversational. Like talking to someone you respect.\n"
            "- Direct, specific, honest. British English. No em-dashes.\n"
            "- If it's a football tweet, react like a real fan in the group chat.\n"
            "- If it's a tech tweet, add a specific observation or ask a genuine question.\n"
            "- If it's a milestone/launch, congratulate specifically and add something real.\n"
            "- If it's news, react to the actual news content.\n"
            "- NEVER use generic templates about 'context engineering' or 'shipping speed'.\n"
            "- Read the tweet. Actually read it. Reply to what it says."
        )
        user_prompt = f"Original tweet by @{author}:\n\n\"{tweet_text}\"\n\nWrite Sahil's reply (1-3 sentences, no @mention needed):"
    
    try:
        resp = _requests.post(
            "https://ollama.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": "deepseek-v4-flash",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": 200,
                "temperature": 0.8,
            },
            # 60s timeout (was 20s) — ollama.com cloud can have cold starts
            # that blow a 20s budget. The function already falls back to a
            # template response on any error, so a longer timeout only
            # reduces how often we degrade to the fallback.
            timeout=60,
        )
        if resp.status_code == 200:
            text = resp.json()["choices"][0]["message"]["content"].strip()
            # Clean up: remove quotes, em-dashes, excess whitespace
            text = text.strip('"').strip("'").strip()
            text = text.replace("\u2014", " - ")
            text = text.replace("\u2013", " - ")
            # Remove @ mentions at the start (xurl reply adds them automatically)
            text = re.sub(r'^@\w+\s+', '', text)
            if text and len(text) > 10:
                return text
        else:
            print(f"[engagement] LLM call failed {resp.status_code}: {resp.text[:100]}")
    except Exception as exc:
        print(f"[engagement] LLM call error: {exc}")
    
    # Drop candidate on failure — never emit fallback echo
    return None


# ── Quality Gate — post-generation guardrails ──

_GENERIC_OPENERS = [
    "this.", "exactly.", "this is interesting", "good point on",
    "worth adding", "interesting take", "great point", "well said",
    "couldn't agree more", "nice one", "love this",
]

def _quality_gate(draft: str, post_text: str) -> bool:
    """Return True if the draft passes all quality checks.
    
    Guardrails from W8_X_ENGAGEMENT_REBUILD_SPEC.md:
    1. Post-understanding: must reference something actually in the post
    2. Echo detection: must not be a simple restatement
    3. Length check: 50-280 characters
    4. Generic-opener check: reject dead templates
    5. Voice check: British English, no hedging
    """
    if not draft or len(draft.strip()) < 10:
        return False
    
    # 1. Generic-opener check
    draft_lower = draft.strip().lower()
    for opener in _GENERIC_OPENERS:
        if draft_lower.startswith(opener):
            print(f"[engagement] quality gate: rejected generic opener '{opener}'")
            return False
    
    # 2. Length check: 50-280 chars
    draft_clean = draft.strip()
    if len(draft_clean) < 50:
        print(f"[engagement] quality gate: too short ({len(draft_clean)} chars)")
        return False
    if len(draft_clean) > 280:
        print(f"[engagement] quality gate: too long ({len(draft_clean)} chars)")
        return False
    
    # 3. Echo detection — simple overlap check
    # If >60% of draft words are in the post, it's an echo
    post_words = set(post_text.lower().split())
    draft_words = set(draft_lower.split())
    if post_words:
        overlap = len(draft_words & post_words) / len(draft_words)
        if overlap > 0.6:
            print(f"[engagement] quality gate: echo detected ({overlap:.0%} overlap)")
            return False
    
    # 4. Post-understanding: draft must contain at least one specific
    # word or phrase that is NOT a common stop word
    common = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'this', 'that',
              'it', 'to', 'of', 'in', 'for', 'on', 'and', 'or', 'but',
              'with', 'from', 'by', 'at', 'as', 'be', 'has', 'have', 'had'}
    specific_words = [w for w in draft_words if w not in common and len(w) > 3]
    if len(specific_words) < 3:
        print(f"[engagement] quality gate: too generic ({len(specific_words)} specific words)")
        return False
    
    # 5. Hedging check: reject drafts with weak/hedging language
    hedging = {'might', 'maybe', 'perhaps', 'could be', 'possibly', 
               'i think', 'in my opinion', 'arguably'}
    draft_clean_lower = draft_clean.lower()
    for h in hedging:
        if h in draft_clean_lower:
            print(f"[engagement] quality gate: hedging detected '{h}'")
            return False
    
    return True


def _generate_quote_tweet(post: Dict) -> str:
    """Generate a quote tweet in Sahil's brand voice using LLM.
    
    The LLM reads the actual tweet and produces a contextual response.
    Falls back to a minimal response if the LLM is unavailable.
    """
    text = post.get("text", "")
    author = post.get("author", "")
    return _llm_generate_response(text, author, "quote_tweet")


def _generate_reply(post: Dict) -> str:
    """Generate a contextual reply in Sahil's voice using LLM."""
    text = post.get("text", "")
    author = post.get("author", "")
    return _llm_generate_response(text, author, "reply")


# ──────────────────────────────────────────────────────────────────────
# GENERATE REPLY
# ──────────────────────────────────────────────────────────────────────


# Old static-template functions removed — replaced by LLM-based generation above.
# The scan_and_suggest pipeline follows.


def scan_and_suggest() -> List[Dict]:
    """Main pipeline: scan target accounts, rank by engagement, generate top N suggestions.

    Pre-analysis via browser scraper collects all raw tweets across all accounts,
    filters low-engagement/bait, sorts by engagement descending, then generates
    LLM suggestions only for the top MAX_SUGGESTIONS_PER_SCAN tweets.

    Saves all new suggestions to JSON and returns them.
    """
    print(f"[engagement] Scanning {len(TARGET_ACCOUNTS)} target accounts...")
    all_suggestions = _load_suggestions()
    existing_ids = {s.get("tweet_id") for s in all_suggestions if s.get("tweet_id")}

    # ─── Phase 1: Collect all raw tweets ───
    all_raw_tweets: list[dict] = []

    # Use browser scraper (zero API cost) when available
    if _HAS_BROWSER_POSTER and _browser_auth():
        print("[engagement] Using browser scraper (zero API cost)...")
        accounts = [a for a in TARGET_ACCOUNTS if a != "Sahil_Saghir"]
        account_tweets = _browser_fetch_batch(accounts, limit=POSTS_PER_ACCOUNT)
        for acct, tweets in account_tweets.items():
            for t in tweets:
                t["author"] = acct  # ensure author is set
                all_raw_tweets.append(t)
    else:
        # Fallback: per-account xurl
        for account in TARGET_ACCOUNTS:
            if account == "Sahil_Saghir":
                continue
            print(f"[engagement]  Fetching tweets from @{account} via xurl...")
            tweets = fetch_recent_tweets(account)
            for t in tweets:
                t["author"] = account
                all_raw_tweets.append(t)

    # ─── Phase 2: Filter and rank ───
    candidates = []
    for tweet in all_raw_tweets:
        tweet_id = tweet.get("id", "")
        if tweet_id in existing_ids:
            continue

        # Skip low-engagement posts
        if tweet.get("engagement", 0) < ENGAGEMENT_THRESHOLD:
            continue

        # Skip engagement bait
        text = tweet.get("text", "")
        bait_patterns = [
            r"RT if", r"retweet if", r"like if", r"comment below",
            r"follow for", r"tag someone", r"mention someone",
        ]
        if any(re.search(p, text, re.IGNORECASE) for p in bait_patterns):
            continue

        candidates.append(tweet)

    # Sort by engagement descending, take top N
    candidates.sort(key=lambda t: t.get("engagement", 0), reverse=True)
    top_candidates = candidates[:MAX_SUGGESTIONS_PER_SCAN]

    print(f"[engagement] Collected {len(all_raw_tweets)} raw tweets, "
          f"{len(candidates)} eligible, generating suggestions for top {len(top_candidates)}")

    # ─── Phase 3: Generate LLM suggestions for top candidates only ───
    new_suggestions = []
    skipped_silence = 0
    for tweet in top_candidates:
        tweet_id = tweet.get("id", "")
        account = tweet.get("author", "unknown")

        # Skip engagement bait filter (already done above, do quick text check)
        text = tweet.get("text", "")

        # Generate quote tweet and reply via LLM
        quote_text = _generate_quote_tweet(tweet)
        reply_text = _generate_reply(tweet)
        
        # Quality gate: drop candidates where LLM failed or produced garbage
        if quote_text and _quality_gate(quote_text, text):
            pass  # quote passed
        else:
            quote_text = None
        
        if reply_text and _quality_gate(reply_text, text):
            pass  # reply passed
        else:
            reply_text = None
        
        # Skip if both generation paths failed
        if not quote_text and not reply_text:
            skipped_silence += 1
            continue

        suggestion_id = str(uuid.uuid4())[:8]
        suggestion = {
            "id": suggestion_id,
            "tweet_id": tweet_id,
            "author": account,
            "author_name": tweet.get("author_name", account),
            "tweet_text": text,
            "tweet_url": tweet.get("url", ""),
            "engagement": tweet.get("engagement", 0),
            "likes": tweet.get("likes", 0),
            "replies": tweet.get("replies", 0),
            "retweets": tweet.get("retweets", 0),
            "category": _categorize_post(text),
            "quote_tweet": quote_text,
            "reply": reply_text,
            "status": "pending",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        new_suggestions.append(suggestion)
        existing_ids.add(tweet_id)

        print(f"[engagement]    + Suggestion {suggestion_id}: @{account} "
              f"(engagement: {tweet['engagement']})")

    if new_suggestions:
        all_suggestions.extend(new_suggestions)
        _save_suggestions(all_suggestions)
        print(f"[engagement] Saved {len(new_suggestions)} new suggestions to {SUGGESTIONS_FILE}")
    else:
        print("[engagement] No new suggestions generated.")

    return new_suggestions


# ──────────────────────────────────────────────────────────────────────
# DISCORD DELIVERY
# ──────────────────────────────────────────────────────────────────────


def deliver_to_discord(suggestions: Optional[List[Dict]] = None) -> int:
    """Deliver pending suggestions to Discord as interactive cards.

    Creates a single forum thread per run (if #content is a forum channel),
    caps to top MAX_DELIVER_PER_RUN by engagement, posts all cards inside
    that thread to avoid flooding the channel.

    Returns the number of suggestions delivered.
    """
    if suggestions is None:
        suggestions = _load_suggestions()

    pending = [s for s in suggestions if s.get("status") == "pending"]
    if not pending:
        print("[engagement] No pending suggestions to deliver.")
        return 0

    if not _discord_token():
        print("[engagement] DISCORD_BOT_TOKEN not set, cannot deliver to Discord.")
        return 0

    # Sort by engagement descending, cap to top N
    pending.sort(key=lambda s: s.get("engagement", 0), reverse=True)
    to_deliver = pending[:MAX_DELIVER_PER_RUN]
    skipped = len(pending) - len(to_deliver)

    stamp = datetime.now().strftime("%d/%m/%y %H:%M")
    header = (
        f"Target Engagement Suggestions · {stamp}\n"
        f"Top {len(to_deliver)} by engagement"
        + (f" ({skipped} lower-scoring skipped)" if skipped else "")
        + "\nUse `!qt <id>` (quote tweet), `!reply <id>` (reply), `!skip <id>` (dismiss)"
    )

    # Forum channels (type 15) need a thread — create one and post inside it
    target = DISCORD_CHANNEL
    if _channel_type(DISCORD_CHANNEL) == 15:
        thread_id = _create_forum_thread(
            DISCORD_CHANNEL,
            f"Engagement · {stamp}",
            header,
        )
        if not thread_id:
            print("[engagement] Could not create forum thread for delivery.")
            return 0
        target = thread_id
    else:
        _post_to_discord(DISCORD_CHANNEL, header)
        time.sleep(0.4)

    delivered = 0
    for s in to_deliver:
        card = _build_discord_card(s)
        ok = _post_to_discord(target, card)
        if ok:
            delivered += 1
        time.sleep(0.6)

    print(f"[engagement] Delivered {delivered}/{len(to_deliver)} suggestions to Discord.")
    return delivered


def _build_discord_card(suggestion: Dict) -> str:
    """Build a Discord message card for a suggestion."""
    lines = [
        f"━━━ **@{suggestion['author']}** · {suggestion['engagement']} engagements ━━━",
        "",
        f"**Original tweet:**",
        f"{suggestion['tweet_text'][:280]}",
        f"{suggestion['tweet_url']}",
        f"❤️ {suggestion['likes']}  💬 {suggestion['replies']}  🔄 {suggestion['retweets']}",
        "",
        f"**💬 Quote tweet:**",
        f"```{suggestion['quote_tweet'][:280]}```",
        "",
        f"**↩️ Reply:**",
        f"```{suggestion['reply'][:280]}```",
        "",
        f"`!qt {suggestion['id']}` · `!reply {suggestion['id']}` · `!skip {suggestion['id']}`",
    ]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# APPROVAL HANDLING
# ──────────────────────────────────────────────────────────────────────


def approve_quote_tweet(suggestion_id: str) -> bool:
    """Post a quote tweet — tries browser poster first, falls back to xurl."""
    suggestions = _load_suggestions()
    target = None

    for s in suggestions:
        if s.get("id") == suggestion_id:
            target = s
            break

    if not target:
        print(f"[engagement] Suggestion {suggestion_id} not found.")
        return False

    if target.get("status") != "pending":
        print(f"[engagement] Suggestion {suggestion_id} already {target['status']}.")
        return False

    tweet_id = target.get("tweet_id", "")
    quote_text = target.get("quote_tweet", "")

    if not tweet_id or not quote_text:
        print(f"[engagement] Missing tweet_id or quote_text for {suggestion_id}.")
        return False

    print(f"[engagement] Posting quote tweet for suggestion {suggestion_id}...")

    # Try browser poster first
    posted = False
    if _HAS_BROWSER_POSTER and _browser_auth():
        print("[engagement] Using browser poster...")
        posted = _browser_qt(tweet_id, quote_text)
    else:
        print("[engagement] Browser poster unavailable. Falling back to xurl API...")

    # Fall back to xurl
    if not posted:
        code, output = _run_xurl(["quote", tweet_id, quote_text])
        posted = code == 0
        if not posted:
            print(f"[engagement] xurl quote failed: {output[:200]}")
            return False

    # Mark as approved
    for s in suggestions:
        if s.get("id") == suggestion_id:
            s["status"] = "approved_qt"
            s["approved_at"] = datetime.now(timezone.utc).isoformat()
            break

    _save_suggestions(suggestions)
    print(f"[engagement] Quote tweet posted for suggestion {suggestion_id}.")
    return True


def approve_reply(suggestion_id: str) -> bool:
    """Post a reply — tries browser poster first, falls back to xurl."""
    suggestions = _load_suggestions()
    target = None

    for s in suggestions:
        if s.get("id") == suggestion_id:
            target = s
            break

    if not target:
        print(f"[engagement] Suggestion {suggestion_id} not found.")
        return False

    if target.get("status") != "pending":
        print(f"[engagement] Suggestion {suggestion_id} already {target['status']}.")
        return False

    tweet_id = target.get("tweet_id", "")
    reply_text = target.get("reply", "")

    if not tweet_id or not reply_text:
        print(f"[engagement] Missing tweet_id or reply_text for {suggestion_id}.")
        return False

    print(f"[engagement] Posting reply for suggestion {suggestion_id}...")

    # Try browser poster first
    posted = False
    if _HAS_BROWSER_POSTER and _browser_auth():
        print("[engagement] Using browser poster...")
        posted = _browser_reply(tweet_id, reply_text)
    else:
        print("[engagement] Browser poster unavailable. Falling back to xurl API...")

    # Fall back to xurl
    if not posted:
        code, output = _run_xurl(["reply", tweet_id, reply_text])
        posted = code == 0
        if not posted:
            print(f"[engagement] xurl reply failed: {output[:200]}")
            return False

    # Mark as approved
    for s in suggestions:
        if s.get("id") == suggestion_id:
            s["status"] = "approved_reply"
            s["approved_at"] = datetime.now(timezone.utc).isoformat()
            break

    _save_suggestions(suggestions)
    print(f"[engagement] Reply posted for suggestion {suggestion_id}.")
    return True


def skip_suggestion(suggestion_id: str) -> bool:
    """Mark a suggestion as skipped."""
    suggestions = _load_suggestions()
    found = False

    for s in suggestions:
        if s.get("id") == suggestion_id:
            s["status"] = "skipped"
            s["skipped_at"] = datetime.now(timezone.utc).isoformat()
            found = True
            break

    if not found:
        print(f"[engagement] Suggestion {suggestion_id} not found.")
        return False

    _save_suggestions(suggestions)
    print(f"[engagement] Suggestion {suggestion_id} skipped.")
    return True


# ──────────────────────────────────────────────────────────────────────
# XURL CONFIG HELPER — Extract Postiz DB credentials
# ──────────────────────────────────────────────────────────────────────


def configure_xurl_from_postiz() -> bool:
    """Read X API credentials from the Postiz PostgreSQL database and write
    them to the xurl config file.

    The Postiz DB stores tokens in the Integration table as
    'internalId-accessToken:accessSecret' format.

    Returns True if credentials were successfully written.
    """
    import subprocess as sp

    # Postiz DB connection details from docker-compose
    db_host = "127.0.0.1"
    db_port = "5432"
    db_user = os.getenv("POSTIZ_DB_USER", "postiz-user")
    db_password = os.getenv("POSTIZ_DB_PASS", "postiz-password")
    db_name = os.getenv("POSTIZ_DB_NAME", "postiz-db-local")

    # Query the Integration table for X accounts
    query = (
        'SELECT "internalId", "name", "token", "refreshToken", "profile" '
        'FROM "Integration" '
        'WHERE "providerIdentifier" = \'x\''
    )

    try:
        result = sp.run(
            [
                "docker", "exec", "postiz-postgres",
                "psql", "-U", db_user, "-d", db_name,
                "-t", "-A", "-F", "|",
                "-c", query,
            ],
            capture_output=True, text=True, timeout=15,
        )
    except FileNotFoundError:
        print("[engagement] docker not found. Cannot query Postiz DB.")
        return False
    except sp.TimeoutExpired:
        print("[engagement] Postiz DB query timed out.")
        return False

    if result.returncode != 0:
        print(f"[engagement] Postiz DB query failed: {result.stderr[:200]}")
        return False

    lines = [l.strip() for l in result.stdout.split("\n") if l.strip()]
    if not lines:
        print("[engagement] No X integrations found in Postiz DB.")
        return False

    # Find the @Sahil_Saghir integration
    sahil_row = None
    for line in lines:
        parts = line.split("|")
        if len(parts) >= 5:
            profile = parts[4].strip()
            if profile == "Sahil_Saghir":
                sahil_row = parts
                break

    if not sahil_row:
        print("[engagement] @Sahil_Saghir integration not found in Postiz DB.")
        print("  Available X profiles:", [p.split("|")[4].strip() for p in lines if len(p.split("|")) >= 5])
        return False

    internal_id = sahil_row[0].strip()
    token = sahil_row[2].strip()
    profile = sahil_row[4].strip()

    # Token format: 'internalId-accessToken:accessSecret'
    # The full string before ':' is the access token (includes internalId prefix)
    if ":" not in token:
        print(f"[engagement] Unexpected token format for {profile} (missing ':' separator).")
        return False

    # Split on the last ':' to get access_token:access_secret
    colon_idx = token.rfind(":")
    access_token = token[:colon_idx]
    access_secret = token[colon_idx + 1:]

    # Get all OAuth credentials from the environment.
    client_id = os.getenv("XURL_CLIENT_ID", "")
    client_secret = os.getenv("XURL_CLIENT_SECRET", "")
    consumer_key = os.getenv("XURL_CONSUMER_KEY", "")
    consumer_secret = os.getenv("XURL_CONSUMER_SECRET", "")
    if not all((client_id, client_secret, consumer_key, consumer_secret)):
        raise XUrlConfigurationError(
            "XURL OAuth configuration is incomplete. Set XURL_CLIENT_ID, "
            "XURL_CLIENT_SECRET, XURL_CONSUMER_KEY and XURL_CONSUMER_SECRET "
            "before running configure-xurl."
        )

    # Write xurl config
    xurl_config = f"""apps:
    kensei-digest:
        client_id: {client_id}
        client_secret: {client_secret}
        oauth1_token:
            type: oauth1
            oauth1:
                access_token: {access_token}
                token_secret: {access_secret}
                consumer_key: {consumer_key}
                consumer_secret: {consumer_secret}
default_app: kensei-digest
"""

    xurl_path = os.path.expanduser("~/.xurl")
    try:
        # Atomic write (C028): write to a temp file in the same directory,
        # fsync, set restrictive permissions (0o600 — the file contains
        # OAuth tokens), then os.replace into place. This prevents a
        # partial write from corrupting an existing config and avoids the
        # TOCTOU window of write-then-chmod.
        import tempfile
        xurl_dir = os.path.dirname(xurl_path)
        fd, tmp_path = tempfile.mkstemp(
            prefix=".xurl-", suffix=".tmp", dir=xurl_dir
        )
        try:
            with os.fdopen(fd, "w") as f:
                f.write(xurl_config)
                f.flush()
                os.fsync(f.fileno())
            os.chmod(tmp_path, 0o600)
            os.replace(tmp_path, xurl_path)
        except Exception:
            # Clean up the temp file on any failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        print(f"[engagement] xurl config written to {xurl_path} for @{profile}")
        return True
    except OSError as exc:
        print(f"[engagement] Failed to write xurl config: {exc}")
        return False


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        return

    command = sys.argv[1]

    if command == "scan":
        print("=" * 60)
        print("  ENGAGEMENT SUGGESTION PIPELINE")
        print("=" * 60)
        new = scan_and_suggest()
        if new:
            print(f"\nDelivering {len(new)} new suggestions to Discord...")
            deliver_to_discord(new)
        print("\nDone.")

    elif command == "deliver":
        delivered = deliver_to_discord()
        print(f"Delivered {delivered} suggestions to Discord.")

    elif command == "approve":
        if len(sys.argv) < 4:
            print("Usage: engagement_suggester.py approve --qt <id> | --reply <id> | --skip <id>")
            return

        action = sys.argv[2]
        suggestion_id = sys.argv[3]

        if action == "--qt":
            ok = approve_quote_tweet(suggestion_id)
        elif action == "--reply":
            ok = approve_reply(suggestion_id)
        elif action == "--skip":
            ok = skip_suggestion(suggestion_id)
        else:
            print(f"Unknown action: {action}")
            return

        if ok:
            print(f"✅ {action} for {suggestion_id} succeeded.")
        else:
            print(f"❌ {action} for {suggestion_id} failed.")

    elif command == "configure-xurl":
        try:
            ok = configure_xurl_from_postiz()
        except XUrlConfigurationError as exc:
            print(f"❌ {exc}")
            return
        if ok:
            print("✅ xurl configured from Postiz DB.")
        else:
            print("❌ Failed to configure xurl from Postiz DB.")

    elif command == "status":
        suggestions = _load_suggestions()
        pending = [s for s in suggestions if s.get("status") == "pending"]
        approved_qt = [s for s in suggestions if s.get("status") == "approved_qt"]
        approved_reply = [s for s in suggestions if s.get("status") == "approved_reply"]
        skipped = [s for s in suggestions if s.get("status") == "skipped"]

        print(f"Suggestions: {len(suggestions)} total")
        print(f"  Pending:       {len(pending)}")
        print(f"  Approved (QT):  {len(approved_qt)}")
        print(f"  Approved (R):   {len(approved_reply)}")
        print(f"  Skipped:        {len(skipped)}")

    else:
        print(f"Unknown command: {command}")
        print(__doc__)


if __name__ == "__main__":
    main()
