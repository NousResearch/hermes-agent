"""
Direct X/Twitter API v2 client for Hermes Simulator.
No x-cli dependency — uses curl via terminal() with bearer token.

Provides:
- get_user(handle) — profile, bio, metrics
- get_tweets(user_id, count) — recent tweets with metrics
- search_tweets(query, count) — search for tweets
- get_user_mentions(user_id, count) — mentions of a user
"""

from hermes_tools import terminal
import json
import os
import time
import urllib.parse

# Bearer token — loaded from env or hardcoded fallback
BEARER = os.environ.get("X_BEARER_TOKEN", "")

MAX_RETRIES = 3
BASE_DELAY = 2  # seconds, exponential backoff: 2s, 4s, 8s


def _api_get(endpoint: str, params: dict = None) -> dict:
    """Make authenticated GET request to X API v2 with retry and error handling."""
    url = f"https://api.twitter.com/2/{endpoint}"
    if params:
        qs = "&".join(f"{k}={urllib.parse.quote(str(v))}" for k, v in params.items())
        url += f"?{qs}"

    for attempt in range(MAX_RETRIES):
        try:
            r = terminal(f'curl -s -w \'\\n%{{http_code}}\' -H "Authorization: Bearer {BEARER}" "{url}"')
            output = r.get("output", "").strip()

            # Split body from status code (last line)
            lines = output.rsplit("\n", 1)
            if len(lines) == 2:
                body, status_str = lines
            else:
                body = output
                status_str = "0"

            try:
                status_code = int(status_str.strip())
            except ValueError:
                status_code = 0

            # Handle specific status codes
            if status_code == 429:
                # Rate limited — retry with backoff
                delay = BASE_DELAY * (2 ** attempt)
                print(f"  [X API] Rate limited (429). Retry {attempt+1}/{MAX_RETRIES} in {delay}s...")
                time.sleep(delay)
                continue

            if status_code in (401, 403):
                return {"error": f"Authentication failed (HTTP {status_code}). Check X_BEARER_TOKEN.", "http_status": status_code}

            if status_code >= 500:
                delay = BASE_DELAY * (2 ** attempt)
                print(f"  [X API] Server error ({status_code}). Retry {attempt+1}/{MAX_RETRIES} in {delay}s...")
                time.sleep(delay)
                continue

            if status_code == 0 and not body:
                # Network error — no response at all
                delay = BASE_DELAY * (2 ** attempt)
                print(f"  [X API] Network error. Retry {attempt+1}/{MAX_RETRIES} in {delay}s...")
                time.sleep(delay)
                continue

            try:
                return json.loads(body)
            except json.JSONDecodeError:
                return {"error": f"Failed to parse response (HTTP {status_code}): {body[:200]}"}

        except Exception as e:
            delay = BASE_DELAY * (2 ** attempt)
            print(f"  [X API] Exception: {e}. Retry {attempt+1}/{MAX_RETRIES} in {delay}s...")
            time.sleep(delay)
            continue

    return {"error": f"All {MAX_RETRIES} retries exhausted for {endpoint}"}


def get_user(handle: str) -> dict:
    """Get user profile by handle."""
    handle = handle.lstrip("@")
    return _api_get(f"users/by/username/{handle}", {
        "user.fields": "description,public_metrics,profile_image_url,created_at,location,url"
    })


def get_tweets(user_id: str, count: int = 20) -> dict:
    """Get user's recent tweets."""
    return _api_get(f"users/{user_id}/tweets", {
        "max_results": max(min(count, 100), 5),
        "tweet.fields": "created_at,public_metrics,text,in_reply_to_user_id,referenced_tweets",
        "exclude": "retweets"  # original tweets only for voice analysis
    })


def get_tweets_with_rts(user_id: str, count: int = 20) -> dict:
    """Get user's recent tweets including retweets (shows interests)."""
    return _api_get(f"users/{user_id}/tweets", {
        "max_results": max(min(count, 100), 5),
        "tweet.fields": "created_at,public_metrics,text,referenced_tweets"
    })


def search_tweets(query: str, count: int = 10) -> dict:
    """Search recent tweets."""
    return _api_get("tweets/search/recent", {
        "query": query,
        "max_results": max(min(count, 100), 10),
        "tweet.fields": "created_at,public_metrics,text,author_id"
    })


def get_user_by_id(user_id: str) -> dict:
    """Get user profile by ID."""
    return _api_get(f"users/{user_id}", {
        "user.fields": "description,public_metrics,username,name"
    })


# ═══════════════════════════════════════════════════════════════
# HIGH-LEVEL INTELLIGENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def profile_user(handle: str) -> dict:
    """Full profile pull: identity + recent tweets (originals only)."""
    user = get_user(handle)
    if "errors" in user or "error" in user:
        return {"error": f"User @{handle} not found", "details": user}

    user_data = user.get("data", {})
    user_id = user_data.get("id")

    result = {
        "profile": user_data,
        "tweets": [],
        "voice_samples": [],
    }

    if user_id:
        # Get original tweets (no RTs) for voice analysis
        tweets = get_tweets(user_id, 20)
        tweet_list = tweets.get("data", [])
        result["tweets"] = tweet_list

        # Extract pure text samples for voice profiling
        # Only exclude retweets and actual replies (has in_reply_to_user_id)
        # Tweets starting with @ are fine if they're standalone mentions
        result["voice_samples"] = [
            t["text"] for t in tweet_list
            if not t.get("text", "").startswith("RT @")
            and not t.get("in_reply_to_user_id")
        ]

    return result


def profile_interactions(handle1: str, handle2: str) -> dict:
    """Find interactions between two users."""
    # Search for replies from handle1 to handle2
    q1 = f"from:{handle1} to:{handle2}"
    q2 = f"from:{handle2} to:{handle1}"

    r1 = search_tweets(q1, 10)
    r2 = search_tweets(q2, 10)

    return {
        f"{handle1}_to_{handle2}": r1.get("data", []),
        f"{handle2}_to_{handle1}": r2.get("data", []),
    }


def get_voice_data(handle: str, count: int = 50) -> dict:
    """Pull maximum voice data: tweets, replies, quote tweets.
    Returns categorized samples for voice profiling."""
    user = get_user(handle)
    if "errors" in user or "error" in user:
        return {"error": f"User @{handle} not found"}

    user_data = user.get("data", {})
    user_id = user_data.get("id")
    if not user_id:
        return {"error": "No user ID found"}

    # Original tweets (exclude RTs)
    originals = get_tweets(user_id, min(count, 100))
    original_list = originals.get("data", [])

    # Categorize — only use in_reply_to_user_id to detect replies
    standalone = []  # not replies
    replies = []     # replies to others

    for t in original_list:
        text = t.get("text", "")
        if t.get("in_reply_to_user_id"):
            replies.append(text)
        else:
            standalone.append(text)

    return {
        "profile": user_data,
        "standalone_tweets": standalone,  # their voice at rest
        "replies": replies,               # their voice in conversation
        "total_samples": len(standalone) + len(replies),
    }


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if not BEARER:
        print("ERROR: X_BEARER_TOKEN not set. Set it in environment or ~/.hermes/.env")
        print("Trying to load from .env...")
        try:
            with open(os.path.expanduser("~/.hermes/.env")) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("X_BEARER_TOKEN="):
                        # Use split with maxsplit=1 to handle values with '=' in them
                        # Also strip surrounding quotes if present
                        val = line.split("=", 1)[1]
                        if val and val[0] in ('"', "'") and val[-1] == val[0]:
                            val = val[1:-1]
                        BEARER = val
                        break
        except Exception as e:
            print(f"  Failed to load .env: {e}")

    if not BEARER:
        print("FATAL: No bearer token found.")
        exit(1)

    # Demo: profile two users
    for handle in ["Teknium", "basedjensen"]:
        print(f"\n{'='*60}")
        print(f"  PROFILING @{handle}")
        print(f"{'='*60}")

        data = profile_user(handle)
        profile = data.get("profile", {})
        print(f"  Name: {profile.get('name')}")
        print(f"  Bio: {profile.get('description')}")
        metrics = profile.get("public_metrics", {})
        print(f"  Followers: {metrics.get('followers_count')}")
        print(f"  Tweets: {metrics.get('tweet_count')}")
        print(f"  Likes given: {metrics.get('like_count')}")

        print(f"\n  Voice samples ({len(data.get('voice_samples', []))}):")
        for sample in data.get("voice_samples", [])[:5]:
            print(f"    > {sample[:120]}")
