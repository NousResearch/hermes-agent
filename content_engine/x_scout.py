"""X auto-suggest scout: surface repurpose candidates from X into the inbox.

Searches Sahil's niches via xurl, ranks by engagement, and stages suggestions
as pending approval items via the G03 content gate. Nothing is posted to
Discord directly: the gate sends an approval card, and only an explicit
``!approve <id>`` will publish the suggestion to the repurpose inbox.

NOTE: the X API is spend-capped until 2026-06-14. This script handles that
gracefully (logs and skips), so it is safe to have scheduled before then.
"""
import json
import subprocess
import time
from typing import Dict, List

XURL = "/home/kensei/.local/bin/xurl"

# Topic queries per target brand (Sahil's selected niches).
BRAND_QUERIES: Dict[str, List[str]] = {
    "sahil_twitter": [
        "Nous Research OR Hermes Agent OR nano-banana -is:retweet lang:en",
        "AI agents OR indie hacker OR build in public -is:retweet lang:en",
    ],
    "sahil_linkedin": [
        "product management OR SaaS OR PropTech -is:retweet lang:en",
    ],
}

SUGGESTIONS_PER_RUN = 4


def _search(query: str, n: int = 10) -> List[dict]:
    """Run an xurl search. Returns tweet dicts, or [] on spend cap / error."""
    try:
        result = subprocess.run(
            [XURL, "search", query, "-n", str(n)],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0 or not result.stdout.strip():
            print(f"[x_scout] xurl returned {result.returncode}: {result.stderr[:120]}")
            return []
        data = json.loads(result.stdout)
    except Exception as exc:  # noqa: BLE001
        print(f"[x_scout] search failed: {exc}")
        return []
    if isinstance(data, dict) and data.get("title") == "SpendCapReached":
        print("[x_scout] X API spend cap reached, skipping (resets 2026-06-14).")
        return []
    if isinstance(data, dict):
        return data.get("data", []) or []
    return data if isinstance(data, list) else []


def _engagement(tweet: dict) -> int:
    m = tweet.get("public_metrics", {}) or {}
    return m.get("like_count", 0) + 2 * m.get("retweet_count", 0) + m.get("reply_count", 0)


def run_scout(channel_id: str = None) -> int:
    """Search, rank, and stage suggestions for approval. Returns count staged."""
    from content_gate import register_for_approval
    from repurpose import INBOX_CHANNEL

    channel_id = channel_id or INBOX_CHANNEL
    candidates = []
    for brand, queries in BRAND_QUERIES.items():
        for q in queries:
            for t in _search(q):
                t["_brand"] = brand
                candidates.append(t)

    if not candidates:
        print("[x_scout] no candidates (spend cap or empty).")
        return 0

    candidates.sort(key=_engagement, reverse=True)
    staged = 0
    for t in candidates[:SUGGESTIONS_PER_RUN]:
        tid = t.get("id")
        text = (t.get("text") or "").replace("\n", " ")[:600]
        link = f"https://x.com/i/web/status/{tid}" if tid else ""
        body = (
            f"💡 Suggestion (angle for `{t['_brand']}`)\n"
            f"> {text}\n\n"
            f"Approve to queue an inspired-by post, or reject to drop."
        )
        register_for_approval(
            body_text=body,
            brand=t["_brand"],
            platform="twitter",
            source="x_scout",
            title=f"X suggestion from {t.get('author_id', 'unknown')}",
            external_url=link,
            metadata={"tweet_id": tid, "engagement": _engagement(t)},
            send_card=True,
            channel_id=channel_id,
        )
        staged += 1
        time.sleep(0.6)

    print(f"[x_scout] staged {staged} suggestion(s) for approval.")
    return staged


if __name__ == "__main__":
    run_scout()
