"""
Subscriber store for the Hermes price-tracker SaaS.

Backed by a single JSON file at $HERMES_HOME/subscribers.json so it requires
zero extra infrastructure.  All mutations are atomic (write-to-tmp + rename).

Schema
------
{
  "subscribers": {
    "user@example.com": {
      "stripe_customer_id": "cus_xxx",
      "stripe_subscription_id": "sub_xxx",
      "plan": "starter",          # "starter" | "pro"
      "url_limit": 5,
      "urls": ["https://example.com/pricing"],
      "active": true,
      "status": "active",         # "active" | "past_due" | "canceled"
      "created_at": "<ISO-8601>",
      "updated_at": "<ISO-8601>"
    }
  }
}
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Pro is effectively unlimited; a large sentinel keeps the quota logic simple
# while the UI renders any value >= 1_000_000 as "Unlimited".
PLAN_LIMITS: dict[str, int] = {
    "starter": 3,
    "pro": 1_000_000,
}


def _hermes_home() -> Path:
    env = os.environ.get("HERMES_HOME", "")
    return Path(env) if env else Path.home() / ".hermes"


def _store_path() -> Path:
    return _hermes_home() / "subscribers.json"


def _load() -> dict:
    p = _store_path()
    if not p.exists():
        return {"subscribers": {}}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"subscribers": {}}


def _save(data: dict) -> None:
    p = _store_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=".subscribers.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def upsert_subscriber(
    email: str,
    stripe_customer_id: str,
    stripe_subscription_id: str,
    plan: str = "starter",
    status: str = "active",
) -> dict:
    """Create or update a subscriber record. Returns the subscriber dict."""
    data = _load()
    subs = data.setdefault("subscribers", {})
    email = email.strip().lower()
    existing = subs.get(email, {})
    url_limit = PLAN_LIMITS.get(plan, PLAN_LIMITS["starter"])
    subs[email] = {
        "stripe_customer_id": stripe_customer_id,
        "stripe_subscription_id": stripe_subscription_id,
        "plan": plan,
        "url_limit": url_limit,
        "urls": existing.get("urls", []),
        "active": status == "active",
        "status": status,
        "created_at": existing.get("created_at", _now()),
        "updated_at": _now(),
    }
    _save(data)
    return subs[email]


def set_subscriber_status(email: str, status: str) -> Optional[dict]:
    """Update subscription status. Returns updated record or None if not found."""
    data = _load()
    email = email.strip().lower()
    sub = data.get("subscribers", {}).get(email)
    if sub is None:
        return None
    sub["status"] = status
    sub["active"] = status == "active"
    sub["updated_at"] = _now()
    _save(data)
    return sub


def deactivate_by_subscription_id(subscription_id: str) -> Optional[str]:
    """Mark a subscriber as canceled by their Stripe subscription ID.
    Returns the email address if found, else None."""
    data = _load()
    for email, sub in data.get("subscribers", {}).items():
        if sub.get("stripe_subscription_id") == subscription_id:
            sub["status"] = "canceled"
            sub["active"] = False
            sub["updated_at"] = _now()
            _save(data)
            return email
    return None


def get_subscriber(email: str) -> Optional[dict]:
    email = email.strip().lower()
    return _load().get("subscribers", {}).get(email)


def add_url(email: str, url: str) -> tuple[bool, str]:
    """Add a URL to a subscriber's tracked list.
    Returns (success, message)."""
    data = _load()
    email = email.strip().lower()
    sub = data.get("subscribers", {}).get(email)
    if sub is None:
        return False, "No active subscription found for that email."
    if not sub.get("active"):
        return False, "Your subscription is not active."
    urls: list = sub.get("urls", [])
    if url in urls:
        return False, "That URL is already in your list."
    if len(urls) >= sub.get("url_limit", 5):
        return False, f"You've reached your plan limit of {sub['url_limit']} URLs."
    urls.append(url)
    sub["urls"] = urls
    sub["updated_at"] = _now()
    _save(data)
    return True, "URL added."


def remove_url(email: str, url: str) -> tuple[bool, str]:
    """Remove a URL from a subscriber's tracked list."""
    data = _load()
    email = email.strip().lower()
    sub = data.get("subscribers", {}).get(email)
    if sub is None:
        return False, "No subscription found."
    urls: list = sub.get("urls", [])
    if url not in urls:
        return False, "URL not in your list."
    urls.remove(url)
    sub["urls"] = urls
    sub["updated_at"] = _now()
    _save(data)
    return True, "URL removed."


def list_subscribers() -> list[dict]:
    """Return all subscribers (admin view — includes email field)."""
    data = _load()
    result = []
    for email, sub in data.get("subscribers", {}).items():
        result.append({"email": email, **sub})
    return sorted(result, key=lambda s: s.get("created_at", ""), reverse=True)
