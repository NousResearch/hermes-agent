"""AIRIES Agent subscription and usage limits via Stripe.

Tracks per-period usage locally and gates agent turns / web fetches.
Stripe Checkout + Billing Portal for paid tiers. Secrets live in .env only.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from hermes_cli.config import cfg_get, get_env_value, load_config
from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_TIER_LIMITS: Dict[str, Dict[str, int]] = {
    "free": {"monthly_turns": 50, "monthly_web_fetches": 25},
    "pro": {"monthly_turns": 2000, "monthly_web_fetches": 500},
    "team": {"monthly_turns": 10000, "monthly_web_fetches": 2500},
}


@dataclass(frozen=True)
class UsageSnapshot:
    tier: str
    status: str
    period: str
    turns_used: int
    turns_limit: int
    web_fetches_used: int
    web_fetches_limit: int
    stripe_customer_id: str = ""
    stripe_subscription_id: str = ""


class AriesSubscriptionManager:
    """Local usage ledger + Stripe subscription state."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or (get_hermes_home() / "airies_subscription.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS usage_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                period_key TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_usage_period ON usage_events(period_key, event_type);

            CREATE TABLE IF NOT EXISTS subscription_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                tier TEXT NOT NULL DEFAULT 'free',
                status TEXT NOT NULL DEFAULT 'active',
                stripe_customer_id TEXT NOT NULL DEFAULT '',
                stripe_subscription_id TEXT NOT NULL DEFAULT '',
                current_period_end REAL NOT NULL DEFAULT 0,
                updated_at REAL NOT NULL DEFAULT 0
            );
            INSERT OR IGNORE INTO subscription_state (id, tier, status) VALUES (1, 'free', 'active');
            """
        )
        self._conn.commit()

    @staticmethod
    def _period_key() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m")

    def _get_state(self) -> sqlite3.Row:
        return self._conn.execute("SELECT * FROM subscription_state WHERE id = 1").fetchone()

    def _tier_limits(self, tier: str) -> Dict[str, int]:
        config = load_config()
        custom = cfg_get(config, "subscription", "tiers", default={}) or {}
        if isinstance(custom, dict) and tier in custom:
            row = custom[tier]
            if isinstance(row, dict):
                return {
                    "monthly_turns": int(row.get("monthly_turns", _TIER_LIMITS.get(tier, _TIER_LIMITS["free"])["monthly_turns"])),
                    "monthly_web_fetches": int(row.get("monthly_web_fetches", _TIER_LIMITS.get(tier, _TIER_LIMITS["free"])["monthly_web_fetches"])),
                }
        return dict(_TIER_LIMITS.get(tier, _TIER_LIMITS["free"]))

    def _count_events(self, event_type: str, period_key: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS c FROM usage_events WHERE event_type = ? AND period_key = ?",
            (event_type, period_key),
        ).fetchone()
        return int(row["c"])

    def record_event(self, event_type: str) -> None:
        if not self._subscription_enabled():
            return
        period = self._period_key()
        self._conn.execute(
            "INSERT INTO usage_events (event_type, period_key, created_at) VALUES (?, ?, ?)",
            (event_type, period, time.time()),
        )
        self._conn.commit()

    def get_snapshot(self) -> UsageSnapshot:
        state = self._get_state()
        tier = state["tier"] or "free"
        limits = self._tier_limits(tier)
        period = self._period_key()
        return UsageSnapshot(
            tier=tier,
            status=state["status"] or "active",
            period=period,
            turns_used=self._count_events("turn", period),
            turns_limit=limits["monthly_turns"],
            web_fetches_used=self._count_events("web_fetch", period),
            web_fetches_limit=limits["monthly_web_fetches"],
            stripe_customer_id=state["stripe_customer_id"] or "",
            stripe_subscription_id=state["stripe_subscription_id"] or "",
        )

    def check_turn_allowed(self) -> Tuple[bool, str]:
        if not self._subscription_enabled():
            return True, ""
        snap = self.get_snapshot()
        if snap.status not in ("active", "trialing"):
            return False, (
                f"AIRIES subscription is {snap.status}. "
                "Run `hermes subscription checkout --tier pro` to reactivate."
            )
        if snap.turns_used >= snap.turns_limit:
            return False, (
                f"Monthly turn limit reached ({snap.turns_used}/{snap.turns_limit} on {snap.tier}). "
                "Upgrade with `hermes subscription checkout --tier pro`."
            )
        return True, ""

    def check_web_fetch_allowed(self) -> Tuple[bool, str]:
        if not self._subscription_enabled():
            return True, ""
        snap = self.get_snapshot()
        if snap.web_fetches_used >= snap.web_fetches_limit:
            return False, (
                f"Monthly web fetch limit reached ({snap.web_fetches_used}/{snap.web_fetches_limit}). "
                "Upgrade your AIRIES plan for more research capacity."
            )
        return True, ""

    def set_subscription(
        self,
        *,
        tier: str,
        status: str = "active",
        stripe_customer_id: str = "",
        stripe_subscription_id: str = "",
        current_period_end: float = 0,
    ) -> None:
        self._conn.execute(
            """
            UPDATE subscription_state
            SET tier = ?, status = ?, stripe_customer_id = ?, stripe_subscription_id = ?,
                current_period_end = ?, updated_at = ?
            WHERE id = 1
            """,
            (tier, status, stripe_customer_id, stripe_subscription_id, current_period_end, time.time()),
        )
        self._conn.commit()

    def create_checkout_session(self, tier: str, success_url: str, cancel_url: str) -> Dict[str, Any]:
        secret = (get_env_value("STRIPE_SECRET_KEY") or "").strip()
        if not secret:
            return {"success": False, "error": "STRIPE_SECRET_KEY is not set in ~/.hermes/.env"}

        price_id = self._price_id_for_tier(tier)
        if not price_id:
            return {"success": False, "error": f"No Stripe price configured for tier '{tier}'"}

        import httpx

        data = {
            "mode": "subscription",
            "success_url": success_url,
            "cancel_url": cancel_url,
            "line_items[0][price]": price_id,
            "line_items[0][quantity]": "1",
            "metadata[tier]": tier,
            "metadata[product]": "airies-agent",
        }
        resp = httpx.post(
            "https://api.stripe.com/v1/checkout/sessions",
            data=data,
            auth=(secret, ""),
            timeout=30,
        )
        if resp.status_code >= 400:
            return {"success": False, "error": f"Stripe error {resp.status_code}: {resp.text}"}
        payload = resp.json()
        return {"success": True, "url": payload.get("url", ""), "session_id": payload.get("id", "")}

    def create_billing_portal(self, return_url: str) -> Dict[str, Any]:
        secret = (get_env_value("STRIPE_SECRET_KEY") or "").strip()
        state = self._get_state()
        customer_id = state["stripe_customer_id"] or ""
        if not secret:
            return {"success": False, "error": "STRIPE_SECRET_KEY is not set"}
        if not customer_id:
            return {"success": False, "error": "No Stripe customer on file — subscribe first"}

        import httpx

        resp = httpx.post(
            "https://api.stripe.com/v1/billing_portal/sessions",
            data={"customer": customer_id, "return_url": return_url},
            auth=(secret, ""),
            timeout=30,
        )
        if resp.status_code >= 400:
            return {"success": False, "error": f"Stripe error {resp.status_code}: {resp.text}"}
        return {"success": True, "url": resp.json().get("url", "")}

    def sync_from_stripe(self) -> Dict[str, Any]:
        secret = (get_env_value("STRIPE_SECRET_KEY") or "").strip()
        state = self._get_state()
        sub_id = state["stripe_subscription_id"] or ""
        if not secret:
            return {"success": False, "error": "STRIPE_SECRET_KEY is not set"}
        if not sub_id:
            return {"success": True, "message": "No subscription to sync (free tier)"}

        import httpx

        resp = httpx.get(
            f"https://api.stripe.com/v1/subscriptions/{sub_id}",
            auth=(secret, ""),
            timeout=30,
        )
        if resp.status_code >= 400:
            return {"success": False, "error": f"Stripe error {resp.status_code}: {resp.text}"}
        sub = resp.json()
        tier = (sub.get("metadata") or {}).get("tier") or state["tier"]
        self.set_subscription(
            tier=tier,
            status=sub.get("status", "active"),
            stripe_customer_id=sub.get("customer", state["stripe_customer_id"]),
            stripe_subscription_id=sub.get("id", sub_id),
            current_period_end=float(sub.get("current_period_end") or 0),
        )
        return {"success": True, "tier": tier, "status": sub.get("status")}

    def apply_checkout_session(self, session_id: str) -> Dict[str, Any]:
        secret = (get_env_value("STRIPE_SECRET_KEY") or "").strip()
        if not secret:
            return {"success": False, "error": "STRIPE_SECRET_KEY is not set"}

        import httpx

        resp = httpx.get(
            f"https://api.stripe.com/v1/checkout/sessions/{session_id}",
            auth=(secret, ""),
            timeout=30,
        )
        if resp.status_code >= 400:
            return {"success": False, "error": f"Stripe error {resp.status_code}: {resp.text}"}
        session = resp.json()
        tier = (session.get("metadata") or {}).get("tier") or "pro"
        self.set_subscription(
            tier=tier,
            status="active",
            stripe_customer_id=session.get("customer", ""),
            stripe_subscription_id=session.get("subscription", ""),
        )
        return {"success": True, "tier": tier}

    def handle_webhook_payload(self, payload: bytes, sig_header: str) -> Dict[str, Any]:
        webhook_secret = (get_env_value("STRIPE_WEBHOOK_SECRET") or "").strip()
        if webhook_secret:
            try:
                import hashlib
                import hmac

                parts = dict(p.split("=", 1) for p in sig_header.split(",") if "=" in p)
                timestamp = parts.get("t", "")
                v1 = parts.get("v1", "")
                signed = f"{timestamp}.{payload.decode()}".encode()
                expected = hmac.new(webhook_secret.encode(), signed, hashlib.sha256).hexdigest()
                if not hmac.compare_digest(expected, v1):
                    return {"success": False, "error": "Invalid webhook signature"}
            except Exception as exc:
                return {"success": False, "error": f"Webhook verification failed: {exc}"}

        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            return {"success": False, "error": "Invalid JSON payload"}

        etype = event.get("type", "")
        obj = (event.get("data") or {}).get("object") or {}

        if etype == "checkout.session.completed":
            tier = (obj.get("metadata") or {}).get("tier") or "pro"
            self.set_subscription(
                tier=tier,
                status="active",
                stripe_customer_id=obj.get("customer", ""),
                stripe_subscription_id=obj.get("subscription", ""),
            )
        elif etype in ("customer.subscription.updated", "customer.subscription.deleted"):
            tier = (obj.get("metadata") or {}).get("tier") or self._get_state()["tier"]
            status = obj.get("status", "canceled")
            if etype.endswith("deleted"):
                tier = "free"
                status = "canceled"
            self.set_subscription(
                tier=tier,
                status=status,
                stripe_customer_id=obj.get("customer", self._get_state()["stripe_customer_id"]),
                stripe_subscription_id=obj.get("id", ""),
                current_period_end=float(obj.get("current_period_end") or 0),
            )
        return {"success": True, "type": etype}

    def _price_id_for_tier(self, tier: str) -> str:
        config = load_config()
        key = f"stripe_price_id_{tier}"
        val = cfg_get(config, "subscription", key, default="") or ""
        if val:
            return str(val).strip()
        env_key = f"STRIPE_PRICE_ID_{tier.upper()}"
        return (get_env_value(env_key) or "").strip()

    @staticmethod
    def _subscription_enabled() -> bool:
        config = load_config()
        return bool(cfg_get(config, "subscription", "enabled", default=True))


_manager: Optional[AriesSubscriptionManager] = None


def get_subscription_manager() -> AriesSubscriptionManager:
    global _manager
    if _manager is None:
        _manager = AriesSubscriptionManager()
    return _manager
