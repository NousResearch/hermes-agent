"""Deterministic service layer for brokerage intent lifecycle management."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from secrets import token_hex
from uuid import uuid4

from brokerage.brokers.base import BrokerAdapter
from brokerage.config import BrokerageSettings
from brokerage.models import TradeEvent, TradeIntent
from brokerage.policy import BrokeragePolicy
from brokerage.storage import SQLiteBrokerageStore


class BrokerageService:
    """Create, confirm, cancel, and inspect brokerage trade intents."""

    def __init__(
        self,
        settings: BrokerageSettings,
        store: SQLiteBrokerageStore,
        policy: BrokeragePolicy,
        broker: BrokerAdapter,
    ):
        self.settings = settings
        self.store = store
        self.policy = policy
        self.broker = broker

    def create_intent(
        self,
        *,
        account_mode: str,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str,
        asset_class: str = "stock",
        limit_price: float | None = None,
        raw_request_text: str | None = None,
        session_id: str | None = None,
        telegram_chat_id: str | None = None,
        market_snapshot: dict | None = None,
    ) -> dict:
        intent = TradeIntent(
            request_id=self._generate_intent_id(),
            account_mode=account_mode,
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            asset_class=asset_class,
            limit_price=limit_price,
            status="pending_confirmation",
        )

        decision = self.policy.validate_new_intent(intent, market_snapshot=market_snapshot)
        if not decision.allowed:
            raise ValueError(decision.reason or "trade intent rejected by policy")

        confirmation_code = self._generate_confirmation_code()
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=self.settings.confirmation_ttl_seconds)
        self.store.create_intent(
            intent,
            confirmation_code=confirmation_code,
            confirmation_expires_at=expires_at,
            raw_request_text=raw_request_text,
            session_id=session_id,
            telegram_chat_id=telegram_chat_id,
        )
        self.store.append_event(TradeEvent(intent_id=intent.request_id, event_type="created", detail=raw_request_text))

        return {
            "intent_id": intent.request_id,
            "status": intent.status,
            "confirmation_code": confirmation_code,
            "preview": self._intent_preview(intent),
            "expires_at": expires_at.isoformat(),
        }

    def confirm_intent(self, intent_id: str, confirmation_text: str, *, now: datetime | None = None) -> dict:
        row = self._require_intent(intent_id)
        if row["status"] != "pending_confirmation":
            raise ValueError("Intent is not in pending_confirmation status")

        intent = self._intent_from_row(row)
        expires_at = self._parse_optional_datetime(row.get("confirmation_expires_at"))
        decision = self.policy.validate_confirmation(
            intent,
            confirmation_text=confirmation_text,
            confirmation_code=row["confirmation_code"],
            expires_at=expires_at,
            now=now,
        )
        if not decision.allowed:
            raise ValueError(decision.reason or "confirmation rejected")

        self.store.update_status(intent_id, "confirmed")
        self.store.append_event(TradeEvent(intent_id=intent_id, event_type="confirmed", detail=confirmation_text))

        try:
            result = self.broker.submit_order(intent)
        except Exception as exc:
            self.store.update_status(intent_id, "submission_error")
            self.store.append_event(
                TradeEvent(intent_id=intent_id, event_type="submission_error", detail=str(exc))
            )
            return {
                "intent_id": intent_id,
                "status": "submission_error",
                "broker_order_id": None,
                "broker_status": None,
                "detail": f"broker submission failed: {exc}",
            }

        if result.accepted:
            self.store.update_status(intent_id, "submitted", ibkr_order_id=result.broker_order_id)
            self.store.append_event(
                TradeEvent(intent_id=intent_id, event_type="submitted", detail=result.broker_status)
            )
            return {
                "intent_id": intent_id,
                "status": "submitted",
                "broker_order_id": result.broker_order_id,
                "broker_status": result.broker_status,
                "detail": result.detail,
            }

        self.store.update_status(intent_id, "rejected")
        self.store.append_event(
            TradeEvent(intent_id=intent_id, event_type="rejected", detail=result.detail or result.broker_status)
        )
        return {
            "intent_id": intent_id,
            "status": "rejected",
            "broker_order_id": result.broker_order_id,
            "broker_status": result.broker_status,
            "detail": result.detail,
        }

    def cancel_intent(self, intent_id: str) -> dict:
        row = self._require_intent(intent_id)
        if row["status"] != "pending_confirmation":
            raise ValueError(f"Only pending_confirmation intents can be cancelled (current: {row['status']})")

        self.store.update_status(intent_id, "cancelled")
        self.store.append_event(TradeEvent(intent_id=intent_id, event_type="cancelled", detail="cancelled by user"))
        updated = self._require_intent(intent_id)
        return updated

    def get_intent(self, intent_id: str) -> dict:
        row = self._require_intent(intent_id)
        return self._refresh_broker_status(row)

    def _refresh_broker_status(self, row: dict) -> dict:
        if row.get("status") != "submitted" or not row.get("ibkr_order_id"):
            return row

        try:
            broker_status = self.broker.get_order_status(
                row["ibkr_order_id"],
                account_mode=row.get("account_mode"),
                expected_quantity=row.get("quantity"),
            )
        except Exception:
            return row

        if not broker_status:
            return row

        live_status = broker_status.get("broker_status") or broker_status.get("status")
        row = dict(row)
        row["broker_status"] = live_status
        if broker_status.get("detail") is not None:
            row["broker_detail"] = broker_status["detail"]

        resolved_status = self._map_broker_status(live_status)
        if resolved_status is None or resolved_status == row["status"]:
            return row

        transitioned = self.store.transition_status(
            row["intent_id"],
            row["status"],
            resolved_status,
        )
        if transitioned:
            self.store.append_event(
                TradeEvent(
                    intent_id=row["intent_id"],
                    event_type=resolved_status,
                    detail=broker_status.get("detail") or live_status,
                )
            )
        refreshed = self._require_intent(row["intent_id"])
        refreshed["broker_status"] = live_status
        if broker_status.get("detail") is not None:
            refreshed["broker_detail"] = broker_status["detail"]
        return refreshed

    def expire_stale_intents(self, *, now: datetime | None = None) -> int:
        timestamp = now or datetime.now(timezone.utc)
        return self.store.expire_pending_confirmations(timestamp)

    def _require_intent(self, intent_id: str) -> dict:
        row = self.store.get_intent(intent_id)
        if row is None:
            raise ValueError(f"Unknown intent: {intent_id}")
        return row

    def _intent_from_row(self, row: dict) -> TradeIntent:
        return TradeIntent(
            request_id=row["intent_id"],
            account_mode=row["account_mode"],
            symbol=row["symbol"],
            side=row["side"],
            quantity=row["quantity"],
            order_type=row["order_type"],
            asset_class=row["asset_class"],
            limit_price=row["limit_price"],
            status=row["status"],
        )

    @staticmethod
    def _map_broker_status(status: str | None) -> str | None:
        if not status:
            return None

        normalized = status.strip().upper()
        if normalized == "FILLED":
            return "filled"
        if normalized in {"CANCELLED", "APICANCELLED"}:
            return "cancelled"
        if normalized == "PARTIALLYFILLED":
            return "submitted"
        if "REJECT" in normalized or normalized == "INACTIVE":
            return "rejected"
        if normalized in {"PENDINGSUBMIT", "PRESUBMITTED", "SUBMITTED", "APIPENDING"}:
            return "submitted"
        return None

    @staticmethod
    def _intent_preview(intent: TradeIntent) -> dict:
        preview = {
            "account_mode": intent.account_mode,
            "side": intent.side,
            "symbol": intent.symbol,
            "quantity": intent.quantity,
            "order_type": intent.order_type,
            "asset_class": intent.asset_class,
        }
        if intent.limit_price is not None:
            preview["limit_price"] = intent.limit_price
        return preview

    @staticmethod
    def _parse_optional_datetime(value: str | None) -> datetime | None:
        if not value:
            return None
        return datetime.fromisoformat(value)

    @staticmethod
    def _generate_intent_id() -> str:
        return f"ti_{uuid4().hex[:12]}"

    @staticmethod
    def _generate_confirmation_code() -> str:
        return f"T-{token_hex(2).upper()}"
