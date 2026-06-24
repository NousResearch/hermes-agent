"""Plaid implementation of :class:`~plugins.finance.provider.FinanceProvider`.

Owns the incremental sync engine: balances via ``/accounts/balance/get`` and
transactions via the cursor-based ``/transactions/sync`` loop, with
investments and liabilities pulled best-effort (they require the matching
products to be enabled on the item). Transactions are only persisted after the
final page so a mid-pagination mutation cannot leave a partial write; on
``TRANSACTIONS_SYNC_MUTATION_DURING_PAGINATION`` the loop restarts from the
cursor it began with.

Reference: https://plaid.com/docs/api/products/transactions/#transactionssync
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from plugins.finance import settings
from plugins.finance.errors import FinanceProviderError
from plugins.finance.plaid import auth
from plugins.finance.plaid.client import PlaidClient
from plugins.finance.provider import FinanceProvider
from plugins.finance.store import FinanceStore

logger = logging.getLogger(__name__)

PROVIDER_NAME = "plaid"

# Provider error codes for optional products that simply aren't enabled on an
# item — treated as "no data", not a failure.
_OPTIONAL_PRODUCT_CODES = {
    "PRODUCTS_NOT_SUPPORTED", "PRODUCT_NOT_READY", "NO_INVESTMENT_ACCOUNTS",
    "NO_LIABILITY_ACCOUNTS", "NO_ACCOUNTS",
}

_MAX_SYNC_RESTARTS = 5


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _balance(account: Dict[str, Any], field: str) -> Any:
    return (account.get("balances") or {}).get(field)


def _account_currency(account: Dict[str, Any]) -> Optional[str]:
    balances = account.get("balances") or {}
    return balances.get("iso_currency_code") or balances.get("unofficial_currency_code")


def _normalize_account(account: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "account_id": account.get("account_id"),
        "name": account.get("name"),
        "official_name": account.get("official_name"),
        "mask": account.get("mask"),
        "type": account.get("type"),
        "subtype": account.get("subtype"),
        "currency": _account_currency(account),
        "current_balance": _balance(account, "current"),
        "available_balance": _balance(account, "available"),
        "credit_limit": _balance(account, "limit"),
    }


def _normalize_transaction(txn: Dict[str, Any]) -> Dict[str, Any]:
    pfc = txn.get("personal_finance_category") or {}
    category_list = txn.get("category") or []
    category_primary = pfc.get("primary") or (category_list[0] if category_list else None)
    category_detailed = pfc.get("detailed") or (" > ".join(category_list) if category_list else None)
    return {
        "transaction_id": txn.get("transaction_id"),
        "account_id": txn.get("account_id"),
        "amount": txn.get("amount"),
        "currency": txn.get("iso_currency_code") or txn.get("unofficial_currency_code"),
        "date": txn.get("date"),
        "datetime": txn.get("datetime"),
        "name": txn.get("name"),
        "merchant_name": txn.get("merchant_name"),
        "category_primary": category_primary,
        "category_detailed": category_detailed,
        "pending": txn.get("pending"),
        "payment_channel": txn.get("payment_channel"),
        "raw": txn,
    }


def _normalize_holdings(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "account_id": holding.get("account_id"),
            "security_id": holding.get("security_id"),
            "quantity": holding.get("quantity"),
            "cost_basis": holding.get("cost_basis"),
            "institution_price": holding.get("institution_price"),
            "institution_value": holding.get("institution_value"),
            "currency": holding.get("iso_currency_code") or holding.get("unofficial_currency_code"),
        }
        for holding in response.get("holdings") or []
    ]


def _normalize_securities(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "security_id": sec.get("security_id"),
            "name": sec.get("name"),
            "ticker_symbol": sec.get("ticker_symbol"),
            "type": sec.get("type"),
            "close_price": sec.get("close_price"),
            "currency": sec.get("iso_currency_code") or sec.get("unofficial_currency_code"),
        }
        for sec in response.get("securities") or []
    ]


def _first_apr(detail: Dict[str, Any]) -> Optional[float]:
    aprs = detail.get("aprs") or []
    if aprs and isinstance(aprs[0], dict):
        return aprs[0].get("apr_percentage")
    return None


def _normalize_liabilities(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    liabilities = response.get("liabilities") or {}
    rows: List[Dict[str, Any]] = []
    for kind in ("credit", "student", "mortgage"):
        for detail in liabilities.get(kind) or []:
            rows.append({
                "account_id": detail.get("account_id"),
                "kind": kind,
                "last_payment_amount": detail.get("last_payment_amount"),
                "last_payment_date": detail.get("last_payment_date"),
                "next_payment_due_date": (
                    detail.get("next_payment_due_date") or detail.get("next_monthly_payment")
                ),
                "minimum_payment_amount": detail.get("minimum_payment_amount"),
                "outstanding_balance": (
                    detail.get("last_statement_balance") or detail.get("outstanding_interest_amount")
                ),
                "apr": _first_apr(detail),
                "detail": detail,
            })
    return rows


class PlaidProvider(FinanceProvider):
    name = PROVIDER_NAME

    def __init__(self, environment: Optional[str] = None) -> None:
        self._environment = environment

    def _resolve_environment(self) -> str:
        return (self._environment or settings.get_plaid_environment() or "sandbox").strip().lower()

    def _client(self, client: Optional[PlaidClient] = None) -> PlaidClient:
        return client or auth.build_client(self._resolve_environment())

    # ------------------------------------------------------------------
    # Connect
    # ------------------------------------------------------------------

    def connect(self, **kwargs: Any) -> Dict[str, Any]:
        store: Optional[FinanceStore] = kwargs.get("store")
        mode = (kwargs.get("mode") or "hosted").strip().lower()
        products: Optional[List[str]] = kwargs.get("products")
        client = self._client(kwargs.get("client"))
        environment = self._resolve_environment()
        do_sync = kwargs.get("do_sync", True)

        if mode == "sandbox":
            linked = auth.sandbox_connect(
                client,
                institution_id=kwargs.get("institution_id") or auth.SANDBOX_INSTITUTION_ID,
                products=products,
                environment=environment,
            )
            return self._finalize_link(store, client, linked, do_sync=do_sync)

        started = auth.start_hosted_link(
            client, products=products, webhook=settings.get_plaid_webhook_url() or None,
            user_id=kwargs.get("user_id"),
        )
        on_link_url: Optional[Callable[[str], None]] = kwargs.get("on_link_url")
        if on_link_url:
            on_link_url(started["hosted_link_url"])

        public_token = auth.poll_for_public_token(
            client, started["link_token"],
            timeout=float(kwargs.get("timeout", 300.0)),
            interval=float(kwargs.get("poll_interval", 3.0)),
        )
        if not public_token:
            return {
                "status": "pending",
                "provider": self.name,
                "hosted_link_url": started["hosted_link_url"],
                "link_token": started["link_token"],
                "message": "Link not completed before timeout. Reopen the URL and rerun connect.",
            }
        linked = auth.complete_link(
            client, public_token, environment=environment,
            institution_id=kwargs.get("institution_id"),
        )
        return self._finalize_link(store, client, linked, do_sync=do_sync)

    def _finalize_link(
        self,
        store: Optional[FinanceStore],
        client: PlaidClient,
        linked: Dict[str, Any],
        *,
        do_sync: bool,
    ) -> Dict[str, Any]:
        result = {"status": "linked", "provider": self.name, **linked}
        if store is not None:
            store.upsert_item(
                linked["item_id"], provider=self.name,
                institution_id=linked.get("institution_id"),
                institution_name=linked.get("institution_name"),
                environment=linked.get("environment"),
            )
            if do_sync:
                result["sync"] = self.sync(store, item_id=linked["item_id"], client=client)
        return result

    # ------------------------------------------------------------------
    # Sync
    # ------------------------------------------------------------------

    def sync(self, store: FinanceStore, *, item_id: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        client = self._client(kwargs.get("client"))
        items = [item_id] if item_id else [it["item_id"] for it in auth.list_token_items()]
        per_item = []
        totals = {"accounts": 0, "added": 0, "modified": 0, "removed": 0}
        for current_item in items:
            summary = self._sync_item(store, client, current_item)
            per_item.append(summary)
            for key in totals:
                totals[key] += summary.get(key, 0) or 0
        return {"provider": self.name, "items_synced": len(items), "totals": totals, "items": per_item}

    def _sync_item(self, store: FinanceStore, client: PlaidClient, item_id: str) -> Dict[str, Any]:
        started_at = _utcnow()
        access_token = auth.get_access_token(item_id)
        if not access_token:
            error = "no stored access token"
            store.record_sync_run(
                item_id=item_id, provider=self.name, started_at=started_at,
                finished_at=_utcnow(), status="error", error=error,
            )
            return {"item_id": item_id, "status": "error", "error": error}

        # Ensure the item row exists before writing FK-bound rows (accounts,
        # transactions). A direct `finance sync` — token restored from backup,
        # or a sync that didn't run through connect() — must still self-heal
        # the local item record from the token metadata.
        token_meta = next(
            (it for it in auth.list_token_items() if it.get("item_id") == item_id), {}
        )
        store.upsert_item(
            item_id, provider=self.name,
            institution_id=token_meta.get("institution_id"),
            institution_name=token_meta.get("institution_name"),
            environment=token_meta.get("environment"),
        )

        summary: Dict[str, Any] = {"item_id": item_id, "status": "ok"}
        try:
            balance_response = client.accounts_balance_get(access_token)
            accounts = [_normalize_account(a) for a in balance_response.get("accounts") or []]
            summary["accounts"] = store.upsert_accounts(item_id, self.name, accounts)

            txn_counts = self._sync_transactions(store, client, item_id, access_token)
            summary.update(txn_counts)

            summary["holdings"] = self._sync_holdings(store, client, item_id, access_token)
            summary["liabilities"] = self._sync_liabilities(store, client, item_id, access_token)

            store.mark_item_synced(item_id, error=None)
            store.record_sync_run(
                item_id=item_id, provider=self.name, started_at=started_at,
                finished_at=_utcnow(), status="ok",
                added=txn_counts.get("added", 0), modified=txn_counts.get("modified", 0),
                removed=txn_counts.get("removed", 0),
            )
        except FinanceProviderError as exc:
            store.mark_item_synced(item_id, error=str(exc))
            store.record_sync_run(
                item_id=item_id, provider=self.name, started_at=started_at,
                finished_at=_utcnow(), status="error", error=str(exc),
            )
            summary["status"] = "error"
            summary["error"] = str(exc)
        return summary

    def _sync_transactions(
        self, store: FinanceStore, client: PlaidClient, item_id: str, access_token: str
    ) -> Dict[str, int]:
        original_cursor = store.get_item_cursor(item_id)
        cursor = original_cursor
        added: List[Dict[str, Any]] = []
        modified: List[Dict[str, Any]] = []
        removed: List[str] = []
        restarts = 0
        while True:
            try:
                response = client.transactions_sync(access_token, cursor=cursor)
            except FinanceProviderError as exc:
                if exc.error_code == "TRANSACTIONS_SYNC_MUTATION_DURING_PAGINATION" and restarts < _MAX_SYNC_RESTARTS:
                    restarts += 1
                    cursor = original_cursor
                    added, modified, removed = [], [], []
                    continue
                raise
            added.extend(_normalize_transaction(t) for t in response.get("added") or [])
            modified.extend(_normalize_transaction(t) for t in response.get("modified") or [])
            removed.extend(
                r.get("transaction_id") for r in response.get("removed") or [] if r.get("transaction_id")
            )
            cursor = response.get("next_cursor")
            if not response.get("has_more"):
                break
        # Only persist after the final page so a mutation cannot leave a partial write.
        counts = store.apply_transactions(
            item_id, self.name, added=added, modified=modified, removed=removed
        )
        store.set_item_cursor(item_id, cursor)
        return counts

    def _sync_holdings(
        self, store: FinanceStore, client: PlaidClient, item_id: str, access_token: str
    ) -> int:
        try:
            response = client.investments_holdings_get(access_token)
        except FinanceProviderError as exc:
            if exc.error_code in _OPTIONAL_PRODUCT_CODES:
                return 0
            logger.debug("Plaid holdings sync skipped for %s: %s", item_id, exc)
            return 0
        store.upsert_securities(_normalize_securities(response))
        return store.upsert_holdings(item_id, self.name, _normalize_holdings(response))

    def _sync_liabilities(
        self, store: FinanceStore, client: PlaidClient, item_id: str, access_token: str
    ) -> int:
        try:
            response = client.liabilities_get(access_token)
        except FinanceProviderError as exc:
            if exc.error_code in _OPTIONAL_PRODUCT_CODES:
                return 0
            logger.debug("Plaid liabilities sync skipped for %s: %s", item_id, exc)
            return 0
        return store.upsert_liabilities(item_id, self.name, _normalize_liabilities(response))

    # ------------------------------------------------------------------
    # Status + disconnect
    # ------------------------------------------------------------------

    def status(self, store: FinanceStore) -> Dict[str, Any]:
        token_items = {it["item_id"]: it for it in auth.list_token_items()}
        items = []
        for item in store.list_items(provider=self.name, include_removed=True):
            item_id = item["item_id"]
            accounts = store.get_accounts()
            account_count = sum(1 for a in accounts if a.get("item_id") == item_id)
            token_meta = token_items.get(item_id, {})
            items.append({
                "item_id": item_id,
                "institution_name": item.get("institution_name") or token_meta.get("institution_name"),
                "environment": item.get("environment") or token_meta.get("environment"),
                "status": item.get("status"),
                "last_sync_at": item.get("last_sync_at"),
                "last_error": item.get("last_error"),
                "accounts": account_count,
                "credential_present": item_id in token_items,
            })
        return {"provider": self.name, "items": items, "last_sync": store.last_sync_run()}

    def disconnect(self, store: FinanceStore, item_id: str, *, purge: bool = False) -> Dict[str, Any]:
        access_token = auth.get_access_token(item_id)
        revoked = False
        if access_token:
            try:
                self._client().item_remove(access_token)
                revoked = True
            except FinanceProviderError as exc:
                logger.debug("Plaid item_remove failed for %s: %s", item_id, exc)
        auth.remove_item_token(item_id)
        if purge:
            store.delete_item(item_id)
        else:
            store.set_item_status(item_id, "disconnected")
        return {"item_id": item_id, "revoked": revoked, "purged": purge, "provider": self.name}
