"""Gated, read-mostly ``finance_*`` model tools backed by the local store.

All query tools read the local SQLite cache only — they never call a provider
API, so the model can ask about balances, spending, net worth, cashflow, and
holdings cheaply and offline. The single exception is ``finance_sync``, which
explicitly refreshes data from the configured provider.

The whole toolset is service-gated: it is only exposed when the active
provider's credentials are configured AND at least one account is linked, so
users who don't use finance pay zero schema/prompt-cache cost. Amounts honor
``finance.privacy_mode`` — in ``summarized`` mode exact figures are bucketed
before they ever reach the model. Access tokens are never loaded here.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, List, Optional

from tools.registry import tool_error, tool_result

from plugins.finance import categorize, privacy, settings
from plugins.finance.errors import FinanceError
from plugins.finance.provider import get_provider
from plugins.finance.store import FinanceStore, default_db_path

_TXN_DEFAULT_LIMIT = 50
_TXN_MAX_LIMIT = 500


# ---------------------------------------------------------------------------
# Availability gate
# ---------------------------------------------------------------------------

def _check_finance_available() -> bool:
    """True when the active provider is configured AND an account is linked.

    Deliberately avoids creating ``finance.db`` as a side effect: if the file
    doesn't exist there can be no linked items, so the tools stay hidden.
    """
    try:
        if settings.get_provider_name() == "plaid":
            from plugins.finance.plaid import auth

            if not auth.credentials_present():
                return False
        path = default_db_path()
        if not path.exists():
            return False
        store = FinanceStore(path)
        try:
            return bool(store.list_items())
        finally:
            store.close()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mode() -> str:
    return settings.get_privacy_mode()


def _amt(value: Optional[float], currency: Optional[str], mode: str) -> Any:
    """Return a numeric amount (full mode) or a bucket label (summarized)."""
    if value is None:
        return None
    if privacy.is_summarized(mode):
        return privacy.bucket_amount(value, currency)
    return round(float(value), 2)


def _resolve_range(args: Dict[str, Any], *, default_days: int) -> Dict[str, Optional[str]]:
    start = (args.get("start_date") or "").strip() or None
    end = (args.get("end_date") or "").strip() or None
    if not start:
        days = args.get("days")
        days = int(days) if isinstance(days, (int, float, str)) and str(days).strip() else default_days
        start = (date.today() - timedelta(days=max(1, days))).isoformat()
    return {"start_date": start, "end_date": end}


def _open_store() -> FinanceStore:
    return FinanceStore()


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def handle_finance_accounts(args: Dict[str, Any], **_kw: Any) -> str:
    mode = _mode()
    account_type = (args.get("account_type") or "").strip().lower()
    store = _open_store()
    try:
        accounts = store.get_accounts()
        net_worth = store.net_worth()
    finally:
        store.close()
    rows = []
    for acct in accounts:
        if account_type and (acct.get("type") or "").lower() != account_type:
            continue
        currency = acct.get("currency")
        rows.append({
            "name": acct.get("name"),
            "official_name": acct.get("official_name"),
            "mask": acct.get("mask"),
            "type": acct.get("type"),
            "subtype": acct.get("subtype"),
            "currency": currency,
            "current_balance": _amt(acct.get("current_balance"), currency, mode),
            "available_balance": _amt(acct.get("available_balance"), currency, mode),
        })
    net_view = {
        cur: {k: _amt(v, cur, mode) for k, v in vals.items()}
        for cur, vals in net_worth.items()
    }
    return tool_result({
        "privacy_mode": mode,
        "account_count": len(rows),
        "accounts": rows,
        "net_worth": net_view,
    })


def handle_finance_transactions(args: Dict[str, Any], **_kw: Any) -> str:
    mode = _mode()
    limit = args.get("limit")
    try:
        limit = int(limit) if limit else _TXN_DEFAULT_LIMIT
    except (TypeError, ValueError):
        limit = _TXN_DEFAULT_LIMIT
    limit = max(1, min(limit, _TXN_MAX_LIMIT))
    order = "amount_desc" if (args.get("sort") or "").strip().lower() in {"amount", "largest"} else "date_desc"
    store = _open_store()
    try:
        txns = store.iter_transactions(
            account_id=(args.get("account_id") or "").strip() or None,
            start_date=(args.get("start_date") or "").strip() or None,
            end_date=(args.get("end_date") or "").strip() or None,
            search=(args.get("search") or "").strip() or None,
            include_pending=bool(args.get("include_pending", True)),
            limit=limit,
            order=order,
        )
        categorizer = categorize.Categorizer(store)
    finally:
        store.close()
    rows = []
    for txn in txns:
        currency = txn.get("currency")
        rows.append({
            "date": txn.get("date"),
            "merchant": categorizer.normalize_merchant(txn),
            "name": txn.get("name"),
            "amount": _amt(txn.get("amount"), currency, mode),
            "currency": currency,
            "category": categorizer.resolve_category(txn),
            "pending": bool(txn.get("pending")),
        })
    return tool_result({"privacy_mode": mode, "count": len(rows), "transactions": rows})


def handle_finance_spending(args: Dict[str, Any], **_kw: Any) -> str:
    mode = _mode()
    date_range = _resolve_range(args, default_days=30)
    include_transfers = bool(args.get("include_transfers", False))
    store = _open_store()
    try:
        summary = categorize.spending_by_category(
            store,
            start_date=date_range["start_date"],
            end_date=date_range["end_date"],
            include_transfers=include_transfers,
        )
    finally:
        store.close()
    categories = [
        {"category": row["category"], "amount": _amt(row["amount"], None, mode)}
        for row in summary["categories"]
    ]
    return tool_result({
        "privacy_mode": mode,
        "start_date": summary["start_date"],
        "end_date": summary["end_date"],
        "transactions": summary["transactions"],
        "total": _amt(summary["total"], None, mode),
        "categories": categories,
    })


def handle_finance_net_worth(args: Dict[str, Any], **_kw: Any) -> str:
    mode = _mode()
    include_history = bool(args.get("include_history", False))
    store = _open_store()
    try:
        net_worth = store.net_worth()
        history = store.net_worth_history() if include_history else []
    finally:
        store.close()
    net_view = {
        cur: {k: _amt(v, cur, mode) for k, v in vals.items()}
        for cur, vals in net_worth.items()
    }
    payload: Dict[str, Any] = {"privacy_mode": mode, "net_worth": net_view}
    if include_history:
        payload["history"] = [
            {"date": point["date"], "net_worth": _amt(point["net_worth"], None, mode)}
            for point in history
        ]
    return tool_result(payload)


def handle_finance_cashflow(args: Dict[str, Any], **_kw: Any) -> str:
    mode = _mode()
    months = args.get("months")
    try:
        months = int(months) if months else 3
    except (TypeError, ValueError):
        months = 3
    store = _open_store()
    try:
        result = categorize.cashflow(store, months=months)
    finally:
        store.close()
    series = [
        {
            "month": row["month"],
            "inflow": _amt(row["inflow"], None, mode),
            "outflow": _amt(row["outflow"], None, mode),
            "net": _amt(row["net"], None, mode),
        }
        for row in result["series"]
    ]
    return tool_result({"privacy_mode": mode, "months": result["months"], "series": series})


def handle_finance_holdings(args: Dict[str, Any], **_kw: Any) -> str:
    mode = _mode()
    store = _open_store()
    try:
        holdings = store.get_holdings()
    finally:
        store.close()
    rows = []
    for holding in holdings:
        currency = holding.get("currency")
        rows.append({
            "account": holding.get("account_name"),
            "security": holding.get("security_name"),
            "ticker": holding.get("ticker_symbol"),
            "type": holding.get("security_type"),
            "quantity": holding.get("quantity"),
            "price": _amt(holding.get("institution_price"), currency, mode),
            "value": _amt(holding.get("institution_value"), currency, mode),
            "cost_basis": _amt(holding.get("cost_basis"), currency, mode),
            "currency": currency,
        })
    return tool_result({"privacy_mode": mode, "count": len(rows), "holdings": rows})


def handle_finance_sync(args: Dict[str, Any], **_kw: Any) -> str:
    item_id = (args.get("item_id") or "").strip() or None
    store = _open_store()
    try:
        provider = get_provider(settings.get_provider_name())
        summary = provider.sync(store, item_id=item_id)
    except FinanceError as exc:
        store.close()
        return tool_error(str(exc))
    finally:
        store.close()
    return tool_result({"success": True, **summary})


# ---------------------------------------------------------------------------
# Schemas + registration table
# ---------------------------------------------------------------------------

FINANCE_ACCOUNTS_SCHEMA = {
    "name": "finance_accounts",
    "description": (
        "List the user's linked financial accounts with their balances and a "
        "net-worth summary, read from the local finance cache."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "account_type": {
                "type": "string",
                "description": "Optional filter: depository, credit, loan, investment, brokerage, other.",
            },
        },
    },
}

FINANCE_TRANSACTIONS_SCHEMA = {
    "name": "finance_transactions",
    "description": (
        "Query recent transactions from the local finance cache. Filter by "
        "account, date range, or text search; sort by date or largest amount."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "account_id": {"type": "string", "description": "Restrict to one account id."},
            "start_date": {"type": "string", "description": "ISO date (YYYY-MM-DD) lower bound."},
            "end_date": {"type": "string", "description": "ISO date (YYYY-MM-DD) upper bound."},
            "search": {"type": "string", "description": "Match merchant or description text."},
            "include_pending": {"type": "boolean", "description": "Include pending transactions (default true)."},
            "sort": {"type": "string", "enum": ["date", "amount"], "description": "Sort order (default date)."},
            "limit": {"type": "integer", "description": f"Max rows (default {_TXN_DEFAULT_LIMIT}, max {_TXN_MAX_LIMIT})."},
        },
    },
}

FINANCE_SPENDING_SCHEMA = {
    "name": "finance_spending",
    "description": (
        "Summarize spending by category over a date range (defaults to the "
        "last 30 days). Transfers between the user's own accounts are excluded."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "days": {"type": "integer", "description": "Trailing window in days when no start_date is given."},
            "start_date": {"type": "string", "description": "ISO date (YYYY-MM-DD) lower bound."},
            "end_date": {"type": "string", "description": "ISO date (YYYY-MM-DD) upper bound."},
            "include_transfers": {"type": "boolean", "description": "Include transfer/payment categories (default false)."},
        },
    },
}

FINANCE_NET_WORTH_SCHEMA = {
    "name": "finance_net_worth",
    "description": "Report total assets, liabilities, and net worth by currency, optionally with a daily trend.",
    "parameters": {
        "type": "object",
        "properties": {
            "include_history": {"type": "boolean", "description": "Include the daily net-worth trend (default false)."},
        },
    },
}

FINANCE_CASHFLOW_SCHEMA = {
    "name": "finance_cashflow",
    "description": "Report monthly inflow, outflow, and net cashflow for the trailing months.",
    "parameters": {
        "type": "object",
        "properties": {
            "months": {"type": "integer", "description": "Number of trailing months (default 3)."},
        },
    },
}

FINANCE_HOLDINGS_SCHEMA = {
    "name": "finance_holdings",
    "description": "List investment holdings (securities, quantities, and values) from linked investment accounts.",
    "parameters": {"type": "object", "properties": {}},
}

FINANCE_SYNC_SCHEMA = {
    "name": "finance_sync",
    "description": (
        "Refresh financial data from the configured provider into the local "
        "cache. This is the only finance tool that contacts the provider."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "item_id": {"type": "string", "description": "Optional: sync only this linked item."},
        },
    },
}

# (tool_name, schema, handler, emoji)
FINANCE_TOOLS = [
    ("finance_accounts", FINANCE_ACCOUNTS_SCHEMA, handle_finance_accounts, "🏦"),
    ("finance_transactions", FINANCE_TRANSACTIONS_SCHEMA, handle_finance_transactions, "🧾"),
    ("finance_spending", FINANCE_SPENDING_SCHEMA, handle_finance_spending, "💸"),
    ("finance_net_worth", FINANCE_NET_WORTH_SCHEMA, handle_finance_net_worth, "📈"),
    ("finance_cashflow", FINANCE_CASHFLOW_SCHEMA, handle_finance_cashflow, "💵"),
    ("finance_holdings", FINANCE_HOLDINGS_SCHEMA, handle_finance_holdings, "📊"),
    ("finance_sync", FINANCE_SYNC_SCHEMA, handle_finance_sync, "🔄"),
]
