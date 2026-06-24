"""Rules-first transaction categorization and spending analytics.

Resolution order for a transaction's category (first match wins):

1. **User override, transaction-scoped** — an explicit per-transaction label.
2. **User override, merchant-scoped** — an explicit label for a merchant name.
3. **Provider category** — Plaid's ``personal_finance_category`` primary.
4. **Merchant rule** — a local ``merchant_rules`` row whose pattern matches and
   carries a category.
5. **LLM fallback** — optional, *off by default*; only ever runs for otherwise
   uncategorized transactions and never on the sync hot path.
6. ``Uncategorized``.

User overrides sit above the automatic sources on purpose: an "override" that
a provider category could outvote would not be an override. Among the
automatic sources the order follows issue #51697 (provider category, then
local merchant rules).

All resolution is in-memory and cheap; merchant rules and overrides are loaded
once per :class:`Categorizer` instance so a batch of transactions does not
re-query SQLite per row.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any, Callable, Dict, List, Optional

UNCATEGORIZED = "Uncategorized"

# Resolved categories that represent money moving between a user's own
# accounts rather than true spending. Excluded from spending totals by default.
_TRANSFER_HINTS = ("transfer", "payment", "credit card payment")


class Categorizer:
    """Resolves categories and normalized merchant names from local rules."""

    def __init__(self, store: Any, *, llm_fallback: Optional[Callable[[Dict[str, Any]], Optional[str]]] = None) -> None:
        self._rules = store.list_merchant_rules()
        self._overrides: Dict[str, str] = {}
        for row in store.list_category_overrides():
            self._overrides[f"{row['scope']}:{row['key']}"] = row["category"]
        self._llm_fallback = llm_fallback

    # -- internal helpers ------------------------------------------------

    @staticmethod
    def _haystack(txn: Dict[str, Any]) -> str:
        return (txn.get("merchant_name") or txn.get("name") or "").lower()

    def _matching_rule(self, txn: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        haystack = self._haystack(txn)
        if not haystack:
            return None
        for rule in self._rules:
            pattern = (rule.get("pattern") or "").lower()
            if not pattern:
                continue
            match_type = (rule.get("match_type") or "substring").lower()
            try:
                if match_type == "exact" and haystack == pattern:
                    return rule
                if match_type == "regex" and re.search(rule["pattern"], haystack, re.IGNORECASE):
                    return rule
                if match_type == "substring" and pattern in haystack:
                    return rule
            except re.error:
                continue
        return None

    # -- public API ------------------------------------------------------

    def normalize_merchant(self, txn: Dict[str, Any]) -> str:
        """Return a clean display name, applying a matching rule if present."""
        rule = self._matching_rule(txn)
        if rule and rule.get("normalized_name"):
            return rule["normalized_name"]
        return txn.get("merchant_name") or txn.get("name") or "Unknown"

    def resolve_category(self, txn: Dict[str, Any]) -> str:
        txn_id = (txn.get("transaction_id") or "").lower()
        if txn_id and f"transaction:{txn_id}" in self._overrides:
            return self._overrides[f"transaction:{txn_id}"]
        merchant_key = (txn.get("merchant_name") or txn.get("name") or "").lower()
        if merchant_key and f"merchant:{merchant_key}" in self._overrides:
            return self._overrides[f"merchant:{merchant_key}"]
        if txn.get("category_primary"):
            return txn["category_primary"]
        rule = self._matching_rule(txn)
        if rule and rule.get("category"):
            return rule["category"]
        if self._llm_fallback is not None:
            try:
                guess = self._llm_fallback(txn)
            except Exception:
                guess = None
            if guess:
                return guess
        return UNCATEGORIZED


def _is_transfer(category: str) -> bool:
    lowered = category.lower()
    return any(hint in lowered for hint in _TRANSFER_HINTS)


def _subtract_months(anchor: date, months: int) -> date:
    month_index = anchor.year * 12 + (anchor.month - 1) - months
    year, month = divmod(month_index, 12)
    return date(year, month + 1, 1)


def spending_by_category(
    store: Any,
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    include_transfers: bool = False,
    categorizer: Optional[Categorizer] = None,
) -> Dict[str, Any]:
    """Aggregate outflows by resolved category over a date range.

    Outflow follows Plaid's sign convention: a positive transaction amount is
    money leaving the account. Transfers/payments between the user's own
    accounts are excluded unless *include_transfers* is set.
    """
    categorizer = categorizer or Categorizer(store)
    txns = store.iter_transactions(start_date=start_date, end_date=end_date)
    totals: Dict[str, float] = {}
    transactions_counted = 0
    for txn in txns:
        amount = txn.get("amount")
        if amount is None or amount <= 0:
            continue
        category = categorizer.resolve_category(txn)
        if not include_transfers and _is_transfer(category):
            continue
        totals[category] = totals.get(category, 0.0) + float(amount)
        transactions_counted += 1
    categories = [
        {"category": cat, "amount": round(total, 2)}
        for cat, total in sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    ]
    return {
        "start_date": start_date,
        "end_date": end_date,
        "total": round(sum(totals.values()), 2),
        "transactions": transactions_counted,
        "categories": categories,
    }


def cashflow(store: Any, *, months: int = 3) -> Dict[str, Any]:
    """Return monthly inflow/outflow/net for the trailing *months* window."""
    months = max(1, int(months))
    start = _subtract_months(date.today().replace(day=1), months - 1)
    txns = store.iter_transactions(start_date=start.isoformat())
    buckets: Dict[str, Dict[str, float]] = {}
    for txn in txns:
        amount = txn.get("amount")
        txn_date = txn.get("date")
        if amount is None or not txn_date:
            continue
        month = txn_date[:7]
        bucket = buckets.setdefault(month, {"inflow": 0.0, "outflow": 0.0})
        if amount > 0:
            bucket["outflow"] += float(amount)
        else:
            bucket["inflow"] += abs(float(amount))
    series = []
    for month in sorted(buckets):
        bucket = buckets[month]
        series.append({
            "month": month,
            "inflow": round(bucket["inflow"], 2),
            "outflow": round(bucket["outflow"], 2),
            "net": round(bucket["inflow"] - bucket["outflow"], 2),
        })
    return {"months": months, "series": series}
