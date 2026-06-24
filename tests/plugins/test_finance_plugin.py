"""Tests for the bundled finance plugin (``plugins/finance/``).

Covers the local SQLite store, privacy formatting, rules-first categorization,
the Plaid REST client error mapping, secure token storage + Hosted Link
onboarding, the cursor-based incremental sync engine (including the
mid-pagination mutation restart), the gated ``finance_*`` tools, the
``hermes finance`` CLI dispatch, and plugin registration.

Everything is mocked — no live network, no real Plaid credentials.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
from unittest import mock

import pytest

from plugins.finance import categorize, privacy, settings
from plugins.finance.errors import (
    FinanceConfigError,
    FinanceProviderError,
    FinanceReauthRequiredError,
)
from plugins.finance.store import FinanceStore, default_db_path


@pytest.fixture(autouse=True)
def _finance_home(tmp_path, monkeypatch):
    """Predictable, isolated HERMES_HOME for every finance test."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _store(tmp_path):
    return FinanceStore(tmp_path / "finance.db")


# ===========================================================================
# Store
# ===========================================================================

def test_store_initializes_schema(tmp_path):
    store = _store(tmp_path)
    # No items / accounts in a fresh DB; queries return empty, not errors.
    assert store.list_items() == []
    assert store.get_accounts() == []
    assert store.net_worth() == {}
    store.close()


def test_net_worth_assets_minus_liabilities(tmp_path):
    store = _store(tmp_path)
    store.upsert_item("i1", provider="plaid")
    store.upsert_accounts("i1", "plaid", [
        {"account_id": "a", "name": "Checking", "type": "depository", "currency": "USD", "current_balance": 5000.0},
        {"account_id": "b", "name": "Card", "type": "credit", "currency": "USD", "current_balance": 1200.0},
    ])
    nw = store.net_worth()["USD"]
    assert nw["assets"] == 5000.0
    assert nw["liabilities"] == 1200.0
    assert nw["net_worth"] == 3800.0
    store.close()


def test_transactions_append_only_tombstone(tmp_path):
    store = _store(tmp_path)
    store.upsert_item("i1", provider="plaid")
    store.apply_transactions("i1", "plaid", added=[
        {"transaction_id": "t1", "account_id": "a", "amount": 10.0, "date": "2026-06-01", "name": "Shop"},
    ])
    assert len(store.iter_transactions()) == 1
    # Removing tombstones (does not hard-delete): excluded by default, visible
    # when include_removed=True.
    store.apply_transactions("i1", "plaid", removed=["t1"])
    assert store.iter_transactions() == []
    all_rows = store.iter_transactions(include_removed=True)
    assert len(all_rows) == 1
    assert all_rows[0]["removed"] == 1
    assert all_rows[0]["removed_at"] is not None
    store.close()


def test_transactions_upsert_is_idempotent(tmp_path):
    store = _store(tmp_path)
    store.upsert_item("i1", provider="plaid")
    txn = {"transaction_id": "t1", "account_id": "a", "amount": 10.0, "date": "2026-06-01", "name": "Shop"}
    store.apply_transactions("i1", "plaid", added=[txn])
    store.apply_transactions("i1", "plaid", modified=[{**txn, "amount": 12.5, "name": "Shop B"}])
    rows = store.iter_transactions()
    assert len(rows) == 1
    assert rows[0]["amount"] == 12.5
    assert rows[0]["name"] == "Shop B"
    store.close()


def test_balance_snapshots_feed_net_worth_history(tmp_path):
    store = _store(tmp_path)
    store.upsert_item("i1", provider="plaid")
    store.upsert_accounts("i1", "plaid", [
        {"account_id": "a", "name": "Checking", "type": "depository", "currency": "USD", "current_balance": 100.0},
    ])
    history = store.net_worth_history()
    assert history and history[-1]["net_worth"] == 100.0
    store.close()


def test_holdings_and_liabilities_roundtrip(tmp_path):
    store = _store(tmp_path)
    store.upsert_item("i1", provider="plaid")
    store.upsert_securities([{"security_id": "s1", "name": "ACME", "ticker_symbol": "ACME", "type": "equity"}])
    store.upsert_holdings("i1", "plaid", [
        {"account_id": "inv", "security_id": "s1", "quantity": 10, "institution_value": 1000.0, "institution_price": 100.0},
    ])
    holdings = store.get_holdings()
    assert holdings[0]["security_name"] == "ACME"
    assert holdings[0]["institution_value"] == 1000.0

    store.upsert_liabilities("i1", "plaid", [
        {"account_id": "c1", "kind": "credit", "minimum_payment_amount": 35.0,
         "next_payment_due_date": "2026-07-01", "apr": 19.99, "detail": {"is_overdue": False}},
    ])
    liabilities = store.get_liabilities()
    assert liabilities[0]["kind"] == "credit"
    assert liabilities[0]["apr"] == 19.99
    assert liabilities[0]["detail"] == {"is_overdue": False}
    store.close()


def test_merchant_rules_and_overrides_crud(tmp_path):
    store = _store(tmp_path)
    rule_id = store.add_merchant_rule(pattern="sbux", normalized_name="Starbucks", category="Coffee")
    assert any(r["id"] == rule_id for r in store.list_merchant_rules())
    store.set_category_override(scope="merchant", key="Starbucks", category="Treats")
    overrides = {o["key"]: o["category"] for o in store.list_category_overrides()}
    assert overrides["starbucks"] == "Treats"
    assert store.delete_merchant_rule(rule_id) is True
    store.close()


# ===========================================================================
# Privacy
# ===========================================================================

@pytest.mark.parametrize("value,expected", [
    (0, "$0"),
    (42, "<$100"),
    (250, "$100-$500"),
    (2500, "$2k-$5k"),
    (1_500_000, "$1M+"),
    (-2500, "-$2k-$5k"),
])
def test_bucket_amount(value, expected):
    assert privacy.bucket_amount(value, "USD") == expected


def test_format_money_modes():
    assert privacy.format_money(2500.5, "USD", mode="full") == "$2,500.50"
    assert privacy.format_money(2500.5, "USD", mode="summarized") == "$2k-$5k"
    assert privacy.format_money(None, "USD") == "n/a"


def test_non_usd_symbol():
    assert privacy.bucket_amount(2500, "EUR") == "EUR 2k-EUR 5k"


def test_normalize_mode():
    assert privacy.normalize_mode("summary") == privacy.SUMMARIZED
    assert privacy.normalize_mode("") == privacy.FULL
    assert privacy.is_summarized("summarized") is True


# ===========================================================================
# Categorization
# ===========================================================================

def test_resolve_category_priority(tmp_path):
    store = _store(tmp_path)
    # Merchant rule + a Plaid category present.
    store.add_merchant_rule(pattern="starbucks", category="Coffee")
    txn = {"transaction_id": "t1", "merchant_name": "Starbucks", "category_primary": "FOOD_AND_DRINK"}
    cat = categorize.Categorizer(store)
    # Plaid category outranks the merchant rule (issue order).
    assert cat.resolve_category(txn) == "FOOD_AND_DRINK"

    # A user override outranks everything.
    store.set_category_override(scope="merchant", key="Starbucks", category="Treats")
    cat = categorize.Categorizer(store)
    assert cat.resolve_category(txn) == "Treats"
    store.close()


def test_resolve_category_falls_back_to_rule_then_uncategorized(tmp_path):
    store = _store(tmp_path)
    store.add_merchant_rule(pattern="acme", category="Shopping")
    cat = categorize.Categorizer(store)
    assert cat.resolve_category({"transaction_id": "t", "name": "ACME STORE"}) == "Shopping"
    assert cat.resolve_category({"transaction_id": "u", "name": "Mystery"}) == categorize.UNCATEGORIZED
    store.close()


def test_normalize_merchant_uses_rule(tmp_path):
    store = _store(tmp_path)
    store.add_merchant_rule(pattern="sq *coffee", normalized_name="Corner Coffee")
    cat = categorize.Categorizer(store)
    assert cat.normalize_merchant({"name": "SQ *COFFEE 123"}) == "Corner Coffee"
    store.close()


def test_spending_excludes_transfers(tmp_path):
    store = _store(tmp_path)
    store.upsert_item("i1", provider="plaid")
    store.apply_transactions("i1", "plaid", added=[
        {"transaction_id": "t1", "account_id": "a", "amount": 30.0, "date": "2026-06-01", "name": "Lunch", "category_primary": "FOOD_AND_DRINK"},
        {"transaction_id": "t2", "account_id": "a", "amount": 100.0, "date": "2026-06-02", "name": "Move", "category_primary": "TRANSFER_OUT"},
        {"transaction_id": "t3", "account_id": "a", "amount": -500.0, "date": "2026-06-03", "name": "Paycheck", "category_primary": "INCOME"},
    ])
    summary = categorize.spending_by_category(store, start_date="2026-05-01")
    cats = {c["category"]: c["amount"] for c in summary["categories"]}
    assert cats == {"FOOD_AND_DRINK": 30.0}  # transfer + income (negative) excluded
    assert summary["total"] == 30.0

    summary_inc = categorize.spending_by_category(store, start_date="2026-05-01", include_transfers=True)
    assert "TRANSFER_OUT" in {c["category"] for c in summary_inc["categories"]}
    store.close()


def test_cashflow_buckets_by_month(tmp_path):
    store = _store(tmp_path)
    store.upsert_item("i1", provider="plaid")
    store.apply_transactions("i1", "plaid", added=[
        {"transaction_id": "t1", "account_id": "a", "amount": -2000.0, "date": "2026-06-01", "name": "Pay"},
        {"transaction_id": "t2", "account_id": "a", "amount": 500.0, "date": "2026-06-15", "name": "Rent"},
    ])
    result = categorize.cashflow(store, months=12)
    june = next(row for row in result["series"] if row["month"] == "2026-06")
    assert june["inflow"] == 2000.0
    assert june["outflow"] == 500.0
    assert june["net"] == 1500.0
    store.close()


# ===========================================================================
# Plaid client (error mapping)
# ===========================================================================

class _FakeResp:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def _client():
    from plugins.finance.plaid.client import PlaidClient

    return PlaidClient("cid", "secret", environment="sandbox")


def test_client_requires_credentials():
    from plugins.finance.plaid.client import PlaidClient

    with pytest.raises(FinanceConfigError):
        PlaidClient("", "", environment="sandbox")


def test_client_base_url_by_environment():
    from plugins.finance.plaid.client import PlaidClient

    assert PlaidClient("c", "s", environment="sandbox").base_url.endswith("sandbox.plaid.com")
    assert PlaidClient("c", "s", environment="production").base_url.endswith("production.plaid.com")


def test_client_success():
    with mock.patch("plugins.finance.plaid.client.httpx.post", return_value=_FakeResp(200, {"ok": True})):
        assert _client()._post("/x", {"a": 1}) == {"ok": True}


def test_client_maps_reauth_error():
    payload = {"error_code": "ITEM_LOGIN_REQUIRED", "error_message": "login"}
    with mock.patch("plugins.finance.plaid.client.httpx.post", return_value=_FakeResp(400, payload)):
        with pytest.raises(FinanceReauthRequiredError):
            _client()._post("/x")


def test_client_maps_config_error():
    payload = {"error_code": "INVALID_API_KEYS", "error_message": "bad keys"}
    with mock.patch("plugins.finance.plaid.client.httpx.post", return_value=_FakeResp(400, payload)):
        with pytest.raises(FinanceConfigError):
            _client()._post("/x")


def test_client_maps_generic_error_with_code():
    payload = {"error_code": "RATE_LIMIT_EXCEEDED", "error_message": "slow down", "request_id": "r1"}
    with mock.patch("plugins.finance.plaid.client.httpx.post", return_value=_FakeResp(429, payload)):
        with pytest.raises(FinanceProviderError) as exc:
            _client()._post("/x")
    assert exc.value.error_code == "RATE_LIMIT_EXCEEDED"
    assert exc.value.request_id == "r1"


# ===========================================================================
# Plaid auth (token storage + onboarding)
# ===========================================================================

def test_token_storage_roundtrip_and_perms():
    from plugins.finance.plaid import auth

    auth.save_item_token("itm1", "access-secret", institution_name="Bank", environment="sandbox")
    assert auth.get_access_token("itm1") == "access-secret"

    # Metadata listing must NOT leak the access token.
    items = auth.list_token_items()
    assert items[0]["item_id"] == "itm1"
    assert "access_token" not in items[0]

    # File is 0600.
    if os.name == "posix":
        mode = stat.S_IMODE(os.stat(auth.token_store_path()).st_mode)
        assert mode == 0o600

    assert auth.remove_item_token("itm1") is True
    assert auth.get_access_token("itm1") is None


def test_credentials_present(monkeypatch):
    from plugins.finance.plaid import auth

    monkeypatch.delenv("PLAID_CLIENT_ID", raising=False)
    monkeypatch.delenv("PLAID_SECRET", raising=False)
    assert auth.credentials_present() is False
    monkeypatch.setenv("PLAID_CLIENT_ID", "cid")
    monkeypatch.setenv("PLAID_SECRET", "secret")
    assert auth.credentials_present() is True
    assert auth.get_plaid_credentials() == ("cid", "secret")


def test_extract_and_poll_public_token():
    from plugins.finance.plaid import auth

    completed = {"link_sessions": [{"results": {"item_add_results": [{"public_token": "pub-123"}]}}]}
    assert auth._extract_public_token(completed) == "pub-123"
    assert auth._extract_public_token({"link_sessions": []}) is None

    class _LinkClient:
        def __init__(self):
            self.calls = 0

        def link_token_get(self, link_token):
            self.calls += 1
            return completed if self.calls >= 2 else {"link_sessions": []}

    client = _LinkClient()
    token = auth.poll_for_public_token(client, "lt", timeout=10, interval=0, sleep=lambda *_: None)
    assert token == "pub-123"


# ===========================================================================
# Plaid provider (sync engine)
# ===========================================================================

class FakePlaidClient:
    """Cursor-keyed fake of the Plaid REST client for sync tests."""

    def __init__(self, *, pages, balance, holdings=None, liabilities=None,
                 holdings_error=None, liabilities_error=None, mutation_cursors=()):
        self.pages = pages  # {cursor_or_None: response_dict}
        self.balance = balance
        self.holdings = holdings or {"holdings": [], "securities": []}
        self.liabilities = liabilities or {"liabilities": {}}
        self.holdings_error = holdings_error
        self.liabilities_error = liabilities_error
        self._mutation_cursors = set(mutation_cursors)
        self.removed_called = False

    def accounts_balance_get(self, access_token):
        return self.balance

    def transactions_sync(self, access_token, *, cursor=None, count=500):
        if cursor in self._mutation_cursors:
            self._mutation_cursors.discard(cursor)
            raise FinanceProviderError(
                "mutation", error_code="TRANSACTIONS_SYNC_MUTATION_DURING_PAGINATION"
            )
        return self.pages[cursor]

    def investments_holdings_get(self, access_token):
        if self.holdings_error:
            raise self.holdings_error
        return self.holdings

    def liabilities_get(self, access_token):
        if self.liabilities_error:
            raise self.liabilities_error
        return self.liabilities

    def item_remove(self, access_token):
        self.removed_called = True
        return {}


def _balance_payload():
    return {"accounts": [
        {"account_id": "a1", "name": "Checking", "type": "depository", "subtype": "checking",
         "balances": {"current": 2500.0, "available": 2400.0, "iso_currency_code": "USD"}},
    ]}


def _txn(tid, amount=10.0):
    return {"transaction_id": tid, "account_id": "a1", "amount": amount, "date": "2026-06-01",
            "name": f"txn-{tid}", "personal_finance_category": {"primary": "GENERAL_MERCHANDISE"}}


def test_provider_sync_paginates_and_persists(tmp_path):
    from plugins.finance.plaid import auth
    from plugins.finance.plaid.provider import PlaidProvider

    auth.save_item_token("item1", "tok", environment="sandbox")
    pages = {
        None: {"added": [_txn("A")], "modified": [], "removed": [], "next_cursor": "c1", "has_more": True},
        "c1": {"added": [_txn("B")], "modified": [], "removed": [], "next_cursor": "c2", "has_more": False},
    }
    fake = FakePlaidClient(pages=pages, balance=_balance_payload())
    store = _store(tmp_path)
    provider = PlaidProvider(environment="sandbox")
    summary = provider.sync(store, item_id="item1", client=fake)

    assert summary["totals"]["added"] == 2
    assert summary["totals"]["accounts"] == 1
    assert store.get_item_cursor("item1") == "c2"
    assert {t["transaction_id"] for t in store.iter_transactions()} == {"A", "B"}
    store.close()


def test_provider_sync_restarts_on_mutation(tmp_path):
    from plugins.finance.plaid import auth
    from plugins.finance.plaid.provider import PlaidProvider

    auth.save_item_token("item1", "tok", environment="sandbox")
    pages = {
        None: {"added": [_txn("A")], "modified": [], "removed": [], "next_cursor": "c1", "has_more": True},
        "c1": {"added": [_txn("B")], "modified": [], "removed": [], "next_cursor": "c2", "has_more": False},
    }
    # Raise a mutation error the first time we request cursor c1; the loop must
    # restart from the original cursor and not double-count "A".
    fake = FakePlaidClient(pages=pages, balance=_balance_payload(), mutation_cursors={"c1"})
    store = _store(tmp_path)
    provider = PlaidProvider(environment="sandbox")
    summary = provider.sync(store, item_id="item1", client=fake)

    assert summary["totals"]["added"] == 2
    assert {t["transaction_id"] for t in store.iter_transactions()} == {"A", "B"}
    store.close()


def test_provider_sync_optional_products_are_best_effort(tmp_path):
    from plugins.finance.plaid import auth
    from plugins.finance.plaid.provider import PlaidProvider

    auth.save_item_token("item1", "tok", environment="sandbox")
    pages = {None: {"added": [], "modified": [], "removed": [], "next_cursor": "c1", "has_more": False}}
    fake = FakePlaidClient(
        pages=pages, balance=_balance_payload(),
        holdings_error=FinanceProviderError("no inv", error_code="NO_INVESTMENT_ACCOUNTS"),
        liabilities_error=FinanceProviderError("no liab", error_code="PRODUCTS_NOT_SUPPORTED"),
    )
    store = _store(tmp_path)
    provider = PlaidProvider(environment="sandbox")
    summary = provider.sync(store, item_id="item1", client=fake)
    item_summary = summary["items"][0]
    assert item_summary["status"] == "ok"
    assert item_summary["holdings"] == 0
    assert item_summary["liabilities"] == 0
    store.close()


def test_provider_sync_records_error_when_no_token(tmp_path):
    from plugins.finance.plaid.provider import PlaidProvider

    store = _store(tmp_path)
    provider = PlaidProvider(environment="sandbox")
    # No saved token for this item id.
    summary = provider.sync(store, item_id="ghost", client=FakePlaidClient(pages={}, balance={}))
    assert summary["items"][0]["status"] == "error"
    assert store.last_sync_run()["status"] == "error"
    store.close()


def test_provider_disconnect_revokes_and_marks(tmp_path, monkeypatch):
    from plugins.finance.plaid import auth
    from plugins.finance.plaid import provider as provider_mod
    from plugins.finance.plaid.provider import PlaidProvider

    auth.save_item_token("item1", "tok", environment="sandbox")
    store = _store(tmp_path)
    store.upsert_item("item1", provider="plaid")
    fake = FakePlaidClient(pages={}, balance={})
    monkeypatch.setattr(provider_mod.auth, "build_client", lambda env: fake)

    result = PlaidProvider(environment="sandbox").disconnect(store, "item1")
    assert result["revoked"] is True
    assert fake.removed_called is True
    assert auth.get_access_token("item1") is None
    assert store.get_item("item1")["status"] == "disconnected"
    store.close()


def test_provider_status_reports_items(tmp_path):
    from plugins.finance.plaid import auth
    from plugins.finance.plaid.provider import PlaidProvider

    auth.save_item_token("item1", "tok", institution_name="Bank", environment="sandbox")
    store = _store(tmp_path)
    store.upsert_item("item1", provider="plaid", institution_name="Bank", environment="sandbox")
    status = PlaidProvider(environment="sandbox").status(store)
    assert status["items"][0]["item_id"] == "item1"
    assert status["items"][0]["credential_present"] is True
    store.close()


# ===========================================================================
# Tools (gating + privacy)
# ===========================================================================

def _seed_default_store():
    store = FinanceStore()  # default (isolated HERMES_HOME) path
    store.upsert_item("item1", provider="plaid", institution_name="Bank", environment="sandbox")
    store.upsert_accounts("item1", "plaid", [
        {"account_id": "a1", "name": "Checking", "type": "depository", "currency": "USD", "current_balance": 2500.0, "available_balance": 2400.0},
        {"account_id": "c1", "name": "Card", "type": "credit", "currency": "USD", "current_balance": 800.0},
    ])
    store.apply_transactions("item1", "plaid", added=[
        {"transaction_id": "t1", "account_id": "a1", "amount": 54.2, "date": "2026-06-01", "name": "Coffee", "merchant_name": "Starbucks", "category_primary": "FOOD_AND_DRINK"},
        {"transaction_id": "t2", "account_id": "a1", "amount": -2000.0, "date": "2026-06-02", "name": "Payroll", "category_primary": "INCOME"},
    ])
    store.close()


def test_tool_gate_requires_creds_and_items(monkeypatch):
    from plugins.finance import tools

    monkeypatch.delenv("PLAID_CLIENT_ID", raising=False)
    monkeypatch.delenv("PLAID_SECRET", raising=False)
    # No creds, no db.
    assert tools._check_finance_available() is False

    monkeypatch.setenv("PLAID_CLIENT_ID", "cid")
    monkeypatch.setenv("PLAID_SECRET", "sec")
    # Creds present but no linked items yet -> still hidden.
    assert tools._check_finance_available() is False

    _seed_default_store()
    assert tools._check_finance_available() is True


def test_tool_accounts_full_and_summarized(monkeypatch):
    from plugins.finance import tools

    _seed_default_store()
    payload = json.loads(tools.handle_finance_accounts({}))
    assert payload["account_count"] == 2
    assert payload["net_worth"]["USD"]["net_worth"] == 1700.0  # 2500 - 800

    monkeypatch.setattr(settings, "get_privacy_mode", lambda: "summarized")
    summarized = json.loads(tools.handle_finance_accounts({}))
    assert summarized["privacy_mode"] == "summarized"
    assert summarized["net_worth"]["USD"]["net_worth"] == "$1k-$2k"
    # No raw float leaks in summarized mode.
    assert isinstance(summarized["accounts"][0]["current_balance"], str)


def test_tool_transactions_and_spending(monkeypatch):
    from plugins.finance import tools

    _seed_default_store()
    txns = json.loads(tools.handle_finance_transactions({"limit": 10}))
    assert txns["count"] == 2
    assert txns["transactions"][0]["category"] in {"FOOD_AND_DRINK", "INCOME"}

    spending = json.loads(tools.handle_finance_spending({"start_date": "2026-05-01"}))
    cats = {c["category"]: c["amount"] for c in spending["categories"]}
    assert cats == {"FOOD_AND_DRINK": 54.2}  # income (negative) + no transfers


def test_tool_cashflow_and_net_worth_history():
    from plugins.finance import tools

    _seed_default_store()
    cashflow = json.loads(tools.handle_finance_cashflow({"months": 12}))
    assert any(row["month"] == "2026-06" for row in cashflow["series"])

    nw = json.loads(tools.handle_finance_net_worth({"include_history": True}))
    assert nw["net_worth"]["USD"]["net_worth"] == 1700.0
    assert isinstance(nw["history"], list)


def test_tool_sync_invokes_provider(monkeypatch):
    from plugins.finance import tools

    _seed_default_store()

    class _FakeProvider:
        def sync(self, store, *, item_id=None, **kw):
            return {"provider": "plaid", "items_synced": 1, "totals": {"added": 3}}

    monkeypatch.setattr(tools, "get_provider", lambda name: _FakeProvider())
    result = json.loads(tools.handle_finance_sync({}))
    assert result["success"] is True
    assert result["items_synced"] == 1


def test_tool_sync_surfaces_finance_error(monkeypatch):
    from plugins.finance import tools

    _seed_default_store()

    class _BadProvider:
        def sync(self, store, *, item_id=None, **kw):
            raise FinanceConfigError("not configured")

    monkeypatch.setattr(tools, "get_provider", lambda name: _BadProvider())
    result = json.loads(tools.handle_finance_sync({}))
    assert "error" in result


# ===========================================================================
# CLI dispatch
# ===========================================================================

def _finance_parser():
    from plugins.finance.cli import register_cli

    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest="cmd")
    finance = subs.add_parser("finance")
    register_cli(finance)
    return parser


def test_cli_disconnect_positional_parsing():
    parser = _finance_parser()
    args = parser.parse_args(["finance", "disconnect", "plaid", "item-9", "--purge"])
    assert args.provider == "plaid"
    assert args.item_id == "item-9"
    assert args.purge is True
    # Provider defaults to plaid when only the item id is supplied.
    args2 = parser.parse_args(["finance", "disconnect", "item-9"])
    assert args2.provider == "plaid"
    assert args2.item_id == "item-9"


def test_cli_status_dispatch(monkeypatch, capsys):
    from plugins.finance import cli

    class _FakeProvider:
        def status(self, store):
            return {"items": [{"item_id": "i1", "institution_name": "Bank", "environment": "sandbox", "status": "active", "accounts": 2}]}

    monkeypatch.setattr(cli, "get_provider", lambda name: _FakeProvider())
    parser = _finance_parser()
    args = parser.parse_args(["finance", "status", "plaid"])
    assert args.func(args) == 0
    assert "Bank" in capsys.readouterr().out


def test_cli_accounts_reads_store(capsys):
    from plugins.finance import cli

    _seed_default_store()
    parser = _finance_parser()
    args = parser.parse_args(["finance", "accounts"])
    assert args.func(args) == 0
    out = capsys.readouterr().out
    assert "Checking" in out
    assert "Net worth" in out


def test_cli_connect_sandbox(monkeypatch, capsys):
    from plugins.finance import cli

    class _FakeProvider:
        def connect(self, **kwargs):
            return {"status": "linked", "provider": "plaid", "item_id": "i1"}

    monkeypatch.setattr(cli, "get_provider", lambda name: _FakeProvider())
    parser = _finance_parser()
    args = parser.parse_args(["finance", "connect", "plaid", "--sandbox", "--no-sync"])
    assert args.func(args) == 0
    assert "linked" in capsys.readouterr().out


def test_cli_missing_credentials_prints_hint(monkeypatch, capsys):
    from plugins.finance import cli

    def _raise(name):
        raise FinanceConfigError("Plaid is not configured.")

    monkeypatch.setattr(cli, "get_provider", _raise)
    parser = _finance_parser()
    args = parser.parse_args(["finance", "sync", "plaid"])
    assert args.func(args) == 1
    out = capsys.readouterr().out
    assert "PLAID_CLIENT_ID" in out


# ===========================================================================
# Plugin registration
# ===========================================================================

def test_register_wires_cli_and_tools():
    import plugins.finance as finance_plugin

    class _Ctx:
        def __init__(self):
            self.tools = []
            self.cli_command = None

        def register_tool(self, **kwargs):
            self.tools.append(kwargs["name"])

        def register_cli_command(self, **kwargs):
            self.cli_command = kwargs["name"]

    ctx = _Ctx()
    finance_plugin.register(ctx)
    assert ctx.cli_command == "finance"
    assert set(ctx.tools) == {
        "finance_accounts", "finance_transactions", "finance_spending",
        "finance_net_worth", "finance_cashflow", "finance_holdings", "finance_sync",
    }


def test_config_defaults_have_finance_section():
    from hermes_cli.config import DEFAULT_CONFIG, OPTIONAL_ENV_VARS

    finance = DEFAULT_CONFIG["finance"]
    assert finance["provider"] == "plaid"
    assert finance["plaid"]["environment"] == "sandbox"
    assert finance["privacy_mode"] == "full"
    assert "PLAID_CLIENT_ID" in OPTIONAL_ENV_VARS
    assert "PLAID_SECRET" in OPTIONAL_ENV_VARS


def test_finance_is_configurable_and_default_off():
    from hermes_cli.tools_config import CONFIGURABLE_TOOLSETS, _DEFAULT_OFF_TOOLSETS

    keys = {ts_key for ts_key, _, _ in CONFIGURABLE_TOOLSETS}
    assert "finance" in keys
    assert "finance" in _DEFAULT_OFF_TOOLSETS


def test_finance_setup_category_prompts_for_plaid_keys():
    """Enabling finance in `hermes tools` must surface the Plaid credential
    prompts, like spotify/homeassistant — not silently toggle on with no setup."""
    from hermes_cli.tools_config import TOOL_CATEGORIES

    entry = TOOL_CATEGORIES["finance"]
    prompted = {
        ev["key"]
        for provider in entry["providers"]
        for ev in provider.get("env_vars", [])
    }
    assert {"PLAID_CLIENT_ID", "PLAID_SECRET"} <= prompted
