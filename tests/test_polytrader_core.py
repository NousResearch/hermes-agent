import importlib
import sys
from types import SimpleNamespace


class FakeOrderArgs:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeOptions:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeOrderType:
    GTC = "GTC"


class FakeSide:
    BUY = "BUY"
    SELL = "SELL"


class FakeApiCreds:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeClobClient:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.__class__.instances.append(self)


def install_fake_clob_v2(monkeypatch):
    FakeClobClient.instances = []
    fake = SimpleNamespace(
        ApiCreds=FakeApiCreds,
        ClobClient=FakeClobClient,
        OrderArgsV2=FakeOrderArgs,
        OrderArgs=FakeOrderArgs,
        PartialCreateOrderOptions=FakeOptions,
        OrderType=FakeOrderType,
        Side=FakeSide,
    )
    monkeypatch.setitem(sys.modules, "py_clob_client_v2", fake)
    monkeypatch.setitem(sys.modules, "py_clob_client_v2.clob_types", fake)
    return fake


def test_settings_default_to_dry_run_and_safe_address_sets_signature_type(monkeypatch):
    from polytrader.config import Settings

    monkeypatch.delenv("DRY_RUN", raising=False)
    monkeypatch.setenv("SAFE_ADDRESS", "0x0000000000000000000000000000000000000abc")
    monkeypatch.delenv("FUNDER_ADDRESS", raising=False)
    monkeypatch.delenv("SIGNATURE_TYPE", raising=False)

    settings = Settings.from_env()

    assert settings.dry_run is True
    assert settings.safe_address == "0x0000000000000000000000000000000000000abc"
    assert settings.funder_address == settings.safe_address
    assert settings.signature_type == 2
    assert hasattr(settings, "max_collateral_per_trade")
    assert not hasattr(settings, "max_usdc_per_trade")


def test_market_selector_uses_pinned_or_chosen_5_minute_up_down_market():
    from polytrader.market_selection import select_5m_updown_market

    events = [
        {
            "slug": "btc-updown-15m-1",
            "title": "Bitcoin Up or Down 15 minute",
            "active": True,
            "closed": False,
            "markets": [{"outcomes": '["Up", "Down"]', "clobTokenIds": '["u15", "d15"]', "enableOrderBook": True}],
        },
        {
            "slug": "eth-updown-5m-1",
            "title": "Ethereum Up or Down 5m",
            "active": True,
            "closed": False,
            "markets": [{"outcomes": ["Up", "Down"], "clobTokenIds": ["ue", "de"], "enableOrderBook": True}],
        },
        {
            "slug": "btc-updown-5m-1",
            "title": "Bitcoin Up or Down - 5-minute window",
            "active": True,
            "closed": False,
            "markets": [{"outcomes": '["Up", "Down"]', "clobTokenIds": '["ub", "db"]', "enableOrderBook": True}],
        },
    ]

    selected = select_5m_updown_market(events, crypto_symbol="BTC")

    assert selected.slug == "btc-updown-5m-1"
    assert selected.up_token_id == "ub"
    assert selected.down_token_id == "db"

    pinned = select_5m_updown_market(events, crypto_symbol="BTC", market_slug="eth-updown-5m-1")
    assert pinned.slug == "eth-updown-5m-1"


def test_fee_aware_evaluator_rejects_edge_that_fees_erase_and_rounds_tick():
    from polytrader.models import MarketMetadata, OrderBookQuote
    from polytrader.strategy import evaluate_buy

    market = MarketMetadata(condition_id="c", token_id="t", tick_size=0.01, neg_risk=False, fee_rate_bps=200, fee_exponent=0)
    quote = OrderBookQuote(bid=0.49, ask=0.51, bid_size=100, ask_size=100)

    rejected = evaluate_buy("FORECAST", market, quote, model_probability=0.525, collateral_size=10, min_edge=0.01)
    assert rejected.action == "SKIP"
    assert rejected.edge_after_fees < 0.01
    assert "fee" in rejected.reason

    accepted = evaluate_buy("FORECAST", market, quote, model_probability=0.55, collateral_size=10, min_edge=0.01)
    assert accepted.action == "BUY"
    assert accepted.price == 0.51
    assert accepted.edge_after_fees >= 0.01


def test_risk_manager_blocks_balance_floor_and_open_positions():
    from polytrader.risk import RiskLimits, check_risk

    limits = RiskLimits(max_collateral_per_trade=10, min_collateral_balance=25, max_open_positions=1)

    assert check_risk(20, 10, open_positions=0, limits=limits).allowed is False
    assert "collateral balance floor" in check_risk(20, 10, open_positions=0, limits=limits).reason
    assert check_risk(100, 11, open_positions=0, limits=limits).allowed is False
    assert check_risk(100, 5, open_positions=1, limits=limits).allowed is False
    assert check_risk(100, 5, open_positions=0, limits=limits).allowed is True


def test_dry_run_execution_never_posts_order(monkeypatch):
    install_fake_clob_v2(monkeypatch)
    from polytrader.execution import ClobV2ExecutionClient
    from polytrader.models import MarketMetadata, TradeDecision

    calls = []

    class FakeClient:
        def create_and_post_order(self, *args, **kwargs):
            calls.append((args, kwargs))

    executor = ClobV2ExecutionClient(client=FakeClient(), dry_run=True)
    receipt = executor.place_order(
        TradeDecision(strategy="FORECAST", action="BUY", token_id="tok", side="BUY", price=0.51, collateral_size=5, edge_after_fees=0.02, reason="ok"),
        MarketMetadata(condition_id="cond", token_id="tok", tick_size=0.01, neg_risk=True, fee_rate_bps=0, fee_exponent=1),
    )

    assert receipt.dry_run is True
    assert receipt.status == "dry_run"
    assert calls == []


def test_live_execution_uses_clob_v2_options(monkeypatch):
    install_fake_clob_v2(monkeypatch)
    import polytrader.execution as execution
    importlib.reload(execution)
    from polytrader.models import MarketMetadata, TradeDecision

    calls = []

    class FakeClient:
        def create_and_post_order(self, *args, **kwargs):
            calls.append((args, kwargs))
            return {"orderID": "abc"}

    executor = execution.ClobV2ExecutionClient(client=FakeClient(), dry_run=False, order_type="GTC", post_only=True)
    receipt = executor.place_order(
        TradeDecision(strategy="FORECAST", action="BUY", token_id="tok", side="BUY", price=0.51, collateral_size=5, edge_after_fees=0.02, reason="ok"),
        MarketMetadata(condition_id="cond", token_id="tok", tick_size=0.01, neg_risk=True, fee_rate_bps=0, fee_exponent=1),
    )

    assert receipt.status == "posted"
    assert calls
    _, kwargs = calls[0]
    assert kwargs["options"].kwargs == {"tick_size": "0.01", "neg_risk": True}
    assert kwargs["order_type"] == "GTC"
    assert kwargs["post_only"] is True


def test_build_clob_v2_client_uses_root_sdk_exports_and_safe_funder(monkeypatch):
    fake = install_fake_clob_v2(monkeypatch)
    import polytrader.execution as execution
    importlib.reload(execution)
    from polytrader.config import Settings

    settings = Settings(
        dry_run=False,
        private_key="test-private-key-placeholder",
        safe_address="0x0000000000000000000000000000000000000abc",
        funder_address="0x0000000000000000000000000000000000000abc",
        signature_type=2,
        clob_api_key="api-key-placeholder",
        clob_api_secret="api-secret-placeholder",
        clob_api_passphrase="passphrase-placeholder",
    )

    client = execution.build_clob_v2_client(settings)

    assert client is fake.ClobClient.instances[0]
    assert client.kwargs["host"] == "https://clob.polymarket.com"
    assert client.kwargs["chain_id"] == 137
    assert client.kwargs["signature_type"] == 2
    assert client.kwargs["funder"] == settings.safe_address
    assert client.kwargs["creds"].kwargs == {
        "api_key": "api-key-placeholder",
        "api_secret": "api-secret-placeholder",
        "api_passphrase": "passphrase-placeholder",
    }


def test_live_mode_refuses_missing_private_key():
    from polytrader.config import Settings
    from polytrader.execution import build_clob_v2_client

    settings = Settings(dry_run=False, private_key=None)

    try:
        build_clob_v2_client(settings)
    except ValueError as exc:
        assert "PRIVATE_KEY" in str(exc)
    else:
        raise AssertionError("live mode without PRIVATE_KEY should fail")


def test_market_data_enriches_fee_metadata_from_clob_v2_client():
    from polytrader.market_data import enrich_market_metadata
    from polytrader.models import SelectedMarket

    class FakeClient:
        def get_clob_market_info(self, condition_id):
            assert condition_id == "cond"
            return {"tick_size": "0.001", "neg_risk": True}

        def get_fee_rate_bps(self, token_id):
            assert token_id == "tok"
            return "50"

        def get_fee_exponent(self, token_id):
            assert token_id == "tok"
            return "2"

    selected = SelectedMarket(slug="btc-updown-5m", title="Bitcoin Up or Down 5m", condition_id="cond", up_token_id="tok", down_token_id="down")
    metadata = enrich_market_metadata(FakeClient(), selected, token_id="tok")

    assert metadata.tick_size == 0.001
    assert metadata.neg_risk is True
    assert metadata.fee_rate_bps == 50
    assert metadata.fee_exponent == 2
