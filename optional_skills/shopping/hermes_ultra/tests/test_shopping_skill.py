"""Comprehensive test suite for Hermes Ultra Shopping Skill.

Tests: selector_loader, parsers (YAML-driven), LLM fallback,
scalper reasoning, database, scorer, trend predictor, and alerts.
"""

import json
import os
import sys
import tempfile
import time

import pytest

# ---------------------------------------------------------------------------
# Ensure the skill scripts are importable
# ---------------------------------------------------------------------------

_project_root = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    os.pardir, os.pardir, os.pardir, os.pardir
)
sys.path.insert(0, os.path.abspath(_project_root))


# ===========================================================================
# Test: Selector Loader
# ===========================================================================

class TestSelectorLoader:
    """Tests for the YAML-driven selector loader."""

    def test_get_selectors_returns_list(self):
        from optional_skills.shopping.hermes_ultra.scripts.parsers.selector_loader import get_selectors
        result = get_selectors("amazon", "name")
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(p, str) for p in result)

    def test_get_selectors_missing_site(self):
        from optional_skills.shopping.hermes_ultra.scripts.parsers.selector_loader import get_selectors
        result = get_selectors("nonexistent_site", "price")
        assert result == []

    def test_get_selectors_missing_key(self):
        from optional_skills.shopping.hermes_ultra.scripts.parsers.selector_loader import get_selectors
        result = get_selectors("amazon", "nonexistent_key")
        assert result == []

    def test_get_selector_returns_string(self):
        from optional_skills.shopping.hermes_ultra.scripts.parsers.selector_loader import get_selector
        result = get_selector("amazon", "price_whole_fraction")
        assert result is not None
        assert isinstance(result, str)

    def test_get_selector_missing_returns_none(self):
        from optional_skills.shopping.hermes_ultra.scripts.parsers.selector_loader import get_selector
        result = get_selector("amazon", "nonexistent")
        assert result is None

    def test_get_site_config_returns_dict(self):
        from optional_skills.shopping.hermes_ultra.scripts.parsers.selector_loader import get_site_config
        cfg = get_site_config("amazon")
        assert isinstance(cfg, dict)
        assert "domains" in cfg
        assert "name" in cfg
        assert "price" in cfg

    def test_get_domains(self):
        from optional_skills.shopping.hermes_ultra.scripts.parsers.selector_loader import get_domains
        domains = get_domains("amazon")
        assert "amazon.com" in domains
        assert "amazon.co.uk" in domains

    def test_get_stock_keywords(self):
        from optional_skills.shopping.hermes_ultra.scripts.parsers.selector_loader import get_stock_keywords
        kw = get_stock_keywords("amazon")
        assert "out_of_stock" in kw
        assert "in_stock" in kw
        assert isinstance(kw["out_of_stock"], list)

    def test_invalidate_cache(self):
        from optional_skills.shopping.hermes_ultra.scripts.parsers.selector_loader import invalidate_cache, get_selectors
        # First call loads cache
        get_selectors("amazon", "name")
        # Invalidate
        invalidate_cache()
        # Should reload on next call without error
        result = get_selectors("amazon", "name")
        assert len(result) > 0

    def test_all_sites_have_domains(self):
        from optional_skills.shopping.hermes_ultra.scripts.parsers.selector_loader import get_site_config
        for site in ["amazon", "ebay", "bestbuy", "newegg", "walmart"]:
            cfg = get_site_config(site)
            assert "domains" in cfg, f"{site} missing domains"
            assert len(cfg["domains"]) > 0, f"{site} has empty domains"


# ===========================================================================
# Test: Parsers (YAML-driven)
# ===========================================================================

class TestParsers:
    """Tests for site-specific parsers."""

    def test_parser_registry_loads(self):
        from optional_skills.shopping.hermes_ultra.scripts.parsers import get_parser, list_supported_sites
        sites = list_supported_sites()
        assert len(sites) >= 8  # 7 specific + generic

    def test_get_parser_amazon(self):
        from optional_skills.shopping.hermes_ultra.scripts.parsers import get_parser
        parser = get_parser("https://www.amazon.com/dp/B0ABCDEF")
        assert parser is not None
        assert "Amazon" in parser.get_site_name()

    def test_get_parser_ebay(self):
        from optional_skills.shopping.hermes_ultra.scripts.parsers import get_parser
        parser = get_parser("https://www.ebay.com/itm/12345")
        assert parser is not None
        assert "eBay" in parser.get_site_name()

    def test_get_parser_generic_fallback(self):
        from optional_skills.shopping.hermes_ultra.scripts.parsers import get_parser
        parser = get_parser("https://unknown-store.com/product/123")
        assert parser is not None
        assert "Generic" in parser.get_site_name()

    def test_amazon_parse_basic(self):
        from optional_skills.shopping.hermes_ultra.scripts.parsers.amazon_global import AmazonGlobalParser
        parser = AmazonGlobalParser()
        html = '''
        <span id="productTitle" class="a-size-large product-title-word-break">
            RTX 4090 Graphics Card
        </span>
        <span class="a-price-whole">1,199</span>
        <span class="a-price-fraction">99</span>
        <div id="availability"><span>In Stock</span></div>
        '''
        result = parser.parse(html, "https://www.amazon.com/dp/B0TEST")
        assert result.name == "RTX 4090 Graphics Card"
        assert result.price == 1199.99
        assert result.stock_status == "in_stock"

    def test_ebay_parse_json_ld(self):
        from optional_skills.shopping.hermes_ultra.scripts.parsers.ebay_global import EbayGlobalParser
        parser = EbayGlobalParser()
        html = '''
        <script type="application/ld+json">
        {"@type": "Product", "name": "Test GPU",
         "offers": {"price": "599.99", "availability": "https://schema.org/InStock"}}
        </script>
        '''
        result = parser.parse(html, "https://www.ebay.com/itm/123")
        assert result.name == "Test GPU"
        assert result.price == 599.99
        assert result.stock_status == "in_stock"

    def test_generic_parse_og_meta(self):
        from optional_skills.shopping.hermes_ultra.scripts.parsers.generic import GenericParser
        parser = GenericParser()
        html = '''
        <meta property="og:title" content="Amazing Product" />
        <meta property="product:price:amount" content="49.99" />
        <meta property="product:price:currency" content="USD" />
        <meta property="product:availability" content="instock" />
        '''
        result = parser.parse(html, "https://some-store.com/product")
        assert result.name == "Amazing Product"
        assert result.price == 49.99
        assert result.currency == "USD"
        assert result.stock_status == "in_stock"

    def test_parser_returns_product_data(self):
        from optional_skills.shopping.hermes_ultra.scripts.parsers.base import ProductData
        from optional_skills.shopping.hermes_ultra.scripts.parsers.bestbuy import BestBuyParser
        parser = BestBuyParser()
        result = parser.parse("<html></html>", "https://www.bestbuy.com/site/test")
        assert isinstance(result, ProductData)


# ===========================================================================
# Test: LLM Fallback
# ===========================================================================

class TestLLMFallback:
    """Tests for the LLM-based extraction fallback."""

    def test_strip_html_to_text(self):
        from optional_skills.shopping.hermes_ultra.scripts.llm_fallback import _strip_html_to_text
        html = "<html><body><h1>  Hello  </h1><script>evil()</script><p>World</p></body></html>"
        text = _strip_html_to_text(html)
        assert "Hello" in text
        assert "World" in text
        assert "evil" not in text
        assert "<" not in text

    def test_strip_html_max_length(self):
        from optional_skills.shopping.hermes_ultra.scripts.llm_fallback import _strip_html_to_text, _MAX_TEXT_LENGTH
        html = "<p>" + "x" * 10000 + "</p>"
        text = _strip_html_to_text(html)
        assert len(text) <= _MAX_TEXT_LENGTH

    def test_build_extraction_prompt(self):
        from optional_skills.shopping.hermes_ultra.scripts.llm_fallback import _build_extraction_prompt
        prompt = _build_extraction_prompt("Some page text", "https://example.com")
        assert "product" in prompt.lower()
        assert "JSON" in prompt
        assert "https://example.com" in prompt

    def test_parse_llm_response_valid(self):
        from optional_skills.shopping.hermes_ultra.scripts.llm_fallback import _parse_llm_response
        response = '{"name": "Test Product", "price": 199.99, "currency": "USD", "stock_status": "in_stock", "seller": "TestStore"}'
        result = _parse_llm_response(response, "USD")
        assert result is not None
        assert result.name == "Test Product"
        assert result.price == 199.99
        assert result.currency == "USD"
        assert result.stock_status == "in_stock"

    def test_parse_llm_response_with_surrounding_text(self):
        from optional_skills.shopping.hermes_ultra.scripts.llm_fallback import _parse_llm_response
        response = 'Here is the data: {"name": "GPU", "price": 499, "currency": "USD", "stock_status": "in_stock", "seller": ""} Hope that helps!'
        result = _parse_llm_response(response, "USD")
        assert result is not None
        assert result.name == "GPU"
        assert result.price == 499.0

    def test_parse_llm_response_invalid(self):
        from optional_skills.shopping.hermes_ultra.scripts.llm_fallback import _parse_llm_response
        result = _parse_llm_response("Not valid json at all", "USD")
        assert result is None

    def test_extract_with_llm_no_client(self):
        from optional_skills.shopping.hermes_ultra.scripts.llm_fallback import extract_with_llm
        result = extract_with_llm("<html>Test</html>", "https://example.com", auxiliary_client=None)
        assert result is None

    def test_extract_with_llm_empty_html(self):
        from optional_skills.shopping.hermes_ultra.scripts.llm_fallback import extract_with_llm
        result = extract_with_llm("", "https://example.com")
        assert result is None

    def test_get_manual_extraction_prompt(self):
        from optional_skills.shopping.hermes_ultra.scripts.llm_fallback import get_manual_extraction_prompt
        html = "<html><body><h1>Product</h1><span class='price'>$99</span></body></html>"
        prompt = get_manual_extraction_prompt(html, "https://store.com/item")
        assert "product" in prompt.lower()
        assert "https://store.com/item" in prompt


# ===========================================================================
# Test: Scalper Reasoning
# ===========================================================================

class TestScalperReasoning:
    """Tests for the MSRP reasoning module."""

    def test_basic_markup(self):
        from optional_skills.shopping.hermes_ultra.scripts.reasoning import analyze_price_reasoning
        result = analyze_price_reasoning(
            product_name="RTX 4090",
            current_price=1800.0,
            original_price=1599.0,
        )
        assert result.msrp == 1599.0
        assert result.markup_pct > 0
        assert "RTX 4090" in result.reasoning
        assert result.recommendation in ("BUY", "WAIT", "AVOID")

    def test_discount_detected(self):
        from optional_skills.shopping.hermes_ultra.scripts.reasoning import analyze_price_reasoning
        result = analyze_price_reasoning(
            product_name="Headphones",
            current_price=150.0,
            original_price=250.0,
        )
        assert result.markup_pct < 0
        assert result.recommendation == "BUY"
        assert "discount" in result.reasoning.lower() or "below" in result.reasoning.lower()

    def test_no_msrp_available(self):
        from optional_skills.shopping.hermes_ultra.scripts.reasoning import analyze_price_reasoning
        result = analyze_price_reasoning(
            product_name="Unknown Widget",
            current_price=50.0,
        )
        assert result.msrp is None
        assert result.recommendation == "WAIT"

    def test_market_prices_used_for_msrp(self):
        from optional_skills.shopping.hermes_ultra.scripts.reasoning import analyze_price_reasoning
        result = analyze_price_reasoning(
            product_name="Keyboard",
            current_price=120.0,
            market_prices=[
                {"site": "Amazon", "price": 100.0},
                {"site": "eBay", "price": 110.0},
                {"site": "Walmart", "price": 95.0},
            ],
        )
        assert result.msrp == 95.0  # min of market prices
        assert result.markup_pct > 0

    def test_avoid_recommendation_for_extreme_markup(self):
        from optional_skills.shopping.hermes_ultra.scripts.reasoning import analyze_price_reasoning
        result = analyze_price_reasoning(
            product_name="GPU",
            current_price=3000.0,
            original_price=1500.0,
        )
        assert result.recommendation == "AVOID"
        assert result.confidence == "high"

    def test_scalper_reasoning_has_factors(self):
        from optional_skills.shopping.hermes_ultra.scripts.reasoning import analyze_price_reasoning
        result = analyze_price_reasoning(
            product_name="PS5",
            current_price=800.0,
            original_price=499.0,
            seller="RandomSeller123",
        )
        assert len(result.factors) > 0

    def test_zero_price_handled(self):
        from optional_skills.shopping.hermes_ultra.scripts.reasoning import analyze_price_reasoning
        result = analyze_price_reasoning(
            product_name="Free Item",
            current_price=0.0,
        )
        assert result.recommendation == "UNKNOWN"

    def test_market_context_populated(self):
        from optional_skills.shopping.hermes_ultra.scripts.reasoning import analyze_price_reasoning
        result = analyze_price_reasoning(
            product_name="Mouse",
            current_price=50.0,
            market_prices=[
                {"site": "Amazon", "price": 45.0},
                {"site": "eBay", "price": 55.0},
            ],
        )
        assert result.market_context != ""


# ===========================================================================
# Test: Database
# ===========================================================================

class TestDatabase:
    """Tests for the SQLite price tracker database."""

    @pytest.fixture
    def db(self, tmp_path):
        from optional_skills.shopping.hermes_ultra.scripts.database import PriceTrackerDB
        db_path = str(tmp_path / "test.db")
        return PriceTrackerDB(db_path=db_path)

    def test_add_and_get_product(self, db):
        from optional_skills.shopping.hermes_ultra.scripts.database import Product
        product = Product(url="https://example.com/p1", name="Test Product")
        added = db.add_product(product)
        assert added.id is not None

        fetched = db.get_product(added.id)
        assert fetched is not None
        assert fetched.name == "Test Product"
        assert fetched.url == "https://example.com/p1"

    def test_get_product_by_url(self, db):
        from optional_skills.shopping.hermes_ultra.scripts.database import Product
        product = Product(url="https://example.com/unique", name="Unique")
        db.add_product(product)

        found = db.get_product_by_url("https://example.com/unique")
        assert found is not None
        assert found.name == "Unique"

    def test_list_products(self, db):
        from optional_skills.shopping.hermes_ultra.scripts.database import Product
        db.add_product(Product(url="https://a.com", name="A"))
        db.add_product(Product(url="https://b.com", name="B"))

        products = db.list_products()
        assert len(products) == 2

    def test_update_price_creates_history(self, db):
        from optional_skills.shopping.hermes_ultra.scripts.database import Product
        product = db.add_product(Product(url="https://x.com", name="X"))
        db.update_product_price(product.id, 100.0, stock_status="in_stock")
        db.update_product_price(product.id, 95.0, stock_status="in_stock")

        history = db.get_price_history(product.id)
        assert len(history) == 2
        assert history[0].price == 95.0  # newest first

    def test_delete_product(self, db):
        from optional_skills.shopping.hermes_ultra.scripts.database import Product
        product = db.add_product(Product(url="https://del.com", name="Delete Me"))
        assert db.delete_product(product.id)
        assert db.get_product(product.id) is None

    def test_settings(self, db):
        db.set_setting("test_key", "test_value")
        assert db.get_setting("test_key") == "test_value"
        assert db.get_setting("missing_key", "default") == "default"

    def test_lifetime_savings(self, db):
        assert db.get_lifetime_savings() == 0.0
        db.add_lifetime_savings(50.0)
        assert db.get_lifetime_savings() == 50.0
        db.add_lifetime_savings(25.0)
        assert db.get_lifetime_savings() == 75.0

    def test_alerts_crud(self, db):
        from optional_skills.shopping.hermes_ultra.scripts.database import Product, Alert
        product = db.add_product(Product(url="https://alert.com", name="Alert Test"))
        alert = Alert(product_id=product.id, alert_type="price_drop", threshold=100.0)
        added = db.add_alert(alert)
        assert added.id is not None

        active = db.get_active_alerts(product.id)
        assert len(active) == 1

        db.deactivate_alert(added.id)
        active = db.get_active_alerts(product.id)
        assert len(active) == 0


# ===========================================================================
# Test: Scoring
# ===========================================================================

class TestScoring:
    """Tests for the deal scoring engine."""

    def test_basic_score(self):
        from optional_skills.shopping.hermes_ultra.scripts.scoring import DealScorer
        scorer = DealScorer()
        result = scorer.calculate(current_price=500.0, target_price=600.0, stock_status="in_stock")
        assert 0 <= result.total_score <= 100
        assert result.label != ""

    def test_below_target_high_score(self):
        from optional_skills.shopping.hermes_ultra.scripts.scoring import DealScorer
        scorer = DealScorer()
        result = scorer.calculate(current_price=400.0, target_price=600.0, stock_status="in_stock")
        assert result.discount_score >= 80

    def test_above_target_low_score(self):
        from optional_skills.shopping.hermes_ultra.scripts.scoring import DealScorer
        scorer = DealScorer()
        result = scorer.calculate(current_price=1200.0, target_price=600.0, stock_status="in_stock")
        assert result.discount_score < 50

    def test_no_target_neutral(self):
        from optional_skills.shopping.hermes_ultra.scripts.scoring import DealScorer
        scorer = DealScorer()
        result = scorer.calculate(current_price=500.0)
        assert result.discount_score == 50

    def test_none_price(self):
        from optional_skills.shopping.hermes_ultra.scripts.scoring import DealScorer
        scorer = DealScorer()
        result = scorer.calculate(current_price=None)
        assert result.total_score == 0


# ===========================================================================
# Test: Scalper Detector
# ===========================================================================

class TestScalperDetector:
    """Tests for scalper detection engine."""

    def test_normal_price(self):
        from optional_skills.shopping.hermes_ultra.scripts.scalper_detector import ScalperDetector
        detector = ScalperDetector()
        result = detector.check(current_price=100.0, price_history=[98, 99, 100, 101, 99])
        assert result.risk_level == "NONE"
        assert not result.is_suspicious

    def test_high_risk_scalper(self):
        from optional_skills.shopping.hermes_ultra.scripts.scalper_detector import ScalperDetector
        detector = ScalperDetector()
        result = detector.check(current_price=200.0, price_history=[100, 105, 98, 102, 100])
        assert result.risk_level == "HIGH"
        assert result.is_suspicious

    def test_cross_site_suspicious(self):
        from optional_skills.shopping.hermes_ultra.scripts.scalper_detector import ScalperDetector
        detector = ScalperDetector()
        market_prices = [
            {"site": "Amazon", "price": 100},
            {"site": "eBay", "price": 180},
        ]
        result = detector.check_cross_site(market_prices)
        assert result is not None
        assert result.is_suspicious

    def test_cross_site_normal(self):
        from optional_skills.shopping.hermes_ultra.scripts.scalper_detector import ScalperDetector
        detector = ScalperDetector()
        market_prices = [
            {"site": "Amazon", "price": 100},
            {"site": "eBay", "price": 105},
        ]
        result = detector.check_cross_site(market_prices)
        assert result is None  # No alert for small spread


# ===========================================================================
# Test: Trend Predictor
# ===========================================================================

class TestTrendPredictor:
    """Tests for trend prediction module."""

    def test_insufficient_data(self):
        from optional_skills.shopping.hermes_ultra.scripts.trend_predictor import TrendPredictor
        predictor = TrendPredictor()
        result = predictor.predict([100.0])
        assert result.direction == "STABLE"
        assert "Need at least 3" in result.analysis_text

    def test_downward_trend(self):
        from optional_skills.shopping.hermes_ultra.scripts.trend_predictor import TrendPredictor
        predictor = TrendPredictor()
        prices = [80, 85, 90, 95, 100]  # newest first = decreasing
        result = predictor.predict(prices)
        assert result.direction == "DOWN"

    def test_upward_trend(self):
        from optional_skills.shopping.hermes_ultra.scripts.trend_predictor import TrendPredictor
        predictor = TrendPredictor()
        prices = [100, 95, 90, 85, 80]  # newest first = increasing
        result = predictor.predict(prices)
        assert result.direction == "UP"

    def test_stable_trend(self):
        from optional_skills.shopping.hermes_ultra.scripts.trend_predictor import TrendPredictor
        predictor = TrendPredictor()
        prices = [100, 99, 100, 101, 100]  # flat
        result = predictor.predict(prices)
        assert result.direction == "STABLE"


# ===========================================================================
# Test: Price Utilities
# ===========================================================================

class TestPriceUtils:
    """Tests for price parsing and currency detection."""

    def test_parse_us_price(self):
        from optional_skills.shopping.hermes_ultra.scripts.parsers.price_utils import parse_price
        assert parse_price("$1,299.99", "USD") == 1299.99

    def test_parse_euro_price(self):
        from optional_skills.shopping.hermes_ultra.scripts.parsers.price_utils import parse_price
        assert parse_price("1.299,00 €", "EUR") == 1299.0

    def test_parse_plain_number(self):
        from optional_skills.shopping.hermes_ultra.scripts.parsers.price_utils import parse_price
        assert parse_price("499", "USD") == 499.0

    def test_parse_empty(self):
        from optional_skills.shopping.hermes_ultra.scripts.parsers.price_utils import parse_price
        assert parse_price("", "USD") is None

    def test_detect_currency_from_url_amazon(self):
        from optional_skills.shopping.hermes_ultra.scripts.parsers.price_utils import detect_currency
        assert detect_currency("https://www.amazon.de/dp/B0TEST") == "EUR"

    def test_detect_currency_from_url_uk(self):
        from optional_skills.shopping.hermes_ultra.scripts.parsers.price_utils import detect_currency
        assert detect_currency("https://www.amazon.co.uk/dp/B0TEST") == "GBP"

    def test_extract_text_first_match(self):
        from optional_skills.shopping.hermes_ultra.scripts.parsers.price_utils import extract_text
        html = '<span id="price">$499.00</span>'
        result = extract_text(html, ['id="price"[^>]*>(.*?)</span>'])
        assert result == "$499.00"

    def test_extract_json_ld(self):
        from optional_skills.shopping.hermes_ultra.scripts.parsers.price_utils import extract_json_ld
        html = '''
        <script type="application/ld+json">
        {"@type": "Product", "name": "Test"}
        </script>
        '''
        blocks = extract_json_ld(html)
        assert len(blocks) == 1
        assert blocks[0]["name"] == "Test"


# ===========================================================================
# Test: Alerts Formatting
# ===========================================================================

class TestAlerts:
    """Tests for alert formatting functions."""

    def test_format_deal_alert(self):
        from optional_skills.shopping.hermes_ultra.scripts.alerts import format_deal_alert
        output = format_deal_alert("Test Product", 499.0, 600.0, 85, "🔥 UNMISSABLE DEAL")
        assert "Test Product" in output
        assert "499" in output

    def test_format_scalper_warning(self):
        from optional_skills.shopping.hermes_ultra.scripts.alerts import format_scalper_warning
        output = format_scalper_warning("GPU", "HIGH", "🚨", 35.0, 1000.0, 1350.0, "Scalper detected")
        assert "GPU" in output

    def test_format_price_table(self):
        from optional_skills.shopping.hermes_ultra.scripts.alerts import format_price_table
        products = [
            {"name": "Product A", "current_price": 100, "target_price": 80, "stock_status": "in_stock", "site": "Amazon"},
            {"name": "Product B", "current_price": 200, "stock_status": "out_of_stock", "site": "eBay"},
        ]
        output = format_price_table(products)
        assert "Product A" in output
        assert "Product B" in output

    def test_format_market_overview(self):
        from optional_skills.shopping.hermes_ultra.scripts.alerts import format_market_overview
        results = [
            {"site": "Amazon", "price": 100, "stock_status": "in_stock"},
            {"site": "eBay", "price": 120, "stock_status": "in_stock"},
        ]
        output = format_market_overview(results, "Test Product")
        assert "Amazon" in output

    def test_format_full_report(self):
        from optional_skills.shopping.hermes_ultra.scripts.alerts import format_full_report
        output = format_full_report(
            product_name="RTX 4090",
            current_price=1499.0,
            target_price=1200.0,
            deal_score=45,
            deal_label="🤔 FAIR DEAL",
            scalper_risk="NONE",
            scalper_text="Normal price range",
            trend_direction="STABLE",
            trend_text="No movement detected",
            reasoning_text="Close to MSRP",
        )
        assert "RTX 4090" in output
        assert "FAIR DEAL" in output or "WAIT" in output

    def test_format_cross_site_scalper_alert(self):
        from optional_skills.shopping.hermes_ultra.scripts.alerts import format_cross_site_scalper_alert
        output = format_cross_site_scalper_alert(
            product_name="GPU",
            min_price=500.0,
            min_site="Amazon",
            max_price=800.0,
            max_site="eBay",
            spread_pct=60.0,
        )
        assert "GPU" in output
        assert "Amazon" in output
        assert "eBay" in output


# ===========================================================================
# Test: WindowsNotificationHook OS Check
# ===========================================================================

class TestNotificationHooks:
    """Tests for notification hook OS awareness."""

    def test_windows_hook_has_os_check(self):
        from optional_skills.shopping.hermes_ultra.scripts.alerts import WindowsNotificationHook
        hook = WindowsNotificationHook()
        assert hasattr(hook, "_is_windows")

    def test_terminal_hook_sends(self):
        from optional_skills.shopping.hermes_ultra.scripts.alerts import TerminalHook
        hook = TerminalHook()
        result = hook.send("Test", "Test message", "info")
        assert result is True
