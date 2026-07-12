"""Unit tests for warashibe multi-market price research + arbitrage helpers."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugins" / "warashibe-reselling"


def load_price_research():
    package_name = "warashibe_reselling_test_plugin"
    for name in list(sys.modules):
        if name == package_name or name.startswith(f"{package_name}."):
            del sys.modules[name]

    pkg_spec = importlib.util.spec_from_file_location(
        package_name,
        PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    assert pkg_spec and pkg_spec.loader
    pkg = importlib.util.module_from_spec(pkg_spec)
    sys.modules[package_name] = pkg

    # Provide package path for relative imports inside price_research.
    import os
    os.environ["HERMES_HOME"] = str(Path.cwd() / ".hermes-test-home")

    core_spec = importlib.util.spec_from_file_location(
        f"{package_name}.core",
        PLUGIN_DIR / "core.py",
    )
    assert core_spec and core_spec.loader
    core_mod = importlib.util.module_from_spec(core_spec)
    sys.modules[f"{package_name}.core"] = core_mod
    core_spec.loader.exec_module(core_mod)

    mod_spec = importlib.util.spec_from_file_location(
        f"{package_name}.price_research",
        PLUGIN_DIR / "price_research.py",
    )
    assert mod_spec and mod_spec.loader
    module = importlib.util.module_from_spec(mod_spec)
    sys.modules[f"{package_name}.price_research"] = module
    mod_spec.loader.exec_module(module)
    return module


class _FakeLocator:
    def __init__(self, nodes):
        self._nodes = list(nodes)

    def count(self):
        return len(self._nodes)

    def nth(self, index):
        return self._nodes[index]

    @property
    def first(self):
        return self._nodes[0] if self._nodes else _FakeNode(None)


class _FakeNode:
    def __init__(self, payload: dict | None):
        self._payload = payload or {}

    def get_attribute(self, name: str):
        return self._payload.get("attrs", {}).get(name)

    def inner_text(self):
        return self._payload.get("text", "")

    def locator(self, selector: str):
        children = self._payload.get("children", {}).get(selector, [])
        if not children and selector.startswith("a"):
            children = [self._payload] if self._payload.get("attrs", {}).get("href") else []
        return _FakeLocator([_FakeNode(c) if not isinstance(c, _FakeNode) else c for c in children])

    def count(self):
        return 1 if self._payload else 0


class _FakePage:
    def __init__(self, mapping: dict[str, list[dict]], evaluate_payload=None):
        self._mapping = mapping
        self._evaluate_payload = evaluate_payload or []

    def locator(self, selector: str):
        return _FakeLocator([_FakeNode(n) for n in self._mapping.get(selector, [])])

    def evaluate(self, _script: str):
        return self._evaluate_payload


def test_mercari_url_uses_keyword_param():
    pr = load_price_research()
    result = pr.search_prices("RTX 3060", platform="mercari", dry_run=True)
    assert "keyword=RTX+3060" in result["url"] or "keyword=RTX%203060" in result["url"]
    assert "status=on_sale" in result["url"]


def test_ebay_and_amazon_in_targets():
    pr = load_price_research()
    assert "ebay" in pr.TARGETS
    assert "amazon_jp" in pr.TARGETS
    assert pr.TARGETS["amazon_jp"]["parser"] == "amazon_official_only"


def test_clean_price_yen_and_usd(monkeypatch):
    pr = load_price_research()
    monkeypatch.setenv("WARASHIBE_USDJPY", "150")
    assert pr._clean_price("33,951 円") == 33951
    assert pr._clean_price("$20.00") == 3000


def test_parse_mercari_item_links():
    pr = load_price_research()
    page = _FakePage(
        {
            "a[href*='/item/']": [
                {"attrs": {"href": "/item/m123"}, "text": "¥ 29,999 ASUS RTX 3060 本体"},
                {"attrs": {"href": "/item/m456"}, "text": "¥ 33,000 GIGABYTE RTX 3060 Ti"},
            ]
        }
    )
    items = pr._parse_mercari_item_links(page, "https://jp.mercari.com/search?keyword=x", 10)
    assert len(items) == 2
    assert items[0]["price"] == 29999


def test_parse_ebay_itm_cards():
    pr = load_price_research()
    page = _FakePage(
        {},
        evaluate_payload=[
            {
                "id": "158023401766",
                "href": "https://www.ebay.com/itm/158023401766",
                "text": "ZOTAC RTX 3060 Ti 中古品 33,951 円 今すぐ買う ＋送料8,134 円",
            },
            {
                "id": "1",
                "href": "https://www.ebay.com/itm/1",
                "text": "Shop on eBay Brand New $20.00",
            },
        ],
    )
    items = pr._parse_ebay_itm_cards(page, "https://www.ebay.com/sch/i.html?_nkw=x", 10)
    assert len(items) == 1
    assert items[0]["price"] == 33951
    assert items[0]["shipping"] == 8134
    assert items[0]["landed_price"] == 33951 + 8134


def test_amazon_official_skips_without_keys(monkeypatch):
    pr = load_price_research()
    monkeypatch.delenv("AMAZON_PAAPI_ACCESS_KEY", raising=False)
    monkeypatch.delenv("PAAPI_ACCESS_KEY", raising=False)
    monkeypatch.delenv("AMAZON_PAAPI_SECRET_KEY", raising=False)
    monkeypatch.delenv("PAAPI_SECRET_KEY", raising=False)
    monkeypatch.delenv("AMAZON_PAAPI_PARTNER_TAG", raising=False)
    monkeypatch.delenv("PAAPI_PARTNER_TAG", raising=False)
    result = pr.search_prices("RTX 3060", platform="amazon_jp", dry_run=False)
    assert result["skipped"] is True
    assert result["items"] == []
    assert "no_html_scrape" in result.get("policy", "") or "scrape" in result.get("skip_reason", "").lower() or "disabled" in result.get("skip_reason", "").lower()


def test_find_arbitrage_go_path(monkeypatch):
    pr = load_price_research()

    def fake_search_markets(keyword, platforms=None, limit=10, dry_run=False):
        return {
            "keyword": keyword,
            "results": [
                {
                    "platform": "mercari",
                    "items": [
                        {
                            "title": "cheap gpu",
                            "price": 20000,
                            "landed_price": 20000,
                            "url": "https://jp.mercari.com/item/m1",
                        }
                    ],
                },
                {
                    "platform": "ebay",
                    "items": [
                        {
                            "title": "expensive gpu",
                            "price": 50000,
                            "landed_price": 50000,
                            "url": "https://www.ebay.com/itm/2",
                        }
                    ],
                },
                {
                    "platform": "yahoo_auction",
                    "items": [],
                },
                {
                    "platform": "amazon_jp",
                    "items": [],
                    "skipped": True,
                },
            ],
        }

    monkeypatch.setattr(pr, "search_markets", fake_search_markets)
    result = pr.find_arbitrage(
        "RTX 3060",
        platforms=["mercari", "ebay"],
        limit=5,
        dry_run=False,
        budget_yen=80000,
    )
    assert result["winner_count"] >= 1
    top = result["winners"][0]
    assert top["buy_platform"] == "mercari"
    assert top["sell_platform"] == "ebay"
    assert top["go"] is True
    assert top["profit"] >= 500


def test_blocked_host_raises():
    pr = load_price_research()
    with pytest.raises(ValueError, match="Blocked"):
        pr._safe_target(pr.TARGETS["mercari"], "https://evil.example/search?keyword=x")
