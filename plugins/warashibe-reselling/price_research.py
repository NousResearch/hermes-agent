"""CloakBrowser-backed public market price research for Warashibe.

Amazon JP is never scraped. Only official SP-API / PA-API when configured.
"""
from __future__ import annotations

import os
import re
import time
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote_plus, urlparse

TARGETS: dict[str, dict[str, Any]] = {
    "mercari": {
        "name": "メルカリ",
        "url": "https://jp.mercari.com/search?keyword={keyword}&status=on_sale",
        "allowed_hosts": {"jp.mercari.com"},
        "parser": "mercari_item_links",
        "backend": "cloakbrowser",
        "currency": "JPY",
    },
    "yahoo_auction": {
        "name": "ヤフオク!",
        "url": "https://auctions.yahoo.co.jp/search/search?p={keyword}",
        "allowed_hosts": {"auctions.yahoo.co.jp"},
        "parser": "yahoo_product",
        "backend": "cloakbrowser",
        "currency": "JPY",
    },
    "ebay": {
        "name": "eBay",
        "url": "https://www.ebay.com/sch/i.html?_nkw={keyword}&LH_BIN=1",
        "allowed_hosts": {"www.ebay.com", "ebay.com"},
        "parser": "ebay_itm_cards",
        "backend": "cloakbrowser",
        "currency": "JPY",  # JP locale often shows 円
    },
    "amazon_jp": {
        "name": "Amazon JP",
        "url": "https://www.amazon.co.jp/s?k={keyword}",
        "allowed_hosts": {"www.amazon.co.jp", "amazon.co.jp"},
        "parser": "amazon_official_only",
        "backend": "amazon_paapi",
        "currency": "JPY",
    },
    "bookoff_online": {
        "name": "ブックオフオンライン",
        "url": "https://www.bookoffonline.co.jp/search?q={keyword}",
        "allowed_hosts": {"www.bookoffonline.co.jp"},
        "parser": "css",
        "backend": "cloakbrowser",
        "currency": "JPY",
        "item": ".item-box",
        "title": ".item-title",
        "price": ".price",
        "fallback_item": ".item-box",
    },
}

_PRICE_RE = re.compile(r"(?:¥|￥|\$|USD)?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*円?")
_YEN_RE = re.compile(r"([0-9][0-9,]*)\s*円")
_USD_RE = re.compile(r"\$\s*([0-9][0-9,]*(?:\.[0-9]+)?)")
_ITEM_HREF_RE = re.compile(r"^/item/[a-zA-Z0-9]+$")
_SHIP_RE = re.compile(r"(?:＋|\+)?\s*送料\s*([0-9][0-9,]*)\s*円")


def _usd_jpy_rate() -> float:
    try:
        return float(os.environ.get("WARASHIBE_USDJPY", "150"))
    except ValueError:
        return 150.0


def _clean_price(value: str) -> int | None:
    """Parse a price string into integer yen when possible."""
    text = value or ""
    yen = _YEN_RE.search(text)
    if yen:
        return int(yen.group(1).replace(",", ""))
    usd = _USD_RE.search(text)
    if usd:
        return int(float(usd.group(1).replace(",", "")) * _usd_jpy_rate())
    match = _PRICE_RE.search(text)
    return int(float(match.group(1).replace(",", ""))) if match else None


def _clean_shipping(value: str) -> int:
    m = _SHIP_RE.search(value or "")
    return int(m.group(1).replace(",", "")) if m else 0


def _safe_target(target: dict[str, Any], url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in target["allowed_hosts"]:
        raise ValueError(f"Blocked research URL: {url}")


def _text(locator: Any) -> str:
    try:
        return " ".join((locator.inner_text() or "").split())
    except Exception:
        return ""


def _abs_url(base_url: str, href: str | None) -> str:
    if not href:
        return base_url
    if href.startswith("/"):
        return f"https://{urlparse(base_url).hostname}{href}"
    return href.split("?")[0] if href.startswith("http") else href


def _parse_mercari_item_links(page: Any, base_url: str, limit: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    rows = page.locator("a[href*='/item/']")
    count = rows.count()
    for index in range(count):
        if len(items) >= limit:
            break
        row = rows.nth(index)
        href = row.get_attribute("href") or ""
        path = urlparse(href).path if href.startswith("http") else href
        if not _ITEM_HREF_RE.match(path.split("?")[0]) and "/item/" not in href:
            continue
        item_url = _abs_url(base_url, href)
        if item_url in seen:
            continue
        seen.add(item_url)
        raw = _text(row)
        if not raw:
            try:
                raw = " ".join((row.locator("xpath=..").inner_text() or "").split())
            except Exception:
                raw = ""
        price = _clean_price(raw)
        title = re.sub(r"^(?:¥|￥)\s*[0-9][0-9,]*\s*", "", raw).strip()
        title = re.sub(r"^[0-9][0-9,]*\s*円\s*", "", title).strip()
        if not title and price is None:
            continue
        items.append(
            {
                "title": title or "(no title)",
                "price": price,
                "shipping": 0,
                "landed_price": price,
                "price_text": f"¥{price:,}" if price is not None else raw,
                "url": item_url,
                "platform": "mercari",
                "currency": "JPY",
            }
        )
    return items


def _parse_yahoo_product(page: Any, base_url: str, limit: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    rows = page.locator("li.Product")
    if rows.count() == 0:
        rows = page.locator(".Product")
    count = min(rows.count(), limit)
    for index in range(count):
        row = rows.nth(index)
        link = row.locator("a[data-auction-title], a.Product__imageLink, a.Product__titleLink, a").first
        href = link.get_attribute("href") if link.count() else ""
        title = (
            link.get_attribute("data-auction-title") if link.count() else None
        ) or _text(row.locator(".Product__titleLink, .Product__title").first)
        if not title:
            title = _text(row)
        price_text = _text(row.locator(".Product__priceValue").first) or _text(row.locator(".Product__price").first)
        if not price_text:
            price_text = _text(row)
        price = _clean_price(price_text)
        if not title and price is None:
            continue
        if title and len(title) > 180:
            title = title[:180].rstrip() + "…"
        items.append(
            {
                "title": (title or "(no title)").strip(),
                "price": price,
                "shipping": 0,
                "landed_price": price,
                "price_text": price_text,
                "url": _abs_url(base_url, href) if href else base_url,
                "platform": "yahoo_auction",
                "currency": "JPY",
            }
        )
    return items


def _parse_ebay_itm_cards(page: Any, base_url: str, limit: int) -> list[dict[str, Any]]:
    """eBay 2026 cards: climb parents of /itm/ links and parse 円 / $ prices."""
    try:
        raw_items = page.evaluate(
            """() => {
              const out = [];
              const seen = new Set();
              for (const a of document.querySelectorAll('a[href*="/itm/"]')) {
                const m = (a.getAttribute('href') || '').match(/\\/itm\\/(\\d{9,})/);
                if (!m || seen.has(m[1])) continue;
                let el = a;
                let text = '';
                for (let i = 0; i < 8; i++) {
                  el = el.parentElement;
                  if (!el) break;
                  const t = (el.innerText || '').replace(/\\s+/g, ' ').trim();
                  if (t.length > 30 && t.length < 600) { text = t; break; }
                }
                if (!text || !/\\d/.test(text)) continue;
                if (/Shop on eBay/i.test(text) && text.length < 120) continue;
                seen.add(m[1]);
                out.push({
                  id: m[1],
                  href: (a.href || '').split('?')[0],
                  text: text.slice(0, 280),
                });
                if (out.length >= 30) break;
              }
              return out;
            }"""
        )
    except Exception:
        raw_items = []

    items: list[dict[str, Any]] = []
    for row in raw_items or []:
        text = row.get("text") or ""
        if re.search(r"Shop on eBay", text, re.I):
            continue
        price = _clean_price(text)
        if price is None or price < 500:
            continue
        ship = _clean_shipping(text)
        title = text
        for noise in ("新しいウィンドウまたはタブに表示されます", "今すぐ買う", "またはベストオファー"):
            title = title.replace(noise, " ")
        title = re.sub(r"[0-9][0-9,]*\s*円", " ", title)
        title = re.sub(r"\$\s*[0-9][0-9,]*(?:\.[0-9]+)?", " ", title)
        title = re.sub(r"＋?\s*送料\s*[0-9][0-9,]*\s*円", " ", title)
        title = " ".join(title.split())[:160]
        if len(title) < 8:
            continue
        landed = price + ship
        items.append(
            {
                "title": title or f"eBay item {row.get('id')}",
                "price": price,
                "shipping": ship,
                "landed_price": landed,
                "price_text": f"¥{price:,}" + (f" +送料¥{ship:,}" if ship else ""),
                "url": row.get("href") or base_url,
                "platform": "ebay",
                "currency": "JPY",
            }
        )
        if len(items) >= limit:
            break
    return items


def _parse_css(page: Any, target: dict[str, Any], base_url: str, limit: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    rows = page.locator(target["item"])
    if rows.count() == 0 and target.get("fallback_item"):
        rows = page.locator(target["fallback_item"])
    count = min(rows.count(), limit)
    for index in range(count):
        row = rows.nth(index)
        title = _text(row.locator(target["title"]).first)
        price_text = _text(row.locator(target["price"]).first)
        if not title and not price_text:
            continue
        link = row.locator("a").first
        href = link.get_attribute("href") if link.count() else ""
        price = _clean_price(price_text)
        items.append(
            {
                "title": title,
                "price": price,
                "shipping": 0,
                "landed_price": price,
                "price_text": price_text,
                "url": _abs_url(base_url, href),
                "platform": "bookoff_online",
                "currency": "JPY",
            }
        )
    return items


def _search_amazon_official(keyword: str, limit: int) -> dict[str, Any]:
    """Official-only Amazon path. Never scrapes amazon.co.jp HTML."""
    result: dict[str, Any] = {
        "keyword": keyword,
        "platform": "amazon_jp",
        "platform_name": "Amazon JP",
        "url": f"https://www.amazon.co.jp/s?k={quote_plus(keyword)}",
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "items": [],
        "count": 0,
        "backend": "amazon_paapi",
        "policy": "no_html_scrape",
    }
    access = os.environ.get("AMAZON_PAAPI_ACCESS_KEY") or os.environ.get("PAAPI_ACCESS_KEY")
    secret = os.environ.get("AMAZON_PAAPI_SECRET_KEY") or os.environ.get("PAAPI_SECRET_KEY")
    partner = os.environ.get("AMAZON_PAAPI_PARTNER_TAG") or os.environ.get("PAAPI_PARTNER_TAG")
    if not (access and secret and partner):
        result["skipped"] = True
        result["skip_reason"] = (
            "Amazon HTML scrape is disabled by policy. "
            "Set AMAZON_PAAPI_ACCESS_KEY / AMAZON_PAAPI_SECRET_KEY / AMAZON_PAAPI_PARTNER_TAG "
            "for Product Advertising API, or use SP-API separately."
        )
        return result

    # Optional dependency; fail soft if package missing.
    try:
        from paapi5_python_sdk.api.default_api import DefaultApi  # type: ignore
        from paapi5_python_sdk.models.search_items_request import SearchItemsRequest  # type: ignore
        from paapi5_python_sdk.models.partner_type import PartnerType  # type: ignore
        from paapi5_python_sdk.models.search_items_resource import SearchItemsResource  # type: ignore
        from paapi5_python_sdk.rest import ApiException  # type: ignore
    except Exception as exc:  # noqa: BLE001
        result["skipped"] = True
        result["skip_reason"] = f"PA-API SDK unavailable: {exc}"
        return result

    host = os.environ.get("AMAZON_PAAPI_HOST", "webservices.amazon.co.jp")
    region = os.environ.get("AMAZON_PAAPI_REGION", "us-west-2")
    try:
        api = DefaultApi(access_key=access, secret_key=secret, host=host, region=region)
        request = SearchItemsRequest(
            partner_tag=partner,
            partner_type=PartnerType.ASSOCIATES,
            keywords=keyword,
            search_index="All",
            item_count=min(limit, 10),
            resources=[
                SearchItemsResource.ITEMINFO_TITLE,
                SearchItemsResource.OFFERS_LISTINGS_PRICE,
                SearchItemsResource.DETAILPAGEURL,
            ],
        )
        response = api.search_items(request)
        search = getattr(response, "search_result", None)
        items_out: list[dict[str, Any]] = []
        for item in (getattr(search, "items", None) or [])[:limit]:
            title = None
            try:
                title = item.item_info.title.display_value
            except Exception:
                title = getattr(item, "asin", "amazon item")
            price = None
            try:
                amount = item.offers.listings[0].price.amount
                price = int(amount)
            except Exception:
                price = None
            url = getattr(item, "detail_page_url", result["url"])
            if price is None:
                continue
            items_out.append(
                {
                    "title": title or "amazon item",
                    "price": price,
                    "shipping": 0,
                    "landed_price": price,
                    "price_text": f"¥{price:,}",
                    "url": url,
                    "platform": "amazon_jp",
                    "currency": "JPY",
                }
            )
        result["items"] = items_out
        result["count"] = len(items_out)
        result["skipped"] = False
        return result
    except Exception as exc:  # noqa: BLE001
        result["skipped"] = True
        result["skip_reason"] = f"PA-API request failed: {exc}"
        return result


def search_prices(
    keyword: str,
    platform: str = "mercari",
    limit: int = 10,
    *,
    delay_seconds: float = 3.0,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Search one public marketplace. Amazon is official-API only."""
    keyword = str(keyword or "").strip()
    if not keyword:
        raise ValueError("keyword must not be empty")
    if platform not in TARGETS:
        raise ValueError(f"Unsupported platform: {platform}. Choose: {', '.join(TARGETS)}")
    limit = max(1, min(int(limit), 50))
    target = TARGETS[platform]
    url = target["url"].format(keyword=quote_plus(keyword))
    _safe_target(target, url)
    result: dict[str, Any] = {
        "keyword": keyword,
        "platform": platform,
        "platform_name": target["name"],
        "url": url,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "items": [],
        "dry_run": dry_run,
        "backend": target.get("backend", "cloakbrowser"),
    }
    if dry_run:
        if target.get("parser") == "amazon_official_only":
            result["policy"] = "no_html_scrape"
        return result

    if target.get("parser") == "amazon_official_only":
        amz = _search_amazon_official(keyword, limit)
        amz["dry_run"] = False
        return amz

    from cloakbrowser import launch

    browser = launch(headless=True, humanize=True)
    try:
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=60_000)
        if delay_seconds > 0:
            time.sleep(min(float(delay_seconds), 8.0))

        parser = target.get("parser", "css")
        if parser == "mercari_item_links":
            items = _parse_mercari_item_links(page, url, limit)
        elif parser == "yahoo_product":
            items = _parse_yahoo_product(page, url, limit)
        elif parser == "ebay_itm_cards":
            items = _parse_ebay_itm_cards(page, url, limit)
        else:
            items = _parse_css(page, target, url, limit)

        result["items"] = items
        result["count"] = len(items)
        return result
    finally:
        browser.close()


def search_markets(
    keyword: str,
    platforms: list[str] | None = None,
    limit: int = 10,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Search selected public markets with one low-volume request per market."""
    selected = platforms or ["mercari", "yahoo_auction", "ebay", "amazon_jp"]
    results = []
    for platform in selected:
        try:
            results.append(search_prices(keyword, platform, limit, dry_run=dry_run))
        except Exception as exc:  # noqa: BLE001
            results.append(
                {
                    "keyword": keyword,
                    "platform": platform,
                    "error": str(exc),
                    "items": [],
                    "count": 0,
                }
            )
    return {
        "keyword": keyword,
        "backend": "mixed",
        "results": results,
    }


def find_arbitrage(
    keyword: str,
    platforms: list[str] | None = None,
    limit: int = 8,
    *,
    dry_run: bool = False,
    min_profit_yen: int | None = None,
    min_profit_rate: float | None = None,
    budget_yen: int | None = None,
) -> dict[str, Any]:
    """Scan markets and rank buy→sell combos that pass warashibe profit gates."""
    # Local import avoids circular issues when CLI loads modules loosely.
    try:
        from . import core
    except ImportError:
        import core  # type: ignore

    selected = platforms or ["mercari", "yahoo_auction", "ebay", "amazon_jp"]
    market = search_markets(keyword, selected, limit, dry_run=dry_run)
    # KPI thresholds (core.KPI_DEFAULTS is the single source of truth)
    profit_yen = int(min_profit_yen) if min_profit_yen is not None else core.KPI_DEFAULTS["min_profit_yen"]
    profit_rate = float(min_profit_rate) if min_profit_rate is not None else core.KPI_DEFAULTS["min_profit_rate"]
    # Arb scans often target mid-ticket niches (GPU/golf); default wider than beginner 1万円.
    budget = int(budget_yen) if budget_yen is not None else int(os.environ.get("WARASHIBE_ARB_BUDGET", "80000"))

    # Shipping/overhead estimates by buy→sell route (yen).
    route_ship = {
        ("mercari", "yahoo_auction"): 700,
        ("yahoo_auction", "mercari"): 700,
        ("mercari", "ebay"): 3500,
        ("yahoo_auction", "ebay"): 3500,
        ("ebay", "mercari"): 4500,
        ("ebay", "yahoo_auction"): 4500,
        ("amazon_jp", "mercari"): 800,
        ("amazon_jp", "yahoo_auction"): 800,
        ("amazon_jp", "ebay"): 3500,
        ("mercari", "amazon_jp"): 900,
        ("yahoo_auction", "amazon_jp"): 900,
        ("ebay", "amazon_jp"): 4500,
    }

    by_platform: dict[str, list[dict[str, Any]]] = {}
    for block in market.get("results") or []:
        plat = block.get("platform")
        if not plat:
            continue
        usable = []
        for item in block.get("items") or []:
            landed = item.get("landed_price")
            if landed is None:
                landed = item.get("price")
            if not isinstance(landed, int) or landed <= 0:
                continue
            usable.append({**item, "landed_price": landed})
        # Drop extreme 1-yen junk if there is a denser cluster.
        if len(usable) >= 3:
            prices = sorted(i["landed_price"] for i in usable)
            median = prices[len(prices) // 2]
            usable = [i for i in usable if i["landed_price"] >= max(500, int(median * 0.25))]
        by_platform[plat] = usable

    combos: list[dict[str, Any]] = []
    for buy_p, buy_items in by_platform.items():
        if not buy_items:
            continue
        buy_item = min(buy_items, key=lambda x: x["landed_price"])
        buy_price = int(buy_item["landed_price"])
        if buy_price > budget:
            continue
        for sell_p, sell_items in by_platform.items():
            if sell_p == buy_p or not sell_items:
                continue
            # Conservative sell: upper quartile, not absolute max.
            sells = sorted(int(i["landed_price"]) for i in sell_items)
            sell_price = sells[int(len(sells) * 0.75)] if len(sells) >= 4 else sells[-1]
            ship = route_ship.get((buy_p, sell_p), 1000)
            profit = core.calc_profit(
                buy_price,
                sell_price,
                platform=sell_p if sell_p in core.PLATFORMS else "mercari",
                shipping_out=ship,
                packaging=int(core.DEFAULTS.get("packaging_cost", 80)),
            )
            # Override go with explicit thresholds (also handles ebay fee_flat quirks).
            go = profit["profit"] >= profit_yen and profit["profit_rate"] >= profit_rate
            sell_item = max(sell_items, key=lambda x: x.get("landed_price") or 0)
            combos.append(
                {
                    "keyword": keyword,
                    "buy_platform": buy_p,
                    "sell_platform": sell_p,
                    "buy_price": buy_price,
                    "sell_price": sell_price,
                    "shipping_out_est": ship,
                    "profit": profit["profit"],
                    "profit_rate": profit["profit_rate"],
                    "platform_fee": profit["platform_fee"],
                    "go": go,
                    "buy_title": buy_item.get("title"),
                    "buy_url": buy_item.get("url"),
                    "sell_sample_title": sell_item.get("title"),
                    "sell_sample_url": sell_item.get("url"),
                }
            )

    combos.sort(key=lambda c: (c["go"], c["profit"]), reverse=True)
    winners = [c for c in combos if c["go"]]
    return {
        "keyword": keyword,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "thresholds": {
            "min_profit_yen": profit_yen,
            "min_profit_rate": profit_rate,
            "budget_yen": budget,
        },
        "market": market,
        "combos": combos,
        "winners": winners,
        "winner_count": len(winners),
    }


PRICE_RESEARCH_SCHEMA = {
    "name": "warashibe_price_research",
    "description": "公開マーケット価格調査（メルカリ/ヤフオク/eBay/Amazon公式API）。購入・ログイン・出品なし。Amazonはスクレイプ禁止。",
    "parameters": {
        "type": "object",
        "properties": {
            "keyword": {"type": "string", "description": "商品名・型番・検索語"},
            "platforms": {"type": "array", "items": {"type": "string"}},
            "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            "dry_run": {"type": "boolean"},
            "arbitrage": {"type": "boolean", "description": "黒字になり得る売買ルートを同時評価"},
        },
        "required": ["keyword"],
    },
}


def handle_price_research(args: dict[str, Any], **_: Any) -> dict[str, Any]:
    platforms = args.get("platforms")
    limit = int(args.get("limit", 10))
    dry_run = bool(args.get("dry_run", False))
    if args.get("arbitrage"):
        return find_arbitrage(args.get("keyword", ""), platforms, limit, dry_run=dry_run)
    return search_markets(args.get("keyword", ""), platforms, limit, dry_run=dry_run)


def check_available() -> bool:
    try:
        import cloakbrowser  # noqa: F401
        return True
    except ImportError:
        return False


__all__ = [
    "search_prices",
    "search_markets",
    "find_arbitrage",
    "handle_price_research",
    "PRICE_RESEARCH_SCHEMA",
    "TARGETS",
]
