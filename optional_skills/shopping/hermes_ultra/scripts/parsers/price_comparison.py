"""Price comparison site parsers — Idealo, PriceSpy, CamelCamelCamel.

Selectors loaded from ``selectors.yaml``.
"""

import logging
import re
from typing import List

from .base import BaseSiteParser, ProductData
from .price_utils import detect_currency, parse_price, extract_text, extract_json_ld, extract_meta
from .selector_loader import get_selectors, get_site_config

logger = logging.getLogger(__name__)


class IdealoParser(BaseSiteParser):
    """Parser for idealo.de price comparison pages."""

    def __init__(self) -> None:
        cfg = get_site_config("idealo")
        self._domains: List[str] = cfg.get("domains", [])

    def get_site_name(self) -> str:
        return "Idealo"

    def get_domains(self) -> List[str]:
        return [d for d in self._domains] + [f"www.{d}" for d in self._domains]

    def can_handle(self, url: str) -> bool:
        return "idealo." in url.lower()

    def parse(self, html: str, url: str = "") -> ProductData:
        data = ProductData()
        currency = detect_currency(url, html)
        data.currency = currency

        for block in extract_json_ld(html):
            if isinstance(block, dict) and "Product" in str(block.get("@type", "")):
                data.name = block.get("name", "")
                offers = block.get("offers", {})
                if isinstance(offers, dict):
                    low = offers.get("lowPrice", offers.get("price", ""))
                    if low:
                        try:
                            data.price = float(low)
                        except (ValueError, TypeError):
                            data.price = parse_price(str(low), currency)
                    pc = offers.get("priceCurrency", "")
                    if pc:
                        data.currency = pc.upper()
                    data.stock_status = "in_stock" if data.price else "out_of_stock"
                if data.name:
                    break

        if not data.name:
            data.name = extract_text(html, get_selectors("idealo", "name"))

        if not data.price:
            price_text = extract_text(html, get_selectors("idealo", "price"))
            if price_text:
                data.price = parse_price(price_text, currency)

        data.seller = "Multiple Sellers (Idealo)"
        if data.price and data.stock_status == "unknown":
            data.stock_status = "in_stock"
        data.image_url = extract_meta(html, "image") or ""
        return data


class PriceSpyParser(BaseSiteParser):
    """Parser for pricespy.com / prisjakt / pricerunner."""

    def __init__(self) -> None:
        cfg = get_site_config("pricespy")
        self._domains: List[str] = cfg.get("domains", [])

    def get_site_name(self) -> str:
        return "PriceSpy"

    def get_domains(self) -> List[str]:
        return [d for d in self._domains] + [f"www.{d}" for d in self._domains]

    def can_handle(self, url: str) -> bool:
        url_lower = url.lower()
        return any(d in url_lower for d in ["pricespy.", "prisjakt.", "pricerunner."])

    def parse(self, html: str, url: str = "") -> ProductData:
        data = ProductData()
        currency = detect_currency(url, html)
        data.currency = currency

        for block in extract_json_ld(html):
            if isinstance(block, dict) and "Product" in str(block.get("@type", "")):
                data.name = block.get("name", "")
                offers = block.get("offers", {})
                if isinstance(offers, dict):
                    low = offers.get("lowPrice", offers.get("price", ""))
                    if low:
                        try:
                            data.price = float(low)
                        except (ValueError, TypeError):
                            data.price = parse_price(str(low), currency)
                    pc = offers.get("priceCurrency", "")
                    if pc:
                        data.currency = pc.upper()
                if data.name:
                    break

        if not data.name:
            data.name = extract_text(html, get_selectors("pricespy", "name"))

        if not data.price:
            price_text = extract_text(html, get_selectors("pricespy", "price"))
            if price_text:
                data.price = parse_price(price_text, currency)

        data.seller = "Multiple Sellers (PriceSpy)"
        data.image_url = extract_meta(html, "image") or ""
        if data.price:
            data.stock_status = "in_stock"
        return data


class CamelParser(BaseSiteParser):
    """Parser for camelcamelcamel.com (Amazon price history)."""

    def __init__(self) -> None:
        cfg = get_site_config("camel")
        self._domains: List[str] = cfg.get("domains", [])

    def get_site_name(self) -> str:
        return "CamelCamelCamel"

    def get_domains(self) -> List[str]:
        return [d for d in self._domains] + [f"www.{d}" for d in self._domains]

    def can_handle(self, url: str) -> bool:
        return "camelcamelcamel.com" in url.lower()

    def parse(self, html: str, url: str = "") -> ProductData:
        data = ProductData()
        data.currency = "USD"

        data.name = extract_text(html, get_selectors("camel", "name"))

        price_text = extract_text(html, get_selectors("camel", "price"))
        if price_text:
            data.price = parse_price(price_text, "USD")

        for pattern in get_selectors("camel", "image"):
            img_match = re.search(pattern, html)
            if img_match:
                data.image_url = img_match.group(1)
                break

        data.seller = "Amazon (via CamelCamelCamel)"
        if data.price:
            data.stock_status = "in_stock"
        return data
