"""Best Buy parser — selectors loaded from ``selectors.yaml``."""

import logging
from typing import List

from .base import BaseSiteParser, ProductData
from .price_utils import parse_price, extract_text, extract_json_ld, extract_meta
from .selector_loader import get_selectors, get_site_config

logger = logging.getLogger(__name__)


class BestBuyParser(BaseSiteParser):
    """Parser for bestbuy.com product pages."""

    def __init__(self) -> None:
        cfg = get_site_config("bestbuy")
        self._domains: List[str] = cfg.get("domains", [])

    def get_site_name(self) -> str:
        return "Best Buy"

    def get_domains(self) -> List[str]:
        return [d for d in self._domains] + [f"www.{d}" for d in self._domains]

    def can_handle(self, url: str) -> bool:
        return any(d in url.lower() for d in self._domains)

    def parse(self, html: str, url: str = "") -> ProductData:
        data = ProductData()
        data.currency = "USD"

        # --- JSON-LD extraction ---
        for block in extract_json_ld(html):
            if isinstance(block, dict) and "Product" in str(block.get("@type", "")):
                data.name = block.get("name", "")
                offers = block.get("offers", {})
                if isinstance(offers, list):
                    offers = offers[0] if offers else {}
                if isinstance(offers, dict):
                    price_val = offers.get("price", "")
                    if price_val:
                        try:
                            data.price = float(price_val)
                        except (ValueError, TypeError):
                            data.price = parse_price(str(price_val), "USD")
                    avail = offers.get("availability", "").lower()
                    if "instock" in avail:
                        data.stock_status = "in_stock"
                    elif "outofstock" in avail:
                        data.stock_status = "out_of_stock"
                if data.name:
                    break

        # --- HTML fallbacks ---
        if not data.name:
            data.name = extract_text(html, get_selectors("bestbuy", "name"))

        if not data.price:
            price_text = extract_text(html, get_selectors("bestbuy", "price"))
            if price_text:
                data.price = parse_price(price_text, "USD")

        if not data.original_price:
            orig_text = extract_text(html, get_selectors("bestbuy", "original_price"))
            if orig_text:
                data.original_price = parse_price(orig_text, "USD")

        if data.stock_status == "unknown":
            html_lower = html.lower()
            if "sold out" in html_lower or "coming soon" in html_lower:
                data.stock_status = "out_of_stock"
            elif "add to cart" in html_lower:
                data.stock_status = "in_stock"
            elif data.price:
                data.stock_status = "in_stock"

        data.seller = "Best Buy"
        data.image_url = extract_meta(html, "image") or ""
        return data
