"""eBay Global parser — handles all eBay domains.

Selectors loaded from ``selectors.yaml``.
"""

import logging
import re
from typing import List

from .base import BaseSiteParser, ProductData
from .price_utils import detect_currency, parse_price, extract_text, extract_json_ld
from .selector_loader import get_selectors, get_site_config

logger = logging.getLogger(__name__)


class EbayGlobalParser(BaseSiteParser):
    """Parser for all eBay product/listing pages worldwide."""

    def __init__(self) -> None:
        cfg = get_site_config("ebay")
        self._domains: List[str] = cfg.get("domains", [])

    def get_site_name(self) -> str:
        return "eBay (Global)"

    def get_domains(self) -> List[str]:
        domains: List[str] = []
        for d in self._domains:
            domains.extend([d, f"www.{d}"])
        return domains

    def can_handle(self, url: str) -> bool:
        return any(d in url.lower() for d in self._domains)

    def parse(self, html: str, url: str = "") -> ProductData:
        data = ProductData()
        currency = detect_currency(url, html)
        data.currency = currency

        # --- Try JSON-LD first ---
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
                            data.price = parse_price(str(price_val), currency)
                    pc = offers.get("priceCurrency", "")
                    if pc:
                        data.currency = pc.upper()
                    avail = offers.get("availability", "").lower()
                    if "instock" in avail:
                        data.stock_status = "in_stock"
                    elif "outofstock" in avail:
                        data.stock_status = "out_of_stock"
                if data.name:
                    break

        # --- HTML fallbacks (from YAML) ---
        if not data.name:
            data.name = extract_text(html, get_selectors("ebay", "name"))

        if not data.price:
            price_text = extract_text(html, get_selectors("ebay", "price"))
            if price_text:
                data.price = parse_price(price_text, currency)

        # Bid price fallback
        if not data.price:
            bid_text = extract_text(html, get_selectors("ebay", "bid_price"))
            if bid_text:
                data.price = parse_price(bid_text, currency)

        # Seller
        data.seller = extract_text(html, get_selectors("ebay", "seller"))

        # Image
        for pattern in get_selectors("ebay", "image"):
            img_match = re.search(pattern, html)
            if img_match:
                data.image_url = img_match.group(1)
                break

        # Stock
        if data.stock_status == "unknown":
            html_lower = html.lower()
            if "ended" in html_lower or "bidding has ended" in html_lower:
                data.stock_status = "out_of_stock"
            elif data.price:
                data.stock_status = "in_stock"

        return data
