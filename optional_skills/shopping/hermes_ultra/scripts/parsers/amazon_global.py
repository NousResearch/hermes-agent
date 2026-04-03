"""Amazon Global parser — handles all Amazon domains.

Supports: amazon.com, amazon.co.uk, amazon.de, amazon.fr, amazon.it,
          amazon.es, amazon.com.tr

All CSS/regex selectors are loaded from ``selectors.yaml`` via
:mod:`selector_loader` so that HTML structure changes only require
a YAML update.
"""

import logging
import re
from typing import List, Optional

from .base import BaseSiteParser, ProductData
from .price_utils import detect_currency, parse_price, extract_text
from .selector_loader import get_selectors, get_selector, get_site_config

logger = logging.getLogger(__name__)


class AmazonGlobalParser(BaseSiteParser):
    """Parser for all Amazon product pages worldwide."""

    def __init__(self) -> None:
        cfg = get_site_config("amazon")
        self._domains: List[str] = cfg.get("domains", [])
        self._domain_names: dict = cfg.get("domain_names", {})

    def get_site_name(self) -> str:
        return "Amazon (Global)"

    def get_domains(self) -> List[str]:
        domains: List[str] = []
        for d in self._domains:
            domains.extend([d, f"www.{d}"])
        return domains

    def can_handle(self, url: str) -> bool:
        url_lower = url.lower()
        return any(d in url_lower for d in self._domains)

    def _detect_amazon_domain(self, url: str) -> str:
        url_lower = url.lower()
        for d in sorted(self._domains, key=len, reverse=True):
            if d in url_lower:
                return d
        return "amazon.com"

    def parse(self, html: str, url: str = "") -> ProductData:
        data = ProductData()
        currency = detect_currency(url, html)
        data.currency = currency

        # --- Product name ---
        data.name = extract_text(html, get_selectors("amazon", "name"))

        # --- Current price (whole + fraction pattern) ---
        wf_pattern = get_selector("amazon", "price_whole_fraction")
        if wf_pattern:
            fraction_match = re.search(wf_pattern, html, re.DOTALL)
            if fraction_match:
                whole = fraction_match.group(1).replace(".", "").replace(",", "")
                frac = fraction_match.group(2)
                try:
                    data.price = float(f"{whole}.{frac}")
                except ValueError:
                    pass

        if not data.price:
            price_text = extract_text(html, get_selectors("amazon", "price"))
            if price_text:
                data.price = parse_price(price_text, currency)

        # Fallback: embedded JSON data
        if not data.price:
            json_pattern = get_selector("amazon", "price_json_fallback")
            if json_pattern:
                price_amount_match = re.search(json_pattern, html)
                if price_amount_match:
                    try:
                        data.price = float(price_amount_match.group(1))
                    except ValueError:
                        pass

        # Fallback: data-price attribute
        if not data.price:
            data_pattern = get_selector("amazon", "price_data_attr")
            if data_pattern:
                twister_match = re.search(data_pattern, html)
                if twister_match:
                    data.price = parse_price(twister_match.group(1), currency)

        # --- Original (list) price ---
        orig_text = extract_text(html, get_selectors("amazon", "original_price"))
        if orig_text:
            data.original_price = parse_price(orig_text, currency)

        # --- Stock status ---
        availability_text = extract_text(html, get_selectors("amazon", "stock")).lower()

        cfg = get_site_config("amazon")
        stock_kw = cfg.get("stock_keywords", {})
        oos_keywords = stock_kw.get("out_of_stock", [])
        limited_keywords = stock_kw.get("limited", [])
        in_stock_keywords = stock_kw.get("in_stock", [])

        if any(w in availability_text for w in oos_keywords):
            data.stock_status = "out_of_stock"
        elif any(w in availability_text for w in limited_keywords):
            data.stock_status = "limited"
        elif any(w in availability_text for w in in_stock_keywords):
            data.stock_status = "in_stock"
        elif "addtocart" in html.lower() or "add-to-cart" in html.lower():
            data.stock_status = "in_stock"
        else:
            data.stock_status = "unknown"

        # --- Seller ---
        data.seller = extract_text(html, get_selectors("amazon", "seller"))
        if not data.seller:
            html_lower = html.lower()
            sorted_domains = sorted(self._domain_names.items(), key=lambda x: len(x[0]), reverse=True)
            if url:
                for d, name in sorted_domains:
                    if d in url.lower():
                        if f"{d} tarafından" in html_lower or "ships from and sold by amazon" in html_lower:
                            data.seller = name
                        break
            else:
                for d, name in sorted_domains:
                    if f"{d} tarafından" in html_lower:
                        data.seller = name
                        break
                if not data.seller and "ships from and sold by amazon" in html_lower:
                    data.seller = "Amazon"

        # --- Image ---
        for pattern in get_selectors("amazon", "image"):
            img_match = re.search(pattern, html)
            if img_match:
                data.image_url = img_match.group(1)
                break

        # --- Category ---
        cat_pattern = get_selector("amazon", "category")
        if cat_pattern:
            cats = re.findall(cat_pattern, html)
            if cats:
                data.category = " > ".join(c.strip() for c in cats if c.strip())

        # --- Rating ---
        rating_patterns = get_selectors("amazon", "rating")
        rating_text = extract_text(html, rating_patterns)
        if rating_text:
            try:
                data.rating = float(rating_text.replace(",", "."))
            except ValueError:
                pass

        # --- Review count ---
        review_patterns = get_selectors("amazon", "review_count")
        review_text = extract_text(html, review_patterns)
        if review_text:
            try:
                data.review_count = int(review_text.replace(".", "").replace(",", ""))
            except ValueError:
                pass

        return data
