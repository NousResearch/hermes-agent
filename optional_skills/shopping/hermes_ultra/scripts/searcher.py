"""Multi-source product searcher — searches Amazon, eBay, Best Buy, Newegg, Walmart concurrently.

Returns price data from ALL available sources so the caller can build a
Market Overview comparison table and pick the Best Deal.
"""

import logging
import re
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from .scraper import StealthScraper

logger = logging.getLogger(__name__)


class ProductSearcher:
    """Searches for products across multiple international stores."""

    MAX_WORKERS: int = 5

    def __init__(self, scraper: Optional[StealthScraper] = None) -> None:
        if scraper is None:
            self.scraper = StealthScraper()
        else:
            self.scraper = scraper

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search_all_sources(self, query: str) -> List[Dict[str, Any]]:
        """Search for a product across all supported stores concurrently.

        Returns a list of results sorted by price (cheapest first).
        Each result: ``{success, site, url, name, price, stock_status}``
        """
        logger.info("Multi-source search for: %s", query)

        search_fns = [
            ("Amazon", self._search_amazon),
            ("eBay", self._search_ebay),
            ("Best Buy", self._search_bestbuy),
            ("Newegg", self._search_newegg),
            ("Walmart", self._search_walmart),
        ]

        from .parsers import _PARSERS, _ensure_loaded
        _ensure_loaded()
        for parser in _PARSERS:
            site_name = parser.get_site_name()
            if site_name not in [s[0] for s in search_fns]:
                search_fns.append(
                    (site_name, lambda q, p=parser: p.search(q, self.scraper))
                )

        results: List[Dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            future_to_site = {
                executor.submit(fn, query): site_name
                for site_name, fn in search_fns
            }
            for future in as_completed(future_to_site):
                site_name = future_to_site[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.debug("Search failed for %s: %s", site_name, e)

        results.sort(key=lambda r: r.get("price") or float("inf"))
        return results

    def search(self, query: str) -> Dict[str, Any]:
        """Compatibility wrapper — returns the cheapest result."""
        all_results = self.search_all_sources(query)
        if not all_results:
            return {
                "success": False,
                "error": "Product not found across any store, or blocked by anti-bot systems.",
            }
        best = all_results[0]
        best["all_results"] = all_results
        return best

    # ------------------------------------------------------------------
    # Per-store search methods
    # ------------------------------------------------------------------

    def _search_amazon(self, query: str) -> Optional[Dict[str, Any]]:
        encoded_query = urllib.parse.quote_plus(query)
        search_url = f"https://www.amazon.com/s?k={encoded_query}"
        result = self.scraper.scrape(search_url)
        if not result.success:
            return None

        html = result.html
        if "api-services-support@amazon.com" in html or "Type the characters you see" in html or "ap_email" in html:
            return {"success": True, "site": "Amazon", "url": search_url, "name": "", "price": None, "stock_status": "unknown", "note": "Amazon bot protection active."}

        matches = re.finditer(r'<a\s+[^>]*href="(/[^"]*?/dp/[A-Z0-9]{10}[^"]*|/dp/[A-Z0-9]{10}[^"]*)"', html, re.IGNORECASE)
        for match in matches:
            product_path = match.group(1)
            if "slredirect" not in product_path:
                base_path = product_path.split("?", 1)[0]
                product_url = f"https://www.amazon.com{base_path}"
                return self._scrape_product_page(product_url, "Amazon")
        return None

    def _search_ebay(self, query: str) -> Optional[Dict[str, Any]]:
        encoded_query = urllib.parse.quote_plus(query)
        search_url = f"https://www.ebay.com/sch/i.html?_nkw={encoded_query}&LH_BIN=1&_sop=15"
        result = self.scraper.scrape(search_url)
        if not result.success:
            return None

        html = result.html
        if "captcha" in html.lower() or "px-captcha" in html.lower():
            return {"success": True, "site": "eBay", "url": search_url, "name": "", "price": None, "stock_status": "unknown", "note": "eBay is currently shielding."}

        matches = re.finditer(r'<a\s+[^>]*href="(https://www\.ebay\.com/itm/[0-9]+[^"]*)"', html, re.IGNORECASE)
        for match in matches:
            product_url = match.group(1).split("?", 1)[0]
            return self._scrape_product_page(product_url, "eBay")
        return None

    def _search_bestbuy(self, query: str) -> Optional[Dict[str, Any]]:
        encoded_query = urllib.parse.quote_plus(query)
        search_url = f"https://www.bestbuy.com/site/searchpage.jsp?st={encoded_query}"
        result = self.scraper.scrape(search_url)
        if not result.success:
            return None

        matches = re.finditer(r'<a\s+[^>]*href="(/site/[^"]+/\d+\.p[^"]*)"', result.html, re.IGNORECASE)
        for match in matches:
            product_path = match.group(1).split("?", 1)[0]
            product_url = f"https://www.bestbuy.com{product_path}"
            return self._scrape_product_page(product_url, "Best Buy")
        return None

    def _search_newegg(self, query: str) -> Optional[Dict[str, Any]]:
        encoded_query = urllib.parse.quote_plus(query)
        search_url = f"https://www.newegg.com/p/pl?d={encoded_query}"
        result = self.scraper.scrape(search_url)
        if not result.success:
            return None

        matches = re.finditer(r'<a\s+[^>]*href="(https://www\.newegg\.com/[^"]+/p/[^"]+)"', result.html, re.IGNORECASE)
        for match in matches:
            product_url = match.group(1).split("?", 1)[0]
            if "/p/pl" in product_url:
                continue
            return self._scrape_product_page(product_url, "Newegg")
        return None

    def _search_walmart(self, query: str) -> Optional[Dict[str, Any]]:
        encoded_query = urllib.parse.quote_plus(query)
        search_url = f"https://www.walmart.com/search?q={encoded_query}"
        result = self.scraper.scrape(search_url)
        if not result.success:
            return None

        matches = re.finditer(r'<a\s+[^>]*href="(/ip/[^"]+)"', result.html, re.IGNORECASE)
        for match in matches:
            product_path = match.group(1).split("?", 1)[0]
            product_url = f"https://www.walmart.com{product_path}"
            return self._scrape_product_page(product_url, "Walmart")
        return None

    # ------------------------------------------------------------------
    # Shared product page scraper
    # ------------------------------------------------------------------

    def _scrape_product_page(self, url: str, site: str) -> Optional[Dict[str, Any]]:
        """Scrape the actual product page and extract price/name via the parser."""
        from .parsers import get_parser

        try:
            result = self.scraper.scrape(url)
            if not result.success:
                return {"success": True, "site": site, "url": url, "name": "", "price": None, "stock_status": "unknown"}

            parser = get_parser(url)
            if not parser:
                return {"success": True, "site": site, "url": url, "name": "", "price": None, "stock_status": "unknown"}

            data = parser.parse(result.html, url)

            # LLM fallback if selectors failed
            if data.price is None:
                from .llm_fallback import extract_with_llm
                llm_data = extract_with_llm(result.html, url)
                if llm_data and llm_data.price:
                    data = llm_data
                    logger.info("LLM fallback extracted price for %s: %s", site, data.price)

            return {
                "success": True,
                "site": site,
                "url": url,
                "name": data.name or "",
                "price": data.price,
                "stock_status": data.stock_status or "unknown",
                "original_price": data.original_price,
                "seller": data.seller or site,
            }
        except Exception as e:
            logger.debug("Failed to scrape %s (%s): %s", site, url, e)
            return {"success": True, "site": site, "url": url, "name": "", "price": None, "stock_status": "unknown"}
