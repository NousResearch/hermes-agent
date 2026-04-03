"""LLM-based fallback extraction for when CSS selectors fail.

When a site updates its HTML structure and the YAML selectors no longer
match, this module strips the HTML down to text and asks an LLM to
extract product name, price, and stock information.

Two modes:
    1. **Automatic** — uses Hermes agent's auxiliary LLM client if available
    2. **Manual** — returns a structured prompt the agent itself can answer
"""

import json
import logging
import re
from typing import Any, Optional

from .parsers.base import ProductData
from .parsers.price_utils import parse_price, detect_currency

logger = logging.getLogger(__name__)

# Maximum characters of page text to send to the LLM
_MAX_TEXT_LENGTH: int = 4000


def _strip_html_to_text(html: str) -> str:
    """Remove HTML tags, scripts, styles; collapse whitespace."""
    # Remove script/style blocks
    text = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    # Remove remaining tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode common entities
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text[:_MAX_TEXT_LENGTH]


def _build_extraction_prompt(page_text: str, url: str) -> str:
    """Build the prompt for the LLM extraction request."""
    return (
        "You are a product data extraction assistant. "
        "Given the following text content from an e-commerce product page, "
        "extract the product information and return it as valid JSON.\n\n"
        f"URL: {url}\n\n"
        f"Page content:\n{page_text}\n\n"
        "Return a JSON object with these exact keys:\n"
        '{"name": "product name", "price": 123.45, "currency": "USD", '
        '"stock_status": "in_stock|out_of_stock|limited|unknown", '
        '"seller": "seller name or empty string"}\n\n'
        "Rules:\n"
        "- price must be a number (float), not a string\n"
        "- If you cannot find a field, use null for price, empty string for others\n"
        "- Detect currency from the page content or URL\n"
        "- Return ONLY the JSON object, nothing else"
    )


def _parse_llm_response(response_text: str, currency: str) -> Optional[ProductData]:
    """Parse the LLM's JSON response into a ProductData object."""
    try:
        # Try to find JSON in the response
        json_match = re.search(r"\{[^{}]*\}", response_text, re.DOTALL)
        if not json_match:
            logger.warning("LLM fallback: No JSON found in response")
            return None

        data_dict = json.loads(json_match.group(0))

        result = ProductData()
        result.name = str(data_dict.get("name", "")).strip()

        price_val = data_dict.get("price")
        if price_val is not None:
            try:
                result.price = float(price_val)
            except (ValueError, TypeError):
                result.price = parse_price(str(price_val), currency)

        result.currency = str(data_dict.get("currency", currency)).upper() or currency
        result.stock_status = str(data_dict.get("stock_status", "unknown"))
        result.seller = str(data_dict.get("seller", ""))

        if result.stock_status not in ("in_stock", "out_of_stock", "limited", "unknown"):
            result.stock_status = "unknown"

        return result

    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning("LLM fallback: Failed to parse response: %s", exc)
        return None


def extract_with_llm(
    html: str,
    url: str,
    auxiliary_client: Any = None,
) -> Optional[ProductData]:
    """Extract product data from HTML using an LLM.

    Args:
        html: Raw HTML of the product page.
        url: The URL the page was fetched from.
        auxiliary_client: An LLM client with a ``chat()`` method.
            If ``None``, returns ``None`` (caller should use manual mode).

    Returns:
        A ``ProductData`` instance on success, or ``None``.
    """
    if not html or not html.strip():
        return None

    page_text = _strip_html_to_text(html)
    if len(page_text) < 50:
        logger.debug("LLM fallback: Page text too short (%d chars)", len(page_text))
        return None

    currency = detect_currency(url, html)
    prompt = _build_extraction_prompt(page_text, url)

    if auxiliary_client is None:
        logger.info(
            "LLM fallback: No auxiliary client available — "
            "returning None (agent should manually extract)"
        )
        return None

    try:
        response = auxiliary_client.chat(prompt)
        if not response:
            return None
        return _parse_llm_response(response, currency)
    except Exception as exc:
        logger.warning("LLM fallback: Client call failed: %s", exc)
        return None


def get_manual_extraction_prompt(html: str, url: str) -> str:
    """Return a prompt the agent itself can answer for manual extraction.

    Use this when no auxiliary LLM client is available — inject the prompt
    as a user message so the Hermes agent itself extracts the data.
    """
    page_text = _strip_html_to_text(html)
    return _build_extraction_prompt(page_text, url)
