#!/usr/bin/env python3
"""Build a prefilled cart URL for a chosen product.

Usage:
  python3 build_cart.py --url 'https://www.terresdecafe.com/products/foo' \
    [--quantity 1] [--json]

Outputs (JSON mode):
  {
    "platform": "shopify" | "woocommerce" | "unknown",
    "cart_url": "https://merchant/cart/12345:1" | null,
    "detected_price_eur": 19.90 | null,
    "needs_user_finish": false | true,
    "strategy": "shopify_cart_add" | "woocommerce_add_to_cart" | "manual_link"
  }

Strategy:
  - Shopify     -> /cart/{variant_id}:{quantity} permalink (1-tap-ish)
  - WooCommerce -> ?add-to-cart=ID&quantity=N
  - Unknown     -> return the product URL with needs_user_finish=true

We never submit payment data; the user finishes on the merchant's UI.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from dataclasses import asdict, dataclass
from urllib.parse import urlparse

from playwright.async_api import async_playwright
from selectolax.parser import HTMLParser


SHOPIFY_MARKERS = ("cdn.shopify.com", "Shopify.theme", "/products/", '"product"')
WOO_MARKERS = ("woocommerce", "wc-block", "wp-content/plugins/woocommerce")
VARIANT_RE = re.compile(r'"id"\s*:\s*(\d{10,})')
WOO_PID_RE = re.compile(r'data-product[_-]?id="(\d+)"|add-to-cart=(\d+)')


@dataclass
class CartResult:
    platform: str
    cart_url: str | None
    detected_price_eur: float | None
    needs_user_finish: bool
    strategy: str


def _detect_platform(html: str) -> str:
    if any(m in html for m in SHOPIFY_MARKERS) and ("Shopify" in html or "/cart/" in html):
        return "shopify"
    lower = html.lower()
    if any(m in lower for m in WOO_MARKERS):
        return "woocommerce"
    return "unknown"


def _detect_price(html: str) -> float | None:
    tree = HTMLParser(html)
    for sel in [
        'meta[property="product:price:amount"]',
        'meta[itemprop="price"]',
        '[itemprop="price"]',
        ".price ins .amount",
        ".price .amount",
        ".price__current .money",
        ".product-price",
    ]:
        node = tree.css_first(sel)
        if not node:
            continue
        raw = node.attributes.get("content") or node.text(strip=True)
        if not raw:
            continue
        m = re.search(r"(\d+[.,]?\d*)", raw.replace(" ", ""))
        if m:
            try:
                return float(m.group(1).replace(",", "."))
            except ValueError:
                continue
    return None


async def build_cart_async(product_url: str, quantity: int) -> CartResult:
    parsed = urlparse(product_url)
    base_url = f"{parsed.scheme}://{parsed.netloc}"

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        try:
            ctx = await browser.new_context(locale="fr-FR")
            page = await ctx.new_page()
            await page.goto(product_url, wait_until="domcontentloaded", timeout=30_000)
            html = await page.content()
        finally:
            await browser.close()

    platform = _detect_platform(html)
    price = _detect_price(html)

    if platform == "shopify":
        m = VARIANT_RE.search(html)
        if m:
            try:
                variant_id = int(m.group(1))
                return CartResult(
                    platform="shopify",
                    cart_url=f"{base_url}/cart/{variant_id}:{quantity}",
                    detected_price_eur=price,
                    needs_user_finish=False,
                    strategy="shopify_cart_add",
                )
            except ValueError:
                pass

    if platform == "woocommerce":
        m = WOO_PID_RE.search(html)
        if m:
            pid = m.group(1) or m.group(2)
            if pid:
                sep = "&" if "?" in product_url else "?"
                return CartResult(
                    platform="woocommerce",
                    cart_url=f"{product_url}{sep}add-to-cart={pid}&quantity={quantity}",
                    detected_price_eur=price,
                    needs_user_finish=False,
                    strategy="woocommerce_add_to_cart",
                )

    return CartResult(
        platform=platform,
        cart_url=product_url,
        detected_price_eur=price,
        needs_user_finish=True,
        strategy="manual_link",
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--url", required=True, help="Product URL")
    p.add_argument("--quantity", type=int, default=1)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    try:
        cart = asyncio.run(build_cart_async(args.url, args.quantity))
    except Exception as exc:  # noqa: BLE001
        err = {"error": str(exc)}
        sys.stdout.write(json.dumps(err, ensure_ascii=False) + "\n")
        return 2

    if args.json:
        sys.stdout.write(json.dumps(asdict(cart), ensure_ascii=False, indent=2) + "\n")
    else:
        sys.stdout.write(
            f"platform: {cart.platform}\n"
            f"strategy: {cart.strategy}\n"
            f"price: {cart.detected_price_eur}\n"
            f"cart_url: {cart.cart_url}\n"
            f"needs_user_finish: {cart.needs_user_finish}\n"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
