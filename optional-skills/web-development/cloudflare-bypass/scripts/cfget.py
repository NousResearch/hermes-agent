#!/usr/bin/env python3
"""Cloudflare bypass — TLS fingerprint impersonation, outputs clean markdown.
Usage: python3 cfget.py <url> | head -N

Three-tier fallback:
  1. curl_cffi (primary) — TLS fingerprint impersonation, handles CF + mid-level protection
  2. cloudscraper (fallback) — basic Cloudflare JS challenge solver
  3. html2text (primary) — HTML → markdown
     regex (fallback) — strip tags, plain text
"""
import sys, re

try:
    from curl_cffi import requests
    HAS_CURL_CFFI = True
except ImportError:
    HAS_CURL_CFFI = False
    import cloudscraper

try:
    import html2text
    HAS_HTML2TEXT = True
except ImportError:
    HAS_HTML2TEXT = False


def get_page(url):
    """Get page content with best available method."""
    if HAS_CURL_CFFI:
        return requests.get(url, impersonate='chrome124', timeout=30)
    else:
        scraper = cloudscraper.create_scraper()
        return scraper.get(url, timeout=30)


def to_markdown(html):
    """Convert HTML to clean markdown."""
    if HAS_HTML2TEXT:
        converter = html2text.HTML2Text()
        converter.ignore_images = True
        converter.body_width = 0
        converter.unicode_snob = True
        md = converter.handle(html)
        md = re.sub(r'\n{3,}', '\n\n', md)
        return md.strip()
    else:
        clean = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
        clean = re.sub(r'<style[^>]*>.*?</style>', '', clean, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', ' ', clean)
        text = re.sub(r'\s+', ' ', text).strip()
        return text


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else input("URL: ")
    r = get_page(url)
    r.raise_for_status()
    print(to_markdown(r.text))


if __name__ == '__main__':
    main()
