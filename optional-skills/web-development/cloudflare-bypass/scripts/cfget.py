#!/usr/bin/env python3
"""Cloudflare bypass — JS challenge'ını geçer, sayfayı markdown basar.
Usage: python3 cfget.py <url> | head -N
"""
import sys, re
import cloudscraper
import html2text

url = sys.argv[1] if len(sys.argv) > 1 else input("URL: ")
scraper = cloudscraper.create_scraper()
r = scraper.get(url, timeout=30)
r.raise_for_status()

converter = html2text.HTML2Text()
converter.ignore_images = True
converter.body_width = 0
converter.unicode_snob = True

md = converter.handle(r.text)
md = re.sub(r'\n{3,}', '\n\n', md)
print(md.strip())
