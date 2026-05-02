"""Pytest for search_offers BLOCKED_DOMAINS filtering + _domain helper.

We don't hit a live SearXNG; we test the pure logic that decides which
URLs survive the filter. The HTTP layer is exercised by --json smoke
in CI separately.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import search_offers as so  # noqa: E402


def test_domain_strips_www() -> None:
    assert so._domain("https://www.maxicoffee.com/foo") == "maxicoffee.com"


def test_domain_lowercase() -> None:
    assert so._domain("https://Example.COM/X") == "example.com"


def test_domain_invalid_returns_empty() -> None:
    assert so._domain("not a url") == ""


def test_blocked_domains_contains_known_aggregators() -> None:
    for d in ("idealo.fr", "youtube.com", "wikipedia.org", "reddit.com"):
        assert d in so.BLOCKED_DOMAINS


def test_blocked_domains_excludes_real_merchants() -> None:
    for d in ("amazon.fr", "fnac.com", "maxicoffee.com", "shopify.com"):
        assert d not in so.BLOCKED_DOMAINS
