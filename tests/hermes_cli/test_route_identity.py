"""Tests for hermes_cli/route_identity.py — normalize_route_base_url."""

import pytest
from unittest.mock import patch


def test_empty_and_whitespace():
    from hermes_cli.route_identity import normalize_route_base_url
    assert normalize_route_base_url("") == ""
    assert normalize_route_base_url(None) == ""


@pytest.mark.parametrize("url,expected", [
    ("http://example.com", "http://example.com"),
    ("HTTPS://EXAMPLE.COM", "https://example.com"),
    ("http://example.com:8080", "http://example.com:8080"),
    ("https://example.com:443", "https://example.com"),
    ("http://example.com:80", "http://example.com"),
])
def test_scheme_and_host_normalization(url, expected):
    from hermes_cli.route_identity import normalize_route_base_url
    assert normalize_route_base_url(url) == expected


def test_trailing_slash_removed():
    from hermes_cli.route_identity import normalize_route_base_url
    assert normalize_route_base_url("http://example.com/") == "http://example.com"
    assert normalize_route_base_url("http://example.com/path/") == "http://example.com/path"


def test_trailing_slash_preserved_when_query_was_empty():
    from hermes_cli.route_identity import normalize_route_base_url
    assert normalize_route_base_url("http://example.com/?") == "http://example.com/?"


def test_query_string_preserved():
    from hermes_cli.route_identity import normalize_route_base_url
    assert normalize_route_base_url("http://example.com/api?key=val") == "http://example.com/api?key=val"


def test_credential_stripped():
    from hermes_cli.route_identity import normalize_route_base_url
    result = normalize_route_base_url("http://USER:PASS@example.com")
    assert "USER:PASS" not in result
    assert "example.com" in result


def test_ipv6_normalized():
    from hermes_cli.route_identity import normalize_route_base_url
    result = normalize_route_base_url("http://[::1]:8080/path")
    assert "[::1]" in result
    assert ":8080" in result


def test_unicode_and_control_chars_return_raw():
    from hermes_cli.route_identity import normalize_route_base_url
    dirty = "http://\nbroken.com"
    assert normalize_route_base_url(dirty) == dirty


def test_fragment_dropped():
    from hermes_cli.route_identity import normalize_route_base_url
    assert normalize_route_base_url("http://example.com/path#section") == "http://example.com/path"
