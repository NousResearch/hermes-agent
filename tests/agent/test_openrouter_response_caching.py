"""Opt-in OpenRouter response caching headers driven by env vars.

See https://openrouter.ai/announcements/response-caching.
"""
import os
from unittest.mock import patch

import pytest

from agent.auxiliary_client import (
    _OR_HEADERS,
    openrouter_default_headers,
    openrouter_feature_headers,
)


@pytest.fixture(autouse=True)
def _clear_cache_env(monkeypatch):
    monkeypatch.delenv("HERMES_OPENROUTER_CACHE", raising=False)
    monkeypatch.delenv("HERMES_OPENROUTER_CACHE_TTL", raising=False)


def test_feature_headers_empty_by_default():
    assert openrouter_feature_headers() == {}


def test_default_headers_match_attribution_when_disabled():
    assert openrouter_default_headers() == dict(_OR_HEADERS)


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "Yes", "on"])
def test_truthy_env_enables_cache_header(monkeypatch, value):
    monkeypatch.setenv("HERMES_OPENROUTER_CACHE", value)
    headers = openrouter_default_headers()
    assert headers["X-OpenRouter-Cache"] == "true"
    # Attribution headers are still present.
    assert headers["HTTP-Referer"] == _OR_HEADERS["HTTP-Referer"]
    assert headers["X-OpenRouter-Title"] == _OR_HEADERS["X-OpenRouter-Title"]


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "maybe"])
def test_non_truthy_env_keeps_cache_header_off(monkeypatch, value):
    monkeypatch.setenv("HERMES_OPENROUTER_CACHE", value)
    assert "X-OpenRouter-Cache" not in openrouter_default_headers()


def test_ttl_env_emits_header_when_cache_enabled(monkeypatch):
    monkeypatch.setenv("HERMES_OPENROUTER_CACHE", "true")
    monkeypatch.setenv("HERMES_OPENROUTER_CACHE_TTL", "3600")
    headers = openrouter_default_headers()
    assert headers["X-OpenRouter-Cache"] == "true"
    assert headers["X-OpenRouter-Cache-TTL"] == "3600"


def test_ttl_ignored_when_cache_disabled(monkeypatch):
    monkeypatch.setenv("HERMES_OPENROUTER_CACHE_TTL", "3600")
    assert openrouter_default_headers() == dict(_OR_HEADERS)


@pytest.mark.parametrize("ttl", ["0", "86401", "abc", "-1", "12.5"])
def test_invalid_ttl_dropped_silently(monkeypatch, ttl):
    """OpenRouter accepts 1..86400 sec; out-of-range/non-int values are skipped.

    Cache header still emitted so the user opts into caching with the
    default TTL rather than a request hard-fail.
    """
    monkeypatch.setenv("HERMES_OPENROUTER_CACHE", "1")
    monkeypatch.setenv("HERMES_OPENROUTER_CACHE_TTL", ttl)
    headers = openrouter_default_headers()
    assert headers["X-OpenRouter-Cache"] == "true"
    assert "X-OpenRouter-Cache-TTL" not in headers


@pytest.mark.parametrize("ttl", ["1", "300", "86400"])
def test_ttl_boundary_values_accepted(monkeypatch, ttl):
    monkeypatch.setenv("HERMES_OPENROUTER_CACHE", "yes")
    monkeypatch.setenv("HERMES_OPENROUTER_CACHE_TTL", ttl)
    assert openrouter_default_headers()["X-OpenRouter-Cache-TTL"] == ttl
