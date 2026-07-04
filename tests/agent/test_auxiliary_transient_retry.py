"""Transient-transport retry count + per-model client-cache isolation.

Two related hardening behaviors for auxiliary calls (which include MoA
reference advisors, a pinned-model path where provider fallback is not a
meaningful recovery):

1. A transient transport blip (connection reset / timeout / 5xx) is retried
   on the SAME provider several times with backoff before giving up — a single
   upstream blip should not silently lose a pinned auxiliary call (root of the
   run2 double-advisor "Connection error" collapse).
2. Two auxiliary calls to the same provider/base_url/key but DIFFERENT models
   get DISTINCT client-cache keys, so a concurrent fan-out (e.g. opus + gpt-5.5
   advisors) never shares one client entry.
"""

from __future__ import annotations

import os
import types
from unittest.mock import patch

import pytest


class _ConnErr(Exception):
    """Stand-in that the transient detector recognizes as a connection blip."""


def test_transient_retry_count_default(monkeypatch):
    from agent import auxiliary_client as ac

    # No config value -> default.
    monkeypatch.setattr(ac, "load_config", lambda: {}, raising=False)
    with patch("hermes_cli.config.load_config", return_value={}), \
         patch("hermes_cli.config.cfg_get", return_value=None):
        assert ac._transient_retry_count() == ac._DEFAULT_TRANSIENT_RETRIES


def test_transient_retry_count_configurable_and_clamped():
    from agent import auxiliary_client as ac

    with patch("hermes_cli.config.cfg_get", return_value=4):
        assert ac._transient_retry_count() == 4
    with patch("hermes_cli.config.cfg_get", return_value=100):
        assert ac._transient_retry_count() == 6  # clamped high
    with patch("hermes_cli.config.cfg_get", return_value=-3):
        assert ac._transient_retry_count() == 0  # clamped low
    with patch("hermes_cli.config.cfg_get", side_effect=RuntimeError):
        assert ac._transient_retry_count() == ac._DEFAULT_TRANSIENT_RETRIES


def test_model_participates_in_client_cache_key():
    """Same provider/base_url/key, different model -> different cache key.

    This is what stops two concurrent advisors from sharing (and racing on)
    one cached client entry."""
    from agent.auxiliary_client import _client_cache_key

    k_opus = _client_cache_key(
        "openrouter", async_mode=False, base_url="https://openrouter.ai/api/v1",
        api_key="K", model="anthropic/claude-opus-4.8",
    )
    k_gpt = _client_cache_key(
        "openrouter", async_mode=False, base_url="https://openrouter.ai/api/v1",
        api_key="K", model="openai/gpt-5.5",
    )
    assert k_opus != k_gpt
    # Same model still collides (cache still works for reuse).
    k_opus2 = _client_cache_key(
        "openrouter", async_mode=False, base_url="https://openrouter.ai/api/v1",
        api_key="K", model="anthropic/claude-opus-4.8",
    )
    assert k_opus == k_opus2


def test_missing_model_key_is_stable():
    """Omitting model (legacy callers) is still a valid, stable key."""
    from agent.auxiliary_client import _client_cache_key

    a = _client_cache_key("openrouter", async_mode=False, base_url="u", api_key="k")
    b = _client_cache_key("openrouter", async_mode=False, base_url="u", api_key="k")
    assert a == b
