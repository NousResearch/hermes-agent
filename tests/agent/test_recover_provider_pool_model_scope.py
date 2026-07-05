# Copyright 2025 Nous Research (Licensed under the Apache License, Version 2.0)
"""Regression tests for model-scoped exhaustion in _recover_provider_pool.

Bug: auxiliary-task recovery (compression, summarization, vision, etc.) called
pool.mark_exhausted_and_rotate() WITHOUT a model_id. On a shared openai-codex
credential, a gpt-5.5 rate-limit therefore wrote a provider-wide block, which
silently swept sibling models (e.g. Spark) into the same cooldown — undoing the
per-model scoping the rest of the fix adds.

These tests assert the model that hit the limit is threaded through so the
block stays scoped to it.
"""

from unittest.mock import MagicMock, patch

from agent.auxiliary_client import _recover_provider_pool


def _rate_limit_exc():
    exc = Exception("Error code: 429 - usage_limit_reached")
    exc.status_code = 429
    return exc


def _auth_exc():
    exc = Exception("Error code: 401 - Unauthorized")
    exc.status_code = 401
    return exc


def _mock_pool():
    pool = MagicMock()
    pool.has_credentials.return_value = True
    pool.try_refresh_current.return_value = None
    pool.mark_exhausted_and_rotate.return_value = MagicMock()  # truthy next entry
    return pool


def test_rate_limit_recovery_threads_model_id():
    """A 429 on gpt-5.5 must be recorded scoped to gpt-5.5, not provider-wide."""
    pool = _mock_pool()
    with patch("agent.auxiliary_client.load_pool", return_value=pool), \
            patch("agent.auxiliary_client._evict_cached_clients"):
        ok = _recover_provider_pool(
            "openai-codex", _rate_limit_exc(), model_id="gpt-5.5"
        )
    assert ok is True
    pool.mark_exhausted_and_rotate.assert_called_once()
    assert pool.mark_exhausted_and_rotate.call_args.kwargs["model_id"] == "gpt-5.5"


def test_auth_recovery_threads_model_id():
    """The auth-error rotation path must also carry the model_id."""
    pool = _mock_pool()
    with patch("agent.auxiliary_client.load_pool", return_value=pool), \
            patch("agent.auxiliary_client._evict_cached_clients"):
        ok = _recover_provider_pool(
            "openai-codex", _auth_exc(), model_id="gpt-5.5"
        )
    assert ok is True
    pool.mark_exhausted_and_rotate.assert_called_once()
    assert pool.mark_exhausted_and_rotate.call_args.kwargs["model_id"] == "gpt-5.5"


def test_recovery_without_model_id_defaults_to_none():
    """Back-compat: omitting model_id still works (provider-wide block)."""
    pool = _mock_pool()
    with patch("agent.auxiliary_client.load_pool", return_value=pool), \
            patch("agent.auxiliary_client._evict_cached_clients"):
        ok = _recover_provider_pool("openai-codex", _rate_limit_exc())
    assert ok is True
    assert pool.mark_exhausted_and_rotate.call_args.kwargs["model_id"] is None
