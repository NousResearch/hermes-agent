"""Seam-identity + behavior regression tests for the R2 error-predicates leaf.

Extraction slice R2 (epic #78647 / #78635): the error-predicate leaf family
moved byte-verbatim from ``agent/auxiliary_client.py`` (window 3615-4017) into
``agent/auxiliary_client_errors.py``.  ``agent.auxiliary_client`` re-exports
every moved name, so all existing import / lazy-import / monkeypatch targets
must keep resolving to the SAME objects — verified here with real ``is``
identity assertions, plus representative classification smoke tests through
the godfile re-export path.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agent import auxiliary_client as godfile
from agent import auxiliary_client_errors as leaf

# Every name moved in the R2 extraction; must exist and be `is`-identical
# on both modules.
MOVED_NAMES = [
    "_AUX_UNHEALTHY_LABEL_ALIASES",
    "_AUX_UNHEALTHY_TTL_SECONDS",
    "_DEFAULT_TRANSIENT_RETRIES",
    "_TRANSIENT_RETRY_BACKOFF_BASE",
    "_aux_unhealthy_logged_at",
    "_aux_unhealthy_until",
    "_is_auth_error",
    "_is_connection_error",
    "_is_model_not_found_error",
    "_is_payment_error",
    "_is_provider_unhealthy",
    "_is_rate_limit_error",
    "_is_timeout_error",
    "_is_transient_transport_error",
    "_is_unsupported_parameter_error",
    "_is_unsupported_temperature_error",
    "_log_skip_unhealthy",
    "_mark_provider_unhealthy",
    "_normalize_chain_label",
    "_nous_portal_account_has_fresh_paid_access",
    "_reset_aux_unhealthy_cache",
    "_transient_retry_count",
]


class _Err(Exception):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


# ── Seam identity ──────────────────────────────────────────────────────────


def test_reexport_identity_for_every_moved_name():
    """The godfile re-exports the exact objects the leaf module owns."""
    for name in MOVED_NAMES:
        assert getattr(godfile, name) is getattr(leaf, name), name
        assert getattr(godfile, name) is not None, name


def test_leaf_owns_all_moved_names():
    for name in MOVED_NAMES:
        assert hasattr(leaf, name), name


# ── Predicate behavior (through the godfile re-export path) ───────────────


def test_is_payment_error():
    assert godfile._is_payment_error(_Err("insufficient credits", 402)) is True
    assert godfile._is_payment_error(_Err("You do not have enough credits")) is True
    assert godfile._is_payment_error(_Err("billing error: quota exceeded", 429)) is True
    assert godfile._is_payment_error(_Err("balance_depleted", 403)) is True
    # 402 wins regardless of message.
    assert godfile._is_payment_error(_Err("some other failure", 402)) is True
    # Non-payment errors are not payment errors.
    assert godfile._is_payment_error(_Err("rate limit reached", 429)) is False
    assert godfile._is_payment_error(_Err("internal error", 500)) is False
    assert godfile._is_payment_error(_Err("model does not exist", 404)) is False


def test_is_rate_limit_error():
    assert godfile._is_rate_limit_error(_Err("rate limit reached", 429)) is True
    assert godfile._is_rate_limit_error(_Err("too many requests, retry after 5s", 429)) is True
    # Generic 429 without billing keywords counts as a rate limit.
    assert godfile._is_rate_limit_error(_Err("try again later", 429)) is True
    # Billing-language 429s belong to _is_payment_error, not here.
    assert godfile._is_rate_limit_error(_Err("credits exhausted", 429)) is False
    # OpenAI SDK RateLimitError without status_code — class-name match.
    assert godfile._is_rate_limit_error(_Err("slow down", None)) is False
    rate_limit = type("RateLimitError", (Exception,), {})("slow down")
    assert godfile._is_rate_limit_error(rate_limit) is True


def test_is_auth_error():
    assert godfile._is_auth_error(_Err("invalid api key", 401)) is True
    assert godfile._is_auth_error(_Err("Error code: 401 - unauthorized")) is True
    # xAI OAuth: 403 + bad-credentials is semantically a 401.
    assert godfile._is_auth_error(_Err("unauthenticated:bad-credentials", 403)) is True
    assert godfile._is_auth_error(_Err("permission denied", 403)) is False
    assert godfile._is_auth_error(_Err("internal error", 500)) is False


def test_is_model_not_found_error():
    # Matches via "does not exist in our configuration" (real Nous proxy body).
    assert godfile._is_model_not_found_error(
        _Err("Model 'gpt-5.4-mini' not found. The requested model does not exist "
             "in our configuration or OpenRouter catalog.", 404)
    ) is True
    assert godfile._is_model_not_found_error(_Err("no such model: foo", 400)) is True
    # Matches via the "the model `" OpenAI-style marker.
    assert godfile._is_model_not_found_error(_Err("The model `x` does not exist", 404)) is True
    assert godfile._is_model_not_found_error(_Err("model not found for provider", 404)) is True
    # Free-tier / credit 404s belong to _is_payment_error, explicitly excluded here.
    assert godfile._is_model_not_found_error(_Err("not available on the free tier", 404)) is False
    assert godfile._is_payment_error(_Err("not available on the free tier", 404)) is True
    # Non-model 404.
    assert godfile._is_model_not_found_error(_Err("route not found", 404)) is False


def test_is_connection_error():
    assert godfile._is_connection_error(_Err("Connection reset by peer")) is True
    assert godfile._is_connection_error(_Err("name or service not known")) is True
    assert godfile._is_connection_error(_Err("incomplete chunked read")) is True
    conn_err = type("APIConnectionError", (Exception,), {})("boom")
    assert godfile._is_connection_error(conn_err) is True
    assert godfile._is_connection_error(_Err("bad request", 400)) is False


def test_is_timeout_error():
    assert godfile._is_timeout_error(_Err("Request timed out.")) is True
    assert godfile._is_timeout_error(_Err("operation timed out after 60s")) is True
    timeout_err = type("APITimeoutError", (Exception,), {})("timed out")
    assert godfile._is_timeout_error(timeout_err) is True
    assert godfile._is_timeout_error(_Err("model does not exist", 404)) is False


def test_is_transient_transport_error():
    assert godfile._is_transient_transport_error(_Err("Connection reset by peer")) is True
    assert godfile._is_transient_transport_error(_Err("upstream blew up", 503)) is True
    assert godfile._is_transient_transport_error(_Err("request timeout", 408)) is True
    assert godfile._is_transient_transport_error(_Err("bad request", 400)) is False
    assert godfile._is_transient_transport_error(_Err("rate limited", 429)) is False


def test_is_unsupported_parameter_error_and_temperature_wrapper():
    assert godfile._is_unsupported_parameter_error(
        _Err("Unsupported parameter: temperature"), "temperature"
    ) is True
    assert godfile._is_unsupported_parameter_error(
        _Err("Unrecognized request argument: max_tokens"), "max_tokens"
    ) is True
    assert godfile._is_unsupported_parameter_error(
        _Err("unknown parameter: seed"), "seed"
    ) is True  # "unknown parameter" is an explicit marker
    assert godfile._is_unsupported_parameter_error(
        _Err("Unexpected field: seed"), "seed"
    ) is False  # param present but no marker phrase
    assert godfile._is_unsupported_parameter_error(
        _Err("Unsupported parameter: temperature"), "max_tokens"
    ) is False  # wrong param
    assert godfile._is_unsupported_temperature_error(
        _Err("Unsupported parameter: temperature")
    ) is True
    assert godfile._is_unsupported_temperature_error(
        _Err("Unsupported parameter: max_tokens")
    ) is False


def test_normalize_chain_label():
    assert godfile._normalize_chain_label("OpenRouter") == "openrouter"
    assert godfile._normalize_chain_label("codex") == "openai-codex"
    assert godfile._normalize_chain_label("custom") == "local/custom"
    assert godfile._normalize_chain_label("deepseek") == "deepseek"
    assert godfile._normalize_chain_label("") == ""


def test_unhealthy_cache_mark_check_reset():
    try:
        godfile._reset_aux_unhealthy_cache()
        assert godfile._is_provider_unhealthy("openrouter") is False
        godfile._mark_provider_unhealthy("openrouter", ttl=60)
        assert godfile._is_provider_unhealthy("openrouter") is True
        # Alias normalization: "codex" marks the "openai-codex" chain label.
        godfile._mark_provider_unhealthy("codex", ttl=60)
        assert godfile._is_provider_unhealthy("openai-codex") is True
        assert godfile._is_provider_unhealthy("nous") is False
    finally:
        godfile._reset_aux_unhealthy_cache()
    assert godfile._is_provider_unhealthy("openrouter") is False


def test_transient_retry_count_default(monkeypatch):
    monkeypatch.setattr(godfile, "load_config", lambda: {}, raising=False)
    with patch("hermes_cli.config.load_config", return_value={}), \
         patch("hermes_cli.config.cfg_get", return_value=None):
        assert godfile._transient_retry_count() == godfile._DEFAULT_TRANSIENT_RETRIES
    # Explicit value, clamped to [0, 6].
    with patch("hermes_cli.config.load_config", return_value={}), \
         patch("hermes_cli.config.cfg_get", return_value=99):
        assert godfile._transient_retry_count() == 6


def test_nous_portal_paid_access_refresh_failure_is_false(monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("portal unreachable")

    monkeypatch.setattr(
        "hermes_cli.nous_account.get_nous_portal_account_info", boom, raising=False
    )
    assert godfile._nous_portal_account_has_fresh_paid_access() is False
