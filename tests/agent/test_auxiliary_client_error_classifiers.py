"""Error-classification predicates for the auxiliary-client fallback chain.

``_is_payment_error`` / ``_is_model_not_found_error`` /
``_is_model_incompatible_error`` / ``_is_rate_limit_error`` decide which fallback
lane a failed auxiliary call may enter; ``_is_timeout_error`` /
``_is_connection_error`` separate a full-budget timeout from a fast connection
drop. ``_refresh_nous_recommended_model`` is the stale-404 companion: it turns
the not-found verdict into a usable replacement model.

Extracted byte-verbatim from ``tests/agent/test_auxiliary_client.py``.

Part of #79917, #78647
"""

import pytest

from agent.auxiliary_client import (
    _NOUS_MODEL,
    _is_connection_error,
    _is_model_incompatible_error,
    _is_model_not_found_error,
    _is_payment_error,
    _is_rate_limit_error,
    _is_timeout_error,
    _refresh_nous_recommended_model,
)
# The parent module's file-level autouse fixture still governs these tests: it
# strips provider env vars and clears the process-wide unhealthy-provider cache.
from tests.agent.test_auxiliary_client import _clean_env  # noqa: F401


# ── Payment / credit exhaustion fallback ─────────────────────────────────


class TestIsPaymentError:
    """_is_payment_error detects 402 and credit-related errors."""

    def test_402_status_code(self):
        exc = Exception("Payment Required")
        exc.status_code = 402
        assert _is_payment_error(exc) is True




    @pytest.mark.parametrize("spelling", ["RESOURCE_EXHAUSTED", "ResourceExhausted", "resource-exhausted"])
    @pytest.mark.parametrize("status", [None, 429])
    def test_resource_exhausted_separator_variants_are_payment(self, spelling, status):
        """NIM / gRPC wrappers serialize the quota signal without the space; the fallback gate
        must read every spelling like the literal ``resource exhausted`` (#85649)."""
        exc = Exception(f"{spelling}: Worker local total request limit reached (32/32)")
        if status is not None:
            exc.status_code = status
        assert _is_payment_error(exc) is True

    def test_403_subscription_required_is_payment(self):
        exc = Exception(
            "this model requires a subscription, upgrade for access: "
            "https://ollama.com/upgrade"
        )
        setattr(exc, "status_code", 403)
        assert _is_payment_error(exc) is True


    def test_404_generic_not_found_is_not_payment(self):
        exc = Exception("Not Found")
        exc.status_code = 404
        assert _is_payment_error(exc) is False





    # ── Daily / monthly quota exhaustion (#26803) ────────────────────────────








class TestIsModelNotFoundError:
    """_is_model_not_found_error detects stale/invalid model 404s, distinct
    from payment errors."""

    def test_nous_openrouter_catalog_404(self):
        """The exact incident error: a Portal-recommended model dropped from
        the Nous → OpenRouter catalog."""
        exc = Exception(
            "Model 'gpt-5.4-mini' not found. The requested model does not "
            "exist in our configuration or OpenRouter catalog."
        )
        exc.status_code = 404
        assert _is_model_not_found_error(exc) is True




    def test_billing_404_is_not_model_not_found(self):
        """Free-tier / credit 404s belong to _is_payment_error, not here —
        the two predicates must not overlap."""
        exc = Exception(
            "Model 'gpt-5' is not available on the free tier. Upgrade."
        )
        exc.status_code = 404
        assert _is_model_not_found_error(exc) is False
        assert _is_payment_error(exc) is True

    def test_out_of_funds_404_is_not_model_not_found(self):
        exc = Exception(
            "Your API key is blocked or out of funds. model_not_found"
        )
        exc.status_code = 404
        # billing keyword wins — payment owns it
        assert _is_model_not_found_error(exc) is False




class TestIsModelIncompatibleError:
    """_is_model_incompatible_error detects 400s where the route cannot run
    the model at all (capability mismatch), distinct from not-found and
    payment errors."""

    def test_codex_chatgpt_account_model_gating(self):
        """The exact incident: an openai-codex/ChatGPT-account fallback asked
        to compress a glm-5.2 conversation."""
        exc = Exception(
            "Error code: 400 - {'detail': \"The 'glm-5.2' model is not "
            "supported when using Codex with a ChatGPT account.\"}"
        )
        exc.status_code = 400
        assert _is_model_incompatible_error(exc) is True



    def test_not_found_is_not_incompatible(self):
        """A model-does-not-exist 400 belongs to _is_model_not_found_error —
        the two predicates must not overlap."""
        exc = Exception("openrouter/foo/bar is not a valid model ID")
        exc.status_code = 400
        assert _is_model_incompatible_error(exc) is False
        assert _is_model_not_found_error(exc) is True

    def test_payment_400_is_not_incompatible(self):
        """A billing 400 that also contains capability-ish phrasing must be
        rejected here — billing keywords win so the payment path owns it and
        the two buckets don't overlap."""
        exc = Exception("insufficient credits: model is not supported on free tier")
        exc.status_code = 400
        assert _is_model_incompatible_error(exc) is False




class TestRefreshNousRecommendedModel:
    """_refresh_nous_recommended_model picks a fresh model after a stale 404."""



    def test_falls_back_to_default_when_portal_unavailable(self, monkeypatch):
        def _boom(**kw):
            raise RuntimeError("portal down")
        monkeypatch.setattr(
            "hermes_cli.models.get_nous_recommended_aux_model", _boom)
        out = _refresh_nous_recommended_model(
            vision=False, stale_model="some/dead-model")
        assert out == _NOUS_MODEL

    def test_returns_none_when_no_distinct_alternative(self, monkeypatch):
        """When the failed model IS the default and the Portal has nothing
        else, there's no usable alternative."""
        monkeypatch.setattr(
            "hermes_cli.models.get_nous_recommended_aux_model",
            lambda **kw: _NOUS_MODEL,
        )
        out = _refresh_nous_recommended_model(
            vision=False, stale_model=_NOUS_MODEL)
        assert out is None


class TestIsRateLimitError:
    """_is_rate_limit_error detects 429 rate-limit errors warranting fallback."""

    def test_429_with_rate_limit_message(self):
        exc = Exception("Rate limit exceeded, try again in 2 seconds")
        exc.status_code = 429
        assert _is_rate_limit_error(exc) is True








    def test_openai_ratelimiterror_classname(self):
        """OpenAI SDK RateLimitError may omit .status_code — detect by class name."""
        class RateLimitError(Exception):
            pass
        exc = RateLimitError("rate limit exceeded")
        # No status_code set, but class name matches
        assert _is_rate_limit_error(exc) is True


class TestIsTimeoutError:
    """_is_timeout_error distinguishes a full-budget timeout from a fast
    connection drop."""

    def test_timed_out_string(self):
        assert _is_timeout_error(Exception("Request timed out.")) is True

    def test_timeout_typename(self):
        class ReadTimeout(Exception):
            pass

        assert _is_timeout_error(ReadTimeout("slow")) is True




class TestIsConnectionError:
    """Tests for _is_connection_error detection."""

    def test_connection_refused(self):
        err = Exception("Connection refused")
        assert _is_connection_error(err) is True



    def test_normal_api_error_not_connection(self):
        err = Exception("Bad Request: invalid model")
        err.status_code = 400
        assert _is_connection_error(err) is False
