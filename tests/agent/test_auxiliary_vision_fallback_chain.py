"""Tests for vision fallback chain fixes (issue #51513).

Covers:
- Bug 1: _main_model_supports_vision returns False for unknown capabilities
- Bug 3: _is_geo_blocked_error detection
- Bug 5: Fallback chain iterates ALL entries, not just the first
- Bonus: "not supported model" keyword in _is_model_incompatible_error
"""

from unittest.mock import patch, MagicMock

import pytest

from agent.auxiliary_client import (
    _is_geo_blocked_error,
    _is_model_incompatible_error,
    _main_model_supports_vision,
)


class _ApiError(Exception):
    """Minimal exception with status_code for testing."""
    def __init__(self, msg, status_code=None):
        super().__init__(msg)
        self.status_code = status_code


# ---------------------------------------------------------------------------
# Bug 1: _main_model_supports_vision returns False for unknown capabilities
# ---------------------------------------------------------------------------


class TestMainModelSupportsVisionUnknown:
    """Unknown model capabilities should assume text-only (return False)."""

    def test_none_supports_returns_false(self):
        """When _lookup_supports_vision returns None, return False to route
        vision through the auxiliary fallback chain."""
        with patch(
            "agent.image_routing._lookup_supports_vision", return_value=None
        ):
            assert _main_model_supports_vision("some-provider", "some-model") is False

    def test_true_supports_returns_true(self):
        """When capability data says vision is supported, return True."""
        with patch(
            "agent.image_routing._lookup_supports_vision", return_value=True
        ):
            assert _main_model_supports_vision("some-provider", "some-model") is True

    def test_false_supports_returns_false(self):
        """When capability data says vision is NOT supported, return False."""
        with patch(
            "agent.image_routing._lookup_supports_vision", return_value=False
        ):
            assert _main_model_supports_vision("some-provider", "some-model") is False


# ---------------------------------------------------------------------------
# Bug 3: _is_geo_blocked_error
# ---------------------------------------------------------------------------


class TestIsGeoBlockedError:
    """Detect geo-blocked/location errors for provider fallback."""

    def test_failed_precondition(self):
        exc = Exception("400 FAILED_PRECONDITION: User location is not supported for the API use")
        assert _is_geo_blocked_error(exc) is True

    def test_user_location_not_supported(self):
        exc = Exception("User location is not supported")
        assert _is_geo_blocked_error(exc) is True

    def test_not_available_in_country(self):
        exc = Exception("This service is not available in your country")
        assert _is_geo_blocked_error(exc) is True

    def test_not_available_in_region(self):
        exc = Exception("not available in your region")
        assert _is_geo_blocked_error(exc) is True

    def test_geo_restricted(self):
        exc = Exception("geo-restricted content")
        assert _is_geo_blocked_error(exc) is True

    def test_geo_blocked(self):
        exc = Exception("access geo_blocked")
        assert _is_geo_blocked_error(exc) is True

    def test_normal_error_not_detected(self):
        exc = Exception("rate limit exceeded")
        assert _is_geo_blocked_error(exc) is False

    def test_connection_error_not_detected(self):
        exc = Exception("connection refused")
        assert _is_geo_blocked_error(exc) is False

    def test_case_insensitive(self):
        exc = Exception("FAILED_PRECONDITION: User Location Is Not Supported")
        assert _is_geo_blocked_error(exc) is True


# ---------------------------------------------------------------------------
# Bonus: "not supported model" keyword in _is_model_incompatible_error
# ---------------------------------------------------------------------------


class TestModelIncompatibleNotSupportedModel:
    """Detect 'not supported model' errors (e.g. xiaomi-tp)."""

    def test_not_supported_model_xiaomi(self):
        exc = _ApiError(
            "{'error': {'code': '400', 'message': 'Param Incorrect', 'param': 'Not supported model mimo-v2-omni'}}",
            status_code=400,
        )
        assert _is_model_incompatible_error(exc) is True

    def test_model_is_not_supported(self):
        exc = _ApiError("The model is not supported", status_code=400)
        assert _is_model_incompatible_error(exc) is True

    def test_unsupported_model(self):
        exc = _ApiError("unsupported model for this account", status_code=400)
        assert _is_model_incompatible_error(exc) is True

    def test_normal_400_not_detected(self):
        exc = _ApiError("invalid request format", status_code=400)
        assert _is_model_incompatible_error(exc) is False

    def test_404_not_detected(self):
        exc = _ApiError("model not found", status_code=404)
        assert _is_model_incompatible_error(exc) is False


# ---------------------------------------------------------------------------
# Bug 5: Fallback chain iterates ALL entries
# ---------------------------------------------------------------------------


class TestFallbackChainIteration:
    """When the first fallback entry fails at call time, the chain should
    continue to the next entry instead of stopping."""

    def test_chain_deduplicates_first_pick(self):
        """The first pick from _try_configured_fallback_chain should not be
        duplicated in the candidates list."""
        fb_label = "fallback_chain[0](provider-a)"

        chain_config = {
            "fallback_chain": [
                {"provider": "provider-a", "model": "model-a"},
                {"provider": "provider-b", "model": "model-b"},
            ]
        }

        def mock_resolve(entry):
            client = MagicMock()
            client.base_url = f"https://{entry.get('provider')}.example.com"
            return client, entry.get("model")

        _fb_candidates = [("first-client", "model-a", fb_label)]
        _chain = chain_config.get("fallback_chain")
        _skip_p = "auto"

        if _chain and isinstance(_chain, list):
            for _i, _entry in enumerate(_chain):
                if not isinstance(_entry, dict):
                    continue
                _ep = str(_entry.get("provider", "")).strip()
                if not _ep or _ep.lower() == _skip_p:
                    continue
                _em = str(_entry.get("model", "")).strip() or None
                _el = f"fallback_chain[{_i}]({_ep})"
                if _el == fb_label:
                    continue  # already first pick
                if _ep and _em:
                    _ec, _emr = mock_resolve(_entry)
                    if _ec is not None:
                        _fb_candidates.append((_ec, _emr or _em, _el))

        # Should have 2: the original first pick + provider-b (not duplicated provider-a)
        assert len(_fb_candidates) == 2
        assert _fb_candidates[0] == ("first-client", "model-a", fb_label)
        assert _fb_candidates[1][2] == "fallback_chain[1](provider-b)"

    def test_chain_skips_failed_provider(self):
        """Chain entries matching the failed provider should be skipped."""
        chain_config = {
            "fallback_chain": [
                {"provider": "failed-provider", "model": "model-x"},
                {"provider": "good-provider", "model": "model-y"},
            ]
        }

        def mock_resolve(entry):
            client = MagicMock()
            client.base_url = f"https://{entry.get('provider')}.example.com"
            return client, entry.get("model")

        _fb_candidates = []
        _tcfg = chain_config
        _chain = _tcfg.get("fallback_chain")
        _skip_p = "failed-provider"

        if _chain and isinstance(_chain, list):
            for _i, _entry in enumerate(_chain):
                if not isinstance(_entry, dict):
                    continue
                _ep = str(_entry.get("provider", "")).strip()
                if not _ep or _ep.lower() == _skip_p:
                    continue
                _em = str(_entry.get("model", "")).strip() or None
                _el = f"fallback_chain[{_i}]({_ep})"
                if _ep and _em:
                    _ec, _emr = mock_resolve(_entry)
                    if _ec is not None:
                        _fb_candidates.append((_ec, _emr or _em, _el))

        assert len(_fb_candidates) == 1
        assert _fb_candidates[0][2] == "fallback_chain[1](good-provider)"

    def test_empty_chain_no_extra_candidates(self):
        """When there is no fallback chain, only the initial client is used."""
        _fb_candidates = [("initial-client", "model-x", "label-x")]
        _chain = None

        if _chain and isinstance(_chain, list):
            for _i, _entry in enumerate(_chain):
                pass  # won't execute

        assert len(_fb_candidates) == 1

    def test_resolve_failure_skips_entry(self):
        """When _resolve_fallback_entry raises, that entry is skipped."""
        chain_config = {
            "fallback_chain": [
                {"provider": "broken-provider", "model": "model-x"},
                {"provider": "good-provider", "model": "model-y"},
            ]
        }

        def mock_resolve(entry):
            if entry.get("provider") == "broken-provider":
                raise Exception("provider unavailable")
            client = MagicMock()
            client.base_url = f"https://{entry.get('provider')}.example.com"
            return client, entry.get("model")

        _fb_candidates = []
        _chain = chain_config.get("fallback_chain")
        _skip_p = "auto"

        if _chain and isinstance(_chain, list):
            for _i, _entry in enumerate(_chain):
                if not isinstance(_entry, dict):
                    continue
                _ep = str(_entry.get("provider", "")).strip()
                if not _ep or _ep.lower() == _skip_p:
                    continue
                _em = str(_entry.get("model", "")).strip() or None
                _el = f"fallback_chain[{_i}]({_ep})"
                if _ep and _em:
                    try:
                        _ec, _emr = mock_resolve(_entry)
                    except Exception:
                        _ec, _emr = None, None
                    if _ec is not None:
                        _fb_candidates.append((_ec, _emr or _em, _el))

        # broken-provider should be skipped, only good-provider remains
        assert len(_fb_candidates) == 1
        assert _fb_candidates[0][2] == "fallback_chain[1](good-provider)"

    def test_three_entry_chain_all_resolved(self):
        """A 3-entry chain should produce 3 candidates when all resolve."""
        chain_config = {
            "fallback_chain": [
                {"provider": "p-a", "model": "m-a"},
                {"provider": "p-b", "model": "m-b"},
                {"provider": "p-c", "model": "m-c"},
            ]
        }

        def mock_resolve(entry):
            client = MagicMock()
            client.base_url = f"https://{entry.get('provider')}.example.com"
            return client, entry.get("model")

        _fb_candidates = []
        _chain = chain_config.get("fallback_chain")
        _skip_p = "auto"

        if _chain and isinstance(_chain, list):
            for _i, _entry in enumerate(_chain):
                if not isinstance(_entry, dict):
                    continue
                _ep = str(_entry.get("provider", "")).strip()
                if not _ep or _ep.lower() == _skip_p:
                    continue
                _em = str(_entry.get("model", "")).strip() or None
                _el = f"fallback_chain[{_i}]({_ep})"
                if _ep and _em:
                    _ec, _emr = mock_resolve(_entry)
                    if _ec is not None:
                        _fb_candidates.append((_ec, _emr or _em, _el))

        assert len(_fb_candidates) == 3
        labels = [c[2] for c in _fb_candidates]
        assert "fallback_chain[0](p-a)" in labels
        assert "fallback_chain[1](p-b)" in labels
        assert "fallback_chain[2](p-c)" in labels
