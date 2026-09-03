"""Tests for the catalog-based reasoning gate in run_agent._supports_reasoning_extra_body.

Covers the new path that consults endpoint_model_supports_reasoning() before
falling through to the _is_openrouter_url() wall, which previously returned
False for all custom/localhost endpoints unconditionally.
"""

from unittest.mock import patch

from run_agent import AIAgent


def _make_custom_agent(base_url="http://127.0.0.1:8977/v1", model="sonnet"):
    """Minimal AIAgent stub pointing at a custom (non-OpenRouter) endpoint."""
    agent = object.__new__(AIAgent)
    agent.provider = "custom"
    agent.base_url = base_url
    agent._base_url_lower = base_url.lower()
    agent.model = model
    agent.api_key = ""
    return agent


class TestCustomProviderReasoningFromCatalog:

    def test_returns_true_when_catalog_says_reasoning_supported(self):
        """A localhost shim that advertises reasoning in /v1/models must enable it."""
        agent = _make_custom_agent()
        with patch(
            "agent.model_metadata.endpoint_model_supports_reasoning",
            return_value=True,
        ):
            result = agent._supports_reasoning_extra_body()
        assert result is True

    def test_falls_through_to_false_when_catalog_says_no_reasoning(self):
        """Model listed but reasoning absent -> catalog returns False -> falls through
        _is_openrouter_url() check -> returns False for non-OpenRouter URL."""
        agent = _make_custom_agent()
        with patch(
            "agent.model_metadata.endpoint_model_supports_reasoning",
            return_value=False,
        ):
            result = agent._supports_reasoning_extra_body()
        assert result is False

    def test_falls_through_to_false_when_catalog_unreachable(self):
        """Catalog unavailable (None) -> falls through -> non-OpenRouter URL -> False."""
        agent = _make_custom_agent()
        with patch(
            "agent.model_metadata.endpoint_model_supports_reasoning",
            return_value=None,
        ):
            result = agent._supports_reasoning_extra_body()
        assert result is False

    def test_exception_in_catalog_lookup_is_swallowed(self):
        """Any exception from catalog lookup must not propagate -- fall through to False."""
        agent = _make_custom_agent()
        with patch(
            "agent.model_metadata.endpoint_model_supports_reasoning",
            side_effect=RuntimeError("network failure"),
        ):
            result = agent._supports_reasoning_extra_body()
        assert result is False

    def test_openrouter_url_skips_the_catalog_entirely(self):
        """OpenRouter must never take the catalog path: its per-model gating below
        is deliberate, and a catalog entry advertising reasoning for a model that
        gating excludes would silently flip behavior."""
        agent = object.__new__(AIAgent)
        agent.provider = "openrouter"
        agent.base_url = "https://openrouter.ai/api/v1"
        agent._base_url_lower = agent.base_url.lower()
        agent.model = "anthropic/claude-sonnet-4-5"
        agent.api_key = ""

        # Even a True catalog verdict must not be consulted for OpenRouter.
        with patch(
            "agent.model_metadata.endpoint_model_supports_reasoning",
            return_value=True,
        ) as mock_catalog:
            agent._supports_reasoning_extra_body()

        mock_catalog.assert_not_called()

    def test_catalog_failure_is_logged_at_debug(self):
        """The catalog probe's except branch must log rather than silently pass,
        so a real bug (e.g. AttributeError) stays diagnosable."""
        agent = _make_custom_agent()
        with patch(
            "agent.model_metadata.endpoint_model_supports_reasoning",
            side_effect=RuntimeError("boom"),
        ):
            with patch("run_agent.logger") as mock_logger:
                result = agent._supports_reasoning_extra_body()

        assert result is False
        assert mock_logger.debug.called
        logged = " ".join(str(a) for a in mock_logger.debug.call_args[0])
        assert "atalog reasoning probe failed" in logged

    def test_nousresearch_url_unaffected_by_catalog(self):
        """nousresearch.com returns True before the catalog lookup runs."""
        agent = object.__new__(AIAgent)
        agent.provider = "custom"
        agent.base_url = "https://inference.nousresearch.com/v1"
        agent._base_url_lower = agent.base_url.lower()
        agent.model = "hermes-3"
        agent.api_key = ""

        # Even if catalog were reachable it must never be called for this host,
        # because the nousresearch.com short-circuit fires first.
        with patch(
            "agent.model_metadata.endpoint_model_supports_reasoning"
        ) as mock_catalog:
            result = agent._supports_reasoning_extra_body()

        assert result is True
        mock_catalog.assert_not_called()

    def test_localhost_without_catalog_returns_false(self):
        """Localhost endpoint with unreachable catalog still returns False,
        not an exception -- guards the regression where AttributeError on
        self._api_key was leaking through the bare except."""
        agent = _make_custom_agent(base_url="http://localhost:8080/v1")
        with patch(
            "agent.model_metadata.endpoint_model_supports_reasoning",
            return_value=None,
        ):
            result = agent._supports_reasoning_extra_body()
        assert result is False
