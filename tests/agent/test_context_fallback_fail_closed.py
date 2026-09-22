"""Fail-closed context fallback (#115637).

When the context-window probe fails, the resolver must assume the cheap
(small) window, not the largest probe tier: a wrong-but-small assumption
costs one unnecessary compression, a wrong-but-large one costs the run.

- Probe failure (and blank model ids) resolve to the smallest probe tier
  at/above MINIMUM_CONTEXT_LENGTH (tiers below it would trip
  ``_enforce_minimum_context`` and refuse to start).
- A size rejection against an *assumed* window steps the working estimate
  down one probe tier, ephemerally (never persisted); an endpoint-advertised
  (explicitly quoted) limit still wins; configured/probed windows are never
  stepped down on a limit-less error.
"""
from unittest.mock import MagicMock, patch

from agent.model_metadata import (
    CONTEXT_PROBE_TIERS,
    DEFAULT_FALLBACK_CONTEXT,
    MINIMUM_CONTEXT_LENGTH,
    get_context_length_from_provider_error,
    get_model_context_length,
    get_next_probe_tier,
    step_down_assumed_context_length,
)

FAIL_CLOSED_CONTEXT = 64_000


class TestProbeFailureFailsClosed:
    def test_fallback_is_smallest_usable_tier_not_largest(self):
        assert DEFAULT_FALLBACK_CONTEXT == FAIL_CLOSED_CONTEXT
        assert DEFAULT_FALLBACK_CONTEXT != CONTEXT_PROBE_TIERS[0]
        assert DEFAULT_FALLBACK_CONTEXT == min(
            t for t in CONTEXT_PROBE_TIERS if t >= MINIMUM_CONTEXT_LENGTH
        )

    def test_probe_failure_yields_smallest_usable_tier(self):
        """Unknown model on a custom endpoint whose probes all fail."""
        with (
            patch("agent.model_metadata._resolve_endpoint_context_length", return_value=None),
            patch("agent.model_metadata._query_ollama_api_show", return_value=None),
            patch("agent.model_metadata._query_local_context_length", return_value=None),
            patch("agent.model_metadata.is_local_endpoint", return_value=False),
            patch("agent.model_metadata.fetch_model_metadata", return_value={}),
            patch("agent.models_dev.lookup_models_dev_context", return_value=None),
        ):
            assert get_model_context_length(
                "totally-unknown-model",
                base_url="https://my-gateway.example.com/v1",
            ) == FAIL_CLOSED_CONTEXT

    def test_blank_model_yields_fallback(self):
        assert get_model_context_length("") == DEFAULT_FALLBACK_CONTEXT

    def test_fallback_warning_names_assumed_window_once(self, monkeypatch, caplog):
        import agent.model_metadata as mm

        monkeypatch.setattr(mm, "_FALLBACK_WARNED", set())
        with (
            patch("agent.model_metadata._resolve_endpoint_context_length", return_value=None),
            patch("agent.model_metadata._query_ollama_api_show", return_value=None),
            patch("agent.model_metadata._query_local_context_length", return_value=None),
            patch("agent.model_metadata.is_local_endpoint", return_value=False),
            patch("agent.model_metadata.fetch_model_metadata", return_value={}),
            patch("agent.models_dev.lookup_models_dev_context", return_value=None),
        ):
            import logging

            with caplog.at_level(logging.WARNING, logger="agent.model_metadata"):
                get_model_context_length("warn-me-model", base_url="https://gw.example.com/v1")
                get_model_context_length("warn-me-model", base_url="https://gw.example.com/v1")
        warnings = [r for r in caplog.records if "Could not determine context length" in r.message]
        assert len(warnings) == 1
        assert "64,000" in warnings[0].message


class TestShrinkOnSizeRejection:
    def test_explicit_provider_limit_still_wins(self):
        assert get_context_length_from_provider_error(
            "prompt is too long: 90000 tokens > 32768 maximum", 64_000
        ) == 32_768

    def test_assumed_window_steps_down_one_tier(self):
        assert step_down_assumed_context_length(DEFAULT_FALLBACK_CONTEXT) == get_next_probe_tier(
            DEFAULT_FALLBACK_CONTEXT
        )
        assert step_down_assumed_context_length(64_000) == 32_000

    def test_configured_window_never_steps_down(self):
        assert step_down_assumed_context_length(1_000_000) is None
        assert step_down_assumed_context_length(140_032) is None

    def test_minimum_tier_never_steps_down(self):
        assert step_down_assumed_context_length(8_000) is None

    def _recovery(self, old_ctx, config_pinned=None):
        from agent.turn_overflow import _Recovery

        agent = MagicMock()
        agent.model = "unknown-model"
        agent.base_url = "https://gw.example.com/v1"
        agent.provider = "custom"
        agent.api_mode = ""
        agent._config_context_length = config_pinned
        agent.context_compressor.context_length = old_ctx
        return _Recovery(
            agent=agent,
            api_messages=[],
            system_message=None,
            effective_task_id=None,
            api_call_count=1,
            max_compression_attempts=3,
            messages=[],
            active_system_prompt=None,
            conversation_history=[],
            approx_tokens=old_ctx,
            compression_attempts=0,
            provider_overflow_recovery_pending=False,
            is_context_length_error=True,
        )

    def test_adopt_steps_down_assumed_window_without_persisting(self):
        from agent.turn_overflow import _adopt_provider_context_limit

        st = self._recovery(64_000)
        with patch("agent.model_metadata.save_provider_context_length") as mock_save:
            # Import inside: turn_overflow imports it lazily from agent.model_metadata.
            new_ctx = _adopt_provider_context_limit(
                st, "Your input exceeds the context window of this model.", 64_000
            )
        assert new_ctx == 32_000
        st.agent.context_compressor.update_model.assert_called_once()
        assert st.agent.context_compressor.update_model.call_args.kwargs["context_length"] == 32_000
        mock_save.assert_not_called()

    def test_adopt_keeps_explicitly_pinned_window(self):
        from agent.turn_overflow import _adopt_provider_context_limit

        st = self._recovery(64_000, config_pinned=64_000)
        with patch("agent.model_metadata.save_provider_context_length") as mock_save:
            new_ctx = _adopt_provider_context_limit(
                st, "Your input exceeds the context window of this model.", 64_000
            )
        assert new_ctx is None
        st.agent.context_compressor.update_model.assert_not_called()
        mock_save.assert_not_called()
