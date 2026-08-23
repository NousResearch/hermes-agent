"""Per-provider output cap (``max_output_tokens`` / ``max_tokens``) on
``custom_providers`` entries.

Contract: the cap survives ``_normalize_custom_provider_entry`` and reaches
the consumers that map it onto ``AIAgent.max_tokens`` — the gateway runtime
resolver (gateway/run.py) and the CLI turn path. Regression context: the
normalizer's unknown-key filter stripped both spellings before
``_lift_max_output_tokens`` ever ran, so a relay with a smaller completion
window than its profile default (e.g. a 128k-context model inheriting a
65536 output reservation) 400'd on every large-prompt request.
"""

from hermes_cli.config import _normalize_custom_provider_entry


class TestNormalizerPreservesOutputCap:
    """The normalizer must not strip the per-provider output cap."""

    def _entry(self, **extra):
        entry = {
            "name": "Aperture",
            "base_url": "http://ai.example.invalid/v1",
            "model": "lumo-max",
        }
        entry.update(extra)
        return entry

    def test_max_output_tokens_survives_normalization(self):
        normalized = _normalize_custom_provider_entry(
            self._entry(max_output_tokens=16384), provider_key="aperture"
        )
        assert normalized["max_output_tokens"] == 16384

    def test_max_tokens_alias_also_accepted(self):
        normalized = _normalize_custom_provider_entry(
            self._entry(max_tokens=16384), provider_key="aperture"
        )
        assert normalized["max_output_tokens"] == 16384

    def test_no_unknown_key_warning_for_cap(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="hermes_cli.config"):
            _normalize_custom_provider_entry(
                self._entry(max_output_tokens=16384), provider_key="aperture"
            )
        assert "unknown config keys" not in caplog.text

    def test_absent_cap_leaves_key_out(self):
        normalized = _normalize_custom_provider_entry(
            self._entry(), provider_key="aperture"
        )
        assert "max_output_tokens" not in normalized

    def test_invalid_cap_values_ignored(self):
        for bad in (0, -5, "16384", 16384.0):
            normalized = _normalize_custom_provider_entry(
                self._entry(max_output_tokens=bad), provider_key="aperture"
            )
            assert "max_output_tokens" not in normalized, repr(bad)


class TestLiftMaxOutputTokens:
    """_lift_max_output_tokens prefers max_output_tokens over max_tokens."""

    def test_prefers_explicit_field_over_alias(self):
        from hermes_cli.runtime_provider import _lift_max_output_tokens

        result = {}
        _lift_max_output_tokens(
            {"max_output_tokens": 8192, "max_tokens": 4096}, result
        )
        assert result["max_output_tokens"] == 8192

    def test_alias_only_entry_lifted(self):
        from hermes_cli.runtime_provider import _lift_max_output_tokens

        result = {}
        _lift_max_output_tokens({"max_tokens": 4096}, result)
        assert result["max_output_tokens"] == 4096

    def test_negative_and_zero_rejected(self):
        from hermes_cli.runtime_provider import _lift_max_output_tokens

        for bad in (0, -1):
            result = {}
            _lift_max_output_tokens({"max_output_tokens": bad}, result)
            assert result == {}


class TestGatewayResolutionOrder:
    """Mirror of gateway/run.py: env > model.max_tokens > per-provider cap.

    The global key must always win over the per-provider fallback.
    """

    def _resolve(self, *, env=None, model_mt=None, provider_mot=None):
        max_tokens = None
        if env is not None:
            try:
                max_tokens = int(env)
            except (ValueError, TypeError):
                max_tokens = None
        elif isinstance(model_mt, int):
            max_tokens = model_mt
        if max_tokens is None:
            if isinstance(provider_mot, int) and provider_mot > 0:
                max_tokens = provider_mot
        return max_tokens

    def test_provider_cap_applies_when_no_global_set(self):
        assert self._resolve(provider_mot=16384) == 16384

    def test_global_model_max_tokens_wins(self):
        assert (
            self._resolve(model_mt=32768, provider_mot=16384) == 32768
        )

    def test_env_wins_over_everything(self):
        assert self._resolve(env="4096", model_mt=32768, provider_mot=16384) == 4096
