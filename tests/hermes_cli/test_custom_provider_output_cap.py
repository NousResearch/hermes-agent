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

    def test_invalid_cap_values_ignored(self, caplog):
        import logging

        for bad in (0, -5, "16384", 16384.0):
            with caplog.at_level(logging.WARNING, logger="hermes_cli.config"):
                normalized = _normalize_custom_provider_entry(
                    self._entry(max_output_tokens=bad), provider_key="aperture"
                )
            assert "max_output_tokens" not in normalized, repr(bad)
            # Present-but-invalid must be loud, not silently dropped.
            assert "must be a positive integer" in caplog.text, repr(bad)


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
    """Exercise the REAL gateway resolver (gateway/run.py
    ``_resolve_runtime_agent_kwargs``): env > model.max_tokens >
    per-provider cap. The global key must always win over the per-provider
    fallback, and a malformed global value must fall through to the cap.
    """

    def _resolve(self, monkeypatch, *, env=None, model_cfg=None, provider_mot=None):
        # _resolve_runtime_agent_kwargs imports these names from
        # hermes_cli.runtime_provider inside the function body, so patching
        # the source module reaches every call.
        import os as _os

        from hermes_cli import runtime_provider as rp
        from gateway import run as gateway_run

        runtime = {"max_output_tokens": provider_mot} if provider_mot is not None else {}
        monkeypatch.setattr(rp, "resolve_runtime_provider", lambda: runtime)
        monkeypatch.setattr(rp, "_get_model_config", lambda: model_cfg or {})
        # AuthError/is_rate_limited_auth_error imports stay real; they are
        # only consulted on the exception path which these tests never hit.

        if env is None:
            monkeypatch.delenv("HERMES_MAX_TOKENS", raising=False)
        else:
            monkeypatch.setenv("HERMES_MAX_TOKENS", str(env))
            assert "HERMES_MAX_TOKENS" in _os.environ

        return gateway_run._resolve_runtime_agent_kwargs()["max_tokens"]

    def test_provider_cap_applies_when_no_global_set(self, monkeypatch):
        assert self._resolve(monkeypatch, provider_mot=16384) == 16384

    def test_global_model_max_tokens_wins(self, monkeypatch):
        result = self._resolve(
            monkeypatch, model_cfg={"max_tokens": 32768}, provider_mot=16384
        )
        assert result == 32768

    def test_env_wins_over_everything(self, monkeypatch):
        result = self._resolve(
            monkeypatch, env="4096", model_cfg={"max_tokens": 32768}, provider_mot=16384
        )
        assert result == 4096

    def test_malformed_global_falls_through_to_cap(self, monkeypatch):
        """model.max_tokens = -1 previously suppressed the cap entirely."""
        result = self._resolve(
            monkeypatch, model_cfg={"max_tokens": -1}, provider_mot=16384
        )
        assert result == 16384

    def test_malformed_env_falls_through_to_cap(self, monkeypatch):
        result = self._resolve(monkeypatch, env="-5", provider_mot=8192)
        assert result == 8192

    def test_nothing_configured_yields_none(self, monkeypatch):
        assert self._resolve(monkeypatch) is None
