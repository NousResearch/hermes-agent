"""Regression: public model-options surface keeps curated models for an
exhausted credential pool while runtime resolution stays non-usable."""
from unittest.mock import patch

from hermes_cli.inventory import build_model_options_payload, load_picker_context
from hermes_cli.model_switch import _credential_pool_is_usable


class _ExhaustedPool:
    def has_credentials(self):
        return True
    def has_available(self):
        return False


def test_model_options_keeps_curated_models_for_exhausted_pool():
    with patch("agent.credential_pool.load_pool", return_value=_ExhaustedPool()):
        assert _credential_pool_is_usable("openai-codex") is False
        payload = build_model_options_payload(load_picker_context(), include_unconfigured=True)
        rows = payload.get("providers", [])
        codex = next((r for r in rows if r.get("slug") == "openai-codex"), None)
        assert codex is not None, "openai-codex must remain visible in the picker"
        models = codex.get("models") or []
        assert len(models) > 0, "curated Codex model list must remain visible"
        for shared in ("gpt-5.4-mini", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra"):
            assert shared in models, f"shared model {shared} missing from curated list"
