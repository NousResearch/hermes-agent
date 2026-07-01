"""Tests for model switch base_url persistence.

Covers:
- Non-custom providers must NOT persist base_url to config.yaml
  (the credential pool provides the correct URL; persisting a stripped
  /v1 for anthropic_messages causes 404 on subsequent chat_completions).
- Custom providers MUST persist base_url (user-specified endpoint).
- Regression: #56158 (opencode-go stale base_url → 404).

The fix lives in ``gateway/slash_commands.py`` in both the picker-callback
and text-command persist paths.  We test the logic indirectly via the
``clear_model_endpoint_credentials`` contract and a focused stub that
replicates the persist block.
"""

from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from hermes_cli.config import clear_model_endpoint_credentials


# ---------------------------------------------------------------------------
# Unit: clear_model_endpoint_credentials
# ---------------------------------------------------------------------------


class TestClearModelEndpointCredentials:
    def test_clear_base_url_removes_it(self):
        cfg = {"default": "m", "provider": "p", "base_url": "https://example.com/v1"}
        clear_model_endpoint_credentials(cfg, clear_base_url=True)
        assert "base_url" not in cfg

    def test_clear_base_url_false_keeps_it(self):
        cfg = {"default": "m", "provider": "p", "base_url": "https://example.com/v1"}
        clear_model_endpoint_credentials(cfg, clear_base_url=False)
        assert cfg["base_url"] == "https://example.com/v1"

    def test_clear_api_key_and_mode(self):
        cfg = {"default": "m", "provider": "p", "api_key": "sk-xxx", "api_mode": "anthropic_messages"}
        clear_model_endpoint_credentials(cfg, clear_base_url=True)
        assert "api_key" not in cfg
        assert "api_mode" not in cfg
        assert "base_url" not in cfg


# ---------------------------------------------------------------------------
# Integration: replicate the persist block logic
# ---------------------------------------------------------------------------


def _simulate_persist_block(result_target_provider: str, result_base_url: str):
    """Replicate the /model persist logic and return the final config dict.

    Mirrors the two persist blocks in ``gateway/slash_commands.py``:
    - Lines 1359-1364 (picker callback)
    - Lines 1596-1601 (text command)
    """
    model_cfg = {"default": "old-model", "provider": "old-provider"}
    model_cfg["default"] = "new-model"
    model_cfg["provider"] = result_target_provider
    # NEW logic: only persist base_url for custom providers
    if str(result_target_provider or "").strip().lower() == "custom" and result_base_url:
        model_cfg["base_url"] = result_base_url
    else:
        clear_model_endpoint_credentials(model_cfg, clear_base_url=True)
    return model_cfg


class TestPersistBlockBaseUrl:
    def test_non_custom_provider_does_not_persist_base_url(self):
        """#56158: opencode-go should not persist stripped /v1 base_url."""
        cfg = _simulate_persist_block("opencode-go", "https://opencode.ai/zen/go")
        assert "base_url" not in cfg, (
            "Non-custom provider must not persist base_url — the pool provides "
            "the correct URL. Persisting a stripped /v1 causes HTTP 404 when "
            "switching to a chat_completions model."
        )

    def test_custom_provider_persists_base_url(self):
        """Custom endpoints must persist the user-specified base_url."""
        cfg = _simulate_persist_block("custom", "http://localhost:11434/v1")
        assert cfg["base_url"] == "http://localhost:11434/v1"

    def test_custom_provider_empty_base_url_not_persisted(self):
        """Custom provider with no base_url should not set it."""
        cfg = _simulate_persist_block("custom", "")
        assert "base_url" not in cfg

    def test_anthropic_provider_does_not_persist_base_url(self):
        """Built-in anthropic should not persist base_url."""
        cfg = _simulate_persist_block("anthropic", "https://api.anthropic.com")
        assert "base_url" not in cfg

    def test_openrouter_provider_does_not_persist_base_url(self):
        """Built-in openrouter should not persist base_url."""
        cfg = _simulate_persist_block("openrouter", "https://openrouter.ai/api/v1")
        assert "base_url" not in cfg
