"""HSK-083 regression: gateway fallback logging must prefer the configured provider name.

Before this fix, ``gateway.run._try_resolve_fallback_provider`` logged only
``runtime.get("provider")`` (the resolved runtime *category*), losing the original
``entry.get("provider") or runtime.get("provider")`` semantic. An Ollama fallback entry
resolves through the OpenAI-compatible path, so its runtime category can read back as
``"openrouter"`` even though the operator configured ``ollama`` — the log must say
``ollama``, matching config, not the internal runtime category.
"""

from __future__ import annotations

import logging
from unittest.mock import patch

from hermes_cli.fallback_config import FallbackResolution


def test_gateway_fallback_log_prefers_configured_provider(monkeypatch, caplog):
    resolution = FallbackResolution(
        runtime={"provider": "openrouter", "api_key": "fb-key", "base_url": "https://openrouter.ai/api/v1"},
        model="llama-4-maverick",
        configured_provider="ollama",
    )

    with (
        patch("gateway.run._load_gateway_runtime_config", return_value={}),
        patch("hermes_cli.fallback_config.resolve_first_available_fallback", return_value=resolution),
    ):
        import gateway.run as gw

        with caplog.at_level(logging.INFO, logger=gw.logger.name):
            result = gw._try_resolve_fallback_provider()

    assert result is not None
    assert result["provider"] == "openrouter"  # runtime kwargs unchanged
    assert result["model"] == "llama-4-maverick"
    assert "ollama" in caplog.text
    assert "openrouter" not in caplog.text
