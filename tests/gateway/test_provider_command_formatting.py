from unittest.mock import patch

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_event(text="/provider", platform=Platform.TELEGRAM,
                user_id="12345", chat_id="67890"):
    source = SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        user_name="testuser",
    )
    return MessageEvent(text=text, source=source)


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    return runner


class TestGatewayProviderFormatting:
    @pytest.mark.asyncio
    async def test_provider_output_uses_aligned_block(self, tmp_path):
        runner = _make_runner()
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            "model:\n"
            "  provider: openrouter\n"
        )

        providers = [
            {"id": "openrouter", "label": "OpenRouter", "authenticated": True, "aliases": ["or"]},
            {"id": "anthropic", "label": "Anthropic", "authenticated": False, "aliases": []},
        ]

        with patch("gateway.run._hermes_home", hermes_home), \
             patch("hermes_cli.models.list_available_providers", return_value=providers):
            result = await runner._handle_provider_command(_make_event())

        assert "🔌 **Current provider:** OpenRouter (`openrouter`)" in result
        assert "**Available providers**" in result
        assert "```text" in result
        assert "✅ openrouter" in result
        assert "OpenRouter (or) ← active" in result
        assert "❌ anthropic" in result
        assert "Switch: `/model provider:model-name`" in result
