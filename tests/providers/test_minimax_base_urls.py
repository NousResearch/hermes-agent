"""Keep MiniMax OpenAI and Anthropic endpoints distinct."""

import pytest

from hermes_cli.auth import PROVIDER_REGISTRY
from hermes_cli.providers import HERMES_OVERLAYS


@pytest.mark.parametrize(
    ("provider_id", "openai_base", "anthropic_base"),
    [
        ("minimax", "https://api.minimax.io/v1", "https://api.minimax.io/anthropic"),
        ("minimax-cn", "https://api.minimaxi.com/v1", "https://api.minimaxi.com/anthropic"),
    ],
)
def test_minimax_protocol_base_urls(provider_id, openai_base, anthropic_base):
    assert PROVIDER_REGISTRY[provider_id].inference_base_url == openai_base

    overlay = HERMES_OVERLAYS[provider_id]
    assert overlay.transport == "anthropic_messages"
    assert overlay.base_url_override == anthropic_base
