"""Bundled plugin adapter for the KarinAI managed image gateway provider."""

from __future__ import annotations

from karinai.runtime.image_gateway_provider import KarinAIImageGatewayProvider


def register(ctx) -> None:
    """Plugin entry point — wire the managed image gateway into the registry."""
    ctx.register_image_gen_provider(KarinAIImageGatewayProvider())
