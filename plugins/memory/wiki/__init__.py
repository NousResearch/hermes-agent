"""Hermes plugin wrapper for the wiki memory provider."""

from llmwiki_hermes.provider.plugin import WikiMemoryProvider


def register(ctx) -> None:
    """Register the wiki memory provider with Hermes."""

    ctx.register_memory_provider(WikiMemoryProvider())


__all__ = ["WikiMemoryProvider", "register"]
