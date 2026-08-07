"""DeepSeek native web-search capability plugin."""

from plugins.web.deepseek.provider import DeepSeekWebSearchProvider


def register(ctx) -> None:
    ctx.register_web_search_provider(DeepSeekWebSearchProvider())
