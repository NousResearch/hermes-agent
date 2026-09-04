"""Deprecated one-release compatibility alias for the Tameru engine."""
from plugins.context_engine.tameru import ExtractiveContextEngine


def register(ctx) -> None:
    ctx.register_context_engine(ExtractiveContextEngine())
