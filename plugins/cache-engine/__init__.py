"""hermes-cache-engine — Context Economy Engine for Hermes.

Safe by default: ``mode: report`` measures what could be saved and changes nothing.
Flip to ``mode: enforce`` once the numbers look right.
"""

__version__ = "0.2.0"

from .config import load_settings
from .engine import CacheEngine


def register(ctx) -> None:
    engine = CacheEngine(load_settings())
    ctx.register_middleware("llm_request", engine.on_llm_request)
