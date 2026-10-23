"""Native handler wiring identities shared by platform adapters."""

from typing import Any


def platform_handler_key(manager: Any, registration: tuple) -> tuple:
    """Keep legacy deduplication unless a registration explicitly allows rewiring."""
    factory, plugin_name = registration
    key = (plugin_name, getattr(factory, "__qualname__", None) or repr(factory))
    generation_for = getattr(manager, "get_platform_handler_factory_generation", None)
    generation = generation_for(registration) if callable(generation_for) else None
    return key + (generation,) if generation is not None else key
