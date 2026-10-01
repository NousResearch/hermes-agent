"""Gateway-owned /model display helpers."""

from __future__ import annotations

from typing import Any

from application_model_switch_persistence import route_changed

_OPAQUE_MODEL_PREFIXES = ("ri.language-model-service..language-model.",)


def format_model_for_display(model_name: str) -> str:
    for prefix in _OPAQUE_MODEL_PREFIXES:
        if model_name and model_name.startswith(prefix):
            return model_name[len(prefix):] or model_name
    return model_name


def resolve_display_context_length(
    model: str,
    provider: str,
    *,
    base_url: str = "",
    api_key: str = "",
    custom_providers: list | None = None,
    config_context_length: int | None = None,
    configured_model: str | None = None,
    configured_provider: str | None = None,
    configured_base_url: str | None = None,
    **_: Any,
) -> int:
    from models.metadata.context import get_model_context_length

    pin = config_context_length
    probe = type(
        "_Route",
        (),
        {
            "new_model": model,
            "target_provider": provider,
            "base_url": base_url,
        },
    )()
    configured = {
        "default": configured_model,
        "provider": configured_provider,
        "base_url": configured_base_url,
    }
    if pin is not None and route_changed(configured, probe):
        pin = None
    return get_model_context_length(
        model,
        base_url=base_url,
        api_key=api_key,
        config_context_length=pin,
        provider=provider,
        custom_providers=custom_providers,
    )


async def resolve_display_context_length_async(*args, **kwargs) -> int:
    import asyncio

    return await asyncio.to_thread(resolve_display_context_length, *args, **kwargs)


__all__ = [
    "format_model_for_display",
    "resolve_display_context_length",
    "resolve_display_context_length_async",
]
