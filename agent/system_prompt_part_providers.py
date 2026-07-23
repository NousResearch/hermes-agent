"""Generic system-prompt part provider registry.

Plugins and edge integrations can register additive prompt parts without adding
product-specific behavior to ``agent.system_prompt``. Providers are process-local
and deterministic for a prompt build; callers cache the final assembled prompt
for the conversation lifetime.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterable, Mapping
from typing import Any

logger = logging.getLogger(__name__)

PromptPartProvider = Callable[..., Mapping[str, str | Iterable[str] | None] | None]
_VALID_TIERS = frozenset({"stable", "context", "volatile"})
_lock = threading.RLock()
_providers: dict[str, PromptPartProvider] = {}


def register_system_prompt_part_provider(name: str, provider: PromptPartProvider) -> None:
    """Register an additive prompt-part provider by stable name."""
    provider_name = str(name or "").strip()
    if not provider_name:
        raise ValueError("provider name must be non-empty")
    if not callable(provider):
        raise TypeError("provider must be callable")
    with _lock:
        _providers[provider_name] = provider
    logger.info("Registered system prompt part provider: %s", provider_name)


def list_system_prompt_part_providers() -> list[str]:
    """Return registered provider names in registration order."""
    with _lock:
        return list(_providers.keys())


def collect_system_prompt_parts(
    agent: Any, *, system_message: str | None = None
) -> dict[str, list[str]]:
    """Collect additive prompt parts from all registered providers."""
    collected: dict[str, list[str]] = {"stable": [], "context": [], "volatile": []}
    with _lock:
        providers = list(_providers.items())
    for name, provider in providers:
        try:
            provided = provider(agent, system_message=system_message)
        except TypeError:
            try:
                provided = provider(agent)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.debug("System prompt part provider %s failed: %s", name, exc)
                continue
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug("System prompt part provider %s failed: %s", name, exc)
            continue
        if not isinstance(provided, Mapping):
            continue
        for tier, value in provided.items():
            if tier not in _VALID_TIERS:
                logger.debug(
                    "System prompt part provider %s returned unknown tier %r",
                    name,
                    tier,
                )
                continue
            collected[tier].extend(_coerce_parts(value))
    return collected


def clear_system_prompt_part_providers_for_tests() -> None:
    """Clear process-local providers for isolated tests."""
    with _lock:
        _providers.clear()


def _coerce_parts(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    try:
        iterator = iter(value)
    except TypeError:
        return []
    parts: list[str] = []
    for item in iterator:
        if isinstance(item, str) and item.strip():
            parts.append(item.strip())
    return parts
