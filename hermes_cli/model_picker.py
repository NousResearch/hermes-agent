"""Shared interactive model picker logic for CLI and gateway.

This module provides the core functionality for the interactive model picker
that allows users to select a provider first, then a model from that provider.

Usage:
    from hermes_cli.model_picker import (
        get_provider_list,
        get_model_list,
        load_favorites,
        save_favorites,
    )
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


HERMES_HOME = Path(os.path.expanduser("~/.hermes"))
MODELS_YAML = HERMES_HOME / "models.yaml"

PICKER_TIMEOUT_SECONDS = 300

PROVIDER_ORDER = [
    "openrouter",
    "nous",
    "openai-codex",
    "copilot",
    "copilot-acp",
    "anthropic",
    "zai",
    "kimi-coding",
    "minimax",
    "minimax-cn",
    "kilocode",
    "opencode-zen",
    "opencode-go",
    "ai-gateway",
    "alibaba",
    "huggingface",
    "deepseek",
    "custom",
]


@dataclass
class ProviderOption:
    """A provider option for the picker."""

    id: str
    label: str
    is_current: bool = False


@dataclass
class ModelOption:
    """A model option for the picker."""

    id: str
    label: str
    is_favorite: bool = False
    is_current: bool = False


def _get_models_yaml_path() -> Path:
    """Get the path to models.yaml, creating directory if needed."""
    HERMES_HOME.mkdir(parents=True, exist_ok=True)
    return MODELS_YAML


def load_favorites() -> dict[str, str]:
    """Load favorites from ~/.hermes/models.yaml.

    Returns:
        Dict mapping provider_id -> model_id for favorites.
    """
    if not MODELS_YAML.exists():
        return {}

    try:
        data = yaml.safe_load(MODELS_YAML.read_text()) or {}
        favs = data.get("favorites", {})
        return {str(k): str(v) for k, v in favs.items()}
    except Exception:
        return {}


def save_favorites(favorites: dict[str, str]) -> None:
    """Save favorites to ~/.hermes/models.yaml.

    Args:
        favorites: Dict mapping provider_id -> model_id.
    """
    path = _get_models_yaml_path()

    existing = {}
    if path.exists():
        try:
            existing = yaml.safe_load(path.read_text()) or {}
        except Exception:
            existing = {}

    existing.setdefault("favorites", {})
    existing["favorites"] = favorites

    path.write_text(yaml.dump(existing, default_flow_style=False))


def add_favorite(provider_id: str, model_id: str) -> None:
    """Add a model as favorite for a provider."""
    favs = load_favorites()
    favs[provider_id] = model_id
    save_favorites(favs)


def remove_favorite(provider_id: str) -> None:
    """Remove a provider's favorite."""
    favs = load_favorites()
    favs.pop(provider_id, None)
    save_favorites(favorites)


def get_provider_list(current_provider: str | None = None) -> list[ProviderOption]:
    """Get list of providers for the picker.

    Args:
        current_provider: The currently active provider ID.

    Returns:
        List of ProviderOption, ordered with current provider first.
    """
    from hermes_cli.models import _PROVIDER_LABELS

    providers = []
    for pid in PROVIDER_ORDER:
        if pid == "custom":
            label = "Custom endpoint"
        else:
            label = _PROVIDER_LABELS.get(pid, pid)

        is_current = pid == current_provider
        providers.append(
            ProviderOption(
                id=pid,
                label=label,
                is_current=is_current,
            )
        )

    current_first = sorted(
        providers, key=lambda p: (not p.is_current, PROVIDER_ORDER.index(p.id))
    )

    return current_first


def get_model_list(
    provider_id: str,
    favorites: dict[str, str] | None = None,
    current_model: str | None = None,
) -> list[ModelOption]:
    """Get list of models for a provider.

    Args:
        provider_id: The provider ID.
        favorites: Dict of provider -> favorite model (from load_favorites).
        current_model: The currently active model.

    Returns:
        List of ModelOption, with favorites first.
    """
    from hermes_cli.models import _PROVIDER_MODELS

    if favorites is None:
        favorites = load_favorites()

    models: list[ModelOption] = []
    favorite_model = favorites.get(provider_id, "")

    provider_models = _PROVIDER_MODELS.get(provider_id, [])

    for model_id in provider_models:
        is_fav = model_id == favorite_model
        is_current = model_id == current_model
        models.append(
            ModelOption(
                id=model_id,
                label=model_id,
                is_favorite=is_fav,
                is_current=is_current,
            )
        )

    sorted_models = sorted(
        models, key=lambda m: (not m.is_favorite, not m.is_current, m.label)
    )

    return sorted_models


def format_provider_selection(
    providers: list[ProviderOption],
    platform: str = "discord",
) -> str:
    """Format provider list for display.

    Args:
        providers: List of ProviderOption.
        platform: Platform format - "discord", "telegram", "cli".

    Returns:
        Formatted string for the platform.
    """
    lines = ["**Select provider:**"]

    for i, p in enumerate(providers, 1):
        current_tag = " *(current)*" if p.is_current else ""
        if platform == "cli":
            marker = "-> " if p.is_current else "   "
            lines.append(f"{marker}{p.label}{current_tag}")
        else:
            lines.append(f"{i}. {p.label}{current_tag}")

    return "\n".join(lines)


def format_model_selection(
    provider_id: str,
    provider_label: str,
    models: list[ModelOption],
    platform: str = "discord",
) -> str:
    """Format model list for display.

    Args:
        provider_id: Provider ID.
        provider_label: Provider display name.
        models: List of ModelOption.
        platform: Platform format.

    Returns:
        Formatted string for the platform.
    """
    lines = [f"**Models for {provider_label}:**"]

    for i, m in enumerate(models, 1):
        tags = []
        if m.is_favorite:
            tags.append("★")
        if m.is_current:
            tags.append("(current)")

        tag_str = " ".join(tags)
        if platform == "cli":
            marker = "-> " if m.is_current else "   "
            lines.append(f"{marker}{m.label} {tag_str}".strip())
        else:
            lines.append(f"{i}. {m.label} {tag_str}".strip())

    return "\n".join(lines)


def format_confirmation(
    provider_id: str,
    provider_label: str,
    model_id: str,
    previous_provider: str | None,
    previous_model: str | None,
    platform: str = "discord",
) -> str:
    """Format confirmation message.

    Args:
        provider_id: New provider ID.
        provider_label: New provider display name.
        model_id: New model ID.
        previous_provider: Previous provider ID.
        previous_model: Previous model ID.
        platform: Platform format.

    Returns:
        Formatted confirmation string.
    """
    new_str = f"{model_id} via {provider_label}"

    if previous_model and previous_provider:
        from hermes_cli.models import _PROVIDER_LABELS

        prev_label = _PROVIDER_LABELS.get(previous_provider, previous_provider)
        prev_str = f"{previous_model} via {prev_label}"
        return f"✓ **Switched to:** {new_str}\n**Previous:** {prev_str}"

    return f"✓ **Switched to:** {new_str}"


def parse_model_choice(
    text: str,
    options: list,
) -> int | None:
    """Parse user's numeric choice from text.

    Args:
        text: User's input text.
        options: List of options (for validation).

    Returns:
        Index of chosen option, or None if invalid.
    """
    text = text.strip()

    try:
        idx = int(text) - 1
        if 0 <= idx < len(options):
            return idx
    except ValueError:
        for i, opt in enumerate(options):
            if hasattr(opt, "id") and opt.id.lower() == text.lower():
                return i
            if hasattr(opt, "label") and opt.label.lower() == text.lower():
                return i

    return None
