"""Shared interactive model picker logic for CLI and gateway.

This module provides the core functionality for the interactive model picker
that allows users to select a provider first, then a model from that provider.

Usage:
    from hermes_cli.model_picker import (
        get_provider_list,
        get_model_list,
    )
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


HERMES_HOME = Path(os.path.expanduser("~/.hermes"))

PROVIDER_ORDER = [
    "ollama-cloud",
    "ollama-local",
    "openai-codex",
    "openrouter",
    "anthropic",
    "google",
    "xai",
    "nous",
    "copilot",
    "copilot-acp",
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
    current_model: str | None = None,
) -> list[ModelOption]:
    """Get list of models for a provider.

    Args:
        provider_id: The provider ID.
        current_model: The currently active model.

    Returns:
        List of ModelOption.
    """
    from hermes_cli.models import _PROVIDER_MODELS

    models: list[ModelOption] = []
    provider_models = _PROVIDER_MODELS.get(provider_id, [])

    for model_id in provider_models:
        is_current = model_id == current_model
        models.append(
            ModelOption(
                id=model_id,
                label=model_id,
                is_current=is_current,
            )
        )

    return models


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
        if m.is_current:
            tags.append("(current)")

        tag_str = " ".join(tags)
        if platform == "cli":
            marker = "-> " if m.is_current else "   "
            lines.append(f"{marker}{m.label} {tag_str}".strip())
        else:
            lines.append(f"{i}. {m.label} {tag_str}".strip())

    return "\n".join(lines)
