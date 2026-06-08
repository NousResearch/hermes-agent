"""Shared /image_model command logic for CLI and gateway.

Image generation backends are plugin-registered providers selected via
``image_gen.provider`` and ``image_gen.model`` in config.yaml. This module keeps
parsing, catalog lookup, formatting, and persistence in one place so the local
CLI and messaging gateway expose the same command surface.
"""

from __future__ import annotations

import logging
import shlex
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ImageModelSwitchResult:
    """Result of an image model switch attempt."""

    success: bool
    message: str
    provider: str = ""
    model: str = ""


def _ensure_discovered() -> None:
    """Discover image generation plugins best-effort."""
    try:
        from hermes_cli.plugins import _ensure_plugins_discovered

        _ensure_plugins_discovered()
    except Exception as exc:  # noqa: BLE001
        logger.debug("image_model plugin discovery skipped: %s", exc)


def _providers() -> dict[str, Any]:
    _ensure_discovered()
    try:
        from agent.image_gen_registry import list_providers

        return {p.name: p for p in list_providers()}
    except Exception as exc:  # noqa: BLE001
        logger.debug("image_model provider list failed: %s", exc)
        return {}


def _load_image_config() -> tuple[dict[str, Any], dict[str, Any]]:
    from hermes_cli.config import load_config

    cfg = load_config()
    if not isinstance(cfg, dict):
        cfg = {}
    img_cfg = cfg.setdefault("image_gen", {})
    if not isinstance(img_cfg, dict):
        img_cfg = {}
        cfg["image_gen"] = img_cfg
    return cfg, img_cfg


def _provider_models(provider: Any) -> list[dict[str, Any]]:
    try:
        raw = provider.list_models() or []
    except Exception as exc:  # noqa: BLE001
        logger.debug("image_model %s.list_models failed: %s", getattr(provider, "name", "?"), exc)
        return []
    models: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"].strip():
            models.append(item)
        elif isinstance(item, str) and item.strip():
            models.append({"id": item.strip(), "display": item.strip()})
    return models


def _default_model(provider: Any, models: list[dict[str, Any]]) -> str:
    try:
        default = provider.default_model()
        if isinstance(default, str) and default.strip():
            return default.strip()
    except Exception:  # noqa: BLE001
        pass
    if models:
        return str(models[0]["id"])
    return ""


def _current_provider_and_model(img_cfg: dict[str, Any], providers: dict[str, Any]) -> tuple[str, str]:
    provider = img_cfg.get("provider")
    provider_name = provider.strip() if isinstance(provider, str) and provider.strip() else ""
    if not provider_name and "fal" in providers:
        provider_name = "fal"
    model = ""
    if provider_name:
        nested = img_cfg.get(provider_name)
        if isinstance(nested, dict):
            raw_nested = nested.get("model")
            if isinstance(raw_nested, str) and raw_nested.strip():
                model = raw_nested.strip()
    raw_model = img_cfg.get("model")
    if not model and isinstance(raw_model, str) and raw_model.strip():
        model = raw_model.strip()
    if provider_name and not model and provider_name in providers:
        models = _provider_models(providers[provider_name])
        model = _default_model(providers[provider_name], models)
    return provider_name, model


def _persist_image_model_switch(target_provider: str, target_model: str) -> None:
    """Persist only the user-owned image_gen keys needed for the switch."""
    from hermes_cli.config import (
        ensure_hermes_home,
        format_managed_message,
        get_config_path,
        is_managed,
        read_raw_config,
    )
    from utils import atomic_yaml_write

    if is_managed():
        raise RuntimeError(format_managed_message("switch image generation model"))

    cfg = read_raw_config()
    if not isinstance(cfg, dict):
        cfg = {}
    img_section = cfg.get("image_gen")
    if not isinstance(img_section, dict):
        img_section = {}
        cfg["image_gen"] = img_section

    img_section["provider"] = target_provider
    provider_section = img_section.get(target_provider)
    if not isinstance(provider_section, dict):
        provider_section = {}

    if target_model:
        img_section["model"] = target_model
        provider_section["model"] = target_model
        img_section[target_provider] = provider_section
    else:
        img_section.pop("model", None)
        provider_section.pop("model", None)
        if provider_section:
            img_section[target_provider] = provider_section
        else:
            img_section.pop(target_provider, None)

    ensure_hermes_home()
    atomic_yaml_write(get_config_path(), cfg, sort_keys=False)


def _split_args(raw_args: str) -> tuple[str, str]:
    """Return (model_input, explicit_provider) for /image_model args."""
    import re as _re

    normalized = _re.sub(r"[\u2012\u2013\u2014\u2015](provider|global)", r"--\1", raw_args or "")
    try:
        parts = shlex.split(normalized)
    except ValueError:
        parts = normalized.split()

    explicit_provider = ""
    filtered: list[str] = []
    i = 0
    while i < len(parts):
        part = parts[i]
        if part == "--global":
            i += 1
            continue
        if part == "--provider" and i + 1 < len(parts):
            explicit_provider = parts[i + 1].strip()
            i += 2
            continue
        filtered.append(part)
        i += 1
    return " ".join(filtered).strip(), explicit_provider


def _parse_target(raw_args: str, current_provider: str, providers: dict[str, Any]) -> tuple[str, str]:
    """Resolve target provider/model from user args.

    Accepted forms:
      provider
      model
      model --provider provider
      provider/model
      provider:model
    """
    model_input, explicit_provider = _split_args(raw_args)
    provider_name = explicit_provider.strip()
    model = model_input.strip()

    if model and not provider_name:
        if ":" in model:
            left, right = model.split(":", 1)
            if left in providers and right.strip():
                provider_name = left.strip()
                model = right.strip()
        elif "/" in model:
            left, right = model.split("/", 1)
            if left in providers and right.strip():
                provider_name = left.strip()
                model = right.strip()

    if model and not provider_name and model in providers:
        provider_name = model
        model = ""

    if not provider_name:
        provider_name = current_provider

    return provider_name, model


def format_image_model_status(max_models_per_provider: int = 8) -> str:
    """Return current image generation provider/model plus usage examples."""
    providers = _providers()
    _cfg, img_cfg = _load_image_config()
    current_provider, current_model = _current_provider_and_model(img_cfg, providers)

    lines = [
        "Image generation model",
        f"Current provider: `{current_provider or 'unset'}`",
        f"Current model: `{current_model or 'default'}`",
        "",
    ]

    if providers:
        lines.append("Available providers:")
        for name, provider in sorted(providers.items()):
            tag = " ← current" if name == current_provider else ""
            lines.append(f"- `{name}`{tag}")
            models = _provider_models(provider)
            shown = [str(m["id"]) for m in models[:max_models_per_provider]]
            if shown:
                suffix = f" (+{len(models) - len(shown)} more)" if len(models) > len(shown) else ""
                lines.append(f"  models: {', '.join(f'`{m}`' for m in shown)}{suffix}")
            default = _default_model(provider, models)
            if default:
                lines.append(f"  default: `{default}`")
    else:
        lines.append("No image generation providers are registered.")

    lines.extend(
        [
            "",
            "Usage:",
            "- `/image_model <provider>`",
            "- `/image_model <model> --provider <provider>`",
            "- `/image_model <provider>/<model>`",
            "- `/image_model <provider>:<model>`",
        ]
    )
    return "\n".join(lines)


def apply_image_model_switch(raw_args: str) -> ImageModelSwitchResult:
    """Persist an image generation provider/model switch to config.yaml."""
    providers = _providers()
    cfg, img_cfg = _load_image_config()
    current_provider, _current_model = _current_provider_and_model(img_cfg, providers)

    if not (raw_args or "").strip():
        return ImageModelSwitchResult(True, format_image_model_status(), current_provider, _current_model)

    target_provider, target_model = _parse_target(raw_args, current_provider, providers)
    if not target_provider:
        return ImageModelSwitchResult(
            False,
            "No image generation provider selected. Run `/image_model` to see available providers.",
        )

    provider = providers.get(target_provider)
    if provider is None:
        available = ", ".join(f"`{name}`" for name in sorted(providers)) or "none"
        return ImageModelSwitchResult(
            False,
            f"Unknown image generation provider `{target_provider}`. Available providers: {available}.",
            target_provider,
            target_model,
        )

    models = _provider_models(provider)
    model_ids = {str(m["id"]) for m in models}
    if not target_model:
        target_model = _default_model(provider, models)

    if target_model and model_ids and target_model not in model_ids:
        sample = ", ".join(f"`{mid}`" for mid in list(sorted(model_ids))[:8])
        more = f" (+{len(model_ids) - 8} more)" if len(model_ids) > 8 else ""
        return ImageModelSwitchResult(
            False,
            f"Provider `{target_provider}` has no image model `{target_model}`. Available: {sample}{more}.",
            target_provider,
            target_model,
        )

    try:
        _persist_image_model_switch(target_provider, target_model)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to save image model switch: %s", exc)
        return ImageModelSwitchResult(
            False,
            f"Failed to save image model switch: {exc}",
            target_provider,
            target_model,
        )

    model_display = target_model or "provider default"
    return ImageModelSwitchResult(
        True,
        f"✓ Image model switched: `{target_provider}` / `{model_display}`\nSaved to config.yaml.",
        target_provider,
        target_model,
    )
