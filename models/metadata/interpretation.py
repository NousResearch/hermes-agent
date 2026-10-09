"""Pure interpretation of raw model catalogue metadata.

This module owns the meaning of capability and metadata fields.  It accepts
raw catalogue-shaped mappings and sparse canonical overrides; catalogue
membership, acquisition, route selection, and configuration loading stay in
callers.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Mapping, Optional

from models.identity import ModelRef
from models.metadata.types import (
    ModelInfo,
    ModelMetadata,
    ModelMetadataPatch,
    ProviderInfo,
    ReasoningMetadata,
)

logger = logging.getLogger(__name__)


# Safe defaults for an explicitly selected model absent from the catalogue.
# Capability fields not present here remain unknown to consumers.
UNKNOWN_MODEL_BASE: dict[str, Any] = {"limit": {"context": 200000}, "tool_call": True}

_DEEPSEEK_FLASH_VISION: dict[str, Any] = {
    "limit": {"context": 1_000_000, "output": 384_000},
    "modalities": {"input": ["text", "image"], "output": ["text"]},
    "tool_call": True,
    "reasoning": True,
    "family": "deepseek-flash",
}

_BUILTIN_MODEL_METADATA: dict[tuple[str, str], dict[str, Any]] = {
    ("openai", "gpt-6-astra"): {
        "limit": {"context": 1_050_000, "output": 128_000},
        "modalities": {"input": ["text", "image"], "output": ["text"]},
        "tool_call": True,
        "reasoning": True,
        "family": "gpt-6",
    },
    # Native DeepSeek V4.1-Flash is multimodal.  Keep this vendor fact here,
    # while the caller remains responsible for deciding which provider route
    # owns the raw model identity.
    ("deepseek", "deepseek-flash"): _DEEPSEEK_FLASH_VISION,
    ("deepseek", "deepseek-v4-flash"): _DEEPSEEK_FLASH_VISION,
    ("deepseek", "deepseek-v4.1-flash"): _DEEPSEEK_FLASH_VISION,
    ("deepseek", "deepseek-v4-flash-vision-exp"): _DEEPSEEK_FLASH_VISION,
}


_OVERRIDE_WARNED_KEYS: set[tuple[str, str]] = set()


def dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def extract_limit(entry: Any, key: str) -> Optional[int]:
    """Return a positive integer from a raw catalogue ``limit`` mapping."""
    value = dict_or_empty(dict_or_empty(entry).get("limit")).get(key)
    return int(value) if isinstance(value, (int, float)) and value > 0 else None


def extract_context(entry: Any) -> Optional[int]:
    """Return a usable catalogue context window, or ``None``."""
    return extract_limit(entry, "context")


def override_int(override: Mapping[str, Any], key: str) -> Optional[int]:
    """Coerce a canonical override field to a positive integer."""
    raw = override.get(key)
    if raw is None:
        return None
    with contextlib.suppress(TypeError, ValueError):
        value = int(raw)
        if value > 0:
            return value
    warn_key = (key, repr(raw))
    if warn_key not in _OVERRIDE_WARNED_KEYS:
        _OVERRIDE_WARNED_KEYS.add(warn_key)
        logger.warning(
            "model_overrides: ignoring invalid %s value %r "
            "(expected a positive integer)",
            key,
            raw,
        )
    return None


def override_to_catalog_shape(
    override: Mapping[str, Any],
) -> tuple[dict[str, Any], Optional[bool]]:
    """Translate canonical override keys to raw catalogue fields.

    Vision is returned separately because its canonical boolean maps to the
    ``modalities.input`` list.  Output limits intentionally remain catalogue
    supplied: the established ``model_overrides`` contract only overrides the
    context window and capability/family fields.
    """
    patch: dict[str, Any] = {}
    context = override_int(override, "context_window")
    if context is not None:
        patch["limit"] = {"context": context}
    for override_key, catalog_key in (
        ("supports_tools", "tool_call"),
        ("supports_reasoning", "reasoning"),
    ):
        if override_key in override:
            patch[catalog_key] = bool(override[override_key])
    vision: Optional[bool] = None
    if "supports_vision" in override:
        vision = bool(override["supports_vision"])
        patch["attachment"] = vision
    if "model_family" in override:
        patch["family"] = str(override["model_family"] or "")
    return patch, vision


def merge_catalog_entry_with_override(
    raw: Mapping[str, Any], override: Mapping[str, Any]
) -> dict[str, Any]:
    """Merge a canonical override into a raw catalogue-shaped entry."""
    shaped, vision_override = override_to_catalog_shape(override)
    raw_dict = dict(raw)
    merged = dict(raw_dict)
    limit_patch = shaped.pop("limit", None)
    if limit_patch:
        merged["limit"] = {
            **dict_or_empty(raw_dict.get("limit")),
            **limit_patch,
        }
    if vision_override is not None:
        modalities = dict(dict_or_empty(raw_dict.get("modalities")))
        input_modalities = modalities.get("input")
        input_modalities = (
            list(input_modalities) if isinstance(input_modalities, list) else []
        )
        if vision_override and "image" not in input_modalities:
            input_modalities.append("image")
        elif not vision_override and "image" in input_modalities:
            input_modalities.remove("image")
        modalities["input"] = input_modalities
        merged["modalities"] = modalities
    merged.update(shaped)
    return merged


def builtin_model_metadata(provider_id: str, model_id: str) -> Optional[dict[str, Any]]:
    """Return built-in vendor metadata for a provider/model pair."""
    return _BUILTIN_MODEL_METADATA.get(
        (str(provider_id or "").strip(), str(model_id or "").strip().lower())
    )


def vision_marker_metadata(*, is_opencode_family: bool, model_id: str) -> Optional[dict[str, Any]]:
    """Return the narrow relay fallback for an explicitly vision-marked model."""
    if not is_opencode_family or "-vision" not in str(model_id or "").strip().lower():
        return None
    return {
        **UNKNOWN_MODEL_BASE,
        "modalities": {"input": ["text", "image"], "output": ["text"]},
    }


def entry_supports_vision(entry: Mapping[str, Any]) -> bool:
    """Prefer valid input modalities over a stale attachment flag."""
    input_modalities = dict_or_empty(entry.get("modalities", {})).get("input")
    return (
        "image" in input_modalities
        if isinstance(input_modalities, list)
        else bool(entry.get("attachment", False))
    )


def model_metadata_patch_from_entry(
    raw: Mapping[str, Any], *, unknown_model: bool = False
) -> ModelMetadataPatch:
    """Interpret one effective raw catalogue entry as canonical metadata facts.

    Missing capability fields remain unknown for an unknown model.  A known
    catalogue entry with an explicit false field, stale attachment flag, or
    malformed modalities value keeps the established false result.
    """
    modalities = dict_or_empty(raw.get("modalities"))
    input_values = modalities.get("input")
    output_values = modalities.get("output")
    input_modalities = tuple(input_values) if isinstance(input_values, list) else None
    output_modalities = tuple(output_values) if isinstance(output_values, list) else None

    if input_modalities is not None:
        supports_vision: Optional[bool] = "image" in input_modalities
    elif "attachment" in raw or "modalities" in raw:
        supports_vision = entry_supports_vision(raw)
    else:
        supports_vision = None if unknown_model else False

    def capability(key: str) -> Optional[bool]:
        if key in raw:
            return bool(raw[key])
        return None if unknown_model else False

    reasoning = ReasoningMetadata(
        supported=capability("reasoning"),
        supported_efforts=(
            tuple(raw["reasoning_efforts"])
            if isinstance(raw.get("reasoning_efforts"), list)
            else None
        ),
        mandatory=(bool(raw["reasoning_mandatory"]) if "reasoning_mandatory" in raw else None),
    )
    return ModelMetadataPatch(
        context_window=extract_limit(raw, "context") or 200000,
        max_output_tokens=extract_limit(raw, "output"),
        max_input_tokens=extract_limit(raw, "input"),
        supports_tools=capability("tool_call"),
        supports_vision=supports_vision,
        supports_reasoning=reasoning.supported,
        supports_structured_output=capability("structured_output"),
        supports_temperature=capability("temperature"),
        input_modalities=input_modalities,
        output_modalities=output_modalities,
        reasoning=reasoning,
        model_family=(str(raw["family"] or "") if "family" in raw else None),
        open_weights=capability("open_weights"),
        release_date=(str(raw["release_date"] or "") if "release_date" in raw else None),
        status=(str(raw["status"] or "") if "status" in raw else None),
        knowledge_cutoff=(str(raw["knowledge"] or "") if "knowledge" in raw else None),
    )


def model_metadata_from_entry(
    ref: ModelRef,
    raw: Mapping[str, Any],
    *,
    unknown_model: bool = False,
    provenance: Optional[Mapping[str, str]] = None,
) -> ModelMetadata:
    """Interpret one effective raw catalogue entry as canonical metadata."""
    patch = model_metadata_patch_from_entry(raw, unknown_model=unknown_model)
    return ModelMetadata(
        ref=ref,
        context_window=patch.context_window,
        max_output_tokens=patch.max_output_tokens,
        max_input_tokens=patch.max_input_tokens,
        supports_tools=patch.supports_tools,
        supports_vision=patch.supports_vision,
        supports_reasoning=patch.supports_reasoning,
        supports_structured_output=patch.supports_structured_output,
        supports_temperature=patch.supports_temperature,
        input_modalities=patch.input_modalities or (),
        output_modalities=patch.output_modalities or (),
        reasoning=patch.reasoning or ReasoningMetadata(),
        model_family=patch.model_family or "",
        open_weights=patch.open_weights,
        release_date=patch.release_date or "",
        status=patch.status or "",
        knowledge_cutoff=patch.knowledge_cutoff or "",
        provenance=dict(provenance or {}),
    )


def model_info_from_entry(
    model_id: str, raw: Mapping[str, Any], provider_id: str
) -> ModelInfo:
    """Interpret one effective raw catalogue entry as ``ModelInfo``."""
    cost = dict_or_empty(raw.get("cost"))
    modalities = dict_or_empty(raw.get("modalities"))

    def modalities_for(key: str) -> tuple[str, ...]:
        values = modalities.get(key) or []
        return tuple(values) if isinstance(values, list) else ()

    def cost_for(key: str) -> Optional[float]:
        return float(cost[key]) if cost.get(key) is not None else None

    return ModelInfo(
        id=model_id,
        name=raw.get("name", "") or model_id,
        family=raw.get("family", "") or "",
        provider_id=provider_id,
        **{
            key: bool(raw.get(key, False))
            for key in (
                "reasoning",
                "tool_call",
                "attachment",
                "temperature",
                "structured_output",
                "open_weights",
            )
        },
        input_modalities=modalities_for("input"),
        output_modalities=modalities_for("output"),
        context_window=extract_limit(raw, "context") or 0,
        max_output=extract_limit(raw, "output") or 0,
        max_input=extract_limit(raw, "input"),
        cost_input=float(cost.get("input", 0) or 0),
        cost_output=float(cost.get("output", 0) or 0),
        cost_cache_read=cost_for("cache_read"),
        cost_cache_write=cost_for("cache_write"),
        knowledge_cutoff=raw.get("knowledge", "") or "",
        release_date=raw.get("release_date", "") or "",
        status=raw.get("status", "") or "",
        interleaved=raw.get("interleaved", False),
    )


def provider_info_from_entry(provider_id: str, raw: Mapping[str, Any]) -> ProviderInfo:
    """Interpret one raw provider catalogue entry as ``ProviderInfo``."""
    env = raw.get("env") or []
    models = raw.get("models") or {}
    return ProviderInfo(
        id=provider_id,
        name=raw.get("name", "") or provider_id,
        env=tuple(env) if isinstance(env, list) else (),
        api=raw.get("api", "") or "",
        doc=raw.get("doc", "") or "",
        model_count=len(models) if isinstance(models, dict) else 0,
    )


__all__ = [
    "UNKNOWN_MODEL_BASE",
    "builtin_model_metadata",
    "dict_or_empty",
    "entry_supports_vision",
    "extract_context",
    "extract_limit",
    "merge_catalog_entry_with_override",
    "model_info_from_entry",
    "model_metadata_from_entry",
    "model_metadata_patch_from_entry",
    "override_int",
    "override_to_catalog_shape",
    "provider_info_from_entry",
    "vision_marker_metadata",
]
