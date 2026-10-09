"""Pure curated Codex model catalogue policy."""

from __future__ import annotations

from typing import Iterable

from models.metadata.context import CODEX_CONTEXT_VARIANT_SUFFIX, has_codex_context_variant

DEFAULT_CODEX_MODELS: tuple[str, ...] = (
    "gpt-6-sol",
    "gpt-6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "gpt-5.5",
    "gpt-5.4-mini",
    "gpt-5.4",
    "gpt-5.3-codex-spark",
)

FORWARD_COMPAT_TEMPLATE_MODELS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("gpt-6-sol", ("gpt-5.6-sol", "gpt-5.5")),
    ("gpt-6-luna", ("gpt-5.6-luna", "gpt-5.5")),
    ("gpt-5.6-sol", ("gpt-5.5", "gpt-5.4")),
    ("gpt-5.6-terra", ("gpt-5.5", "gpt-5.4")),
    ("gpt-5.6-luna", ("gpt-5.5", "gpt-5.4")),
    ("gpt-5.5", ("gpt-5.4", "gpt-5.4-mini")),
    ("gpt-5.3-codex-spark", ("gpt-5.4", "gpt-5.5")),
)


def dedupe_model_ids(model_ids: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(model_ids))


def add_forward_compat_models(model_ids: list[str]) -> list[str]:
    ordered = dedupe_model_ids(model_ids)
    seen = set(ordered)
    for synthetic_model, template_models in FORWARD_COMPAT_TEMPLATE_MODELS:
        if synthetic_model not in seen and any(template in seen for template in template_models):
            ordered.append(synthetic_model)
            seen.add(synthetic_model)
    return ordered


def add_context_variants(model_ids: list[str]) -> list[str]:
    out: list[str] = []
    present = set(model_ids)
    for model_id in model_ids:
        out.append(model_id)
        variant = model_id + CODEX_CONTEXT_VARIANT_SUFFIX
        if variant in present or variant in out:
            continue
        if has_codex_context_variant(model_id):
            out.append(variant)
    return out


def finalize_codex_models(model_ids: list[str]) -> list[str]:
    return add_context_variants(add_forward_compat_models(model_ids))


def drop_undiscovered_astra(model_ids: list[str]) -> list[str]:
    """Remove account-gated Astra aliases unless live discovery supplied them."""
    from models.metadata.reasoning import is_astra_model

    return [model for model in model_ids if not is_astra_model(model)]


def curated_codex_models() -> tuple[str, ...]:
    return tuple(finalize_codex_models(list(DEFAULT_CODEX_MODELS)))


__all__ = [
    "DEFAULT_CODEX_MODELS",
    "FORWARD_COMPAT_TEMPLATE_MODELS",
    "add_context_variants",
    "add_forward_compat_models",
    "curated_codex_models",
    "dedupe_model_ids",
    "drop_undiscovered_astra",
    "finalize_codex_models",
]