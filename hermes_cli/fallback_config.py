"""Helpers for reading the effective fallback provider chain from config."""

from __future__ import annotations

import os
from typing import Any


def _normalized_base_url(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().rstrip("/")


def resolve_entry_api_key(entry: dict[str, Any] | None) -> str | None:
    """API key for one fallback entry: inline ``api_key``, else ``key_env``.

    Mirrors the custom-provider convention (``key_env`` names the env var
    holding the key; ``api_key_env`` accepted as an alias). Returns None when
    neither yields a non-empty value, letting ``resolve_runtime_provider``
    fall through to the provider's standard credential resolution.
    """
    if not isinstance(entry, dict):
        return None
    inline = str(entry.get("api_key") or "").strip()
    if inline:
        return inline
    key_env = str(entry.get("key_env") or entry.get("api_key_env") or "").strip()
    if key_env:
        return os.getenv(key_env, "").strip() or None
    return None


def _split_provider_model(text: str) -> tuple[str, str] | None:
    """Split a ``provider:model`` string into ``(provider, model)``.

    Mirrors the ``provider:model`` grammar of
    :func:`hermes_cli.models.parse_model_input`: the split is anchored to the
    **first** colon so model ids that themselves contain colons stay intact
    (``ollama-cloud:nemotron-3-nano:30b`` → provider ``ollama-cloud``, model
    ``nemotron-3-nano:30b``; ``openrouter:qwen/qwen3.6-plus:free`` → provider
    ``openrouter``, model ``qwen/qwen3.6-plus:free``).

    The ``custom:<name>:<model>`` triple is special-cased so a qualified custom
    endpoint id keeps its ``custom:<name>`` prefix
    (``custom:my-endpoint:claude-sonnet-4.6`` → provider ``custom:my-endpoint``,
    model ``claude-sonnet-4.6``).

    Returns ``None`` when the string can't be interpreted as ``provider:model``
    (no colon, or an empty provider/model half).
    """
    colon = text.find(":")
    if colon <= 0:
        return None
    provider = text[:colon].strip()
    model = text[colon + 1:].strip()
    if not provider or not model:
        return None
    # custom:<name>:<model> → provider "custom:<name>", model "<model>",
    # matching the named-custom-provider case in parse_model_input.
    if provider.lower() == "custom" and ":" in model:
        second = model.find(":")
        custom_name = model[:second].strip()
        actual_model = model[second + 1:].strip()
        if custom_name and actual_model:
            return (f"custom:{custom_name}", actual_model)
    return (provider, model)


def _coerce_entry(entry: Any, *, allow_strings: bool) -> dict[str, Any] | None:
    """Coerce a single raw fallback entry into a ``{provider, model, ...}`` dict.

    Two on-disk shapes are accepted:

    * ``dict`` — the canonical form written by ``hermes fallback`` and the CLI,
      and the only shape the config validator accepts for either key.
    * ``"provider:model"`` string — only when ``allow_strings`` is set. This is
      the shape produced by the desktop *Model → Fallback Models* settings
      field, which is rendered by the generic comma-separated ``list`` editor
      and whose help text reads *"Backup provider:model entries to try if the
      default model fails."* String coercion is scoped to ``fallback_providers``
      (the desktop list field); the legacy ``fallback_model`` key stays
      dict-only to match its documented, validated contract.

    Returns ``None`` for shapes that can't be interpreted (e.g. a bare model
    with no ``provider:`` prefix, or a non-string/non-dict value).
    """
    if isinstance(entry, dict):
        return entry
    if allow_strings and isinstance(entry, str):
        split = _split_provider_model(entry.strip())
        if split is None:
            return None
        provider, model = split
        return {"provider": provider, "model": model}
    return None


def _iter_fallback_entries(raw: Any, *, allow_strings: bool = False) -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        candidates: list[Any] = [raw]
    elif isinstance(raw, str):
        candidates = [raw] if allow_strings else []
    elif isinstance(raw, list):
        candidates = list(raw)
    else:
        return []

    entries: list[dict[str, Any]] = []
    for raw_entry in candidates:
        entry = _coerce_entry(raw_entry, allow_strings=allow_strings)
        if entry is None:
            continue
        provider = str(entry.get("provider") or "").strip()
        model = str(entry.get("model") or "").strip()
        if not provider or not model:
            continue

        normalized = dict(entry)
        normalized["provider"] = provider
        normalized["model"] = model

        base_url = _normalized_base_url(entry.get("base_url"))
        if base_url:
            normalized["base_url"] = base_url

        entries.append(normalized)
    return entries


def _entry_identity(entry: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(entry.get("provider") or "").strip().lower(),
        str(entry.get("model") or "").strip().lower(),
        _normalized_base_url(entry.get("base_url")).lower(),
    )


def get_fallback_chain(config: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return the effective fallback chain merged across old and new config keys.

    ``fallback_providers`` remains the primary source of truth and keeps its
    order. Legacy ``fallback_model`` entries are appended afterwards unless
    they target the same provider/model/base_url route as an earlier entry.
    The returned list always contains fresh dict copies.

    ``"provider:model"`` string entries are only honored for the
    ``fallback_providers`` list — the shape the desktop settings field writes.
    ``fallback_model`` stays dict-only to match its documented/validated
    contract.
    """

    config = config or {}
    chain: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for key, allow_strings in (("fallback_providers", True), ("fallback_model", False)):
        for entry in _iter_fallback_entries(config.get(key), allow_strings=allow_strings):
            identity = _entry_identity(entry)
            if identity in seen:
                continue
            seen.add(identity)
            chain.append(entry)

    return chain
