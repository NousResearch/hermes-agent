"""ACP model picker: deduplicated ``provider:model`` rows from the Hermes inventory + named endpoints."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Callable

from acp.schema import ModelInfo, SessionModelState

logger = logging.getLogger("acp_adapter.server")

# Per-provider row cap (clients render all `availableModels` in one dropdown; mirrors the
# MoA picker cap). Not a total cap; the current model is always kept via the fallback insert.
ACP_MAX_MODELS_PER_PROVIDER = 200


def _named_custom_provider_catalogs() -> list[tuple[str, str, list[tuple[str, str]]]]:
    """``(slug, label, [(model_id, description), ...])`` for named endpoints (v12 ``providers:``
    and legacy ``custom_providers:``), which canonical provider enumeration never lists.

    Models = the entry's declared models, refreshed from the live ``/models`` listing when a
    credential exists and ``discover_models`` isn't disabled; declared models survive a failed
    discovery (some endpoints have no ``/models`` route). Slugs use the ``custom:<name>`` shape
    ``parse_model_input``/``resolve_runtime_provider`` resolve, so choice ids round-trip."""
    try:
        from hermes_cli.config import (get_compatible_custom_providers, is_provider_enabled, load_config)
        from hermes_cli.model_switch import _declared_model_ids, _entry_models_discovered, _models_config_is_allowlist
        from hermes_cli.model_switch_providers import _NativePickerModelList, _fetch_picker_live_models
        from hermes_cli.model_switch_providers import _discover_flag
        from hermes_cli.models_local import should_use_ollama_native_catalog
        from hermes_cli.providers import custom_provider_slug
    except ImportError:
        return []

    try:
        cfg = load_config()
        entries = get_compatible_custom_providers(cfg)
    except Exception:
        logger.debug("Could not load named custom providers", exc_info=True)
        return []

    # ``get_compatible_custom_providers`` drops ``enabled``; read disabled keys from raw config.
    raw_providers = cfg.get("providers") if isinstance(cfg, dict) else None
    disabled_keys = {
        str(key).strip().lower()
        for key, raw in (raw_providers.items() if isinstance(raw_providers, dict) else ())
        if isinstance(raw, dict) and not is_provider_enabled(raw)
    }

    def _entry_catalog(entry: dict) -> tuple[str, str, list[tuple[str, str]]] | None:
        field = lambda key: str(entry.get(key) or "").strip()  # noqa: E731
        provider_key, name, base_url = field("provider_key"), field("name"), field("base_url")
        if provider_key.lower() in disabled_keys or not name or not base_url:
            return None
        slug = custom_provider_slug(name, provider_key)

        api_key = field("api_key")
        if not api_key:
            key_env = str(entry.get("key_env") or entry.get("api_key_env") or "").strip()
            api_key = os.environ.get(key_env, "").strip() if key_env else ""

        models_cfg = entry.get("models")
        declared = [m for m in dict.fromkeys([field("model"), *_declared_model_ids(models_cfg)]) if m]

        native_headers = entry.get("extra_headers") or None
        is_ollama_key = provider_key.lower() in {"ollama", "custom:ollama"}
        is_native_ollama = should_use_ollama_native_catalog(
            provider_key if is_ollama_key else "custom", base_url, headers=native_headers
        )
        if not api_key and not declared and not is_native_ollama:
            return None  # nothing to discover with and nothing declared: not addressable

        model_ids = list(declared)
        live = None
        if _discover_flag(entry) and (api_key or is_native_ollama):
            try:
                live = _fetch_picker_live_models(
                    api_key, base_url, provider_key if is_native_ollama and is_ollama_key else "custom",
                    _models_config_is_allowlist(models_cfg, _entry_models_discovered(entry)),
                    headers=native_headers, timeout=1.5, api_mode=entry.get("api_mode"),
                )
            except Exception:
                live = None
            if isinstance(live, _NativePickerModelList):
                model_ids = list(live)
            elif live is not None:
                model_ids = declared + [m for m in live if m not in declared]

        if not model_ids and not isinstance(live, _NativePickerModelList):
            return None
        return slug, name, [(mid, "") for mid in model_ids]

    catalogs = [_entry_catalog(entry) for entry in entries if isinstance(entry, dict)]
    return [c for c in catalogs if c is not None]


def _semantic_provider(provider_id: str, normalize_provider: Callable[[str], str]) -> str:
    raw = str(provider_id or "").strip().lower()
    if raw in {"ollama", "custom:ollama"}:
        return "ollama"
    if raw.startswith("custom:"):
        return raw
    return normalize_provider(raw)


def _empty_catalog_applies(
    provider_id: str, empty_authoritative: set[str], normalize_provider: Callable[[str], str]
) -> bool:
    """True when a named endpoint with an authoritative-empty catalog owns ``provider_id``."""
    raw = str(provider_id or "").strip().lower()
    normalized = normalize_provider(raw)
    if normalized == "custom":
        return any(
            candidate == raw
            or f"custom:{candidate}" == raw
            or (raw == "custom" and candidate == "custom")
            for candidate in empty_authoritative
        )
    return any(
        candidate == raw
        or candidate == f"custom:{normalized}"
        or candidate == f"custom:{raw}"
        or normalize_provider(candidate) == normalized
        for candidate in empty_authoritative
    )


def _choice_provider(model_id: str) -> str:
    """Provider prefix of an encoded choice id; longest configured ``custom:`` slug wins."""
    parts = model_id.split(":")
    if parts[:1] == ["custom"] and len(parts) > 1:
        from hermes_cli.models import _configured_custom_provider_ids

        lowered = model_id.lower()
        for candidate in sorted(
            (p for p in _configured_custom_provider_ids() if p.startswith("custom:")), key=len, reverse=True,
        ):
            if lowered.startswith(candidate + ":"):
                return candidate
        return "custom"
    return parts[0]


def encode_model_choice(provider: str | None, model: str | None) -> str:
    """``provider:model`` so ACP clients keep provider context."""
    raw_model = str(model or "").strip()
    if not raw_model:
        return ""
    raw_provider = str(provider or "").strip().lower()
    return f"{raw_provider}:{raw_model}" if raw_provider else raw_model


@dataclass
class _ModelCatalog:
    """Deduplicated ACP model rows from the inventory + named endpoints.

    Dedupes on the encoded choice id AND a semantic ``provider:model`` id (``ollama`` ==
    ``custom:ollama``). A bare/``custom`` current provider whose base_url matches an ollama
    inventory row is resolved to ``custom:ollama``."""

    normalize_provider: Callable[[str], str]
    current_model: str
    current_choice_provider: str
    current_base_url: str
    models: list[ModelInfo] = field(default_factory=list)
    seen_ids: set[str] = field(default_factory=set)
    seen_semantic_ids: set[str] = field(default_factory=set)
    empty_authoritative: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.current_choice_provider == "ollama":
            self.current_choice_provider = "custom:ollama"
        self._identity_resolved = self.current_choice_provider not in {"", "custom"}

    def semantic(self, provider_id: str) -> str:
        return _semantic_provider(provider_id, self.normalize_provider)

    def add(self, provider_id: str, model_id: str, name: str, description: str) -> None:
        choice_id = encode_model_choice(provider_id, model_id)
        semantic_id = f"{self.semantic(provider_id)}:{model_id}"
        if not choice_id or choice_id in self.seen_ids or semantic_id in self.seen_semantic_ids:
            return
        self.models.append(ModelInfo(model_id=choice_id, name=name, description=description))
        self.seen_ids.add(choice_id)
        self.seen_semantic_ids.add(semantic_id)

    def add_inventory_rows(self, rows: list, provider_label: Callable[[str], str]) -> None:
        for row in rows:
            raw_row_provider = str(row.get("slug") or "").strip().lower()
            row_provider = self.normalize_provider(raw_row_provider)
            row_base_url = str(row.get("api_url") or "").strip().rstrip("/").lower()
            if row.get("native_catalog_empty"):
                self.empty_authoritative.add(raw_row_provider)
            if not self._identity_resolved and raw_row_provider in {"ollama", "custom:ollama"} and (
                self.current_base_url and row_base_url == self.current_base_url
            ):
                self.current_choice_provider = "custom:ollama"
                self._identity_resolved = True
            row_models = row.get("models")
            if not row_provider or not isinstance(row_models, (list, tuple)):
                continue
            provider_name = str(row.get("name") or "").strip() or provider_label(row_provider)
            encoded_provider = (
                "custom:ollama" if raw_row_provider == "ollama"
                else raw_row_provider if raw_row_provider.startswith("custom:")
                else row_provider
            )
            for model_entry in row_models:
                if isinstance(model_entry, dict):
                    model_entry = model_entry.get("id") or model_entry.get("model") or model_entry.get("name")
                rendered_model = str(model_entry or "").strip()
                if not rendered_model:
                    continue
                is_current = rendered_model == self.current_model and (
                    self.semantic(encoded_provider) == self.semantic(self.current_choice_provider)
                )
                self.add(
                    encoded_provider, rendered_model, f"{provider_name} · {rendered_model}",
                    f"Provider: {provider_name}" + (" • current" if is_current else ""),
                )

    def add_named_catalogs(self, catalogs: list, normalized_provider: str) -> None:
        """Named user-defined endpoints (providers: / custom_providers:) are invisible
        to canonical enumeration — append them like the TUI /model picker. An empty
        catalog marks that slug authoritative-empty."""
        for named_slug, named_label, named_catalog in catalogs:
            if not named_catalog:
                self.empty_authoritative.add(str(named_slug).strip().lower())
                continue
            for named_model, named_desc in named_catalog:
                is_current = named_slug == normalized_provider and named_model == self.current_model
                parts = [f"Provider: {named_label}", str(named_desc or "").strip(), "current" if is_current else ""]
                self.add(named_slug, named_model, named_model, " • ".join(part for part in parts if part))


def build_model_state(model: str, provider: str, base_url: str) -> SessionModelState | None:
    """Picker state from the shared inventory + named endpoints; ``None`` when nothing is listable
    (caller falls back to a single current-model row). Raises on inventory failure."""
    from hermes_cli.inventory import build_models_payload, load_picker_context
    from hermes_cli.models import normalize_provider, provider_label

    normalized_provider = normalize_provider(provider)
    context = load_picker_context().with_overrides(
        current_provider=normalized_provider, current_model=model, current_base_url=base_url,
    )
    payload = build_models_payload(
        context, explicit_only=True, include_unconfigured=False, picker_hints=False,
        canonical_order=True, pricing=False, capabilities=False, refresh=False,
        probe_custom_providers=False, probe_current_custom_provider=False, max_models=ACP_MAX_MODELS_PER_PROVIDER,
    )

    cat = _ModelCatalog(
        normalize_provider=normalize_provider, current_model=model,
        current_choice_provider=str(provider or "").strip().lower(),
        current_base_url=base_url.strip().rstrip("/").lower(),
    )
    cat.add_inventory_rows(payload.get("providers") or [], provider_label)
    cat.add_named_catalogs(_named_custom_provider_catalogs(), normalized_provider)
    available_models = cat.models

    def empty_applies(provider_id: str) -> bool:
        return _empty_catalog_applies(provider_id, cat.empty_authoritative, normalize_provider)

    if cat.empty_authoritative:
        available_models = [m for m in available_models if not empty_applies(_choice_provider(m.model_id))]

    current_is_empty = empty_applies(cat.current_choice_provider)
    if current_is_empty:
        available_models = [m for m in available_models if " • current" not in str(m.description or "")]
    current_model_id = "" if current_is_empty else encode_model_choice(cat.current_choice_provider, model)
    if current_model_id and current_model_id not in {item.model_id for item in available_models}:
        provider_name = provider_label(normalized_provider)
        available_models.insert(0, ModelInfo(
            model_id=current_model_id, name=f"{provider_name} · {model}",
            description=f"Provider: {provider_name} • current",
        ))

    if not available_models and current_is_empty:
        return SessionModelState(available_models=[], current_model_id="")
    if available_models:
        return SessionModelState(
            available_models=available_models,
            current_model_id=current_model_id if current_model_id or current_is_empty else available_models[0].model_id,
        )
    return None
