"""Centralized model registry and resolver for Hermes.

Provides a single entry point for resolving any model reference
(legacy config dicts, new ``provider/model`` string IDs) into a
consistent ``ResolvedModel`` object.

Design principles
-----------------
* Dependency-light — no direct imports of gateway, CLI, or auth modules.
  Credential resolution is left to callers.
* Backward compatible — reads ``model.provider`` + ``model.default``
  blocks, ``fallback_providers`` lists, and legacy ``fallback_model``
  dict without requiring config migration.
* Optional ``model_registry`` config section — when absent, the
  registry falls back entirely to legacy config reading.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional


# ── Public helper functions ───────────────────────────────────────────

def model_id(provider: str, model: str) -> str:
    """Build a canonical ``provider/model`` ID string."""
    return f"{provider}/{model}"


def split_model_id(id_str: str) -> tuple[str, str]:
    """Split ``provider/model`` at the first slash.

    The model part may itself contain slashes (e.g.
    ``anthropic/claude-sonnet-4``).
    """
    slash = id_str.find("/")
    if slash <= 0:
        raise ValueError(
            f"invalid model ID {id_str!r}: expected provider/model"
        )
    return id_str[:slash], id_str[slash + 1:]


def legacy_entry_to_ref(entry: Dict[str, Any]) -> ModelRef:
    """Convert a legacy ``{provider, model}`` dict to a ``ModelRef``."""
    provider = str(entry.get("provider", "")).strip()
    model = str(entry.get("model", "")).strip()
    if not provider or not model:
        raise ValueError(
            f"invalid fallback entry {entry}: must have provider and model"
        )
    return ModelRef(provider=provider, model=model)


def resolved_to_legacy_dict(resolved: ResolvedModel) -> Dict[str, Any]:
    """Convert a ``ResolvedModel`` back to a legacy dict.

    Produces a ``{provider, model, base_url?, api_mode?}`` dict
    suitable for writing into ``fallback_providers`` config.
    """
    d: Dict[str, Any] = {
        "provider": resolved.provider,
        "model": resolved.model,
    }
    if resolved.base_url:
        d["base_url"] = resolved.base_url
    if resolved.api_mode:
        d["api_mode"] = resolved.api_mode
    return d


# ── Core types ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelRef:
    """Parsed model identity (provider + model).

    The ``id`` property yields the canonical
    ``provider/model`` string.
    """
    provider: str
    model: str

    @property
    def id(self) -> str:
        return model_id(self.provider, self.model)


@dataclass(frozen=True)
class ResolvedModel:
    """Fully resolved model with all runtime-relevant metadata.

    This object is what every internal consumer uses once a model
    reference has been resolved through the registry.
    """
    id: str
    provider: str
    model: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    api_mode: Optional[str] = None
    context_length: Optional[int] = None
    capabilities: Dict[str, Any] = field(default_factory=dict)
    provider_config: Dict[str, Any] = field(default_factory=dict)
    model_config: Dict[str, Any] = field(default_factory=dict)


# ── ModelRegistry ─────────────────────────────────────────────────────

class ModelRegistry:
    """Centralized model registry and resolver.

    Parameters
    ----------
    config : dict
        The full Hermes config dict (``~/.hermes/config.yaml``).
    env : mapping, optional
        Environment variables (default ``os.environ``).
    """
    def __init__(
        self,
        config: dict,
        *,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self._config = config
        self._env = env or {}
        self._registry_config = config.get("model_registry") or {}
        self._model_cfg = self._config.get("model", {})
        if not isinstance(self._model_cfg, dict):
            self._model_cfg = {}

    # ── parse_ref ─────────────────────────────────────────────────

    def parse_ref(
        self,
        value: str | dict | None,
        *,
        default_provider: str | None = None,
    ) -> ModelRef:
        """Parse a model reference into a ``ModelRef``.

        Accepted forms
        ~~~~~~~~~~~~~~
        * String ID: ``"openrouter/anthropic/claude-sonnet-4"``
        * Legacy dict: ``{"provider": "openrouter", "model": "anthropic/claude-sonnet-4"}``
        * ``None`` or empty string → use ``default_provider`` with model ``""``.
        """
        if value is None or (isinstance(value, str) and not value.strip()):
            return ModelRef(provider=default_provider or "", model="")

        if isinstance(value, str):
            provider, model = split_model_id(value)
            return ModelRef(provider=provider, model=model)

        if isinstance(value, dict):
            return legacy_entry_to_ref(value)

        raise TypeError(
            f"model reference must be str or dict, got {type(value).__name__}"
        )

    # ── resolve ───────────────────────────────────────────────────

    def resolve(
        self,
        value: str | dict | None,
        *,
        role: str = "main",
        default_provider: str | None = None,
    ) -> ResolvedModel:
        """Parse a reference and enrich it with config-level metadata.

        The ``role`` parameter is informational (``"main"``,
        ``"fallback"``, ``"auxiliary"``) and used for error messages.
        """
        ref = self.parse_ref(value, default_provider=default_provider)

        # Merge provider-level config from model_registry.providers.
        provider_cfg: Dict[str, Any] = {}
        models_list: List[Dict[str, Any]] = []
        reg_providers = self._registry_config.get("providers") or {}
        if ref.provider in reg_providers:
            prov_group = reg_providers[ref.provider]
            if isinstance(prov_group, dict):
                provider_cfg = dict(prov_group)
                provider_cfg.pop("models", None)
                models_list = provider_cfg.pop("models", [])
                provider_cfg.pop("models", None)
            models_list = provider_cfg.pop("models", []) if isinstance(prov_group, dict) else []

        # Merge model-level overrides from model_registry.providers.<p>.models.
        model_cfg: Dict[str, Any] = {}
        for entry in models_list:
            if isinstance(entry, dict) and entry.get("id") == ref.id:
                model_cfg = dict(entry)
                model_cfg.pop("id", None)
                break

        caps = dict(model_cfg.pop("capabilities", {}) or {})

        return ResolvedModel(
            id=ref.id,
            provider=ref.provider,
            model=ref.model,
            base_url=provider_cfg.get("base_url") or model_cfg.get("base_url"),
            api_key=provider_cfg.get("api_key") or model_cfg.get("api_key"),
            api_mode=model_cfg.get("api_mode"),
            context_length=model_cfg.get("context_length"),
            capabilities=caps,
            provider_config=provider_cfg,
            model_config=model_cfg,
        )

    # ── main ──────────────────────────────────────────────────────

    def main(self) -> ResolvedModel:
        """Resolve the primary/main model from config.

        Reads ``model.provider`` + ``model.default`` (or ``model.name``
        or ``model.model``).
        """
        if not isinstance(self._model_cfg, dict):
            raise ValueError("no main model configured")

        provider = str(self._model_cfg.get("provider") or "").strip()
        model = str(
            self._model_cfg.get("default")
            or self._model_cfg.get("name")
            or self._model_cfg.get("model")
            or ""
        ).strip()

        if not provider or not model:
            raise ValueError("no main model configured")

        ref = ModelRef(provider=provider, model=model)

        # Merge model_registry.provider config for this provider.
        provider_cfg: Dict[str, Any] = {}
        model_cfg: Dict[str, Any] = {}
        reg_providers = self._registry_config.get("providers") or {}
        if ref.provider in reg_providers:
            prov_group = reg_providers[ref.provider]
            if isinstance(prov_group, dict):
                provider_cfg = dict(prov_group)
                models_list = provider_cfg.pop("models", [])
                provider_cfg.pop("models", None)
                for entry in models_list:
                    if isinstance(entry, dict) and entry.get("id") == ref.id:
                        model_cfg = dict(entry)
                        model_cfg.pop("id", None)
                        break

        caps = dict(model_cfg.pop("capabilities", {}) or {})

        # Also merge fields directly from the model block (base_url,
        # context_length, api_mode) — these are set by the user in
        # config.yaml under model:.
        base_url = provider_cfg.get("base_url") or model_cfg.get("base_url")
        api_key = provider_cfg.get("api_key") or model_cfg.get("api_key")
        api_mode = model_cfg.get("api_mode")
        context_length = model_cfg.get("context_length")
        if isinstance(self._model_cfg, dict):
            if not base_url and self._model_cfg.get("base_url"):
                base_url = self._model_cfg["base_url"]
            if not api_mode and self._model_cfg.get("api_mode"):
                api_mode = self._model_cfg["api_mode"]
            if context_length is None and self._model_cfg.get("context_length"):
                context_length = self._model_cfg["context_length"]

        return ResolvedModel(
            id=ref.id,
            provider=ref.provider,
            model=ref.model,
            base_url=base_url,
            api_key=api_key,
            api_mode=api_mode,
            context_length=context_length,
            capabilities=caps,
            provider_config=provider_cfg,
            model_config=model_cfg,
        )

    # ── fallback_chain ────────────────────────────────────────────

    def fallback_chain(self) -> List[ResolvedModel]:
        """Resolve the fallback provider chain.

        Reads ``fallback_providers`` (list of dicts) first, then falls
        back to legacy ``fallback_model`` (single dict).
        """
        fallback_raw = self._config.get("fallback_providers")
        if fallback_raw is None:
            fallback_raw = self._config.get("fallback_model")

        entries: List[tuple[ModelRef, Dict[str, Any]]] = []
        if isinstance(fallback_raw, list):
            for entry in fallback_raw:
                if isinstance(entry, dict):
                    try:
                        ref = legacy_entry_to_ref(entry)
                        entries.append((ref, entry))
                    except ValueError:
                        continue
        elif isinstance(fallback_raw, dict):
            # Legacy fallback_model (single dict).
            try:
                ref = legacy_entry_to_ref(fallback_raw)
                entries.append((ref, fallback_raw))
            except ValueError:
                pass

        if not entries:
            return []

        return [
            self._resolve_from_ref(ref, raw_entry=raw_entry, role="fallback")
            for ref, raw_entry in entries
        ]

    # ── auxiliary ─────────────────────────────────────────────────

    def auxiliary(self, task: str) -> ResolvedModel:
        """Resolve the model for an auxiliary task.

        If the task config has ``provider: "auto"`` (or is missing),
        returns the main model.
        """
        main = self.main()

        aux_cfg = (self._config.get("auxiliary") or {})
        if not isinstance(aux_cfg, dict):
            return main

        task_cfg = aux_cfg.get(task)
        if not isinstance(task_cfg, dict):
            return main

        provider = str(task_cfg.get("provider") or "").strip()
        if provider == "auto" or not provider:
            return main

        model = str(task_cfg.get("model") or "").strip()
        if not model:
            return main

        ref = ModelRef(provider=provider, model=model)
        return self._resolve_from_ref(ref, role=task)

    # ── to_legacy_agent_kwargs ────────────────────────────────────

    def to_legacy_agent_kwargs(self, resolved: ResolvedModel) -> Dict[str, Any]:
        """Convert a ``ResolvedModel`` into kwargs suitable for
        ``AIAgent.__init__`` / gateway runtime.
        """
        kwargs: Dict[str, Any] = {
            "provider": resolved.provider,
            "model": resolved.model,
        }
        if resolved.base_url:
            kwargs["base_url"] = resolved.base_url
        if resolved.api_key:
            kwargs["api_key"] = resolved.api_key
        if resolved.api_mode:
            kwargs["api_mode"] = resolved.api_mode
        return kwargs

    # ── internal ──────────────────────────────────────────────────

    def _resolve_from_ref(
        self,
        ref: ModelRef,
        *,
        role: str,
        raw_entry: Dict[str, Any] | None = None,
    ) -> ResolvedModel:
        """Helper: resolve a ModelRef into a ResolvedModel.

        Looks up provider-level config from ``model_registry``
        if available.  Falls back to plain identity.
        """
        provider_cfg: Dict[str, Any] = {}
        model_cfg: Dict[str, Any] = {}
        reg_providers = self._registry_config.get("providers") or {}
        if ref.provider in reg_providers:
            prov_group = reg_providers[ref.provider]
            if isinstance(prov_group, dict):
                provider_cfg = dict(prov_group)
                models_list = provider_cfg.pop("models", [])
                provider_cfg.pop("models", None)
                for entry in models_list:
                    if isinstance(entry, dict) and entry.get("id") == ref.id:
                        model_cfg = dict(entry)
                        model_cfg.pop("id", None)
                        break

        caps = dict(model_cfg.pop("capabilities", {}) or {})

        # Merge raw entry fields (base_url, api_mode) from config.
        base_url = provider_cfg.get("base_url") or model_cfg.get("base_url")
        api_key = provider_cfg.get("api_key") or model_cfg.get("api_key")
        api_mode = model_cfg.get("api_mode")
        context_length = model_cfg.get("context_length")
        if raw_entry:
            if not base_url and raw_entry.get("base_url"):
                base_url = raw_entry["base_url"]
            if not api_mode and raw_entry.get("api_mode"):
                api_mode = raw_entry["api_mode"]

        return ResolvedModel(
            id=ref.id,
            provider=ref.provider,
            model=ref.model,
            base_url=base_url,
            api_key=api_key,
            api_mode=api_mode,
            context_length=context_length,
            capabilities=caps,
            provider_config=provider_cfg,
            model_config=model_cfg,
        )
