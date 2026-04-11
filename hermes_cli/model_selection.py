"""Canonical model-selection primitives shared across Hermes surfaces.

Stable hierarchy:

  source -> provider -> model

Where source is one of the long-lived top-level categories:

  - ``openrouter``
  - ``oauth``
  - ``other``

Providers and models are dynamic data beneath those stable categories.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

from agent.credential_pool import load_pool
from hermes_cli.auth import (
    PROVIDER_REGISTRY,
    get_codex_auth_status,
    get_auth_status,
    get_nous_auth_status,
    get_qwen_auth_status,
)
from hermes_cli.codex_models import get_codex_model_ids
from hermes_cli.model_normalize import normalize_model_for_provider
from hermes_cli.models import OPENROUTER_MODELS, _PROVIDER_MODELS, normalize_provider
from hermes_cli.providers import custom_provider_slug, get_label

StatusKind = Literal["none", "success", "error"]

_QWEN_OAUTH_MODELS: tuple[str, ...] = (
    "qwen3-coder-plus",
    "qwen3-coder",
)

_OTHER_PROVIDER_ORDER: tuple[str, ...] = (
    "anthropic",
    "copilot",
    "copilot-acp",
    "gemini",
    "zai",
    "kimi-coding",
    "minimax",
    "minimax-cn",
    "deepseek",
    "xai",
    "kilocode",
    "opencode-zen",
    "opencode-go",
    "ai-gateway",
    "alibaba",
    "huggingface",
)

_OPENROUTER_VENDOR_LABELS: dict[str, str] = {
    "anthropic": "Anthropic",
    "arcee-ai": "Arcee",
    "deepseek": "DeepSeek",
    "google": "Google",
    "meta-llama": "Meta Llama",
    "minimax": "MiniMax",
    "moonshotai": "Moonshot",
    "nvidia": "Nvidia",
    "openai": "OpenAI",
    "qwen": "Qwen",
    "stepfun": "StepFun",
    "x-ai": "X.AI",
    "xiaomi": "Xiaomi",
    "z-ai": "Z.AI",
}


@dataclass(frozen=True)
class SourceCategory:
    id: str
    label: str
    status_label: str = ""
    status_kind: StatusKind = "none"
    current: bool = False


@dataclass(frozen=True)
class ProviderNode:
    id: str
    source_id: str
    provider_slug: str
    token: str
    label: str
    status_label: str = ""
    status_kind: StatusKind = "none"
    current: bool = False


@dataclass(frozen=True)
class ModelNode:
    id: str
    provider_id: str
    provider_slug: str
    model_id: str
    token: str
    label: str
    enabled: bool
    current: bool = False


@dataclass(frozen=True)
class ModelSwitchRequest:
    provider_slug: str
    model_id: str


@dataclass(frozen=True)
class SelectionItem:
    kind: Literal["source", "provider", "model"]
    id: str
    token: str
    label: str
    status_label: str = ""
    status_kind: StatusKind = "none"
    enabled: bool = True
    current: bool = False
    request: Optional[ModelSwitchRequest] = None


@dataclass(frozen=True)
class SelectionView:
    level: Literal["source", "provider", "model"]
    breadcrumb: str
    items: tuple[SelectionItem, ...]

    def default_cursor(self) -> int:
        for idx, item in enumerate(self.items):
            if item.current and item.enabled:
                return idx
        for idx, item in enumerate(self.items):
            if item.enabled:
                return idx
        return 0


@dataclass(frozen=True)
class ModelSelectionTree:
    sources: tuple[SourceCategory, ...]
    providers_by_source: dict[str, tuple[ProviderNode, ...]]
    models_by_provider: dict[str, tuple[ModelNode, ...]]

    def source(self, source_id: str) -> Optional[SourceCategory]:
        for source in self.sources:
            if source.id == source_id:
                return source
        return None

    def providers(self, source_id: str) -> tuple[ProviderNode, ...]:
        return self.providers_by_source.get(source_id, ())

    def provider(self, provider_id: str) -> Optional[ProviderNode]:
        for providers in self.providers_by_source.values():
            for provider in providers:
                if provider.id == provider_id:
                    return provider
        return None

    def models(self, provider_id: str) -> tuple[ModelNode, ...]:
        return self.models_by_provider.get(provider_id, ())


class ModelSelectionController:
    """Stateful navigator for the canonical selection tree."""

    def __init__(self, tree: ModelSelectionTree):
        self.tree = tree
        self.source_id: Optional[str] = None
        self.provider_id: Optional[str] = None
        self.cursor: int = 0
        self._sync_cursor()

    @property
    def active(self) -> bool:
        return True

    @property
    def level(self) -> Literal["source", "provider", "model"]:
        if self.provider_id is not None:
            return "model"
        if self.source_id is not None:
            return "provider"
        return "source"

    def current_view(self) -> SelectionView:
        if self.provider_id is not None:
            provider = self.tree.provider(self.provider_id)
            assert provider is not None
            source = self.tree.source(provider.source_id)
            breadcrumb = f"{source.label} / {provider.label}" if source is not None else provider.label
            items = tuple(
                SelectionItem(
                    kind="model",
                    id=model.id,
                    token=model.token,
                    label=model.label,
                    status_label=(
                        provider.status_label
                        or (source.status_label if source is not None else "")
                    ) if not model.enabled else "",
                    status_kind=(
                        provider.status_kind
                        if provider.status_kind != "none"
                        else (source.status_kind if source is not None else "none")
                    ) if not model.enabled else "none",
                    enabled=model.enabled,
                    current=model.current,
                    request=ModelSwitchRequest(
                        provider_slug=model.provider_slug,
                        model_id=model.model_id,
                    ),
                )
                for model in self.tree.models(self.provider_id)
            )
            return SelectionView(level="model", breadcrumb=breadcrumb, items=items)

        if self.source_id is not None:
            source = self.tree.source(self.source_id)
            breadcrumb = source.label if source is not None else ""
            items = tuple(
                SelectionItem(
                    kind="provider",
                    id=provider.id,
                    token=provider.token,
                    label=provider.label,
                    status_label=provider.status_label,
                    status_kind=provider.status_kind,
                    current=provider.current,
                )
                for provider in self.tree.providers(self.source_id)
            )
            return SelectionView(level="provider", breadcrumb=breadcrumb, items=items)

        items = tuple(
            SelectionItem(
                kind="source",
                id=source.id,
                token=source.id,
                label=source.label,
                status_label=source.status_label,
                status_kind=source.status_kind,
                current=source.current,
            )
            for source in self.tree.sources
        )
        return SelectionView(level="source", breadcrumb="", items=items)

    def move_up(self) -> None:
        view = self.current_view()
        if not view.items:
            return
        self.cursor = (self.cursor - 1) % len(view.items)

    def move_down(self) -> None:
        view = self.current_view()
        if not view.items:
            return
        self.cursor = (self.cursor + 1) % len(view.items)

    def set_source(self, source_id: str) -> bool:
        if self.tree.source(source_id) is None:
            return False
        self.source_id = source_id
        self.provider_id = None
        self._sync_cursor()
        return True

    def set_provider(self, provider_id: str) -> bool:
        provider = self.tree.provider(provider_id)
        if provider is None:
            return False
        self.source_id = provider.source_id
        self.provider_id = provider_id
        self._sync_cursor()
        return True

    def selected_item(self) -> Optional[SelectionItem]:
        view = self.current_view()
        if not view.items:
            return None
        return view.items[max(0, min(self.cursor, len(view.items) - 1))]

    def activate_index(self, index: int) -> Optional[ModelSwitchRequest]:
        view = self.current_view()
        if not view.items:
            return None
        self.cursor = max(0, min(index, len(view.items) - 1))
        return self.enter()

    def enter(self) -> Optional[ModelSwitchRequest]:
        item = self.selected_item()
        if item is None:
            return None
        if item.kind == "source":
            self.source_id = item.id
            self.provider_id = None
            self._sync_cursor()
            return None
        if item.kind == "provider":
            provider = self.tree.provider(item.id)
            if provider is None:
                return None
            if not self.tree.models(item.id):
                if provider.status_kind == "error":
                    return None
                return ModelSwitchRequest(
                    provider_slug=provider.provider_slug,
                    model_id="",
                )
            self.provider_id = item.id
            self._sync_cursor()
            return None
        if not item.enabled:
            return None
        return item.request

    def back(self) -> bool:
        if self.provider_id is not None:
            self.provider_id = None
            self._sync_cursor()
            return True
        if self.source_id is not None:
            self.source_id = None
            self._sync_cursor()
            return True
        return False

    def _sync_cursor(self) -> None:
        self.cursor = self.current_view().default_cursor()


def _status_label(configured: bool) -> tuple[str, StatusKind]:
    return ("configured", "success") if configured else ("login", "error")


def selection_item_meta_text(item: SelectionItem) -> str:
    parts: list[str] = []
    if item.current:
        parts.append("current")
    if item.status_label:
        parts.append(item.status_label)
    return " · ".join(parts)


def _openrouter_has_credentials() -> bool:
    try:
        pool = load_pool("openrouter")
        return bool(pool and pool.has_credentials())
    except Exception:
        return False


def _oauth_status(provider_slug: str) -> dict:
    if provider_slug == "openai-codex":
        return get_codex_auth_status()
    if provider_slug == "nous":
        return get_nous_auth_status()
    if provider_slug == "qwen-oauth":
        return get_qwen_auth_status()
    return {"logged_in": False}


def _vendor_label(vendor_slug: str) -> str:
    label = _OPENROUTER_VENDOR_LABELS.get(vendor_slug)
    if label:
        return label
    bits = [part for part in vendor_slug.replace("_", "-").split("-") if part]
    return " ".join(part[:1].upper() + part[1:] for part in bits) or vendor_slug


def _append_variant(label: str, variant: str) -> str:
    known = {
        "free": "Free",
        "fast": "Fast",
        "extended": "Extended",
    }
    suffix = known.get(variant.lower(), variant.replace("-", " ").title())
    return f"{label} {suffix}".strip()


def _humanize_generic_model(model_id: str) -> str:
    bare = model_id.split("/", 1)[1] if "/" in model_id else model_id
    variant = ""
    if ":" in bare:
        bare, variant = bare.split(":", 1)
    tokens = []
    for part in bare.split("-"):
        if not part:
            continue
        lower = part.lower()
        if lower in {"gpt", "glm", "qwen"}:
            tokens.append(lower.upper())
        elif lower in {"x", "ai"}:
            tokens.append(lower.upper())
        elif lower == "mimo":
            tokens.append("MiMo")
        elif lower.replace(".", "").isdigit():
            tokens.append(part)
        else:
            tokens.append(part[:1].upper() + part[1:])
    label = " ".join(tokens) or bare
    return _append_variant(label, variant) if variant else label


def _humanize_openai_model(model_id: str) -> str:
    bare = model_id.split("/", 1)[1] if "/" in model_id else model_id
    variant = ""
    if ":" in bare:
        bare, variant = bare.split(":", 1)
    if bare.startswith("gpt-"):
        remainder = bare[len("gpt-"):]
        parts = remainder.split("-")
        label = f"GPT-{parts[0]}"
        if len(parts) > 1:
            label += " " + " ".join(part.upper() if part == "o" else part.capitalize() for part in parts[1:])
        return _append_variant(label, variant) if variant else label
    if bare.startswith(("o1-", "o3-", "o4-")):
        family, *rest = bare.split("-")
        label = family.upper()
        if rest:
            label += " " + " ".join(part.capitalize() for part in rest)
        return _append_variant(label, variant) if variant else label
    return _humanize_generic_model(model_id)


def _humanize_anthropic_model(model_id: str) -> str:
    bare = model_id.split("/", 1)[1] if "/" in model_id else model_id
    variant = ""
    if ":" in bare:
        bare, variant = bare.split(":", 1)
    for prefix, family in (
        ("claude-opus-", "Opus"),
        ("claude-sonnet-", "Sonnet"),
        ("claude-haiku-", "Haiku"),
    ):
        if bare.startswith(prefix):
            version = bare[len(prefix):].replace("-", ".")
            label = f"{family} {version}".strip()
            return _append_variant(label, variant) if variant else label
    return _humanize_generic_model(model_id)


def humanize_model_label(model_id: str, *, provider_slug: str = "") -> str:
    bare = model_id.split("/", 1)[1] if "/" in model_id else model_id
    if provider_slug == "openai-codex" or bare.startswith(("gpt-", "o1-", "o3-", "o4-")):
        return _humanize_openai_model(model_id)
    if bare.startswith("claude-"):
        return _humanize_anthropic_model(model_id)
    return _humanize_generic_model(model_id)


def _oauth_current_model_matches(current_model: str, model_id: str) -> bool:
    normalized = (current_model or "").strip()
    return normalized == model_id or normalized.endswith(f"/{model_id}")


def _provider_configured_status(
    provider_slug: str,
    *,
    user_providers: Optional[dict[str, Any]] = None,
    custom_providers: Optional[list[dict[str, Any]]] = None,
) -> tuple[str, StatusKind]:
    if provider_slug.startswith("custom:"):
        if isinstance(custom_providers, list):
            for entry in custom_providers:
                if not isinstance(entry, dict):
                    continue
                display_name = (entry.get("name") or "").strip()
                api_url = (
                    entry.get("base_url", "")
                    or entry.get("url", "")
                    or entry.get("api", "")
                    or ""
                ).strip()
                if display_name and api_url and custom_provider_slug(display_name) == provider_slug:
                    return _status_label(True)
        return _status_label(False)

    if user_providers and provider_slug in user_providers:
        entry = user_providers.get(provider_slug)
        configured = isinstance(entry, dict) and bool(
            (entry.get("api") or entry.get("url") or entry.get("base_url") or "").strip()
        )
        return _status_label(configured)

    status = get_auth_status(provider_slug)
    configured = bool(status.get("configured") or status.get("logged_in"))
    return _status_label(configured)


def _provider_token(provider_slug: str) -> str:
    return provider_slug.strip().lower().replace(" ", "-")


def _provider_display_name(provider_slug: str) -> str:
    pconfig = PROVIDER_REGISTRY.get(provider_slug)
    if pconfig is not None and getattr(pconfig, "name", ""):
        return pconfig.name
    return get_label(provider_slug)


def _provider_model_rows(
    *,
    provider_id: str,
    provider_slug: str,
    model_ids: list[str],
    configured: bool,
    normalized_provider: str,
    current_model: str,
) -> tuple[ModelNode, ...]:
    normalized_current = (
        normalize_model_for_provider(current_model, provider_slug)
        if normalized_provider == provider_slug
        else ""
    )
    return tuple(
        ModelNode(
            id=f"{provider_id}:{model_id}",
            provider_id=provider_id,
            provider_slug=provider_slug,
            model_id=model_id,
            token=model_id,
            label=humanize_model_label(model_id, provider_slug=provider_slug),
            enabled=configured,
            current=_oauth_current_model_matches(normalized_current, model_id),
        )
        for model_id in model_ids
    )


def build_model_selection_tree(
    current_provider: str = "",
    current_model: str = "",
    user_providers: Optional[dict[str, Any]] = None,
    custom_providers: Optional[list[dict[str, Any]]] = None,
) -> ModelSelectionTree:
    normalized_provider = normalize_provider(current_provider) if current_provider else ""
    oauth_provider_slugs = {"openai-codex", "nous", "qwen-oauth"}
    current_source = "openrouter" if normalized_provider == "openrouter" else (
        "oauth" if normalized_provider in oauth_provider_slugs else (
            "other" if normalized_provider else ""
        )
    )

    openrouter_configured = _openrouter_has_credentials()
    openrouter_status_label, openrouter_status_kind = _status_label(openrouter_configured)

    sources = (
        SourceCategory(
            id="openrouter",
            label="OpenRouter",
            status_label=openrouter_status_label,
            status_kind=openrouter_status_kind,
            current=current_source == "openrouter",
        ),
        SourceCategory(
            id="oauth",
            label="OAuth",
            current=current_source == "oauth",
        ),
    )

    current_openrouter_model = normalize_model_for_provider(current_model, "openrouter")
    grouped: dict[str, list[str]] = {}
    ordered_vendors: list[str] = []
    for model_id, _desc in OPENROUTER_MODELS:
        if "/" not in model_id:
            continue
        vendor, _bare = model_id.split("/", 1)
        if vendor not in grouped:
            grouped[vendor] = []
            ordered_vendors.append(vendor)
        grouped[vendor].append(model_id)

    openrouter_providers: list[ProviderNode] = []
    models_by_provider: dict[str, tuple[ModelNode, ...]] = {}
    for vendor in ordered_vendors:
        provider_id = f"openrouter:{vendor}"
        openrouter_providers.append(
            ProviderNode(
                id=provider_id,
                source_id="openrouter",
                provider_slug="openrouter",
                token=vendor,
                label=_vendor_label(vendor),
                current=(
                    normalized_provider == "openrouter"
                    and current_openrouter_model.startswith(f"{vendor}/")
                ),
            )
        )
        models_by_provider[provider_id] = tuple(
            ModelNode(
                id=f"{provider_id}:{model_id}",
                provider_id=provider_id,
                provider_slug="openrouter",
                model_id=model_id,
                token=model_id.split("/", 1)[1] if "/" in model_id else model_id,
                label=humanize_model_label(model_id),
                enabled=openrouter_configured,
                current=(
                    normalized_provider == "openrouter"
                    and current_openrouter_model == model_id
                ),
            )
            for model_id in grouped[vendor]
        )

    oauth_providers: list[ProviderNode] = []
    oauth_defs = (
        ("openai", "openai-codex", "OpenAI", get_codex_auth_status(), get_codex_model_ids(access_token=None)),
        ("nous", "nous", "Nous", get_nous_auth_status(), list(_PROVIDER_MODELS.get("nous", ()))),
        ("qwen", "qwen-oauth", "Qwen", get_qwen_auth_status(), list(_QWEN_OAUTH_MODELS)),
    )
    for token, provider_slug, label, status, model_ids in oauth_defs:
        configured = bool(status.get("logged_in") or status.get("configured"))
        status_label, status_kind = _status_label(configured)
        provider_id = f"oauth:{token}"
        oauth_providers.append(
            ProviderNode(
                id=provider_id,
                source_id="oauth",
                provider_slug=provider_slug,
                token=token,
                label=label,
                status_label=status_label,
                status_kind=status_kind,
                current=normalized_provider == provider_slug,
            )
        )
        normalized_current = normalize_model_for_provider(current_model, provider_slug) if normalized_provider == provider_slug else ""
        models_by_provider[provider_id] = tuple(
            ModelNode(
                id=f"{provider_id}:{model_id}",
                provider_id=provider_id,
                provider_slug=provider_slug,
                model_id=model_id,
                token=model_id.split("/", 1)[1] if "/" in model_id and provider_slug == "nous" else model_id,
                label=humanize_model_label(model_id, provider_slug=provider_slug),
                enabled=configured,
                current=_oauth_current_model_matches(normalized_current, model_id),
            )
            for model_id in model_ids
        )

    other_providers: list[ProviderNode] = []
    seen_other_slugs: set[str] = set()

    for provider_slug in _OTHER_PROVIDER_ORDER:
        model_ids = list(_PROVIDER_MODELS.get(provider_slug, ()))
        if not model_ids:
            continue
        status_label, status_kind = _provider_configured_status(
            provider_slug,
            user_providers=user_providers,
            custom_providers=custom_providers,
        )
        configured = status_kind != "error"
        provider_id = f"other:{provider_slug}"
        other_providers.append(
            ProviderNode(
                id=provider_id,
                source_id="other",
                provider_slug=provider_slug,
                token=_provider_token(provider_slug),
                label=_provider_display_name(provider_slug),
                status_label=status_label,
                status_kind=status_kind,
                current=normalized_provider == provider_slug,
            )
        )
        models_by_provider[provider_id] = _provider_model_rows(
            provider_id=provider_id,
            provider_slug=provider_slug,
            model_ids=model_ids,
            configured=configured,
            normalized_provider=normalized_provider,
            current_model=current_model,
        )
        seen_other_slugs.add(provider_slug)

    if user_providers and isinstance(user_providers, dict):
        for provider_slug, entry in user_providers.items():
            if provider_slug in seen_other_slugs or not isinstance(entry, dict):
                continue
            default_model = (entry.get("default_model") or "").strip()
            model_ids: list[str] = []
            if default_model:
                model_ids.append(default_model)
            elif normalized_provider == provider_slug and current_model:
                model_ids.append(current_model)
            status_label, status_kind = _provider_configured_status(
                provider_slug,
                user_providers=user_providers,
                custom_providers=custom_providers,
            )
            configured = status_kind != "error"
            provider_id = f"other:{provider_slug}"
            other_providers.append(
                ProviderNode(
                    id=provider_id,
                    source_id="other",
                    provider_slug=provider_slug,
                    token=_provider_token(provider_slug),
                    label=(entry.get("name") or provider_slug).strip() or provider_slug,
                    status_label=status_label,
                    status_kind=status_kind,
                    current=normalized_provider == provider_slug,
                )
            )
            models_by_provider[provider_id] = _provider_model_rows(
                provider_id=provider_id,
                provider_slug=provider_slug,
                model_ids=model_ids,
                configured=configured,
                normalized_provider=normalized_provider,
                current_model=current_model,
            )
            seen_other_slugs.add(provider_slug)

    if custom_providers and isinstance(custom_providers, list):
        for entry in custom_providers:
            if not isinstance(entry, dict):
                continue
            display_name = (entry.get("name") or "").strip()
            if not display_name:
                continue
            provider_slug = custom_provider_slug(display_name)
            if provider_slug in seen_other_slugs:
                continue
            model_id = (entry.get("model") or "").strip() or (
                current_model if normalized_provider == provider_slug and current_model else ""
            )
            status_label, status_kind = _provider_configured_status(
                provider_slug,
                user_providers=user_providers,
                custom_providers=custom_providers,
            )
            configured = status_kind != "error"
            provider_id = f"other:{provider_slug}"
            other_providers.append(
                ProviderNode(
                    id=provider_id,
                    source_id="other",
                    provider_slug=provider_slug,
                    token=_provider_token(provider_slug),
                    label=display_name,
                    status_label=status_label,
                    status_kind=status_kind,
                    current=normalized_provider == provider_slug,
                )
            )
            models_by_provider[provider_id] = _provider_model_rows(
                provider_id=provider_id,
                provider_slug=provider_slug,
                model_ids=[model_id] if model_id else [],
                configured=configured,
                normalized_provider=normalized_provider,
                current_model=current_model,
            )
            seen_other_slugs.add(provider_slug)

    sources = (
        sources[0],
        sources[1],
        SourceCategory(
            id="other",
            label="Other providers",
            current=current_source == "other",
        ),
    )

    return ModelSelectionTree(
        sources=sources,
        providers_by_source={
            "openrouter": tuple(openrouter_providers),
            "oauth": tuple(oauth_providers),
            "other": tuple(other_providers),
        },
        models_by_provider=models_by_provider,
    )
