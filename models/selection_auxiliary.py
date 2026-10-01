"""Pure auxiliary-model selection over caller-supplied provider/model facts."""

from __future__ import annotations

import re
from collections.abc import Iterable

from models.selection_defaults import select_default_model
from models.selection_types import ModelSelection
from providers.identity import normalize_provider


_FAST_MODEL_FAMILIES = (
    "gpt-mini-latest", "gpt-nano-latest", "claude-haiku-latest", "gemini-flash-latest",
    "gpt-5.4-nano", "gpt-5.4-mini", "gpt-5-mini", "haiku-4.5", "gemini-3.6-flash",
    "flash-lite", "-nano", "-mini", "-flash", "haiku",
)

_FAST_MODEL_EXCLUDES = (
    "thinking", "reason", "-r1", "minilm", ":batch", ":free",
    "o1-", "o3-", "o4-", "codex", "audio", "-vl", "embed",
    "-tts", "-transcribe", "-realtime", "-image", "-search-preview",
)

_FAST_AUXILIARY_TASKS = frozenset({"title_generation"})


def auxiliary_task_prefers_fast_model(
    task: str | None,
    preference_enabled: bool,
) -> bool:
    """Whether a task may apply an explicitly enabled fast-model preference."""

    return bool(preference_enabled and task in _FAST_AUXILIARY_TASKS)


def _ordered_ids(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        model = str(value or "").strip()
        key = model.lower()
        if model and key not in seen:
            seen.add(key)
            out.append(model)
    return tuple(out)


def _model_recency_key(model_id: str) -> tuple:
    return tuple(
        (1, float(part), "") if index % 2 else (0, 0.0, part)
        for index, part in enumerate(re.split(r"(\d+(?:\.\d+)?)", model_id.lower()))
        if part
    )


def _allowed(model_id: str, allowed_model_ids: frozenset[str] | None) -> bool:
    if not allowed_model_ids:
        return True
    return model_id in allowed_model_ids or model_id.split(":", 1)[0] in allowed_model_ids


def select_fast_auxiliary_model(
    provider: str,
    model_ids: Iterable[str],
    *,
    allowed_model_ids: Iterable[str] | None = None,
) -> ModelSelection:
    """Pick the newest latency-oriented model from an already-fetched catalog."""

    provider_id = normalize_provider(provider)
    allowed = (
        frozenset(str(value).strip() for value in allowed_model_ids if str(value).strip())
        if allowed_model_ids is not None
        else None
    )
    catalog = sorted(
        (mid for mid in _ordered_ids(model_ids) if _allowed(mid, allowed)),
        key=_model_recency_key,
        reverse=True,
    )
    ranked: list[str] = []
    seen: set[str] = set()
    for family in _FAST_MODEL_FAMILIES:
        for model_id in catalog:
            lowered = model_id.lower()
            if (
                family in lowered
                and not any(token in lowered for token in _FAST_MODEL_EXCLUDES)
                and lowered not in seen
            ):
                seen.add(lowered)
                ranked.append(model_id)
    return select_default_model(provider_id, ranked, purpose="auxiliary_fast")


def select_auxiliary_model(
    provider: str,
    *,
    main_model: str = "",
    resolved_aux_model: str = "",
    default_aux_model: str = "",
    live_model_ids: Iterable[str] = (),
    prefer_fast: bool = False,
    allowed_model_ids: Iterable[str] | None = None,
    purpose: str = "auxiliary",
) -> ModelSelection:
    """Choose a provider-local auxiliary model from already-known facts."""

    provider_id = normalize_provider(provider)
    allowed = (
        frozenset(str(value).strip() for value in allowed_model_ids if str(value).strip())
        if allowed_model_ids is not None
        else None
    )
    if prefer_fast:
        fast = select_fast_auxiliary_model(
            provider_id, live_model_ids, allowed_model_ids=allowed,
        )
        if fast.selected is not None:
            return fast

    choices: list[str] = []
    if prefer_fast and resolved_aux_model:
        choices.append(resolved_aux_model)
    if default_aux_model:
        choices.append(default_aux_model)
    if main_model:
        choices.append(main_model)
    eligible = [mid for mid in _ordered_ids(choices) if _allowed(mid, allowed)]
    return select_default_model(provider_id, eligible, purpose=purpose)


def select_auxiliary_fallback_model(
    provider: str,
    *,
    preferred_model: str = "",
    fallback_model: str = "",
    excluded_model: str = "",
    allowed_model_ids: Iterable[str] | None = None,
) -> ModelSelection:
    """Choose a fallback-lane model from an external recommendation then provider fallback."""

    allowed = (
        frozenset(str(value).strip() for value in allowed_model_ids if str(value).strip())
        if allowed_model_ids is not None
        else None
    )
    excluded = str(excluded_model or "").strip().lower()
    choices = [
        mid for mid in _ordered_ids((preferred_model, fallback_model))
        if mid.lower() != excluded and _allowed(mid, allowed)
    ]
    return select_default_model(
        normalize_provider(provider), choices, purpose="auxiliary_fallback"
    )


def select_vision_auxiliary_model(
    provider: str,
    *,
    explicit_model: str = "",
    vision_default: str = "",
    main_model: str = "",
    main_supports_vision: bool | None = None,
) -> ModelSelection:
    """Choose explicit vision model, provider vision default, then usable main model."""

    choices: list[str] = []
    if explicit_model:
        choices.append(explicit_model)
    if vision_default:
        choices.append(vision_default)
    if main_model and main_supports_vision is not False:
        choices.append(main_model)
    return select_default_model(
        normalize_provider(provider),
        _ordered_ids(choices),
        purpose="auxiliary_vision",
    )


def selected_auxiliary_model_id(selection: ModelSelection) -> str:
    return selection.selected.ref.model if selection.selected is not None else ""


__all__ = [
    "auxiliary_task_prefers_fast_model",
    "select_auxiliary_fallback_model",
    "select_auxiliary_model",
    "select_fast_auxiliary_model",
    "select_vision_auxiliary_model",
    "selected_auxiliary_model_id",
]
