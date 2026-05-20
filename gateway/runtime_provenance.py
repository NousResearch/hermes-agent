"""Gateway runtime provenance helpers.

The gateway sometimes carries queued context notes across turns (for example a
session-scoped ``/model`` switch note). Those notes are hints, not authority.
The live runtime chosen for the actual agent turn wins, and conflicts are logged
so stale injected metadata cannot make the assistant self-identify as the wrong
provider/model.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Mapping

logger = logging.getLogger(__name__)

_LEGACY_SWITCH_RE = re.compile(
    r"model was just switched from (?P<previous>.+?) to (?P<requested>.+?) "
    r"via (?P<provider>.+?)\. Adjust your self-identification accordingly",
    re.IGNORECASE,
)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _normalize_provider(value: Any) -> str:
    provider = _clean(value).lower().replace(" ", "-").replace("_", "-")
    aliases = {
        "anthropic-claude": "anthropic",
        "claude": "anthropic",
        "openai-codex-oauth": "openai-codex",
        "codex": "openai-codex",
        "openai-codex": "openai-codex",
    }
    return aliases.get(provider, provider)


def _normalize_model(value: Any) -> str:
    model = _clean(value).lower()
    if not model:
        return ""
    return model.rsplit("/", 1)[-1]


def _runtime_label(*, provider: Any, model: Any) -> str:
    runtime_model = _clean(model) or "unknown-model"
    runtime_provider = _normalize_provider(provider) or "unknown-provider"
    return f"{runtime_provider}/{runtime_model}"


def build_model_switch_note(
    *,
    previous_model: str,
    requested_model: str,
    requested_provider: str,
) -> dict[str, str]:
    """Return a structured one-shot model switch note for the next turn."""
    provider = _normalize_provider(requested_provider) or _clean(requested_provider)
    model = _clean(requested_model)
    previous = _clean(previous_model)
    return {
        "previous_model": previous,
        "requested_model": model,
        "requested_provider": provider,
        "text": (
            f"[Runtime metadata note: /model requested switch from {previous or 'unknown'} "
            f"to {provider}/{model or 'unknown'}. This queued note is advisory; "
            "live runtime metadata/footer for the current turn is authoritative "
            "for self-identification.]"
        ),
    }


def _coerce_pending_note(note: Any) -> tuple[str, str, str, str]:
    """Return (text, previous_model, requested_model, requested_provider)."""
    if isinstance(note, Mapping):
        text = _clean(note.get("text"))
        previous = _clean(note.get("previous_model"))
        requested = _clean(note.get("requested_model") or note.get("model"))
        provider = _clean(note.get("requested_provider") or note.get("provider"))
        return text, previous, requested, provider

    text = _clean(note)
    match = _LEGACY_SWITCH_RE.search(text)
    if not match:
        return text, "", "", ""
    return (
        text,
        _clean(match.group("previous")),
        _clean(match.group("requested")),
        _clean(match.group("provider")),
    )


def _conflicts(
    *,
    requested_model: str,
    requested_provider: str,
    runtime_model: str,
    runtime_provider: str,
) -> bool:
    if requested_model and runtime_model:
        if _normalize_model(requested_model) != _normalize_model(runtime_model):
            return True
    if requested_provider and runtime_provider:
        if _normalize_provider(requested_provider) != _normalize_provider(runtime_provider):
            return True
    return False


def resolve_self_identification_note(
    pending_note: Any,
    *,
    runtime_model: str,
    runtime_provider: str,
    session_key: str = "",
    log: logging.Logger | None = None,
) -> str:
    """Resolve a queued self-identification note against live runtime metadata.

    The returned text is safe to prepend to the next user turn. If queued
    metadata conflicts with the actual runtime for this turn, the conflict is
    both logged and surfaced to the model as diagnostic context, with explicit
    instruction to prefer the live runtime/footer.
    """
    text, previous_model, requested_model, requested_provider = _coerce_pending_note(pending_note)
    runtime_label = _runtime_label(provider=runtime_provider, model=runtime_model)
    requested_label = _runtime_label(provider=requested_provider, model=requested_model)

    if _conflicts(
        requested_model=requested_model,
        requested_provider=requested_provider,
        runtime_model=runtime_model,
        runtime_provider=runtime_provider,
    ):
        active_logger = log or logger
        active_logger.warning(
            "runtime provenance conflict: session=%s queued=%s live=%s previous_model=%s",
            session_key or "",
            requested_label,
            runtime_label,
            previous_model or "",
        )
        return (
            f"[Runtime provenance conflict: queued /model switch requested "
            f"{requested_label}, but the live runtime is {runtime_label}. "
            "Prefer the live runtime/footer for self-identification; treat the "
            "queued note as stale diagnostic context, not truth.]"
        )

    if text:
        return f"{text}\n[Live runtime for this turn: {runtime_label}.]"
    return f"[Live runtime for this turn: {runtime_label}.]"
