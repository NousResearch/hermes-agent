"""Pure provider detection policy for explicit model selection."""

from __future__ import annotations

from dataclasses import dataclass

from models.identity import ModelRef
from providers.identity import normalize_provider


@dataclass(frozen=True, slots=True)
class ExplicitDetectionFacts:
    """Caller-supplied discovery and availability facts for provider detection."""

    current_catalog_model: str = ""
    current_provider_owns_model: bool = False
    named_provider_candidate: ModelRef | None = None
    static_candidates: tuple[ModelRef, ...] = ()
    current_static_owns_model: bool = False
    openrouter_candidate: ModelRef | None = None
    declared_provider_candidate: ModelRef | None = None
    eligible_providers: tuple[str, ...] = ()
    allow_first_guess: bool = False

    def candidate_providers(self) -> tuple[str, ...]:
        refs = (
            *(self.static_candidates),
            *(ref for ref in (self.named_provider_candidate,
                              self.openrouter_candidate,
                              self.declared_provider_candidate) if ref is not None),
        )
        return tuple(dict.fromkeys(ref.provider for ref in refs if ref.provider))


def _same_provider(left: str, right: str) -> bool:
    return normalize_provider(left) == normalize_provider(right)


def select_detected_model(
    raw_model: str,
    current_provider: str,
    facts: ExplicitDetectionFacts,
) -> ModelRef | None:
    """Apply the historical provider-detection precedence to materialized facts."""

    raw = str(raw_model or "").strip()
    current = str(current_provider or "").strip().lower()
    if facts.current_catalog_model:
        return ModelRef(current, facts.current_catalog_model)
    if facts.current_provider_owns_model:
        return ModelRef(current, raw)
    if facts.named_provider_candidate is not None:
        return facts.named_provider_candidate

    eligible = {normalize_provider(provider) for provider in facts.eligible_providers}
    first_guess: ModelRef | None = None

    for candidate in facts.static_candidates:
        if _same_provider(candidate.provider, current) or normalize_provider(candidate.provider) in eligible:
            return candidate
        first_guess = first_guess or candidate

    if facts.current_static_owns_model:
        return ModelRef(current, raw)

    candidate = facts.openrouter_candidate
    if candidate is not None:
        if _same_provider(candidate.provider, current):
            if candidate.model == raw:
                return ModelRef(current, raw)
            return candidate
        if normalize_provider(candidate.provider) in eligible:
            return candidate
        first_guess = first_guess or candidate

    candidate = facts.declared_provider_candidate
    if candidate is not None:
        if _same_provider(candidate.provider, current) or normalize_provider(candidate.provider) in eligible:
            return candidate
        first_guess = first_guess or candidate

    if facts.allow_first_guess and first_guess is not None:
        return first_guess

    # A configured provider prefix is an explicit declaration, not a guess.
    if facts.declared_provider_candidate is not None:
        return facts.declared_provider_candidate
    return None


__all__ = ["ExplicitDetectionFacts", "select_detected_model"]
