"""Pure interpretation of an explicit model-selection request."""

from __future__ import annotations

from dataclasses import dataclass, replace

from models.aliases import MODEL_ALIASES, resolve_declared_model_id, resolve_model_alias
from models.identity import ModelRef, normalize_model_ref, parse_model_ref
from models.selection_detection import ExplicitDetectionFacts, select_detected_model
from models.selection_types import ModelSelection, SelectionCandidate, SelectionRequest
from providers.identity import normalize_provider


@dataclass(frozen=True, slots=True)
class ExplicitAlias:
    name: str
    ref: ModelRef


@dataclass(frozen=True, slots=True)
class ExplicitProviderFacts:
    provider: str
    alias_models: tuple[str, ...] = ()
    catalog_models: tuple[str, ...] = ()
    model_aliases: tuple[tuple[str, str], ...] = ()
    identity_aliases: tuple[str, ...] = ()
    aggregator: bool = False
    normalization_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        provider = normalize_provider(self.provider)
        aliases = tuple(dict.fromkeys(
            str(value or "").strip().lower()
            for value in (provider, *self.identity_aliases)
            if str(value or "").strip()
        ))
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "identity_aliases", aliases)


class ExplicitSelectionError(ValueError):
    def __init__(
        self, code: str, raw_input: str, *, providers: tuple[str, ...] = ()
    ) -> None:
        self.code = code
        self.raw_input = raw_input
        self.providers = providers
        super().__init__(code)


def _facts_for(
    provider: str, facts: tuple[ExplicitProviderFacts, ...]
) -> ExplicitProviderFacts | None:
    target = normalize_provider(provider)
    return next(
        (item for item in facts if target == item.provider or target in item.identity_aliases),
        None,
    )


def _providers_match(
    left: str, right: str, facts: tuple[ExplicitProviderFacts, ...]
) -> bool:
    a, b = normalize_provider(left), normalize_provider(right)
    if a == b:
        return True
    left_facts, right_facts = _facts_for(a, facts), _facts_for(b, facts)
    left_ids = set(left_facts.identity_aliases if left_facts else (a,))
    right_ids = set(right_facts.identity_aliases if right_facts else (b,))
    return bool(left_ids & right_ids)


def _direct_alias(
    raw: str,
    provider: str,
    aliases: tuple[ExplicitAlias, ...],
    facts: tuple[ExplicitProviderFacts, ...],
    *,
    keep_provider: bool,
) -> tuple[ModelRef, str] | None:
    key = raw.lower()
    direct = next((item for item in aliases if item.name.lower() == key), None)
    if direct is not None:
        target = provider if keep_provider else direct.ref.provider
        matched = (
            direct.name
            if not keep_provider or _providers_match(direct.ref.provider, provider, facts)
            else ""
        )
        return ModelRef(target, direct.ref.model), matched

    matches = [item for item in aliases if item.ref.model.lower() == key]
    if not matches:
        return None
    chosen = next(
        (item for item in matches if _providers_match(item.ref.provider, provider, facts)),
        matches[0],
    )
    target = provider if keep_provider else chosen.ref.provider
    matched = (
        chosen.name
        if not keep_provider or _providers_match(chosen.ref.provider, provider, facts)
        else ""
    )
    return ModelRef(target, chosen.ref.model), matched


def _provider_alias(
    raw: str, provider: str, fact: ExplicitProviderFacts | None
) -> tuple[ModelRef, str] | None:
    if fact is None:
        return None
    aliases = dict(fact.model_aliases)
    declared = resolve_declared_model_id(
        raw, fact.provider, fact.alias_models, provider_aliases=aliases
    )
    if aliases and declared is not None:
        return ModelRef(fact.provider, declared), raw.lower()
    if raw.lower() not in MODEL_ALIASES:
        return None
    resolved = resolve_model_alias(
        raw, fact.provider, fact.alias_models, provider_aliases=aliases
    )
    return (ModelRef(fact.provider, resolved), raw.lower()) if resolved else None


def _alias_resolution(
    raw: str,
    provider: str,
    facts: tuple[ExplicitProviderFacts, ...],
    aliases: tuple[ExplicitAlias, ...],
    *,
    keep_provider: bool = False,
) -> tuple[ModelRef, str] | None:
    direct = _direct_alias(raw, provider, aliases, facts, keep_provider=keep_provider)
    if direct is not None:
        return direct
    return _provider_alias(raw, provider, _facts_for(provider, facts))


def _catalog_match(raw: str, fact: ExplicitProviderFacts | None) -> str | None:
    if fact is None or not fact.aggregator:
        return None
    wanted = raw.lower()
    return next((mid for mid in fact.catalog_models if mid.lower() == wanted), None) or next(
        (
            mid for mid in fact.catalog_models
            if "/" in mid and mid.split("/", 1)[1].lower() == wanted
        ),
        None,
    )


def _normalized(ref: ModelRef, facts: tuple[ExplicitProviderFacts, ...]) -> ModelRef:
    fact = _facts_for(ref.provider, facts)
    return normalize_model_ref(
        ref, known_ids=fact.normalization_ids if fact is not None else ()
    )


def _finish(
    ref: ModelRef,
    facts: tuple[ExplicitProviderFacts, ...],
    *,
    source: str,
    alias: str = "",
) -> ModelSelection:
    from models.selection import select_model

    ref = _normalized(ref, facts)
    candidate = SelectionCandidate(ref=ref, source=source)
    result = select_model(
        SelectionRequest(candidates=(candidate,), purpose="explicit", explicit=ref)
    )
    return replace(result, matched_alias=alias)


def explicit_provider_hint(
    raw_input: str,
    *,
    known_provider_ids: tuple[str, ...] = (),
    named_custom_provider_ids: tuple[str, ...] = (),
) -> str:
    """Provider qualified by raw input, for caller-owned fact acquisition only."""

    return parse_model_ref(
        raw_input,
        "",
        known_provider_ids=known_provider_ids,
        named_custom_provider_ids=named_custom_provider_ids,
    ).provider


def select_explicit_model(
    raw_input: str,
    current_provider: str,
    *,
    explicit_provider: str = "",
    provider_facts: tuple[ExplicitProviderFacts, ...] = (),
    direct_aliases: tuple[ExplicitAlias, ...] = (),
    configured_matches: tuple[ModelRef, ...] = (),
    fallback_providers: tuple[str, ...] = (),
    known_provider_ids: tuple[str, ...] = (),
    named_custom_provider_ids: tuple[str, ...] = (),
    detection: ExplicitDetectionFacts | None = None,
    block_provider_fallback: bool = False,
    hold_current_provider: bool = False,
    moa_ref: ModelRef | None = None,
) -> ModelSelection:
    """Interpret an explicit model request from already-materialized facts."""

    raw = str(raw_input or "").strip()
    current = normalize_provider(current_provider)
    explicit = normalize_provider(explicit_provider)

    if moa_ref is not None and not explicit:
        return _finish(moa_ref, provider_facts, source="moa")

    if not explicit:
        qualified = parse_model_ref(
            raw, "", known_provider_ids=known_provider_ids,
            named_custom_provider_ids=named_custom_provider_ids,
        )
        if qualified.provider:
            explicit, raw = qualified.provider, qualified.model

    if explicit:
        resolved = _alias_resolution(
            raw, explicit, provider_facts, direct_aliases, keep_provider=True
        )
        ref, alias = resolved if resolved is not None else (ModelRef(explicit, raw), "")
        return _finish(ref, provider_facts, source="explicit", alias=alias)

    resolved = _alias_resolution(raw, current, provider_facts, direct_aliases)
    if resolved is not None:
        ref, alias = resolved
        return _finish(ref, provider_facts, source="alias", alias=alias)

    if raw.lower() in MODEL_ALIASES:
        for provider in fallback_providers:
            if _providers_match(provider, current, provider_facts):
                continue
            fact = _facts_for(provider, provider_facts)
            if fact is None:
                continue
            resolved = _alias_resolution(raw, fact.provider, provider_facts, direct_aliases)
            if resolved is not None:
                ref, alias = resolved
                return _finish(ref, provider_facts, source="alias_fallback", alias=alias)
        raise ExplicitSelectionError("alias_unavailable", raw)

    current_fact = _facts_for(current, provider_facts)
    catalogued = _catalog_match(raw, current_fact)
    if catalogued is not None:
        return _finish(ModelRef(current, catalogued), provider_facts, source="catalog")

    if block_provider_fallback:
        return _finish(ModelRef(current, raw), provider_facts, source="current")

    if configured_matches:
        current_match = next(
            (
                ref for ref in configured_matches
                if _providers_match(ref.provider, current, provider_facts)
            ),
            None,
        )
        if current_match is not None:
            return _finish(
                ModelRef(current, current_match.model),
                provider_facts,
                source="configured",
            )
        if len(configured_matches) > 1:
            raise ExplicitSelectionError(
                "ambiguous_configured", raw,
                providers=tuple(sorted(ref.provider for ref in configured_matches)),
            )
        configured = configured_matches[0]
        return _finish(configured, provider_facts, source="configured")

    if hold_current_provider:
        return _finish(ModelRef(current, raw), provider_facts, source="current")

    detected = select_detected_model(raw, current, detection) if detection is not None else None
    if detected is not None:
        return _finish(detected, provider_facts, source="catalog_detection")
    return _finish(ModelRef(current, raw), provider_facts, source="current")
