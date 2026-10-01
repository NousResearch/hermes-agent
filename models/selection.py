"""Pure provider/model selection over a caller-supplied canonical universe."""

from __future__ import annotations

from dataclasses import replace

from models.identity import ModelRef
from models.metadata.types import ModelMetadata
from models.selection_types import (
    CapabilityRequirements,
    CandidateRejection,
    ModelSelection,
    SelectionCandidate,
    SelectionConstraints,
    SelectionPolicy,
    SelectionReason,
    SelectionRequest,
)
from providers.routing import InvocationRequest, resolve_invocation_route


def build_selection_candidate(
    ref: ModelRef,
    metadata: ModelMetadata | None = None,
    route_request: InvocationRequest | None = None,
    *,
    catalogued: bool = False,
    source: str = "",
    stable_order: int = 0,
) -> SelectionCandidate:
    """Build one candidate from already-known facts.

    A route is optional because identity selection occurs before credentials and
    runtime route context are available on some callers.
    """

    route = None
    if route_request is not None:
        route = resolve_invocation_route(
            replace(route_request, provider=ref.provider, model=ref.model)
        )
    return SelectionCandidate(ref, metadata, route, catalogued, source, stable_order)


def _rejections(
    candidate: SelectionCandidate, constraints: SelectionConstraints
) -> tuple[str, ...]:
    reasons: list[str] = []
    if constraints.allowed_providers and candidate.ref.provider not in constraints.allowed_providers:
        reasons.append("provider_not_allowed")
    if constraints.allowed_models and candidate.ref not in constraints.allowed_models:
        reasons.append("model_not_allowed")
    if candidate.ref in constraints.excluded_models:
        reasons.append("model_excluded")
    if constraints.require_catalogued and not candidate.catalogued:
        reasons.append("not_catalogued")

    route = candidate.route
    if constraints.allowed_api_modes and (
        route is None or route.api_mode not in constraints.allowed_api_modes
    ):
        reasons.append("api_mode_not_allowed")
    if constraints.allowed_runtime_kinds and (
        route is None or route.runtime_kind not in constraints.allowed_runtime_kinds
    ):
        reasons.append("runtime_kind_not_allowed")

    metadata, req = candidate.metadata, constraints.capabilities
    for name, required in (
        ("tools", req.tools),
        ("vision", req.vision),
        ("reasoning", req.reasoning),
        ("structured_output", req.structured_output),
    ):
        if required is None:
            continue
        actual = getattr(metadata, f"supports_{name}", None) if metadata is not None else None
        if actual is not required:
            reasons.append(f"requires_{name}={required}")
    if req.minimum_context_window is not None and (
        metadata is None
        or metadata.context_window is None
        or metadata.context_window < req.minimum_context_window
    ):
        reasons.append(f"minimum_context_window={req.minimum_context_window}")
    return tuple(reasons)


def _index(value, values: tuple) -> int:
    try:
        return values.index(value)
    except ValueError:
        return len(values)


def _rank(candidate: SelectionCandidate, policy: SelectionPolicy) -> tuple:
    catalog_rank = (
        0 if policy.prefer_catalogued and candidate.catalogued
        else int(policy.prefer_catalogued)
    )
    return (
        _index(candidate.ref, policy.preferred),
        _index(candidate.ref.provider, policy.provider_order),
        catalog_rank,
        candidate.stable_order,
        candidate.ref.provider,
        candidate.ref.model,
    )


def select_model(request: SelectionRequest) -> ModelSelection:
    """Apply hard constraints, then deterministic policy preferences."""

    accepted: list[SelectionCandidate] = []
    rejected: list[CandidateRejection] = []
    for candidate in request.candidates:
        reasons = _rejections(candidate, request.constraints)
        if reasons:
            rejected.append(CandidateRejection(candidate.ref, reasons))
        else:
            accepted.append(candidate)

    if request.explicit is not None:
        selected = next((item for item in accepted if item.ref == request.explicit), None)
        if selected is None:
            if not any(item.ref == request.explicit for item in rejected):
                rejected.append(CandidateRejection(request.explicit, ("not_in_universe",)))
            return ModelSelection(
                None, (), tuple(rejected), SelectionReason.EXPLICIT_UNAVAILABLE,
                request.purpose, request.policy.name,
            )
        return ModelSelection(
            selected, (selected,), tuple(rejected), SelectionReason.EXPLICIT_MATCH,
            request.purpose, request.policy.name,
        )

    eligible = tuple(sorted(accepted, key=lambda item: _rank(item, request.policy)))
    if not eligible:
        return ModelSelection(
            None, (), tuple(rejected), SelectionReason.NO_ELIGIBLE_CANDIDATE,
            request.purpose, request.policy.name,
        )
    selected = eligible[0]
    reason = (
        SelectionReason.PREFERRED_MATCH
        if selected.ref in request.policy.preferred
        else SelectionReason.POLICY_DEFAULT
    )
    return ModelSelection(
        selected, eligible, tuple(rejected), reason, request.purpose, request.policy.name
    )


from models.selection_auxiliary import (  # noqa: E402
    auxiliary_task_prefers_fast_model,
    select_auxiliary_fallback_model,
    select_auxiliary_model,
    select_fast_auxiliary_model,
    select_vision_auxiliary_model,
    selected_auxiliary_model_id,
)
from models.selection_detection import ExplicitDetectionFacts, select_detected_model  # noqa: E402
from models.selection_defaults import select_default_model, select_nous_default_model  # noqa: E402
from models.selection_picker import list_picker_candidates, picker_model_ids  # noqa: E402

from models.selection_explicit import (  # noqa: E402
    ExplicitAlias,
    ExplicitProviderFacts,
    ExplicitSelectionError,
    explicit_provider_hint,
    select_explicit_model,
)


__all__ = [
    "CapabilityRequirements",
    "CandidateRejection",
    "ExplicitAlias",
    "ExplicitDetectionFacts",
    "ExplicitProviderFacts",
    "ExplicitSelectionError",
    "explicit_provider_hint",
    "ModelSelection",
    "SelectionCandidate",
    "SelectionConstraints",
    "SelectionPolicy",
    "SelectionReason",
    "SelectionRequest",
    "build_selection_candidate",
    "select_auxiliary_fallback_model",
    "auxiliary_task_prefers_fast_model",
    "select_auxiliary_model",
    "select_fast_auxiliary_model",
    "select_vision_auxiliary_model",
    "selected_auxiliary_model_id",
    "select_default_model",
    "select_nous_default_model",
    "select_detected_model",
    "select_explicit_model",
    "list_picker_candidates",
    "picker_model_ids",
    "select_model",
]
