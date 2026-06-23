"""Pure routing and orchestration for durable knowledge writes."""

from __future__ import annotations

from dataclasses import dataclass

from knowledge.adapters.base import ExistingKnowledge, KnowledgeAdapter
from knowledge.policy import RoutePolicy
from knowledge.types import (
    DuplicatePolicy,
    KnowledgeDestination,
    KnowledgeWriteRequest,
    KnowledgeWriteResult,
    RouteAction,
    RouteDecision,
)

MAX_KNOWLEDGE_CONTENT_CHARS = 120_000


def route_knowledge_write(
    request: KnowledgeWriteRequest,
    policy: RoutePolicy | None = None,
) -> RouteDecision:
    if request.destination_override:
        destination = request.destination_override
        if destination is KnowledgeDestination.STATIC_KNOWLEDGE:
            return RouteDecision(
                destination=destination,
                action=RouteAction.WRITE_DOCUMENT,
                reason="destination_override",
                requires_title=True,
                snapshot=True,
                content_type=request.content_type,
                backend=(policy or RoutePolicy()).default_static_backend,
            )
        if destination is KnowledgeDestination.DYNAMIC_MEMORY:
            return RouteDecision(
                destination=destination,
                action=RouteAction.RETAIN_MEMORY,
                reason="destination_override",
                content_type=request.content_type,
                backend=(policy or RoutePolicy()).default_dynamic_backend,
            )
        return RouteDecision(
            destination=destination,
            action=RouteAction.DRY_RUN_ONLY if destination is KnowledgeDestination.REVIEW else RouteAction.SKIP,
            reason="destination_override",
            content_type=request.content_type,
        )
    return (policy or RoutePolicy()).decide(request.content_type)


@dataclass
class KnowledgeRouter:
    static_adapter: KnowledgeAdapter
    dynamic_adapter: KnowledgeAdapter
    policy: RoutePolicy = RoutePolicy()

    def write(self, request: KnowledgeWriteRequest) -> KnowledgeWriteResult:
        if not request.content.strip():
            return KnowledgeWriteResult(
                success=False,
                destination=KnowledgeDestination.REVIEW,
                action="validate",
                error="content is required",
            )
        if len(request.content) > MAX_KNOWLEDGE_CONTENT_CHARS:
            return KnowledgeWriteResult(
                success=False,
                destination=KnowledgeDestination.REVIEW,
                action="validate",
                error=f"content exceeds {MAX_KNOWLEDGE_CONTENT_CHARS} characters",
            )

        decision = route_knowledge_write(request, self.policy)
        if request.dry_run or decision.destination in {KnowledgeDestination.NONE, KnowledgeDestination.REVIEW}:
            return KnowledgeWriteResult(
                success=True,
                destination=decision.destination,
                action=str(decision.action),
                written=False,
                dry_run=True,
                decision=decision,
            )

        if decision.requires_title and not request.title.strip():
            return KnowledgeWriteResult(
                success=False,
                destination=decision.destination,
                action="validate",
                written=False,
                decision=decision,
                error="title is required for static knowledge writes",
            )

        if decision.destination is KnowledgeDestination.STATIC_KNOWLEDGE:
            return self._write_static(request, decision)
        if decision.destination is KnowledgeDestination.DYNAMIC_MEMORY:
            return self._write_dynamic(request, decision)
        return KnowledgeWriteResult(
            success=False,
            destination=decision.destination,
            action=str(decision.action),
            decision=decision,
            error=f"Unsupported destination: {decision.destination}",
        )

    def _write_static(self, request: KnowledgeWriteRequest, decision: RouteDecision) -> KnowledgeWriteResult:
        existing: list[ExistingKnowledge] = []
        if request.duplicate_policy is not DuplicatePolicy.CREATE_NEW:
            existing = self.static_adapter.search_existing(request)

        if existing:
            if request.duplicate_policy is DuplicatePolicy.SKIP_EXISTING:
                return KnowledgeWriteResult(
                    success=True,
                    destination=decision.destination,
                    action="skip_existing",
                    written=False,
                    decision=decision,
                    existing=[item.to_dict() for item in existing],
                )
            if request.duplicate_policy is DuplicatePolicy.REVIEW:
                return KnowledgeWriteResult(
                    success=True,
                    destination=KnowledgeDestination.REVIEW,
                    action="review_existing",
                    written=False,
                    dry_run=True,
                    decision=decision,
                    existing=[item.to_dict() for item in existing],
                )
            if request.duplicate_policy is DuplicatePolicy.UPDATE_EXISTING:
                wr = self.static_adapter.update(existing[0], request)
                return KnowledgeWriteResult(
                    success=wr.success,
                    destination=decision.destination,
                    action=wr.action,
                    written=wr.success,
                    id=wr.id,
                    path=wr.path,
                    backend=wr.backend,
                    decision=decision,
                    existing=[item.to_dict() for item in existing],
                    error=wr.error,
                    result=wr.data,
                )

        wr = self.static_adapter.write(request)
        return KnowledgeWriteResult(
            success=wr.success,
            destination=decision.destination,
            action=wr.action,
            written=wr.success,
            id=wr.id,
            path=wr.path,
            backend=wr.backend,
            decision=decision,
            error=wr.error,
            result=wr.data,
        )

    def _write_dynamic(self, request: KnowledgeWriteRequest, decision: RouteDecision) -> KnowledgeWriteResult:
        tags = tuple(dict.fromkeys((*request.tags, "knowledge-router", "hindsight")))
        routed = KnowledgeWriteRequest(
            content_type=request.content_type,
            content=request.content,
            title=request.title,
            context=request.context,
            tags=tags,
            dry_run=request.dry_run,
            destination_override=request.destination_override,
            idempotency_key=request.idempotency_key,
            update_mode=request.update_mode,
            duplicate_policy=request.duplicate_policy,
            metadata=request.metadata,
            notebook=request.notebook,
            path=request.path,
        )
        wr = self.dynamic_adapter.write(routed)
        return KnowledgeWriteResult(
            success=wr.success,
            destination=decision.destination,
            action=wr.action,
            written=wr.success,
            id=wr.id,
            path=wr.path,
            backend=wr.backend,
            decision=decision,
            error=wr.error,
            result=wr.data,
        )
