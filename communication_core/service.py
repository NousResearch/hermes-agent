"""Application service for account-explicit communication workflows.

The service is intentionally consumed by the CLI and skills, not registered as
a model tool.  It owns orchestration and policy while adapters and repositories
remain independently testable.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, Iterable

from .adapters import (
    CommunicationAdapter,
    CommunicationOrchestrator,
    FacebookCommunicationAdapter,
    FakeCommunicationAdapter,
    NormalizedConversation,
    NormalizedEvent,
    NormalizedIdentity,
    NormalizedMessage,
    TelegramCommunicationAdapter,
    VKCommunicationAdapter,
)
from .errors import (
    AccountUnavailableError,
    CapabilityUnsupportedError,
    RouteDeniedError,
    ScopeViolationError,
)
from .repository import CommunicationRepository, stable_id, utc_now


class CommunicationService:
    """Coordinate adapters without ever selecting an implicit account."""

    def __init__(
        self,
        repository: CommunicationRepository | None = None,
        *,
        orchestrator: CommunicationOrchestrator | None = None,
        register_builtin_adapters: bool = True,
    ) -> None:
        self.repository = repository or CommunicationRepository()
        self.orchestrator = orchestrator or CommunicationOrchestrator(self.repository)
        if register_builtin_adapters:
            self._register_if_absent(FacebookCommunicationAdapter())
            self._register_if_absent(TelegramCommunicationAdapter())
            self._register_if_absent(VKCommunicationAdapter())

    def _register_if_absent(self, adapter: CommunicationAdapter) -> None:
        if adapter.provider not in self.orchestrator._adapters:
            self.orchestrator.register(adapter)

    def register_adapter(self, adapter: CommunicationAdapter) -> None:
        self.orchestrator.register(adapter)

    def initialize(self) -> dict[str, Any]:
        return {
            "database": str(self.repository.db_path),
            "schema_version": self.repository.initialize(),
        }

    def account_health(self, connected_account_id: str) -> dict[str, Any]:
        return self.orchestrator.health(connected_account_id)

    @staticmethod
    def _safe_issue_detail(error: BaseException) -> str:
        """Return a stable diagnostic without paths, payloads, or credentials."""
        return f"{type(error).__name__}: adapter operation failed"

    def _ensure_identity(
        self, connected_account_id: str, item: NormalizedIdentity
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        existing = self.repository.get_identity_by_external(
            connected_account_id, item.external_id
        )
        if existing and existing.get("person_id"):
            person_id = str(existing["person_id"])
        else:
            person_id = stable_id("person", connected_account_id, item.external_id)
            if self.repository.get_person(person_id) is None:
                self.repository.create_person(
                    item.display_name or f"Contact {item.external_id}",
                    person_id=person_id,
                )
        return self.repository.upsert_identity(
            connected_account_id=connected_account_id,
            external_id=item.external_id,
            display_name=item.display_name,
            profile_ref=item.profile_ref,
            person_id=person_id,
            provenance=item.provenance,
            observed_at=item.observed_at,
        )

    def sync(self, connected_account_id: str, *, mode: str = "incremental") -> dict[str, Any]:
        self.orchestrator.adapter_for(connected_account_id)
        with self.repository.account_sync_lock(connected_account_id):
            return self._sync_unlocked(connected_account_id, mode=mode)

    def _sync_unlocked(self, connected_account_id: str, *, mode: str) -> dict[str, Any]:
        if mode not in {"full", "incremental", "retry"}:
            raise ValueError("sync mode must be full, incremental, or retry")
        account, adapter = self.orchestrator.adapter_for(connected_account_id)
        run_id = self.repository.start_sync_run(connected_account_id, mode=mode)
        stats = {
            "contacts": 0,
            "conversations": 0,
            "messages_inserted": 0,
            "messages_observed": 0,
            "events": 0,
            "issues": 0,
        }
        resources: list[tuple[str, str, Any]] = [
            ("contacts", "contacts.read", adapter.sync_contacts),
            ("conversations", "conversations.read", adapter.sync_conversations),
            ("messages", "messages.read", adapter.sync_messages),
            ("events", "events.read", adapter.sync_events),
        ]
        try:
            for resource, capability, loader in resources:
                if not adapter.capabilities.supports(capability):
                    continue
                cursor = None if mode == "full" else self.repository.get_cursor(
                    connected_account_id, resource
                )
                try:
                    items = tuple(loader(account, cursor=cursor))
                    self._store_resource(connected_account_id, resource, items, stats)
                    next_cursor = self._next_cursor(items)
                    if next_cursor:
                        self.repository.set_cursor(
                            connected_account_id, resource, next_cursor
                        )
                except BaseException as error:
                    stats["issues"] += 1
                    self.repository.add_sync_issue(
                        run_id,
                        connected_account_id,
                        code=f"{resource.upper()}_SYNC_FAILED",
                        detail_redacted=self._safe_issue_detail(error),
                        retryable=True,
                    )
            status = "succeeded" if stats["issues"] == 0 else "partial"
            self.repository.finish_sync_run(run_id, status=status, stats=stats)
            return {"run_id": run_id, "status": status, "stats": stats}
        except BaseException:
            self.repository.finish_sync_run(run_id, status="failed", stats=stats)
            raise

    @staticmethod
    def _next_cursor(items: Iterable[Any]) -> str | None:
        values = [str(item.observed_at) for item in items if item.observed_at]
        return max(values) if values else None

    def _store_resource(
        self,
        connected_account_id: str,
        resource: str,
        items: Iterable[Any],
        stats: dict[str, int],
    ) -> None:
        if resource == "contacts":
            for item in items:
                assert isinstance(item, NormalizedIdentity)
                self._ensure_identity(connected_account_id, item)
                stats["contacts"] += 1
            return
        if resource == "conversations":
            for item in items:
                assert isinstance(item, NormalizedConversation)
                identity, endpoint = self._ensure_identity(
                    connected_account_id,
                    NormalizedIdentity(
                        external_id=item.identity_external_id,
                        observed_at=item.observed_at,
                        provenance=item.provenance,
                    ),
                )
                del identity
                self.repository.upsert_conversation(
                    connected_account_id=connected_account_id,
                    endpoint_id=endpoint["id"],
                    external_id=item.external_id,
                    kind=item.kind,
                    title=item.title,
                    provenance=item.provenance,
                    observed_at=item.observed_at or utc_now(),
                )
                stats["conversations"] += 1
            return
        if resource == "messages":
            for item in items:
                assert isinstance(item, NormalizedMessage)
                identity = self.repository.get_identity_by_external(
                    connected_account_id, item.identity_external_id
                )
                conversation = self.repository.get_conversation_by_external(
                    connected_account_id, item.conversation_external_id
                )
                if identity is None or conversation is None:
                    raise ScopeViolationError(
                        "message references an identity or conversation outside its account"
                    )
                endpoint = self.repository.get_endpoint(conversation["endpoint_id"])
                if endpoint is None or endpoint["platform_identity_id"] != identity["id"]:
                    raise ScopeViolationError(
                        "message identity does not match its account-scoped conversation"
                    )
                _, inserted = self.repository.upsert_message(
                    connected_account_id=connected_account_id,
                    endpoint_id=endpoint["id"],
                    conversation_id=conversation["id"],
                    external_id=item.external_id,
                    direction=item.direction,
                    body=item.body,
                    sent_at=item.sent_at,
                    sender_identity_id=identity["id"],
                    provenance=item.provenance,
                    observed_at=item.observed_at or utc_now(),
                )
                stats["messages_observed"] += 1
                stats["messages_inserted"] += int(inserted)
            return
        if resource == "events":
            for item in items:
                assert isinstance(item, NormalizedEvent)
                identity = self.repository.get_identity_by_external(
                    connected_account_id, item.identity_external_id
                )
                if identity is None or not identity.get("person_id"):
                    raise ScopeViolationError(
                        "event references an identity outside its account"
                    )
                with self.repository.read_connection() as connection:
                    endpoint = connection.execute(
                        """SELECT * FROM contact_endpoints
                           WHERE connected_account_id = ? AND platform_identity_id = ?""",
                        (connected_account_id, identity["id"]),
                    ).fetchone()
                assert endpoint is not None
                self.repository.add_contact_event(
                    person_id=identity["person_id"],
                    connected_account_id=connected_account_id,
                    endpoint_id=endpoint["id"],
                    event_type=item.event_type,
                    external_id=item.external_id,
                    happened_at=item.happened_at,
                    timezone=item.timezone,
                    data=item.data,
                    provenance=item.provenance,
                )
                stats["events"] += 1

    def route_dry_run(
        self,
        *,
        person_id: str,
        source_endpoint_id: str,
        target_endpoint_id: str,
        actor: str = "user",
    ) -> dict[str, Any]:
        source = self.repository.get_endpoint(source_endpoint_id)
        target = self.repository.get_endpoint(target_endpoint_id)
        allowed = True
        reasons: list[str] = []
        if source is None or target is None:
            allowed = False
            reasons.append("source or target endpoint does not exist")
        else:
            if source_endpoint_id == target_endpoint_id:
                allowed = False
                reasons.append("source and target endpoints are identical")
            if source["status"] != "active" or target["status"] != "active":
                allowed = False
                reasons.append("both endpoints must be active")
            if not self.repository.account_link_allowed(
                source["connected_account_id"], target["connected_account_id"]
            ):
                allowed = False
                reasons.append("directed account link is not explicitly allowed")
            with self.repository.read_connection() as connection:
                people = connection.execute(
                    """SELECT person_id FROM platform_identities WHERE id IN (?, ?)""",
                    (source["platform_identity_id"], target["platform_identity_id"]),
                ).fetchall()
            if len(people) != 2 or any(row[0] != person_id for row in people):
                allowed = False
                reasons.append("both endpoints must resolve to the selected person")
        explanation = "; ".join(reasons) if reasons else "explicit account policy and person route allow this target"
        if source is not None and target is not None:
            self.repository.audit_route(
                person_id=person_id,
                source_account_id=source["connected_account_id"],
                target_account_id=target["connected_account_id"],
                source_endpoint_id=source_endpoint_id,
                target_endpoint_id=target_endpoint_id,
                action="dry_run",
                allowed=allowed,
                explanation=explanation,
                actor=actor,
            )
        return {
            "allowed": allowed,
            "person_id": person_id,
            "source_endpoint_id": source_endpoint_id,
            "target_endpoint_id": target_endpoint_id,
            "explanation": explanation,
        }

    def apply_route(
        self,
        *,
        person_id: str,
        source_endpoint_id: str,
        target_endpoint_id: str,
        actor: str = "user",
        reason: str = "explicit route application after dry-run",
    ) -> dict[str, Any]:
        decision = self.route_dry_run(
            person_id=person_id,
            source_endpoint_id=source_endpoint_id,
            target_endpoint_id=target_endpoint_id,
            actor=actor,
        )
        if not decision["allowed"]:
            raise RouteDeniedError(decision["explanation"])
        route = self.repository.set_person_route(
            person_id=person_id,
            source_endpoint_id=source_endpoint_id,
            target_endpoint_id=target_endpoint_id,
            actor=actor,
            reason=reason,
        )
        return {**route, "route_version": self.repository.route_version(route)}

    def create_draft(
        self,
        *,
        person_id: str,
        source_endpoint_id: str,
        payload: str,
        actor: str = "user",
    ) -> dict[str, Any]:
        route = self.repository.get_route(person_id, source_endpoint_id)
        if route is None:
            raise RouteDeniedError("no enabled person route exists for source endpoint")
        decision = self.route_dry_run(
            person_id=person_id,
            source_endpoint_id=source_endpoint_id,
            target_endpoint_id=route["target_endpoint_id"],
            actor=actor,
        )
        if not decision["allowed"]:
            raise RouteDeniedError(decision["explanation"])
        source = self.repository.get_endpoint(source_endpoint_id)
        target = self.repository.get_endpoint(route["target_endpoint_id"])
        assert source is not None and target is not None
        with self.repository.read_connection() as connection:
            recipient = connection.execute(
                """SELECT i.external_id, i.display_name, a.provider
                   FROM platform_identities i
                   JOIN connected_accounts a ON a.id = i.observed_via_account_id
                   WHERE i.id = ?""",
                (target["platform_identity_id"],),
            ).fetchone()
        if recipient is None:
            raise ScopeViolationError("target identity is unavailable")
        preview = [{
            "person_id": person_id,
            "endpoint_id": target["id"],
            "connected_account_id": target["connected_account_id"],
            "provider": recipient["provider"],
            "external_id": recipient["external_id"],
            "display_name": recipient["display_name"],
        }]
        return self.repository.create_draft(
            person_id=person_id,
            source_account_id=source["connected_account_id"],
            source_endpoint_id=source_endpoint_id,
            target_account_id=target["connected_account_id"],
            endpoint_id=target["id"],
            route_version=self.repository.route_version(route),
            recipients=preview,
            payload=payload,
        )

    def approve_draft(
        self, draft_id: str, *, actor: str = "user", ttl_minutes: int = 30
    ) -> dict[str, Any]:
        if ttl_minutes < 1 or ttl_minutes > 1440:
            raise ValueError("approval TTL must be between 1 and 1440 minutes")
        expires_at = (
            datetime.now(UTC) + timedelta(minutes=ttl_minutes)
        ).isoformat(timespec="microseconds").replace("+00:00", "Z")
        return self.repository.approve_draft(
            draft_id, actor=actor, expires_at=expires_at
        )

    def enqueue(self, approval_id: str, *, idempotency_key: str | None = None) -> dict[str, Any]:
        return self.repository.enqueue_approved(
            approval_id,
            idempotency_key=idempotency_key or f"communication:{uuid.uuid4().hex}",
        )

    def execute_test_sink(self, outbox_id: str) -> dict[str, Any]:
        """Execute exactly one fake-provider item; production providers fail closed."""
        item = self.repository.get_outbox(outbox_id)
        if item is None:
            raise KeyError(outbox_id)
        account, adapter = self.orchestrator.adapter_for(item["connected_account_id"])
        if account["provider"] != "fake" or not isinstance(adapter, FakeCommunicationAdapter):
            raise CapabilityUnsupportedError(
                "outbox execution is restricted to the fake test sink"
            )
        claim_token = f"claim:{uuid.uuid4().hex}"
        expires_at = (
            datetime.now(UTC) + timedelta(minutes=5)
        ).isoformat(timespec="microseconds").replace("+00:00", "Z")
        claimed = self.repository.claim_outbox(
            outbox_id, claim_token=claim_token, expires_at=expires_at
        )
        if claimed is None:
            raise AccountUnavailableError("outbox item is not pending")
        endpoint = self.repository.get_endpoint(claimed["endpoint_id"])
        assert endpoint is not None
        try:
            evidence = adapter.send_approved(
                account,
                endpoint=endpoint,
                payload=claimed["payload"],
                idempotency_key=claimed["idempotency_key"],
            )
            self.repository.complete_outbox(
                outbox_id,
                claim_token=claim_token,
                status="sent",
                evidence=evidence,
            )
        except BaseException as error:
            self.repository.complete_outbox(
                outbox_id,
                claim_token=claim_token,
                status="uncertain",
                evidence={"observed": False},
                error_redacted=self._safe_issue_detail(error),
            )
            raise
        return self.repository.get_outbox(outbox_id) or {}
