"""Shared, account-explicit adapter contract and orchestrator."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Protocol

from .errors import (
    AccountRequiredError,
    AccountUnavailableError,
    CapabilityUnsupportedError,
    ScopeViolationError,
)


@dataclass(frozen=True)
class AdapterCapabilities:
    contacts_read: bool = False
    profiles_read: bool = False
    conversations_read: bool = False
    messages_read: bool = False
    groups_read: bool = False
    events_read: bool = False
    receipts_read: bool = False
    messages_send: bool = False

    def names(self) -> tuple[str, ...]:
        mapping = {
            "contacts.read": self.contacts_read,
            "profiles.read": self.profiles_read,
            "conversations.read": self.conversations_read,
            "messages.read": self.messages_read,
            "groups.read": self.groups_read,
            "events.read": self.events_read,
            "receipts.read": self.receipts_read,
            "messages.send": self.messages_send,
        }
        return tuple(name for name, enabled in mapping.items() if enabled)

    def supports(self, capability: str) -> bool:
        return capability in self.names()


@dataclass(frozen=True)
class NormalizedIdentity:
    external_id: str
    display_name: str | None = None
    profile_ref: str | None = None
    observed_at: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NormalizedConversation:
    external_id: str
    identity_external_id: str
    kind: str = "direct"
    title: str | None = None
    observed_at: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NormalizedMessage:
    external_id: str | None
    conversation_external_id: str
    identity_external_id: str
    direction: str
    body: str
    sent_at: str
    observed_at: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NormalizedEvent:
    external_id: str | None
    identity_external_id: str
    event_type: str
    happened_at: str
    observed_at: str | None = None
    timezone: str = "UTC"
    data: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NormalizedProfile:
    external_id: str
    display_name: str | None = None
    profile_ref: str | None = None
    observed_at: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NormalizedGroup:
    external_id: str
    title: str | None = None
    member_external_ids: tuple[str, ...] = ()
    observed_at: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NormalizedReceipt:
    external_id: str
    message_external_id: str
    status: str
    observed_at: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)


class CommunicationAdapter(ABC):
    """Read adapter contract; every call receives one exact account record."""

    provider: str
    capabilities: AdapterCapabilities

    def _require_account(self, account: dict[str, Any]) -> None:
        account_id = str(account.get("id") or "").strip()
        if not account_id:
            raise AccountRequiredError("connected_account_id is required")
        if account.get("provider") != self.provider:
            raise ScopeViolationError(
                f"adapter provider {self.provider!r} cannot use account provider "
                f"{account.get('provider')!r}"
            )
        if not bool(account.get("enabled", 0)):
            raise AccountUnavailableError(f"connected account {account_id} is disabled")

    def _require_capability(self, capability: str) -> None:
        if not self.capabilities.supports(capability):
            raise CapabilityUnsupportedError(
                f"{self.provider} adapter does not support {capability}"
            )

    @abstractmethod
    def health(self, account: dict[str, Any]) -> dict[str, Any]:
        """Return redacted health and auth state for an exact account."""

    def sync_contacts(
        self, account: dict[str, Any], *, cursor: str | None = None
    ) -> Iterable[NormalizedIdentity]:
        self._require_account(account)
        self._require_capability("contacts.read")
        raise CapabilityUnsupportedError("contacts.read is declared but not implemented")

    def sync_profiles(
        self, account: dict[str, Any], *, cursor: str | None = None
    ) -> Iterable[NormalizedProfile]:
        self._require_account(account)
        self._require_capability("profiles.read")
        raise CapabilityUnsupportedError("profiles.read is declared but not implemented")

    def sync_conversations(
        self, account: dict[str, Any], *, cursor: str | None = None
    ) -> Iterable[NormalizedConversation]:
        self._require_account(account)
        self._require_capability("conversations.read")
        raise CapabilityUnsupportedError("conversations.read is declared but not implemented")

    def sync_messages(
        self, account: dict[str, Any], *, cursor: str | None = None
    ) -> Iterable[NormalizedMessage]:
        self._require_account(account)
        self._require_capability("messages.read")
        raise CapabilityUnsupportedError("messages.read is declared but not implemented")

    def sync_events(
        self, account: dict[str, Any], *, cursor: str | None = None
    ) -> Iterable[NormalizedEvent]:
        self._require_account(account)
        self._require_capability("events.read")
        raise CapabilityUnsupportedError("events.read is declared but not implemented")

    def sync_groups(
        self, account: dict[str, Any], *, cursor: str | None = None
    ) -> Iterable[NormalizedGroup]:
        self._require_account(account)
        self._require_capability("groups.read")
        raise CapabilityUnsupportedError("groups.read is declared but not implemented")

    def sync_receipts(
        self, account: dict[str, Any], *, cursor: str | None = None
    ) -> Iterable[NormalizedReceipt]:
        self._require_account(account)
        self._require_capability("receipts.read")
        raise CapabilityUnsupportedError("receipts.read is declared but not implemented")

    def send_approved(
        self,
        account: dict[str, Any],
        *,
        endpoint: dict[str, Any],
        payload: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        self._require_account(account)
        self._require_capability("messages.send")
        raise CapabilityUnsupportedError("production writes are not implemented")


class CommunicationReadConnector(Protocol):
    """Injected API/test-server connector; every call remains account-explicit."""

    def health(self, account: dict[str, Any]) -> dict[str, Any]: ...
    def sync_contacts(self, account: dict[str, Any], *, cursor: str | None = None) -> Iterable[NormalizedIdentity]: ...
    def sync_profiles(self, account: dict[str, Any], *, cursor: str | None = None) -> Iterable[NormalizedProfile]: ...
    def sync_conversations(self, account: dict[str, Any], *, cursor: str | None = None) -> Iterable[NormalizedConversation]: ...
    def sync_messages(self, account: dict[str, Any], *, cursor: str | None = None) -> Iterable[NormalizedMessage]: ...
    def sync_events(self, account: dict[str, Any], *, cursor: str | None = None) -> Iterable[NormalizedEvent]: ...
    def sync_groups(self, account: dict[str, Any], *, cursor: str | None = None) -> Iterable[NormalizedGroup]: ...
    def sync_receipts(self, account: dict[str, Any], *, cursor: str | None = None) -> Iterable[NormalizedReceipt]: ...


class FixtureReadAdapter(CommunicationAdapter):
    """Deterministic read-only adapter used by fixtures and test servers."""

    def __init__(
        self,
        provider: str,
        *,
        contacts: Iterable[NormalizedIdentity] = (),
        profiles: Iterable[NormalizedProfile] = (),
        conversations: Iterable[NormalizedConversation] = (),
        messages: Iterable[NormalizedMessage] = (),
        events: Iterable[NormalizedEvent] = (),
        groups: Iterable[NormalizedGroup] = (),
        receipts: Iterable[NormalizedReceipt] = (),
        capabilities: AdapterCapabilities | None = None,
    ) -> None:
        self.provider = provider
        self.capabilities = capabilities or AdapterCapabilities(
            contacts_read=True,
            profiles_read=True,
            conversations_read=True,
            messages_read=True,
            events_read=True,
        )
        self._contacts = tuple(contacts)
        self._profiles = tuple(profiles)
        self._conversations = tuple(conversations)
        self._messages = tuple(messages)
        self._events = tuple(events)
        self._groups = tuple(groups)
        self._receipts = tuple(receipts)

    def health(self, account: dict[str, Any]) -> dict[str, Any]:
        self._require_account(account)
        return {
            "connected_account_id": account["id"],
            "provider": self.provider,
            "auth_status": account.get("auth_status", "unknown"),
            "health_status": "healthy",
            "capabilities": list(self.capabilities.names()),
            "fixture": True,
        }

    def sync_contacts(self, account: dict[str, Any], *, cursor=None):
        self._require_account(account)
        self._require_capability("contacts.read")
        return iter(self._contacts)

    def sync_profiles(self, account: dict[str, Any], *, cursor=None):
        self._require_account(account)
        self._require_capability("profiles.read")
        return iter(self._profiles)

    def sync_conversations(self, account: dict[str, Any], *, cursor=None):
        self._require_account(account)
        self._require_capability("conversations.read")
        return iter(self._conversations)

    def sync_messages(self, account: dict[str, Any], *, cursor=None):
        self._require_account(account)
        self._require_capability("messages.read")
        return iter(self._messages)

    def sync_events(self, account: dict[str, Any], *, cursor=None):
        self._require_account(account)
        self._require_capability("events.read")
        return iter(self._events)

    def sync_groups(self, account: dict[str, Any], *, cursor=None):
        self._require_account(account)
        self._require_capability("groups.read")
        return iter(self._groups)

    def sync_receipts(self, account: dict[str, Any], *, cursor=None):
        self._require_account(account)
        self._require_capability("receipts.read")
        return iter(self._receipts)


class FacebookCommunicationAdapter(FixtureReadAdapter):
    """Facebook read contract backed by the existing canonical CRM repository.

    Browser/E2EE synchronization remains owned by ``facebook_core``.  This
    adapter is the migration bridge that exposes its verified local records to
    Communication Core without creating a second Facebook browser stack.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        super().__init__(
            "facebook",
            capabilities=AdapterCapabilities(
                contacts_read=True,
                profiles_read=True,
                conversations_read=True,
                messages_read=True,
                events_read=True,
            ),
        )
        self.db_path = Path(db_path) if db_path is not None else None

    def _repository(self):
        from universal_browser_manager.facebook_core.repository import FacebookRepository

        return FacebookRepository(self.db_path) if self.db_path else FacebookRepository()

    def health(self, account: dict[str, Any]) -> dict[str, Any]:
        self._require_account(account)
        repository = self._repository()
        exists = repository.db_path.is_file()
        return {
            "connected_account_id": account["id"],
            "provider": self.provider,
            "auth_status": account.get("auth_status", "unknown"),
            "health_status": "healthy" if exists else "failed",
            "database_present": exists,
            "capabilities": list(self.capabilities.names()),
            "write_actions_enabled": False,
        }

    def sync_contacts(self, account: dict[str, Any], *, cursor=None):
        self._require_account(account)
        self._require_capability("contacts.read")
        for row in self._repository().list_friends(limit=500):
            external_id = row.get("canonical_key") or row.get("profile_url") or str(row["id"])
            yield NormalizedIdentity(
                external_id=str(external_id),
                display_name=row.get("name"),
                profile_ref=row.get("profile_url"),
                provenance={"legacy_crm_id": str(row["id"]), "verified": True},
            )

    def sync_profiles(self, account: dict[str, Any], *, cursor=None):
        self._require_account(account)
        self._require_capability("profiles.read")
        for contact in self.sync_contacts(account, cursor=cursor):
            yield NormalizedProfile(
                external_id=contact.external_id,
                display_name=contact.display_name,
                profile_ref=contact.profile_ref,
                observed_at=contact.observed_at,
                provenance=contact.provenance,
            )

    def _read_rows(self, sql: str) -> list[sqlite3.Row]:
        path = self._repository().db_path
        if not path.is_file():
            raise AccountUnavailableError("Facebook CRM database is unavailable")
        connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA query_only = ON")
            return connection.execute(sql).fetchall()
        finally:
            connection.close()

    @staticmethod
    def _friend_external(row: sqlite3.Row) -> str:
        return str(row["canonical_key"] or row["profile_url"] or row["friend_id"])

    @staticmethod
    def _conversation_external(row: sqlite3.Row) -> str:
        return str(row["thread_url"] or f"friend:{row['friend_id']}")

    def sync_conversations(self, account: dict[str, Any], *, cursor=None):
        self._require_account(account)
        self._require_capability("conversations.read")
        rows = self._read_rows(
            """SELECT id AS friend_id, canonical_key, profile_url, thread_url,
                      name, updated_at FROM friends ORDER BY id"""
        )
        for row in rows:
            observed = row["updated_at"]
            if cursor and observed and observed <= cursor:
                continue
            yield NormalizedConversation(
                external_id=self._conversation_external(row),
                identity_external_id=self._friend_external(row),
                kind="direct",
                title=row["name"],
                observed_at=observed,
                provenance={"legacy_crm_id": str(row["friend_id"]), "verified": True},
            )

    def sync_messages(self, account: dict[str, Any], *, cursor=None):
        self._require_account(account)
        self._require_capability("messages.read")
        rows = self._read_rows(
            """SELECT m.id, m.message_key, m.message_id,
                      COALESCE(m.message_text, m.text, '') AS body,
                      COALESCE(m.sent_at, m.timestamp, m.created_at) AS sent_at,
                      m.direction, m.source_system, m.source_record_id,
                      f.id AS friend_id, f.canonical_key, f.profile_url, f.thread_url
               FROM messages m JOIN friends f ON f.id = m.friend_id
               ORDER BY m.friend_id, sent_at, m.id"""
        )
        for row in rows:
            if cursor and row["sent_at"] and row["sent_at"] <= cursor:
                continue
            yield NormalizedMessage(
                external_id=str(row["message_id"] or row["message_key"] or row["id"]),
                conversation_external_id=self._conversation_external(row),
                identity_external_id=self._friend_external(row),
                direction={"sent": "outgoing", "received": "incoming"}.get(row["direction"], "system"),
                body=row["body"],
                sent_at=row["sent_at"],
                observed_at=row["sent_at"],
                provenance={
                    "legacy_crm_id": str(row["id"]),
                    "legacy_source_system": row["source_system"],
                    "legacy_source_record_id": row["source_record_id"],
                    "verified": True,
                },
            )

    def sync_events(self, account: dict[str, Any], *, cursor=None):
        self._require_account(account)
        self._require_capability("events.read")
        rows = self._read_rows(
            """SELECT x.id, x.interaction_key, x.type, x.details,
                      x.interacted_at, x.source_system, x.source_record_id,
                      f.id AS friend_id, f.canonical_key, f.profile_url, f.thread_url
               FROM interactions x JOIN friends f ON f.id = x.friend_id
               ORDER BY x.friend_id, x.interacted_at, x.id"""
        )
        for row in rows:
            if cursor and row["interacted_at"] and row["interacted_at"] <= cursor:
                continue
            yield NormalizedEvent(
                external_id=str(row["interaction_key"] or row["id"]),
                identity_external_id=self._friend_external(row),
                event_type=f"facebook.{row['type']}",
                happened_at=row["interacted_at"],
                observed_at=row["interacted_at"],
                data={"details": row["details"]},
                provenance={
                    "legacy_crm_id": str(row["id"]),
                    "legacy_source_system": row["source_system"],
                    "legacy_source_record_id": row["source_record_id"],
                    "verified": True,
                },
            )


class ConnectorReadAdapter(FixtureReadAdapter):
    """Fixture-compatible adapter that can delegate to a real/test connector."""

    def __init__(self, provider: str, *, connector: CommunicationReadConnector | None = None, **fixtures: Any) -> None:
        self._configured = connector is not None or bool(fixtures)
        if not self._configured:
            fixtures["capabilities"] = AdapterCapabilities()
        super().__init__(provider, **fixtures)
        self.connector = connector

    def health(self, account: dict[str, Any]) -> dict[str, Any]:
        self._require_account(account)
        if self.connector:
            return self.connector.health(account)
        if not self._configured:
            return {
                "connected_account_id": account["id"],
                "provider": self.provider,
                "auth_status": account.get("auth_status", "unknown"),
                "health_status": "failed",
                "capabilities": [],
                "reason": "read connector is not configured",
            }
        return super().health(account)

    def sync_contacts(self, account: dict[str, Any], *, cursor=None):
        self._require_account(account)
        return self.connector.sync_contacts(account, cursor=cursor) if self.connector else super().sync_contacts(account, cursor=cursor)

    def sync_profiles(self, account: dict[str, Any], *, cursor=None):
        self._require_account(account)
        if self.connector and hasattr(self.connector, "sync_profiles"):
            return self.connector.sync_profiles(account, cursor=cursor)
        return super().sync_profiles(account, cursor=cursor)

    def sync_conversations(self, account: dict[str, Any], *, cursor=None):
        self._require_account(account)
        return self.connector.sync_conversations(account, cursor=cursor) if self.connector else super().sync_conversations(account, cursor=cursor)

    def sync_messages(self, account: dict[str, Any], *, cursor=None):
        self._require_account(account)
        return self.connector.sync_messages(account, cursor=cursor) if self.connector else super().sync_messages(account, cursor=cursor)

    def sync_events(self, account: dict[str, Any], *, cursor=None):
        self._require_account(account)
        return self.connector.sync_events(account, cursor=cursor) if self.connector else super().sync_events(account, cursor=cursor)

    def sync_groups(self, account: dict[str, Any], *, cursor=None):
        self._require_account(account)
        if self.connector and hasattr(self.connector, "sync_groups"):
            return self.connector.sync_groups(account, cursor=cursor)
        return super().sync_groups(account, cursor=cursor)

    def sync_receipts(self, account: dict[str, Any], *, cursor=None):
        self._require_account(account)
        if self.connector and hasattr(self.connector, "sync_receipts"):
            return self.connector.sync_receipts(account, cursor=cursor)
        return super().sync_receipts(account, cursor=cursor)


class TelegramCommunicationAdapter(ConnectorReadAdapter):
    """Telegram personal/group communication adapter, separate from News ingest."""

    def __init__(self, **fixtures: Any) -> None:
        super().__init__("telegram", **fixtures)


class VKCommunicationAdapter(ConnectorReadAdapter):
    """Account-scoped VK read adapter; live writes are intentionally absent."""

    def __init__(self, **fixtures: Any) -> None:
        super().__init__("vk", **fixtures)


class DatingCommunicationAdapter(FixtureReadAdapter):
    """Read-only adapter for an explicitly named dating pilot.

    ``provider`` must be the user-approved site identifier; constructing a
    generic or unnamed pilot is rejected.
    """

    def __init__(self, provider: str, *, pilot_confirmed: bool, **fixtures: Any) -> None:
        normalized = provider.strip().lower()
        if not pilot_confirmed or not normalized or normalized in {"dating", "generic"}:
            raise ValueError("dating pilot provider must be explicitly user-confirmed")
        super().__init__(normalized, **fixtures)


class FakeCommunicationAdapter(FixtureReadAdapter):
    """Only adapter allowed to execute the outbox during this goal."""

    def __init__(self, **fixtures: Any) -> None:
        capabilities = fixtures.pop(
            "capabilities",
            AdapterCapabilities(
                contacts_read=True,
                profiles_read=True,
                conversations_read=True,
                messages_read=True,
                groups_read=True,
                events_read=True,
                receipts_read=True,
                messages_send=True,
            ),
        )
        super().__init__("fake", capabilities=capabilities, **fixtures)
        self.deliveries: dict[str, dict[str, Any]] = {}

    def send_approved(
        self,
        account: dict[str, Any],
        *,
        endpoint: dict[str, Any],
        payload: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        self._require_account(account)
        self._require_capability("messages.send")
        delivery = self.deliveries.get(idempotency_key)
        if delivery is None:
            delivery = {
                "observed": True,
                "delivery_id": f"fake:{len(self.deliveries) + 1}",
                "connected_account_id": account["id"],
                "endpoint_id": endpoint["id"],
                "payload": payload,
            }
            self.deliveries[idempotency_key] = delivery
        return dict(delivery)


class CommunicationOrchestrator:
    """Resolve an adapter only after an explicit account lookup."""

    def __init__(self, repository: Any) -> None:
        self.repository = repository
        self._adapters: dict[str, CommunicationAdapter] = {}

    def register(self, adapter: CommunicationAdapter) -> None:
        if adapter.provider in self._adapters:
            raise ValueError(f"adapter already registered for {adapter.provider}")
        self._adapters[adapter.provider] = adapter

    def adapter_for(self, connected_account_id: str) -> tuple[dict[str, Any], CommunicationAdapter]:
        if not str(connected_account_id or "").strip():
            raise AccountRequiredError("connected_account_id is required")
        account = self.repository.get_account(connected_account_id)
        if account is None:
            raise AccountUnavailableError(
                f"connected account {connected_account_id!r} does not exist"
            )
        if not bool(account["enabled"]):
            raise AccountUnavailableError(
                f"connected account {connected_account_id!r} is disabled"
            )
        adapter = self._adapters.get(account["provider"])
        if adapter is None:
            raise CapabilityUnsupportedError(
                f"no adapter registered for provider {account['provider']!r}"
            )
        return account, adapter

    def health(self, connected_account_id: str) -> dict[str, Any]:
        account, adapter = self.adapter_for(connected_account_id)
        return adapter.health(account)

    def capabilities(self, connected_account_id: str) -> tuple[str, ...]:
        _, adapter = self.adapter_for(connected_account_id)
        return adapter.capabilities.names()
