"""SQLite repositories enforcing Communication Core ownership boundaries."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterator, Sequence

from hermes_constants import get_hermes_home

from .errors import AccountUnavailableError, ApprovalInvalidError, DatabaseMissingError, ScopeViolationError
from .schema import LATEST_SCHEMA_VERSION, connect, current_version, migrate


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def stable_id(prefix: str, *parts: str) -> str:
    value = "\x1f".join(str(part) for part in parts)
    return f"{prefix}_{uuid.uuid5(uuid.NAMESPACE_URL, value).hex}"


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def json_value(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


class CommunicationRepository:
    """Profile-aware canonical storage; reads never create a missing database."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = (
            Path(db_path)
            if db_path is not None
            else get_hermes_home() / "communication" / "communication.db"
        )

    def initialize(self) -> int:
        return migrate(self.db_path)

    def schema_version(self) -> int:
        with self.read_connection() as connection:
            return current_version(connection)

    @contextmanager
    def read_connection(self) -> Iterator[sqlite3.Connection]:
        if not self.db_path.is_file():
            raise DatabaseMissingError(
                f"communication database does not exist: {self.db_path}"
            )
        connection = connect(self.db_path, readonly=True)
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        if not self.db_path.is_file():
            raise DatabaseMissingError(
                "communication database is not initialized; run "
                "`hermes communication init`"
            )
        connection = connect(self.db_path)
        connection.execute("BEGIN IMMEDIATE")
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _rows(rows: Sequence[sqlite3.Row]) -> list[dict[str, Any]]:
        return [dict(row) for row in rows]

    @staticmethod
    def _row(row: sqlite3.Row | None) -> dict[str, Any] | None:
        return dict(row) if row is not None else None

    def add_account(
        self,
        *,
        provider: str,
        account_namespace: str,
        label: str,
        owner_profile: str,
        credential_ref: str | None = None,
        browser_profile_ref: str | None = None,
        auth_status: str = "unknown",
        capabilities: Sequence[str] = (),
        write_policy: str = "disabled",
        account_id: str | None = None,
    ) -> dict[str, Any]:
        provider = provider.strip().lower()
        namespace = account_namespace.strip()
        if not provider or not namespace or not label.strip() or not owner_profile.strip():
            raise ValueError("provider, namespace, label, and owner_profile are required")
        if credential_ref and any(marker in credential_ref.lower() for marker in ("token=", "password=", "secret=")):
            raise ValueError("credential_ref must reference a secret, not contain one")
        now = utc_now()
        account_id = account_id or new_id("acct")
        with self.transaction() as connection:
            connection.execute(
                """INSERT INTO connected_accounts(
                       id, provider, account_namespace, label, owner_profile,
                       credential_ref, browser_profile_ref, auth_status,
                       capabilities_json, write_policy, created_at, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    account_id,
                    provider,
                    namespace,
                    label.strip(),
                    owner_profile.strip(),
                    credential_ref,
                    browser_profile_ref,
                    auth_status,
                    json_text(sorted(set(capabilities))),
                    write_policy,
                    now,
                    now,
                ),
            )
        return self.get_account(account_id) or {}

    def get_account(self, account_id: str) -> dict[str, Any] | None:
        with self.read_connection() as connection:
            row = connection.execute(
                "SELECT * FROM connected_accounts WHERE id = ?", (account_id,)
            ).fetchone()
        result = self._row(row)
        if result:
            result["capabilities"] = json_value(result.pop("capabilities_json"), [])
            result["rate_limit_state"] = json_value(
                result.pop("rate_limit_state_json"), {}
            )
        return result

    def list_accounts(self, *, include_disabled: bool = False) -> list[dict[str, Any]]:
        where = "" if include_disabled else "WHERE enabled = 1"
        with self.read_connection() as connection:
            rows = connection.execute(
                f"SELECT * FROM connected_accounts {where} ORDER BY provider, label, id"
            ).fetchall()
        result = self._rows(rows)
        for item in result:
            item["capabilities"] = json_value(item.pop("capabilities_json"), [])
            item.pop("rate_limit_state_json", None)
            item.pop("credential_ref", None)
        return result

    def set_account_health(
        self,
        account_id: str,
        *,
        auth_status: str,
        health_status: str,
        last_seen_at: str | None = None,
    ) -> None:
        with self.transaction() as connection:
            changed = connection.execute(
                """UPDATE connected_accounts
                   SET auth_status = ?, health_status = ?, last_seen_at = ?, updated_at = ?
                   WHERE id = ?""",
                (auth_status, health_status, last_seen_at or utc_now(), utc_now(), account_id),
            ).rowcount
            if changed != 1:
                raise KeyError(account_id)

    def disable_account(self, account_id: str) -> None:
        """Disable one scope and pause its endpoints/routes without fallback."""
        now = utc_now()
        with self.transaction() as connection:
            changed = connection.execute(
                "UPDATE connected_accounts SET enabled = 0, updated_at = ? WHERE id = ?",
                (now, account_id),
            ).rowcount
            if changed != 1:
                raise KeyError(account_id)
            connection.execute(
                "UPDATE contact_endpoints SET status = 'disabled', updated_at = ? "
                "WHERE connected_account_id = ?",
                (now, account_id),
            )
            connection.execute(
                """UPDATE person_channel_routes SET enabled = 0, updated_at = ?
                   WHERE source_endpoint_id IN (
                       SELECT id FROM contact_endpoints WHERE connected_account_id = ?
                   ) OR target_endpoint_id IN (
                       SELECT id FROM contact_endpoints WHERE connected_account_id = ?
                   )""",
                (now, account_id, account_id),
            )

    def create_person(
        self,
        display_name: str,
        *,
        timezone: str = "UTC",
        pii_policy: str = "minimal",
        person_id: str | None = None,
    ) -> dict[str, Any]:
        if not display_name.strip():
            raise ValueError("display_name is required")
        now = utc_now()
        person_id = person_id or new_id("person")
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO persons VALUES (?, ?, ?, ?, ?, ?)",
                (person_id, display_name.strip(), timezone, pii_policy, now, now),
            )
            connection.execute(
                "INSERT INTO communication_journeys VALUES (?, ?, ?, ?)",
                (stable_id("journey", person_id), person_id, now, now),
            )
        return self.get_person(person_id) or {}

    def get_person(self, person_id: str) -> dict[str, Any] | None:
        with self.read_connection() as connection:
            row = connection.execute(
                "SELECT * FROM persons WHERE id = ?", (person_id,)
            ).fetchone()
        return self._row(row)

    def person_detail(self, person_id: str) -> dict[str, Any]:
        """Return the canonical person and its identities/endpoints.

        This repository boundary keeps CLI and skill consumers independent of
        the physical schema and makes the account/identity/endpoint split
        explicit in one application-facing result.
        """
        person = self.get_person(person_id)
        if person is None:
            raise KeyError(person_id)
        with self.read_connection() as connection:
            identities = connection.execute(
                """SELECT i.id, i.person_id, i.provider,
                          i.observed_via_account_id, i.external_id,
                          i.display_name, i.profile_ref, i.observed_at,
                          e.id AS endpoint_id,
                          e.connected_account_id AS endpoint_account_id,
                          e.status AS endpoint_status
                   FROM platform_identities i
                   LEFT JOIN contact_endpoints e ON e.platform_identity_id = i.id
                   WHERE i.person_id = ?
                   ORDER BY i.provider, i.id, e.id""",
                (person_id,),
            ).fetchall()
        return {"person": person, "identities": self._rows(identities)}

    def search_people(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        query = query.strip()
        if not query:
            raise ValueError("query is required")
        escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        with self.read_connection() as connection:
            rows = connection.execute(
                """SELECT DISTINCT p.* FROM persons p
                   LEFT JOIN platform_identities i ON i.person_id = p.id
                   WHERE p.display_name LIKE ? ESCAPE '\\' COLLATE NOCASE
                      OR i.display_name LIKE ? ESCAPE '\\' COLLATE NOCASE
                   ORDER BY p.display_name COLLATE NOCASE, p.id LIMIT ?""",
                (f"%{escaped}%", f"%{escaped}%", max(1, min(int(limit), 500))),
            ).fetchall()
        return self._rows(rows)

    def find_duplicate_candidates(self) -> list[dict[str, Any]]:
        """Suggest, but never auto-merge, explainable identity candidates."""
        with self.read_connection() as connection:
            rows = connection.execute(
                """SELECT a.id AS first_person_id, b.id AS second_person_id,
                          a.display_name,
                          EXISTS(
                              SELECT 1 FROM platform_identities ia
                              JOIN platform_identities ib
                                ON ia.profile_ref = ib.profile_ref
                               AND ia.id <> ib.id
                              WHERE ia.person_id = a.id AND ib.person_id = b.id
                                AND ia.profile_ref IS NOT NULL
                                AND trim(ia.profile_ref) <> ''
                          ) AS shared_profile_ref
                   FROM persons a JOIN persons b ON a.id < b.id
                   WHERE lower(trim(a.display_name)) = lower(trim(b.display_name))
                   ORDER BY a.display_name, a.id, b.id"""
            ).fetchall()
        result = []
        for row in rows:
            strong = bool(row["shared_profile_ref"])
            result.append(
                {
                    "first_person_id": row["first_person_id"],
                    "second_person_id": row["second_person_id"],
                    "display_name": row["display_name"],
                    "confidence": 0.9 if strong else 0.35,
                    "reasons": (
                        ["same normalized name", "same explicit profile reference"]
                        if strong
                        else ["same normalized name only; insufficient without manual evidence"]
                    ),
                    "auto_merge": False,
                }
            )
        return result

    def search_all(self, query: str, limit: int = 20) -> dict[str, Any]:
        """Search canonical domains while returning IDs, not private body text."""
        query = query.strip()
        if not query:
            raise ValueError("query is required")
        escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        pattern = f"%{escaped}%"
        bounded = max(1, min(int(limit), 500))
        with self.read_connection() as connection:
            people = connection.execute(
                """SELECT DISTINCT p.id,
                          CASE WHEN p.pii_policy = 'restricted'
                               THEN '[redacted]' ELSE p.display_name END AS display_name,
                          CASE WHEN p.pii_policy = 'restricted'
                               THEN '[redacted]' ELSE p.timezone END AS timezone,
                          p.pii_policy
                   FROM persons p LEFT JOIN platform_identities i ON i.person_id = p.id
                   WHERE p.display_name LIKE ? ESCAPE '\\' COLLATE NOCASE
                      OR i.display_name LIKE ? ESCAPE '\\' COLLATE NOCASE
                   ORDER BY p.display_name, p.id LIMIT ?""",
                (pattern, pattern, bounded),
            ).fetchall()
            identities = connection.execute(
                """SELECT i.id, i.person_id, i.provider,
                          i.observed_via_account_id,
                          CASE WHEN p.pii_policy = 'restricted'
                               THEN '[redacted]' ELSE i.external_id END AS external_id,
                          CASE WHEN p.pii_policy = 'restricted'
                               THEN '[redacted]' ELSE i.display_name END AS display_name
                   FROM platform_identities i
                   LEFT JOIN persons p ON p.id = i.person_id
                   WHERE i.display_name LIKE ? ESCAPE '\\' COLLATE NOCASE
                      OR i.external_id LIKE ? ESCAPE '\\' COLLATE NOCASE
                   ORDER BY i.provider, i.id LIMIT ?""",
                (pattern, pattern, bounded),
            ).fetchall()
            conversations = connection.execute(
                """SELECT c.id, c.connected_account_id, c.endpoint_id, c.provider,
                          CASE WHEN p.pii_policy = 'restricted'
                               THEN '[redacted]' ELSE c.external_id END AS external_id,
                          CASE WHEN p.pii_policy = 'restricted'
                               THEN '[redacted]' ELSE c.title END AS title,
                          i.person_id
                   FROM conversations c
                   JOIN contact_endpoints e ON e.id = c.endpoint_id
                   JOIN platform_identities i ON i.id = e.platform_identity_id
                   LEFT JOIN persons p ON p.id = i.person_id
                   WHERE c.title LIKE ? ESCAPE '\\' COLLATE NOCASE
                      OR c.external_id LIKE ? ESCAPE '\\' COLLATE NOCASE
                   ORDER BY c.observed_at DESC, c.id LIMIT ?""",
                (pattern, pattern, bounded),
            ).fetchall()
            messages = connection.execute(
                """SELECT m.id, m.connected_account_id, m.endpoint_id,
                          m.conversation_id, m.provider, m.sent_at, i.person_id,
                          'body matched; content redacted' AS match_explanation
                   FROM messages m
                   JOIN contact_endpoints e ON e.id = m.endpoint_id
                   JOIN platform_identities i ON i.id = e.platform_identity_id
                   JOIN persons p ON p.id = i.person_id
                   WHERE p.pii_policy <> 'restricted'
                     AND m.body LIKE ? ESCAPE '\\' COLLATE NOCASE
                   ORDER BY m.sent_at DESC, m.id LIMIT ?""",
                (pattern, bounded),
            ).fetchall()
            events = connection.execute(
                """SELECT ev.id, ev.person_id, ev.connected_account_id,
                          ev.endpoint_id, ev.event_type,
                          CASE WHEN p.pii_policy = 'restricted'
                               THEN '[redacted]' ELSE ev.external_id END AS external_id,
                          ev.happened_at
                   FROM contact_events ev
                   JOIN persons p ON p.id = ev.person_id
                   WHERE ev.event_type LIKE ? ESCAPE '\\' COLLATE NOCASE
                      OR ev.external_id LIKE ? ESCAPE '\\' COLLATE NOCASE
                   ORDER BY ev.happened_at DESC, ev.id LIMIT ?""",
                (pattern, pattern, bounded),
            ).fetchall()
        return {
            "query": query,
            "people": self._rows(people),
            "identities": self._rows(identities),
            "conversations": self._rows(conversations),
            "messages": self._rows(messages),
            "events": self._rows(events),
            "message_bodies_redacted": True,
        }

    def upsert_identity(
        self,
        *,
        connected_account_id: str,
        external_id: str,
        display_name: str | None,
        profile_ref: str | None = None,
        person_id: str | None = None,
        provenance: dict[str, Any] | None = None,
        observed_at: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        account = self.get_account(connected_account_id)
        if account is None:
            raise KeyError(connected_account_id)
        now = utc_now()
        observed_at = observed_at or now
        identity_id = stable_id(
            "identity", account["provider"], connected_account_id, external_id
        )
        endpoint_id = stable_id("endpoint", connected_account_id, identity_id)
        with self.transaction() as connection:
            connection.execute(
                """INSERT INTO platform_identities(
                       id, person_id, provider, observed_via_account_id, external_id,
                       display_name, profile_ref, provenance_json, observed_at,
                       created_at, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(provider, observed_via_account_id, external_id) DO UPDATE SET
                       display_name = excluded.display_name,
                       profile_ref = COALESCE(excluded.profile_ref, platform_identities.profile_ref),
                       provenance_json = excluded.provenance_json,
                       observed_at = excluded.observed_at,
                       sync_version = platform_identities.sync_version + 1,
                       updated_at = excluded.updated_at""",
                (
                    identity_id,
                    person_id,
                    account["provider"],
                    connected_account_id,
                    external_id,
                    display_name,
                    profile_ref,
                    json_text(provenance or {}),
                    observed_at,
                    now,
                    now,
                ),
            )
            identity = connection.execute(
                """SELECT * FROM platform_identities
                   WHERE provider = ? AND observed_via_account_id = ? AND external_id = ?""",
                (account["provider"], connected_account_id, external_id),
            ).fetchone()
            assert identity is not None
            if person_id and identity["person_id"] not in (None, person_id):
                raise ScopeViolationError("identity already belongs to another person")
            if person_id and identity["person_id"] is None:
                connection.execute(
                    "UPDATE platform_identities SET person_id = ?, updated_at = ? WHERE id = ?",
                    (person_id, now, identity["id"]),
                )
            connection.execute(
                """INSERT INTO contact_endpoints(id, connected_account_id,
                       platform_identity_id, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(connected_account_id, platform_identity_id)
                   DO UPDATE SET updated_at = excluded.updated_at""",
                (endpoint_id, connected_account_id, identity["id"], now, now),
            )
            endpoint = connection.execute(
                "SELECT * FROM contact_endpoints WHERE connected_account_id = ? "
                "AND platform_identity_id = ?",
                (connected_account_id, identity["id"]),
            ).fetchone()
            identity = connection.execute(
                "SELECT * FROM platform_identities WHERE id = ?", (identity["id"],)
            ).fetchone()
        return dict(identity), dict(endpoint)

    def get_identity_by_external(
        self, connected_account_id: str, external_id: str
    ) -> dict[str, Any] | None:
        with self.read_connection() as connection:
            row = connection.execute(
                """SELECT i.* FROM platform_identities i
                   JOIN connected_accounts a ON a.id = i.observed_via_account_id
                   WHERE i.observed_via_account_id = ? AND i.external_id = ?""",
                (connected_account_id, external_id),
            ).fetchone()
        return self._row(row)

    def get_endpoint(self, endpoint_id: str) -> dict[str, Any] | None:
        with self.read_connection() as connection:
            row = connection.execute(
                "SELECT * FROM contact_endpoints WHERE id = ?", (endpoint_id,)
            ).fetchone()
        return self._row(row)

    def upsert_conversation(
        self,
        *,
        connected_account_id: str,
        endpoint_id: str,
        external_id: str,
        kind: str,
        title: str | None,
        provenance: dict[str, Any],
        observed_at: str,
    ) -> dict[str, Any]:
        account = self.get_account(connected_account_id)
        endpoint = self.get_endpoint(endpoint_id)
        if account is None or endpoint is None:
            raise KeyError("account or endpoint not found")
        if endpoint["connected_account_id"] != connected_account_id:
            raise ScopeViolationError("conversation endpoint belongs to another account")
        conversation_id = stable_id(
            "conversation", account["provider"], connected_account_id, external_id
        )
        now = utc_now()
        with self.transaction() as connection:
            connection.execute(
                """INSERT INTO conversations(
                       id, connected_account_id, endpoint_id, provider, external_id,
                       kind, title, provenance_json, observed_at, created_at, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(provider, connected_account_id, external_id) DO UPDATE SET
                       endpoint_id = excluded.endpoint_id,
                       kind = excluded.kind,
                       title = excluded.title,
                       provenance_json = excluded.provenance_json,
                       observed_at = excluded.observed_at,
                       sync_version = conversations.sync_version + 1,
                       updated_at = excluded.updated_at""",
                (
                    conversation_id,
                    connected_account_id,
                    endpoint_id,
                    account["provider"],
                    external_id,
                    kind,
                    title,
                    json_text(provenance),
                    observed_at,
                    now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM conversations WHERE provider = ? "
                "AND connected_account_id = ? AND external_id = ?",
                (account["provider"], connected_account_id, external_id),
            ).fetchone()
        return dict(row)

    def get_conversation_by_external(
        self, connected_account_id: str, external_id: str
    ) -> dict[str, Any] | None:
        with self.read_connection() as connection:
            row = connection.execute(
                "SELECT * FROM conversations WHERE connected_account_id = ? AND external_id = ?",
                (connected_account_id, external_id),
            ).fetchone()
        return self._row(row)

    def upsert_message(
        self,
        *,
        connected_account_id: str,
        endpoint_id: str,
        conversation_id: str,
        external_id: str | None,
        direction: str,
        body: str,
        sent_at: str,
        sender_identity_id: str | None,
        provenance: dict[str, Any],
        observed_at: str,
    ) -> tuple[dict[str, Any], bool]:
        with self.read_connection() as connection:
            conversation = connection.execute(
                "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
        if conversation is None or conversation["connected_account_id"] != connected_account_id:
            raise ScopeViolationError("message conversation belongs to another account")
        if conversation["endpoint_id"] != endpoint_id:
            raise ScopeViolationError("message endpoint does not match conversation")
        endpoint = self.get_endpoint(endpoint_id)
        if endpoint is None:
            raise ScopeViolationError("message endpoint is unavailable")
        if sender_identity_id and endpoint["platform_identity_id"] != sender_identity_id:
            raise ScopeViolationError(
                "message sender identity does not match the conversation contact"
            )
        account = self.get_account(connected_account_id)
        assert account is not None
        fingerprint = hashlib.sha256(
            "\x1f".join(
                (conversation_id, sender_identity_id or "", direction, sent_at, body)
            ).encode("utf-8")
        ).hexdigest()
        message_id = stable_id(
            "message",
            account["provider"],
            connected_account_id,
            external_id or fingerprint,
        )
        now = utc_now()
        with self.transaction() as connection:
            existing = connection.execute(
                "SELECT id FROM messages WHERE connected_account_id = ? "
                "AND stable_fingerprint = ?",
                (connected_account_id, fingerprint),
            ).fetchone()
            connection.execute(
                """INSERT INTO messages(
                       id, connected_account_id, endpoint_id, conversation_id,
                       provider, external_id, stable_fingerprint, direction,
                       sender_identity_id, body, sent_at, observed_at,
                       provenance_json, created_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(connected_account_id, stable_fingerprint) DO UPDATE SET
                       observed_at = excluded.observed_at,
                       provenance_json = excluded.provenance_json,
                       sync_version = messages.sync_version + 1""",
                (
                    message_id,
                    connected_account_id,
                    endpoint_id,
                    conversation_id,
                    account["provider"],
                    external_id,
                    fingerprint,
                    direction,
                    sender_identity_id,
                    body,
                    sent_at,
                    observed_at,
                    json_text(provenance),
                    now,
                ),
            )
            inserted = existing is None
            row = connection.execute(
                "SELECT * FROM messages WHERE connected_account_id = ? "
                "AND stable_fingerprint = ?",
                (connected_account_id, fingerprint),
            ).fetchone()
        return dict(row), inserted

    def start_sync_run(
        self,
        connected_account_id: str,
        *,
        mode: str,
        endpoint_id: str | None = None,
        retry_of_id: str | None = None,
    ) -> str:
        run_id = new_id("sync")
        with self.transaction() as connection:
            connection.execute(
                """INSERT INTO sync_runs(
                       id, connected_account_id, endpoint_id, mode, status,
                       started_at, retry_of_id
                   ) VALUES (?, ?, ?, ?, 'running', ?, ?)""",
                (run_id, connected_account_id, endpoint_id, mode, utc_now(), retry_of_id),
            )
        return run_id

    def finish_sync_run(
        self, run_id: str, *, status: str, stats: dict[str, Any]
    ) -> None:
        with self.transaction() as connection:
            connection.execute(
                "UPDATE sync_runs SET status = ?, stats_json = ?, finished_at = ? WHERE id = ?",
                (status, json_text(stats), utc_now(), run_id),
            )
            if status == "succeeded":
                connection.execute(
                    """UPDATE connected_accounts SET last_successful_sync_at = ?,
                       health_status = 'healthy', updated_at = ?
                       WHERE id = (SELECT connected_account_id FROM sync_runs WHERE id = ?)""",
                    (utc_now(), utc_now(), run_id),
                )

    def add_sync_issue(
        self,
        run_id: str,
        connected_account_id: str,
        *,
        code: str,
        detail_redacted: str,
        retryable: bool,
        endpoint_id: str | None = None,
    ) -> str:
        issue_id = new_id("issue")
        with self.transaction() as connection:
            connection.execute(
                """INSERT INTO sync_issues VALUES
                   (?, ?, ?, ?, ?, ?, ?, 'open', ?, NULL)""",
                (
                    issue_id,
                    run_id,
                    connected_account_id,
                    endpoint_id,
                    code,
                    detail_redacted,
                    int(retryable),
                    utc_now(),
                ),
            )
        return issue_id

    def set_cursor(
        self,
        connected_account_id: str,
        resource: str,
        value: str,
        *,
        endpoint_id: str | None = None,
    ) -> None:
        with self.transaction() as connection:
            if endpoint_id is None:
                connection.execute(
                    """INSERT INTO sync_cursors(
                           connected_account_id, endpoint_id, resource, cursor_value, updated_at
                       ) VALUES (?, NULL, ?, ?, ?)
                       ON CONFLICT(connected_account_id, resource)
                       WHERE endpoint_id IS NULL DO UPDATE SET
                           cursor_value = excluded.cursor_value,
                           sync_version = sync_cursors.sync_version + 1,
                           updated_at = excluded.updated_at""",
                    (connected_account_id, resource, value, utc_now()),
                )
            else:
                connection.execute(
                    """INSERT INTO sync_cursors(
                           connected_account_id, endpoint_id, resource, cursor_value, updated_at
                       ) VALUES (?, ?, ?, ?, ?)
                       ON CONFLICT(connected_account_id, endpoint_id, resource) DO UPDATE SET
                           cursor_value = excluded.cursor_value,
                           sync_version = sync_cursors.sync_version + 1,
                           updated_at = excluded.updated_at""",
                    (connected_account_id, endpoint_id, resource, value, utc_now()),
                )

    def get_cursor(
        self, connected_account_id: str, resource: str, endpoint_id: str | None = None
    ) -> str | None:
        with self.read_connection() as connection:
            if endpoint_id is None:
                row = connection.execute(
                    """SELECT cursor_value FROM sync_cursors
                       WHERE connected_account_id = ? AND endpoint_id IS NULL AND resource = ?""",
                    (connected_account_id, resource),
                ).fetchone()
            else:
                row = connection.execute(
                    """SELECT cursor_value FROM sync_cursors
                       WHERE connected_account_id = ? AND endpoint_id = ? AND resource = ?""",
                    (connected_account_id, endpoint_id, resource),
                ).fetchone()
        return str(row[0]) if row else None

    def sync_status(self, connected_account_id: str) -> dict[str, Any]:
        with self.read_connection() as connection:
            runs = connection.execute(
                """SELECT * FROM sync_runs WHERE connected_account_id = ?
                   ORDER BY started_at DESC LIMIT 20""",
                (connected_account_id,),
            ).fetchall()
            issues = connection.execute(
                """SELECT * FROM sync_issues WHERE connected_account_id = ?
                   AND status IN ('open', 'retrying', 'quarantined')
                   ORDER BY created_at DESC""",
                (connected_account_id,),
            ).fetchall()
        return {"runs": self._rows(runs), "issues": self._rows(issues)}

    @contextmanager
    def account_sync_lock(
        self, connected_account_id: str, *, ttl_seconds: int = 300
    ) -> Iterator[str]:
        owner_token = new_id("synclock")
        now = utc_now()
        expires_at = (
            datetime.now(UTC) + timedelta(seconds=max(1, ttl_seconds))
        ).isoformat(timespec="microseconds").replace("+00:00", "Z")
        with self.transaction() as connection:
            connection.execute(
                "DELETE FROM sync_locks WHERE connected_account_id = ? AND expires_at <= ?",
                (connected_account_id, now),
            )
            acquired = connection.execute(
                """INSERT OR IGNORE INTO sync_locks VALUES (?, ?, ?, ?)""",
                (connected_account_id, owner_token, expires_at, now),
            ).rowcount
        if acquired != 1:
            raise AccountUnavailableError(
                f"sync already running for connected account {connected_account_id}"
            )
        try:
            yield owner_token
        finally:
            with self.transaction() as connection:
                connection.execute(
                    "DELETE FROM sync_locks WHERE connected_account_id = ? AND owner_token = ?",
                    (connected_account_id, owner_token),
                )

    def allow_account_link(
        self,
        source_account_id: str,
        target_account_id: str,
        *,
        allowed: bool,
        actor: str,
        reason: str,
    ) -> None:
        if source_account_id == target_account_id:
            raise ValueError("source and target accounts must differ")
        with self.transaction() as connection:
            connection.execute(
                """INSERT INTO account_link_policies VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(source_account_id, target_account_id) DO UPDATE SET
                       allowed = excluded.allowed, actor = excluded.actor,
                       reason = excluded.reason, updated_at = excluded.updated_at""",
                (
                    source_account_id,
                    target_account_id,
                    int(allowed),
                    actor,
                    reason,
                    utc_now(),
                ),
            )

    def account_link_allowed(self, source_account_id: str, target_account_id: str) -> bool:
        with self.read_connection() as connection:
            row = connection.execute(
                """SELECT allowed FROM account_link_policies
                   WHERE source_account_id = ? AND target_account_id = ?""",
                (source_account_id, target_account_id),
            ).fetchone()
        return bool(row and row[0])

    def list_routes(self, person_id: str | None = None) -> list[dict[str, Any]]:
        where = "WHERE person_id = ?" if person_id else ""
        params = (person_id,) if person_id else ()
        with self.read_connection() as connection:
            rows = connection.execute(
                f"SELECT * FROM person_channel_routes {where} ORDER BY person_id, id",
                params,
            ).fetchall()
        return self._rows(rows)

    def set_person_route(
        self,
        *,
        person_id: str,
        source_endpoint_id: str,
        target_endpoint_id: str,
        actor: str,
        reason: str,
    ) -> dict[str, Any]:
        source = self.get_endpoint(source_endpoint_id)
        target = self.get_endpoint(target_endpoint_id)
        if source is None or target is None:
            raise KeyError("source or target endpoint not found")
        if not self.account_link_allowed(
            source["connected_account_id"], target["connected_account_id"]
        ):
            raise ScopeViolationError("directed AccountLinkPolicy is default-deny")
        with self.read_connection() as connection:
            identity_people = connection.execute(
                """SELECT person_id FROM platform_identities WHERE id IN (?, ?)""",
                (source["platform_identity_id"], target["platform_identity_id"]),
            ).fetchall()
        if any(row[0] != person_id for row in identity_people):
            raise ScopeViolationError("route endpoints must belong to the selected person")
        route_id = stable_id("route", person_id, source_endpoint_id)
        now = utc_now()
        with self.transaction() as connection:
            connection.execute(
                """INSERT INTO person_channel_routes VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)
                   ON CONFLICT(person_id, source_endpoint_id) DO UPDATE SET
                       target_endpoint_id = excluded.target_endpoint_id,
                       actor = excluded.actor, reason = excluded.reason,
                       enabled = 1, updated_at = excluded.updated_at""",
                (
                    route_id,
                    person_id,
                    source_endpoint_id,
                    target_endpoint_id,
                    actor,
                    reason,
                    now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM person_channel_routes WHERE person_id = ? AND source_endpoint_id = ?",
                (person_id, source_endpoint_id),
            ).fetchone()
        return dict(row)

    def route_version(self, route: dict[str, Any]) -> str:
        return hashlib.sha256(
            json_text(
                {
                    "id": route["id"],
                    "person_id": route["person_id"],
                    "source_endpoint_id": route["source_endpoint_id"],
                    "target_endpoint_id": route["target_endpoint_id"],
                    "updated_at": route["updated_at"],
                    "enabled": route["enabled"],
                }
            ).encode("utf-8")
        ).hexdigest()

    def get_route(self, person_id: str, source_endpoint_id: str) -> dict[str, Any] | None:
        with self.read_connection() as connection:
            row = connection.execute(
                """SELECT * FROM person_channel_routes
                   WHERE person_id = ? AND source_endpoint_id = ? AND enabled = 1""",
                (person_id, source_endpoint_id),
            ).fetchone()
        return self._row(row)

    def audit_route(
        self,
        *,
        person_id: str | None,
        source_account_id: str,
        target_account_id: str,
        source_endpoint_id: str | None,
        target_endpoint_id: str | None,
        action: str,
        allowed: bool,
        explanation: str,
        actor: str,
    ) -> str:
        audit_id = new_id("routeaudit")
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO route_audit VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    audit_id,
                    person_id,
                    source_account_id,
                    target_account_id,
                    source_endpoint_id,
                    target_endpoint_id,
                    action,
                    int(allowed),
                    explanation,
                    actor,
                    utc_now(),
                ),
            )
        return audit_id

    def timeline(
        self,
        person_id: str,
        *,
        endpoint_id: str | None = None,
        start_at: str | None = None,
        end_at: str | None = None,
    ) -> dict[str, Any]:
        message_filters = ["i.person_id = ?"]
        message_params: list[Any] = [person_id]
        if endpoint_id:
            message_filters.append("m.endpoint_id = ?")
            message_params.append(endpoint_id)
        if start_at:
            message_filters.append("m.sent_at >= ?")
            message_params.append(start_at)
        if end_at:
            message_filters.append("m.sent_at < ?")
            message_params.append(end_at)

        event_filters = ["ev.person_id = ?"]
        event_params: list[Any] = [person_id]
        episode_filters = ["j.person_id = ?"]
        episode_params: list[Any] = [person_id]
        transition_filters = ["j.person_id = ?"]
        transition_params: list[Any] = [person_id]
        if endpoint_id:
            event_filters.append("ev.endpoint_id = ?")
            event_params.append(endpoint_id)
            episode_filters.append("ep.endpoint_id = ?")
            episode_params.append(endpoint_id)
            transition_filters.append(
                "(tr.from_endpoint_id = ? OR tr.to_endpoint_id = ?)"
            )
            transition_params.extend((endpoint_id, endpoint_id))
        if start_at:
            event_filters.append("ev.happened_at >= ?")
            event_params.append(start_at)
            episode_filters.append("ep.started_at >= ?")
            episode_params.append(start_at)
            transition_filters.append("tr.happened_at >= ?")
            transition_params.append(start_at)
        if end_at:
            event_filters.append("ev.happened_at < ?")
            event_params.append(end_at)
            episode_filters.append("ep.started_at < ?")
            episode_params.append(end_at)
            transition_filters.append("tr.happened_at < ?")
            transition_params.append(end_at)
        with self.read_connection() as connection:
            messages = connection.execute(
                f"""SELECT m.id, m.connected_account_id, m.endpoint_id,
                          m.conversation_id, m.provider, m.direction,
                          CASE WHEN p.pii_policy = 'restricted'
                               THEN '[redacted]' ELSE m.body END AS body,
                          m.sent_at, m.observed_at, m.provenance_json
                   FROM messages m
                   JOIN contact_endpoints e ON e.id = m.endpoint_id
                   JOIN platform_identities i ON i.id = e.platform_identity_id
                   JOIN persons p ON p.id = i.person_id
                   WHERE {' AND '.join(message_filters)}
                   ORDER BY m.sent_at, m.id""",
                message_params,
            ).fetchall()
            events = connection.execute(
                f"""SELECT ev.id, ev.person_id, ev.connected_account_id,
                           ev.endpoint_id, ev.event_type,
                           CASE WHEN p.pii_policy = 'restricted'
                                THEN '[redacted]' ELSE ev.external_id END AS external_id,
                           ev.happened_at, ev.timezone,
                           CASE WHEN p.pii_policy = 'restricted'
                                THEN '{{}}' ELSE ev.data_json END AS data_json,
                           CASE WHEN p.pii_policy = 'restricted'
                                THEN '{{"redacted":true}}' ELSE ev.provenance_json
                           END AS provenance_json,
                           ev.observed_at, ev.sync_version
                    FROM contact_events ev JOIN persons p ON p.id = ev.person_id
                    WHERE {' AND '.join(event_filters)}
                    ORDER BY ev.happened_at, ev.id""",
                event_params,
            ).fetchall()
            episodes = connection.execute(
                f"""SELECT ep.* FROM channel_episodes ep
                    JOIN communication_journeys j ON j.id = ep.journey_id
                    WHERE {' AND '.join(episode_filters)}
                    ORDER BY ep.started_at, ep.id""",
                episode_params,
            ).fetchall()
            transitions = connection.execute(
                f"""SELECT tr.* FROM channel_transitions tr
                    JOIN communication_journeys j ON j.id = tr.journey_id
                    WHERE {' AND '.join(transition_filters)}
                    ORDER BY tr.happened_at, tr.id""",
                transition_params,
            ).fetchall()
        return {
            "person_id": person_id,
            "messages": self._rows(messages),
            "events": self._rows(events),
            "episodes": self._rows(episodes),
            "transitions": self._rows(transitions),
        }

    def list_messages(self, person_id: str) -> list[dict[str, Any]]:
        return self.timeline(person_id)["messages"]

    def record_transition(
        self,
        *,
        person_id: str,
        from_endpoint_id: str,
        to_endpoint_id: str,
        initiator: str,
        evidence_type: str,
        evidence_ref: str,
        happened_at: str | None = None,
    ) -> dict[str, Any]:
        if initiator not in {"person", "user", "inbound_resume"}:
            raise ValueError("transition initiator must be explicit")
        if evidence_type == "delivery_failure":
            raise ScopeViolationError(
                "delivery failure cannot move a conversation to another channel"
            )
        if initiator == "person" and evidence_type != "person_request":
            raise ScopeViolationError("person transition requires person_request evidence")
        if initiator == "inbound_resume" and evidence_type != "incoming_message":
            raise ScopeViolationError(
                "inbound resume requires exact incoming_message evidence"
            )
        if from_endpoint_id == to_endpoint_id:
            raise ValueError("transition endpoints must differ")
        journey_id = stable_id("journey", person_id)
        happened_at = happened_at or utc_now()
        transition_id = new_id("transition")
        now = utc_now()
        with self.transaction() as connection:
            for endpoint_id in (from_endpoint_id, to_endpoint_id):
                row = connection.execute(
                    """SELECT i.person_id FROM contact_endpoints e
                       JOIN platform_identities i ON i.id = e.platform_identity_id
                       WHERE e.id = ?""",
                    (endpoint_id,),
                ).fetchone()
                if row is None or row[0] != person_id:
                    raise ScopeViolationError("transition endpoint belongs to another person")
            connection.execute(
                """INSERT INTO channel_transitions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    transition_id,
                    journey_id,
                    from_endpoint_id,
                    to_endpoint_id,
                    initiator,
                    evidence_type,
                    evidence_ref,
                    happened_at,
                    now,
                ),
            )
            connection.execute(
                """UPDATE channel_episodes SET ended_at = ?, end_reason = ?
                   WHERE journey_id = ? AND endpoint_id = ? AND ended_at IS NULL""",
                (happened_at, "channel_transition", journey_id, from_endpoint_id),
            )
            connection.execute(
                """INSERT INTO channel_episodes VALUES (?, ?, ?, ?, NULL, ?, NULL, ?)""",
                (
                    new_id("episode"),
                    journey_id,
                    to_endpoint_id,
                    happened_at,
                    initiator,
                    now,
                ),
            )
            previous_state = (
                "return_by_request"
                if initiator in {"person", "inbound_resume"}
                else "paused"
            )
            connection.execute(
                """INSERT INTO channel_preferences VALUES (?, ?, ?, ?)
                   ON CONFLICT(person_id, endpoint_id) DO UPDATE SET
                       state = excluded.state, updated_at = excluded.updated_at""",
                (person_id, from_endpoint_id, previous_state, now),
            )
            connection.execute(
                """INSERT INTO channel_preferences VALUES (?, ?, 'active', ?)
                   ON CONFLICT(person_id, endpoint_id) DO UPDATE SET
                       state = 'active', updated_at = excluded.updated_at""",
                (person_id, to_endpoint_id, now),
            )
            row = connection.execute(
                "SELECT * FROM channel_transitions WHERE id = ?", (transition_id,)
            ).fetchone()
        return dict(row)

    def resume_from_inbound(
        self, *, person_id: str, endpoint_id: str, message_id: str
    ) -> dict[str, Any]:
        with self.read_connection() as connection:
            message = connection.execute(
                """SELECT m.id, m.direction, m.endpoint_id, i.person_id
                   FROM messages m
                   JOIN contact_endpoints e ON e.id = m.endpoint_id
                   JOIN platform_identities i ON i.id = e.platform_identity_id
                   WHERE m.id = ?""",
                (message_id,),
            ).fetchone()
            if (
                message is None
                or message["direction"] != "incoming"
                or message["endpoint_id"] != endpoint_id
                or message["person_id"] != person_id
            ):
                raise ScopeViolationError(
                    "inbound resume evidence must be an incoming message for the exact endpoint"
                )
            active = connection.execute(
                """SELECT endpoint_id FROM channel_preferences
                   WHERE person_id = ? AND state = 'active'""",
                (person_id,),
            ).fetchone()
        if active is not None and active[0] == endpoint_id:
            return {
                "person_id": person_id,
                "endpoint_id": endpoint_id,
                "state": "active",
                "evidence_ref": message_id,
                "transitioned": False,
            }
        if active is None:
            now = utc_now()
            journey_id = stable_id("journey", person_id)
            with self.transaction() as connection:
                connection.execute(
                    """INSERT INTO channel_preferences VALUES (?, ?, 'active', ?)
                       ON CONFLICT(person_id, endpoint_id) DO UPDATE SET
                           state = 'active', updated_at = excluded.updated_at""",
                    (person_id, endpoint_id, now),
                )
                connection.execute(
                    """INSERT INTO channel_episodes VALUES
                       (?, ?, ?, ?, NULL, 'inbound_resume', NULL, ?)""",
                    (new_id("episode"), journey_id, endpoint_id, now, now),
                )
            return {
                "person_id": person_id,
                "endpoint_id": endpoint_id,
                "state": "active",
                "evidence_ref": message_id,
                "transitioned": False,
            }
        return self.record_transition(
            person_id=person_id,
            from_endpoint_id=active[0],
            to_endpoint_id=endpoint_id,
            initiator="inbound_resume",
            evidence_type="incoming_message",
            evidence_ref=message_id,
        )

    def create_group(self, name: str, *, exclusion: bool = False) -> dict[str, Any]:
        now = utc_now()
        group_id = new_id("group")
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO contact_groups VALUES (?, ?, ?, ?, ?)",
                (group_id, name.strip(), int(exclusion), now, now),
            )
            row = connection.execute(
                "SELECT * FROM contact_groups WHERE id = ?", (group_id,)
            ).fetchone()
        return dict(row)

    def add_group_member(self, group_id: str, person_id: str) -> None:
        with self.transaction() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO contact_group_members VALUES (?, ?, ?)",
                (group_id, person_id, utc_now()),
            )

    def list_groups(self) -> list[dict[str, Any]]:
        with self.read_connection() as connection:
            rows = connection.execute(
                """SELECT g.*, COUNT(m.person_id) AS member_count
                   FROM contact_groups g LEFT JOIN contact_group_members m ON m.group_id = g.id
                   GROUP BY g.id ORDER BY g.name COLLATE NOCASE"""
            ).fetchall()
        return self._rows(rows)

    def group_preview(self, group_id: str) -> dict[str, Any]:
        with self.read_connection() as connection:
            group = connection.execute(
                "SELECT * FROM contact_groups WHERE id = ?", (group_id,)
            ).fetchone()
            members = connection.execute(
                """SELECT p.id, p.display_name, p.timezone
                   FROM contact_group_members gm JOIN persons p ON p.id = gm.person_id
                   WHERE gm.group_id = ? ORDER BY p.id""",
                (group_id,),
            ).fetchall()
        if group is None:
            raise KeyError(group_id)
        people = self._rows(members)
        payload = {"group_id": group_id, "people": people}
        return {
            **dict(group),
            "members": people,
            "preview_hash": hashlib.sha256(json_text(payload).encode("utf-8")).hexdigest(),
        }

    def create_segment(self, name: str, query: dict[str, Any]) -> dict[str, Any]:
        allowed = {"timezone", "tag", "priority_min", "last_touch_before"}
        if not set(query).issubset(allowed):
            raise ValueError("segment query contains unsupported fields")
        segment_id = new_id("segment")
        now = utc_now()
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO smart_segments VALUES (?, ?, ?, ?, ?)",
                (segment_id, name.strip(), json_text(query), now, now),
            )
            row = connection.execute(
                "SELECT * FROM smart_segments WHERE id = ?", (segment_id,)
            ).fetchone()
        result = dict(row)
        result["query"] = json_value(result.pop("query_json"), {})
        return result

    def segment_preview(self, segment_id: str) -> dict[str, Any]:
        with self.read_connection() as connection:
            segment = connection.execute(
                "SELECT * FROM smart_segments WHERE id = ?", (segment_id,)
            ).fetchone()
            if segment is None:
                raise KeyError(segment_id)
            query = json_value(segment["query_json"], {})
            rows = connection.execute(
                """SELECT p.*, COALESCE(r.priority, 0) AS priority,
                          COALESCE(r.tags_json, '[]') AS tags_json,
                          r.last_touch_at
                   FROM persons p LEFT JOIN relationship_states r ON r.person_id = p.id
                   ORDER BY p.id"""
            ).fetchall()
        matches: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            reasons: list[str] = []
            if "timezone" in query:
                if item["timezone"] != query["timezone"]:
                    continue
                reasons.append(f"timezone={query['timezone']}")
            if "priority_min" in query:
                if int(item["priority"]) < int(query["priority_min"]):
                    continue
                reasons.append(f"priority>={query['priority_min']}")
            if "tag" in query:
                tags = json_value(item["tags_json"], [])
                if query["tag"] not in tags:
                    continue
                reasons.append(f"tag={query['tag']}")
            if "last_touch_before" in query:
                if item["last_touch_at"] and item["last_touch_at"] >= query["last_touch_before"]:
                    continue
                reasons.append(f"last_touch_before={query['last_touch_before']}")
            matches.append({"person_id": item["id"], "reasons": reasons})
        immutable = {"segment_id": segment_id, "members": matches}
        return {
            "segment_id": segment_id,
            "query": query,
            "members": matches,
            "preview_hash": hashlib.sha256(json_text(immutable).encode("utf-8")).hexdigest(),
        }

    def add_contact_event(
        self,
        *,
        person_id: str,
        event_type: str,
        happened_at: str,
        timezone: str = "UTC",
        connected_account_id: str | None = None,
        endpoint_id: str | None = None,
        external_id: str | None = None,
        data: dict[str, Any] | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        event_id = stable_id(
            "event",
            connected_account_id or "manual",
            external_id or new_id("manual"),
            event_type,
        )
        now = utc_now()
        with self.transaction() as connection:
            connection.execute(
                """INSERT INTO contact_events(
                       id, person_id, connected_account_id, endpoint_id, event_type,
                       external_id, happened_at, timezone, data_json,
                       provenance_json, observed_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(connected_account_id, external_id, event_type) DO UPDATE SET
                       happened_at = excluded.happened_at,
                       data_json = excluded.data_json,
                       provenance_json = excluded.provenance_json,
                       observed_at = excluded.observed_at,
                       sync_version = contact_events.sync_version + 1""",
                (
                    event_id,
                    person_id,
                    connected_account_id,
                    endpoint_id,
                    event_type,
                    external_id,
                    happened_at,
                    timezone,
                    json_text(data or {}),
                    json_text(provenance or {}),
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM contact_events WHERE id = ?", (event_id,)
            ).fetchone()
        return dict(row)

    def create_draft(
        self,
        *,
        person_id: str,
        source_account_id: str,
        source_endpoint_id: str,
        target_account_id: str,
        endpoint_id: str,
        route_version: str,
        recipients: Sequence[dict[str, Any]],
        payload: str,
    ) -> dict[str, Any]:
        endpoint = self.get_endpoint(endpoint_id)
        source_endpoint = self.get_endpoint(source_endpoint_id)
        if endpoint is None or endpoint["connected_account_id"] != target_account_id:
            raise ScopeViolationError("draft endpoint does not belong to target account")
        if (
            source_endpoint is None
            or source_endpoint["connected_account_id"] != source_account_id
        ):
            raise ScopeViolationError(
                "draft source endpoint does not belong to source account"
            )
        draft_id = new_id("draft")
        payload_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        now = utc_now()
        with self.transaction() as connection:
            connection.execute(
                """INSERT INTO drafts(
                       id, person_id, connected_account_id, endpoint_id,
                       source_account_id, source_endpoint_id, route_version,
                       recipient_preview_json, payload, payload_hash, status,
                       created_at, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'draft', ?, ?)""",
                (
                    draft_id,
                    person_id,
                    target_account_id,
                    endpoint_id,
                    source_account_id,
                    source_endpoint_id,
                    route_version,
                    json_text(list(recipients)),
                    payload,
                    payload_hash,
                    now,
                    now,
                ),
            )
            row = connection.execute("SELECT * FROM drafts WHERE id = ?", (draft_id,)).fetchone()
        return dict(row)

    def get_draft(self, draft_id: str) -> dict[str, Any] | None:
        with self.read_connection() as connection:
            row = connection.execute("SELECT * FROM drafts WHERE id = ?", (draft_id,)).fetchone()
        return self._row(row)

    def list_drafts(self, status: str | None = None) -> list[dict[str, Any]]:
        where = "WHERE status = ?" if status else ""
        params = (status,) if status else ()
        with self.read_connection() as connection:
            rows = connection.execute(
                f"SELECT * FROM drafts {where} ORDER BY created_at DESC", params
            ).fetchall()
        return self._rows(rows)

    def cancel_draft(self, draft_id: str) -> None:
        with self.transaction() as connection:
            connection.execute(
                "UPDATE drafts SET status = 'cancelled', updated_at = ? "
                "WHERE id = ? AND status = 'draft'",
                (utc_now(), draft_id),
            )
            connection.execute(
                "UPDATE approvals SET status = 'invalidated' "
                "WHERE draft_id = ? AND status = 'active'",
                (draft_id,),
            )

    def approve_draft(
        self, draft_id: str, *, actor: str, expires_at: str
    ) -> dict[str, Any]:
        approval_id = new_id("approval")
        now = utc_now()
        invalid_reason: str | None = None
        row: sqlite3.Row | None = None
        with self.transaction() as connection:
            draft = connection.execute(
                "SELECT * FROM drafts WHERE id = ?", (draft_id,)
            ).fetchone()
            if draft is None or draft["status"] != "draft":
                raise ApprovalInvalidError("draft is not approvable")
            route = connection.execute(
                """SELECT * FROM person_channel_routes
                   WHERE person_id = ? AND source_endpoint_id = ?
                   AND target_endpoint_id = ? AND enabled = 1""",
                (draft["person_id"], draft["source_endpoint_id"], draft["endpoint_id"]),
            ).fetchone()
            if route is None or self.route_version(dict(route)) != draft["route_version"]:
                connection.execute(
                    "UPDATE drafts SET status = 'invalidated', updated_at = ? WHERE id = ?",
                    (now, draft_id),
                )
                invalid_reason = "route changed after draft creation"
            else:
                recipients_hash = hashlib.sha256(
                    draft["recipient_preview_json"].encode("utf-8")
                ).hexdigest()
                connection.execute(
                    """INSERT INTO approvals(
                           id, draft_id, actor, person_id, source_account_id,
                           source_endpoint_id, target_account_id, endpoint_id,
                           recipient_preview_hash, payload_hash, route_version,
                           status, approved_at, expires_at, consumed_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, NULL)""",
                    (
                        approval_id,
                        draft_id,
                        actor,
                        draft["person_id"],
                        draft["source_account_id"],
                        draft["source_endpoint_id"],
                        draft["connected_account_id"],
                        draft["endpoint_id"],
                        recipients_hash,
                        draft["payload_hash"],
                        draft["route_version"],
                        now,
                        expires_at,
                    ),
                )
                connection.execute(
                    "UPDATE drafts SET status = 'approved', updated_at = ? WHERE id = ?",
                    (now, draft_id),
                )
                row = connection.execute(
                    "SELECT * FROM approvals WHERE id = ?", (approval_id,)
                ).fetchone()
        if invalid_reason:
            raise ScopeViolationError(invalid_reason)
        assert row is not None
        return dict(row)

    def reject_approval(self, approval_id: str) -> None:
        with self.transaction() as connection:
            connection.execute(
                "UPDATE approvals SET status = 'rejected' WHERE id = ? AND status = 'active'",
                (approval_id,),
            )

    def enqueue_approved(self, approval_id: str, *, idempotency_key: str) -> dict[str, Any]:
        """Atomically revalidate and consume an approval while inserting outbox."""
        outbox_id = new_id("outbox")
        now = utc_now()
        with self.transaction() as connection:
            approval = connection.execute(
                "SELECT * FROM approvals WHERE id = ?", (approval_id,)
            ).fetchone()
            if approval is None or approval["status"] != "active" or approval["expires_at"] <= now:
                if approval is not None and approval["status"] == "active":
                    connection.execute(
                        "UPDATE approvals SET status = 'expired' WHERE id = ?",
                        (approval_id,),
                    )
                    connection.commit()
                raise ApprovalInvalidError("approval is not active")
            draft = connection.execute(
                "SELECT * FROM drafts WHERE id = ?", (approval["draft_id"],)
            ).fetchone()
            if draft is None or draft["status"] != "approved":
                raise ApprovalInvalidError("approved draft is unavailable")
            recipient_hash = hashlib.sha256(
                draft["recipient_preview_json"].encode("utf-8")
            ).hexdigest()
            exact = (
                approval["person_id"] == draft["person_id"]
                and approval["source_account_id"] == draft["source_account_id"]
                and approval["source_endpoint_id"] == draft["source_endpoint_id"]
                and approval["target_account_id"] == draft["connected_account_id"]
                and approval["endpoint_id"] == draft["endpoint_id"]
                and approval["payload_hash"] == draft["payload_hash"]
                and approval["recipient_preview_hash"] == recipient_hash
                and approval["route_version"] == draft["route_version"]
            )
            if not exact:
                connection.execute(
                    "UPDATE approvals SET status = 'invalidated' WHERE id = ?",
                    (approval_id,),
                )
                connection.execute(
                    "UPDATE drafts SET status = 'invalidated', updated_at = ? WHERE id = ?",
                    (now, draft["id"]),
                )
                # Persist fail-closed state before surfacing the typed error.
                # The surrounding context manager's rollback is then a no-op.
                connection.commit()
                raise ScopeViolationError("approval no longer matches exact draft target")
            consumed = connection.execute(
                """UPDATE approvals SET status = 'consumed', consumed_at = ?
                   WHERE id = ? AND status = 'active'""",
                (now, approval_id),
            ).rowcount
            if consumed != 1:
                raise ApprovalInvalidError("approval was concurrently consumed")
            connection.execute(
                """INSERT INTO outbox_items(
                       id, draft_id, approval_id, person_id, connected_account_id,
                       endpoint_id, payload_hash, idempotency_key, created_at, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    outbox_id,
                    draft["id"],
                    approval_id,
                    draft["person_id"],
                    draft["connected_account_id"],
                    draft["endpoint_id"],
                    draft["payload_hash"],
                    idempotency_key,
                    now,
                    now,
                ),
            )
            connection.execute(
                "UPDATE drafts SET status = 'queued', updated_at = ? WHERE id = ?",
                (now, draft["id"]),
            )
            connection.execute(
                "INSERT INTO outbox_events VALUES (?, ?, NULL, 'pending', '{}', ?)",
                (new_id("outboxevent"), outbox_id, now),
            )
            row = connection.execute(
                "SELECT * FROM outbox_items WHERE id = ?", (outbox_id,)
            ).fetchone()
        return dict(row)

    def claim_outbox(self, outbox_id: str, *, claim_token: str, expires_at: str) -> dict[str, Any] | None:
        now = utc_now()
        with self.transaction() as connection:
            expired = connection.execute(
                """SELECT id FROM outbox_items
                   WHERE status = 'in_progress' AND claim_expires_at <= ?""",
                (now,),
            ).fetchall()
            connection.execute(
                """UPDATE outbox_items SET status = 'uncertain', updated_at = ?,
                       error_redacted = 'claim expired; reconciliation required'
                   WHERE status = 'in_progress' AND claim_expires_at <= ?""",
                (now, now),
            )
            for expired_item in expired:
                connection.execute(
                    """INSERT INTO outbox_events VALUES
                       (?, ?, 'in_progress', 'uncertain', ?, ?)""",
                    (
                        new_id("outboxevent"),
                        expired_item["id"],
                        json_text({"reason": "claim_expired", "requires_reconciliation": True}),
                        now,
                    ),
                )
            changed = connection.execute(
                """UPDATE outbox_items SET status = 'in_progress', claim_token = ?,
                       claim_expires_at = ?, attempt_count = attempt_count + 1, updated_at = ?
                   WHERE id = ? AND status = 'pending'""",
                (claim_token, expires_at, now, outbox_id),
            ).rowcount
            if changed != 1:
                return None
            row = connection.execute(
                """SELECT o.*, d.payload FROM outbox_items o
                   JOIN drafts d ON d.id = o.draft_id WHERE o.id = ?""",
                (outbox_id,),
            ).fetchone()
            connection.execute(
                "INSERT INTO outbox_events VALUES (?, ?, 'pending', 'in_progress', ?, ?)",
                (new_id("outboxevent"), outbox_id, json_text({"claim_token": claim_token}), now),
            )
        return dict(row)

    def complete_outbox(
        self,
        outbox_id: str,
        *,
        claim_token: str,
        status: str,
        evidence: dict[str, Any],
        error_redacted: str | None = None,
    ) -> None:
        if status == "sent" and not evidence.get("observed"):
            raise ValueError("sent requires observed postcondition evidence")
        if status not in {"sent", "failed", "uncertain"}:
            raise ValueError("invalid terminal outbox status")
        now = utc_now()
        with self.transaction() as connection:
            changed = connection.execute(
                """UPDATE outbox_items SET status = ?, postcondition_json = ?,
                       error_redacted = ?, claim_token = NULL, claim_expires_at = NULL,
                       updated_at = ? WHERE id = ? AND status = 'in_progress'
                       AND claim_token = ?""",
                (status, json_text(evidence), error_redacted, now, outbox_id, claim_token),
            ).rowcount
            if changed != 1:
                raise ValueError("outbox claim is not owned by caller")
            connection.execute(
                "INSERT INTO outbox_events VALUES (?, ?, 'in_progress', ?, ?, ?)",
                (new_id("outboxevent"), outbox_id, status, json_text(evidence), now),
            )

    def get_outbox(self, outbox_id: str) -> dict[str, Any] | None:
        with self.read_connection() as connection:
            row = connection.execute(
                "SELECT * FROM outbox_items WHERE id = ?", (outbox_id,)
            ).fetchone()
        return self._row(row)

    def get_group(self, group_id: str) -> dict[str, Any] | None:
        with self.read_connection() as connection:
            group = connection.execute(
                "SELECT * FROM contact_groups WHERE id = ?", (group_id,)
            ).fetchone()
        if group is None:
            return None
        return {"group": dict(group), **self.group_preview(group_id)}

    def list_route_audit(self, person_id: str | None = None) -> list[dict[str, Any]]:
        where = "WHERE person_id = ?" if person_id else ""
        params = (person_id,) if person_id else ()
        with self.read_connection() as connection:
            rows = connection.execute(
                f"SELECT * FROM route_audit {where} ORDER BY created_at DESC, id DESC",
                params,
            ).fetchall()
        return self._rows(rows)

    def analyze_conversation(self, conversation_id: str) -> dict[str, Any]:
        """Return explainable evidence signals; never infer hidden psychology."""
        with self.read_connection() as connection:
            conversation = connection.execute(
                "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
            if conversation is None:
                raise KeyError(conversation_id)
            person = connection.execute(
                """SELECT i.person_id FROM contact_endpoints e
                   JOIN platform_identities i ON i.id = e.platform_identity_id
                   WHERE e.id = ?""",
                (conversation["endpoint_id"],),
            ).fetchone()
            if person is None or not person["person_id"]:
                raise ScopeViolationError("conversation is not linked to a canonical person")
            person_id = person["person_id"]
            pii_policy = connection.execute(
                "SELECT pii_policy FROM persons WHERE id = ?", (person_id,)
            ).fetchone()[0]
            if pii_policy == "restricted":
                raise ScopeViolationError(
                    "restricted PII policy forbids conversation body analysis"
                )
            messages = connection.execute(
                """SELECT id, direction, body, sent_at FROM messages
                   WHERE conversation_id = ? ORDER BY sent_at, id""",
                (conversation_id,),
            ).fetchall()
        positive_words = {"thanks", "thank", "great", "good", "спасибо", "хорошо", "рад"}
        negative_words = {"sorry", "problem", "bad", "извини", "проблем", "плохо"}
        positive: list[str] = []
        negative: list[str] = []
        commitments: list[dict[str, Any]] = []
        topic_evidence: dict[str, list[str]] = {}
        stopwords = {
            "about", "after", "before", "could", "should", "would", "there",
            "their", "which", "what", "when", "where", "with", "this", "that",
            "можно", "когда", "тогда", "этого", "который", "потом", "будет",
        }
        for index, row in enumerate(messages):
            lowered = row["body"].casefold()
            tokens = set(lowered.replace("!", " ").replace("?", " ").split())
            for raw_token in tokens:
                token = "".join(character for character in raw_token if character.isalnum() or character in "-_" )
                if len(token) >= 5 and token not in stopwords and not token.isdigit():
                    topic_evidence.setdefault(token, []).append(row["id"])
            if tokens & positive_words:
                positive.append(row["id"])
            if any(word in lowered for word in negative_words):
                negative.append(row["id"])
            if row["direction"] == "incoming" and row["body"].rstrip().endswith("?"):
                answered = any(
                    later["direction"] == "outgoing" for later in messages[index + 1 :]
                )
                if not answered:
                    commitments.append(
                        {"kind": "unanswered_question", "message_id": row["id"], "summary": row["body"]}
                    )
            if any(marker in lowered for marker in ("i will", "обещаю", "сделаю", "пришлю")):
                commitments.append(
                    {"kind": "promise", "message_id": row["id"], "summary": row["body"]}
                )
            if any(marker in lowered for marker in ("agreed", "договорились", "согласовано")):
                commitments.append(
                    {"kind": "agreement", "message_id": row["id"], "summary": row["body"]}
                )
        persisted: list[dict[str, Any]] = []
        now = utc_now()
        with self.transaction() as connection:
            for item in commitments:
                commitment_id = stable_id(
                    "commitment", item["message_id"], item["kind"]
                )
                connection.execute(
                    """INSERT INTO commitments(
                           id, person_id, message_id, kind, summary, status,
                           evidence_json, created_at, updated_at
                       ) VALUES (?, ?, ?, ?, ?, 'open', ?, ?, ?)
                       ON CONFLICT(message_id, kind) WHERE message_id IS NOT NULL
                       DO UPDATE SET summary = excluded.summary,
                           evidence_json = excluded.evidence_json,
                           updated_at = excluded.updated_at""",
                    (
                        commitment_id,
                        person_id,
                        item["message_id"],
                        item["kind"],
                        item["summary"],
                        json_text({"message_id": item["message_id"], "method": "explicit deterministic rule"}),
                        now,
                        now,
                    ),
                )
                persisted.append({**item, "id": commitment_id})
            last_touch = messages[-1]["sent_at"] if messages else None
            next_reason = (
                "open unanswered question"
                if any(item["kind"] == "unanswered_question" for item in commitments)
                else None
            )
            connection.execute(
                """INSERT INTO relationship_states(
                       person_id, last_touch_at, next_action_at,
                       next_action_reason, updated_at
                   ) VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(person_id) DO UPDATE SET
                       last_touch_at = excluded.last_touch_at,
                       next_action_at = COALESCE(excluded.next_action_at, relationship_states.next_action_at),
                       next_action_reason = COALESCE(excluded.next_action_reason, relationship_states.next_action_reason),
                       updated_at = excluded.updated_at""",
                (person_id, last_touch, now if next_reason else None, next_reason, now),
            )
        delta = len(positive) - len(negative)
        tone = "more positive markers" if delta > 0 else "more negative markers" if delta < 0 else "balanced markers"
        midpoint = max(1, len(messages) // 2)
        first_ids = {row["id"] for row in messages[:midpoint]}
        first_score = sum(message_id in first_ids for message_id in positive) - sum(
            message_id in first_ids for message_id in negative
        )
        second_score = (len(positive) - len(negative)) - first_score
        important_topics = [
            {"topic": topic, "message_ids": sorted(set(ids)), "count": len(ids)}
            for topic, ids in sorted(
                topic_evidence.items(), key=lambda pair: (-len(pair[1]), pair[0])
            )[:10]
        ]
        return {
            "conversation_id": conversation_id,
            "person_id": person_id,
            "commitments": persisted,
            "important_topics": important_topics,
            "tone_signal": {
                "label": tone,
                "positive_evidence_message_ids": positive,
                "negative_evidence_message_ids": negative,
                "first_half_score": first_score,
                "second_half_score": second_score,
                "change": second_score - first_score,
                "method": "transparent keyword counts; not a psychological classification",
            },
            "message_count": len(messages),
        }

    def daily_brief(self, for_date: str | None = None) -> dict[str, Any]:
        date_value = for_date or utc_now()[:10]
        with self.read_connection() as connection:
            people = connection.execute(
                """SELECT p.id, p.display_name, p.timezone, r.priority,
                          r.last_touch_at, r.next_action_at, r.next_action_reason
                   FROM persons p LEFT JOIN relationship_states r ON r.person_id = p.id
                   WHERE (r.next_action_at IS NOT NULL AND substr(r.next_action_at, 1, 10) <= ?)
                      OR EXISTS (SELECT 1 FROM commitments c
                                 WHERE c.person_id = p.id AND c.status = 'open')
                   ORDER BY COALESCE(r.priority, 0) DESC, p.display_name, p.id""",
                (date_value,),
            ).fetchall()
            items: list[dict[str, Any]] = []
            for person in people:
                commitments = connection.execute(
                    """SELECT id, kind, summary, due_at FROM commitments
                       WHERE person_id = ? AND status = 'open'
                       ORDER BY COALESCE(due_at, '9999'), id""",
                    (person["id"],),
                ).fetchall()
                reasons = [row["kind"] for row in commitments]
                if person["next_action_reason"]:
                    reasons.append(person["next_action_reason"])
                items.append(
                    {
                        **dict(person),
                        "open_commitments": self._rows(commitments),
                        "why": sorted(set(reasons)),
                    }
                )
        return {"date": date_value, "items": items, "explainable": True}

    def plan_greetings(self, for_date: str | None = None) -> dict[str, Any]:
        from datetime import date
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

        requested = date.fromisoformat(for_date) if for_date else None
        with self.read_connection() as connection:
            rows = connection.execute(
                """SELECT e.*, p.timezone, p.display_name FROM contact_events e
                   JOIN persons p ON p.id = e.person_id
                   WHERE e.event_type = 'birthday' ORDER BY p.id, e.id"""
            ).fetchall()
            excluded_people = {
                row[0]
                for row in connection.execute(
                    """SELECT DISTINCT m.person_id FROM contact_group_members m
                       JOIN contact_groups g ON g.id = m.group_id WHERE g.exclusion = 1"""
                ).fetchall()
            }
        planned: list[dict[str, Any]] = []
        now = datetime.now(UTC)
        for row in rows:
            try:
                local_date = requested or now.astimezone(ZoneInfo(row["timezone"])).date()
            except ZoneInfoNotFoundError:
                local_date = requested or now.date()
            event_date = date.fromisoformat(row["happened_at"][:10])
            if (event_date.month, event_date.day) != (local_date.month, local_date.day):
                continue
            status = "excluded" if row["person_id"] in excluded_people else "planned"
            reason = "member of exclusion group" if status == "excluded" else "birthday in person's local timezone"
            delivery_id = stable_id("greeting", row["person_id"], row["id"], local_date.isoformat())
            with self.transaction() as connection:
                connection.execute(
                    """INSERT INTO greeting_deliveries(
                           id, person_id, event_id, local_date, status, reason, created_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(person_id, event_id, local_date) DO NOTHING""",
                    (delivery_id, row["person_id"], row["id"], local_date.isoformat(), status, reason, utc_now()),
                )
                delivery = connection.execute(
                    """SELECT * FROM greeting_deliveries
                       WHERE person_id = ? AND event_id = ? AND local_date = ?""",
                    (row["person_id"], row["id"], local_date.isoformat()),
                ).fetchone()
            planned.append(dict(delivery))
        return {"date": requested.isoformat() if requested else None, "items": planned}

    def list_greetings(self, for_date: str | None = None) -> list[dict[str, Any]]:
        where = "WHERE local_date = ?" if for_date else ""
        params = (for_date,) if for_date else ()
        with self.read_connection() as connection:
            rows = connection.execute(
                f"SELECT * FROM greeting_deliveries {where} ORDER BY local_date, person_id, id",
                params,
            ).fetchall()
        return self._rows(rows)

    def merge_people(
        self,
        winner_person_id: str,
        merged_person_id: str,
        *,
        actor: str,
        evidence: dict[str, Any],
    ) -> dict[str, Any]:
        if winner_person_id == merged_person_id:
            raise ValueError("winner and merged person must differ")
        if not evidence:
            raise ValueError("manual merge evidence is required")
        audit_id = new_id("mergeaudit")
        now = utc_now()
        with self.transaction() as connection:
            winner = connection.execute("SELECT * FROM persons WHERE id = ?", (winner_person_id,)).fetchone()
            merged = connection.execute("SELECT * FROM persons WHERE id = ?", (merged_person_id,)).fetchone()
            if winner is None or merged is None:
                raise KeyError("winner or merged person does not exist")
            identity_ids = [row[0] for row in connection.execute(
                "SELECT id FROM platform_identities WHERE person_id = ?", (merged_person_id,)
            ).fetchall()]
            event_ids = [row[0] for row in connection.execute(
                "SELECT id FROM contact_events WHERE person_id = ?", (merged_person_id,)
            ).fetchall()]
            commitment_ids = [row[0] for row in connection.execute(
                "SELECT id FROM commitments WHERE person_id = ?", (merged_person_id,)
            ).fetchall()]
            group_rows = connection.execute(
                """SELECT m.group_id,
                          EXISTS(SELECT 1 FROM contact_group_members w
                                 WHERE w.group_id = m.group_id AND w.person_id = ?) AS winner_had
                   FROM contact_group_members m WHERE m.person_id = ?""",
                (winner_person_id, merged_person_id),
            ).fetchall()
            merged_journey = stable_id("journey", merged_person_id)
            winner_journey = stable_id("journey", winner_person_id)
            merged_journey_had = connection.execute(
                "SELECT 1 FROM communication_journeys WHERE id = ?", (merged_journey,)
            ).fetchone() is not None
            winner_journey_had = connection.execute(
                "SELECT 1 FROM communication_journeys WHERE id = ?", (winner_journey,)
            ).fetchone() is not None
            episode_ids = [row[0] for row in connection.execute(
                "SELECT id FROM channel_episodes WHERE journey_id = ?", (merged_journey,)
            ).fetchall()]
            transition_ids = [row[0] for row in connection.execute(
                "SELECT id FROM channel_transitions WHERE journey_id = ?", (merged_journey,)
            ).fetchall()]
            route_ids = [row[0] for row in connection.execute(
                "SELECT id FROM person_channel_routes WHERE person_id = ?", (merged_person_id,)
            ).fetchall()]
            preference_rows = connection.execute(
                """SELECT p.endpoint_id, p.state, p.updated_at,
                          w.state AS winner_state, w.updated_at AS winner_updated_at
                   FROM channel_preferences p
                   LEFT JOIN channel_preferences w
                     ON w.person_id = ? AND w.endpoint_id = p.endpoint_id
                   WHERE p.person_id = ?""",
                (winner_person_id, merged_person_id),
            ).fetchall()
            snapshot = {
                "merged_person": dict(merged),
                "identity_ids": identity_ids,
                "event_ids": event_ids,
                "commitment_ids": commitment_ids,
                "groups": [{"group_id": row[0], "winner_had": bool(row[1])} for row in group_rows],
                "episode_ids": episode_ids,
                "transition_ids": transition_ids,
                "route_ids": route_ids,
                "merged_journey_had": merged_journey_had,
                "winner_journey_had": winner_journey_had,
                "preferences": [dict(row) for row in preference_rows],
            }
            connection.execute("UPDATE platform_identities SET person_id = ? WHERE person_id = ?", (winner_person_id, merged_person_id))
            connection.execute("UPDATE contact_events SET person_id = ? WHERE person_id = ?", (winner_person_id, merged_person_id))
            connection.execute("UPDATE commitments SET person_id = ? WHERE person_id = ?", (winner_person_id, merged_person_id))
            for group in group_rows:
                connection.execute(
                    "INSERT OR IGNORE INTO contact_group_members VALUES (?, ?, ?)",
                    (group[0], winner_person_id, now),
                )
            connection.execute("DELETE FROM contact_group_members WHERE person_id = ?", (merged_person_id,))
            if merged_journey_had:
                connection.execute(
                    "INSERT OR IGNORE INTO communication_journeys VALUES (?, ?, ?, ?)",
                    (winner_journey, winner_person_id, now, now),
                )
                connection.execute("UPDATE channel_episodes SET journey_id = ? WHERE journey_id = ?", (winner_journey, merged_journey))
                connection.execute("UPDATE channel_transitions SET journey_id = ? WHERE journey_id = ?", (winner_journey, merged_journey))
                connection.execute("DELETE FROM communication_journeys WHERE id = ?", (merged_journey,))
            connection.execute("UPDATE person_channel_routes SET person_id = ? WHERE person_id = ?", (winner_person_id, merged_person_id))
            for preference in preference_rows:
                connection.execute(
                    "INSERT OR IGNORE INTO channel_preferences VALUES (?, ?, ?, ?)",
                    (
                        winner_person_id,
                        preference["endpoint_id"],
                        preference["state"],
                        preference["updated_at"],
                    ),
                )
            connection.execute("DELETE FROM channel_preferences WHERE person_id = ?", (merged_person_id,))
            connection.execute(
                """INSERT INTO identity_merge_audit VALUES
                   (?, ?, ?, ?, ?, ?, 'merge', ?)""",
                (audit_id, winner_person_id, merged_person_id, actor, json_text(evidence), json_text(snapshot), now),
            )
        return {"audit_id": audit_id, "winner_person_id": winner_person_id, "merged_person_id": merged_person_id, "action": "merge"}

    def unmerge_people(self, merge_audit_id: str, *, actor: str) -> dict[str, Any]:
        now = utc_now()
        audit_id = new_id("mergeaudit")
        with self.transaction() as connection:
            merge = connection.execute(
                "SELECT * FROM identity_merge_audit WHERE id = ? AND action = 'merge'",
                (merge_audit_id,),
            ).fetchone()
            if merge is None:
                raise KeyError(merge_audit_id)
            prior = connection.execute(
                """SELECT 1 FROM identity_merge_audit WHERE action = 'unmerge'
                   AND json_extract(evidence_json, '$.merge_audit_id') = ?""",
                (merge_audit_id,),
            ).fetchone()
            if prior:
                raise ValueError("merge has already been reversed")
            snapshot = json_value(merge["snapshot_json"], {})
            loser = merge["merged_person_id"]
            winner = merge["winner_person_id"]
            loser_journey = stable_id("journey", loser)
            winner_journey = stable_id("journey", winner)
            if snapshot.get("merged_journey_had"):
                connection.execute(
                    "INSERT OR IGNORE INTO communication_journeys VALUES (?, ?, ?, ?)",
                    (loser_journey, loser, now, now),
                )
            for identity_id in snapshot.get("identity_ids", []):
                connection.execute("UPDATE platform_identities SET person_id = ? WHERE id = ?", (loser, identity_id))
            for event_id in snapshot.get("event_ids", []):
                connection.execute("UPDATE contact_events SET person_id = ? WHERE id = ?", (loser, event_id))
            for commitment_id in snapshot.get("commitment_ids", []):
                connection.execute("UPDATE commitments SET person_id = ? WHERE id = ?", (loser, commitment_id))
            for episode_id in snapshot.get("episode_ids", []):
                connection.execute("UPDATE channel_episodes SET journey_id = ? WHERE id = ?", (loser_journey, episode_id))
            for transition_id in snapshot.get("transition_ids", []):
                connection.execute("UPDATE channel_transitions SET journey_id = ? WHERE id = ?", (loser_journey, transition_id))
            for route_id in snapshot.get("route_ids", []):
                connection.execute("UPDATE person_channel_routes SET person_id = ? WHERE id = ?", (loser, route_id))
            for group in snapshot.get("groups", []):
                connection.execute(
                    "INSERT OR IGNORE INTO contact_group_members VALUES (?, ?, ?)",
                    (group["group_id"], loser, now),
                )
                if not group["winner_had"]:
                    connection.execute(
                        "DELETE FROM contact_group_members WHERE group_id = ? AND person_id = ?",
                        (group["group_id"], winner),
                    )
            preferences = snapshot.get("preferences")
            if preferences is None:  # Backward-compatible rollback of v2 snapshots.
                preferences = [
                    {"endpoint_id": endpoint_id, "state": "active", "updated_at": now,
                     "winner_state": None, "winner_updated_at": None}
                    for endpoint_id in snapshot.get("preference_endpoints", [])
                ]
            for preference in preferences:
                endpoint_id = preference["endpoint_id"]
                connection.execute(
                    "INSERT OR REPLACE INTO channel_preferences VALUES (?, ?, ?, ?)",
                    (loser, endpoint_id, preference["state"], preference["updated_at"]),
                )
                if preference.get("winner_state") is None:
                    connection.execute(
                        "DELETE FROM channel_preferences WHERE person_id = ? AND endpoint_id = ?",
                        (winner, endpoint_id),
                    )
                else:
                    connection.execute(
                        "INSERT OR REPLACE INTO channel_preferences VALUES (?, ?, ?, ?)",
                        (
                            winner,
                            endpoint_id,
                            preference["winner_state"],
                            preference["winner_updated_at"],
                        ),
                    )
            if snapshot.get("merged_journey_had") and not snapshot.get("winner_journey_had"):
                remaining = connection.execute(
                    """SELECT EXISTS(SELECT 1 FROM channel_episodes WHERE journey_id = ?)
                              OR EXISTS(SELECT 1 FROM channel_transitions WHERE journey_id = ?)""",
                    (winner_journey, winner_journey),
                ).fetchone()[0]
                if not remaining:
                    connection.execute(
                        "DELETE FROM communication_journeys WHERE id = ?", (winner_journey,)
                    )
            evidence = {"merge_audit_id": merge_audit_id}
            connection.execute(
                """INSERT INTO identity_merge_audit VALUES
                   (?, ?, ?, ?, ?, ?, 'unmerge', ?)""",
                (audit_id, winner, loser, actor, json_text(evidence), merge["snapshot_json"], now),
            )
        return {"audit_id": audit_id, "merge_audit_id": merge_audit_id, "action": "unmerge"}

    def list_open_issues(self) -> list[dict[str, Any]]:
        with self.read_connection() as connection:
            rows = connection.execute(
                """SELECT * FROM sync_issues
                   WHERE status IN ('open', 'retrying', 'quarantined')
                   ORDER BY created_at DESC"""
            ).fetchall()
        return self._rows(rows)
