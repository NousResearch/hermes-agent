"""Tenant-qualified principal and alias authority."""

from __future__ import annotations

import json
import sqlite3
from typing import Iterable

from .errors import IdentityConflict, UnknownIdentity
from .ledger import HermesTagLedger
from .model import ExternalIdentity, Principal, canonical_json, new_id, utc_text


class IdentityStore:
    def __init__(self, ledger: HermesTagLedger) -> None:
        self.ledger = ledger

    def create_principal(
        self,
        display_name: str,
        *,
        roles: Iterable[str] = (),
        guest: bool = False,
        principal_id: str | None = None,
    ) -> Principal:
        principal = Principal(
            principal_id=principal_id or new_id("principal"),
            display_name=display_name,
            roles=tuple(roles),
            guest=guest,
        )
        with self.ledger.transaction() as connection:
            connection.execute(
                """
                INSERT INTO hermes_tag_principals(
                    principal_id, display_name, roles_json, guest, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    principal.principal_id,
                    principal.display_name,
                    canonical_json(principal.roles),
                    1 if principal.guest else 0,
                    principal.created_at,
                ),
            )
        self.ledger.append_receipt(
            event_id=new_id("event"),
            kind="principal.created",
            payload={
                "principal_id": principal.principal_id,
                "roles": principal.roles,
                "guest": principal.guest,
            },
        )
        return principal

    def get_principal(self, principal_id: str) -> Principal:
        connection = self.ledger.connection()
        try:
            row = connection.execute(
                "SELECT * FROM hermes_tag_principals WHERE principal_id=?",
                (principal_id,),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            raise UnknownIdentity(f"unknown principal {principal_id}")
        return Principal(
            principal_id=row["principal_id"],
            display_name=row["display_name"],
            roles=tuple(json.loads(row["roles_json"])),
            guest=bool(row["guest"]),
            created_at=row["created_at"],
        )

    def bind_alias(
        self,
        identity: ExternalIdentity,
        principal_id: str,
        *,
        allow_rebind: bool = False,
    ) -> Principal:
        principal = self.get_principal(principal_id)
        timestamp = utc_text()
        previous_principal: str | None = None
        with self.ledger.transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM hermes_tag_aliases WHERE alias_key=?",
                (identity.key,),
            ).fetchone()
            if existing is not None and existing["revoked_at"] is None:
                previous_principal = existing["principal_id"]
                if previous_principal == principal_id:
                    return principal
                if not allow_rebind:
                    raise IdentityConflict(
                        "external identity is already bound to another principal"
                    )
                connection.execute(
                    "UPDATE hermes_tag_aliases SET revoked_at=? WHERE alias_key=?",
                    (timestamp, identity.key),
                )
                historical_key = f"{identity.key}:{timestamp}"
                connection.execute(
                    "UPDATE hermes_tag_aliases SET alias_key=? WHERE alias_key=?",
                    (historical_key, identity.key),
                )
            try:
                connection.execute(
                    """
                    INSERT INTO hermes_tag_aliases(
                        alias_key, platform, profile, scope_id, external_id,
                        display_name, principal_id, bound_at, revoked_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)
                    """,
                    (
                        identity.key,
                        identity.platform,
                        identity.profile,
                        identity.scope_id,
                        identity.external_id,
                        identity.display_name,
                        principal_id,
                        timestamp,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise IdentityConflict("alias uniqueness conflict") from exc
        self.ledger.append_receipt(
            event_id=new_id("event"),
            kind="identity.rebound" if previous_principal else "identity.bound",
            payload={
                "alias_key": identity.key,
                "principal_id": principal_id,
                "previous_principal_id": previous_principal,
                "platform": identity.platform,
                "profile": identity.profile,
                "scope_id_digest": identity.key[:16],
            },
        )
        return principal

    def resolve(self, identity: ExternalIdentity) -> Principal:
        connection = self.ledger.connection()
        try:
            row = connection.execute(
                """
                SELECT p.*
                FROM hermes_tag_aliases a
                JOIN hermes_tag_principals p
                  ON p.principal_id = a.principal_id
                WHERE a.alias_key=? AND a.revoked_at IS NULL
                """,
                (identity.key,),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            raise UnknownIdentity("external identity has no active principal binding")
        return Principal(
            principal_id=row["principal_id"],
            display_name=row["display_name"],
            roles=tuple(json.loads(row["roles_json"])),
            guest=bool(row["guest"]),
            created_at=row["created_at"],
        )

    def resolve_or_guest(
        self,
        identity: ExternalIdentity,
        *,
        allow_guest: bool,
    ) -> Principal:
        try:
            return self.resolve(identity)
        except UnknownIdentity:
            if not allow_guest:
                raise
        display_name = identity.display_name or f"guest-{identity.key[:8]}"
        principal = self.create_principal(display_name, guest=True, roles=("guest",))
        try:
            return self.bind_alias(identity, principal.principal_id)
        except IdentityConflict:
            return self.resolve(identity)

    def revoke_alias(self, identity: ExternalIdentity) -> None:
        timestamp = utc_text()
        with self.ledger.transaction() as connection:
            changed = connection.execute(
                """
                UPDATE hermes_tag_aliases
                SET revoked_at=?
                WHERE alias_key=? AND revoked_at IS NULL
                """,
                (timestamp, identity.key),
            ).rowcount
        if not changed:
            raise UnknownIdentity("external identity has no active binding")
        self.ledger.append_receipt(
            event_id=new_id("event"),
            kind="identity.revoked",
            payload={"alias_key": identity.key},
        )
