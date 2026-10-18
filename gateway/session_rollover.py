"""SessionStore receipt lifecycle and fresh-identity rollover cutover."""

from __future__ import annotations

import contextlib
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional
from gateway.session_persistence import _DB_UNPINNED

if TYPE_CHECKING:
    from gateway.session import SessionEntry, SessionSource

logger = logging.getLogger("gateway.session")
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class SessionRolloverMixin:
    """Native SessionDB-backed rollover APIs and receipt-first stale-route recovery."""

    @staticmethod
    def _rollover_profile_home(db) -> str:
        return str(Path(db.db_path).resolve().parent)

    def _rollover_db(self, session_key: str):
        db = self._db_for_key(session_key)
        if db is None or not callable(getattr(db, "begin_gateway_rollover", None)):
            raise RuntimeError("Native SQLite session storage is required for automatic rollover")
        return db

    def _rollover_route_identity(self, entry: SessionEntry) -> dict[str, Any]:
        if entry.origin is None or entry.platform is None:
            raise ValueError("Rollover requires a resolved native routing identity")
        from gateway.session import sanitize_model_override
        from gateway.session_prompt_pin import sanitize_prompt_pin

        route_entry = entry.to_dict()
        route_entry.pop("active_turn_token", None)
        route_entry.pop("active_turn_started_at", None)
        return {
            "origin": entry.origin.to_dict(),
            "platform": entry.platform.value,
            "model_override": sanitize_model_override(entry.model_override),
            "prompt_pin": sanitize_prompt_pin(entry.prompt_pin),
            "transport_profile": entry.transport_profile,
            "route_entry": route_entry,
        }

    def _validate_rollover_home(self, db, profile_home: str) -> str:
        if not isinstance(profile_home, str) or not profile_home:
            raise ValueError("Rollover profile_home is required")
        actual_home = self._rollover_profile_home(db)
        if str(Path(profile_home).resolve()) != actual_home:
            raise ValueError("Rollover profile_home does not own this session database")
        return actual_home

    @staticmethod
    def _invalidate_stale_rollover(db, session_key: str, receipt: dict[str, Any], reason: str) -> None:
        """CAS-invalidate only the exact attempt whose route snapshot went stale."""
        try:
            db.set_gateway_rollover_phase(
                session_key, receipt["new_session_id"], receipt["phase"], "invalidated", reason,
            )
        except ValueError:
            # A newer receipt or phase won the database transaction before this stale writer.
            pass

    def begin_rollover(
        self, session_key: str, expected_session_id: str, generation: Any, *,
        profile_home: str, packet_path: str,
    ) -> dict[str, Any]:
        """Durably claim the old route and allocate its sole successor before packet authoring."""
        if not session_key or not expected_session_id:
            raise ValueError("Rollover authoring requires a session key and predecessor id")
        if not isinstance(packet_path, str) or not Path(packet_path).is_absolute():
            raise ValueError("Rollover packet_path must be absolute")
        self._ensure_loaded()
        with self._lock:
            entry = self._entries.get(session_key)
            if entry is None or entry.session_id != expected_session_id:
                raise ValueError("Rollover route changed before authoring")
            route_identity = self._rollover_route_identity(entry)

        db = self._rollover_db(session_key)
        owner_home = self._validate_rollover_home(db, profile_home)
        receipt = db.begin_gateway_rollover({
            "session_key": session_key,
            "old_session_id": expected_session_id,
            "generation": generation,
            "profile_home": owner_home,
            "packet_path": packet_path,
            "reason": "",
            "phase": "authoring",
            "owner_committed": False,
            "route_identity": route_identity,
        })
        with self._lock:
            current = self._entries.get(session_key)
            if current is entry and current.session_id == expected_session_id:
                return receipt
        self._invalidate_stale_rollover(
            db, session_key, receipt, "The lane route changed during rollover authoring.",
        )
        raise ValueError("Rollover route changed during authoring")

    def prepare_rollover(
        self, session_key: str, expected_session_id: str, generation: Any, binding: dict, *,
        expected_new_session_id: str,
    ) -> dict[str, Any]:
        """Seal the authoring receipt with the immutable packet and exact restart/bootstrap text."""
        if not isinstance(binding, dict):
            raise TypeError("Rollover binding must be a dict")
        packet_path = binding.get("packet_path")
        digest = binding.get("packet_sha256")
        restart_message = binding.get("restart_message")
        bootstrap_prompt = binding.get("bootstrap_prompt")
        if not isinstance(packet_path, str) or not Path(packet_path).is_absolute():
            raise ValueError("Rollover packet_path must be absolute")
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise ValueError("Rollover packet_sha256 must be a SHA-256 hex digest")
        if not isinstance(restart_message, str) or not restart_message:
            raise ValueError("Rollover restart_message must be non-empty exact text")
        if (
            not isinstance(bootstrap_prompt, str)
            or not bootstrap_prompt.endswith("\n\n" + restart_message)
            or not bootstrap_prompt[: -(len(restart_message) + 2)].strip()
        ):
            raise ValueError("Rollover bootstrap_prompt must bind the exact restart_message")
        if not session_key or not expected_session_id:
            raise ValueError("Rollover preparation requires a session key and predecessor id")
        self._ensure_loaded()
        with self._lock:
            entry = self._entries.get(session_key)
            if entry is None or entry.session_id != expected_session_id:
                raise ValueError("Rollover route changed before preparation")

        db = self._rollover_db(session_key)
        owner_home = self._validate_rollover_home(db, binding.get("profile_home"))
        receipt = db.prepare_gateway_rollover({
            "session_key": session_key,
            "old_session_id": expected_session_id,
            "generation": generation,
            "profile_home": owner_home,
            "packet_path": packet_path,
            "packet_sha256": digest,
            "restart_message": restart_message,
            "bootstrap_prompt": bootstrap_prompt,
            "reason": binding.get("reason") or "",
        }, expected_new_session_id=expected_new_session_id)
        with self._lock:
            current = self._entries.get(session_key)
            if current is entry and current.session_id == expected_session_id:
                return receipt
        self._invalidate_stale_rollover(
            db, session_key, receipt, "The lane route changed during rollover preparation.",
        )
        raise ValueError("Rollover route changed during preparation")

    def commit_rollover(
        self, session_key: str, expected_session_id: str, generation: Any, *,
        expected_new_session_id: str,
    ) -> SessionEntry:
        """Transfer DB ownership, then publish its committed successor only if the route snapshot holds."""
        if not session_key or not expected_session_id:
            raise ValueError("Rollover commit requires a session key and predecessor id")
        self._ensure_loaded()
        with self._lock:
            observed = self._entries.get(session_key)

        db = self._rollover_db(session_key)
        receipt = db.get_gateway_rollover(session_key)
        if receipt is None:
            raise ValueError("No prepared rollover receipt exists")
        if receipt.get("new_session_id") != expected_new_session_id:
            raise ValueError("Rollover attempt changed before commit")
        if receipt.get("old_session_id") != expected_session_id:
            raise ValueError("Rollover predecessor changed before commit")
        if receipt.get("generation") != generation:
            raise ValueError("Rollover generation changed before commit")

        with self._lock:
            current = self._entries.get(session_key)
            route_changed = current is not observed or (
                current is not None and current.session_id not in {
                    expected_session_id, expected_new_session_id,
                }
            )
            if (
                not route_changed and receipt.get("owner_committed") is True
                and current is not None and current.session_id == expected_new_session_id
                and current.prev_session_id == expected_session_id
            ):
                return current
        if route_changed:
            if not receipt.get("owner_committed"):
                self._invalidate_stale_rollover(
                    db, session_key, receipt, "The lane route changed before rollover cutover.",
                )
            raise ValueError("Rollover route changed before commit")

        committed = db.commit_gateway_rollover(
            session_key, expected_session_id, generation,
            expected_new_session_id=expected_new_session_id,
        )
        successor = self._entry_from_rollover_receipt(committed, owner_committed=True)
        with self._lock:
            current = self._entries.get(session_key)
            if current is not observed:
                if (
                    current is not None and current.session_id == expected_new_session_id
                    and current.prev_session_id == expected_session_id
                ):
                    return current
                raise ValueError("Rollover route changed after database cutover")
            if current is not None and current.session_id not in {
                expected_session_id, expected_new_session_id,
            }:
                raise ValueError("Rollover route changed after database cutover")
            self._entries[session_key] = successor
        try:
            with self._lock:
                routing_data, routing_generation = self._snapshot_routing_locked()
            self._persist_routing_data(routing_data, routing_generation)
        except Exception as exc:
            # The DB cutover and receipt are already committed; route recovery must replay this same ID.
            logger.warning(
                "Rollover route publication failed after commit for %r; successor %s is durable: %s",
                session_key, successor.session_id, exc, exc_info=True,
            )
        return successor

    def get_rollover(self, session_key: str) -> Optional[dict[str, Any]]:
        if not session_key:
            return None
        db = self._db_for_key(session_key)
        getter = getattr(db, "get_gateway_rollover", None) if db else None
        return getter(session_key) if callable(getter) else None

    def set_rollover_phase(
        self, session_key: str, new_session_id: str, expected_phase: str, phase: str, reason: str = "",
    ) -> dict[str, Any]:
        db = self._rollover_db(session_key)
        return db.set_gateway_rollover_phase(
            session_key, new_session_id, expected_phase, phase, reason,
        )

    def list_pending_rollovers(
        self, *, profile_homes=(), recover_interrupted: bool = False,
    ) -> list[dict[str, Any]]:
        """List pending receipts in served homes; only startup classifies interrupted work."""
        self._ensure_loaded()
        with self._lock:
            keys = list(self._entries)
        allowed_homes = {str(Path(home).resolve()) for home in profile_homes}
        if allowed_homes:
            for home in allowed_homes:
                db_path = Path(home) / "state.db"
                if db_path.is_file():
                    self._open_session_db_for_active_scope(db_path=db_path)
        else:
            for key in keys:
                self._db_for_key(key)
        with getattr(self, "_db_handles_lock", self._lock):
            databases = list(dict.fromkeys(db for db in getattr(self, "_db_handles", {}).values() if db))
        pinned = self._pinned_db()
        if pinned is not _DB_UNPINNED and pinned is not None:
            databases.append(pinned)
        receipts: dict[tuple[str, str], dict[str, Any]] = {}
        recovered = False
        for db in dict.fromkeys(databases):
            if allowed_homes and self._rollover_profile_home(db) not in allowed_homes:
                continue
            lister = getattr(db, "list_gateway_rollovers", None)
            if not callable(lister):
                continue
            for receipt in lister():
                if receipt["phase"] in {"complete", "invalidated"}:
                    continue
                receipt, changed = self._reconcile_rollover_receipt(db, receipt, recover_interrupted)
                recovered = recovered or changed
                if receipt is not None:
                    receipt_key = (str(Path(db.db_path).resolve()), receipt["session_key"])
                    receipts[receipt_key] = receipt
        if recovered:
            self._save_entries()
        return list(receipts.values())

    def _reconcile_rollover_receipt(self, db, receipt, recover_interrupted):
        key = receipt["session_key"]
        owner_db = self._db_for_key(key)
        if owner_db is None or Path(owner_db.db_path).resolve() != Path(db.db_path).resolve():
            raise RuntimeError("Rollover receipt is stored in the wrong profile database")
        if self._validate_rollover_home(db, receipt.get("profile_home")) != receipt["profile_home"]:
            raise RuntimeError("Rollover receipt profile does not match its owning database")
        with self._lock:
            current = self._entries.get(key)
        if current is not None and current.session_id not in {
            receipt.get("old_session_id"), receipt.get("new_session_id"),
        }:
            self._invalidate_stale_rollover(
                db, key, receipt, "The lane route changed before rollover recovery.",
            )
            return None, False
        if recover_interrupted and receipt["phase"] in {
            "authoring", "prepared", "bootstrapping", "admitting",
        }:
            reason = (
                "Rollover authoring was interrupted; reconcile before continuing."
                if not receipt.get("owner_committed") else
                "Successor bootstrap or admission was interrupted; reconcile uncertain outcomes."
            )
            receipt = db.set_gateway_rollover_phase(
                key, receipt["new_session_id"], receipt["phase"], "blocked", reason,
            )
        recovered = False
        if receipt["phase"] in {"blocked", "committed", "bootstrapping", "admitting"}:
            route = self._entry_from_rollover_receipt(receipt, owner_committed=bool(receipt.get("owner_committed")))
            if self._rollover_route_is_durable(db, receipt, route.session_id):
                with self._lock:
                    current = self._entries.get(key)
                    route_changed = current is not None and current.session_id not in {
                        receipt.get("old_session_id"), receipt.get("new_session_id"),
                    }
                    already_current = current is not None and current.session_id == route.session_id and (
                        not receipt.get("owner_committed") or current.prev_session_id == receipt["old_session_id"]
                    )
                    if not route_changed and not already_current:
                        self._entries[key] = route
                        recovered = True
                if route_changed:
                    self._invalidate_stale_rollover(
                        db, key, receipt, "The lane route changed during rollover recovery.",
                    )
                    return None, False
            elif receipt.get("owner_committed"):
                successor = db.get_session(receipt["new_session_id"])
                if successor is not None and successor.get("end_reason") == "session_reset":
                    self._invalidate_stale_rollover(
                        db, key, receipt, "The committed successor was explicitly reset.",
                    )
                    return None, False
        return receipt, recovered

    def _entry_from_rollover_receipt(self, receipt: dict[str, Any], *, owner_committed: bool) -> SessionEntry:
        from gateway.session import SessionEntry

        route = receipt.get("route_identity")
        if not isinstance(route, dict) or not isinstance(route.get("origin"), dict):
            raise RuntimeError("Rollover receipt is missing its durable route identity")
        old_id = receipt["old_session_id"]
        if not owner_committed:
            route_entry = route.get("route_entry")
            if not isinstance(route_entry, dict):
                raise RuntimeError("Rollover receipt is missing its predecessor route")
            old_entry = dict(route_entry)
            old_entry["session_key"] = receipt["session_key"]
            old_entry["session_id"] = old_id
            old_entry.pop("active_turn_token", None)
            old_entry.pop("active_turn_started_at", None)
            return SessionEntry.from_dict(old_entry)
        now = datetime.now().isoformat()
        return SessionEntry.from_dict({
            "session_key": receipt["session_key"],
            "session_id": receipt["new_session_id"],
            "created_at": now,
            "updated_at": now,
            "origin": route["origin"],
            "display_name": receipt.get("display_name"),
            "platform": receipt.get("source"),
            "chat_type": receipt.get("chat_type") or route["origin"].get("chat_type", "dm"),
            "transport_profile": receipt.get("transport_profile"),
            "model_override": route.get("model_override"),
            "prompt_pin": route.get("prompt_pin"),
            "prev_session_id": old_id,
            "is_fresh_reset": True,
        })

    @staticmethod
    def _rollover_route_is_durable(db, receipt: dict[str, Any], session_id: str) -> bool:
        old = db.get_session(receipt["old_session_id"])
        if old is None or old.get("session_key") != receipt["session_key"]:
            return False
        if not receipt.get("owner_committed"):
            return old.get("ended_at") is None and session_id == receipt["old_session_id"]
        new = db.get_session(receipt["new_session_id"])
        return bool(
            old.get("end_reason") == "session_reset"
            and new is not None
            and new.get("parent_session_id") == receipt["old_session_id"]
            and new.get("_reset_from") == receipt["old_session_id"]
            and new.get("session_key") == receipt["session_key"]
            and new.get("ended_at") is None
            and session_id == receipt["new_session_id"]
        )

    def _stale_entry_verdict(self, key: str, entry, row):
        """Use the committed receipt before generic stale-peer recovery can lose reset lineage."""
        if row.get("end_reason") == "session_reset" and entry.origin is not None:
            recovered = self._recover_rollover_route(key, entry.origin, lock_held=True)
            if recovered is not None and recovered.session_id != entry.session_id:
                logger.warning(
                    "gateway.session: restoring committed rollover route %r from %s to %s",
                    key, entry.session_id, recovered.session_id,
                )
                return recovered
        return super()._stale_entry_verdict(key, entry, row)


    def _recover_rollover_route(
        self, session_key: str, source: SessionSource, *, lock_held: bool = False,
    ) -> Optional[SessionEntry]:
        """Recover the receipt's owner before timer/reset or stale-row recovery can act."""
        receipt = self.get_rollover(session_key)
        if receipt is None or receipt["phase"] in {"complete", "invalidated"}:
            return None
        with (contextlib.nullcontext() if lock_held else self._lock):
            current = self._entries.get(session_key)
            if current is not None and current.session_id not in {
                receipt["old_session_id"], receipt["new_session_id"],
            }:
                return None
        route_identity = receipt.get("route_identity") or {}
        stored_origin = route_identity.get("origin")
        if not isinstance(stored_origin, dict):
            raise RuntimeError("Rollover receipt is missing its durable route identity")
        expected_origin = type(source).from_dict(stored_origin)
        from gateway.session import is_shared_multi_user_session

        shared_route = is_shared_multi_user_session(
            source,
            group_sessions_per_user=getattr(self.config, "group_sessions_per_user", True),
            thread_sessions_per_user=getattr(self.config, "thread_sessions_per_user", False),
        )
        fields = ("platform", "chat_id", "chat_type", "thread_id", "scope_id", "profile")
        if not shared_route:
            fields += ("user_id",)
        for field in fields:
            if getattr(expected_origin, field, None) != getattr(source, field, None):
                raise RuntimeError("Rollover source no longer matches the bound route")
        if self._generate_session_key(source) != session_key:
            raise RuntimeError("Rollover source no longer maps to the bound native session key")
        from gateway.session_identity import transport_profile_of

        if transport_profile_of(source) != route_identity.get("transport_profile"):
            raise RuntimeError("Rollover source transport no longer matches the bound route")
        owner_committed = bool(receipt.get("owner_committed"))
        db = self._rollover_db(session_key)
        route_session_id = receipt["new_session_id"] if owner_committed else receipt["old_session_id"]
        if not self._rollover_route_is_durable(db, receipt, route_session_id):
            if owner_committed:
                successor = db.get_session(receipt["new_session_id"])
                if (
                    successor is None
                    or successor.get("parent_session_id") != receipt["old_session_id"]
                    or successor.get("_reset_from") != receipt["old_session_id"]
                    or successor.get("session_key") != session_key
                ):
                    raise RuntimeError("Committed rollover receipt has no matching durable successor")
                if successor.get("ended_at") is not None:
                    return None
                raise RuntimeError("Committed rollover receipt has inconsistent predecessor state")
            return None
        recovered = self._entry_from_rollover_receipt(receipt, owner_committed=owner_committed)
        with (contextlib.nullcontext() if lock_held else self._lock):
            current = self._entries.get(session_key)
            if current is not None and current.session_id not in {
                receipt["old_session_id"], receipt["new_session_id"],
            }:
                return None
            if current is not None and current.session_id == route_session_id:
                if not owner_committed or current.prev_session_id == receipt["old_session_id"]:
                    return current
            self._entries[session_key] = recovered
        return recovered
