"""Durable bot-chain delivery routing to the session-owned profile database."""

from typing import Any, Dict, Optional


class BotChainAdmissionUnavailable(RuntimeError):
    """No authoritative durable admission: the caller must execute zero turns."""


class SessionBotChainStoreMixin:
    def bot_chain_publication_guard(self, session_id: str, platform_message_id: str, owner_token: str):
        """Synchronous worker-side publication fence in the receipt-owning DB."""
        db = self._db_for_session_id(session_id)
        if db is None:
            raise BotChainAdmissionUnavailable(f"no owning session store for {session_id}")
        return db.bot_chain_publication_guard(session_id, platform_message_id, owner_token)

    def admit_bot_chain_delivery(
        self, session_id: str, platform_message_id: str, chain_name: str
    ) -> str:
        """Durably admit an inbound bot-chain event; see SessionDB.

        Returns ``"admitted"`` / ``"running"`` / ``"settled"``. Unlike the
        read-only dedupe probe above, errors PROPAGATE: when the receipt
        cannot be persisted the caller must refuse to execute, because
        running the chain without a durable admission row would reopen the
        duplicate-execution window on redelivery. A session with no owning
        durable store (a named profile whose home cannot be resolved, the
        same state ``_append_transcript_message`` refuses to write into)
        raises :class:`BotChainAdmissionUnavailable` — a synthetic
        "admitted" here would execute the chain with no receipt at all and
        re-execute it on every platform redelivery.
        """
        db = self._db_for_session_id(session_id)
        if db is None:
            raise BotChainAdmissionUnavailable(
                f"no owning session store for {session_id}; "
                "refusing bot-chain admission without a durable receipt"
            )
        return db.admit_bot_chain_delivery(session_id, platform_message_id, chain_name)

    def mark_bot_chain_delivery_running(
        self, session_id: str, platform_message_id: str
    ) -> Optional[str]:
        """Atomic execution claim; None/raise means zero model turns.

        Returns the claim's ``owner_token`` on success — required by the
        heartbeat renewal, settlement, and release calls that follow.
        """
        db = self._db_for_session_id(session_id)
        if db is None:
            raise BotChainAdmissionUnavailable(
                f"no owning session store for {session_id}; "
                "cannot durably claim bot-chain execution"
            )
        return db.mark_bot_chain_delivery_running(session_id, platform_message_id)

    def renew_bot_chain_delivery_claim(
        self, session_id: str, platform_message_id: str, owner_token: str
    ) -> bool:
        """Heartbeat: extend the claim lease while execution is in flight.

        Returns False when there is no owning store or the claim is no
        longer ours (reclaimed after an expiry gap).
        """
        db = self._db_for_session_id(session_id)
        if db is None:
            return False
        return db.renew_bot_chain_delivery_claim(
            session_id, platform_message_id, owner_token
        )

    def settle_bot_chain_delivery(
        self,
        session_id: str,
        platform_message_id: str,
        *,
        outcome: str,
        detail: str = "",
        owner_token: Optional[str] = None,
    ) -> bool:
        db = self._db_for_session_id(session_id)
        if db is None:
            return False
        return db.settle_bot_chain_delivery(
            session_id,
            platform_message_id,
            outcome=outcome,
            detail=detail,
            owner_token=owner_token,
        )

    def get_bot_chain_delivery(
        self, session_id: str, platform_message_id: str
    ) -> Optional[Dict[str, Any]]:
        """Read back the admission receipt (authoritative chain identity).

        Returns None when no owning store exists or no receipt was recorded;
        callers must not execute without an authoritative identity readback.
        """
        db = self._db_for_session_id(session_id)
        if db is None:
            return None
        return db.get_bot_chain_delivery(session_id, platform_message_id)

    def release_bot_chain_delivery_claim(
        self,
        session_id: str,
        platform_message_id: str,
        owner_token: Optional[str] = None,
    ) -> bool:
        """Best-effort release of THIS claim's execution row.

        Called from the settlement-failure path: a receipt left ``running``
        would stand redeliveries down until the lease expires. Scoped by
        ``owner_token`` in SessionDB, so a concurrent or newer claim is
        never revoked. Returns False when there is no owning store (no
        receipt exists there either) or no own ``running`` row to release.
        """
        db = self._db_for_session_id(session_id)
        if db is None:
            return False
        return db.release_bot_chain_delivery_claim(
            session_id, platform_message_id, owner_token
        )
