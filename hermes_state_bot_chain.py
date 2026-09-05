"""Bot-chain persistence for SessionDB: delivery admission ledger and the
canonical Bot Chat chain receipt.

Mixin contract: this is a plain mixin class consumed by
``hermes_state.SessionDB``. It defines no ``__init__`` and no state of its
own; methods access the host's attributes (``self._execute_write``,
``self._read_ctx``) established by ``SessionDB.__init__``. It must never
import hermes_state (cycle).

Two durable mechanisms live here:

1. ``bot_chain_deliveries`` — the inbound delivery admission ledger. A
   ``$Bot`` chain triggered by a platform message executes model turns with
   durable side effects (isolated sessions, canonical Bot Chat history).
   At-least-once platform delivery must therefore meet an idempotent
   recipient: the admission row is written BEFORE any model execution and
   is the single authority deciding whether a (re)delivered platform
   message may start a chain. States:

     admitted  — receipt persisted; no execution claim was durably
                 recorded, so NO side effect can have happened yet. A
                 redelivery that finds this state resumes the work under
                 the ORIGINAL chain identity instead of abandoning it.
     running   — a process holds the atomic execution claim (recorded
                 BEFORE the first side effect). The claim is a bounded
                 lease: the owner proves liveness by renewing
                 ``lease_expires_at`` while it executes. A redelivery that
                 finds a live owner inside its lease stands down; an
                 expired lease (host/container identity changed, machine
                 renamed, state directory restored — anything a pid+host
                 pair can never disprove) is reclaimed to ``admitted``,
                 and durable per-step history then decides which side
                 effects already happened.
     settled   — terminal; ``outcome`` binds the chain result
                 (completed | failed | cancelled).

   The first admission binds the chain identity (``chain_name``); later
   (re)deliveries of the same platform message reuse it and never
   overwrite it. Claims, releases, and settlements are scoped by a
   per-claim ``owner_token`` so a stale runtime generation can never
   touch a newer owner's row.

2. The chain receipt (``display_metadata.bot_chain.chain``) stamped onto
   canonical Bot Chat message rows at publish time. Recovery from Bot
   Chat may skip re-execution only on this exact chain-identity match,
   never on prompt text.
"""

import contextlib
import json
import logging
import math
import os
import sqlite3
import time
import uuid
from typing import Any, Dict, Optional, Sequence

# Moved methods logged under the "hermes_state" logger before the split;
# keep that logger identity so log filtering/capture behavior is unchanged.
logger = logging.getLogger("hermes_state")


class BotChainClaimLostError(RuntimeError):
    """The receipt no longer authorizes this runtime to publish a step."""


class SessionBotChainMixin:
    """See module docstring — mixin for SessionDB (Bot-chain cluster)."""

    #: Seconds a ``running`` execution claim stays authoritative without a
    #: renewal. Long enough that a healthy owner renews several times per
    #: window; short enough that a dead runtime generation becomes
    #: reclaimable in minutes, not never.
    BOT_CHAIN_CLAIM_LEASE_SECONDS = 120.0

    #: Key under which a bot-chain receipt lives inside ``display_metadata``.
    #: The receipt is the durable, chain-qualified proof that a canonical
    #: Bot Chat turn came from one exact chain identity
    #: (``{"bot_chain": {"chain": <conversation_name>}}``). Recovery may
    #: skip re-execution only on this identity match, never on prompt text.
    BOT_CHAIN_RECEIPT_METADATA_KEY = "bot_chain"

    @contextlib.contextmanager
    def _bot_chain_fence(self):
        """Serialize receipt ownership changes with bounded local publication.

        A separate SQLite lock file avoids holding state.db's writer while a
        publisher writes another profile (or the SAME state.db). Native SQLite
        locking works across processes and supported filesystems on all OSes;
        the kernel releases it after a crash. Never held during model execution.
        """
        # ponytail: one fence per ingress DB; shard per receipt if publication contention matters.
        conn = sqlite3.connect(f"{self.db_path}.bot-chain-lock", timeout=5, isolation_level=None)
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield
        finally:
            conn.close()

    def _bot_chain_write(self, fn):
        with self._bot_chain_fence():
            return self._execute_write(fn)

    @contextlib.contextmanager
    def bot_chain_publication_guard(self, session_id: str, platform_message_id: str, owner_token: str):
        """Fence the claim check AND publication against concurrent reclaim."""
        with self._bot_chain_fence():
            row = self.get_bot_chain_delivery(session_id, platform_message_id)
            if not (
                owner_token and row and row["state"] == "running"
                and row["owner_token"] == owner_token
                and (row["lease_expires_at"] or 0) > time.time()
            ):
                raise BotChainClaimLostError("Bot-chain execution claim expired or changed owner")
            yield

    _BOT_CHAIN_DELIVERIES_DDL = """
        session_id TEXT NOT NULL,
        platform_message_id TEXT NOT NULL,
        chain_name TEXT NOT NULL,
        state TEXT NOT NULL,
        outcome TEXT,
        detail TEXT,
        owner_pid INTEGER,
        owner_host TEXT,
        owner_token TEXT,
        lease_expires_at REAL,
        admitted_at REAL NOT NULL,
        updated_at REAL NOT NULL,
        PRIMARY KEY (session_id, platform_message_id)
    """

    def _ensure_bot_chain_deliveries_table(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS bot_chain_deliveries ("
            + self._BOT_CHAIN_DELIVERIES_DDL
            + ")"
        )
        # Deployments that already ran a pre-lease build have the table
        # without the newer columns; CREATE IF NOT EXISTS cannot add them,
        # so migrate in place.
        have = {
            row[1]
            for row in conn.execute("PRAGMA table_info('bot_chain_deliveries')")
        }
        for column, ddl in (
            ("owner_pid", "ALTER TABLE bot_chain_deliveries ADD COLUMN owner_pid INTEGER"),
            ("owner_host", "ALTER TABLE bot_chain_deliveries ADD COLUMN owner_host TEXT"),
            ("owner_token", "ALTER TABLE bot_chain_deliveries ADD COLUMN owner_token TEXT"),
            ("lease_expires_at", "ALTER TABLE bot_chain_deliveries ADD COLUMN lease_expires_at REAL"),
        ):
            if column not in have:
                conn.execute(ddl)

    @staticmethod
    def _claim_owner() -> tuple:
        import socket

        return os.getpid(), socket.gethostname()

    @staticmethod
    def _owner_alive(owner_pid, owner_host) -> bool:
        """Conservative liveness check for a recorded claim owner.

        Fails CLOSED: a foreign host, an empty pid, or any OS-level doubt
        reads as alive, because a false reclaim would allow a second
        execution — the exact failure this table exists to prevent. This
        check only ever runs INSIDE an unexpired lease window (a fast
        dead-owner reclaim); bounded staleness is the lease's job, so a
        runtime generation that can never be probed again still expires.
        (PID reuse after death is accepted residual risk inside the
        window: a recycled pid keeps the row claimed until the lease
        lapses, never double-executes.)
        """
        try:
            pid = int(owner_pid) if owner_pid is not None else 0
        except (TypeError, ValueError):
            return True
        if pid <= 0:
            return True
        import socket

        if not owner_host or owner_host != socket.gethostname():
            return True
        try:
            import psutil

            return psutil.pid_exists(pid)
        except ImportError:
            pass
        except Exception:
            return True
        if os.name == "nt":
            return True
        try:
            os.kill(pid, 0)  # windows-footgun: ok — nt returns above
        except ProcessLookupError:
            return False
        except PermissionError:
            return True  # exists, owned by someone else
        except Exception:
            return True
        return True

    def admit_bot_chain_delivery(
        self,
        session_id: str,
        platform_message_id: str,
        chain_name: str,
    ) -> str:
        """Durably admit an inbound bot-chain event, exactly once.

        Returns one of:

        * ``"admitted"`` — the caller holds a fresh or resumed admission and
          MUST win the atomic execution claim via
          :meth:`mark_bot_chain_delivery_running` before any side effect,
          then settle via :meth:`settle_bot_chain_delivery`. Also returned
          when a previous attempt died before its claim (state was still
          ``admitted`` — no side effect can exist), when its claim lease
          expired (a prior runtime generation that stopped renewing), or
          when its claim owner is provably dead; the recorded chain
          identity is reused, never replaced, so recovery continues the
          SAME chain.
        * ``"running"`` — a live owner holds an unexpired execution claim;
          the caller must not execute anything.
        * ``"settled"`` — this platform message already ran to settlement;
          the caller must not execute anything.
        """
        def _do(conn: sqlite3.Connection) -> str:
            # The clock starts inside the write transaction: time spent
            # waiting on the SQLite lock must not come out of the lease
            # window the caller believes it was granted.
            now = time.time()
            self._ensure_bot_chain_deliveries_table(conn)
            cursor = conn.execute(
                "INSERT OR IGNORE INTO bot_chain_deliveries "
                "(session_id, platform_message_id, chain_name, state, "
                "admitted_at, updated_at) VALUES (?, ?, ?, 'admitted', ?, ?)",
                (session_id, platform_message_id, chain_name, now, now),
            )
            if cursor.rowcount:
                return "admitted"
            row = conn.execute(
                "SELECT state, owner_pid, owner_host, lease_expires_at, "
                "updated_at FROM bot_chain_deliveries "
                "WHERE session_id = ? AND platform_message_id = ?",
                (session_id, platform_message_id),
            ).fetchone()
            if row[0] == "settled":
                return "settled"
            if row[0] == "running":
                # The lease is the bounded staleness authority. Legacy rows
                # written before the lease columns existed have no recorded
                # expiry; their last update acts as an implicit lease start,
                # so even they eventually become reclaimable instead of
                # wedging the receipt forever.
                lease_expires_at = row[3]
                if lease_expires_at is None:
                    lease_expires_at = row[4] + self.BOT_CHAIN_CLAIM_LEASE_SECONDS
                if lease_expires_at > now and self._owner_alive(row[1], row[2]):
                    return "running"
            # Resume/reclaim: reset to a fresh admission under the ORIGINAL
            # chain identity. The next step for the caller is the atomic
            # claim — admission alone never authorizes side effects.
            conn.execute(
                "UPDATE bot_chain_deliveries SET state = 'admitted', "
                "outcome = NULL, detail = NULL, owner_pid = NULL, "
                "owner_host = NULL, owner_token = NULL, "
                "lease_expires_at = NULL, updated_at = ? "
                "WHERE session_id = ? AND platform_message_id = ?",
                (now, session_id, platform_message_id),
            )
            return "admitted"

        return self._bot_chain_write(_do)

    def mark_bot_chain_delivery_running(
        self,
        session_id: str,
        platform_message_id: str,
        *,
        lease_seconds: Optional[float] = None,
    ) -> Optional[str]:
        """Atomically claim execution for an admitted receipt.

        Returns the new claim's ``owner_token`` only for the single caller
        that transitioned the row from ``admitted`` to ``running``; every
        concurrent or late contender gets ``None`` and MUST execute zero
        model turns. The claim records the owning process AND a bounded
        lease; the owner must renew via
        :meth:`renew_bot_chain_delivery_claim` while executing, and both
        :meth:`settle_bot_chain_delivery` and
        :meth:`release_bot_chain_delivery_claim` require the returned
        token. This write is the durable execution claim: it must land
        BEFORE any side effect, and a lost or failed claim means the chain
        does not run.
        """
        pid, host = self._claim_owner()
        owner_token = uuid.uuid4().hex
        lease = float(
            lease_seconds
            if lease_seconds is not None
            else self.BOT_CHAIN_CLAIM_LEASE_SECONDS
        )
        if not math.isfinite(lease) or lease <= 0:
            raise ValueError("bot-chain claim lease must be finite and positive")

        def _do(conn: sqlite3.Connection) -> bool:
            # Lease window opens only once the write transaction is held.
            now = time.time()
            self._ensure_bot_chain_deliveries_table(conn)
            cursor = conn.execute(
                "UPDATE bot_chain_deliveries SET state = 'running', "
                "owner_pid = ?, owner_host = ?, owner_token = ?, "
                "lease_expires_at = ?, updated_at = ? "
                "WHERE session_id = ? AND platform_message_id = ? AND "
                "state = 'admitted'",
                (pid, host, owner_token, now + lease, now,
                 session_id, platform_message_id),
            )
            return cursor.rowcount == 1

        return owner_token if self._bot_chain_write(_do) else None

    def renew_bot_chain_delivery_claim(
        self,
        session_id: str,
        platform_message_id: str,
        owner_token: str,
        *,
        lease_seconds: Optional[float] = None,
    ) -> bool:
        """Extend this claim's lease; the executing owner's heartbeat.

        Returns True only while THIS claim (matched by ``owner_token``)
        still owns the ``running`` row. ``False`` means the claim was lost
        — reclaimed after an expiry gap — and the caller must stop
        renewing; settlement will likewise refuse the stale token.
        """
        if not owner_token:
            return False
        lease = float(
            lease_seconds
            if lease_seconds is not None
            else self.BOT_CHAIN_CLAIM_LEASE_SECONDS
        )
        if not math.isfinite(lease) or lease <= 0:
            raise ValueError("bot-chain claim lease must be finite and positive")

        def _do(conn: sqlite3.Connection) -> bool:
            # Lease window opens only once the write transaction is held.
            now = time.time()
            cursor = conn.execute(
                "UPDATE bot_chain_deliveries SET lease_expires_at = ?, "
                "updated_at = ? "
                "WHERE session_id = ? AND platform_message_id = ? AND "
                "state = 'running' AND owner_token = ? AND lease_expires_at > ?",
                (now + lease, now, session_id, platform_message_id,
                 str(owner_token), now),
            )
            return cursor.rowcount == 1

        return bool(self._bot_chain_write(_do))

    def release_bot_chain_delivery_claim(
        self,
        session_id: str,
        platform_message_id: str,
        owner_token: str,
    ) -> bool:
        """Release THIS claim's ``running`` row back to ``admitted``.

        Used when execution finished but the settlement write failed: left
        in ``running``, the receipt would stand redeliveries down until the
        lease expired. Released back to ``admitted``, a redelivery resumes
        under the recorded chain identity and recovers every durably
        persisted step instead of re-executing blindly.

        The release is scoped by ``owner_token``, so it can never revoke a
        concurrent or newer execution's claim. Returns True only when this
        claim's own ``running`` row moved back to ``admitted``; a foreign
        claim, an ``admitted`` row, and a ``settled`` row all return False
        untouched.
        """
        if not owner_token:
            return False

        def _do(conn: sqlite3.Connection) -> bool:
            now = time.time()
            self._ensure_bot_chain_deliveries_table(conn)
            cursor = conn.execute(
                "UPDATE bot_chain_deliveries SET state = 'admitted', "
                "owner_pid = NULL, owner_host = NULL, owner_token = NULL, "
                "lease_expires_at = NULL, updated_at = ? "
                "WHERE session_id = ? AND platform_message_id = ? AND "
                "state = 'running' AND owner_token = ?",
                (now, session_id, platform_message_id, str(owner_token)),
            )
            return cursor.rowcount == 1

        return self._bot_chain_write(_do)

    def settle_bot_chain_delivery(
        self,
        session_id: str,
        platform_message_id: str,
        *,
        outcome: str,
        detail: str = "",
        owner_token: Optional[str] = None,
    ) -> bool:
        """Bind the terminal chain outcome to the admission receipt.

        Owner-scoped: only the current claim's ``owner_token`` may settle a
        ``running`` row. A stale runtime generation whose claim was
        reclaimed can no longer overwrite the newer owner's outcome — the
        reclaiming delivery settles the truthful terminal state after
        recovering the durable step frontier. Returns True when this call
        performed the settlement; False when the row was not this claim's
        to settle (already settled, reclaimed, or foreign).
        """
        if not owner_token:
            return False

        def _do(conn: sqlite3.Connection) -> bool:
            now = time.time()
            cursor = conn.execute(
                "UPDATE bot_chain_deliveries SET state = 'settled', "
                "outcome = ?, detail = ?, lease_expires_at = NULL, "
                "updated_at = ? "
                "WHERE session_id = ? AND platform_message_id = ? AND "
                "state = 'running' AND owner_token = ? AND lease_expires_at > ?",
                (outcome, detail, now, session_id,
                 platform_message_id, str(owner_token), now),
            )
            return cursor.rowcount == 1

        return bool(self._bot_chain_write(_do))

    def get_bot_chain_delivery(
        self, session_id: str, platform_message_id: str
    ) -> Optional[Dict[str, Any]]:
        """Return the admission row as a dict (or None). Diagnostic/tests.

        Self-migrating: a table created by a pre-lease build lacks the
        columns this SELECT reads, and treating that schema error as "no
        receipt" would let the caller fall through to a weaker legacy
        dedupe path and strand the real receipt. On a schema miss ("no
        such table" / "no such column") the migration runs first and the
        read is retried once. A plain miss (no row) costs no write, and
        any other OperationalError propagates — a locked or broken DB is
        NOT authoritative proof that no receipt exists.
        """

        def _read(conn: sqlite3.Connection):
            return conn.execute(
                "SELECT session_id, platform_message_id, chain_name, "
                "state, outcome, detail, owner_token, lease_expires_at, "
                "admitted_at, updated_at "
                "FROM bot_chain_deliveries "
                "WHERE session_id = ? AND platform_message_id = ?",
                (session_id, platform_message_id),
            ).fetchone()

        schema_miss = False
        with self._read_ctx() as conn:
            try:
                row = _read(conn)
            except sqlite3.OperationalError as exc:
                reason = str(exc).lower()
                if "no such table" not in reason and "no such column" not in reason:
                    raise
                schema_miss = True
                row = None
        if schema_miss:
            # A pre-lease schema (or no table at all). Migrate (idempotent)
            # and read once more before concluding "no receipt".
            self._execute_write(self._ensure_bot_chain_deliveries_table)
            with self._read_ctx() as conn:
                row = _read(conn)
        if row is None:
            return None
        return {
            "session_id": row[0],
            "platform_message_id": row[1],
            "chain_name": row[2],
            "state": row[3],
            "outcome": row[4],
            "detail": row[5],
            "owner_token": row[6],
            "lease_expires_at": row[7],
            "admitted_at": row[8],
            "updated_at": row[9],
        }

    def stamp_bot_chain_receipt(
        self, message_row_ids: Sequence[int], chain_name: str
    ) -> int:
        """Merge a bot-chain receipt into each listed row's display_metadata.

        Used when an isolated chain session is promoted (renamed) into the
        canonical Bot Chat: the rename retitles the session, so the exact
        chain identity has to survive on the message rows themselves.
        Returns the number of rows stamped.
        """
        ids = [int(row_id) for row_id in message_row_ids if row_id is not None]
        chain_name = str(chain_name or "").strip()
        if not ids or not chain_name:
            return 0
        receipt_json = json.dumps({"chain": chain_name})

        def _do(conn):
            cursor = conn.executemany(
                "UPDATE messages SET display_metadata = json_set("
                "COALESCE(NULLIF(display_metadata, ''), '{}'), "
                "'$.bot_chain', json(?)) WHERE id = ?",
                [(receipt_json, row_id) for row_id in ids],
            )
            return cursor.rowcount

        return int(self._execute_write(_do))
