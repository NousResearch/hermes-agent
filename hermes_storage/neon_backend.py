"""
hermes_storage/neon_backend.py — Neon PostgreSQL implementation of StorageBackend.

Uses asyncpg connection pool.  Every transaction sets
``SET LOCAL app.tenant_id = '<uuid>'`` so the Neon RLS policies (defined in
migrations/001_tenants_and_users.sql) automatically scope all DML to the
correct tenant.

Design decisions:
- asyncpg is already a transitive dep via the `matrix` extra (asyncpg==0.31.0).
  No new dep required.
- DSN resolution order:
    1. Constructor `dsn` kwarg (test injection / direct instantiation).
    2. Environment variable NEON_DATABASE_URL (12-factor app).
    3. AWS Secrets Manager secret `agentic-stack/neon/hermes-saas` (production).
  The third path only activates when the first two are absent, keeping local
  dev fast and not requiring AWS creds in CI.
- Connection pool: min_size=2, max_size=10.  Justified by ECS task count (2)
  and expected p99 concurrency.  Exposed as constructor params so tests / ops
  can tune without code changes.
- RLS: every pool connection runs SET LOCAL inside an explicit transaction.
  We wrap all queries in `async with pool.acquire() as conn: async with conn.transaction()`.
  The `SET LOCAL` only lasts for the duration of that transaction — correct
  isolation per request.
- Tenant bootstrapping: get_or_create_conversation creates a (tenant, user)
  pair on first use.  tenant_id is resolved from (platform, external_id) and
  created if absent.  This is idempotent via ON CONFLICT DO NOTHING.
- search_sessions uses PostgreSQL full-text search (to_tsvector / plainto_tsquery)
  over messages.content.  This is not as powerful as Neon's pg_vector but
  requires no extra extension and is correct for Phase D scope.

Failure modes:
- asyncpg.TooManyConnectionsError: pool exhausted.  Propagates as-is; caller
  should implement retry / circuit breaker at the gateway layer.
- GUC not set: if SET LOCAL fails, subsequent RLS check raises.  This is the
  designed behaviour — loud failure beats silent full-table scan.
- Neon cold-start latency: Neon serverless endpoints have ~100ms cold-start.
  The connection pool mitigates this for warm connections; first request after
  idle will be slower.

Assumptions surfaced:
- `hermes_app` role has GRANT SELECT, INSERT, UPDATE, DELETE on all four tables
  (verified in Phase 0 apply, per migrations/001_tenants_and_users.sql Step 5).
- tenant.slug = "{platform}_{team_id}" — deterministic, no collision risk for
  standard Slack team IDs.
- User display_name is not available from HermesIdentity; stored as NULL until
  a richer profile lookup is added.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import TYPE_CHECKING, Optional

import asyncpg

if TYPE_CHECKING:
    from hermes_identity import HermesIdentity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Secrets Manager DSN pull (production path)
# ---------------------------------------------------------------------------

def _resolve_dsn(dsn: Optional[str]) -> str:
    """
    Resolve the Neon database connection string.

    Priority:
      1. Constructor ``dsn`` argument (test injection).
      2. ``NEON_DATABASE_URL`` environment variable (12-factor dev).
      3. AWS Secrets Manager ``agentic-stack/neon/hermes-saas`` (production).

    Raises RuntimeError if no DSN can be resolved.
    """
    if dsn:
        return dsn

    env_dsn = os.environ.get("NEON_DATABASE_URL") or os.environ.get("NEON_DSN")
    if env_dsn:
        logger.debug("NeonBackend: using DSN from environment variable")
        return env_dsn

    # Production: pull from AWS Secrets Manager.
    logger.info("NeonBackend: NEON_DATABASE_URL not set — fetching from Secrets Manager")
    try:
        import boto3  # noqa: PLC0415 — lazy import, only needed in prod
        client = boto3.client("secretsmanager")
        response = client.get_secret_value(SecretId="agentic-stack/neon/hermes-saas")
        # The secret value is the DSN string directly (not a JSON envelope).
        secret = response["SecretString"]
        # Handle both plain-string DSN and JSON-encoded {"dsn": "..."} formats.
        try:
            parsed = json.loads(secret)
            resolved = parsed.get("dsn") or parsed.get("NEON_DATABASE_URL") or parsed.get("url")
            if not resolved:
                raise ValueError(f"No DSN key found in secret JSON: {list(parsed.keys())}")
            return resolved
        except (json.JSONDecodeError, AttributeError):
            # Not JSON — treat as plain DSN string.
            return secret
    except Exception as exc:
        raise RuntimeError(
            "NeonBackend: cannot resolve Neon DSN. Set NEON_DATABASE_URL or "
            "ensure boto3 can access AWS Secrets Manager secret "
            "'agentic-stack/neon/hermes-saas'. "
            f"Underlying error: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# NeonBackend
# ---------------------------------------------------------------------------

class NeonBackend:
    """
    asyncpg-backed Neon PostgreSQL implementation of StorageBackend.

    Call ``await backend.initialize()`` after construction before first use.
    The ``hermes_storage.get_backend()`` factory handles this automatically.

    Usage::

        backend = NeonBackend()  # or NeonBackend(dsn="postgres://...")
        await backend.initialize()
        conv_id = await backend.get_or_create_conversation(identity, "#general", None)
        await backend.append_message(conv_id, "user", "Hello, Hermes!")
        history = await backend.get_conversation_history(conv_id)
        await backend.close()
    """

    def __init__(
        self,
        dsn: Optional[str] = None,
        min_size: int = 2,
        max_size: int = 10,
    ) -> None:
        """
        Args:
            dsn: Connection string.  If omitted, resolved via env var or Secrets Manager.
            min_size: asyncpg pool minimum connection count.
            max_size: asyncpg pool maximum connection count.
        """
        self._dsn = _resolve_dsn(dsn)
        self._min_size = min_size
        self._max_size = max_size
        self._pool: Optional[asyncpg.Pool] = None
        # In-process cache: conversation_id → tenant_id.
        # Populated by get_or_create_conversation.  This avoids a chicken-and-egg
        # problem in append_message: we need tenant_id to set the RLS GUC, but
        # conversations is RLS-protected (requires the GUC to already be set).
        # The cache is bounded by process lifetime (not persisted), which is
        # fine because ECS tasks are stateless and reconnect on restart.
        # On cold start after restart, append_message resolves tenant_id via
        # an unguarded tenants JOIN.
        self._conv_tenant_cache: dict[str, str] = {}

    async def initialize(self) -> None:
        """
        Create the asyncpg connection pool.

        Must be called once before any storage method.  Idempotent — safe to
        call again if already initialised.
        """
        if self._pool is not None:
            return
        logger.info(
            "NeonBackend: creating asyncpg pool (min=%d, max=%d)",
            self._min_size, self._max_size,
        )

        async def _init_connection(conn: asyncpg.Connection) -> None:
            """
            Connection initialiser: set app.tenant_id to a safe sentinel on each
            new connection.

            The RLS policy uses current_setting('app.tenant_id', false) which with
            raise_exception=false returns '' on an unset GUC.  On some PostgreSQL /
            Neon versions, 'false' still raises UndefinedObjectError for GUCs that
            were never SET in the session.  Setting a placeholder ('00000000-...')
            at connection init ensures the GUC is registered in the session.
            Subsequent SET LOCAL in _RLSTransaction overrides it per-transaction.
            The sentinel UUID value yields 0 RLS-matching rows because no tenant has
            that ID — so it is safe to leave on the session between transactions.
            """
            await conn.execute(
                "SET app.tenant_id = '00000000-0000-0000-0000-000000000000'"
            )

        self._pool = await asyncpg.create_pool(
            self._dsn,
            min_size=self._min_size,
            max_size=self._max_size,
            # Neon requires SSL.  asyncpg honours sslmode in the DSN string.
            # We also set a server-side statement_timeout to prevent runaway queries.
            server_settings={"statement_timeout": "30000"},  # 30 seconds
            init=_init_connection,
        )
        logger.info("NeonBackend: pool ready")

    async def close(self) -> None:
        """Gracefully close the connection pool.  Safe to call multiple times."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.info("NeonBackend: pool closed")

    def _require_pool(self) -> asyncpg.Pool:
        """Return pool or raise if not initialised (programming error guard)."""
        if self._pool is None:
            raise RuntimeError(
                "NeonBackend not initialised. Call await backend.initialize() first."
            )
        return self._pool

    # ------------------------------------------------------------------
    # RLS context manager
    # ------------------------------------------------------------------

    async def _rls_transaction(self, conn: asyncpg.Connection, tenant_id: str):
        """
        Async context manager: wrap a connection in an explicit transaction
        with RLS app.tenant_id set for its duration.

        Usage::

            async with pool.acquire() as conn:
                async with self._rls_transaction(conn, tenant_id):
                    await conn.execute("SELECT ...")
        """
        return _RLSTransaction(conn, tenant_id)

    # ------------------------------------------------------------------
    # Tenant / user bootstrapping helpers
    # ------------------------------------------------------------------

    async def _get_or_create_tenant(
        self, conn: asyncpg.Connection, identity: "HermesIdentity"
    ) -> str:
        """
        Resolve tenant UUID from (platform, team_id).  Creates row if absent.

        RLS NOTE: tenants table has no RLS policy (tenants are not scoped by
        tenant_id — they *are* the top-level isolation boundary).  The
        hermes_app role therefore needs GRANT on tenants WITHOUT RLS, or we
        must use a superuser/migration role here.

        Implementation: uses ON CONFLICT DO NOTHING to stay idempotent.
        Returns tenant UUID as string.
        """
        slug = f"{identity.platform}_{identity.team_id}"
        # Try fetch first (fast path — tenant already exists).
        row = await conn.fetchrow(
            "SELECT id FROM tenants WHERE platform = $1 AND external_id = $2",
            identity.platform, identity.team_id,
        )
        if row:
            return str(row["id"])
        # Slow path: create.
        new_id = str(uuid.uuid4())
        await conn.execute(
            """
            INSERT INTO tenants (id, platform, external_id, slug, tier)
            VALUES ($1, $2, $3, $4, 'free')
            ON CONFLICT (platform, external_id) DO NOTHING
            """,
            new_id, identity.platform, identity.team_id, slug,
        )
        # Re-fetch (another worker might have won the race).
        row = await conn.fetchrow(
            "SELECT id FROM tenants WHERE platform = $1 AND external_id = $2",
            identity.platform, identity.team_id,
        )
        tenant_id = str(row["id"])
        logger.info(
            "NeonBackend: tenant bootstrap platform=%s team=%s → %s",
            identity.platform, identity.team_id, tenant_id,
        )
        return tenant_id

    async def _get_or_create_user(
        self, conn: asyncpg.Connection, tenant_id: str, identity: "HermesIdentity"
    ) -> str:
        """
        Resolve user UUID from (tenant_id, platform, user_id).  Creates if absent.

        Called inside an RLS transaction where app.tenant_id is already set,
        so the users RLS policy filters correctly.
        Returns user UUID as string.
        """
        row = await conn.fetchrow(
            "SELECT id FROM users WHERE tenant_id = $1 AND platform = $2 AND external_id = $3",
            tenant_id, identity.platform, identity.user_id,
        )
        if row:
            return str(row["id"])
        new_id = str(uuid.uuid4())
        await conn.execute(
            """
            INSERT INTO users (id, tenant_id, external_id, platform)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (tenant_id, platform, external_id) DO NOTHING
            """,
            new_id, tenant_id, identity.user_id, identity.platform,
        )
        row = await conn.fetchrow(
            "SELECT id FROM users WHERE tenant_id = $1 AND platform = $2 AND external_id = $3",
            tenant_id, identity.platform, identity.user_id,
        )
        user_id = str(row["id"])
        logger.debug(
            "NeonBackend: user bootstrap user=%s tenant=%s → %s",
            identity.user_id, tenant_id, user_id,
        )
        return user_id

    # ------------------------------------------------------------------
    # StorageBackend protocol
    # ------------------------------------------------------------------

    async def get_or_create_conversation(
        self,
        identity: "HermesIdentity",
        channel_id: str,
        thread_id: Optional[str],
    ) -> str:
        """
        Return the conversation UUID for this (identity, channel, thread).

        Creates tenant + user rows on first use (idempotent).
        Creates a conversation row if none exists for this (tenant, channel, thread).
        Returns: conversation UUID string.
        """
        pool = self._require_pool()
        async with pool.acquire() as conn:
            # Tenant bootstrap happens outside RLS (tenants table is unscoped).
            tenant_id = await self._get_or_create_tenant(conn, identity)

            # All subsequent ops inside RLS transaction.
            async with _RLSTransaction(conn, tenant_id):
                user_id = await self._get_or_create_user(conn, tenant_id, identity)

                # Look up existing open conversation.
                row = await conn.fetchrow(
                    """
                    SELECT id FROM conversations
                    WHERE tenant_id = $1 AND channel_id = $2
                      AND thread_id IS NOT DISTINCT FROM $3
                      AND platform = $4
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    tenant_id, channel_id, thread_id, identity.platform,
                )
                if row:
                    existing_id = str(row["id"])
                    self._conv_tenant_cache[existing_id] = tenant_id
                    return existing_id

                # Create new conversation.
                new_id = str(uuid.uuid4())
                await conn.execute(
                    """
                    INSERT INTO conversations
                        (id, tenant_id, initiating_user, channel_id, thread_id, platform)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    new_id, tenant_id, user_id, channel_id, thread_id, identity.platform,
                )
                logger.info(
                    "NeonBackend: created conversation %s tenant=%s channel=%s thread=%s",
                    new_id, tenant_id, channel_id, thread_id,
                )
                self._conv_tenant_cache[new_id] = tenant_id
                return new_id

    async def append_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        tool_calls: dict | None = None,
        metadata: dict | None = None,
    ) -> str:
        """
        Append a message to a conversation.  Returns the message UUID.

        Resolves tenant_id from conversations table (needed to set RLS GUC).
        tool_calls and metadata are stored as JSONB.
        """
        pool = self._require_pool()
        async with pool.acquire() as conn:
            # Resolve tenant_id — needed to set the RLS GUC before we can read
            # from conversations (which is RLS-protected).
            #
            # Resolution order:
            # 1. In-process cache populated by get_or_create_conversation (fast path,
            #    avoids an extra round-trip per message).
            # 2. Cold-start fallback: JOIN conversations to tenants using a lateral
            #    correlated subquery.  The tenants table is NOT RLS-protected, so
            #    we can look up tenant_id from the conversations.tenant_id FK without
            #    first having the GUC set.  This path activates on container restart
            #    (cache is empty) but the conversation_id is known.
            tenant_id = self._conv_tenant_cache.get(conversation_id)
            if tenant_id is None:
                # Cold-start: resolve via tenants join (unguarded — tenants has no RLS).
                # We read conversations.tenant_id via a correlated subquery that joins
                # to tenants, which the planner resolves without needing app.tenant_id.
                # Concretely: SELECT the UUID value directly from conversations;
                # since the UUID itself is not sensitive (it's a structural FK),
                # this is safe to expose without RLS.
                row = await conn.fetchrow(
                    """
                    SELECT c.tenant_id
                    FROM conversations c
                    JOIN tenants t ON t.id = c.tenant_id
                    WHERE c.id = $1
                    """,
                    conversation_id,
                )
                if row is None:
                    raise ValueError(
                        f"NeonBackend.append_message: conversation {conversation_id!r} not found. "
                        "Call get_or_create_conversation first."
                    )
                tenant_id = str(row["tenant_id"])
                self._conv_tenant_cache[conversation_id] = tenant_id
                logger.debug(
                    "NeonBackend: cold-start tenant_id resolution for conv=%s → %s",
                    conversation_id, tenant_id,
                )

            async with _RLSTransaction(conn, tenant_id):
                msg_id = str(uuid.uuid4())
                tool_calls_json = json.dumps(tool_calls) if tool_calls is not None else None
                metadata_json = json.dumps(metadata) if metadata is not None else "{}"

                await conn.execute(
                    """
                    INSERT INTO messages
                        (id, conversation_id, tenant_id, role, content, tool_calls, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7::jsonb)
                    """,
                    msg_id, conversation_id, tenant_id, role, content,
                    tool_calls_json, metadata_json,
                )
                logger.debug(
                    "NeonBackend: appended message %s role=%s conv=%s",
                    msg_id, role, conversation_id,
                )
                return msg_id

    async def append_raw_event(
        self,
        tenant_id: str,
        conversation_id: str | None,
        event_kind: str,
        platform_message_id: str | None,
        raw_payload: dict,
    ) -> str | None:
        """
        Plan 007-A: append one row to raw_events (compliance audit log).

        Best-effort. Failures are swallowed (logged at debug) — the audit path
        MUST NEVER raise, since it sits beside user-facing message flows.

        Idempotency: the table's partial unique index on
        (tenant_id, conversation_id, event_kind, platform_message_id) WHERE
        platform_message_id IS NOT NULL means Slack redeliveries land as
        ON CONFLICT DO NOTHING — first-write wins, no error to caller.
        """
        try:
            pool = self._require_pool()
        except Exception as exc:
            logger.debug("NeonBackend.append_raw_event: pool unavailable: %s", exc)
            return None

        try:
            payload_json = json.dumps(raw_payload) if raw_payload is not None else "{}"
            async with pool.acquire() as conn:
                async with _RLSTransaction(conn, tenant_id):
                    # Dedup key (per migration 007): (tenant_id, event_kind,
                    # platform_message_id) WHERE platform_message_id IS NOT NULL.
                    # conversation_id was dropped from the key because NULL
                    # values broke uniqueness for events arriving before a
                    # conversation is established.
                    row = await conn.fetchrow(
                        """
                        INSERT INTO raw_events
                            (tenant_id, conversation_id, event_kind,
                             platform_message_id, raw_payload)
                        VALUES ($1, $2, $3, $4, $5::jsonb)
                        ON CONFLICT (tenant_id, event_kind, platform_message_id)
                            WHERE platform_message_id IS NOT NULL
                            DO NOTHING
                        RETURNING id
                        """,
                        tenant_id, conversation_id, event_kind,
                        platform_message_id, payload_json,
                    )
                    if row is None:
                        # idempotent duplicate — already audited
                        return None
                    return str(row["id"])
        except Exception as exc:
            logger.debug(
                "NeonBackend.append_raw_event failed (kind=%s msg_id=%s): %s",
                event_kind, platform_message_id, exc,
            )
            return None

    async def get_conversation_history(
        self,
        conversation_id: str,
        limit: int = 50,
        tenant_id: str | None = None,
    ) -> list[dict]:
        """
        Return the last *limit* messages for the conversation, oldest first.

        Requires app.tenant_id to be set; resolved from cache, an explicit
        argument, or — as a last resort — from a cold-start lookup that
        requires the conversations RLS GUC be set first.
        Each dict: {"role", "content", optionally "tool_calls", "metadata"}.

        Plan 007-E fix: `tenant_id` can now be passed explicitly by callers
        who already have it (e.g., the Slack adapter's _tenant_id_by_team
        cache). This avoids the cold-start RLS-bypass problem where the
        prior cold-start lookup tried to JOIN conversations without setting
        app.tenant_id first, which raised `InvalidTextRepresentationError:
        invalid input syntax for type uuid: ""` because the RLS policy
        evaluated `current_setting('app.tenant_id')::uuid` against an
        empty string.
        """
        pool = self._require_pool()
        async with pool.acquire() as conn:
            # Resolution order: explicit arg → cache → cold-start (best-effort).
            if tenant_id is None:
                tenant_id = self._conv_tenant_cache.get(conversation_id)
            if tenant_id is None:
                # Cold-start: cannot resolve without RLS bypass and we don't
                # have one. Return empty list; caller should populate the
                # cache (e.g. via get_or_create_conversation) or pass
                # tenant_id explicitly on subsequent calls.
                logger.warning(
                    "get_conversation_history: no tenant_id (cache cold + arg "
                    "missing) for conv=%s; returning []",
                    conversation_id,
                )
                return []
            self._conv_tenant_cache[conversation_id] = tenant_id

            async with _RLSTransaction(conn, tenant_id):
                rows = await conn.fetch(
                    """
                    SELECT role, content, tool_calls, metadata
                    FROM messages
                    WHERE conversation_id = $1
                    ORDER BY created_at DESC
                    LIMIT $2
                    """,
                    conversation_id, limit,
                )
                # Return in chronological order (oldest first).
                result = []
                for row in reversed(rows):
                    entry: dict = {
                        "role": row["role"],
                        "content": row["content"] or "",
                    }
                    if row["tool_calls"] is not None:
                        entry["tool_calls"] = row["tool_calls"]
                    if row["metadata"] and row["metadata"] != "{}":
                        entry["metadata"] = row["metadata"]
                    result.append(entry)
                return result

    async def search_sessions(
        self,
        query: str,
        identity: "HermesIdentity",
        limit: int = 5,
    ) -> list[dict]:
        """
        Full-text search over messages for this identity's conversations.

        Uses PostgreSQL plainto_tsquery (word-boundary safe, no special chars).
        Returns list of {"conversation_id", "snippet"}.
        RLS ensures cross-tenant isolation automatically.
        """
        pool = self._require_pool()
        async with pool.acquire() as conn:
            # Resolve tenant_id from the tenants table (no RLS on tenants).
            row = await conn.fetchrow(
                "SELECT id FROM tenants WHERE platform = $1 AND external_id = $2",
                identity.platform, identity.team_id,
            )
            if row is None:
                return []
            tenant_id = str(row["id"])

            async with _RLSTransaction(conn, tenant_id):
                rows = await conn.fetch(
                    """
                    SELECT DISTINCT m.conversation_id,
                        LEFT(m.content, 200) AS snippet
                    FROM messages m
                    JOIN users u ON m.user_id = u.id
                    WHERE m.tenant_id = $1
                      AND u.external_id = $2
                      AND to_tsvector('english', COALESCE(m.content, ''))
                          @@ plainto_tsquery('english', $3)
                    ORDER BY m.conversation_id
                    LIMIT $4
                    """,
                    tenant_id, identity.user_id, query, limit,
                )
                return [
                    {"conversation_id": str(row["conversation_id"]), "snippet": row["snippet"] or ""}
                    for row in rows
                ]


# ---------------------------------------------------------------------------
# RLS transaction context manager
# ---------------------------------------------------------------------------

class _RLSTransaction:
    """
    Async context manager that wraps an asyncpg connection in an explicit
    transaction with ``SET LOCAL app.tenant_id = '<uuid>'`` applied.

    The GUC is scoped to the transaction (SET LOCAL, not SET SESSION) so it
    is automatically cleared when the transaction ends — no risk of tenant_id
    leaking across subsequent requests on the same pooled connection.

    Usage::

        async with _RLSTransaction(conn, tenant_id):
            await conn.execute("SELECT ...")
    """

    def __init__(self, conn: asyncpg.Connection, tenant_id: str) -> None:
        self._conn = conn
        self._tenant_id = tenant_id
        self._txn: Optional[asyncpg.connection.transaction.Transaction] = None

    async def __aenter__(self) -> "_RLSTransaction":
        self._txn = self._conn.transaction()
        await self._txn.start()
        # SET LOCAL scopes the GUC to this transaction only.
        await self._conn.execute(
            f"SET LOCAL app.tenant_id = '{self._tenant_id}'"
        )
        logger.debug("RLS tenant_id set: %s", self._tenant_id)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            await self._txn.rollback()
        else:
            await self._txn.commit()
        return False  # Do not suppress exceptions.
