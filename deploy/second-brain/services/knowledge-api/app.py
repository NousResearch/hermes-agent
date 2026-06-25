import os
import asyncio
import json
import time
from pathlib import Path
from typing import Any

import asyncpg
import httpx
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException


app = FastAPI(title="Company Knowledge API", version="0.1.0")

DATABASE_URL = os.environ["DATABASE_URL"]
REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")
INGEST_QUEUE_NAME = os.environ.get("INGEST_QUEUE_NAME", "second_brain:ingest_jobs")
SOURCE_SCAN_QUEUE_NAME = os.environ.get("SOURCE_SCAN_QUEUE_NAME", "second_brain:source_scan_jobs")
QUERY_WORKSPACE_CONCURRENCY = int(os.environ.get("QUERY_WORKSPACE_CONCURRENCY", "4"))
LIGHTRAG_ROOT = Path(os.environ.get("LIGHTRAG_ROOT", "/data/lightrag"))
MIN_SOURCE_INTERVAL_MINUTES = int(os.environ.get("MIN_SOURCE_INTERVAL_MINUTES", "15"))
MAX_SOURCE_INTERVAL_MINUTES = int(os.environ.get("MAX_SOURCE_INTERVAL_MINUTES", str(60 * 24 * 30)))
PUBLIC_WORKSPACE = "company_public"
C_LEVEL_WORKSPACE = "department_c_level"
ENABLED_WORKSPACES = [PUBLIC_WORKSPACE, C_LEVEL_WORKSPACE]
C_LEVEL_TARGETS = {"c_level", "c-level", "clevel", "c level", "department_c_level", "admin", "executive", "board"}
C_LEVEL_CLASSIFICATIONS = {"confidential", "restricted", "c_level", "c-level"}
SOURCE_TYPES = {"notion", "drive_public"}
SOURCE_SECRET_KEYS = {"notion_api_key", "api_key", "token", "access_token", "secret"}
LIGHTRAG_API_KEY = os.environ.get("LIGHTRAG_API_KEY", "")
LIGHTRAG_URLS = {
    PUBLIC_WORKSPACE: os.environ.get("LIGHTRAG_COMPANY_PUBLIC_URL"),
    C_LEVEL_WORKSPACE: os.environ.get("LIGHTRAG_DEPARTMENT_C_LEVEL_URL"),
}


@app.on_event("startup")
async def startup() -> None:
    app.state.pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
    app.state.redis = redis.from_url(REDIS_URL, decode_responses=True)
    await ensure_runtime_schema()
    for workspace in ENABLED_WORKSPACES:
        base = LIGHTRAG_ROOT / "workspaces" / workspace
        for child in ("docs", "kv_store", "vector_store", "graph_store"):
            (base / child).mkdir(parents=True, exist_ok=True)


@app.on_event("shutdown")
async def shutdown() -> None:
    await app.state.redis.aclose()
    await app.state.pool.close()


async def ensure_runtime_schema() -> None:
    async with app.state.pool.acquire() as conn:
        await conn.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS queued_at timestamptz")
        await conn.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS indexed_at timestamptz")
        await conn.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS ingest_error text")
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS document_ingest_payloads (
              document_id uuid PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
              workspace_slug text NOT NULL,
              title text NOT NULL,
              source_text text NOT NULL,
              attempts integer NOT NULL DEFAULT 0,
              created_at timestamptz NOT NULL DEFAULT now(),
              updated_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        await ensure_source_schema(conn)
        await ensure_analytics_schema(conn)


async def ensure_source_schema(conn: Any) -> None:
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS document_sources (
          id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
          name text NOT NULL,
          source_type text NOT NULL CHECK (source_type IN ('notion', 'drive_public')),
          target text NOT NULL DEFAULT 'public',
          workspace_slug text NOT NULL,
          classification text NOT NULL CHECK (classification IN ('public', 'internal', 'confidential', 'restricted')),
          config jsonb NOT NULL DEFAULT '{}'::jsonb,
          interval_minutes integer NOT NULL DEFAULT 1440,
          enabled boolean NOT NULL DEFAULT true,
          next_scan_at timestamptz,
          last_scan_at timestamptz,
          last_status text,
          last_error text,
          created_at timestamptz NOT NULL DEFAULT now(),
          updated_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS source_scan_runs (
          id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
          source_id uuid NOT NULL REFERENCES document_sources(id) ON DELETE CASCADE,
          trigger text NOT NULL CHECK (trigger IN ('manual', 'scheduled', 'startup')),
          status text NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'running', 'complete', 'failed')),
          queued_at timestamptz NOT NULL DEFAULT now(),
          started_at timestamptz,
          finished_at timestamptz,
          items_found integer NOT NULL DEFAULT 0,
          documents_queued integer NOT NULL DEFAULT 0,
          error text
        )
        """
    )
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS source_items (
          source_id uuid NOT NULL REFERENCES document_sources(id) ON DELETE CASCADE,
          external_id text NOT NULL,
          checksum text NOT NULL,
          document_id uuid REFERENCES documents(id) ON DELETE SET NULL,
          title text NOT NULL,
          source_uri text,
          last_seen_at timestamptz NOT NULL DEFAULT now(),
          updated_at timestamptz NOT NULL DEFAULT now(),
          PRIMARY KEY (source_id, external_id)
        )
        """
    )


async def ensure_analytics_schema(conn: Any) -> None:
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS query_events (
          id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
          actor_email text,
          actor_role text NOT NULL DEFAULT 'member',
          actor_groups text[] NOT NULL DEFAULT '{}'::text[],
          query_text text NOT NULL,
          mode text NOT NULL DEFAULT 'mix',
          allowed_workspaces text[] NOT NULL DEFAULT '{}'::text[],
          status text NOT NULL CHECK (status IN ('ok', 'error')),
          latency_ms integer NOT NULL DEFAULT 0,
          error text,
          created_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS query_workspace_events (
          id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
          query_event_id uuid NOT NULL REFERENCES query_events(id) ON DELETE CASCADE,
          workspace_slug text NOT NULL,
          latency_ms integer NOT NULL DEFAULT 0,
          status text NOT NULL CHECK (status IN ('ok', 'error')),
          reference_count integer NOT NULL DEFAULT 0,
          error text,
          created_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS query_document_hits (
          id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
          query_event_id uuid NOT NULL REFERENCES query_events(id) ON DELETE CASCADE,
          workspace_slug text NOT NULL,
          reference_id text,
          title text,
          source_uri text,
          rank integer NOT NULL DEFAULT 0,
          document_id uuid REFERENCES documents(id) ON DELETE SET NULL,
          created_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_query_events_created_at ON query_events (created_at DESC)")
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_query_events_actor ON query_events (actor_email, created_at DESC)")
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_query_workspace_events_workspace ON query_workspace_events (workspace_slug, created_at DESC)"
    )
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_query_document_hits_lookup ON query_document_hits (workspace_slug, title, created_at DESC)"
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/workspaces")
async def workspaces() -> dict[str, Any]:
    rows = await app.state.pool.fetch(
        """
        SELECT slug, name, visibility_boundary
        FROM rag_workspaces
        WHERE slug = ANY($1::text[])
        ORDER BY array_position($1::text[], slug)
        """,
        ENABLED_WORKSPACES,
    )
    return {
        "workspaces": [dict(row) for row in rows],
        "lightrag_root": str(LIGHTRAG_ROOT),
    }


def normalize_value(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", " ").replace("-", " ")


def route_workspace(payload: dict[str, Any]) -> str:
    explicit_workspace = str(payload.get("workspace") or "").strip()
    if explicit_workspace == C_LEVEL_WORKSPACE:
        return C_LEVEL_WORKSPACE
    if explicit_workspace == PUBLIC_WORKSPACE:
        return PUBLIC_WORKSPACE

    target_values = {
        normalize_value(payload.get("target")),
        normalize_value(payload.get("workspace")),
        normalize_value(payload.get("visibility")),
        normalize_value(payload.get("department")),
    }
    classification = normalize_value(payload.get("classification"))
    if target_values & {normalize_value(value) for value in C_LEVEL_TARGETS}:
        return C_LEVEL_WORKSPACE
    if classification in {normalize_value(value) for value in C_LEVEL_CLASSIFICATIONS}:
        return C_LEVEL_WORKSPACE
    return PUBLIC_WORKSPACE


def allowed_workspaces(groups: list[str]) -> list[str]:
    normalized = {str(group).strip().lower() for group in groups}
    allowed = [PUBLIC_WORKSPACE]
    if "role_admin" in normalized:
        allowed.append(C_LEVEL_WORKSPACE)
    return allowed


def coerce_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "enabled"}


def normalize_source_type(value: Any) -> str:
    source_type = str(value or "").strip().lower().replace("-", "_")
    if source_type in {"drive", "google_drive", "gdrive"}:
        source_type = "drive_public"
    if source_type not in SOURCE_TYPES:
        raise ValueError("source_type must be notion or drive_public")
    return source_type


def clamp_source_interval(value: Any) -> int:
    try:
        interval = int(value)
    except (TypeError, ValueError):
        interval = 1440
    return max(MIN_SOURCE_INTERVAL_MINUTES, min(interval, MAX_SOURCE_INTERVAL_MINUTES))


def source_config_from_value(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        loaded = json.loads(value)
        return loaded if isinstance(loaded, dict) else {}
    return {}


def redact_source_config(config: dict[str, Any]) -> dict[str, Any]:
    redacted: dict[str, Any] = {}
    for key, value in config.items():
        lower = str(key).lower()
        if lower in SOURCE_SECRET_KEYS or lower.endswith(("_key", "_token", "_secret")):
            redacted[key] = "set" if value else ""
        else:
            redacted[key] = value
    return redacted


def normalize_source_payload(payload: dict[str, Any], *, existing_config: dict[str, Any] | None = None) -> dict[str, Any]:
    name = str(payload.get("name") or "").strip()
    if not name:
        raise ValueError("name is required")
    source_type = normalize_source_type(payload.get("source_type") or payload.get("type"))
    config = dict(existing_config or {})
    config.update(source_config_from_value(payload.get("config")))
    for key in (
        "notion_api_key",
        "notion_page_id",
        "notion_page_url",
        "notion_data_source_id",
        "notion_search_query",
        "drive_url",
        "drive_urls",
    ):
        if payload.get(key) not in {None, ""}:
            config[key] = payload[key]

    if source_type == "notion" and not str(config.get("notion_api_key") or "").strip():
        raise ValueError("notion_api_key is required for notion sources")
    if source_type == "drive_public" and not (config.get("drive_url") or config.get("drive_urls")):
        raise ValueError("drive_url or drive_urls is required for drive_public sources")

    classification = str(payload.get("classification") or "internal").strip().lower()
    if classification not in {"public", "internal", "confidential", "restricted"}:
        raise ValueError("classification must be public, internal, confidential, or restricted")
    target = str(payload.get("target") or "public").strip().lower().replace("-", "_")
    workspace = route_workspace({"target": target, "classification": classification})
    if workspace == C_LEVEL_WORKSPACE:
        target = "c_level"
    else:
        target = "public"

    return {
        "name": name,
        "source_type": source_type,
        "target": target,
        "workspace": workspace,
        "classification": classification,
        "config": config,
        "interval_minutes": clamp_source_interval(payload.get("interval_minutes")),
        "enabled": coerce_bool(payload.get("enabled"), True),
    }


def serialize_source_row(row: Any) -> dict[str, Any]:
    data = dict(row)
    data["id"] = str(data["id"])
    data["config"] = redact_source_config(source_config_from_value(data.get("config")))
    for key in ("next_scan_at", "last_scan_at", "created_at", "updated_at"):
        value = data.get(key)
        if hasattr(value, "isoformat"):
            data[key] = value.isoformat()
    return data


def serialize_scan_run(row: Any) -> dict[str, Any]:
    data = dict(row)
    data["id"] = str(data["id"])
    data["source_id"] = str(data["source_id"])
    for key in ("queued_at", "started_at", "finished_at"):
        value = data.get(key)
        if hasattr(value, "isoformat"):
            data[key] = value.isoformat()
    return data


def analytics_int(value: Any, default: int = 0) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return default


def analytics_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return default


def analytics_iso(value: Any) -> Any:
    return value.isoformat() if hasattr(value, "isoformat") else value


def clamp_analytics_days(value: Any) -> int:
    try:
        days = int(value)
    except (TypeError, ValueError):
        days = 30
    return max(1, min(days, 365))


def clamp_analytics_limit(value: Any) -> int:
    try:
        limit = int(value)
    except (TypeError, ValueError):
        limit = 20
    return max(1, min(limit, 100))


def compact_query_text(value: Any, limit: int = 2000) -> str:
    return " ".join(str(value or "").split())[:limit]


def reference_field(ref: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = ref.get(key)
        if value not in {None, ""}:
            return str(value)
    return None


def coerce_document_id(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        import uuid

        return str(uuid.UUID(text))
    except ValueError:
        return None


def extract_document_hits(answers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    ranks_by_workspace: dict[str, int] = {}
    for answer in answers:
        workspace = str(answer.get("workspace") or "unknown")
        result = answer.get("result") or {}
        references = result.get("references") if isinstance(result, dict) else None
        if isinstance(references, dict):
            references = list(references.values())
        if not isinstance(references, list):
            continue
        for ref in references:
            ref_data = ref if isinstance(ref, dict) else {"reference_id": str(ref)}
            reference_id = reference_field(ref_data, "reference_id", "id", "chunk_id", "doc_id")
            source_uri = reference_field(ref_data, "source_uri", "file_path", "url", "source")
            title = reference_field(ref_data, "title", "file_path", "source", "source_uri", "url", "reference_id", "id")
            document_id = coerce_document_id(reference_field(ref_data, "document_id", "doc_id"))
            dedupe_key = (workspace, reference_id or source_uri or title or "")
            if not dedupe_key[1] or dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            ranks_by_workspace[workspace] = ranks_by_workspace.get(workspace, 0) + 1
            hits.append(
                {
                    "workspace_slug": workspace,
                    "rank": ranks_by_workspace[workspace],
                    "reference_id": reference_id,
                    "title": title,
                    "source_uri": source_uri or title,
                    "document_id": document_id,
                }
            )
    return hits


async def record_query_analytics(
    pool: Any,
    *,
    actor_email: str | None,
    actor_role: str,
    actor_groups: list[str],
    query_text: str,
    mode: str,
    allowed_workspaces: list[str],
    answers: list[dict[str, Any]],
    latency_ms: int,
    status: str,
    error: str | None,
) -> None:
    hits = extract_document_hits(answers)
    reference_counts: dict[str, int] = {}
    for hit in hits:
        workspace = hit["workspace_slug"]
        reference_counts[workspace] = reference_counts.get(workspace, 0) + 1

    async with pool.acquire() as conn:
        event_id = await conn.fetchval(
            """
            INSERT INTO query_events (
              actor_email, actor_role, actor_groups, query_text, mode,
              allowed_workspaces, status, latency_ms, error
            )
            VALUES ($1, $2, $3::text[], $4, $5, $6::text[], $7, $8, $9)
            RETURNING id
            """,
            actor_email,
            actor_role or "member",
            actor_groups,
            compact_query_text(query_text),
            mode or "mix",
            allowed_workspaces,
            status,
            max(0, int(latency_ms)),
            error,
        )
        for answer in answers:
            workspace = str(answer.get("workspace") or "unknown")
            workspace_error = answer.get("error")
            await conn.execute(
                """
                INSERT INTO query_workspace_events (
                  query_event_id, workspace_slug, latency_ms, status,
                  reference_count, error
                )
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                event_id,
                workspace,
                max(0, analytics_int(answer.get("latency_ms"), latency_ms)),
                "error" if workspace_error else "ok",
                reference_counts.get(workspace, 0),
                str(workspace_error) if workspace_error else None,
            )
        if hits:
            await conn.executemany(
                """
                INSERT INTO query_document_hits (
                  query_event_id, workspace_slug, reference_id, title,
                  source_uri, rank, document_id
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7::uuid)
                """,
                [
                    (
                        event_id,
                        hit["workspace_slug"],
                        hit["reference_id"],
                        hit["title"],
                        hit["source_uri"],
                        hit["rank"],
                        hit["document_id"],
                    )
                    for hit in hits
                ],
            )


def serialize_analytics_rows(rows: list[Any]) -> list[dict[str, Any]]:
    serialized = []
    for row in rows:
        data = dict(row)
        serialized.append({key: analytics_iso(value) for key, value in data.items()})
    return serialized


def build_analytics_response(
    *,
    period_days: int,
    limit: int,
    summary: dict[str, Any] | None,
    top_documents: list[Any],
    top_users: list[Any],
    recent_queries: list[Any],
    workspace_usage: list[Any],
    top_questions: list[Any],
) -> dict[str, Any]:
    summary = dict(summary or {})
    total_queries = analytics_int(summary.get("total_queries"))
    successful_queries = analytics_int(summary.get("successful_queries"))
    return {
        "period_days": period_days,
        "limit": limit,
        "summary": {
            "total_queries": total_queries,
            "unique_users": analytics_int(summary.get("unique_users")),
            "successful_queries": successful_queries,
            "failed_queries": analytics_int(summary.get("failed_queries")),
            "success_rate": (successful_queries / total_queries) if total_queries else 0,
            "avg_latency_ms": analytics_float(summary.get("avg_latency_ms")),
            "document_hits": analytics_int(summary.get("document_hits")),
        },
        "top_documents": serialize_analytics_rows(top_documents),
        "top_users": serialize_analytics_rows(top_users),
        "recent_queries": serialize_analytics_rows(recent_queries),
        "workspace_usage": serialize_analytics_rows(workspace_usage),
        "top_questions": serialize_analytics_rows(top_questions),
    }


@app.post("/query")
async def query(payload: dict[str, Any]) -> dict[str, Any]:
    started_at = time.monotonic()
    user_groups = payload.get("groups") or []
    allowed = allowed_workspaces(user_groups)
    headers = {"X-API-Key": LIGHTRAG_API_KEY} if LIGHTRAG_API_KEY else {}
    semaphore = asyncio.Semaphore(max(1, QUERY_WORKSPACE_CONCURRENCY))

    async def query_workspace(client: httpx.AsyncClient, workspace: str) -> dict[str, Any]:
        workspace_started_at = time.monotonic()
        base_url = LIGHTRAG_URLS.get(workspace)
        if not base_url:
            return {
                "workspace": workspace,
                "error": "workspace is not configured",
                "latency_ms": int((time.monotonic() - workspace_started_at) * 1000),
            }
        async with semaphore:
            try:
                response = await client.post(
                    f"{base_url}/query",
                    json={
                        "query": payload.get("query"),
                        "mode": payload.get("mode", "mix"),
                        "include_references": payload.get("include_references", True),
                    },
                    headers=headers,
                )
                response.raise_for_status()
                return {
                    "workspace": workspace,
                    "result": response.json(),
                    "latency_ms": int((time.monotonic() - workspace_started_at) * 1000),
                }
            except Exception as exc:
                return {
                    "workspace": workspace,
                    "error": str(exc),
                    "latency_ms": int((time.monotonic() - workspace_started_at) * 1000),
                }

    async with httpx.AsyncClient(timeout=180) as client:
        answers = await asyncio.gather(*(query_workspace(client, workspace) for workspace in allowed))
    latency_ms = int((time.monotonic() - started_at) * 1000)
    status = "ok" if any(not answer.get("error") for answer in answers) else "error"
    error = None if status == "ok" else "; ".join(str(answer.get("error")) for answer in answers if answer.get("error"))
    try:
        await record_query_analytics(
            app.state.pool,
            actor_email=str(payload.get("actor_email") or "") or None,
            actor_role=str(payload.get("actor_role") or "member"),
            actor_groups=[str(group) for group in user_groups],
            query_text=str(payload.get("query") or ""),
            mode=str(payload.get("mode") or "mix"),
            allowed_workspaces=allowed,
            answers=answers,
            latency_ms=latency_ms,
            status=status,
            error=error,
        )
    except Exception as exc:
        print(f"query analytics write failed: {exc}", flush=True)
    return {
        "status": "ok",
        "query": payload.get("query"),
        "allowed_workspaces": allowed,
        "answers": answers,
    }


@app.post("/documents/text")
async def ingest_text(payload: dict[str, Any]) -> dict[str, Any]:
    workspace = route_workspace(payload)
    if workspace not in LIGHTRAG_URLS:
        return {"status": "error", "reason": f"unknown workspace {workspace}"}

    title = payload["title"]
    text = payload["text"]
    classification = payload.get("classification", "internal")
    department = payload.get("department") or ("c_level" if workspace == C_LEVEL_WORKSPACE else "public")
    source_text = (
        f"Title: {title}\n"
        f"Workspace: {workspace}\n"
        f"Department: {department}\n"
        f"Classification: {classification}\n\n"
        f"{text}"
    )

    async with app.state.pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO documents (title, department_slug, classification, status, queued_at)
            VALUES ($1, $2, $3, 'queued', now())
            RETURNING id
            """,
            title,
            department,
            classification,
        )
        workspace_row = await conn.fetchrow(
            "SELECT id FROM rag_workspaces WHERE slug = $1",
            workspace,
        )
        if workspace_row:
            await conn.execute(
                """
                INSERT INTO document_workspace_membership (document_id, workspace_id)
                VALUES ($1, $2)
                ON CONFLICT DO NOTHING
                """,
                row["id"],
                workspace_row["id"],
            )
        await conn.execute(
            """
            INSERT INTO document_ingest_payloads (document_id, workspace_slug, title, source_text)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (document_id) DO UPDATE
            SET workspace_slug = EXCLUDED.workspace_slug,
                title = EXCLUDED.title,
                source_text = EXCLUDED.source_text,
                updated_at = now()
            """,
            row["id"],
            workspace,
            title,
            source_text,
        )

    await app.state.redis.rpush(INGEST_QUEUE_NAME, str(row["id"]))

    return {
        "status": "queued",
        "document_id": str(row["id"]),
        "workspace": workspace,
        "queue": INGEST_QUEUE_NAME,
    }


@app.get("/sources")
async def list_sources() -> dict[str, Any]:
    rows = await app.state.pool.fetch(
        """
        SELECT id, name, source_type, target, workspace_slug, classification, config,
               interval_minutes, enabled, next_scan_at, last_scan_at, last_status,
               last_error, created_at, updated_at
        FROM document_sources
        ORDER BY created_at DESC
        """
    )
    return {
        "sources": [serialize_source_row(row) for row in rows],
        "scan_queue": SOURCE_SCAN_QUEUE_NAME,
        "scan_queue_depth": await app.state.redis.llen(SOURCE_SCAN_QUEUE_NAME),
    }


@app.post("/sources")
async def create_source(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        source = normalize_source_payload(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    async with app.state.pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO document_sources (
              name, source_type, target, workspace_slug, classification, config,
              interval_minutes, enabled, next_scan_at
            )
            VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8,
                    CASE WHEN $8 THEN now() ELSE NULL END)
            RETURNING id, name, source_type, target, workspace_slug, classification,
                      config, interval_minutes, enabled, next_scan_at, last_scan_at,
                      last_status, last_error, created_at, updated_at
            """,
            source["name"],
            source["source_type"],
            source["target"],
            source["workspace"],
            source["classification"],
            json.dumps(source["config"]),
            source["interval_minutes"],
            source["enabled"],
        )
    return {"status": "created", "source": serialize_source_row(row)}


@app.patch("/sources/{source_id}")
async def update_source(source_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        async with app.state.pool.acquire() as conn:
            existing = await conn.fetchrow(
                """
                SELECT id, name, source_type, target, classification, config,
                       interval_minutes, enabled
                FROM document_sources
                WHERE id = $1::uuid
                """,
                source_id,
            )
            if not existing:
                raise HTTPException(status_code=404, detail="source not found")
            merged = {
                "name": payload.get("name", existing["name"]),
                "source_type": payload.get("source_type", existing["source_type"]),
                "target": payload.get("target", existing["target"]),
                "classification": payload.get("classification", existing["classification"]),
                "interval_minutes": payload.get("interval_minutes", existing["interval_minutes"]),
                "enabled": payload.get("enabled", existing["enabled"]),
                "config": payload.get("config") or {},
            }
            source = normalize_source_payload(
                merged,
                existing_config=source_config_from_value(existing["config"]),
            )
            reset_schedule = coerce_bool(payload.get("reset_schedule"), True)
            row = await conn.fetchrow(
                """
                UPDATE document_sources
                SET name = $2,
                    source_type = $3,
                    target = $4,
                    workspace_slug = $5,
                    classification = $6,
                    config = $7::jsonb,
                    interval_minutes = $8,
                    enabled = $9,
                    next_scan_at = CASE
                      WHEN NOT $9 THEN NULL
                      WHEN $10 THEN now()
                      ELSE next_scan_at
                    END,
                    updated_at = now()
                WHERE id = $1::uuid
                RETURNING id, name, source_type, target, workspace_slug, classification,
                          config, interval_minutes, enabled, next_scan_at, last_scan_at,
                          last_status, last_error, created_at, updated_at
                """,
                source_id,
                source["name"],
                source["source_type"],
                source["target"],
                source["workspace"],
                source["classification"],
                json.dumps(source["config"]),
                source["interval_minutes"],
                source["enabled"],
                reset_schedule,
            )
    except asyncpg.PostgresError as exc:
        raise HTTPException(status_code=400, detail="invalid source id") from exc
    return {"status": "updated", "source": serialize_source_row(row)}


@app.post("/sources/{source_id}/scan")
async def queue_source_scan(source_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    trigger = str((payload or {}).get("trigger") or "manual").strip().lower()
    if trigger not in {"manual", "scheduled", "startup"}:
        trigger = "manual"
    try:
        async with app.state.pool.acquire() as conn:
            source = await conn.fetchrow("SELECT id FROM document_sources WHERE id = $1::uuid", source_id)
            if not source:
                raise HTTPException(status_code=404, detail="source not found")
            run = await conn.fetchrow(
                """
                INSERT INTO source_scan_runs (source_id, trigger, status)
                VALUES ($1::uuid, $2, 'queued')
                RETURNING id, source_id, trigger, status, queued_at, started_at,
                          finished_at, items_found, documents_queued, error
                """,
                source_id,
                trigger,
            )
    except asyncpg.PostgresError as exc:
        raise HTTPException(status_code=400, detail="invalid source id") from exc
    await app.state.redis.rpush(SOURCE_SCAN_QUEUE_NAME, str(run["id"]))
    return {"status": "queued", "run": serialize_scan_run(run), "queue": SOURCE_SCAN_QUEUE_NAME}


@app.get("/sources/{source_id}/runs")
async def source_runs(source_id: str) -> dict[str, Any]:
    try:
        async with app.state.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, source_id, trigger, status, queued_at, started_at,
                       finished_at, items_found, documents_queued, error
                FROM source_scan_runs
                WHERE source_id = $1::uuid
                ORDER BY queued_at DESC
                LIMIT 20
                """,
                source_id,
            )
    except asyncpg.PostgresError as exc:
        raise HTTPException(status_code=400, detail="invalid source id") from exc
    return {"runs": [serialize_scan_run(row) for row in rows]}


@app.get("/documents/{document_id}")
async def document_status(document_id: str) -> dict[str, Any]:
    try:
        async with app.state.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, title, department_slug, classification, status, queued_at,
                       indexed_at, ingest_error, created_at
                FROM documents
                WHERE id = $1::uuid
                """,
                document_id,
            )
    except asyncpg.PostgresError as exc:
        raise HTTPException(status_code=400, detail="invalid document id") from exc
    if not row:
        raise HTTPException(status_code=404, detail="document not found")
    return {key: str(value) if key == "id" or hasattr(value, "isoformat") else value for key, value in dict(row).items()}


@app.get("/queue/status")
async def queue_status() -> dict[str, Any]:
    async with app.state.pool.acquire() as conn:
        rows = await conn.fetch("SELECT status, count(*) AS count FROM documents GROUP BY status ORDER BY status")
    return {
        "queue": INGEST_QUEUE_NAME,
        "queue_depth": await app.state.redis.llen(INGEST_QUEUE_NAME),
        "source_scan_queue": SOURCE_SCAN_QUEUE_NAME,
        "source_scan_queue_depth": await app.state.redis.llen(SOURCE_SCAN_QUEUE_NAME),
        "document_status_counts": {row["status"]: row["count"] for row in rows},
        "worker_expected": True,
    }


@app.get("/analytics")
async def analytics(days: int = 30, limit: int = 20) -> dict[str, Any]:
    period_days = clamp_analytics_days(days)
    row_limit = clamp_analytics_limit(limit)
    async with app.state.pool.acquire() as conn:
        summary = await conn.fetchrow(
            """
            SELECT
              count(*)::int AS total_queries,
              count(DISTINCT actor_email)::int AS unique_users,
              count(*) FILTER (WHERE status = 'ok')::int AS successful_queries,
              count(*) FILTER (WHERE status = 'error')::int AS failed_queries,
              coalesce(avg(latency_ms), 0)::float AS avg_latency_ms,
              (
                SELECT count(*)::int
                FROM query_document_hits h
                JOIN query_events qh ON qh.id = h.query_event_id
                WHERE qh.created_at >= now() - ($1::int * interval '1 day')
              ) AS document_hits
            FROM query_events q
            WHERE q.created_at >= now() - ($1::int * interval '1 day')
            """,
            period_days,
        )
        top_documents = await conn.fetch(
            """
            SELECT
              coalesce(d.title, h.title, h.source_uri, h.reference_id, 'unknown') AS title,
              coalesce(d.source_uri, h.source_uri) AS source_uri,
              h.workspace_slug,
              count(*)::int AS hits,
              count(DISTINCT q.actor_email)::int AS unique_users,
              max(h.created_at) AS last_accessed_at
            FROM query_document_hits h
            JOIN query_events q ON q.id = h.query_event_id
            LEFT JOIN documents d ON d.id = h.document_id
            WHERE q.created_at >= now() - ($1::int * interval '1 day')
            GROUP BY coalesce(d.title, h.title, h.source_uri, h.reference_id, 'unknown'),
                     coalesce(d.source_uri, h.source_uri),
                     h.workspace_slug
            ORDER BY hits DESC, last_accessed_at DESC
            LIMIT $2
            """,
            period_days,
            row_limit,
        )
        top_users = await conn.fetch(
            """
            SELECT
              coalesce(actor_email, 'unknown') AS actor_email,
              max(actor_role) AS actor_role,
              count(*)::int AS query_count,
              count(*) FILTER (WHERE status = 'error')::int AS failed_queries,
              max(created_at) AS last_query_at
            FROM query_events
            WHERE created_at >= now() - ($1::int * interval '1 day')
            GROUP BY coalesce(actor_email, 'unknown')
            ORDER BY query_count DESC, last_query_at DESC
            LIMIT $2
            """,
            period_days,
            row_limit,
        )
        recent_queries = await conn.fetch(
            """
            SELECT
              q.created_at,
              coalesce(q.actor_email, 'unknown') AS actor_email,
              q.actor_role,
              q.query_text,
              q.mode,
              q.allowed_workspaces,
              q.status,
              q.latency_ms,
              q.error,
              count(h.id)::int AS document_count
            FROM query_events q
            LEFT JOIN query_document_hits h ON h.query_event_id = q.id
            WHERE q.created_at >= now() - ($1::int * interval '1 day')
            GROUP BY q.id
            ORDER BY q.created_at DESC
            LIMIT $2
            """,
            period_days,
            row_limit,
        )
        workspace_usage = await conn.fetch(
            """
            SELECT
              w.workspace_slug,
              count(*)::int AS query_count,
              count(*) FILTER (WHERE w.status = 'error')::int AS error_count,
              coalesce(avg(w.latency_ms), 0)::float AS avg_latency_ms,
              sum(w.reference_count)::int AS reference_count,
              max(w.created_at) AS last_query_at
            FROM query_workspace_events w
            JOIN query_events q ON q.id = w.query_event_id
            WHERE q.created_at >= now() - ($1::int * interval '1 day')
            GROUP BY w.workspace_slug
            ORDER BY query_count DESC, w.workspace_slug
            LIMIT $2
            """,
            period_days,
            row_limit,
        )
        top_questions = await conn.fetch(
            """
            SELECT
              query_text,
              count(*)::int AS count,
              count(DISTINCT actor_email)::int AS unique_users,
              max(created_at) AS last_asked_at
            FROM query_events
            WHERE created_at >= now() - ($1::int * interval '1 day')
            GROUP BY query_text
            ORDER BY count DESC, last_asked_at DESC
            LIMIT $2
            """,
            period_days,
            row_limit,
        )
    return build_analytics_response(
        period_days=period_days,
        limit=row_limit,
        summary=dict(summary or {}),
        top_documents=top_documents,
        top_users=top_users,
        recent_queries=recent_queries,
        workspace_usage=workspace_usage,
        top_questions=top_questions,
    )
