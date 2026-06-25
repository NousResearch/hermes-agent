import os
import asyncio
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
QUERY_WORKSPACE_CONCURRENCY = int(os.environ.get("QUERY_WORKSPACE_CONCURRENCY", "4"))
LIGHTRAG_ROOT = Path(os.environ.get("LIGHTRAG_ROOT", "/data/lightrag"))
DEPARTMENT_WORKSPACES = [
    "company_public",
    "company_internal",
    "department_marketing",
    "department_financial",
    "department_hr",
    "department_engineering",
    "department_c_level",
]
LIGHTRAG_API_KEY = os.environ.get("LIGHTRAG_API_KEY", "")
LIGHTRAG_URLS = {
    "company_public": os.environ.get("LIGHTRAG_COMPANY_PUBLIC_URL"),
    "company_internal": os.environ.get("LIGHTRAG_COMPANY_INTERNAL_URL"),
    "department_marketing": os.environ.get("LIGHTRAG_DEPARTMENT_MARKETING_URL"),
    "department_financial": os.environ.get("LIGHTRAG_DEPARTMENT_FINANCIAL_URL"),
    "department_hr": os.environ.get("LIGHTRAG_DEPARTMENT_HR_URL"),
    "department_engineering": os.environ.get("LIGHTRAG_DEPARTMENT_ENGINEERING_URL"),
    "department_c_level": os.environ.get("LIGHTRAG_DEPARTMENT_C_LEVEL_URL"),
}


@app.on_event("startup")
async def startup() -> None:
    app.state.pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
    app.state.redis = redis.from_url(REDIS_URL, decode_responses=True)
    await ensure_runtime_schema()
    for workspace in DEPARTMENT_WORKSPACES:
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


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/workspaces")
async def workspaces() -> dict[str, Any]:
    rows = await app.state.pool.fetch(
        "SELECT slug, name, visibility_boundary FROM rag_workspaces ORDER BY slug"
    )
    return {
        "workspaces": [dict(row) for row in rows],
        "lightrag_root": str(LIGHTRAG_ROOT),
    }


def route_workspace(payload: dict[str, Any]) -> str:
    visibility = payload.get("visibility", "department")
    department = str(payload.get("department", "")).lower().replace(" ", "_")
    if visibility == "company":
        classification = payload.get("classification", "internal")
        return "company_public" if classification == "public" else "company_internal"
    if visibility == "department":
        return f"department_{department}"
    raise ValueError("only company and department visibility are enabled in this MVP")


def allowed_workspaces(groups: list[str]) -> list[str]:
    allowed = ["company_public", "company_internal"]
    for group in groups:
        if group.startswith("department_") and group in LIGHTRAG_URLS:
            allowed.append(group)
    return sorted(set(allowed))


@app.post("/query")
async def query(payload: dict[str, Any]) -> dict[str, Any]:
    user_groups = payload.get("groups") or []
    allowed = allowed_workspaces(user_groups)
    headers = {"X-API-Key": LIGHTRAG_API_KEY} if LIGHTRAG_API_KEY else {}
    semaphore = asyncio.Semaphore(max(1, QUERY_WORKSPACE_CONCURRENCY))

    async def query_workspace(client: httpx.AsyncClient, workspace: str) -> dict[str, Any]:
        base_url = LIGHTRAG_URLS.get(workspace)
        if not base_url:
            return {"workspace": workspace, "error": "workspace is not configured"}
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
                return {"workspace": workspace, "result": response.json()}
            except Exception as exc:
                return {"workspace": workspace, "error": str(exc)}

    async with httpx.AsyncClient(timeout=180) as client:
        answers = await asyncio.gather(*(query_workspace(client, workspace) for workspace in allowed))
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
    department = payload.get("department")
    source_text = (
        f"Title: {title}\n"
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
        "document_status_counts": {row["status"]: row["count"] for row in rows},
        "worker_expected": True,
    }
