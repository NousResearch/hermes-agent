import asyncio
import os
from typing import Any

import asyncpg
import httpx
import redis.asyncio as redis


DATABASE_URL = os.environ["DATABASE_URL"]
REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")
INGEST_QUEUE_NAME = os.environ.get("INGEST_QUEUE_NAME", "second_brain:ingest_jobs")
INGEST_WORKER_CONCURRENCY = int(os.environ.get("INGEST_WORKER_CONCURRENCY", "2"))
INGEST_MAX_ATTEMPTS = int(os.environ.get("INGEST_MAX_ATTEMPTS", "3"))
INGEST_RETRY_BACKOFF_SECONDS = int(os.environ.get("INGEST_RETRY_BACKOFF_SECONDS", "10"))
LIGHTRAG_API_KEY = os.environ.get("LIGHTRAG_API_KEY", "")
LIGHTRAG_URLS = {
    "company_public": os.environ.get("LIGHTRAG_COMPANY_PUBLIC_URL"),
    "department_c_level": os.environ.get("LIGHTRAG_DEPARTMENT_C_LEVEL_URL"),
}


async def ensure_runtime_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
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


async def bootstrap_queue(pool: asyncpg.Pool, redis_client: redis.Redis) -> None:
    rows = await pool.fetch(
        """
        SELECT p.document_id
        FROM document_ingest_payloads p
        JOIN documents d ON d.id = p.document_id
        WHERE d.status IN ('queued', 'indexing')
        ORDER BY d.queued_at NULLS LAST, d.created_at
        """
    )
    if rows:
        await redis_client.rpush(INGEST_QUEUE_NAME, *(str(row["document_id"]) for row in rows))
        print(f"bootstrapped {len(rows)} ingest jobs", flush=True)


async def mark_failed(pool: asyncpg.Pool, document_id: str, error: str, *, retry: bool) -> None:
    status = "queued" if retry else "failed"
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE documents
            SET status = $2, ingest_error = $3
            WHERE id = $1::uuid
            """,
            document_id,
            status,
            error[:4000],
        )


async def process_document(pool: asyncpg.Pool, redis_client: redis.Redis, document_id: str) -> None:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT p.document_id, p.workspace_slug, p.title, p.source_text, p.attempts,
                   d.status
            FROM document_ingest_payloads p
            JOIN documents d ON d.id = p.document_id
            WHERE p.document_id = $1::uuid
            """,
            document_id,
        )
        if not row:
            return
        if row["status"] == "indexed":
            await conn.execute("DELETE FROM document_ingest_payloads WHERE document_id = $1::uuid", document_id)
            return
        attempts = int(row["attempts"]) + 1
        await conn.execute(
            """
            UPDATE document_ingest_payloads
            SET attempts = $2, updated_at = now()
            WHERE document_id = $1::uuid
            """,
            document_id,
            attempts,
        )
        await conn.execute(
            """
            UPDATE documents
            SET status = 'indexing', ingest_error = NULL
            WHERE id = $1::uuid
            """,
            document_id,
        )

    workspace = str(row["workspace_slug"])
    base_url = LIGHTRAG_URLS.get(workspace)
    if not base_url:
        await mark_failed(pool, document_id, f"workspace is not configured: {workspace}", retry=False)
        return

    headers = {"X-API-Key": LIGHTRAG_API_KEY} if LIGHTRAG_API_KEY else {}
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(
                f"{base_url}/documents/text",
                json={"text": row["source_text"], "file_source": row["title"]},
                headers=headers,
            )
            if response.is_error:
                raise RuntimeError(f"LightRAG HTTP {response.status_code}: {response.text[:2000]}")
            lightrag_result: dict[str, Any] = response.json()
    except Exception as exc:
        retry = attempts < INGEST_MAX_ATTEMPTS
        await mark_failed(pool, document_id, str(exc), retry=retry)
        if retry:
            await asyncio.sleep(INGEST_RETRY_BACKOFF_SECONDS)
            await redis_client.rpush(INGEST_QUEUE_NAME, document_id)
        print(f"ingest failed document_id={document_id} attempts={attempts} retry={retry}: {exc}", flush=True)
        return

    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE documents
            SET status = 'indexed', indexed_at = now(), ingest_error = NULL
            WHERE id = $1::uuid
            """,
            document_id,
        )
        await conn.execute("DELETE FROM document_ingest_payloads WHERE document_id = $1::uuid", document_id)
    print(f"indexed document_id={document_id} workspace={workspace} result={lightrag_result}", flush=True)


async def worker_loop(worker_id: int, pool: asyncpg.Pool, redis_client: redis.Redis) -> None:
    while True:
        item = await redis_client.blpop(INGEST_QUEUE_NAME, timeout=10)
        if not item:
            continue
        _, document_id = item
        try:
            await process_document(pool, redis_client, str(document_id))
        except Exception as exc:
            print(f"worker={worker_id} unhandled error document_id={document_id}: {exc}", flush=True)


async def main() -> None:
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=max(2, INGEST_WORKER_CONCURRENCY + 1))
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    try:
        await ensure_runtime_schema(pool)
        await bootstrap_queue(pool, redis_client)
        await asyncio.gather(
            *(worker_loop(worker_id, pool, redis_client) for worker_id in range(INGEST_WORKER_CONCURRENCY))
        )
    finally:
        await redis_client.aclose()
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
