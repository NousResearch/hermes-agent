import asyncio
import hashlib
import json
import os
import re
from typing import Any
from urllib.parse import parse_qs, urlparse

import asyncpg
import httpx
import redis.asyncio as redis


DATABASE_URL = os.environ["DATABASE_URL"]
REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")
INGEST_QUEUE_NAME = os.environ.get("INGEST_QUEUE_NAME", "second_brain:ingest_jobs")
SOURCE_SCAN_QUEUE_NAME = os.environ.get("SOURCE_SCAN_QUEUE_NAME", "second_brain:source_scan_jobs")
INGEST_WORKER_CONCURRENCY = int(os.environ.get("INGEST_WORKER_CONCURRENCY", "2"))
SOURCE_SCAN_WORKER_CONCURRENCY = int(os.environ.get("SOURCE_SCAN_WORKER_CONCURRENCY", "1"))
SOURCE_SCHEDULER_POLL_SECONDS = int(os.environ.get("SOURCE_SCHEDULER_POLL_SECONDS", "60"))
SOURCE_SCAN_MAX_PAGES = int(os.environ.get("SOURCE_SCAN_MAX_PAGES", "50"))
SOURCE_MAX_TEXT_BYTES = int(os.environ.get("SOURCE_MAX_TEXT_BYTES", str(5 * 1024 * 1024)))
INGEST_MAX_ATTEMPTS = int(os.environ.get("INGEST_MAX_ATTEMPTS", "3"))
INGEST_RETRY_BACKOFF_SECONDS = int(os.environ.get("INGEST_RETRY_BACKOFF_SECONDS", "10"))
NOTION_VERSION = os.environ.get("NOTION_VERSION", "2026-03-11")
LIGHTRAG_API_KEY = os.environ.get("LIGHTRAG_API_KEY", "")
LIGHTRAG_URLS = {
    "company_public": os.environ.get("LIGHTRAG_COMPANY_PUBLIC_URL"),
    "department_c_level": os.environ.get("LIGHTRAG_DEPARTMENT_C_LEVEL_URL"),
}
NOTION_ID_RE = re.compile(r"([0-9a-fA-F]{32})")


async def ensure_runtime_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS queued_at timestamptz")
        await conn.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS indexed_at timestamptz")
        await conn.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS ingest_error text")
        await conn.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS checksum text")
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
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS document_dedupe_keys (
              workspace_slug text NOT NULL,
              checksum text NOT NULL,
              document_id uuid REFERENCES documents(id) ON DELETE SET NULL,
              created_at timestamptz NOT NULL DEFAULT now(),
              updated_at timestamptz NOT NULL DEFAULT now(),
              PRIMARY KEY (workspace_slug, checksum)
            )
            """
        )
        await backfill_document_checksums(conn)
        await conn.execute(
            """
            INSERT INTO document_dedupe_keys (workspace_slug, checksum, document_id)
            SELECT DISTINCT ON (rw.slug, d.checksum)
                   rw.slug, d.checksum, d.id
            FROM documents d
            JOIN document_workspace_membership dwm ON dwm.document_id = d.id
            JOIN rag_workspaces rw ON rw.id = dwm.workspace_id
            WHERE d.checksum IS NOT NULL
            ORDER BY rw.slug, d.checksum,
                     CASE WHEN d.status = 'indexed' THEN 0 ELSE 1 END,
                     d.created_at
            ON CONFLICT (workspace_slug, checksum) DO NOTHING
            """
        )


async def backfill_document_checksums(conn: Any) -> None:
    rows = await conn.fetch(
        """
        SELECT d.id, p.source_text
        FROM documents d
        JOIN document_ingest_payloads p ON p.document_id = d.id
        WHERE d.checksum IS NULL AND p.source_text IS NOT NULL
        """
    )
    for row in rows:
        checksum = checksum_text(extract_ingest_body(row["source_text"]))
        await conn.execute(
            "UPDATE documents SET checksum = $1 WHERE id = $2 AND checksum IS NULL",
            checksum,
            row["id"],
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
    scan_rows = await pool.fetch(
        """
        SELECT id
        FROM source_scan_runs
        WHERE status IN ('queued', 'running')
        ORDER BY queued_at
        """
    )
    if scan_rows:
        await redis_client.rpush(SOURCE_SCAN_QUEUE_NAME, *(str(row["id"]) for row in scan_rows))
        print(f"bootstrapped {len(scan_rows)} source scan jobs", flush=True)


def source_config_from_value(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        loaded = json.loads(value)
        return loaded if isinstance(loaded, dict) else {}
    return {}


def checksum_text(text: str) -> str:
    return hashlib.sha256(str(text or "").strip().encode("utf-8")).hexdigest()


def extract_ingest_body(source_text: str) -> str:
    parts = str(source_text or "").split("\n\n", 1)
    return parts[1] if len(parts) == 2 else parts[0]


def compact_error_text(text: str, limit: int = 500) -> str:
    return " ".join(str(text or "").split())[:limit]


def canonical_uuid(raw: str) -> str:
    value = raw.replace("-", "").lower()
    if len(value) != 32:
        raise ValueError("expected 32 hex characters")
    return f"{value[0:8]}-{value[8:12]}-{value[12:16]}-{value[16:20]}-{value[20:32]}"


def extract_notion_id(value: str) -> str:
    text = str(value or "").strip()
    match = NOTION_ID_RE.search(text.replace("-", ""))
    if not match:
        raise ValueError("could not find a Notion page or data source id")
    return canonical_uuid(match.group(1))


def title_from_notion_page(page: dict[str, Any]) -> str:
    for prop in (page.get("properties") or {}).values():
        title = prop.get("title") if isinstance(prop, dict) else None
        if title:
            text = "".join(part.get("plain_text", "") for part in title if isinstance(part, dict)).strip()
            if text:
                return text
    return page.get("id", "Notion Page")


def parse_drive_public_link(url: str) -> dict[str, str]:
    parsed = urlparse(str(url or "").strip())
    host = parsed.netloc.lower()
    path = parsed.path
    query = parse_qs(parsed.query)
    if "docs.google.com" in host:
        match = re.search(r"/(document|spreadsheets|presentation)/d/([^/]+)", path)
        if not match:
            raise ValueError("unsupported Google Docs URL")
        kind = {
            "document": "google_doc",
            "spreadsheets": "google_sheet",
            "presentation": "google_slide",
        }[match.group(1)]
        return {"kind": kind, "file_id": match.group(2)}
    if "drive.google.com" in host:
        folder = re.search(r"/drive/folders/([^/]+)", path)
        if folder:
            return {"kind": "drive_folder", "file_id": folder.group(1)}
        file_match = re.search(r"/file/d/([^/]+)", path)
        if file_match:
            return {"kind": "drive_file", "file_id": file_match.group(1)}
        if query.get("id"):
            return {"kind": "drive_file", "file_id": query["id"][0]}
    raise ValueError("unsupported Drive public URL")


def drive_public_download_url(parsed: dict[str, str]) -> str:
    kind = parsed["kind"]
    file_id = parsed["file_id"]
    if kind == "google_doc":
        return f"https://docs.google.com/document/d/{file_id}/export?format=txt"
    if kind == "google_sheet":
        return f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=csv"
    if kind == "google_slide":
        return f"https://docs.google.com/presentation/d/{file_id}/export/txt"
    if kind == "drive_file":
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    raise ValueError("public Drive folder listing requires Drive API/OAuth; add file or Google Docs links instead")


def source_workspace(target: str, classification: str) -> str:
    normalized_target = str(target or "public").strip().lower().replace("-", "_")
    normalized_classification = str(classification or "internal").strip().lower()
    if normalized_target in {"c_level", "clevel", "department_c_level"} or normalized_classification in {
        "confidential",
        "restricted",
    }:
        return "department_c_level"
    return "company_public"


async def notion_request(
    client: httpx.AsyncClient,
    method: str,
    path: str,
    api_key: str,
    *,
    body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    response = await client.request(
        method,
        f"https://api.notion.com{path}",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Notion-Version": NOTION_VERSION,
            "Content-Type": "application/json",
        },
        json=body,
    )
    if response.is_error:
        raise RuntimeError(f"Notion HTTP {response.status_code}: {compact_error_text(response.text)}")
    return response.json()


async def fetch_notion_markdown(client: httpx.AsyncClient, api_key: str, page_id: str) -> tuple[str, bool]:
    data = await notion_request(client, "GET", f"/v1/pages/{page_id}/markdown", api_key)
    text = data.get("markdown") or ""
    for block_id in data.get("unknown_block_ids") or []:
        try:
            extra = await notion_request(client, "GET", f"/v1/pages/{block_id}/markdown", api_key)
        except Exception:
            continue
        if extra.get("markdown"):
            text += f"\n\n{extra['markdown']}"
    return text.strip(), bool(data.get("truncated"))


async def collect_notion_documents(config: dict[str, Any]) -> list[dict[str, Any]]:
    api_key = str(config.get("notion_api_key") or "").strip()
    if not api_key:
        raise RuntimeError("notion_api_key is missing")
    max_pages = max(1, min(int(config.get("max_pages") or SOURCE_SCAN_MAX_PAGES), 100))
    page_ids: list[str] = []
    for key in ("notion_page_id", "page_id"):
        if config.get(key):
            page_ids.append(extract_notion_id(str(config[key])))
    for key in ("notion_page_url", "page_url"):
        if config.get(key):
            page_ids.append(extract_notion_id(str(config[key])))
    for key in ("notion_page_ids", "page_ids", "notion_page_urls", "page_urls"):
        values = config.get(key) or []
        if isinstance(values, str):
            values = [item.strip() for item in values.split(",") if item.strip()]
        for value in values:
            page_ids.append(extract_notion_id(str(value)))

    documents: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=120) as client:
        data_source_id = config.get("notion_data_source_id") or config.get("data_source_id")
        if data_source_id:
            cursor = None
            while len(page_ids) < max_pages:
                body: dict[str, Any] = {"page_size": min(100, max_pages - len(page_ids))}
                if cursor:
                    body["start_cursor"] = cursor
                result = await notion_request(
                    client,
                    "POST",
                    f"/v1/data_sources/{extract_notion_id(str(data_source_id))}/query",
                    api_key,
                    body=body,
                )
                for item in result.get("results") or []:
                    if item.get("object") == "page" and item.get("id"):
                        page_ids.append(str(item["id"]))
                if not result.get("has_more") or not result.get("next_cursor"):
                    break
                cursor = result["next_cursor"]

        if not page_ids:
            cursor = None
            while len(page_ids) < max_pages:
                body = {
                    "page_size": min(100, max_pages - len(page_ids)),
                    "filter": {"property": "object", "value": "page"},
                }
                if config.get("notion_search_query"):
                    body["query"] = str(config["notion_search_query"])
                if cursor:
                    body["start_cursor"] = cursor
                result = await notion_request(client, "POST", "/v1/search", api_key, body=body)
                for item in result.get("results") or []:
                    if item.get("object") == "page" and item.get("id"):
                        page_ids.append(str(item["id"]))
                if not result.get("has_more") or not result.get("next_cursor"):
                    break
                cursor = result["next_cursor"]

        seen: set[str] = set()
        for page_id in page_ids[:max_pages]:
            canonical_id = extract_notion_id(page_id)
            if canonical_id in seen:
                continue
            seen.add(canonical_id)
            page = await notion_request(client, "GET", f"/v1/pages/{canonical_id}", api_key)
            markdown, truncated = await fetch_notion_markdown(client, api_key, canonical_id)
            if not markdown:
                continue
            title = title_from_notion_page(page)
            documents.append(
                {
                    "external_id": f"notion:{canonical_id}",
                    "title": title,
                    "source_uri": page.get("url") or f"notion://page/{canonical_id}",
                    "text": f"# {title}\n\n{markdown}\n\nNotion truncated: {truncated}",
                }
            )
    return documents


async def collect_drive_public_documents(config: dict[str, Any]) -> list[dict[str, Any]]:
    urls = config.get("drive_urls") or config.get("drive_url") or []
    if isinstance(urls, str):
        urls = [item.strip() for item in urls.split(",") if item.strip()]
    if not urls:
        raise RuntimeError("drive_url or drive_urls is missing")
    documents: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
        for url in urls:
            parsed = parse_drive_public_link(str(url))
            download_url = drive_public_download_url(parsed)
            response = await client.get(download_url)
            if response.is_error:
                raise RuntimeError(f"Drive HTTP {response.status_code}: {compact_error_text(response.text)}")
            content = response.content[:SOURCE_MAX_TEXT_BYTES]
            text = content.decode(response.encoding or "utf-8", errors="replace").strip()
            if not text:
                continue
            title = str(config.get("title") or config.get("name") or f"Drive {parsed['file_id']}").strip()
            documents.append(
                {
                    "external_id": f"drive:{parsed['kind']}:{parsed['file_id']}",
                    "title": title,
                    "source_uri": str(url),
                    "text": text,
                }
            )
    return documents


async def collect_source_documents(source_type: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    if source_type == "notion":
        return await collect_notion_documents(config)
    if source_type == "drive_public":
        return await collect_drive_public_documents(config)
    raise RuntimeError(f"unsupported source type: {source_type}")


async def queue_scanned_document(
    pool: asyncpg.Pool,
    redis_client: redis.Redis,
    *,
    source_id: str,
    workspace: str,
    classification: str,
    document: dict[str, Any],
) -> bool:
    text = str(document.get("text") or "").strip()
    if not text:
        return False
    checksum = checksum_text(text)
    external_id = str(document["external_id"])
    document_id: Any = None
    async with pool.acquire() as conn:
        async with conn.transaction():
            existing = await conn.fetchrow(
                """
                SELECT checksum
                FROM source_items
                WHERE source_id = $1::uuid AND external_id = $2
                """,
                source_id,
                external_id,
            )
            if existing and existing["checksum"] == checksum:
                await conn.execute(
                    """
                    UPDATE source_items
                    SET last_seen_at = now(), updated_at = now()
                    WHERE source_id = $1::uuid AND external_id = $2
                    """,
                    source_id,
                    external_id,
                )
                return False
            title = str(document.get("title") or external_id).strip()[:500]
            source_uri = str(document.get("source_uri") or external_id)
            duplicate = await conn.fetchrow(
                """
                SELECT document_id
                FROM document_dedupe_keys
                WHERE workspace_slug = $1 AND checksum = $2
                """,
                workspace,
                checksum,
            )
            if duplicate and duplicate["document_id"]:
                await conn.execute(
                    """
                    INSERT INTO source_items (source_id, external_id, checksum, document_id, title, source_uri)
                    VALUES ($1::uuid, $2, $3, $4, $5, $6)
                    ON CONFLICT (source_id, external_id) DO UPDATE
                    SET checksum = EXCLUDED.checksum,
                        document_id = EXCLUDED.document_id,
                        title = EXCLUDED.title,
                        source_uri = EXCLUDED.source_uri,
                        last_seen_at = now(),
                        updated_at = now()
                    """,
                    source_id,
                    external_id,
                    checksum,
                    duplicate["document_id"],
                    title,
                    source_uri,
                )
                return False
            source_text = (
                f"Title: {title}\n"
                f"Workspace: {workspace}\n"
                f"Source: {source_uri}\n"
                f"Classification: {classification}\n\n"
                f"{text}"
            )
            row = await conn.fetchrow(
                """
                INSERT INTO documents (title, source_uri, department_slug, classification, checksum, status, queued_at)
                VALUES ($1, $2, $3, $4, $5, 'queued', now())
                RETURNING id
                """,
                title,
                source_uri,
                "c_level" if workspace == "department_c_level" else "public",
                classification,
                checksum,
            )
            document_id = row["id"]
            dedupe_row = await conn.fetchrow(
                """
                INSERT INTO document_dedupe_keys (workspace_slug, checksum, document_id)
                VALUES ($1, $2, $3)
                ON CONFLICT (workspace_slug, checksum) DO NOTHING
                RETURNING document_id
                """,
                workspace,
                checksum,
                document_id,
            )
            if not dedupe_row:
                duplicate = await conn.fetchrow(
                    """
                    SELECT document_id
                    FROM document_dedupe_keys
                    WHERE workspace_slug = $1 AND checksum = $2
                    """,
                    workspace,
                    checksum,
                )
                await conn.execute("DELETE FROM documents WHERE id = $1", document_id)
                await conn.execute(
                    """
                    INSERT INTO source_items (source_id, external_id, checksum, document_id, title, source_uri)
                    VALUES ($1::uuid, $2, $3, $4, $5, $6)
                    ON CONFLICT (source_id, external_id) DO UPDATE
                    SET checksum = EXCLUDED.checksum,
                        document_id = EXCLUDED.document_id,
                        title = EXCLUDED.title,
                        source_uri = EXCLUDED.source_uri,
                        last_seen_at = now(),
                        updated_at = now()
                    """,
                    source_id,
                    external_id,
                    checksum,
                    duplicate["document_id"] if duplicate else None,
                    title,
                    source_uri,
                )
                return False
            workspace_row = await conn.fetchrow("SELECT id FROM rag_workspaces WHERE slug = $1", workspace)
            if workspace_row:
                await conn.execute(
                    """
                    INSERT INTO document_workspace_membership (document_id, workspace_id)
                    VALUES ($1, $2)
                    ON CONFLICT DO NOTHING
                    """,
                    document_id,
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
                document_id,
                workspace,
                title,
                source_text,
            )
            await conn.execute(
                """
                INSERT INTO source_items (source_id, external_id, checksum, document_id, title, source_uri)
                VALUES ($1::uuid, $2, $3, $4, $5, $6)
                ON CONFLICT (source_id, external_id) DO UPDATE
                SET checksum = EXCLUDED.checksum,
                    document_id = EXCLUDED.document_id,
                    title = EXCLUDED.title,
                    source_uri = EXCLUDED.source_uri,
                    last_seen_at = now(),
                    updated_at = now()
                """,
                source_id,
                external_id,
                checksum,
                document_id,
                title,
                source_uri,
            )
    await redis_client.rpush(INGEST_QUEUE_NAME, str(document_id))
    return True


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


async def process_source_scan(pool: asyncpg.Pool, redis_client: redis.Redis, run_id: str) -> None:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT r.id AS run_id, r.source_id, r.status, s.name, s.source_type,
                   s.target, s.workspace_slug, s.classification, s.config,
                   s.interval_minutes, s.enabled
            FROM source_scan_runs r
            JOIN document_sources s ON s.id = r.source_id
            WHERE r.id = $1::uuid
            """,
            run_id,
        )
        if not row or row["status"] == "complete":
            return
        await conn.execute(
            """
            UPDATE source_scan_runs
            SET status = 'running', started_at = now(), error = NULL
            WHERE id = $1::uuid
            """,
            run_id,
        )

    source_id = str(row["source_id"])
    config = source_config_from_value(row["config"])
    config.setdefault("name", row["name"])
    workspace = row["workspace_slug"] or source_workspace(row["target"], row["classification"])
    try:
        documents = await collect_source_documents(row["source_type"], config)
        queued = 0
        for document in documents:
            if await queue_scanned_document(
                pool,
                redis_client,
                source_id=source_id,
                workspace=workspace,
                classification=row["classification"],
                document=document,
            ):
                queued += 1
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE source_scan_runs
                SET status = 'complete',
                    finished_at = now(),
                    items_found = $2,
                    documents_queued = $3,
                    error = NULL
                WHERE id = $1::uuid
                """,
                run_id,
                len(documents),
                queued,
            )
            await conn.execute(
                """
                UPDATE document_sources
                SET last_scan_at = now(),
                    last_status = 'complete',
                    last_error = NULL,
                    next_scan_at = CASE
                      WHEN enabled THEN now() + make_interval(mins => interval_minutes)
                      ELSE NULL
                    END,
                    updated_at = now()
                WHERE id = $1::uuid
                """,
                source_id,
            )
        print(
            f"source scan complete run_id={run_id} source_id={source_id} items={len(documents)} queued={queued}",
            flush=True,
        )
    except Exception as exc:
        error = str(exc)[:4000]
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE source_scan_runs
                SET status = 'failed', finished_at = now(), error = $2
                WHERE id = $1::uuid
                """,
                run_id,
                error,
            )
            await conn.execute(
                """
                UPDATE document_sources
                SET last_scan_at = now(),
                    last_status = 'failed',
                    last_error = $2,
                    next_scan_at = CASE
                      WHEN enabled THEN now() + make_interval(mins => interval_minutes)
                      ELSE NULL
                    END,
                    updated_at = now()
                WHERE id = $1::uuid
                """,
                source_id,
                error,
            )
        print(f"source scan failed run_id={run_id} source_id={source_id}: {error}", flush=True)


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


async def source_scan_worker_loop(worker_id: int, pool: asyncpg.Pool, redis_client: redis.Redis) -> None:
    while True:
        item = await redis_client.blpop(SOURCE_SCAN_QUEUE_NAME, timeout=10)
        if not item:
            continue
        _, run_id = item
        try:
            await process_source_scan(pool, redis_client, str(run_id))
        except Exception as exc:
            print(f"source_worker={worker_id} unhandled error run_id={run_id}: {exc}", flush=True)


async def source_scheduler_loop(pool: asyncpg.Pool, redis_client: redis.Redis) -> None:
    while True:
        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT id
                    FROM document_sources s
                    WHERE enabled = true
                      AND (next_scan_at IS NULL OR next_scan_at <= now())
                      AND NOT EXISTS (
                        SELECT 1
                        FROM source_scan_runs r
                        WHERE r.source_id = s.id
                          AND r.status IN ('queued', 'running')
                      )
                    ORDER BY COALESCE(next_scan_at, created_at)
                    LIMIT 10
                    """
                )
                run_ids: list[str] = []
                for row in rows:
                    run = await conn.fetchrow(
                        """
                        INSERT INTO source_scan_runs (source_id, trigger, status)
                        VALUES ($1, 'scheduled', 'queued')
                        RETURNING id
                        """,
                        row["id"],
                    )
                    await conn.execute(
                        """
                        UPDATE document_sources
                        SET next_scan_at = now() + make_interval(mins => interval_minutes),
                            updated_at = now()
                        WHERE id = $1
                        """,
                        row["id"],
                    )
                    run_ids.append(str(run["id"]))
            if run_ids:
                await redis_client.rpush(SOURCE_SCAN_QUEUE_NAME, *run_ids)
                print(f"scheduled {len(run_ids)} source scans", flush=True)
        except Exception as exc:
            print(f"source scheduler error: {exc}", flush=True)
        await asyncio.sleep(max(10, SOURCE_SCHEDULER_POLL_SECONDS))


async def main() -> None:
    pool = await asyncpg.create_pool(
        DATABASE_URL,
        min_size=1,
        max_size=max(3, INGEST_WORKER_CONCURRENCY + SOURCE_SCAN_WORKER_CONCURRENCY + 1),
    )
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    try:
        await ensure_runtime_schema(pool)
        await bootstrap_queue(pool, redis_client)
        await asyncio.gather(
            *(worker_loop(worker_id, pool, redis_client) for worker_id in range(INGEST_WORKER_CONCURRENCY)),
            *(
                source_scan_worker_loop(worker_id, pool, redis_client)
                for worker_id in range(SOURCE_SCAN_WORKER_CONCURRENCY)
            ),
            source_scheduler_loop(pool, redis_client),
        )
    finally:
        await redis_client.aclose()
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
