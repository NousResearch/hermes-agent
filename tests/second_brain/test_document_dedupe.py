import asyncio
import importlib.util
import json
import sys
import types
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_module(monkeypatch, name: str, path: Path):
    monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost/test")
    monkeypatch.setenv("LIGHTRAG_COMPANY_PUBLIC_URL", "http://company-public")
    monkeypatch.setenv("LIGHTRAG_DEPARTMENT_C_LEVEL_URL", "http://c-level")
    if "asyncpg" not in sys.modules:
        asyncpg_stub = types.ModuleType("asyncpg")
        asyncpg_stub.PostgresError = Exception
        asyncpg_stub.Pool = object
        asyncpg_stub.create_pool = None
        monkeypatch.setitem(sys.modules, "asyncpg", asyncpg_stub)
    if "redis.asyncio" not in sys.modules:
        redis_pkg = types.ModuleType("redis")
        redis_asyncio_stub = types.ModuleType("redis.asyncio")
        redis_asyncio_stub.Redis = object
        redis_asyncio_stub.from_url = None
        redis_pkg.asyncio = redis_asyncio_stub
        monkeypatch.setitem(sys.modules, "redis", redis_pkg)
        monkeypatch.setitem(sys.modules, "redis.asyncio", redis_asyncio_stub)

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeAcquire:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakeTransaction:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakePool:
    def __init__(self, conn):
        self.conn = conn

    def acquire(self):
        return FakeAcquire(self.conn)


class FakeRedis:
    def __init__(self):
        self.pushed = []

    async def rpush(self, *args):
        self.pushed.append(args)


class DuplicateAwareConn:
    existing_id = uuid.UUID("11111111-1111-1111-1111-111111111111")
    new_id = uuid.UUID("22222222-2222-2222-2222-222222222222")
    workspace_id = uuid.UUID("33333333-3333-3333-3333-333333333333")

    def __init__(self):
        self.execute_calls = []
        self.fetchrow_calls = []

    def transaction(self):
        return FakeTransaction()

    async def fetchrow(self, sql, *args):
        self.fetchrow_calls.append((sql, args))
        if "FROM source_items" in sql:
            return None
        if "INSERT INTO document_dedupe_keys" in sql:
            return None
        if "FROM document_dedupe_keys" in sql:
            return {"document_id": self.existing_id}
        if "INSERT INTO documents" in sql:
            return {"id": self.new_id}
        if "SELECT id FROM rag_workspaces" in sql:
            return {"id": self.workspace_id}
        return None

    async def execute(self, sql, *args):
        self.execute_calls.append((sql, args))


class LightragBackfillConn:
    doc_id = uuid.UUID("55555555-5555-5555-5555-555555555555")

    def __init__(self):
        self.execute_calls = []

    async def fetch(self, sql, *args):
        if args == ("company_public",):
            return [{"title": "Company Handbook", "document_ids": [self.doc_id]}]
        return []

    async def execute(self, sql, *args):
        self.execute_calls.append((sql, args))


def test_manual_text_ingest_skips_existing_workspace_checksum(monkeypatch):
    module = load_module(
        monkeypatch,
        "second_brain_knowledge_api_dedupe_manual",
        ROOT / "deploy/second-brain/services/knowledge-api/app.py",
    )
    conn = DuplicateAwareConn()
    redis_client = FakeRedis()
    module.app.state.pool = FakePool(conn)
    module.app.state.redis = redis_client

    result = asyncio.run(
        module.ingest_text(
            {
                "title": "Handbook Copy",
                "text": "Same company handbook text",
                "target": "public",
                "classification": "internal",
            }
        )
    )

    assert result["status"] == "duplicate"
    assert result["document_id"] == str(conn.existing_id)
    assert result["workspace"] == "company_public"
    assert redis_client.pushed == []
    assert not any("INSERT INTO documents" in sql for sql, _ in conn.fetchrow_calls)


def test_source_scan_skips_duplicate_content_from_other_source(monkeypatch):
    module = load_module(
        monkeypatch,
        "second_brain_worker_dedupe_source",
        ROOT / "deploy/second-brain/services/knowledge-api/worker.py",
    )
    conn = DuplicateAwareConn()
    redis_client = FakeRedis()

    queued = asyncio.run(
        module.queue_scanned_document(
            FakePool(conn),
            redis_client,
            source_id=str(uuid.UUID("44444444-4444-4444-4444-444444444444")),
            workspace="company_public",
            classification="internal",
            document={
                "external_id": "drive-doc-1",
                "title": "Handbook Copy",
                "source_uri": "https://example.test/handbook",
                "text": "Same company handbook text",
            },
        )
    )

    assert queued is False
    assert redis_client.pushed == []
    assert not any("INSERT INTO documents" in sql for sql, _ in conn.fetchrow_calls)
    assert any("INSERT INTO source_items" in sql for sql, _ in conn.execute_calls)


def test_backfill_document_checksum_from_lightrag_full_docs(monkeypatch, tmp_path):
    module = load_module(
        monkeypatch,
        "second_brain_knowledge_api_dedupe_lightrag_backfill",
        ROOT / "deploy/second-brain/services/knowledge-api/app.py",
    )
    monkeypatch.setattr(module, "LIGHTRAG_ROOT", tmp_path)
    storage = tmp_path / "workspaces/company_public/rag_storage/company_public"
    storage.mkdir(parents=True)
    (storage / "kv_store_full_docs.json").write_text(
        json.dumps(
            {
                "doc-1": {
                    "content": (
                        "Title: Company Handbook\n"
                        "Workspace: company_public\n"
                        "Classification: internal\n\n"
                        "Same company handbook text"
                    )
                }
            }
        )
    )
    conn = LightragBackfillConn()

    asyncio.run(module.backfill_document_checksums_from_lightrag(conn))

    assert any(
        "UPDATE documents SET checksum" in sql
        and args[0] == module.checksum_text("Same company handbook text")
        and args[1] == conn.doc_id
        for sql, args in conn.execute_calls
    )
