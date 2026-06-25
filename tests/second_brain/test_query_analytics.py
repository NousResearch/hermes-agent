import asyncio
import importlib.util
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
        asyncpg_stub.create_pool = None
        monkeypatch.setitem(sys.modules, "asyncpg", asyncpg_stub)
    if "redis.asyncio" not in sys.modules:
        redis_pkg = types.ModuleType("redis")
        redis_asyncio_stub = types.ModuleType("redis.asyncio")
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


class FakePool:
    def __init__(self, conn):
        self.conn = conn

    def acquire(self):
        return FakeAcquire(self.conn)


class FakeConn:
    def __init__(self):
        self.event_id = uuid.UUID("11111111-1111-1111-1111-111111111111")
        self.fetchval_calls = []
        self.execute_calls = []
        self.executemany_calls = []

    async def fetchval(self, sql, *args):
        self.fetchval_calls.append((sql, args))
        return self.event_id

    async def execute(self, sql, *args):
        self.execute_calls.append((sql, args))

    async def executemany(self, sql, rows):
        self.executemany_calls.append((sql, rows))


def test_extract_document_hits_deduplicates_references(monkeypatch):
    module = load_module(
        monkeypatch,
        "second_brain_knowledge_api_analytics_extract",
        ROOT / "deploy/second-brain/services/knowledge-api/app.py",
    )

    answers = [
        {
            "workspace": "company_public",
            "result": {
                "references": [
                    {"reference_id": "doc-1", "file_path": "Company Handbook.md"},
                    {"reference_id": "doc-1", "file_path": "Company Handbook.md"},
                    {"reference_id": "doc-2", "file_path": "Onboarding.md"},
                ]
            },
        },
        {"workspace": "department_c_level", "error": "timeout"},
    ]

    hits = module.extract_document_hits(answers)

    assert hits == [
        {
            "workspace_slug": "company_public",
            "rank": 1,
            "reference_id": "doc-1",
            "title": "Company Handbook.md",
            "source_uri": "Company Handbook.md",
            "document_id": None,
        },
        {
            "workspace_slug": "company_public",
            "rank": 2,
            "reference_id": "doc-2",
            "title": "Onboarding.md",
            "source_uri": "Onboarding.md",
            "document_id": None,
        },
    ]


def test_record_query_analytics_writes_event_workspaces_and_hits(monkeypatch):
    module = load_module(
        monkeypatch,
        "second_brain_knowledge_api_analytics_record",
        ROOT / "deploy/second-brain/services/knowledge-api/app.py",
    )
    conn = FakeConn()
    pool = FakePool(conn)
    answers = [
        {"workspace": "company_public", "result": {"references": [{"reference_id": "r1", "file_path": "Guide.md"}]}},
        {"workspace": "department_c_level", "error": "not configured"},
    ]

    asyncio.run(
        module.record_query_analytics(
            pool,
            actor_email="user@example.com",
            actor_role="member",
            actor_groups=["company_all"],
            query_text="What changed in the handbook?",
            mode="mix",
            allowed_workspaces=["company_public", "department_c_level"],
            answers=answers,
            latency_ms=321,
            status="ok",
            error=None,
        )
    )

    assert "INSERT INTO query_events" in conn.fetchval_calls[0][0]
    assert conn.fetchval_calls[0][1][0] == "user@example.com"
    assert conn.fetchval_calls[0][1][3] == "What changed in the handbook?"
    assert [call[1][1] for call in conn.execute_calls] == ["company_public", "department_c_level"]
    assert conn.execute_calls[0][1][4] == 1
    assert conn.execute_calls[1][1][3] == "error"
    assert conn.executemany_calls[0][1][0][2] == "r1"
    assert conn.executemany_calls[0][1][0][3] == "Guide.md"


def test_build_analytics_response_serializes_admin_metrics(monkeypatch):
    module = load_module(
        monkeypatch,
        "second_brain_knowledge_api_analytics_response",
        ROOT / "deploy/second-brain/services/knowledge-api/app.py",
    )

    payload = module.build_analytics_response(
        period_days=30,
        limit=10,
        summary={
            "total_queries": 3,
            "unique_users": 2,
            "successful_queries": 2,
            "failed_queries": 1,
            "avg_latency_ms": 125.4,
            "document_hits": 5,
        },
        top_documents=[
            {
                "title": "Company Handbook.md",
                "source_uri": "Company Handbook.md",
                "workspace_slug": "company_public",
                "hits": 4,
                "unique_users": 2,
                "last_accessed_at": "2026-06-25T05:30:00+00:00",
            }
        ],
        top_users=[{"actor_email": "user@example.com", "query_count": 2, "last_query_at": "2026-06-25"}],
        recent_queries=[
            {
                "created_at": "2026-06-25T05:31:00+00:00",
                "actor_email": "user@example.com",
                "actor_role": "member",
                "query_text": "What changed?",
                "status": "ok",
                "latency_ms": 100,
                "allowed_workspaces": ["company_public"],
                "document_count": 1,
            }
        ],
        workspace_usage=[{"workspace_slug": "company_public", "query_count": 3, "error_count": 1, "avg_latency_ms": 88.8}],
        top_questions=[{"query_text": "What changed?", "count": 2, "last_asked_at": "2026-06-25"}],
    )

    assert payload["summary"]["success_rate"] == 2 / 3
    assert payload["summary"]["document_hits"] == 5
    assert payload["top_documents"][0]["title"] == "Company Handbook.md"
    assert payload["recent_queries"][0]["query_text"] == "What changed?"
    assert payload["workspace_usage"][0]["error_count"] == 1
