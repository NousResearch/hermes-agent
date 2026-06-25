import importlib.util
import sys
import types
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


def test_source_payload_normalizes_schedule_and_redacts_secret(monkeypatch):
    module = load_module(
        monkeypatch,
        "second_brain_knowledge_api_sources",
        ROOT / "deploy/second-brain/services/knowledge-api/app.py",
    )

    payload = module.normalize_source_payload(
        {
            "name": "Ops Notion",
            "source_type": "notion",
            "target": "c_level",
            "classification": "restricted",
            "interval_minutes": 2,
            "config": {
                "notion_api_key": "secret_notion_key",
                "notion_page_url": "https://www.notion.so/acme/Ops-Runbook-1234567890abcdef1234567890abcdef",
            },
        }
    )

    assert payload["source_type"] == "notion"
    assert payload["workspace"] == "department_c_level"
    assert payload["interval_minutes"] == module.MIN_SOURCE_INTERVAL_MINUTES
    assert payload["config"]["notion_api_key"] == "secret_notion_key"
    assert module.redact_source_config(payload["config"])["notion_api_key"] == "set"


def test_drive_source_requires_public_link(monkeypatch):
    module = load_module(
        monkeypatch,
        "second_brain_knowledge_api_drive_source",
        ROOT / "deploy/second-brain/services/knowledge-api/app.py",
    )

    payload = module.normalize_source_payload(
        {
            "name": "Drive Docs",
            "source_type": "drive_public",
            "config": {"drive_url": "https://docs.google.com/document/d/1abcDEFghiJKLmnopQrsTuvWxYz1234567890/edit"},
        }
    )

    assert payload["target"] == "public"
    assert payload["workspace"] == "company_public"
    assert payload["config"]["drive_url"].startswith("https://docs.google.com/document/")


def test_worker_parses_notion_and_drive_document_links(monkeypatch):
    module = load_module(
        monkeypatch,
        "second_brain_worker_sources",
        ROOT / "deploy/second-brain/services/knowledge-api/worker.py",
    )

    assert (
        module.extract_notion_id("https://www.notion.so/acme/Ops-Runbook-1234567890abcdef1234567890abcdef?pvs=4")
        == "12345678-90ab-cdef-1234-567890abcdef"
    )

    parsed = module.parse_drive_public_link(
        "https://docs.google.com/document/d/1abcDEFghiJKLmnopQrsTuvWxYz1234567890/edit"
    )
    assert parsed == {
        "kind": "google_doc",
        "file_id": "1abcDEFghiJKLmnopQrsTuvWxYz1234567890",
    }
    assert module.drive_public_download_url(parsed) == (
        "https://docs.google.com/document/d/1abcDEFghiJKLmnopQrsTuvWxYz1234567890/export?format=txt"
    )

    folder = module.parse_drive_public_link("https://drive.google.com/drive/folders/1FolderId")
    assert folder["kind"] == "drive_folder"
