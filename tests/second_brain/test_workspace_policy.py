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


def test_knowledge_api_routes_only_public_and_c_level(monkeypatch):
    module = load_module(
        monkeypatch,
        "second_brain_knowledge_api",
        ROOT / "deploy/second-brain/services/knowledge-api/app.py",
    )

    assert module.ENABLED_WORKSPACES == ["company_public", "department_c_level"]
    assert module.route_workspace({"title": "All hands", "text": "hello"}) == "company_public"
    assert (
        module.route_workspace(
            {
                "title": "Legacy department upload",
                "department": "marketing",
                "visibility": "department",
                "classification": "internal",
            }
        )
        == "company_public"
    )
    assert module.route_workspace({"target": "c_level", "title": "Board"}) == "department_c_level"
    assert module.route_workspace({"classification": "restricted", "title": "Board"}) == "department_c_level"


def test_knowledge_api_allows_c_level_only_for_admin(monkeypatch):
    module = load_module(
        monkeypatch,
        "second_brain_knowledge_api_auth",
        ROOT / "deploy/second-brain/services/knowledge-api/app.py",
    )

    assert module.allowed_workspaces([]) == ["company_public"]
    assert module.allowed_workspaces(["company_all"]) == ["company_public"]
    assert module.allowed_workspaces(["department_marketing", "department_hr"]) == ["company_public"]
    assert module.allowed_workspaces(["department_c_level"]) == ["company_public"]
    assert module.allowed_workspaces(["role_admin"]) == ["company_public", "department_c_level"]


def test_gateway_normalizes_two_workspace_roles(monkeypatch):
    module = load_module(
        monkeypatch,
        "second_brain_gateway",
        ROOT / "deploy/second-brain/services/company-ai-gateway/app.py",
    )

    assert module.normalize_groups(["company_all", "department_marketing", "role_admin"]) == [
        "company_all",
        "role_admin",
    ]
    assert module.query_groups_for_auth({"role": "member", "groups": ["department_marketing"]}) == ["company_all"]
    assert module.query_groups_for_auth({"role": "admin", "groups": ["role_admin"]}) == ["role_admin"]
    assert module.visible_workspace_slugs({"role": "member", "groups": ["department_c_level"]}) == {
        "company_public"
    }
    assert module.visible_workspace_slugs({"role": "admin", "groups": ["role_admin"]}) == {
        "company_public",
        "department_c_level",
    }
