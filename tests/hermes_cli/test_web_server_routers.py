"""adopt_router: mount a plain APIRouter's own routes instead of FastAPI's per-route copy."""

import ast
import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi import Depends, FastAPI
from fastapi.routing import APIRouter
from fastapi.testclient import TestClient

from hermes_cli import web_server_routers
from hermes_cli.web_server_routers import adopt_router

REPO = Path(__file__).resolve().parents[2]


def _router(**kwargs) -> APIRouter:
    router = APIRouter(**kwargs)

    @router.get("/ping")
    def ping(n: int = 1):
        return {"pong": n}

    return router


def _ids(routes) -> set[int]:
    return {id(r) for r in routes}


def test_plain_router_routes_are_shared_and_served():
    app, router = FastAPI(), _router()
    adopt_router(app, router)
    assert _ids(router.routes) <= _ids(app.router.routes)
    assert TestClient(app).get("/ping", params={"n": 3}).json() == {"pong": 3}


def test_router_with_its_own_settings_is_included_as_before():
    """Prefix, router-level dependencies and an app with global dependencies all change the built
    routes, so those still go through include_router (fresh copies)."""
    seen = []
    cases = [
        (FastAPI(), _router(prefix="/v1"), "/v1/ping"),
        (FastAPI(), _router(dependencies=[Depends(lambda: seen.append("router"))]), "/ping"),
        (FastAPI(dependencies=[Depends(lambda: seen.append("app"))]), _router(), "/ping"),
    ]
    for app, router, path in cases:
        adopt_router(app, router)
        assert not _ids(router.routes) & _ids(app.router.routes)
        assert TestClient(app).get(path).json() == {"pong": 1}
    assert seen == ["router", "app"]


def test_app_router_that_merges_into_routes_includes_the_router():
    """add_api_route merges the app router's own prefix, tags, responses and callbacks into every
    route include_router builds, so an app router carrying any of them falls back to include_router."""
    def callback():
        pass

    callbacks = APIRouter()
    callbacks.add_api_route("/cb", callback, methods=["POST"])
    apps = [FastAPI(responses={418: {"description": "teapot"}}), FastAPI(callbacks=callbacks.routes)]
    tagged, prefixed = FastAPI(), FastAPI()
    tagged.router.tags = ["app"]
    prefixed.router.prefix = "/api"
    for app in (*apps, tagged, prefixed):
        router = _router()
        adopt_router(app, router)
        assert not _ids(router.routes) & _ids(app.router.routes)
    assert "418" in apps[0].openapi()["paths"]["/ping"]["get"]["responses"]
    assert "callbacks" in apps[1].openapi()["paths"]["/ping"]["get"]
    assert TestClient(prefixed).get("/api/ping").json() == {"pong": 1}


@pytest.mark.skipif("strict_content_type" not in inspect.signature(FastAPI).parameters,
                    reason="FastAPI without strict_content_type (before 0.130)")
def test_app_with_lax_content_type_includes_the_router():
    """include_router hands the app's strict_content_type to routes that left it at the default; an
    adopted route keeps the default (strict), so a lax app falls back to include_router and its
    routes accept a JSON body sent without a Content-Type header."""
    app, router = FastAPI(strict_content_type=False), APIRouter()

    @router.post("/echo")
    def echo(body: dict):
        return body

    adopt_router(app, router)
    assert not _ids(router.routes) & _ids(app.router.routes)
    reply = TestClient(app).post("/echo", content=b'{"a": 1}')
    assert reply.json() == {"a": 1}


def test_router_with_its_own_lifespan_is_included(monkeypatch):
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(_app):
        yield

    app, router = FastAPI(), _router(lifespan=lifespan)
    adopt_router(app, router)
    assert not _ids(router.routes) & _ids(app.router.routes)

    # A FastAPI whose default lifespan cannot be told apart from a custom one adopts nothing.
    monkeypatch.setattr(web_server_routers, "_DEFAULT_LIFESPAN", None)
    app, router = FastAPI(), _router()
    adopt_router(app, router)
    assert not _ids(router.routes) & _ids(app.router.routes)


def test_module_uses_only_public_fastapi_and_starlette_names():
    """Private names (e.g. fastapi.routing._DefaultLifespan, new in FastAPI 0.130) break the whole
    dashboard import on older FastAPI inside the supported range (fastapi>=0.104)."""
    tree = ast.parse(Path(web_server_routers.__file__).read_text(encoding="utf-8-sig"))
    private = [(node.module, alias.name) for node in ast.walk(tree)
               if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0]
               in {"fastapi", "starlette"} for alias in node.names if alias.name.startswith("_")]
    assert private == []


def test_dashboard_app_holds_one_copy_of_each_module_route():
    from hermes_cli import web_server
    from hermes_cli.web_routers import sessions, status

    for router in (status.router, status.logs_router, sessions.list_router):
        assert router.routes and _ids(router.routes) <= _ids(web_server.app.router.routes)


# ── Contract: adopted dashboard == include_router dashboard ─────────────────────────────────────


def _dashboard_snapshot(include_everything: bool) -> dict:
    """Run in a fresh interpreter: build the dashboard app (adopting routers, or including every one
    as FastAPI would) and describe each route field by field plus the OpenAPI schema."""
    import enum

    import hermes_cli.web_server_routers as mounting
    from fastapi.datastructures import DefaultPlaceholder

    if include_everything:
        mounting.adopt_router = lambda app, router: app.include_router(router)
    from hermes_cli import web_server

    def norm(value):
        if isinstance(value, DefaultPlaceholder):
            return {"default": norm(value.value)}
        if isinstance(value, enum.Enum):
            return f"{type(value).__qualname__}.{value.name}"
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, dict):
            return {str(k): norm(v) for k, v in value.items()}
        if isinstance(value, (set, frozenset)):
            return sorted(norm(v) for v in value)
        if isinstance(value, (list, tuple)):
            return [norm(v) for v in value]
        if hasattr(value, "dependency"):  # params.Depends / Security
            return {"depends": norm(value.dependency), "use_cache": value.use_cache,
                    "scopes": norm(getattr(value, "scopes", None))}
        if hasattr(value, "__qualname__"):
            return f"{getattr(value, '__module__', '')}.{value.__qualname__}"
        return f"<{type(value).__module__}.{type(value).__qualname__} instance>"

    def dependant(dep):
        if dep is None:
            return None
        params = {kind: sorted(p.name for p in getattr(dep, f"{kind}_params"))
                  for kind in ("path", "query", "header", "cookie", "body")}
        return {"call": norm(dep.call), **params,
                "dependencies": [dependant(d) for d in dep.dependencies]}

    # dependency_overrides_provider is the documented difference (see web_server_routers).
    fields = ("path", "path_format", "name", "methods", "endpoint", "dependencies", "tags",
              "summary", "description", "response_description", "responses", "deprecated",
              "operation_id", "unique_id", "include_in_schema", "status_code", "response_model",
              "response_class", "callbacks", "openapi_extra", "strict_content_type",
              "response_model_include", "response_model_exclude", "response_model_by_alias",
              "response_model_exclude_unset", "response_model_exclude_defaults",
              "response_model_exclude_none", "generate_unique_id_function")
    routes = []
    for route in web_server.app.router.routes:
        row = {"type": norm(type(route))}
        row.update({f: norm(getattr(route, f)) for f in fields if hasattr(route, f)})
        # Compare what the request handler applies: include_router stores the app's plain True, an
        # adopted route keeps the router default DefaultPlaceholder(True).
        strict = row.get("strict_content_type")
        if isinstance(strict, dict):
            row["strict_content_type"] = strict["default"]
        row["dependant"] = dependant(getattr(route, "dependant", None))
        row["response_field"] = norm(getattr(getattr(route, "response_field", None), "type_", None))
        row["attrs"] = sorted(vars(route))
        routes.append(row)
    return {"routes": routes, "openapi": web_server.app.openapi()}


def _snapshot_in_subprocess(include_everything: bool) -> dict:
    source = inspect.getsource(_dashboard_snapshot)
    code = f"import json\n{source}\nprint(json.dumps(_dashboard_snapshot({include_everything}), sort_keys=True))"
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, encoding="utf-8",
                         timeout=180, check=False)
    assert out.returncode == 0, out.stderr[-4000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


# Every attribute a dashboard route may carry. The compared fields above plus state FastAPI and
# Starlette derive from them (dependency graph, body/response fields, path regex, ASGI app) and the
# documented dependency_overrides_provider difference. A FastAPI upgrade that adds a route attribute
# fails test_adopted_dashboard_matches_include_router_dashboard until someone decides whether
# include_router merges it (then compare it above) or not (then list it here).
_KNOWN_ROUTE_ATTRS = frozenset({
    "path", "path_format", "name", "methods", "endpoint", "dependencies", "tags", "summary",
    "description", "response_description", "responses", "deprecated", "operation_id", "unique_id",
    "include_in_schema", "status_code", "response_model", "response_class", "callbacks",
    "openapi_extra", "strict_content_type", "response_model_include", "response_model_exclude",
    "response_model_by_alias", "response_model_exclude_unset", "response_model_exclude_defaults",
    "response_model_exclude_none", "generate_unique_id_function",
    "app", "path_regex", "param_convertors", "dependant", "_flat_dependant", "_embed_body_fields",
    "body_field", "response_field", "response_fields", "dependency_overrides_provider",
    "_base_app",  # Starlette Mount (the static /assets mount)
    "secure_cloned_response_field",  # FastAPI < 0.130: derived from response_model
    # FastAPI >= 0.135: derived from the endpoint's own return annotation, never merged
    "is_json_stream", "is_sse_stream", "stream_item_field", "stream_item_type",
})


def test_adopted_dashboard_matches_include_router_dashboard():
    """Every adopted route must equal what include_router would have built, field by field and in
    the same order, and the OpenAPI schema must be identical; a FastAPI upgrade that makes
    include_router merge something new fails here instead of silently changing the dashboard."""
    adopted, included = _snapshot_in_subprocess(False), _snapshot_in_subprocess(True)
    assert len(adopted["routes"]) == len(included["routes"]) > 300
    for got, want in zip(adopted["routes"], included["routes"], strict=True):
        assert got == want, (got.get("path"), got.get("name"))
        unknown = set(got["attrs"]) - _KNOWN_ROUTE_ATTRS
        assert not unknown, (got.get("path"), sorted(unknown))
    assert adopted["openapi"] == included["openapi"]
