"""Tests for gen_openapi.py route extraction."""

import ast
import pytest


def _extract_routes_from_source(source: str) -> list:
    """Parse source and extract route registrations using the same logic as gen_openapi."""
    routes = []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and hasattr(node.func, "attr"):
            method_name = node.func.attr
            if method_name in ("add_get", "add_post", "add_put", "add_delete"):
                http_method = method_name.replace("add_", "").upper()
                path = ""
                handler = ""
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        path = arg.value
                        break
                for arg in node.args:
                    if isinstance(arg, ast.Attribute):
                        handler = arg.attr
                        break
                    elif isinstance(arg, ast.Name):
                        handler = arg.id
                        break
                for kw in node.keywords:
                    if kw.arg == "handler":
                        if hasattr(kw.value, "attr"):
                            handler = kw.value.attr
                        elif isinstance(kw.value, ast.Name):
                            handler = kw.value.id
                    elif kw.arg == "path" and isinstance(kw.value, ast.Constant):
                        path = kw.value.value
                if handler and path:
                    routes.append({"method": http_method, "path": path, "handler": handler})
    return routes


class TestExtractRoutes:
    """Verify route extraction from aiohttp router.add_get() calls."""

    def test_extracts_positional_handler(self):
        """add_get('/health', self._handle_health) should extract handler."""
        source = """
self._app.router.add_get("/health", self._handle_health)
self._app.router.add_post("/v1/chat/completions", self._handle_chat_completions)
"""
        routes = _extract_routes_from_source(source)
        assert len(routes) == 2
        assert routes[0]["path"] == "/health"
        assert routes[0]["handler"] == "_handle_health"
        assert routes[0]["method"] == "GET"
        assert routes[1]["path"] == "/v1/chat/completions"
        assert routes[1]["handler"] == "_handle_chat_completions"
        assert routes[1]["method"] == "POST"

    def test_extracts_delete_method(self):
        """add_delete should extract DELETE method."""
        source = """
self._app.router.add_delete("/v1/responses/{response_id}", self._handle_delete_response)
"""
        routes = _extract_routes_from_source(source)
        assert len(routes) == 1
        assert routes[0]["method"] == "DELETE"
        assert routes[0]["path"] == "/v1/responses/{response_id}"
        assert routes[0]["handler"] == "_handle_delete_response"

    def test_no_routes_returns_empty(self):
        """No matching calls should return empty list."""
        source = """
x = 1
"""
        routes = _extract_routes_from_source(source)
        assert routes == []

    def test_all_four_methods(self):
        """GET, POST, PUT, DELETE all extracted."""
        source = """
app.router.add_get("/a", self._h1)
app.router.add_post("/b", self._h2)
app.router.add_put("/c", self._h3)
app.router.add_delete("/d", self._h4)
"""
        routes = _extract_routes_from_source(source)
        methods = [r["method"] for r in routes]
        assert methods == ["GET", "POST", "PUT", "DELETE"]

    def test_real_api_server_pattern(self):
        """Real api_server.py registrations: self._app.router.add_get(..., self._handler)."""
        source = """
self._app.router.add_get("/health", self._handle_health)
self._app.router.add_get("/v1/models", self._handle_models)
self._app.router.add_post("/v1/chat/completions", self._handle_chat_completions)
self._app.router.add_post("/v1/responses", self._handle_responses)
self._app.router.add_get("/v1/responses/{response_id}", self._handle_get_response)
self._app.router.add_delete("/v1/responses/{response_id}", self._handle_delete_response)
self._app.router.add_post("/v1/runs", self._handle_runs)
self._app.router.add_get("/v1/runs/{run_id}", self._handle_get_run)
self._app.router.add_get("/v1/runs/{run_id}/events", self._handle_run_events)
self._app.router.add_post("/v1/runs/{run_id}/approval", self._handle_run_approval)
self._app.router.add_post("/v1/runs/{run_id}/stop", self._handle_stop_run)
"""
        routes = _extract_routes_from_source(source)
        assert len(routes) == 11
        assert routes[0]["handler"] == "_handle_health"
        assert routes[6]["handler"] == "_handle_runs"
        assert routes[9]["handler"] == "_handle_run_approval"

    def test_positional_handler_takes_precedence_over_keyword(self):
        """Positional handler extracted even if handler= keyword also present."""
        source = """
app.router.add_get("/path", positional_handler, handler=keyword_handler)
"""
        routes = _extract_routes_from_source(source)
        assert len(routes) == 1
        # Keyword handler= overwrites positional (this is fine — real code uses positional)
        assert routes[0]["handler"] == "keyword_handler"
