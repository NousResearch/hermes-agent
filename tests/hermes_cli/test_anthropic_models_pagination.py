"""Anthropic /v1/models cursor pagination — regression tests.

The endpoint defaults to a 20-item page and signals continuation via
``has_more``/``last_id``/``after_id``. An unpaginated read silently drops
every model past the first page (bug class ported from
OpenHands/OpenHands#16758). Covers ``_fetch_anthropic_models`` and the
anthropic provider-plugin ``fetch_models``.
"""

from __future__ import annotations

import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from hermes_cli.models import _fetch_anthropic_models

MODELS = [f"claude-fake-{i:03d}" for i in range(55)]


class _PagedHandler(BaseHTTPRequestHandler):
    page_cap = 20  # server-side max page size (forces pagination even at limit=1000)
    repeat_cursor = False  # simulate a buggy server that never advances

    def log_message(self, *args):  # noqa: D102
        pass

    def do_GET(self):  # noqa: N802
        u = urlparse(self.path)
        if not u.path.endswith("/models"):
            self.send_response(404)
            self.end_headers()
            return
        q = parse_qs(u.query)
        limit = min(int(q.get("limit", ["20"])[0]), self.page_cap)
        after = q.get("after_id", [None])[0]
        start = MODELS.index(after) + 1 if after in MODELS else 0
        page = MODELS[start : start + limit]
        last_id = page[-1] if page else None
        if self.repeat_cursor and after is not None:
            last_id = after  # cursor never advances
        body = {
            "data": [{"id": m, "type": "model"} for m in page],
            "has_more": True if self.repeat_cursor else start + limit < len(MODELS),
            "first_id": page[0] if page else None,
            "last_id": last_id,
        }
        raw = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture()
def paged_server():
    handler = type("Handler", (_PagedHandler,), {})
    srv = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}", handler
    finally:
        srv.shutdown()


def _load_plugin_profile():
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "anthropic_provider_under_test",
        root / "plugins" / "model-providers" / "anthropic" / "__init__.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.anthropic


class TestFetchAnthropicModelsPagination:
    def test_follows_cursor_across_all_pages(self, paged_server):
        base, _handler = paged_server
        got = _fetch_anthropic_models(base_url=base, api_key="sk-ant-api-test")
        assert got is not None
        assert len(got) == len(MODELS)
        assert set(got) == set(MODELS)

    def test_single_page_catalog_still_works(self, paged_server):
        base, handler = paged_server
        handler.page_cap = 1000  # whole catalog fits one page
        got = _fetch_anthropic_models(base_url=base, api_key="sk-ant-api-test")
        assert got is not None and len(got) == len(MODELS)

    def test_repeated_cursor_terminates(self, paged_server):
        base, handler = paged_server
        handler.repeat_cursor = True
        got = _fetch_anthropic_models(base_url=base, api_key="sk-ant-api-test")
        # Must not hang or loop forever; returns the de-duped pages it saw.
        assert got is not None
        assert 0 < len(got) <= len(MODELS)


class TestPluginFetchModelsPagination:
    def test_follows_cursor_across_all_pages(self, paged_server):
        base, _handler = paged_server
        profile = _load_plugin_profile()
        got = profile.fetch_models(api_key="sk-ant-api-test", base_url=base)
        assert got is not None
        assert len(got) == len(MODELS)

    def test_no_api_key_returns_none(self):
        profile = _load_plugin_profile()
        assert profile.fetch_models(api_key=None) is None
