import asyncio
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from threading import Thread

import pytest

from agent import secret_scope
from hermes_constants import set_hermes_home_override, reset_hermes_home_override
from plugins.web.firecrawl import provider as fc


@pytest.fixture
def api():
    pytest.importorskip("firecrawl")
    class RequestLog(list):
        on_request = None

    calls = RequestLog()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            key = self.headers.get("Authorization", "")
            calls.append((key, self.path, body))
            if calls.on_request is not None:
                calls.on_request()
            status = 402 if key.endswith("-empty") else 200
            data = {"success": False, "error": "Payment Required: Insufficient credits " + key}
            if status == 200:
                data = {"success": True, "data": (
                    {"web": [{"url": "https://example.com", "title": key}]}
                    if self.path == "/v2/search" else
                    {"markdown": "HTTP fixture content", "metadata": {"sourceURL": "https://example.com", "title": key}}
                )}
            payload = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", calls
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


@contextmanager
def profile(home):
    ht = set_hermes_home_override(home)
    st = secret_scope.set_secret_scope(secret_scope.build_profile_secret_scope(home))
    try:
        yield
    finally:
        secret_scope.reset_secret_scope(st)
        reset_hermes_home_override(ht)


@pytest.mark.parametrize("cache_enabled", [False, True])
@pytest.mark.parametrize("shared_keys", [False, True])
def test_sdk_dispatch_profile_a_b_a_and_config_freshness(api, tmp_path, monkeypatch, cache_enabled, shared_keys):
    from tools import web_tools
    url, calls = api
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("FIRECRAWL_API_KEYS", '["launch-profile-must-not-leak"]')
    homes = [tmp_path / "a", tmp_path / "b"]
    def prefix(home):
        return "shared" if shared_keys else home.name
    for home in homes:
        home.mkdir()
        (home / "config.yaml").write_text(f"web:\n  cache_enabled: {str(cache_enabled).lower()}\n")
        (home / ".env").write_text(f'FIRECRAWL_API_KEYS=\'["{prefix(home)}-empty", "{prefix(home)}-ok"]\'\nFIRECRAWL_API_URL={url}\n')
    old = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    try:
        for home in [homes[0], homes[1], homes[0]]:
            with profile(home):
                assert web_tools._get_backend() == "firecrawl"
                assert fc.check_firecrawl_api_key()
                assert not fc._use_keyless_ring()
                result = json.loads(web_tools.web_search_tool("profile isolation"))
                assert result["success"], result
                assert result["data"]["web"][0]["title"] == f"Bearer {prefix(home)}-ok"
                result = json.loads(asyncio.run(web_tools.web_extract_tool(["https://example.com"])))
                assert result["results"][0]["content"] == "HTTP fixture content"
        a, b = map(prefix, homes)
        expected = [f"Bearer {a}-empty", f"Bearer {a}-ok", f"Bearer {a}-ok", f"Bearer {b}-empty", f"Bearer {b}-ok", f"Bearer {b}-ok"]
        if not cache_enabled:
            expected += [f"Bearer {a}-ok", f"Bearer {a}-ok"]
        assert [c[0] for c in calls] == expected
        (homes[0] / ".env").write_text(f'FIRECRAWL_API_KEYS=\'["replacement-ok", "a-empty"]\'\nFIRECRAWL_API_URL={url}\n')
        with profile(homes[0]):
            result = json.loads(web_tools.web_search_tool("profile isolation"))
            assert result["data"]["web"][0]["title"] == "Bearer replacement-ok"
            result = json.loads(asyncio.run(web_tools.web_extract_tool(["https://example.com"])))
            assert result["results"][0]["content"] == "HTTP fixture content"
        assert calls[-1][0] == "Bearer replacement-ok"
        assert calls[-1][1] == "/v2/scrape"
        empty = tmp_path / "empty"
        empty.mkdir()
        with profile(empty):
            assert not web_tools._has_env("FIRECRAWL_API_KEYS")
            assert fc._get_direct_firecrawl_config() is None
    finally:
        secret_scope.set_multiplex_active(old)


@pytest.mark.parametrize("operation", ["search", "scrape"])
@pytest.mark.parametrize("edit_at", ["request", "lookup"])
def test_cache_identity_survives_inflight_key_edit(api, tmp_path, monkeypatch, operation, edit_at):
    from tools import web_tools, web_result_cache
    endpoint, calls = api
    (tmp_path / "config.yaml").write_text("web:\n  backend: firecrawl\n  cache_enabled: true\n")

    def set_key(key):
        (tmp_path / ".env").write_text(
            f'FIRECRAWL_API_KEYS=\'{json.dumps([key])}\'\nFIRECRAWL_API_URL={endpoint}\n')

    def invoke():
        if operation == "search":
            result = json.loads(web_tools.web_search_tool("inflight identity"))
            return result["data"]["web"][0]["title"]
        result = json.loads(asyncio.run(web_tools.web_extract_tool(["https://example.com"])))
        return result["results"][0]["title"]

    set_key("old-account")
    if edit_at == "request":
        calls.on_request = lambda: set_key("new-account")
    else:
        target, name = ((web_result_cache.search_memo, "lookup") if operation == "search"
                        else (web_result_cache, "extract_cache_get"))
        original = getattr(target, name)

        def edit_before_lookup(*args, **kwargs):
            set_key("new-account")
            return original(*args, **kwargs)

        monkeypatch.setattr(target, name, edit_before_lookup)

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert invoke() == ("Bearer old-account" if edit_at == "request" else "Bearer new-account")
    calls.on_request = None
    if edit_at == "lookup":
        monkeypatch.setattr(target, name, original)
    assert invoke() == "Bearer new-account"
    count = len(calls)
    assert invoke() == "Bearer new-account"
    assert len(calls) == count
    set_key("old-account")
    assert invoke() == "Bearer old-account"
    assert {c[0] for c in calls} == {"Bearer old-account", "Bearer new-account"}


def test_all_keys_tried_once_then_no_network_and_no_rescue(api, tmp_path, monkeypatch, caplog):
    from tools import web_tools
    url, calls = api
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("web:\n  backend: firecrawl\n  cache_enabled: false\n")
    keys = [f"key-{i}-empty" for i in range(40)]
    monkeypatch.setenv("FIRECRAWL_API_KEYS", json.dumps(keys + keys))
    monkeypatch.setenv("FIRECRAWL_API_URL", url)
    for _ in range(2):
        result = json.loads(web_tools.web_search_tool("exhausted"))
        assert result["success"] is False
        assert "exhausted" in result["error"].lower()
    results = json.loads(asyncio.run(web_tools.web_extract_tool(["https://example.com"])))
    assert "exhausted" in results["results"][0]["error"].lower()
    assert [c[0] for c in calls] == ["Bearer " + key for key in keys]
    for key in keys:
        assert key not in json.dumps(result) + json.dumps(results) + caplog.text
