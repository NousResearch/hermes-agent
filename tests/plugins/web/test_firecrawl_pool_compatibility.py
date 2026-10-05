import asyncio
from types import SimpleNamespace

import httpx
import pytest

from plugins.web.firecrawl import provider as fc


@pytest.mark.parametrize("mode", ["single", "self-hosted", "empty-pool", "nous", "plural-precedence"])
def test_legacy_routes_and_explicit_gateway_do_not_use_pool(tmp_path, monkeypatch, mode):
    from tools import web_tools
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(web_tools, "_firecrawl_client", None)
    monkeypatch.delenv("FIRECRAWL_API_KEYS", raising=False)
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    monkeypatch.delenv("FIRECRAWL_API_URL", raising=False)
    selection = "nous" if mode == "nous" else "firecrawl"
    (tmp_path / "config.yaml").write_text(f"web:\n  backend: {selection}\n")
    if mode != "self-hosted":
        monkeypatch.setenv("FIRECRAWL_API_KEY", "legacy-key")
    if mode == "self-hosted":
        monkeypatch.setenv("FIRECRAWL_API_URL", "http://localhost:3002")
    if mode == "empty-pool":
        monkeypatch.setenv("FIRECRAWL_API_KEYS", "[]")
    if mode == "nous":
        monkeypatch.setenv("FIRECRAWL_API_KEYS", "malformed-secret-must-be-ignored")
        monkeypatch.setattr(fc._gateway, "resolve_managed_tool_gateway", lambda *a, **kw: SimpleNamespace(
            nous_user_token="managed-token", gateway_origin="https://gateway.example"))
    if mode == "plural-precedence":
        monkeypatch.setenv("FIRECRAWL_API_KEYS", '[" primary-key ", "primary-key", "backup-key"]')
    constructed = []

    def factory(**kwargs):
        constructed.append(kwargs)
        return SimpleNamespace(search=lambda **kw: {"web": []})

    monkeypatch.setattr(fc, "Firecrawl", factory)
    assert fc.FirecrawlWebSearchProvider().search("test")["success"]
    expected = {
        "single": {"api_key": "legacy-key"},
        "self-hosted": {"api_url": "http://localhost:3002"},
        "empty-pool": {"api_key": "legacy-key"},
        "nous": {"api_key": "managed-token", "api_url": "https://gateway.example"},
        "plural-precedence": {"api_key": "primary-key", "max_retries": 0},
    }
    assert constructed == [expected[mode]]


def test_simultaneous_first_use_shares_one_profile_pool(tmp_path, monkeypatch):
    from agent import secret_scope
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    (tmp_path / "config.yaml").write_text("web:\n  backend: firecrawl\n")
    calls = []

    class Client:
        def __init__(self, api_key, **kwargs):
            self.key = api_key

        def search(self, **kwargs):
            calls.append(self.key)
            if self.key == "empty-key":
                response = httpx.Response(402, request=httpx.Request("POST", "https://api.firecrawl.dev/v2/search"))
                raise httpx.HTTPStatusError("empty-key", request=response.request, response=response)
            return {"web": []}

    monkeypatch.setattr(fc, "Firecrawl", Client)
    ht = set_hermes_home_override(tmp_path)
    st = secret_scope.set_secret_scope({"FIRECRAWL_API_KEYS": '["empty-key", "working-key"]'})
    old = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    try:
        async def parallel():
            p = fc.FirecrawlWebSearchProvider()
            return await asyncio.gather(*(asyncio.to_thread(p.search, "test") for _ in range(16)))
        assert all(r["success"] for r in asyncio.run(parallel()))
        assert calls == ["empty-key"] + ["working-key"] * 16
    finally:
        secret_scope.set_multiplex_active(old)
        secret_scope.reset_secret_scope(st)
        reset_hermes_home_override(ht)
