"""Recall diagnostics through real provider discovery, config, and HTTP client."""

import logging
import os
from pathlib import Path
import time

import httpx
import pytest
import yaml

from hermes_cli import config
from plugins.memory import load_memory_provider


@pytest.fixture
def recall_provider(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for name in list(os.environ):
        if name.startswith("OPENVIKING_"):
            monkeypatch.delenv(name, raising=False)
    (home / "config.yaml").write_text(yaml.safe_dump({
        "memory": {"provider": "openviking", "openviking": {
            "endpoint": "http://127.0.0.1:19473",
            "user": "private-user-fixture",
            "recall_timeout_seconds": 1.5,
            "recall_request_timeout_seconds": 0.75,
            "recall_prefer_abstract": True,
        }},
    }), encoding="utf-8")
    config._LOAD_CONFIG_CACHE.clear()
    requests = []
    behavior = {"error": None, "fallback": False}

    def handle(request):
        requests.append(request)
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok", "healthy": True, "version": "test"})
        if request.url.path == "/api/v1/system/status":
            return httpx.Response(200, json={"result": {"user": "private-user-fixture"}})
        if request.url.path == "/api/v1/content/read":
            return httpx.Response(200, json={"result": {"content": ""}})
        if request.url.path == "/api/v1/fs/ls":
            return httpx.Response(200, json={"result": []})
        assert request.url.path in {"/api/v1/search/search", "/api/v1/search/find"}
        if behavior["error"] is not None and not (
                behavior["fallback"] and request.url.path.endswith("/find")):
            raise behavior["error"]("private-query-fixture in transport error")
        return httpx.Response(200, json={"result": {"memories": [{
            "uri": "viking://user/test/memories/events/decision.md",
            "abstract": "A remembered decision.", "score": 0.9,
        }]}})

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        monkeypatch.setattr(httpx, "get", client.get)
        monkeypatch.setattr(httpx, "post", client.post)
        provider = load_memory_provider("openviking", register_skills=False)
        assert provider is not None
        provider.initialize("recall-test", hermes_home=str(home))
        assert provider._client is not None
        requests.clear()
        try:
            yield provider, behavior, requests
        finally:
            provider.shutdown()
    config._LOAD_CONFIG_CACHE.clear()


@pytest.mark.parametrize("error", [
    httpx.ReadTimeout, httpx.ConnectTimeout, httpx.WriteTimeout, httpx.PoolTimeout, TimeoutError,
])
def test_recall_timeout_warns_without_identity_probe_or_private_text(recall_provider, caplog, error):
    provider, behavior, requests = recall_provider
    behavior["error"] = error
    with caplog.at_level(logging.WARNING, logger="plugins.memory.openviking"):
        assert provider._search_prefetch_context("private-query-fixture") == ""
    warnings = [r for r in caplog.records if r.name == "plugins.memory.openviking" and r.levelno == logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "recall timed out" in message
    assert error.__name__ in message
    assert "budget_s=1.5" in message
    assert "request_s=0.75" in message
    assert "private-query-fixture" not in message
    assert "private-user-fixture" not in message
    assert [request.url.path for request in requests] == ["/api/v1/search/find"]
    assert requests[0].extensions["timeout"]["read"] <= 0.75


def test_recovered_session_search_timeout_keeps_recall_and_stays_quiet(recall_provider, caplog):
    provider, behavior, requests = recall_provider
    behavior.update(error=httpx.ReadTimeout, fallback=True)
    with caplog.at_level(logging.WARNING, logger="plugins.memory.openviking"):
        result = provider._search_prefetch_context("private-query-fixture", session_id="session")
    assert "A remembered decision." in result
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert [request.url.path for request in requests] == ["/api/v1/search/search", "/api/v1/search/find"]


def test_public_prefetch_warns_when_session_search_exhausts_total_budget(recall_provider, monkeypatch, caplog):
    provider, behavior, requests = recall_provider
    provider.prefetch("")  # Complete the once-per-session profile stage through HTTP.
    requests.clear()
    now = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])

    def exhaust_budget(message):
        now[0] += 2.0
        return httpx.ReadTimeout(message)

    behavior["error"] = exhaust_budget
    with caplog.at_level(logging.WARNING, logger="plugins.memory.openviking"):
        assert provider.prefetch("private-query-fixture") == ""
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "TimeoutError" in warnings[0].getMessage()
    assert "budget_s=1.5" in warnings[0].getMessage()
    assert [request.url.path for request in requests] == ["/api/v1/search/search"]


@pytest.mark.parametrize("error", [ValueError, httpx.ConnectError])
def test_non_timeout_search_failure_keeps_debug_level(recall_provider, caplog, error):
    provider, behavior, requests = recall_provider
    behavior["error"] = error
    with caplog.at_level(logging.DEBUG, logger="plugins.memory.openviking"):
        assert provider._search_prefetch_context("private-query-fixture") == ""
    assert any("context search failed" in r.getMessage() and r.levelno == logging.DEBUG for r in caplog.records)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert [request.url.path for request in requests] == ["/api/v1/search/find"]
