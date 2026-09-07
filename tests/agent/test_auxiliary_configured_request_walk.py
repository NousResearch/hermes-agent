"""Configured backups must recover from request failures, without widening policy."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
import yaml

from agent import auxiliary_client as aux


class ProviderFailure(Exception):
    def __init__(self, status: int, message: str = "provider unavailable"):
        super().__init__(message)
        self.status_code = status


def response(text="recovered"):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


@pytest.fixture
def routes(monkeypatch, tmp_path):
    """Only credential/client acquisition is replaced; run the actual call and chain code."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    calls = []
    outcomes = {"primary": ProviderFailure(429), "first": ProviderFailure(503), "second": response()}
    chain = [
        {"provider": "openrouter", "model": "first"},
        {"provider": "openrouter", "model": "second"},
    ]
    config = {"provider": "custom", "model": "primary", "timeout": 30, "fallback_chain": chain}
    (home / "config.yaml").write_text(yaml.safe_dump({"auxiliary": {"title_generation": config}}))
    monkeypatch.setattr(aux, "_get_auxiliary_task_config", lambda task: config)
    monkeypatch.setattr(aux, "_get_task_extra_body", lambda task: {})
    monkeypatch.setattr(aux, "_transient_retry_count", lambda: 0)
    monkeypatch.setattr(aux, "_recover_provider_pool", lambda *a, **kw: None)
    monkeypatch.setattr(aux, "_refresh_provider_credentials", lambda *a, **kw: False)
    monkeypatch.setattr(aux, "_is_provider_unhealthy", lambda *a, **kw: False)
    monkeypatch.setattr(aux, "_mark_provider_unhealthy", lambda *a, **kw: None)

    def client(model, asynchronous=False):
        def create(**kwargs):
            calls.append((model, kwargs.get("timeout")))
            value = outcomes[model]
            if callable(value):
                value = value()
            if isinstance(value, BaseException):
                raise value
            return value
        async def async_create(**kwargs):
            return create(**kwargs)
        return SimpleNamespace(base_url="https://example.invalid/v1", chat=SimpleNamespace(completions=SimpleNamespace(create=async_create if asynchronous else create)))

    monkeypatch.setattr(aux, "_get_cached_client", lambda provider, model=None, **kw: (client(model or "primary", kw.get("async_mode", False)), model or "primary"))
    monkeypatch.setattr(aux, "_resolve_fallback_entry", lambda entry: (client(entry["model"]), entry["model"]))
    monkeypatch.setattr(aux, "_to_async_client", lambda c, model, **kw: (client(model, True), model))
    monkeypatch.setattr(aux, "_try_payment_fallback", lambda *a, **kw: (None, None, ""))
    monkeypatch.setattr(aux, "_try_main_agent_model_fallback", lambda *a, **kw: (None, None, ""))
    return SimpleNamespace(calls=calls, outcomes=outcomes, config=config, chain=chain)


def invoke(asynchronous=False):
    kwargs = dict(task="title_generation", provider="custom", model="primary", messages=[{"role": "user", "content": "synthetic"}], timeout=30)
    return asyncio.run(aux.async_call_llm(**kwargs)) if asynchronous else aux.call_llm(**kwargs)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_first_backup_503_advances_to_second_and_reports_actual_route(routes, asynchronous):
    result = invoke(asynchronous)
    assert result.choices[0].message.content == "recovered"
    assert [name for name, _ in routes.calls] == ["primary", "first", "second"]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("status,message", [(503, "overloaded"), (404, "model_not_found")])
def test_primary_capacity_or_missing_model_uses_configured_backup(routes, asynchronous, status, message):
    routes.outcomes["primary"] = ProviderFailure(status, message)
    routes.outcomes["first"] = response()
    assert invoke(asynchronous).choices[0].message.content == "recovered"
    assert list(dict.fromkeys(name for name, _ in routes.calls)) == ["primary", "first"]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("status", [400, 403])
def test_backup_request_policy_error_does_not_try_another_provider(routes, asynchronous, status):
    routes.outcomes["first"] = ProviderFailure(status, "request rejected by policy")
    with pytest.raises(ProviderFailure) as exc:
        invoke(asynchronous)
    assert exc.value.status_code == status
    assert [name for name, _ in routes.calls] == ["primary", "first"]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_explicit_primary_auth_failure_stays_on_its_provider(routes, asynchronous):
    routes.outcomes["primary"] = ProviderFailure(401, "invalid credential")
    with pytest.raises(ProviderFailure):
        invoke(asynchronous)
    assert all(name == "primary" for name, _ in routes.calls)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_explicit_cancel_is_never_recovery(routes, asynchronous):
    routes.outcomes["first"] = aux.AuxiliaryExplicitCancellation()
    with pytest.raises(aux.AuxiliaryExplicitCancellation):
        invoke(asynchronous)
    assert [name for name, _ in routes.calls] == ["primary", "first"]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_duplicate_deployment_is_not_retried(routes, asynchronous):
    routes.chain.insert(1, dict(routes.chain[0]))
    assert invoke(asynchronous).choices[0].message.content == "recovered"
    assert [name for name, _ in routes.calls] == ["primary", "first", "second"]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_payment_failure_skips_sibling_models_sharing_credential(routes, asynchronous):
    routes.outcomes["first"] = ProviderFailure(402, "insufficient credits")
    routes.chain.append({"provider": "gemini", "model": "third"})
    routes.outcomes["third"] = response("independent")
    assert invoke(asynchronous).choices[0].message.content == "independent"
    assert [name for name, _ in routes.calls] == ["primary", "first", "third"]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_exhaustion_raises_last_request_error_without_restarting_chain(routes, asynchronous):
    routes.outcomes["second"] = ProviderFailure(503, "last backup unavailable")
    with pytest.raises(ProviderFailure, match="last backup unavailable"):
        invoke(asynchronous)
    assert [name for name, _ in routes.calls] == ["primary", "first", "second"]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_explicit_total_budget_caps_entry_timeout_and_does_not_leak(routes, asynchronous):
    routes.config["fallback_total_timeout"] = 5
    routes.chain[0]["timeout"] = 240
    assert invoke(asynchronous).choices[0].message.content == "recovered"
    assert all(0 < timeout <= 5 for name, timeout in routes.calls if name != "primary")
    assert aux._AUX_FALLBACK_DEADLINE.get() is None


@pytest.mark.parametrize("asynchronous", [False, True])
def test_expired_budget_prevents_another_backup(routes, asynchronous, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(aux.time, "monotonic", lambda: clock[0])
    routes.config["fallback_total_timeout"] = 1
    def expire():
        clock[0] += 2
        return ProviderFailure(503)
    routes.outcomes["first"] = expire
    with pytest.raises(TimeoutError, match="fallback budget exhausted"):
        invoke(asynchronous)
    assert [name for name, _ in routes.calls] == ["primary", "first"]
    assert aux._AUX_FALLBACK_DEADLINE.get() is None


@pytest.mark.parametrize("asynchronous", [False, True])
def test_real_sdk_and_config_reach_second_backup_over_loopback(tmp_path, monkeypatch, asynchronous):
    """Real config loader, credentials resolver, SDK and HTTP; only the provider is synthetic."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    seen = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_GET(self):
            body = json.dumps({"data": [{"id": m} for m in ["primary", "first", "second"]]}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body)
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            model = payload["model"]; seen.append(model)
            status = {"primary": 429, "first": 503, "second": 200}[model]
            if status == 200:
                value = {"id": "synthetic", "object": "chat.completion", "created": 0, "model": model, "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "loopback-recovered"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
            else:
                value = {"error": {"message": "rate limit" if status == 429 else "provider overloaded", "type": "capacity", "code": str(status)}}
            body = json.dumps(value).encode()
            self.send_response(status); self.send_header("Content-Type", "application/json"); self.send_header("Retry-After", "0.001"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True); worker.start()
    base = f"http://127.0.0.1:{server.server_port}/v1"
    home = tmp_path / "real-home"; home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    config = {"model": {"provider": "custom", "default": "primary", "base_url": base, "api_key": "synthetic"}, "auxiliary": {"transient_retries": 0, "title_generation": {"provider": "custom", "model": "primary", "base_url": base, "api_key": "synthetic", "timeout": 5, "fallback_total_timeout": 15, "fallback_chain": [{"provider": "custom", "model": m, "base_url": base, "api_key": "synthetic"} for m in ["first", "second"]]}}}
    (home / "config.yaml").write_text(yaml.safe_dump(config))
    route_info = {}
    kwargs = dict(task="title_generation", messages=[{"role": "user", "content": "synthetic"}], max_tokens=16, route_info=route_info)
    try:
        result = asyncio.run(aux.async_call_llm(**kwargs)) if asynchronous else aux.call_llm(**kwargs)
        assert result.choices[0].message.content == "loopback-recovered"
        assert list(dict.fromkeys(seen)) == ["primary", "first", "second"]
        assert route_info["model"] == "second"
    finally:
        server.shutdown(); server.server_close(); worker.join(timeout=2)
