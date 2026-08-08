"""Regression coverage for exact runtime-route diagnostics in ``hermes doctor``."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hermes_cli import config as config_module
from hermes_cli import doctor, doctor_routes


def test_validate_route_config_rejects_misplaced_model_fallback_chain():
    issues = doctor_routes.validate_route_config({
        "model": {
            "provider": "openrouter",
            "default": "openai/gpt-4.1-mini",
            "fallback_providers": [{"provider": "deepseek", "model": "deepseek-chat"}],
        }
    })

    assert any(issue.path == "model.fallback_providers" for issue in issues)
    assert any("top-level fallback_providers" in issue.fix for issue in issues)


def test_validate_route_config_rejects_stringified_fallback_chain():
    issues = doctor_routes.validate_route_config({
        "fallback_providers": ("[{'provider': 'deepseek', 'model': 'deepseek-chat'}]")
    })

    assert any(issue.path == "fallback_providers" for issue in issues)
    assert any("YAML list" in issue.message for issue in issues)


@pytest.mark.parametrize(
    "entry",
    [
        "deepseek/deepseek-chat",
        {"provider": "deepseek"},
        {"model": "deepseek-chat"},
        {"provider": 123, "model": "deepseek-chat"},
    ],
)
def test_validate_route_config_rejects_invalid_fallback_entries(entry):
    issues = doctor_routes.validate_route_config({"fallback_providers": [entry]})

    assert any(issue.path.startswith("fallback_providers[0]") for issue in issues)


def test_validate_route_config_accepts_well_formed_routes():
    issues = doctor_routes.validate_route_config({
        "model": {"provider": "openrouter", "default": "openai/gpt-4.1-mini"},
        "fallback_providers": [{"provider": "deepseek", "model": "deepseek-chat"}],
        "delegation": {
            "provider": "nvidia",
            "model": "qwen/qwen3.5-122b-a10b",
            "child_timeout_seconds": 0,
            "max_spawn_depth": 2,
            "max_concurrent_children": 3,
        },
        "auxiliary": {
            "compression": {
                "provider": "deepseek",
                "model": "deepseek-chat",
            }
        },
    })

    assert issues == []


def test_validate_route_config_accepts_disabled_empty_fallback():
    assert doctor_routes.validate_route_config({"fallback_model": {}}) == []
    assert doctor_routes.validate_route_config({"fallback_providers": []}) == []


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("child_timeout_seconds", -1),
        ("child_timeout_seconds", "600"),
        ("max_spawn_depth", 0),
        ("max_spawn_depth", "2"),
        ("max_concurrent_children", 0),
    ],
)
def test_validate_route_config_rejects_malformed_delegation_limits(key, value):
    issues = doctor_routes.validate_route_config({"delegation": {key: value}})

    assert any(issue.path == f"delegation.{key}" for issue in issues)


def test_collect_configured_routes_covers_every_explicit_runtime_surface():
    routes = doctor_routes.collect_configured_routes({
        "model": {"provider": "openrouter", "default": "openai/gpt-4.1-mini"},
        "fallback_providers": [{"provider": "deepseek", "model": "deepseek-chat"}],
        "delegation": {
            "provider": "nvidia",
            "model": "qwen/qwen3.5-122b-a10b",
        },
        "auxiliary": {
            "compression": {
                "provider": "xiaomi",
                "model": "mimo-v2-flash",
            },
            "title_generation": {"provider": "auto", "model": ""},
        },
    })

    by_label = {route.label: route for route in routes}
    assert set(by_label) == {
        "model.primary",
        "fallback[0]",
        "delegation",
        "auxiliary.compression",
    }
    assert by_label["auxiliary.compression"].task == "compression"
    assert by_label["auxiliary.compression"].provider == "xiaomi"


def test_collect_configured_routes_uses_neutral_label_for_legacy_fallback():
    routes = doctor_routes.collect_configured_routes({
        "model": {"provider": "openrouter", "default": "openai/gpt-4.1-mini"},
        "fallback_model": {"provider": "deepseek", "model": "deepseek-chat"},
    })

    assert [route.label for route in routes] == ["model.primary", "fallback[0]"]


def test_probe_route_executes_exact_route_without_fallback(monkeypatch):
    calls = {}

    class FakeCompletions:
        def create(self, **kwargs):
            calls["kwargs"] = kwargs
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="OK"))]
            )

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions()),
        base_url="https://api.example.test/v1",
    )

    def fake_resolve(provider, model=None, **kwargs):
        calls["resolve"] = (provider, model, kwargs)
        return fake_client, model

    monkeypatch.setattr(doctor_routes, "resolve_provider_client", fake_resolve)
    monkeypatch.setattr(
        doctor_routes,
        "_get_task_extra_body",
        lambda task: {"diagnostic_task": task},
    )

    def fake_build(provider, model, messages, **kwargs):
        calls["build"] = (provider, model, kwargs)
        return {
            "model": model,
            "messages": messages,
            "timeout": kwargs["timeout"],
        }

    monkeypatch.setattr(doctor_routes, "_build_call_kwargs", fake_build)

    result = doctor_routes.probe_route(
        doctor_routes.RouteSpec(
            label="auxiliary.compression",
            provider="xiaomi",
            model="mimo-v2-flash",
            base_url="https://api.example.test/v1",
            api_key="secret",
            task="compression",
        ),
        timeout=7,
    )

    assert result.ok is True
    assert calls["resolve"] == (
        "xiaomi",
        "mimo-v2-flash",
        {
            "explicit_base_url": "https://api.example.test/v1",
            "explicit_api_key": "secret",
            "api_mode": None,
        },
    )
    assert calls["kwargs"]["timeout"] == 7
    assert calls["build"][2]["extra_body"] == {"diagnostic_task": "compression"}


def test_probe_route_reports_auth_failure_and_redacts_secret(monkeypatch):
    class FakeCompletions:
        def create(self, **kwargs):
            raise RuntimeError("HTTP 401 api_key=not-a-real-key")

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions()),
        base_url="https://api.example.test/v1",
    )
    monkeypatch.setattr(
        doctor_routes,
        "resolve_provider_client",
        lambda *args, **kwargs: (fake_client, "broken-model"),
    )
    monkeypatch.setattr(
        doctor_routes,
        "_build_call_kwargs",
        lambda *args, **kwargs: {"model": "broken-model", "messages": []},
    )

    result = doctor_routes.probe_route(
        doctor_routes.RouteSpec(
            label="delegation",
            provider="openrouter",
            model="broken-model",
            api_key="not-a-real-key",
        )
    )

    assert result.ok is False
    assert "401" in result.detail
    assert "not-a-real-key" not in result.detail


def _install_doctor_route_config(monkeypatch):
    config = {"model": {"provider": "openrouter", "default": "openai/gpt-4.1-mini"}}
    monkeypatch.setattr(config_module, "read_raw_config", lambda: config)
    monkeypatch.setattr(config_module, "load_config", lambda: config)


def test_doctor_default_discloses_unprobed_routes_without_blocking(monkeypatch, capsys):
    _install_doctor_route_config(monkeypatch)
    issues = []

    doctor._check_configured_runtime_routes(issues, probe_routes=False)

    output = capsys.readouterr().out
    assert issues == []
    assert "Live inference routes were not probed" in output
    assert "--probe-routes" in output


def test_doctor_live_route_failure_is_blocking(monkeypatch, capsys):
    _install_doctor_route_config(monkeypatch)
    monkeypatch.setattr(
        doctor_routes,
        "probe_route",
        lambda route: doctor_routes.ProbeResult(route, False, "HTTP 401 invalid key"),
    )
    issues = []

    doctor._check_configured_runtime_routes(issues, probe_routes=True)

    output = capsys.readouterr().out
    assert any("failed live inference" in issue for issue in issues)
    assert "HTTP 401 invalid key" in output


def test_doctor_unprobeable_route_is_blocking(monkeypatch, capsys):
    _install_doctor_route_config(monkeypatch)
    monkeypatch.setattr(
        doctor_routes,
        "probe_route",
        lambda route: doctor_routes.ProbeResult(
            route,
            False,
            "transport has no chat-completions interface",
            skipped=True,
        ),
    )
    issues = []

    doctor._check_configured_runtime_routes(issues, probe_routes=True)

    output = capsys.readouterr().out
    assert any("could not be live-probed" in issue for issue in issues)
    assert "transport has no chat-completions interface" in output
