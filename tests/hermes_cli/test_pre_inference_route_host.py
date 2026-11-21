"""Pre-inference routing host extension (issue #104).

Covers the host half of the Hermes pre-inference routing contract:
``hermes_cli.middleware.apply_pre_inference_route_middleware`` plus the
``tui_gateway.agent_factory`` pre-inference seam.

The AutoRouter consumer itself is tested on the AutoRouter side; the optional
``AUTOROUTER_SOURCE``-gated case wires the real consumer into a scratch
HERMES_HOME end to end.
"""
from __future__ import annotations

import os
import pytest

from hermes_cli import plugins
from hermes_cli.middleware import (
    PRE_INFERENCE_ROUTE_MIDDLEWARE,
    PreInferenceRouteResult,
    apply_pre_inference_route_middleware,
)
from hermes_cli.plugins import PluginManager

_CONSUMER_PLUGIN = """
import os

_OUTCOMES = {
    'route': {'status': 'route', 'provider': 'nvidia', 'model': 'nvidia-free',
              'reason': 'fixture route'},
    'refuse': {'status': 'refuse', 'reason': 'fixture refusal: no eligible free model'},
    'passthrough': {'status': 'passthrough', 'reason': 'native_routing_disabled'},
    'malformed': {'status': 'route', 'provider': 'NOT A PROVIDER!', 'model': 'x'},
    'nondict': 'not a dict',
    'none': None,
    'raise': RuntimeError('consumer exploded'),
    'toolongmodel': {'status': 'route', 'provider': 'nvidia', 'model': 'm' * 500,
                     'reason': 'too long'},
    'controlmodel': {'status': 'route', 'provider': 'nvidia', 'model': 'bad' + chr(10) + 'model',
                     'reason': 'control chars'},
    'keylike': {'status': 'route', 'provider': 'nvidia',
                'model': 'sk-abcdefghijklmnopqrstuvwxyz123456', 'reason': 'leaked'},
    'badprovider': {'status': 'route', 'provider': 'NOT A PROVIDER!',
                    'model': 'nvidia-free', 'reason': 'bad provider'},
}

def register(ctx):
    def consume(**kwargs):
        outcome = _OUTCOMES[os.environ['FIXTURE_OUTCOME']]
        if isinstance(outcome, Exception):
            raise outcome
        if outcome is None:
            return None
        return outcome if not isinstance(outcome, dict) else dict(outcome)

    ctx.register_middleware('pre_inference_route', consume)
"""


def _write_plugin(home, name, source, extra_yaml=""):
    plugin_dir = home / "plugins" / name
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(f"name: {name}\nversion: 0.1.0\n{extra_yaml}")
    (plugin_dir / "__main__.py").write_text("")  # unused; package import uses __init__.py
    (plugin_dir / "__init__.py").write_text(source)


@pytest.fixture
def consumer_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    _write_plugin(home, "autorouter_fixture", _CONSUMER_PLUGIN)
    (home / "config.yaml").write_text(
        "plugins:\n  enabled: [autorouter_fixture]\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("FIXTURE_OUTCOME", "route")
    manager = PluginManager()
    manager.discover_and_load()
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    return home


def _route(status):
    return apply_pre_inference_route_middleware({"session_id": "sess-1"})


def test_no_consumer_registered_returns_none(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("plugins:\n  enabled: []\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = PluginManager()
    manager.discover_and_load()
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    assert apply_pre_inference_route_middleware({"session_id": "sess-x"}) is None


@pytest.mark.parametrize("outcome,expected", [
    ("route", ("route", "nvidia", "nvidia-free")),
    ("refuse", ("refuse", None, None)),
    ("passthrough", ("passthrough", None, None)),
])
def test_consumer_outcomes(consumer_home, monkeypatch, outcome, expected):
    monkeypatch.setenv("FIXTURE_OUTCOME", outcome)
    route = _route(outcome)
    assert route is not None
    assert route.status == expected[0]
    assert route.provider == expected[1]
    assert route.model == expected[2]


def test_malformed_result_is_fail_closed(consumer_home, monkeypatch):
    monkeypatch.setenv("FIXTURE_OUTCOME", "malformed")
    route = _route("malformed")
    assert route is not None
    assert route.status == "refuse"
    assert "malformed" in route.reason


def test_metadata_is_sanitized_before_delivery(consumer_home, monkeypatch, tmp_path):
    captured = {}

    def probe(**kwargs):
        captured.update(kwargs)
        return {"status": "passthrough"}

    plugins.get_plugin_manager()._middleware[PRE_INFERENCE_ROUTE_MIDDLEWARE] = [probe]
    apply_pre_inference_route_middleware(
        {"session_id": "sess-2", "note": "Bearer sk-abc123def456ghi789"})
    assert captured["note"] == "[redacted]"
    assert captured["session_id"] == "sess-2"


def test_pre_inference_override_never_rewrites_explicit_pick():
    from tui_gateway import agent_factory

    override = {"model": "explicit/picked", "provider": "openrouter"}
    assert agent_factory._pre_inference_route_override(
        "s", "sess", "k", override, None, {"model": {}}, None) == (override, None)
    assert agent_factory._pre_inference_route_override(
        "s", "sess", "k", None, "explicit-provider", {"model": {}}, None) == (None, "explicit-provider")


def test_pre_inference_override_route_and_refusal(consumer_home, monkeypatch):
    from tui_gateway import agent_factory

    monkeypatch.setenv("FIXTURE_OUTCOME", "route")
    model_override, provider_override = agent_factory._pre_inference_route_override(
        "s", "sess-77", "k", None, None,
        {"model": {"default": "cfg-default", "provider": "cfg-provider"}}, None)
    assert model_override == {"model": "nvidia-free", "provider": "nvidia"}
    assert provider_override is None

    monkeypatch.setenv("FIXTURE_OUTCOME", "refuse")
    with pytest.raises(RuntimeError, match="refused"):
        agent_factory._pre_inference_route_override(
            "s", "sess-78", "k", None, None,
            {"model": {"default": "cfg-default", "provider": "cfg-provider"}}, None)

    monkeypatch.setenv("FIXTURE_OUTCOME", "passthrough")
    assert agent_factory._pre_inference_route_override(
        "s", "sess-79", "k", None, None, {"model": {}}, None) == (None, None)


@pytest.mark.skipif(not os.environ.get("AUTOROUTER_SOURCE"),
                    reason="AUTOROUTER_SOURCE not set to an autorouter checkout")
def test_real_autorouter_consumer_end_to_end(tmp_path, monkeypatch):
    """Optional cross-repository proof: install the real AutoRouter plugin into a
    scratch HERMES_HOME, opt in, seed its catalog, and confirm the ordinary
    pre-inference seam routes a new session with no other configuration."""
    import sys
    from pathlib import Path

    source = Path(os.environ["AUTOROUTER_SOURCE"])
    home = tmp_path / "home"
    plugin = home / "plugins" / "autorouter"
    plugin.parent.mkdir(parents=True)
    # Vendored link to the checkout: relative imports inside the package keep working.
    os.symlink(source, plugin)
    cfg_dir = home / "plugin-data" / "autorouter"
    sys.path.insert(0, str(source))
    monkeypatch.setenv("HERMES_HOME", str(home))

    import importlib.util
    spec = importlib.util.spec_from_file_location("autorouter", source / "__init__.py")
    pkg = importlib.util.module_from_spec(spec)
    pkg.__path__ = [str(source)]
    pkg.__package__ = "autorouter"
    sys.modules["autorouter"] = pkg
    spec.loader.exec_module(pkg)
    import autorouter.autorouter_core.storage.state as state_mod

    store = state_mod.StateStore(cfg_dir)
    now = 1.0
    expiry = 9999999999.0
    import json as _json
    store.upsert_catalog("nvidia", "nvidia-free", context_length=65536,
                         capabilities={"tools": "supported"}, cost_tier="free")
    store.upsert_capability(
        "nvidia", "nvidia-free",
        _json.dumps({
            "provider": "nvidia", "model": "nvidia-free",
            "field_evidence": {
                "context_window": {"metadata_source": "provider",
                                   "retrieved_at": now, "expires_at": expiry},
                "tools": {"metadata_source": "provider",
                          "retrieved_at": now, "expires_at": expiry},
                "cost_tier": {"metadata_source": "provider",
                              "retrieved_at": now, "expires_at": expiry},
            },
            "context_window": 65536, "tools": "supported", "cost_tier": "free",
            "source": "provider",
        }),
        metadata_source="provider", retrieved_at=now, expires_at=expiry)
    (home / "config.yaml").write_text(
        "plugins:\n  enabled: [autorouter]\n  entries:\n    autorouter:\n"
        "      settings:\n        enabled: true\n        enable_native_routing: true\n"
        "        providers: [nvidia]\n        requires_tools: true\n"
    )
    manager = PluginManager()
    manager.discover_and_load()
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)

    from tui_gateway import agent_factory
    model_override, provider_override = agent_factory._pre_inference_route_override(
        "s", "sess-e2e", "k", None, None, {"model": {}}, None)
    assert model_override == {"model": "nvidia-free", "provider": "nvidia"}


# ---------------------------------------------------------------------------
# Dispatch 108 hardening: registered-consumer failures are refusals
# ---------------------------------------------------------------------------

def test_consumer_exception_is_fail_closed_refusal(consumer_home, monkeypatch):
    monkeypatch.setenv("FIXTURE_OUTCOME", "raise")
    route = apply_pre_inference_route_middleware({"session_id": "sess-raise"})
    assert route is not None
    assert route.status == "refuse"
    assert "no result" in route.reason or "malformed" in route.reason


def test_consumer_exception_raises_before_agent_creation(consumer_home, monkeypatch):
    from tui_gateway import agent_factory

    monkeypatch.setenv("FIXTURE_OUTCOME", "raise")
    with pytest.raises(RuntimeError, match="refused"):
        agent_factory._pre_inference_route_override(
            "s", "sess-raise-2", "k", None, None,
            {"model": {"default": "cfg-default", "provider": "cfg-provider"}}, None)


def test_consumer_none_result_is_fail_closed_refusal(consumer_home, monkeypatch):
    monkeypatch.setenv("FIXTURE_OUTCOME", "none")
    route = apply_pre_inference_route_middleware({"session_id": "sess-none"})
    assert route is not None
    assert route.status == "refuse"


def test_consumer_none_result_raises_before_agent_creation(consumer_home, monkeypatch):
    from tui_gateway import agent_factory

    monkeypatch.setenv("FIXTURE_OUTCOME", "none")
    with pytest.raises(RuntimeError, match="no result"):
        agent_factory._pre_inference_route_override(
            "s", "sess-none-2", "k", None, None,
            {"model": {"default": "cfg-default", "provider": "cfg-provider"}}, None)


def test_consumer_non_dict_result_is_refusal(consumer_home, monkeypatch):
    monkeypatch.setenv("FIXTURE_OUTCOME", "nondict")
    route = apply_pre_inference_route_middleware({"session_id": "sess-nondict"})
    assert route is not None
    assert route.status == "refuse"


def test_malformed_provider_refused(consumer_home, monkeypatch):
    monkeypatch.setenv("FIXTURE_OUTCOME", "badprovider")
    route = apply_pre_inference_route_middleware({"session_id": "sess-badprov"})
    assert route is not None and route.status == "refuse"


def test_model_too_long_refused(consumer_home, monkeypatch):
    monkeypatch.setenv("FIXTURE_OUTCOME", "toolongmodel")
    route = apply_pre_inference_route_middleware({"session_id": "sess-long"})
    assert route is not None and route.status == "refuse"


def test_model_with_control_chars_refused(consumer_home, monkeypatch):
    monkeypatch.setenv("FIXTURE_OUTCOME", "controlmodel")
    route = apply_pre_inference_route_middleware({"session_id": "sess-ctrl"})
    assert route is not None and route.status == "refuse"


def test_model_with_key_like_material_refused(consumer_home, monkeypatch):
    monkeypatch.setenv("FIXTURE_OUTCOME", "keylike")
    route = apply_pre_inference_route_middleware({"session_id": "sess-keylike"})
    assert route is not None and route.status == "refuse"
    assert "sk-" not in (route.model or "")



def test_raising_consumer_does_not_block_a_later_valid_consumer(monkeypatch):
    def bad(**kwargs):
        raise RuntimeError("boom")

    def good(**kwargs):
        return {"status": "route", "provider": "nvidia", "model": "nvidia-free",
                "reason": "fixture route"}

    monkeypatch.setenv("HERMES_HOME", "/nonexistent")
    from hermes_cli import plugins as _plugins
    from hermes_cli.plugins import PluginManager

    manager = PluginManager()
    monkeypatch.setattr(_plugins, "get_plugin_manager", lambda: manager)
    manager._middleware = {PRE_INFERENCE_ROUTE_MIDDLEWARE: [bad, good]}
    route = apply_pre_inference_route_middleware({"session_id": "sess-multi"})
    assert route is not None
    assert route.status == "route"
    assert route.provider == "nvidia"


def test_all_invalid_consumers_fail_closed(monkeypatch):
    def bad(**kwargs):
        raise RuntimeError("boom")

    def ugly(**kwargs):
        return {"status": "route", "provider": "!!", "model": "x"}

    from hermes_cli import plugins as _plugins
    from hermes_cli.plugins import PluginManager

    manager = PluginManager()
    monkeypatch.setattr(_plugins, "get_plugin_manager", lambda: manager)
    manager._middleware = {PRE_INFERENCE_ROUTE_MIDDLEWARE: [bad, ugly]}
    route = apply_pre_inference_route_middleware({"session_id": "sess-multi-2"})
    assert route is not None
    assert route.status == "refuse"


def test_explicit_model_pick_never_rewritten_even_when_consumer_raises(consumer_home, monkeypatch):
    from tui_gateway import agent_factory

    monkeypatch.setenv("FIXTURE_OUTCOME", "raise")
    override = {"model": "explicit/picked", "provider": "openrouter"}
    assert agent_factory._pre_inference_route_override(
        "s", "sess", "k", override, None, {"model": {}}, None) == (override, None)


def test_explicit_provider_pick_never_rewritten_even_when_consumer_raises(consumer_home, monkeypatch):
    from tui_gateway import agent_factory

    monkeypatch.setenv("FIXTURE_OUTCOME", "raise")
    assert agent_factory._pre_inference_route_override(
        "s", "sess", "k", None, "explicit-provider", {"model": {}}, None) == (None, "explicit-provider")


def test_refusal_happens_before_provider_client_construction(consumer_home, monkeypatch):
    from tui_gateway import agent_factory

    built = []

    def _sentinel(*args, **kwargs):
        built.append((args, kwargs))
        raise AssertionError("provider client must not be built on refusal")

    monkeypatch.setattr(agent_factory, "_resolve_agent_model_runtime", _sentinel)
    monkeypatch.setenv("FIXTURE_OUTCOME", "refuse")
    with pytest.raises(RuntimeError, match="refused"):
        agent_factory._pre_inference_route_override(
            "s", "sess-preprov", "k", None, None,
            {"model": {"default": "cfg-default", "provider": "cfg-provider"}}, None)
    assert built == []


def test_explicit_pick_path_never_touches_the_consumer(consumer_home, monkeypatch):
    from tui_gateway import agent_factory
    from hermes_cli import plugins as _plugins

    def _boom(**kwargs):
        raise AssertionError("consumer must not run for explicit picks")

    _plugins.get_plugin_manager()._middleware[PRE_INFERENCE_ROUTE_MIDDLEWARE] = [_boom]
    override = {"model": "explicit/picked", "provider": "openrouter"}
    assert agent_factory._pre_inference_route_override(
        "s", "sess", "k", override, None, {"model": {}}, None) == (override, None)


def test_stock_behavior_when_plugin_absent(tmp_path, monkeypatch):
    from hermes_cli import plugins as _plugins
    from hermes_cli.plugins import PluginManager

    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("plugins:\n  enabled: []\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = PluginManager()
    manager.discover_and_load()
    monkeypatch.setattr(_plugins, "get_plugin_manager", lambda: manager)

    from tui_gateway import agent_factory

    assert apply_pre_inference_route_middleware({"session_id": "sess-stock"}) is None
    assert agent_factory._pre_inference_route_override(
        "s", "sess-stock", "k", None, None, {"model": {}}, None) == (None, None)


def test_passthrough_consumer_proceeds_stock(consumer_home, monkeypatch):
    from tui_gateway import agent_factory

    monkeypatch.setenv("FIXTURE_OUTCOME", "passthrough")
    assert agent_factory._pre_inference_route_override(
        "s", "sess-pass", "k", None, None, {"model": {}}, None) == (None, None)


def test_zero_network_on_consumer_failure(consumer_home, monkeypatch):
    import socket

    def _boom(*args, **kwargs):
        raise AssertionError("network access is forbidden on the refusal path")

    monkeypatch.setattr(socket, "socket", _boom)
    monkeypatch.setattr(socket, "create_connection", _boom)
    monkeypatch.setenv("FIXTURE_OUTCOME", "raise")
    route = apply_pre_inference_route_middleware({"session_id": "sess-net"})
    assert route is not None and route.status == "refuse"


def test_metadata_is_bounded_and_sanitized(consumer_home, monkeypatch):
    captured = {}

    def probe(**kwargs):
        captured.update(kwargs)
        return {"status": "passthrough"}

    from hermes_cli import plugins as _plugins

    _plugins.get_plugin_manager()._middleware[PRE_INFERENCE_ROUTE_MIDDLEWARE] = [probe]
    long_id = "s" * 5000
    with_ctrl = "sess\x00with\x7fctrl"
    route = apply_pre_inference_route_middleware(
        {"session_id": long_id, "platform": with_ctrl,
         "note": "Bearer sk-abcdefghijklmnopqrstuvwxyz123456"})
    assert route is not None and route.status == "passthrough"
    assert len(captured["session_id"]) <= 256
    assert "\x00" not in captured["platform"] and "\x7f" not in captured["platform"]
    assert captured["note"] == "[redacted]"


def test_receipt_is_bounded_and_redacted(consumer_home, monkeypatch):
    from hermes_cli import plugins as _plugins

    def probe(**kwargs):
        return {"status": "refuse", "reason": "Bearer sk-abcdefghijklmnopqrstuvwxyz123456",
                "receipt": {"trace": "x" * 5000, "ok": True}}

    _plugins.get_plugin_manager()._middleware[PRE_INFERENCE_ROUTE_MIDDLEWARE] = [probe]
    route = apply_pre_inference_route_middleware({"session_id": "sess-receipt"})
    assert route is not None and route.status == "refuse"
    assert len(route.reason) <= 500
    assert "sk-" not in route.reason
    assert len(route.receipt["trace"]) <= 500
