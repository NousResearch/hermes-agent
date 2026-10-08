"""Turn-route plugin command and middleware regression tests."""

import pytest

from hermes_cli.middleware import apply_turn_route_middleware, public_turn_route
from hermes_cli.plugins import PluginManager, invoke_plugin_command
from tests.hermes_cli.test_plugins import _make_plugin_dir


@pytest.mark.parametrize("parameter_name", ["session_id", "session_key", "platform"])
def test_legacy_plugin_command_parameter_names_keep_raw_args(parameter_name):
    calls = []
    handlers = {
        "session_id": lambda session_id: calls.append(session_id),
        "session_key": lambda session_key: calls.append(session_key),
        "platform": lambda platform: calls.append(platform),
    }

    invoke_plugin_command(
        handlers[parameter_name], "literal-user-args",
        session_id="physical-session", session_key="durable-session", platform="telegram",
    )

    assert calls == ["literal-user-args"]


def test_plugin_command_context_reaches_opt_in_parameters():
    calls = []

    def keyword_context(raw_args, *, session_key=None):
        calls.append((raw_args, session_key))

    def all_context(raw_args, **context):
        calls.append((raw_args, context))

    invoke_plugin_command(keyword_context, "literal-user-args", session_key="durable-session")
    invoke_plugin_command(
        all_context, "literal-user-args",
        session_id="physical-session", session_key="durable-session", platform="telegram",
    )

    assert calls == [
        ("literal-user-args", "durable-session"),
        ("literal-user-args", {
            "session_id": "physical-session", "session_key": "durable-session", "platform": "telegram",
        }),
    ]


def test_turn_route_trace_contains_plugin_identity_and_redacted_original(tmp_path, monkeypatch):
    plugins_dir = tmp_path / "hermes_test" / "plugins"
    _make_plugin_dir(
        plugins_dir,
        "router_plugin",
        register_body=(
            "ctx.register_middleware('turn_route', lambda **kw: {"
            "'route': {**kw['route'], 'marker': ('redacted' if 'api_key' not in repr(kw['original_redacted_route']) else 'leaked')}, "
            "'source': 'router'})"
        ),
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_test"))

    mgr = PluginManager()
    mgr.discover_and_load()
    monkeypatch.setattr("hermes_cli.plugins._delivery_manager", lambda: mgr)

    route = public_turn_route(
        "configured-model",
        {
            "provider": "custom",
            "requested_provider": "custom:alpha",
            "api_mode": "chat_completions",
            "api_key": "provider-token-value",
            "command": "hermes-acp",
            "args": ["--api-key", "acp-token-value"],
        },
    )
    result = apply_turn_route_middleware(route, session_id="physical", session_key="durable")

    assert result.payload["marker"] == "redacted"
    assert result.trace == [{"plugin": "router_plugin", "source": "router"}]
    assert "provider-token-value" not in repr(result.original_payload)
    assert "acp-token-value" not in repr(result.original_payload)


@pytest.mark.parametrize("followed_by_noop", [False, True])
def test_failed_turn_route_middleware_mutation_is_discarded(monkeypatch, followed_by_noop):
    """A failed route callback cannot change the effective route or poison later callbacks."""
    route = {"model": "configured-model", "provider": "configured-provider", "runtime": {}}
    seen_by_noop = []

    def mutates_then_raises(route, **_kwargs):
        route["model"] = "wrong-model"
        raise RuntimeError("route callback failed")

    def noop(route, **_kwargs):
        seen_by_noop.append(route.copy())

    manager = PluginManager()
    manager._middleware["turn_route"] = [mutates_then_raises]
    if followed_by_noop:
        manager._middleware["turn_route"].append(noop)
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)

    result = apply_turn_route_middleware(route)

    assert result.payload == route
    assert result.changed is False
    assert result.trace == []
    assert route["model"] == "configured-model"
    assert seen_by_noop == ([route] if followed_by_noop else [])


def test_successful_turn_route_middleware_callbacks_chain(monkeypatch):
    """Each successful route decision is the next callback's input."""
    route = {"model": "configured-model", "provider": "configured-provider", "runtime": {}}
    seen_by_second = []

    def select_intermediate(route, **_kwargs):
        route["model"] = "intermediate-model"
        return {"route": route}

    def refine_route(route, **_kwargs):
        seen_by_second.append(route.copy())
        route["model"] = "final-model"
        return {"route": route}

    manager = PluginManager()
    manager._middleware["turn_route"] = [select_intermediate, refine_route]
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)

    result = apply_turn_route_middleware(route)

    assert seen_by_second == [{**route, "model": "intermediate-model"}]
    assert result.payload["model"] == "final-model"
    assert result.changed is True
    assert result.trace == [{"source": "plugin"}, {"source": "plugin"}]
    assert route["model"] == "configured-model"
