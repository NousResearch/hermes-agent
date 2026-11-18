"""Public plugin pollers must never construct a warmer or consult live session storage."""

import json

import pytest

from hermes_cli.plugins import PluginContext, PluginManager
from hermes_cli.plugins_manifest import PluginManifest
from tests.agent.prompt_cache_lifecycle.test_warmer import priced


def test_unobserved_session_is_an_observational_json_snapshot(tmp_path, monkeypatch):
    manager = PluginManager(scope_key=str(tmp_path))
    ctx = PluginContext(PluginManifest(name="cache-observer"), manager)

    def forbidden(*args, **kwargs):
        raise AssertionError("a status read must not construct a controller")

    monkeypatch.setattr("agent.prompt_cache_lifecycle.warmer.get_prompt_cache_warmer", forbidden)
    for _ in range(3):
        result = ctx.prompt_cache_warm_status(session_id="not-observed")
        assert json.loads(json.dumps(result, allow_nan=False)) == {
            "managed": False, "next_warm_at": None, "reason": "unavailable",
        }


OBSERVER_PLUGIN = '''
import json
def register(ctx):
    def poll(args, **kwargs):
        assert hasattr(ctx, "prompt_cache_warm_status")
        return json.dumps(ctx.prompt_cache_warm_status(session_id=args["session_id"]), allow_nan=False)
    ctx.register_tool(name="cache_status_probe", toolset="cache_status_probe",
        schema={"name": "cache_status_probe", "description": "Observe cache warming",
                "parameters": {"type": "object", "properties": {"session_id": {"type": "string"}}}},
        handler=poll)
'''


@pytest.mark.platforms("any")  # host isolation starts an actual child process
@pytest.mark.parametrize("isolation", ["host", "in_process"])
def test_real_plugin_discovery_and_scalar_host_path(tmp_path, monkeypatch, isolation, request):
    from agent.prompt_cache_lifecycle import warmer as lifecycle
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tests.agent.prompt_cache_lifecycle.test_warmer import _agent, _arm
    from tests.hermes_cli.test_plugin_host import _home_with_plugins
    from tools.registry import registry

    request.getfixturevalue("priced")
    _home_with_plugins(tmp_path, monkeypatch, {"cache-observer": OBSERVER_PLUGIN}, isolation=isolation)
    agent = _agent()
    assert _arm(agent) is None
    manager = PluginManager()
    manager.discover_and_load()
    try:
        assert manager._plugins["cache-observer"].error is None
        token = set_hermes_home_override(tmp_path / "unrelated-poller-home")
        try:
            scheduled = json.loads(registry.dispatch("cache_status_probe", {"session_id": "s"}, scope=manager.scope_key))
            missing = json.loads(registry.dispatch("cache_status_probe", {"session_id": "unobserved"}, scope=manager.scope_key))
        finally:
            reset_hermes_home_override(token)
        assert scheduled["managed"] is True and scheduled["reason"] == "scheduled"
        assert isinstance(scheduled["next_warm_at"], float)
        assert missing == {"managed": False, "next_warm_at": None, "reason": "unavailable"}
        lifecycle.cancel_prompt_cache_warming(agent)
        expired = json.loads(registry.dispatch("cache_status_probe", {"session_id": "s"}, scope=manager.scope_key))
        assert expired == {"managed": True, "next_warm_at": None, "reason": "expiring"}
    finally:
        lifecycle.cancel_prompt_cache_warming(agent)
        manager.unload()
        if isolation == "host":
            manager._plugin_host().shutdown()
