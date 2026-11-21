import threading

from tests.agent.test_plugin_stream_hooks import _agent, _patch_test_callback_manager


def test_unknown_observer_does_not_suppress_status_but_keeps_moa_streaming(monkeypatch):
    from agent.turn_api_call import _should_stream

    agent = _agent()
    agent.provider = "moa"
    output = []
    agent._print_fn = output.append
    manager = _patch_test_callback_manager(monkeypatch, {})
    manager._discovered = False
    agent._vprint("request status")
    assert output == ["request status"]
    assert _should_stream(agent) is True
    assert manager._discovered is False


def test_busy_stream_observer_keeps_moa_streaming(monkeypatch):
    from agent.turn_api_call import _should_stream

    agent = _agent()
    agent.provider = "moa"
    manager = _patch_test_callback_manager(monkeypatch, {"on_stream_delta": [lambda **kwargs: None]})
    manager._discovery_lock = threading.Lock()
    manager._discovery_lock.acquire()
    try:
        assert _should_stream(agent) is True
    finally:
        manager._discovery_lock.release()


def test_stream_capability_lookup_does_not_discover(monkeypatch, tmp_path):
    from agent import plugin_stream_hooks as psh
    from hermes_cli import plugins
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    home = tmp_path / "undiscovered-stream-observer"
    manager = plugins.PluginManager(scope_key=str(home))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    discovery_calls = []

    def discover_and_load():
        discovery_calls.append(True)
        manager._discovered = True

    monkeypatch.setattr(manager, "discover_and_load", discover_and_load)
    monkeypatch.setattr(psh, "stream_reasoning_deltas_enabled", lambda: True)
    token = set_hermes_home_override(home)
    try:
        assert psh.has_stream_observer_hooks() is False
        assert psh.has_stream_observer_hooks(include_unknown=True) is True
        assert psh.has_reasoning_stream_observer_hooks() is True
        assert discovery_calls == []
        assert manager._discovered is False
    finally:
        reset_hermes_home_override(token)
