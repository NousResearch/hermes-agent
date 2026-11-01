from types import SimpleNamespace

from agent import relay_runtime
from hermes_cli import lifecycle, observability, plugins


def test_invoke_hook_notifies_builtin_observers_before_plugins(monkeypatch):
    calls = []
    manager = SimpleNamespace(
        invoke_hook=lambda name, **kwargs: calls.append(("plugin", name, kwargs)) or ["ok"]
    )
    monkeypatch.setattr(
        observability,
        "observe_lifecycle",
        lambda name, **kwargs: calls.append(("builtin", name, kwargs)),
    )
    monkeypatch.setattr(plugins, "invoke_hook", manager.invoke_hook)

    result = lifecycle.invoke_hook("on_session_start", session_id="session-1")

    assert result == ["ok"]
    assert [call[0] for call in calls] == ["builtin", "plugin"]


def test_finalize_session_closes_core_before_plugin_export(monkeypatch):
    calls = []
    manager = SimpleNamespace(
        invoke_hook=lambda name, **kwargs: calls.append(("plugin", name, kwargs)) or []
    )
    coordinator = SimpleNamespace(
        finalize_conversation=lambda **kwargs: calls.append(("core", kwargs))
    )
    monkeypatch.setattr(
        observability,
        "observe_lifecycle",
        lambda name, **kwargs: calls.append(("builtin", name, kwargs)),
    )
    monkeypatch.setattr(plugins, "invoke_hook", manager.invoke_hook)
    monkeypatch.setattr(relay_runtime, "SESSION_COORDINATOR", coordinator)
    monkeypatch.setattr(relay_runtime, "current_profile_key", lambda: "profile-1")

    lifecycle.finalize_session(session_id="session-1", platform="cli")

    assert [call[0] for call in calls] == ["builtin", "core", "plugin"]
    assert calls[1][1] == {
        "profile_key": "profile-1",
        "session_id": "session-1",
    }


def test_finalize_session_drains_outbound_queue_last(monkeypatch):
    """Regression: webhook delivery is a fire-and-forget daemon thread; without the
    bounded drain at finalize, a session close during process shutdown enqueued its
    event and exited before the POST left — subscribers never saw the retire."""
    import agent.outbound_webhooks as outbound

    calls = []
    manager = SimpleNamespace(
        invoke_hook=lambda name, **kwargs: calls.append(("plugin", name)) or []
    )
    monkeypatch.setattr(
        observability,
        "observe_lifecycle",
        lambda name, **kwargs: calls.append(("builtin", name)),
    )
    monkeypatch.setattr(plugins, "invoke_hook", manager.invoke_hook)
    monkeypatch.setattr(
        outbound, "flush", lambda timeout=5.0: calls.append(("drain", timeout)) or True
    )

    lifecycle.finalize_session(session_id="", platform="cli")

    # Builtin observers first (they enqueue), plugins next, drain last so the
    # FIFO — including this finalize — is fully handed off before returning.
    assert [call[0] for call in calls] == ["builtin", "plugin", "drain"]
    assert calls[2][1] <= 5.0  # bounded: shutdown must never hang on a wedged endpoint


def test_finalize_session_survives_drain_failure(monkeypatch):
    """A flush that raises (import trouble, wedged queue) is swallowed: session close
    never turns into a shutdown crash."""
    import agent.outbound_webhooks as outbound

    def _boom(timeout=5.0):
        raise RuntimeError("endpoint wedged")

    monkeypatch.setattr(observability, "observe_lifecycle", lambda name, **kwargs: None)
    monkeypatch.setattr(plugins, "invoke_hook", lambda name, **kwargs: ["msg"])
    monkeypatch.setattr(outbound, "flush", _boom)

    assert lifecycle.finalize_session(session_id=None, platform="cli") == ["msg"]


def test_plugin_only_dispatch_does_not_reenter_builtin_observers(monkeypatch):
    manager = SimpleNamespace(invoke_hook=lambda name, **kwargs: [name, kwargs])
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    monkeypatch.setattr(
        observability,
        "observe_lifecycle",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected")),
    )

    assert plugins.invoke_hook("custom", value=1) == ["custom", {"value": 1}]
