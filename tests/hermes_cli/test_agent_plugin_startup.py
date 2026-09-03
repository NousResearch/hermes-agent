"""Regression coverage for plugin discovery at agent construction time."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_interactive_agent_discovers_plugins_before_credential_setup(monkeypatch):
    import cli
    import hermes_cli.cli_agent_setup_mixin as setup_mixin
    import hermes_cli.plugins as plugins

    events: list[str] = []
    monkeypatch.setattr(
        cli,
        "_prepare_deferred_agent_startup",
        lambda: events.append("deferred"),
    )
    monkeypatch.setattr(
        plugins,
        "discover_plugins",
        lambda: events.append("plugins"),
    )

    agent_setup = SimpleNamespace(
        agent=None,
        finalize_preloaded_skills=lambda: events.append("skills"),
        _install_tool_callbacks=lambda: events.append("callbacks"),
        _ensure_tirith_security=lambda: events.append("tirith"),
        _ensure_runtime_credentials=lambda: False,
    )

    result = setup_mixin.CLIAgentSetupMixin._init_agent(agent_setup)

    assert result is False
    assert events == ["skills", "deferred", "plugins", "callbacks", "tirith"]


def test_oneshot_discovers_plugins_before_agent_build(monkeypatch):
    import hermes_cli.config
    import hermes_cli.oneshot as oneshot
    import hermes_cli.plugins as plugins
    import run_agent  # noqa: F401 - warm imports before recording ordering

    events: list[str] = []
    monkeypatch.setattr(
        plugins,
        "discover_plugins",
        lambda: events.append("plugins"),
    )

    def fail_config_load():
        events.append("config")
        raise RuntimeError("stop before provider setup")

    monkeypatch.setattr(hermes_cli.config, "load_config", fail_config_load)

    with pytest.raises(RuntimeError, match="stop before provider setup"):
        oneshot._run_agent("test prompt")

    assert events.index("plugins") < events.index("config", events.index("plugins"))


def test_profile_manager_discovery_is_idempotent_and_isolated(tmp_path):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    import hermes_cli.plugins as plugins

    home_a = tmp_path / "profile-a"
    home_b = tmp_path / "profile-b"
    for home, marker in ((home_a, "a"), (home_b, "b")):
        plugin_dir = home / "plugins" / "marker"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "plugin.yaml").write_text(
            "name: marker\nversion: '1.0.0'\n"
        )
        (plugin_dir / "__init__.py").write_text(
            f"def register(ctx):\n    ctx._manager._plugin_skills['marker'] = {marker!r}\n"
        )
        (home / "config.yaml").write_text(
            "plugins:\n  enabled:\n    - marker\n"
        )

    plugins._reset_plugin_managers_for_tests()

    token_a = set_hermes_home_override(str(home_a))
    try:
        manager_a = plugins.get_plugin_manager()
        manager_a.discover_and_load()
        loaded_plugin_count = len(manager_a._plugins)
        manager_a.discover_and_load()
        assert manager_a._plugin_skills["marker"] == "a"
        assert len(manager_a._plugins) == loaded_plugin_count
    finally:
        reset_hermes_home_override(token_a)

    token_b = set_hermes_home_override(str(home_b))
    try:
        manager_b = plugins.get_plugin_manager()
        manager_b.discover_and_load()
        assert manager_b is not manager_a
        assert manager_b._plugin_skills["marker"] == "b"
    finally:
        reset_hermes_home_override(token_b)

    plugins._reset_plugin_managers_for_tests()


def test_oneshot_session_start_is_once_before_turn_with_normalized_identity():
    from hermes_cli import oneshot

    events = []
    lifecycle = SimpleNamespace(
        invoke_hook=lambda name, **kwargs: events.append((name, kwargs)),
        finalize_session=lambda **kwargs: events.append(("finalize", kwargs)),
    )
    agent = SimpleNamespace(
        session_id="oneshot-session",
        platform="cli",
        model="normalized-model",
        provider="normalized-provider",
    )

    def run_conversation(_prompt):
        lifecycle.invoke_hook("on_session_start")
        events.append(("turn", {}))
        return {"final_response": "done"}

    agent.run_conversation = run_conversation

    result = oneshot._run_oneshot_conversation(agent, "hello", lifecycle)

    assert result["final_response"] == "done"
    assert [event[0] for event in events] == ["on_session_start", "turn"]
    assert events[0][1] == {
        "session_id": "oneshot-session",
        "platform": "cli",
        "model": "normalized-model",
        "provider": "normalized-provider",
    }


def test_oneshot_success_emits_end_then_finalize_with_shutdown_reason():
    from hermes_cli import oneshot

    events = []
    lifecycle = SimpleNamespace(
        invoke_hook=lambda name, **kwargs: events.append((name, kwargs)),
        finalize_session=lambda **kwargs: events.append(("on_session_finalize", kwargs)),
    )
    agent = SimpleNamespace(
        session_id="oneshot-session",
        platform="cli",
        model="model",
        provider="provider",
    )

    oneshot._notify_successful_oneshot_lifecycle(
        agent, {"final_response": "done"}, lifecycle
    )

    assert [event[0] for event in events] == [
        "on_session_end",
        "on_session_finalize",
    ]
    assert all(event[1]["reason"] == "shutdown" for event in events)
    assert all(event[1]["session_id"] == "oneshot-session" for event in events)


@pytest.mark.parametrize(
    "result",
    [
        {"final_response": "failed", "failed": True},
        {"final_response": "partial", "partial": True},
        {"final_response": "interrupted", "interrupted": True},
    ],
)
def test_oneshot_failed_or_interrupted_run_does_not_finalize(result):
    from hermes_cli import oneshot

    events = []
    lifecycle = SimpleNamespace(
        invoke_hook=lambda name, **kwargs: events.append((name, kwargs)),
        finalize_session=lambda **kwargs: events.append(("on_session_finalize", kwargs)),
    )
    agent = SimpleNamespace(
        session_id="oneshot-session",
        platform="cli",
        model="model",
        provider="provider",
    )

    oneshot._notify_successful_oneshot_lifecycle(agent, result, lifecycle)

    assert events == []
