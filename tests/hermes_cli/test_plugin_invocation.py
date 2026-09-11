"""Trusted plugin invocation context and command availability contracts."""

from __future__ import annotations

import asyncio
import pytest

from hermes_cli import plugins
from hermes_cli.plugin_invocation import (
    PluginInvocationContext,
    PluginInvocationContextUnavailable,
    _bind_plugin_invocation,
    _new_local_plugin_invocation,
    _revoke_plugin_invocation,
)
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


def _invocation(**overrides) -> PluginInvocationContext:
    values = {
        "_profile": "default",
        "_session_id": "session-1",
        "_platform": "cli",
        "_authenticated_actor": None,
        "_target": "session-1",
        "_chat_id": None,
        "_thread_id": None,
        "_origin": None,
        "_execution_kind": "root",
    }
    values.update({f"_{key}": value for key, value in overrides.items()})
    return PluginInvocationContext(**values)


@pytest.fixture()
def neutral_consumer(monkeypatch):
    manager = PluginManager(scope_key="/tmp/hermes-plugin-invocation-test")
    context = PluginContext(
        PluginManifest(name="neutral-context-consumer", source="user"), manager
    )
    seen = []

    def handler(raw_args):
        invocation = context.invocation
        seen.append(
            (
                raw_args,
                invocation,
                {
                    "session_id": invocation.session_id,
                    "platform": invocation.platform,
                    "authenticated_actor": invocation.authenticated_actor,
                    "execution_kind": invocation.execution_kind,
                },
            )
        )
        return "handled"

    context.register_command(
        "context-probe",
        handler,
        availability=lambda invocation: invocation.platform == "cli",
    )
    monkeypatch.setattr(plugins, "_ensure_plugins_discovered", lambda: manager)
    return context, manager, seen


def test_context_is_immutable_and_unavailable_outside_dispatch(neutral_consumer):
    context, _manager, _seen = neutral_consumer

    with pytest.raises(PluginInvocationContextUnavailable):
        _ = context.invocation
    with pytest.raises((AttributeError, TypeError)):
        _invocation().platform = "gateway"


def test_local_context_uses_refresh_free_host_actor_and_active_profile(monkeypatch):
    import hermes_cli.auth as auth
    import hermes_cli.auth_nous as auth_nous
    import hermes_cli.profiles as profiles

    monkeypatch.setattr(
        auth,
        "get_provider_auth_state",
        lambda provider: {"access_token": "local-jwt", "scope": "inference:invoke"}
        if provider == "nous"
        else None,
    )
    monkeypatch.setattr(auth_nous, "_state_invoke_jwt_status", lambda _state, _token: None)
    monkeypatch.setattr(
        auth_nous, "_decode_jwt_claims", lambda _token: {"sub": "  nas-user-7  "}
    )
    monkeypatch.setattr("hermes_cli.anon_auth.is_guest_state", lambda _state: False)
    monkeypatch.setattr(
        auth,
        "resolve_nous_runtime_credentials",
        lambda *_args, **_kwargs: pytest.fail("discovery must not refresh credentials"),
    )
    monkeypatch.setattr(profiles, "get_active_profile_name", lambda: "work")

    invocation = _new_local_plugin_invocation(
        platform="cli", session_id="local-session"
    )

    assert invocation.profile == "work"
    assert invocation.authenticated_actor == "nas-user-7"
    assert invocation.session_id == "local-session"
    assert invocation.target == "local-session"


def test_local_context_rejects_expired_access_even_with_refresh(monkeypatch):
    import hermes_cli.auth as auth
    import hermes_cli.auth_nous as auth_nous

    state = {"access_token": "expired-jwt", "refresh_token": "still-present"}
    monkeypatch.setattr(auth, "get_provider_auth_state", lambda _provider: state)
    monkeypatch.setattr(
        auth_nous, "_state_invoke_jwt_status", lambda _state, _token: "invoke_jwt_expiring"
    )
    monkeypatch.setattr("hermes_cli.anon_auth.is_guest_state", lambda _state: False)

    invocation = _new_local_plugin_invocation(platform="tui")

    assert invocation.authenticated_actor is None


def test_local_context_rejects_wrong_scope_or_missing_jwt_subject(monkeypatch):
    import hermes_cli.auth as auth
    import hermes_cli.auth_nous as auth_nous

    monkeypatch.setattr(
        auth,
        "get_provider_auth_state",
        lambda _provider: {"access_token": "local-jwt", "scope": "wrong:scope"},
    )
    monkeypatch.setattr(
        auth_nous,
        "_state_invoke_jwt_status",
        lambda _state, _token: "missing_inference_invoke_scope",
    )
    monkeypatch.setattr(auth_nous, "_decode_jwt_claims", lambda _token: {"sub": "nas-user-7"})
    monkeypatch.setattr("hermes_cli.anon_auth.is_guest_state", lambda _state: False)

    wrong_scope = _new_local_plugin_invocation(platform="cli")
    assert wrong_scope.authenticated_actor is None

    monkeypatch.setattr(auth_nous, "_state_invoke_jwt_status", lambda _state, _token: None)
    monkeypatch.setattr(auth_nous, "_decode_jwt_claims", lambda _token: {})
    missing_subject = _new_local_plugin_invocation(platform="cli")
    assert missing_subject.authenticated_actor is None


def test_local_context_rejects_guest_or_mismatched_subject(monkeypatch):
    import hermes_cli.auth as auth
    import hermes_cli.auth_nous as auth_nous

    state = {"access_token": "local-jwt", "user_id": "different-user"}
    monkeypatch.setattr(auth, "get_provider_auth_state", lambda _provider: state)
    monkeypatch.setattr(auth_nous, "_state_invoke_jwt_status", lambda _state, _token: None)
    monkeypatch.setattr(
        auth_nous, "_decode_jwt_claims", lambda _token: {"sub": "nas-user-7"}
    )
    monkeypatch.setattr("hermes_cli.anon_auth.is_guest_state", lambda _state: True)
    assert _new_local_plugin_invocation(platform="cli").authenticated_actor is None

    monkeypatch.setattr("hermes_cli.anon_auth.is_guest_state", lambda _state: False)
    assert _new_local_plugin_invocation(platform="cli").authenticated_actor is None


def test_local_context_fails_closed_on_state_reader_error(monkeypatch):
    import hermes_cli.auth as auth

    def fail(_provider):
        raise OSError("unreadable auth store")

    monkeypatch.setattr(auth, "get_provider_auth_state", fail)

    assert _new_local_plugin_invocation(platform="cli").authenticated_actor is None


def test_local_context_does_not_cross_profile_scope(monkeypatch):
    import hermes_cli.profiles as profiles
    import hermes_cli.plugin_invocation as invocation_mod

    monkeypatch.setattr(profiles, "get_active_profile_name", lambda: "work")
    monkeypatch.setattr(invocation_mod, "_local_authenticated_actor", lambda: "nas-user-7")

    invocation = _new_local_plugin_invocation(platform="tui", profile="personal")

    assert invocation.profile == "personal"
    assert invocation.authenticated_actor is None


def test_availability_filters_discovery_and_dispatch(neutral_consumer):
    _context, _manager, _seen = neutral_consumer
    allowed = _invocation()
    denied = _invocation(platform="telegram")

    assert "context-probe" not in plugins.get_plugin_commands()
    assert "context-probe" not in plugins.get_plugin_commands(denied)
    assert plugins.get_plugin_command_handler("context-probe", denied) is None
    assert "context-probe" in plugins.get_plugin_commands(allowed)
    assert plugins.get_plugin_command_handler("context-probe", allowed) is not None
    assert plugins.is_plugin_command_registered("context-probe") is True


def test_expired_context_denies_before_always_true_predicate(monkeypatch):
    manager = PluginManager(scope_key="/tmp/hermes-plugin-expired-test")
    context = PluginContext(PluginManifest(name="neutral-consumer", source="user"), manager)
    predicate_calls = []

    def always_true(invocation):
        predicate_calls.append(invocation)
        return True

    context.register_command("expired", lambda _raw: None, availability=always_true)
    monkeypatch.setattr(plugins, "_ensure_plugins_discovered", lambda: manager)
    invocation = _invocation()
    _revoke_plugin_invocation(invocation)

    assert plugins.get_plugin_commands(invocation) == {}
    assert plugins.get_plugin_command_handler("expired", invocation) is None
    assert predicate_calls == []


def test_bound_handler_sees_exact_context_and_binding_resets(neutral_consumer):
    context, _manager, seen = neutral_consumer
    invocation = _invocation(authenticated_actor="user-7")
    handler = plugins.get_plugin_command_handler("context-probe", invocation)

    with _bind_plugin_invocation(invocation):
        assert handler("raw ARG") == "handled"
        assert context.invocation is invocation

    assert seen == [
        (
            "raw ARG",
            invocation,
            {
                "session_id": "session-1",
                "platform": "cli",
                "authenticated_actor": "user-7",
                "execution_kind": "root",
            },
        )
    ]
    with pytest.raises(PluginInvocationContextUnavailable, match="expired"):
        _ = invocation.session_id
    with pytest.raises(PluginInvocationContextUnavailable):
        _ = context.invocation


def test_predicate_exception_and_awaitable_fail_closed(monkeypatch, caplog):
    manager = PluginManager(scope_key="/tmp/hermes-plugin-predicate-test")
    context = PluginContext(PluginManifest(name="neutral-consumer", source="user"), manager)
    context.register_command(
        "raises", lambda _raw: None, availability=lambda _invocation: 1 / 0
    )

    async def async_predicate(_invocation):
        return True

    context.register_command("async", lambda _raw: None, availability=async_predicate)
    monkeypatch.setattr(plugins, "_ensure_plugins_discovered", lambda: manager)

    assert plugins.get_plugin_commands(_invocation()) == {}
    assert "command availability failed (ZeroDivisionError)" in caplog.text
    assert "division by zero" not in caplog.text
    assert "Traceback" not in caplog.text
    assert "returned an awaitable" in caplog.text


def test_non_callable_predicate_is_rejected():
    manager = PluginManager(scope_key="/tmp/hermes-plugin-predicate-type-test")
    context = PluginContext(PluginManifest(name="neutral-consumer", source="user"), manager)

    with pytest.raises(TypeError, match="availability must be callable"):
        context.register_command("bad", lambda _raw: None, availability=True)


def test_legacy_command_without_predicate_is_unchanged(monkeypatch):
    manager = PluginManager(scope_key="/tmp/hermes-plugin-legacy-test")
    context = PluginContext(PluginManifest(name="legacy-consumer", source="user"), manager)
    handler = lambda raw: raw
    context.register_command("legacy", handler)
    monkeypatch.setattr(plugins, "_ensure_plugins_discovered", lambda: manager)

    assert plugins.get_plugin_commands()["legacy"]["handler"] is handler
    assert plugins.get_plugin_command_handler("legacy") is handler


def test_binding_is_isolated_between_async_tasks(neutral_consumer):
    context, _manager, _seen = neutral_consumer

    async def observe(invocation):
        with _bind_plugin_invocation(invocation):
            await asyncio.sleep(0)
            return context.invocation.session_id

    async def run_observers():
        return await asyncio.gather(
            observe(_invocation(session_id="first")),
            observe(_invocation(session_id="second")),
        )

    first, second = asyncio.run(run_observers())

    assert first == "first"
    assert second == "second"


def test_spawn_task_does_not_inherit_command_authority(neutral_consumer):
    context, _manager, _seen = neutral_consumer

    async def observe_background():
        await asyncio.sleep(0)
        with pytest.raises(PluginInvocationContextUnavailable):
            _ = context.invocation
        return "background"

    async def run_background():
        with _bind_plugin_invocation(_invocation()):
            task = context.spawn_task(observe_background(), name="neutral-background")
        return await task

    assert asyncio.run(run_background()) == "background"


def test_async_result_helper_thread_preserves_context(neutral_consumer):
    context, _manager, _seen = neutral_consumer
    invocation = _invocation(session_id="threaded")

    async def async_handler():
        await asyncio.sleep(0)
        return context.invocation

    async def run_handler():
        with _bind_plugin_invocation(invocation):
            return plugins.resolve_plugin_command_result(async_handler())

    observed = asyncio.run(run_handler())

    assert observed is invocation


def test_cli_command_dispatch_binds_trusted_context(neutral_consumer, monkeypatch):
    import cli as cli_mod

    context, _manager, seen = neutral_consumer
    cli = object.__new__(cli_mod.HermesCLI)
    cli.session_id = "cli-session"
    cli.config = {"quick_commands": {}}
    monkeypatch.setattr(cli_mod, "_ensure_skill_commands", lambda: {})
    monkeypatch.setattr(cli_mod, "get_skill_bundles", lambda: {})
    monkeypatch.setattr(cli_mod, "_cprint", lambda _text: None)

    assert cli._process_unregistered_slash("/context-probe MiXeD", "/context-probe mixed")

    raw_args, invocation, snapshot = seen.pop()
    assert raw_args == "MiXeD"
    assert snapshot == {
        "session_id": "cli-session",
        "platform": "cli",
        "authenticated_actor": None,
        "execution_kind": "root",
    }
    with pytest.raises(PluginInvocationContextUnavailable, match="expired"):
        _ = invocation.session_id
    with pytest.raises(PluginInvocationContextUnavailable):
        _ = context.invocation


def test_cli_unavailable_registered_command_stops_as_unknown(neutral_consumer, monkeypatch):
    import cli as cli_mod

    _context, _manager, seen = neutral_consumer
    cli = object.__new__(cli_mod.HermesCLI)
    cli.session_id = "cli-session"
    cli.config = {"quick_commands": {}}
    cli._expand_slash_prefix = lambda *_args: "unknown"
    monkeypatch.setattr(cli_mod, "_ensure_skill_commands", lambda: {})
    monkeypatch.setattr(cli_mod, "get_skill_bundles", lambda: {})
    monkeypatch.setattr(
        cli_mod,
        "_cli_plugin_invocation",
        lambda _cli: _invocation(platform="tui"),
    )

    assert cli._process_unregistered_slash("/context-probe", "/context-probe") == "unknown"
    assert seen == []
