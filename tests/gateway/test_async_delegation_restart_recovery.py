"""Gateway wiring for durable async-delegation restart recovery."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig
from gateway.run import GatewayRunner, _format_gateway_process_notification
from tools.process_registry import format_process_notification


def _record():
    return {
        "delegation_id": "deleg_resume",
        "profile": "default",
        "source": {
            "kind": "single",
            "tasks": [{"goal": "finish report", "context": "draft exists", "role": "leaf"}],
        },
        "execution": {
            "model": "persisted-model",
            "provider": "custom",
            "base_url": "https://example.invalid/v1",
            "api_mode": "chat_completions",
            "toolsets": ["file"],
            "max_iterations": 40,
            "parent_depth": 0,
            "credential_ref": {"provider": "named-provider"},
        },
        "route": {
            "platform": "telegram",
            "session_key": "agent:main:telegram:dm:123",
            "parent_session_id": "sess-parent",
        },
        "attempt": {"generation": 1, "redispatch_count": 1},
    }


def test_restart_event_uses_existing_process_notification_formatter():
    event = {
        "type": "async_delegation_restarted",
        "delegation_id": "deleg_resume",
        "attempt_generation": 1,
        "redispatch_count": 1,
        "goals": ["finish report"],
    }
    shared = format_process_notification(event)
    gateway = _format_gateway_process_notification(event)
    assert gateway is not None
    assert gateway == shared
    assert "RESTARTED" in gateway
    assert "recovery launch 1/2" in gateway


def test_gateway_recovery_runs_once_and_reresolves_current_credentials(tmp_path, monkeypatch):
    runner = object.__new__(GatewayRunner)
    runner.__dict__["config"] = SimpleNamespace(multiplex_profiles=False)
    runner._current_boot_id = MagicMock(return_value="200:2.0")
    built_parent = MagicMock()
    returned_runner = MagicMock()

    def recover_side_effect(**kwargs):
        produced = kwargs["runner_factory"](_record(), "CONTINUE after restart")
        assert produced[0] is returned_runner
        produced[1]()
        return {"claimed": 1, "exhausted": 0, "failed_validation": 0}

    runtime = {
        "api_key": "CURRENT-KEY",
        "base_url": "https://current.invalid/v1",
        "provider": "custom",
        "api_mode": "chat_completions",
        "command": None,
        "args": [],
        "credential_pool": None,
    }

    with patch("hermes_cli.profiles.profiles_to_serve", return_value=[("default", Path(tmp_path))]), \
         patch("hermes_cli.config.load_config", return_value={"delegation": {"resume_on_restart": True}}), \
         patch("tools.async_delegation.recover_async_delegations", side_effect=recover_side_effect) as recover, \
         patch("tools.async_delegation.enqueue_pending_outbox", return_value=2) as replay, \
         patch("gateway.run._resolve_runtime_agent_kwargs_for_provider", return_value=runtime) as resolve, \
         patch("run_agent.AIAgent", return_value=built_parent) as agent_cls, \
         patch("tools.delegate_tool.build_recovered_delegation_runner", return_value=returned_runner) as build:
        first = runner._recover_async_delegations_once()
        second = runner._recover_async_delegations_once()

    assert first == {"claimed": 1, "queued": 2, "exhausted": 0, "failed_validation": 0}
    assert second == {"claimed": 0, "queued": 0}
    recover.assert_called_once()
    replay.assert_called_once()
    resolve.assert_called_once_with("named-provider")
    assert agent_cls.call_args.kwargs["api_key"] == "CURRENT-KEY"
    assert agent_cls.call_args.kwargs["model"] == "persisted-model"
    built_parent.interrupt.assert_called_once()
    build.assert_called_once_with(_record(), "CONTINUE after restart", built_parent)


@pytest.mark.asyncio
async def test_gateway_start_runs_recovery_with_no_connected_adapters(
    tmp_path, monkeypatch
):
    """A degraded boot gets its one recovery pass; reconnect must not own it."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runner = GatewayRunner(
        GatewayConfig(platforms={}, sessions_dir=tmp_path / "sessions")
    )
    recover = MagicMock(return_value={"claimed": 0, "queued": 0})
    monkeypatch.setattr(runner, "_recover_async_delegations_once", recover)

    assert await runner.start() is True

    recover.assert_called_once_with()


def test_gateway_recovery_continues_after_one_profile_registry_fails(tmp_path):
    runner = object.__new__(GatewayRunner)
    runner.__dict__["config"] = SimpleNamespace(multiplex_profiles=True)
    runner._current_boot_id = MagicMock(return_value="200:2.0")
    homes = [tmp_path / "broken", tmp_path / "healthy"]

    with patch(
        "hermes_cli.profiles.profiles_to_serve",
        return_value=[("broken", homes[0]), ("healthy", homes[1])],
    ), patch(
        "hermes_cli.config.load_config",
        return_value={"delegation": {"resume_on_restart": True}},
    ), patch(
        "tools.async_delegation.recover_async_delegations",
        side_effect=[RuntimeError("corrupt registry"), {"claimed": 1}],
    ) as recover, patch(
        "tools.async_delegation.enqueue_pending_outbox", return_value=1
    ) as replay:
        summary = runner._recover_async_delegations_once()

    assert summary == {"claimed": 1, "queued": 1, "exhausted": 0, "failed_validation": 0}
    assert recover.call_count == 2
    replay.assert_called_once()


def test_gateway_recovery_reresolves_direct_endpoint_key_from_current_config(
    tmp_path,
):
    runner = object.__new__(GatewayRunner)
    runner.__dict__["config"] = SimpleNamespace(multiplex_profiles=False)
    runner._current_boot_id = MagicMock(return_value="200:2.0")
    record = _record()
    record["execution"]["base_url"] = "https://persisted-direct.invalid/v1"
    record["execution"]["credential_ref"] = {
        "source": "delegation_config",
        "parent_provider": "openrouter",
    }
    built_parent = MagicMock()
    returned_runner = MagicMock()

    def recover_side_effect(**kwargs):
        kwargs["runner_factory"](record, "CONTINUE after restart")
        return {"claimed": 1, "exhausted": 0, "failed_validation": 0}

    with patch(
        "hermes_cli.profiles.profiles_to_serve",
        return_value=[("default", Path(tmp_path))],
    ), patch(
        "hermes_cli.config.load_config",
        return_value={
            "delegation": {
                "resume_on_restart": True,
                "api_key": "CURRENT-DIRECT-KEY",
            }
        },
    ), patch(
        "tools.async_delegation.recover_async_delegations",
        side_effect=recover_side_effect,
    ), patch(
        "tools.async_delegation.enqueue_pending_outbox", return_value=0
    ), patch(
        "gateway.run._resolve_runtime_agent_kwargs_for_provider",
        side_effect=AssertionError("configured direct key must not resolve parent provider"),
    ) as resolve, patch(
        "run_agent.AIAgent", return_value=built_parent
    ) as agent_cls, patch(
        "tools.delegate_tool.build_recovered_delegation_runner",
        return_value=returned_runner,
    ):
        summary = runner._recover_async_delegations_once()

    assert summary["claimed"] == 1
    resolve.assert_not_called()
    assert agent_cls.call_args.kwargs["api_key"] == "CURRENT-DIRECT-KEY"
    assert agent_cls.call_args.kwargs["base_url"] == "https://persisted-direct.invalid/v1"


def test_process_event_source_does_not_reuse_wrong_profile_origin():
    from gateway.platforms.base import Platform
    from gateway.session import SessionSource

    runner = object.__new__(GatewayRunner)
    wrong = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_type="dm",
        profile="default",
    )
    runner.session_store = MagicMock()
    runner.session_store._ensure_loaded = MagicMock()
    runner.session_store._entries = {
        "agent:main:telegram:dm:123": SimpleNamespace(origin=wrong)
    }
    runner._get_cached_session_source = MagicMock(return_value=wrong)

    source = runner._build_process_event_source({
        "type": "async_delegation",
        "session_key": "agent:main:telegram:dm:123",
        "platform": "telegram",
        "chat_type": "dm",
        "chat_id": "123",
        "profile": "work",
    })

    assert source is not None
    assert source.profile == "work"


@pytest.mark.asyncio
async def test_async_injection_ack_outcome_distinguishes_delivery_from_ended_session():
    from gateway.session import SessionSource
    from gateway.platforms.base import Platform

    runner = object.__new__(GatewayRunner)
    adapter = MagicMock()
    adapter.handle_message = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._profile_adapters = {}
    runner.session_store = MagicMock()
    runner.session_store._ensure_loaded = MagicMock()
    runner.session_store._entries = {
        "agent:main:telegram:dm:123": SimpleNamespace(
            origin=SessionSource(
                platform=Platform.TELEGRAM,
                chat_id="123",
                chat_type="dm",
            )
        )
    }
    current = SimpleNamespace(session_id="sess-parent")
    runner.__dict__["_async_session_store"] = SimpleNamespace(
        _store=runner.session_store,
        get_or_create_session=AsyncMock(return_value=current),
    )
    event = {
        "type": "async_delegation",
        "session_key": "agent:main:telegram:dm:123",
        "parent_session_id": "sess-parent",
    }

    assert await runner._inject_watch_notification("done", event) == "delivered"
    adapter.handle_message.assert_awaited_once()

    current.session_id = "sess-new"
    assert await runner._inject_watch_notification("done", event) == "dropped"
    assert adapter.handle_message.await_count == 1
