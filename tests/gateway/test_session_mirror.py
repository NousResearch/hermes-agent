"""Gateway behavior contracts for opt-in Kanban session mirrors (issue #116940)."""

from __future__ import annotations

import asyncio
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast

import pytest

from gateway.platforms.event import MessageEvent
from gateway.run_turn import GatewayTurnMixin
from gateway.session import Platform, SessionSource
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_db_session_mirror import list_mirrors


@pytest.fixture
def mirror_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return home


def _enabled_config(*, profiles=("default",), platforms=("telegram",)):
    return {
        "kanban": {
            "session_mirror": {
                "enabled": True,
                "profiles": list(profiles),
                "platforms": list(platforms),
                "mode": "read_only",
                "retention_days": 30,
            },
        },
    }


class _Runner:
    def __init__(self, callback):
        self.callback = callback
        self.model_calls = 0

    def _profile_scope_for_source(self, _source) -> Any:
        return nullcontext()

    async def _run_agent_inner(self, message, context_prompt, history, source, session_id, **kwargs):
        self.model_calls += 1
        return await self.callback(message, source, session_id)


class _ScopedRunner(_Runner):
    def __init__(self, callback, profile_homes):
        super().__init__(callback)
        self.config = type("Config", (), {"multiplex_profiles": True})()
        self.profile_homes = profile_homes

    def _profile_scope_for_source(self, source) -> Any:
        return GatewayTurnMixin._profile_scope_for_source(cast(GatewayTurnMixin, self), source)

    def _profile_scope_key_for_source(self, source):
        return GatewayTurnMixin._profile_scope_key_for_source(cast(GatewayTurnMixin, self), source)

    def _resolve_profile_home_for_source(self, source):
        return self.profile_homes[source.profile]


async def _execute(runner, source, event, message):
    return await GatewayTurnMixin._run_agent(
        runner, message, "", [], source, "session-1", inbound_message_id=event.message_id,
        _gateway_event=event,
    )


def test_gateway_turn_records_real_outcomes_idempotently_without_extra_model_calls(mirror_home, monkeypatch):
    import gateway.run

    monkeypatch.setattr(gateway.run, "_load_gateway_config", lambda: _enabled_config())
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat-1", thread_id="topic-2", profile="default")
    response = {"final_response": "answer", "completed": True}

    async def run(message, _source, _session_id):
        if message == "raise":
            raise RuntimeError("model execution failed")
        if message == "cancel":
            raise asyncio.CancelledError
        return response

    async def scenario():
        runner = _Runner(run)
        event = MessageEvent(text="private user text must not be stored", source=source, message_id="msg-1")
        assert await _execute(runner, source, event, "first") == response
        # A redelivery with the same platform message id reuses its terminal mirror.
        assert await _execute(runner, source, event, "duplicate") == response

        failed = MessageEvent(text="secret failure request", source=source, message_id="msg-failed")
        with pytest.raises(RuntimeError, match="model execution failed"):
            await _execute(runner, source, failed, "raise")

        cancelled = MessageEvent(text="cancel request", source=source, message_id="msg-cancelled")
        with pytest.raises(asyncio.CancelledError):
            await _execute(runner, source, cancelled, "cancel")

        assert runner.model_calls == 4
        with kbc.connect_closing(board="default") as conn:
            rows = list_mirrors(conn, limit=20)
        by_message = {row["message_id"]: row for row in rows}
        assert set(by_message) == {"msg-1", "msg-failed", "msg-cancelled"}
        assert by_message["msg-1"]["status"] == "completed"
        assert by_message["msg-failed"]["status"] == "failed"
        assert by_message["msg-cancelled"]["status"] == "cancelled"
        assert all(row["profile"] == "default" and row["platform"] == "telegram" for row in rows)
        assert all("private user text" not in str(row) and "secret failure request" not in str(row) for row in rows)

    asyncio.run(scenario())


def test_routed_profile_scope_uses_the_matching_profile_allowlist(mirror_home, monkeypatch):
    import gateway.run
    from hermes_constants import get_hermes_home

    default_home = mirror_home
    worker_home = mirror_home / "profiles" / "worker"
    worker_home.mkdir(parents=True)
    configs = {
        str(default_home): _enabled_config(profiles=("default",)),
        str(worker_home): _enabled_config(profiles=("worker",)),
    }
    monkeypatch.setattr(gateway.run, "_load_gateway_config", lambda: configs[str(get_hermes_home())])

    async def run(_message, _source, _session_id):
        return {"completed": True}

    async def scenario():
        runner = _ScopedRunner(run, {"default": default_home, "worker": worker_home})
        for index, profile in enumerate(("default", "worker", "default")):
            source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat-1", profile=profile)
            event = MessageEvent(text="private", source=source, message_id=f"scope-{index}")
            await _execute(runner, source, event, "private")

        with kbc.connect_closing(board="default") as conn:
            rows = list_mirrors(conn)
        assert [(row["profile"], row["message_id"]) for row in rows] == [
            ("default", "scope-2"), ("worker", "scope-1"), ("default", "scope-0"),
        ]
        assert runner.model_calls == 3

    asyncio.run(scenario())


def test_mirror_is_fail_closed_for_disabled_unselected_internal_and_unidentified_events(
    mirror_home, monkeypatch,
):
    import gateway.run

    configs = [
        {"kanban": {"session_mirror": {"enabled": False, "profiles": ["default"], "platforms": ["telegram"]}}},
        _enabled_config(profiles=()),
        _enabled_config(platforms=("discord",)),
        {"kanban": {"session_mirror": {"enabled": True, "profiles": "default", "platforms": ["telegram"]}}},
        {"kanban": {"session_mirror": {"enabled": True, "profiles": ["default"], "platforms": [None]}}},
        {"kanban": {"session_mirror": {"enabled": True, "profiles": ["default"], "platforms": ["telegram"], "retention_days": -1}}},
    ]
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat-1", profile="default")

    async def run(_message, _source, _session_id):
        return {"final_response": "ok", "completed": True}

    async def scenario():
        runner = _Runner(run)
        events = [MessageEvent(text="input", source=source, message_id=f"off-{i}") for i in range(len(configs))]
        events.extend((
            MessageEvent(text="internal", source=source, message_id="internal", internal=True),
            MessageEvent(text="no id", source=source, message_id=None),
        ))
        for index, event in enumerate(events):
            config = configs[index] if index < len(configs) else _enabled_config()
            monkeypatch.setattr(gateway.run, "_load_gateway_config", lambda config=config: config)
            await _execute(runner, source, event, "plain text")

        with kbc.connect_closing(board="default") as conn:
            assert list_mirrors(conn, include_archived=True) == []
        assert runner.model_calls == len(events)

    asyncio.run(scenario())


def test_routed_runtime_profile_and_storage_errors_do_not_block_turn(mirror_home, monkeypatch):
    import gateway.run

    monkeypatch.setattr(gateway.run, "_load_gateway_config", lambda: _enabled_config(profiles=("worker",)))
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat-1", profile="worker")

    async def run(_message, _source, _session_id):
        return {"completed": True}

    async def scenario():
        runner = _Runner(run)
        event = MessageEvent(text="secret", source=source, message_id="routed-id")
        await _execute(runner, source, event, "secret")
        with kbc.connect_closing(board="default") as conn:
            rows = list_mirrors(conn)
        assert len(rows) == 1
        assert rows[0]["profile"] == "worker" and rows[0]["platform"] == "telegram"
        assert "secret" not in str(rows[0])
        monkeypatch.setattr(kbc, "connect_closing", lambda **_kwargs: (_ for _ in ()).throw(OSError("disk unavailable")))
        next_event = MessageEvent(text="private", source=source, message_id="storage-error")
        assert await _execute(runner, source, next_event, "private") == {"completed": True}
        assert runner.model_calls == 2

    asyncio.run(scenario())


def test_followup_mirror_uses_queued_event_identity(mirror_home, monkeypatch):
    import gateway.run

    monkeypatch.setattr(gateway.run, "_load_gateway_config", lambda: _enabled_config())
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat-1", profile="default")

    async def run(_message, _source, _session_id):
        return {"completed": True}

    async def scenario():
        runner = _Runner(run)
        for mid in ("opening-id", "queued-id"):
            event = MessageEvent(text="private", source=source, message_id=mid)
            await _execute(runner, source, event, "private")
        with kbc.connect_closing(board="default") as conn:
            rows = list_mirrors(conn)
        assert {row["message_id"] for row in rows} == {"opening-id", "queued-id"}
        assert runner.model_calls == 2

    asyncio.run(scenario())
