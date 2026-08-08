"""Tests for Discord's explicit /review multi-specialist pipeline."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.discord.adapter import DiscordAdapter


class _FakeTree:
    def __init__(self):
        self.commands = {}

    def command(self, *, name, description):
        def decorator(fn):
            self.commands[name] = fn
            return fn

        return decorator

    def add_command(self, command):
        self.commands[command.name] = command

    def get_commands(self):
        return [SimpleNamespace(name=name) for name in self.commands]


@pytest.fixture
def adapter():
    instance = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    instance._client = SimpleNamespace(
        tree=_FakeTree(),
        get_channel=MagicMock(return_value=None),
        fetch_channel=AsyncMock(),
        user=SimpleNamespace(id=99999, name="HermesBot"),
    )
    instance._check_slash_authorization = AsyncMock(return_value=True)
    instance._threads = MagicMock()
    return instance


@pytest.mark.anyio
async def test_registers_review_as_explicit_native_slash_command(adapter):
    adapter._handle_review_slash = AsyncMock()

    adapter._register_slash_commands()

    command = adapter._client.tree.commands["review"]
    assert "thread" in adapter._client.tree.commands
    interaction = SimpleNamespace()
    await command(interaction, topic="Review this schema", kanban=True)

    adapter._handle_review_slash.assert_awaited_once_with(
        interaction,
        "Review this schema",
        create_kanban_tasks=True,
    )


@pytest.mark.anyio
async def test_handle_review_slash_creates_thread_and_detaches_pipeline(adapter):
    interaction = SimpleNamespace(
        channel_id=123,
        user=SimpleNamespace(id=42, display_name="Jezza"),
        response=SimpleNamespace(defer=AsyncMock()),
        followup=SimpleNamespace(send=AsyncMock()),
    )
    adapter._create_thread = AsyncMock(
        return_value={"success": True, "thread_id": "555", "thread_name": "review-schema"}
    )
    adapter._run_review_pipeline = AsyncMock()

    await adapter._handle_review_slash(
        interaction,
        "Review this schema",
        create_kanban_tasks=True,
    )
    await asyncio.sleep(0)

    adapter._create_thread.assert_awaited_once()
    assert adapter._create_thread.await_args.kwargs["auto_archive_duration"] == 4320
    interaction.response.defer.assert_awaited_once_with(ephemeral=True)
    interaction.followup.send.assert_awaited_once()
    assert interaction.followup.send.await_args.kwargs["ephemeral"] is True
    assert "<#555>" in interaction.followup.send.await_args.args[0]
    adapter._threads.mark.assert_called_once_with("555")
    adapter._run_review_pipeline.assert_awaited_once_with(
        thread_id="555",
        topic="Review this schema",
        create_kanban_tasks=True,
        invoking_user="42",
        invoking_channel_id="123",
    )


@pytest.mark.anyio
async def test_review_pipeline_runs_five_specialists_and_ten_rebuttals(adapter):
    thread = SimpleNamespace(send=AsyncMock())
    adapter._client.get_channel.return_value = thread
    active_calls = 0
    max_active_calls = 0

    async def fake_review_agent(prompt):
        nonlocal active_calls, max_active_calls
        active_calls += 1
        max_active_calls = max(max_active_calls, active_calls)
        await asyncio.sleep(0)
        active_calls -= 1
        return f"answer: {prompt[:40]}"

    adapter._run_review_agent = AsyncMock(side_effect=fake_review_agent)
    adapter._create_review_kanban_tasks = AsyncMock(return_value=["t_12345678"])

    await adapter._run_review_pipeline(
        thread_id="555",
        topic="Review this schema",
        create_kanban_tasks=True,
        invoking_user="42",
        invoking_channel_id="123",
    )

    assert adapter._run_review_agent.await_count == 16
    calls = [call.args[0] for call in adapter._run_review_agent.await_args_list]
    assert sum("INDEPENDENT PERSPECTIVE" in prompt for prompt in calls) == 5
    assert sum("CROSS-REBUTTAL" in prompt for prompt in calls) == 10
    assert sum("MODERATOR SYNTHESIS" in prompt for prompt in calls) == 1
    assert max_active_calls == 5
    for prompt in calls[:5]:
        assert "Take a position" in prompt
        assert "Top 3 risks or strengths" in prompt
    for prompt in calls[5:15]:
        assert "Stronger case" in prompt
        assert "Preserve from A" in prompt
        assert "Preserve from B" in prompt
    synthesis_prompt = calls[-1]
    for heading in ("Consensus", "Contested", "Rejected", "Action Items"):
        assert f"## {heading}" in synthesis_prompt
    adapter._create_review_kanban_tasks.assert_awaited_once()
    assert thread.send.await_count == 16


@pytest.mark.anyio
async def test_run_review_agent_uses_minimax_m3_chat_subprocess(adapter):
    process = SimpleNamespace(
        communicate=AsyncMock(return_value=(b"specialist output\n", b"")),
        returncode=0,
        kill=MagicMock(),
        wait=AsyncMock(),
    )

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=process)) as spawn:
        output = await adapter._run_review_agent("Inspect the design")

    assert output == "specialist output"
    assert spawn.await_args.args[:5] == (
        "hermes",
        "chat",
        "--model",
        "minimax-m3",
        "--quiet",
    )
    assert "--query" in spawn.await_args.args


@pytest.mark.anyio
async def test_create_review_kanban_tasks_auto_fires_ready_assigned_cards(adapter):
    synthesis = """## Consensus
Ship the validated design.

## Contested
None.

## Rejected
Manual approval.

## Action Items
- Add schema validation
- Cover timeout handling
"""
    outputs = iter(
        [
            json.dumps({"id": "t_11111111", "status": "ready", "assignee": "worker-improve"}),
            json.dumps({"id": "t_22222222", "status": "ready", "assignee": "worker-improve"}),
        ]
    )

    async def fake_exec(*args, **kwargs):
        process = SimpleNamespace(
            communicate=AsyncMock(return_value=(next(outputs).encode(), b"")),
            returncode=0,
        )
        return process

    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec) as spawn:
        created = await adapter._create_review_kanban_tasks(
            synthesis,
            invoking_user="42",
            invoking_channel_id="123",
        )

    assert created == ["t_11111111", "t_22222222"]
    assert spawn.call_count == 2
    for call in spawn.call_args_list:
        args = call.args
        assert args[:3] == ("hermes", "kanban", "create")
        assert args[3] in ("Add schema validation", "Cover timeout handling")
        assert "--assignee" in args
        assert args[args.index("--assignee") + 1] == "worker-improve"
        assert "--initial-status" in args
        assert args[args.index("--initial-status") + 1] == "running"
        assert "--json" in args


@pytest.mark.anyio
async def test_create_review_kanban_tasks_requires_action_items_heading(adapter):
    with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as spawn:
        created = await adapter._create_review_kanban_tasks(
            "## Consensus\n- Do not create this as a task",
            invoking_user="42",
            invoking_channel_id="123",
        )

    assert created == []
    spawn.assert_not_awaited()
