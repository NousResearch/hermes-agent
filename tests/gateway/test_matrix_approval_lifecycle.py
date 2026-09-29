"""Matrix cards follow the core decision and preserve the requesting profile."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent import secret_scope
from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from gateway.run import _profile_runtime_scope
from hermes_constants import get_hermes_home
from plugins.platforms.matrix.adapter import MatrixAdapter
from plugins.platforms.matrix.approval_cards import generate_command_summary
from tools import approval
from tools.approval_gateway_wait import _ApprovalEntry


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["reaction", "interrupted", "session_closed", "timeout", "expired_typed", "disconnect"])
async def test_card_controls_and_terminal_work_remain_with_the_owner(tmp_path, monkeypatch, boundary):
    homes = [tmp_path / "a", tmp_path / "b", tmp_path / "a"]
    adapter = MatrixAdapter(PlatformConfig(enabled=True, token="test", extra={
        "homeserver": "https://matrix.example.org", "allowed_users": "@owner:example.org,@other:example.org",
    }))
    adapter._client = SimpleNamespace(api=SimpleNamespace(session=SimpleNamespace(close=AsyncMock())))
    adapter._send_reaction = AsyncMock(return_value="$reaction")
    adapter._redact_bot_approval_reactions = AsyncMock()
    adapter._send_invalid_reaction_feedback = AsyncMock(return_value=True)
    edits = []
    visible = asyncio.Event()

    async def edit(room, event_id, body, **kwargs):
        edits.append((Path(get_hermes_home()), event_id, body))
        if len(edits) == 3:
            visible.set()
        return SendResult(success=True, message_id="$replacement")

    adapter.edit_message = AsyncMock(side_effect=edit)
    adapter.send = AsyncMock(side_effect=[SendResult(success=True, message_id=f"$card-{i}") for i in range(3)])
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    entries = []
    try:
        for index, home in enumerate(homes):
            home.mkdir(exist_ok=True)
            with _profile_runtime_scope(home, {}):
                entry = _ApprovalEntry({"command": f"rm -rf /tmp/card-{index}"})
                session = f"agent:{home.name}:matrix:room:{index}"
                entries.append((session, entry))
                with approval._lock:
                    approval._gateway_queues[session] = [entry]
                await adapter.send_exec_approval(
                    chat_id="!room:example.org", session_key=session,
                    command=entry.data["command"], description="recursive deletion", allow_session=False,
                    metadata={**entry.data, "requester_user_id": "@owner:example.org", "thread_id": "$root"},
                )

        for index, (session, entry) in enumerate(entries):
            target = f"$card-{index}"
            prompt = adapter._approval_prompts_by_event[target]
            with _profile_runtime_scope(tmp_path / "other", {"GATEWAY_ALLOW_ALL_USERS": "true"}):
                assert not prompt.owner_context.run(adapter._is_authorized_user, "@intruder:example.org")
                await adapter._handle_approval_reaction("!room:example.org", target, "✅", "@other:example.org")
                await adapter._handle_approval_reaction("!wrong:example.org", target, "✅", "@owner:example.org")
                await adapter._handle_approval_reaction("!room:example.org", target, "♾️", "@owner:example.org")
            assert approval.has_blocking_approval(session, entry.approval_id)
            assert prompt.terminal_choice is None
            if boundary == "reaction":
                with _profile_runtime_scope(tmp_path / "other", {}):
                    await adapter._handle_approval_reaction("!room:example.org", target, "✅", "@owner:example.org")
            elif boundary == "expired_typed":
                prompt.expires_at = entry.expires_at = 0
                assert approval.resolve_gateway_approval(session, "once", approval_id=entry.approval_id) == 0
                assert entry.settle is not None
                entry.settle("session_closed")
            elif boundary != "disconnect":
                assert entry.settle is not None
                with approval._lock:
                    approval._gateway_queues.pop(session)
                entry.settle(boundary)

        if boundary == "disconnect":
            await adapter.disconnect()
            assert all(entry.cancelled for _, entry in entries)
        else:
            await asyncio.wait_for(visible.wait(), timeout=2)
            labels = {"reaction": "Approved once", "timeout": "Expired", "expired_typed": "Expired", "interrupted": "Cancelled", "session_closed": "Cancelled"}
            assert [(home, event, labels[boundary] in body) for home, event, body in edits] == [
                (home, f"$card-{index}", True) for index, home in enumerate(homes)
            ]
        assert adapter._approval_prompts_by_event == {}
        assert adapter._approval_prompt_by_session == {}
    finally:
        for session, _ in entries:
            approval.clear_session(session)
        for task in getattr(adapter, "_approval_tasks", set()):
            task.cancel()
        await asyncio.gather(*getattr(adapter, "_approval_tasks", set()), return_exceptions=True)


@pytest.mark.asyncio
async def test_reaction_and_core_completion_retract_each_seeded_reaction_once(monkeypatch):
    monkeypatch.setenv("MATRIX_ALLOWED_USERS", "@owner:example.org")
    adapter = MatrixAdapter(PlatformConfig(enabled=True, token="test", extra={"homeserver": "https://matrix.example.org"}))
    adapter._client = SimpleNamespace()
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="$card"))
    adapter._send_reaction = AsyncMock(side_effect=[f"$seed-{index}" for index in range(4)])
    redactions = []
    adapter._schedule_reaction_redaction = lambda room, event_id, reason="": redactions.append(event_id)
    session = "agent:main:matrix:room:retract"
    entry = _ApprovalEntry({"command": "rm -rf /tmp/card"})
    with approval._lock:
        approval._gateway_queues[session] = [entry]
    retract = adapter._redact_bot_approval_reactions
    retractions = []
    completion_retracted = asyncio.Event()

    async def observed_retract(room_id, prompt):
        await retract(room_id, prompt)
        retractions.append(room_id)
        if len(retractions) == 2:
            completion_retracted.set()

    async def edit(room, event_id, body, **kwargs):
        if not completion_retracted.is_set():
            entry.settle("resolved")
            await asyncio.wait_for(completion_retracted.wait(), timeout=2)
        return SendResult(success=True, message_id="$replacement")

    adapter._redact_bot_approval_reactions = observed_retract
    adapter.edit_message = AsyncMock(side_effect=edit)
    try:
        await adapter.send_exec_approval(
            chat_id="!room:example.org", session_key=session, command=entry.data["command"],
            metadata={**entry.data, "requester_user_id": "@owner:example.org"},
        )
        prompt = adapter._approval_prompts_by_event["$card"]
        await adapter._handle_approval_reaction("!room:example.org", "$card", "✅", "@owner:example.org")
        await prompt.lifecycle_task
        assert (redactions, prompt.terminal_visible, adapter.edit_message.await_count) == (
            ["$seed-0", "$seed-1", "$seed-2", "$seed-3"], True, 1,
        )
    finally:
        approval.clear_session(session)


@pytest.mark.parametrize("policy", ["local_only", "local_preferred", "remote_redacted"])
def test_summary_redacts_every_input_and_output_at_the_auxiliary_boundary(monkeypatch, policy):
    token = "sk-proj-" + "X" * 40
    seen = []

    def call(**kwargs):
        seen.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=f"Summary {token}"))])

    monkeypatch.setattr("agent.auxiliary_client.call_llm", call)
    monkeypatch.setattr("plugins.platforms.matrix.approval_cards._resolve_approval_summary_route", lambda: {
        "provider": "custom", "model": "test-model", "base_url": "http://127.0.0.1:1234/v1", "api_key": "test",
    })
    result = generate_command_summary(command=f"echo {token}", description=f"guard {token}", provider_policy=policy)
    assert result is not None
    assert token not in result
    assert len(seen) == 1
    assert token not in repr(seen[0]["messages"])


@pytest.mark.parametrize("policy", ["local_only", "local_preferred", "remote_redacted"])
def test_summary_uses_each_profiles_configured_auxiliary_client(tmp_path, monkeypatch, policy):
    from tests.fakes.fake_llm_provider import FakeLLMServer, Text, write_hermes_home

    token = "sk-proj-" + "Y" * 40
    with (
        FakeLLMServer(aux=lambda request: Text("Profile A interpretation")) as first,
        FakeLLMServer(aux=lambda request: Text("Profile B interpretation")) as second,
    ):
        homes = [tmp_path / "a", tmp_path / "b"]
        for home, server in zip(homes, (first, second)):
            write_hermes_home(home, server.base_url, extra_config=(
                "auxiliary:\n  transient_retries: 0\n  approval:\n"
                "    provider: custom\n    model: fake-model\n"
                f"    base_url: {server.base_url}\n"
            ))
        results = []
        monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
        for home in (homes[0], homes[1], homes[0]):
            with _profile_runtime_scope(home):
                results.append(generate_command_summary(
                    command=f"echo {token}", description=f"guard {token}", provider_policy=policy,
                ))
        assert results == ["Profile A interpretation", "Profile B interpretation", "Profile A interpretation"]
        assert [len(first.aux_requests()), len(second.aux_requests())] == [2, 1]
        assert [first.main_requests(), second.main_requests()] == [[], []]
        for request in first.aux_requests() + second.aux_requests():
            assert token not in repr(request["messages"])


@pytest.mark.asyncio
@pytest.mark.parametrize("local_primary", [True, False])
async def test_local_preferred_honours_the_configured_remote_deadline(monkeypatch, local_primary):
    from plugins.platforms.matrix.approval_cards import MatrixApprovalSummaryConfig
    from plugins.platforms.matrix.approval_lifecycle import _MatrixApprovalPrompt

    calls = []

    def call(**kwargs):
        calls.append((kwargs["timeout"], kwargs.get("allow_provider_fallback", True)))
        if local_primary and len(calls) == 1:
            raise RuntimeError("local model unavailable")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Prints a marker."))])

    monkeypatch.setattr("agent.auxiliary_client.call_llm", call)
    monkeypatch.setattr("plugins.platforms.matrix.approval_cards._resolve_approval_summary_route", lambda: {
        "provider": "custom", "model": "test-model",
        "base_url": "http://127.0.0.1:1234/v1" if local_primary else "https://llm.example.org/v1",
        "api_key": "test",
    })
    adapter = MatrixAdapter.__new__(MatrixAdapter)
    adapter.edit_message = AsyncMock(return_value=SendResult(success=True, message_id="$replacement"))
    prompt = _MatrixApprovalPrompt("session", "!room:example.org", "$card", command="echo marker")
    config = MatrixApprovalSummaryConfig(
        enabled=True, provider_policy="local_preferred", local_timeout_seconds=40, remote_timeout_seconds=7,
    )
    adapter._schedule_approval_summary(prompt, config)
    task = prompt.summary_task
    assert task is not None
    await task
    assert calls == ([(40, False), (7, True)] if local_primary else [(7, True)])
