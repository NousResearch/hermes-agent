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
                assert not adapter._is_authorized_user("@intruder:example.org")
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
                assert entry.settle is not None and entry.result is None
                with approval._lock:
                    approval._gateway_queues.pop(session)
                entry.settle("timeout")
            elif boundary != "disconnect":
                assert entry.settle is not None
                with approval._lock:
                    approval._gateway_queues.pop(session)
                entry.settle(boundary)

        if boundary == "disconnect":
            await adapter.disconnect()
            assert all(entry.cancelled for _, entry in entries)
        await asyncio.wait_for(visible.wait(), timeout=2)
        labels = {
            "reaction": "Approved once", "timeout": "Expired", "expired_typed": "Expired",
            "interrupted": "Cancelled", "session_closed": "Cancelled", "disconnect": "Cancelled",
        }
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
    prompt = _MatrixApprovalPrompt("session", "!room:example.org", "$card", command="echo marker", approval_id="approval-1")
    config = MatrixApprovalSummaryConfig(
        enabled=True, provider_policy="local_preferred", local_timeout_seconds=40, remote_timeout_seconds=7,
    )
    adapter._schedule_approval_summary(prompt, config)
    task = prompt.summary_task
    assert task is not None
    await task
    assert calls == ([(40, False), (7, True)] if local_primary else [(7, True)])


class _FakeClock:
    """Monotonic time that advances only when the card lifecycle sleeps."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        self.sleeps.append(delay)
        self.now += delay


async def _card_with_failing_edits(monkeypatch, session: str, send_event: AsyncMock):
    monkeypatch.setenv("MATRIX_ALLOWED_USERS", "@owner:example.org")
    adapter = MatrixAdapter(PlatformConfig(enabled=True, token="test", extra={"homeserver": "https://matrix.example.org"}))
    adapter._client = SimpleNamespace(send_message_event=send_event)
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="$card"))
    adapter._send_reaction = AsyncMock(return_value="$seed")
    adapter._schedule_reaction_redaction = lambda *args, **kwargs: None
    clock = _FakeClock()
    adapter._approval_clock, adapter._approval_sleep = clock.monotonic, clock.sleep
    entry = _ApprovalEntry({"command": "rm -rf /tmp/card"})
    with approval._lock:
        approval._gateway_queues[session] = [entry]
    await adapter.send_exec_approval(
        chat_id="!room:example.org", session_key=session, command=entry.data["command"],
        metadata={**entry.data, "requester_user_id": "@owner:example.org"},
    )
    return adapter, adapter._approval_prompts_by_event["$card"], clock


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["forbidden", "not_found", "over_local_limit"])
async def test_permanent_terminal_edit_failure_stops_retrying(monkeypatch, caplog, failure):
    from mautrix.errors import MForbidden, MNotFound

    errors = {"forbidden": MForbidden(403, "You are not in this room"), "not_found": MNotFound(404, "Unknown event")}
    send_event = AsyncMock(side_effect=errors.get(failure))
    session = f"agent:main:matrix:room:{failure}"
    adapter, prompt, clock = await _card_with_failing_edits(monkeypatch, session, send_event)
    if failure == "over_local_limit":
        adapter.max_message_length = 40
    try:
        await asyncio.wait_for(adapter._complete_matrix_approval(prompt, "timeout"), timeout=2)
        gave_up = [record.getMessage() for record in caplog.records if record.levelname == "ERROR"]
        assert (send_event.await_count, clock.sleeps, prompt.terminal_visible, len(gave_up)) == (
            0 if failure == "over_local_limit" else 1, [], False, 1,
        )
    finally:
        approval.clear_session(session)


@pytest.mark.asyncio
async def test_transient_terminal_edit_failure_backs_off_and_gives_up(monkeypatch, caplog):
    from plugins.platforms.matrix import approval_lifecycle

    send_event = AsyncMock(side_effect=ConnectionResetError("Connection reset by peer"))
    session = "agent:main:matrix:room:transient"
    adapter, prompt, clock = await _card_with_failing_edits(monkeypatch, session, send_event)
    retry = approval_lifecycle._TERMINAL_EDIT_RETRY
    try:
        await asyncio.wait_for(adapter._complete_matrix_approval(prompt, "timeout"), timeout=2)
        gave_up = [record.getMessage() for record in caplog.records if record.levelname == "ERROR"]
        assert (
            send_event.await_count == len(clock.sleeps) + 1,
            clock.sleeps == sorted(clock.sleeps),
            clock.sleeps[0] < clock.sleeps[-1] == retry.max_delay,
            sum(clock.sleeps) <= retry.horizon,
            prompt.terminal_visible,
            len(gave_up),
        ) == (True, True, True, True, False, 1)
    finally:
        approval.clear_session(session)


@pytest.mark.asyncio
async def test_card_claims_reactions_until_its_terminal_edit_lands(monkeypatch):
    from mautrix.errors import MForbidden

    send_event = AsyncMock(side_effect=MForbidden(403, "You are not in this room"))
    session = "agent:main:matrix:room:claim"
    adapter, prompt, _clock = await _card_with_failing_edits(monkeypatch, session, send_event)
    try:
        await adapter._complete_matrix_approval(prompt, "timeout")
        claimed = await adapter._handle_approval_reaction("!room:example.org", "$card", "✅", "@owner:example.org")
        assert (claimed, prompt.terminal_visible, send_event.await_count) == (True, False, 1)
    finally:
        approval.clear_session(session)


@pytest.mark.asyncio
async def test_disconnect_bounds_the_cancelled_card_edits(monkeypatch):
    never = asyncio.Event()

    async def hang(*args, **kwargs):
        await never.wait()

    session = "agent:main:matrix:room:close"
    adapter, prompt, _clock = await _card_with_failing_edits(monkeypatch, session, AsyncMock(side_effect=hang))
    adapter._approval_close_timeout = 0.05
    entry = approval._gateway_queues[session][0]
    try:
        await asyncio.wait_for(adapter._close_matrix_approvals(), timeout=1)
        assert (bool(entry.cancelled), prompt.terminal_visible, adapter._approval_prompts_by_event) == (True, False, {})
    finally:
        approval.clear_session(session)


@pytest.mark.asyncio
async def test_card_without_approval_id_never_resolves_another_request(monkeypatch):
    monkeypatch.setenv("MATRIX_ALLOWED_USERS", "@owner:example.org")
    adapter = MatrixAdapter(PlatformConfig(enabled=True, token="test", extra={"homeserver": "https://matrix.example.org"}))
    adapter._client = SimpleNamespace()
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="$card"))
    adapter._send_reaction = AsyncMock(return_value="$seed")
    adapter.edit_message = AsyncMock(return_value=SendResult(success=True, message_id="$edit"))
    session = "agent:main:matrix:room:no-id"
    older, newer = _ApprovalEntry({"command": "rm -rf /older"}), _ApprovalEntry({"command": "rm -rf /newer"})
    with approval._lock:
        approval._gateway_queues[session] = [older, newer]
    try:
        sent = await adapter.send_exec_approval(
            chat_id="!room:example.org", session_key=session, command="rm -rf /newer",
            metadata={"requester_user_id": "@owner:example.org"},
        )
        reacted = await adapter._handle_approval_reaction("!room:example.org", "$card", "✅", "@owner:example.org")
        assert (sent.success, reacted, older.result, newer.result) == (False, False, None, None)
    finally:
        approval.clear_session(session)
        for task in getattr(adapter, "_approval_tasks", set()):
            task.cancel()


@pytest.mark.asyncio
@pytest.mark.parametrize("notice", ["edit_failed", "expired"])
async def test_card_notices_stay_in_the_card_thread(monkeypatch, notice):
    monkeypatch.setenv("MATRIX_ALLOWED_USERS", "@owner:example.org")
    adapter = MatrixAdapter(PlatformConfig(enabled=True, token="test", extra={"homeserver": "https://matrix.example.org"}))
    adapter._client = SimpleNamespace()
    sent = []

    async def send_room_message(room_id, content):
        sent.append(content)
        return f"$event-{len(sent)}"

    adapter._send_room_message = send_room_message
    adapter._send_reaction = AsyncMock(return_value="$seed")
    adapter._schedule_reaction_redaction = lambda *args, **kwargs: None
    adapter.edit_message = AsyncMock(return_value=SendResult(success=notice == "expired", message_id="$edit", error="offline"))
    session = f"agent:main:matrix:room:thread-{notice}"
    entry = _ApprovalEntry({"command": "rm -rf /tmp/card"})
    with approval._lock:
        approval._gateway_queues[session] = [entry]
    try:
        await adapter.send_exec_approval(
            chat_id="!room:example.org", session_key=session, command=entry.data["command"],
            metadata={**entry.data, "requester_user_id": "@owner:example.org", "thread_id": "$root"},
        )
        prompt = adapter._approval_prompts_by_event["$event-1"]
        if notice == "expired":
            prompt.expires_at = entry.expires_at = 0
            await adapter._handle_approval_reaction("!room:example.org", "$event-1", "✅", "@owner:example.org")
        else:
            await adapter._finalize_matrix_approval_prompt("!room:example.org", "$event-1", prompt, choice="deny")
        assert [content["m.relates_to"] for content in sent[1:]] == [{
            "rel_type": "m.thread", "event_id": "$root", "is_falling_back": True,
            "m.in_reply_to": {"event_id": "$event-1"},
        }]
    finally:
        approval.clear_session(session)
        for task in getattr(adapter, "_approval_tasks", set()):
            task.cancel()
