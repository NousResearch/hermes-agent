"""Security and completion-truth tests for the canary API control surface."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.canonical_writer_boundary import trusted_runtime_envelope
from gateway.canonical_writer_handlers import (
    RuntimeContext,
    _require_exact_runtime_epoch,
)
from gateway.platforms.api_server import (
    APIServerAdapter,
    _effective_uid_for_systemd_credential,
    _session_stream_outcome,
)
from gateway.session_context import clear_session_vars
from hermes_state import SessionDB


def _credential_config() -> PlatformConfig:
    return PlatformConfig(
        enabled=True,
        extra={
            "host": "127.0.0.1",
            "port": 8642,
            "key_credential": "api-server.key",
        },
    )


def _write_credential(directory: Path, value: bytes = b"canary-control-key\n") -> Path:
    path = directory / "api-server.key"
    path.write_bytes(value)
    path.chmod(0o600)
    return path


def _session_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app.router.add_post("/api/sessions", adapter._handle_create_session)
    app.router.add_post(
        "/api/sessions/{session_id}/chat",
        adapter._handle_session_chat,
    )
    app.router.add_post(
        "/api/sessions/{session_id}/chat/stream",
        adapter._handle_session_chat_stream,
    )
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    app.router.add_post("/v1/responses", adapter._handle_responses)
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    app.router.add_get("/v1/runs/{run_id}/events", adapter._handle_run_events)
    app.router.add_post("/v1/runs/{run_id}/stop", adapter._handle_stop_run)
    return app


@pytest.mark.asyncio
async def test_systemd_credential_starts_authenticated_loopback_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credential_dir = tmp_path / "credentials"
    credential_dir.mkdir()
    _write_credential(credential_dir)
    monkeypatch.setenv("CREDENTIALS_DIRECTORY", str(credential_dir))
    monkeypatch.delenv("API_SERVER_KEY", raising=False)

    adapter = APIServerAdapter(_credential_config())
    adapter._session_db = SessionDB(tmp_path / "state.db")
    app = _session_app(adapter)
    async with TestClient(TestServer(app)) as client:
        denied = await client.post("/api/sessions", json={})
        assert denied.status == 401
        accepted = await client.post(
            "/api/sessions",
            json={},
            headers={"Authorization": "Bearer canary-control-key"},
        )
        assert accepted.status == 201
    adapter._session_db.close()


@pytest.mark.parametrize("credential_name", ["../key", "/tmp/key", "nested/key", ""])
def test_systemd_credential_rejects_non_basename(
    credential_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CREDENTIALS_DIRECTORY", str(tmp_path))
    monkeypatch.delenv("API_SERVER_KEY", raising=False)
    config = PlatformConfig(
        enabled=True,
        extra={"key_credential": credential_name},
    )
    with pytest.raises(ValueError, match="safe basename"):
        APIServerAdapter(config)


def test_systemd_credential_boundary_rejects_missing_posix_uid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr("gateway.platforms.api_server.os.geteuid")
    with pytest.raises(ValueError, match="POSIX UID"):
        _effective_uid_for_systemd_credential()


def test_systemd_credential_rejects_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = tmp_path / "secret"
    secret.write_bytes(b"secret")
    secret.chmod(0o600)
    credential_dir = tmp_path / "credentials"
    credential_dir.mkdir()
    (credential_dir / "api-server.key").symlink_to(secret)
    monkeypatch.setenv("CREDENTIALS_DIRECTORY", str(credential_dir))
    monkeypatch.delenv("API_SERVER_KEY", raising=False)
    with pytest.raises(ValueError, match="metadata is unsafe"):
        APIServerAdapter(_credential_config())


def test_systemd_credential_rejects_inline_or_environment_conflict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credential_dir = tmp_path / "credentials"
    credential_dir.mkdir()
    _write_credential(credential_dir)
    monkeypatch.setenv("CREDENTIALS_DIRECTORY", str(credential_dir))
    monkeypatch.setenv("API_SERVER_KEY", "environment-key")
    with pytest.raises(ValueError, match="cannot be combined"):
        APIServerAdapter(_credential_config())

    monkeypatch.delenv("API_SERVER_KEY")
    config = _credential_config()
    config.extra["key"] = "inline-key"
    with pytest.raises(ValueError, match="cannot be combined"):
        APIServerAdapter(config)


def _event_payloads(body: str) -> dict[str, dict]:
    events: dict[str, dict] = {}
    for block in body.split("\n\n"):
        lines = block.splitlines()
        event = next(
            (line.removeprefix("event: ") for line in lines if line.startswith("event: ")),
            None,
        )
        data = next(
            (json.loads(line.removeprefix("data: ")) for line in lines if line.startswith("data: ")),
            None,
        )
        if event is not None and data is not None:
            events[event] = data
    return events


def test_non_mapping_agent_result_is_never_completed() -> None:
    outcome = _session_stream_outcome("unexpected result")
    assert outcome == {
        "status": "failed",
        "assistant_event": "assistant.failed",
        "run_event": "run.failed",
        "completed": False,
        "partial": False,
        "interrupted": False,
        "failed": True,
        "incomplete": True,
        "finish_reason": "error",
        "turn_exit_reason": "invalid_agent_result",
    }


@pytest.mark.parametrize(
    ("result", "status", "finish_reason"),
    [
        ({"final_response": "legacy"}, "completed", "stop"),
        ({"completed": False}, "incomplete", "error"),
        ({"incomplete": True}, "incomplete", "error"),
        ({"partial": True}, "partial", "error"),
        ({"interrupted": True}, "interrupted", "error"),
        (
            {
                "partial": True,
                "outcome_code": "output_truncated",
                "error": "output was truncated",
            },
            "partial",
            "length",
        ),
    ],
)
def test_agent_outcome_normalization_is_shared_and_mechanical(
    result: dict,
    status: str,
    finish_reason: str,
) -> None:
    outcome = _session_stream_outcome(result)
    assert outcome["status"] == status
    assert outcome["finish_reason"] == finish_reason
    assert outcome["completed"] is (status == "completed")
    assert outcome["incomplete"] is (status != "completed")


def test_free_form_truncation_text_does_not_select_length_finish_reason() -> None:
    outcome = _session_stream_outcome(
        {"partial": True, "error": "output was truncated"}
    )
    assert outcome["status"] == "partial"
    assert outcome["finish_reason"] == "error"


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("completed", "false"),
        ("partial", 1),
        ("interrupted", "no"),
        ("failed", 0),
        ("incomplete", None),
    ],
)
def test_non_boolean_outcome_flags_are_invalid_agent_results(
    flag: str,
    value,
) -> None:
    outcome = _session_stream_outcome({"final_response": "text", flag: value})
    assert outcome["status"] == "failed"
    assert outcome["completed"] is False
    assert outcome["incomplete"] is True
    assert outcome["finish_reason"] == "error"
    assert outcome["turn_exit_reason"] == "invalid_agent_result"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("result", "expected_status", "expected_http"),
    [
        (
            {
                "final_response": "partial answer",
                "completed": False,
                "turn_exit_reason": "budget_exhausted",
            },
            "incomplete",
            200,
        ),
        ("not a mapping", "failed", 502),
    ],
)
async def test_synchronous_session_chat_exposes_incomplete_outcome(
    tmp_path: Path,
    result,
    expected_status: str,
    expected_http: int,
) -> None:
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._session_db = SessionDB(tmp_path / "state.db")
    session_id = adapter._session_db.create_session("sync-outcome", "api_server")
    mock_run = AsyncMock(return_value=(result, {"total_tokens": 3}))

    app = _session_app(adapter)
    with patch.object(adapter, "_run_agent", mock_run):
        async with TestClient(TestServer(app)) as client:
            response = await client.post(
                f"/api/sessions/{session_id}/chat",
                json={"message": "continue until complete"},
            )
            payload = await response.json()

    assert response.status == expected_http
    assert response.headers["X-Hermes-Completed"] == "false"
    if expected_http == 200:
        assert payload["message"]["content"] == "partial answer"
        outcome = payload["outcome"]
    else:
        outcome = payload["error"]["hermes"]
    assert outcome["status"] == expected_status
    assert outcome["completed"] is False
    assert outcome["incomplete"] is True
    adapter._session_db.close()


@pytest.mark.asyncio
async def test_session_stream_non_mapping_result_emits_failed_terminal_event(
    tmp_path: Path,
) -> None:
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._session_db = SessionDB(tmp_path / "state.db")
    session_id = adapter._session_db.create_session("stream-invalid", "api_server")

    app = _session_app(adapter)
    with patch.object(
        adapter,
        "_run_agent",
        new=AsyncMock(return_value=("not a mapping", {"total_tokens": 0})),
    ):
        async with TestClient(TestServer(app)) as client:
            response = await client.post(
                f"/api/sessions/{session_id}/chat/stream",
                json={"message": "finish"},
            )
            events = _event_payloads(await response.text())

    assert "assistant.completed" not in events
    assert "run.completed" not in events
    assert events["assistant.failed"]["incomplete"] is True
    assert events["run.failed"]["completed"] is False
    assert events["run.failed"]["turn_exit_reason"] == "invalid_agent_result"
    adapter._session_db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("result", "expected_http", "expected_status"),
    [
        (
            {
                "final_response": "unfinished answer",
                "completed": False,
                "turn_exit_reason": "budget_exhausted",
            },
            200,
            "incomplete",
        ),
        ("not a mapping", 502, "failed"),
        (
            {"final_response": "malformed result", "completed": "false"},
            200,
            "failed",
        ),
    ],
)
async def test_chat_completions_sync_never_reports_incomplete_as_stop(
    result,
    expected_http: int,
    expected_status: str,
) -> None:
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    app = _session_app(adapter)
    with patch.object(
        adapter,
        "_run_agent",
        new=AsyncMock(return_value=(result, {"total_tokens": 2})),
    ):
        async with TestClient(TestServer(app)) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-5.6-sol",
                    "messages": [{"role": "user", "content": "finish"}],
                },
            )
            payload = await response.json()

    assert response.status == expected_http
    assert response.headers["X-Hermes-Completed"] == "false"
    if expected_http == 200:
        assert payload["choices"][0]["finish_reason"] == "error"
        outcome = payload["hermes"]
    else:
        outcome = payload["error"]["hermes"]
    assert outcome["status"] == expected_status
    assert outcome["completed"] is False
    assert outcome["incomplete"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result",
    [
        {
            "final_response": "unfinished answer",
            "completed": False,
            "turn_exit_reason": "budget_exhausted",
        },
        "not a mapping",
    ],
)
async def test_chat_completions_sse_never_reports_incomplete_as_stop(result) -> None:
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    app = _session_app(adapter)
    with patch.object(
        adapter,
        "_run_agent",
        new=AsyncMock(return_value=(result, {"total_tokens": 2})),
    ):
        async with TestClient(TestServer(app)) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-5.6-sol",
                    "stream": True,
                    "messages": [{"role": "user", "content": "finish"}],
                },
            )
            body = await response.text()

    terminal_chunks = []
    for line in body.splitlines():
        if not line.startswith("data: {"):
            continue
        chunk = json.loads(line.removeprefix("data: "))
        if chunk["choices"][0].get("finish_reason") is not None:
            terminal_chunks.append(chunk)
    assert len(terminal_chunks) == 1
    terminal = terminal_chunks[0]
    assert terminal["choices"][0]["finish_reason"] == "error"
    assert terminal["hermes"]["completed"] is False
    assert terminal["hermes"]["incomplete"] is True


@pytest.mark.asyncio
async def test_responses_sync_stores_incomplete_without_advancing_conversation() -> None:
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    app = _session_app(adapter)
    incomplete_result = {
        "final_response": "finished only step one",
        "completed": False,
        "partial": True,
        "turn_exit_reason": "budget_exhausted",
        "messages": [
            {"role": "user", "content": "finish both steps"},
            {"role": "assistant", "content": "finished only step one"},
        ],
    }
    complete_result = {
        "final_response": "fresh complete answer",
        "messages": [
            {"role": "user", "content": "start fresh"},
            {"role": "assistant", "content": "fresh complete answer"},
        ],
    }

    async with TestClient(TestServer(app)) as client:
        with patch.object(
            adapter,
            "_run_agent",
            new=AsyncMock(return_value=(incomplete_result, {"total_tokens": 2})),
        ):
            response = await client.post(
                "/v1/responses",
                json={
                    "input": "finish both steps",
                    "conversation": "honest-sync",
                },
            )
            payload = await response.json()

        assert response.status == 200
        assert response.headers["X-Hermes-Completed"] == "false"
        assert payload["status"] == "incomplete"
        assert payload["hermes"]["completed"] is False
        assert payload["hermes"]["partial"] is True
        stored = adapter._response_store.get(payload["id"])
        assert stored is not None
        assert stored["response"]["status"] == "incomplete"
        assert adapter._response_store.get_conversation("honest-sync") is None

        with patch.object(
            adapter,
            "_run_agent",
            new=AsyncMock(return_value=(complete_result, {"total_tokens": 2})),
        ) as next_run:
            follow_up = await client.post(
                "/v1/responses",
                json={"input": "start fresh", "conversation": "honest-sync"},
            )
            assert follow_up.status == 200
        assert next_run.call_args.kwargs["conversation_history"] == []


@pytest.mark.asyncio
async def test_responses_sync_invalid_result_is_failed_not_completed() -> None:
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    app = _session_app(adapter)
    with patch.object(
        adapter,
        "_run_agent",
        new=AsyncMock(return_value=("not a mapping", {"total_tokens": 0})),
    ):
        async with TestClient(TestServer(app)) as client:
            response = await client.post(
                "/v1/responses",
                json={"input": "finish", "conversation": "invalid-sync"},
            )
            payload = await response.json()

    assert payload["status"] == "failed"
    assert payload["hermes"]["turn_exit_reason"] == "invalid_agent_result"
    assert payload["hermes"]["completed"] is False
    assert adapter._response_store.get(payload["id"])["response"]["status"] == "failed"
    assert adapter._response_store.get_conversation("invalid-sync") is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("result", "terminal_event", "terminal_status"),
    [
        (
            {
                "final_response": "partial streamed answer",
                "completed": False,
                "partial": True,
                "turn_exit_reason": "budget_exhausted",
            },
            "response.incomplete",
            "incomplete",
        ),
        ("not a mapping", "response.failed", "failed"),
    ],
)
async def test_responses_sse_never_emits_or_chains_completed_for_incomplete_result(
    result,
    terminal_event: str,
    terminal_status: str,
) -> None:
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    app = _session_app(adapter)
    with patch.object(
        adapter,
        "_run_agent",
        new=AsyncMock(return_value=(result, {"total_tokens": 1})),
    ):
        async with TestClient(TestServer(app)) as client:
            response = await client.post(
                "/v1/responses",
                json={
                    "input": "finish",
                    "stream": True,
                    "conversation": f"honest-sse-{terminal_status}",
                },
            )
            body = await response.text()

    events = _event_payloads(body)
    assert "response.completed" not in events
    assert terminal_event in events
    terminal = events[terminal_event]["response"]
    assert terminal["status"] == terminal_status
    assert terminal["hermes"]["completed"] is False
    stored = adapter._response_store.get(terminal["id"])
    assert stored is not None
    assert stored["response"]["status"] == terminal_status
    assert (
        adapter._response_store.get_conversation(
            f"honest-sse-{terminal_status}"
        )
        is None
    )


def _run_event_payloads(body: str) -> list[dict]:
    return [
        json.loads(line.removeprefix("data: "))
        for line in body.splitlines()
        if line.startswith("data: {")
    ]


def _local_revoke_receipt(session_key: str, epoch: str) -> dict:
    return {
        "success": True,
        "writer_required": False,
        "scope_revoked": False,
        "session_key_sha256": hashlib.sha256(session_key.encode()).hexdigest(),
        "capability_epoch_sha256": epoch,
        "authority_active": False,
        "revocation_event_id": None,
        "inserted": False,
        "deduped": False,
    }


def test_durable_revoke_receipt_cannot_confirm_a_different_epoch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import approval

    session_key = "exact-session-key"
    expected_epoch = "a" * 64
    monkeypatch.setattr(
        approval,
        "revoke_session_capabilities_durably",
        lambda _key, reason: {
            "success": True,
            "session_key_sha256": hashlib.sha256(session_key.encode()).hexdigest(),
            "capability_epoch_sha256": "b" * 64,
            "scope_type": "session",
            "scope_revoked": True,
            "authority_active": False,
            "revocation_event_id": "00000000-0000-4000-8000-000000000001",
            "inserted": True,
            "deduped": False,
        },
    )

    with pytest.raises(RuntimeError, match="exact authority tombstone"):
        APIServerAdapter._revoke_api_server_run_capabilities(
            session_key,
            expected_epoch,
        )


def test_nil_durable_revoke_event_cannot_confirm_terminal_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import approval

    session_key = "nil-event-session-key"
    expected_epoch = "a" * 64
    nil_event_id = "00000000-0000-0000-0000-000000000000"
    monkeypatch.setattr(
        approval,
        "revoke_session_capabilities_durably",
        lambda _key, reason: {
            "success": True,
            "session_key_sha256": hashlib.sha256(session_key.encode()).hexdigest(),
            "capability_epoch_sha256": expected_epoch,
            "scope_type": "session",
            "scope_revoked": True,
            "authority_active": False,
            "revocation_event_id": nil_event_id,
            "inserted": True,
            "deduped": False,
        },
    )

    with pytest.raises(RuntimeError, match="exact authority tombstone"):
        APIServerAdapter._revoke_api_server_run_capabilities(
            session_key,
            expected_epoch,
        )

    assert APIServerAdapter._api_cleanup_allows_terminal([{
        "authority_created": True,
        "durable_revoke_succeeded": True,
        "local_clear_succeeded": True,
        "authority_active": False,
        "writer_required": True,
        "revocation_event_id": nil_event_id,
        "inserted": True,
        "deduped": False,
    }]) is False


@pytest.mark.asyncio
async def test_cleanup_degraded_cannot_release_post_disconnect_terminal() -> None:
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    cleanup_ref = [{
        "authority_created": True,
        "durable_revoke_succeeded": True,
        "local_clear_succeeded": False,
        "authority_active": False,
        "writer_required": True,
        "revocation_event_id": "00000000-0000-4000-8000-000000000001",
        "inserted": True,
        "deduped": False,
    }]

    async def completed_agent():
        return {"final_response": "must wait"}, {}

    task = asyncio.create_task(completed_agent())
    await asyncio.sleep(0)
    assert await adapter._interrupt_and_await_api_task(
        task,
        None,
        cleanup_ref,
        reason="test disconnect",
    ) is False

    cleanup_ref[0]["local_clear_succeeded"] = True
    assert await adapter._interrupt_and_await_api_task(
        task,
        None,
        cleanup_ref,
        reason="test disconnect",
    ) is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("result", "event_name", "status_name"),
    [
        (
            {
                "final_response": "unfinished",
                "completed": False,
                "partial": True,
                "turn_exit_reason": "budget_exhausted",
            },
            "run.partial",
            "partial",
        ),
        ("not a mapping", "run.failed", "failed"),
    ],
)
async def test_runs_surface_uses_honest_terminal_outcome(
    monkeypatch: pytest.MonkeyPatch,
    result,
    event_name: str,
    status_name: str,
) -> None:
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    agent = MagicMock()
    agent.run_conversation.return_value = result
    agent.session_prompt_tokens = 1
    agent.session_completion_tokens = 1
    agent.session_total_tokens = 2
    monkeypatch.setattr(adapter, "_create_agent", lambda **_kwargs: agent)
    monkeypatch.setattr(
        adapter,
        "_revoke_api_server_run_capabilities",
        _local_revoke_receipt,
    )

    app = _session_app(adapter)
    async with TestClient(TestServer(app)) as client:
        started = await client.post("/v1/runs", json={"input": "finish"})
        run_id = (await started.json())["run_id"]
        events_response = await client.get(f"/v1/runs/{run_id}/events")
        events = _run_event_payloads(await events_response.text())
        status_response = await client.get(f"/v1/runs/{run_id}")
        status = await status_response.json()

    terminal = next(event for event in events if event["event"] == event_name)
    assert all(event["event"] != "run.completed" for event in events)
    assert terminal["completed"] is False
    assert terminal["incomplete"] is True
    assert status["status"] == status_name
    assert status["last_event"] == event_name
    assert status["completed"] is False


@pytest.mark.asyncio
async def test_runs_surface_revoke_retry_blocks_terminal_until_confirmed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    agent = MagicMock()
    agent.run_conversation.return_value = {
        "final_response": "must not escape as success"
    }
    agent.session_prompt_tokens = 1
    agent.session_completion_tokens = 1
    agent.session_total_tokens = 2
    monkeypatch.setattr(adapter, "_create_agent", lambda **_kwargs: agent)

    attempts = 0

    def fail_once_then_confirm(key: str, epoch: str) -> dict:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("writer revoke not confirmed")
        return _local_revoke_receipt(key, epoch)

    monkeypatch.setattr(
        adapter,
        "_revoke_api_server_run_capabilities",
        fail_once_then_confirm,
    )
    app = _session_app(adapter)
    async with TestClient(TestServer(app)) as client:
        started = await client.post("/v1/runs", json={"input": "finish"})
        run_id = (await started.json())["run_id"]
        events_response = await client.get(f"/v1/runs/{run_id}/events")
        events = _run_event_payloads(await events_response.text())
        status_response = await client.get(f"/v1/runs/{run_id}")
        status = await status_response.json()

    blocked_index = next(
        index
        for index, event in enumerate(events)
        if event["event"] == "run.cleanup_blocked"
    )
    completed_index = next(
        index
        for index, event in enumerate(events)
        if event["event"] == "run.completed"
    )
    assert blocked_index < completed_index
    assert all(event["event"] != "run.failed" for event in events)
    assert status["status"] == "completed"
    assert status["last_event"] == "run.completed"
    assert attempts >= 2


@pytest.mark.asyncio
async def test_stop_keeps_run_nonterminal_until_blocking_revoke_confirms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gateway.platforms.api_server as api_server_module

    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    agent = MagicMock()
    agent.run_conversation.return_value = {
        "final_response": "finished before cleanup",
        "completed": True,
    }
    agent.session_prompt_tokens = 1
    agent.session_completion_tokens = 1
    agent.session_total_tokens = 2
    monkeypatch.setattr(adapter, "_create_agent", lambda **_kwargs: agent)
    monkeypatch.setattr(
        api_server_module,
        "API_CLEANUP_SHIELD_TIMEOUT_SECONDS",
        0.01,
    )
    revoke_entered = threading.Event()
    release_revoke = threading.Event()

    def blocking_revoke(key: str, epoch: str) -> dict:
        revoke_entered.set()
        if not release_revoke.wait(timeout=5):
            raise RuntimeError("test did not release exact revoke")
        return _local_revoke_receipt(key, epoch)

    monkeypatch.setattr(
        adapter,
        "_revoke_api_server_run_capabilities",
        blocking_revoke,
    )
    app = _session_app(adapter)
    try:
        async with TestClient(TestServer(app)) as client:
            started = await client.post("/v1/runs", json={"input": "finish"})
            run_id = (await started.json())["run_id"]
            assert await asyncio.to_thread(revoke_entered.wait, 1)

            stopped = await client.post(f"/v1/runs/{run_id}/stop")
            assert (await stopped.json())["status"] == "stopping"
            status = await (await client.get(f"/v1/runs/{run_id}")).json()
            assert status["status"] == "stopping"
            assert status["terminal"] is False
            assert run_id in adapter._active_run_tasks
            assert run_id in adapter._run_approval_sessions

            release_revoke.set()
            events_response = await client.get(f"/v1/runs/{run_id}/events")
            events = _run_event_payloads(await events_response.text())
            assert any(event["event"] == "run.completed" for event in events)
    finally:
        release_revoke.set()


@pytest.mark.asyncio
async def test_session_stream_concurrent_validation_reserves_exactly_one_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._max_concurrent_runs = 1
    monkeypatch.setattr(
        adapter,
        "_get_existing_session_or_404",
        lambda _session_id: ({"id": "session-1"}, None),
    )
    monkeypatch.setattr(
        adapter,
        "_conversation_history_for_session",
        lambda _session_id: [],
    )
    validation_gate = asyncio.Event()
    release_agent = asyncio.Event()
    arrivals = 0

    async def synchronized_body(_request):
        nonlocal arrivals
        arrivals += 1
        if arrivals == 2:
            validation_gate.set()
        await validation_gate.wait()
        return {"message": "run exact canary"}, None

    async def blocking_run_agent(**_kwargs):
        adapter._inflight_agent_runs += 1
        try:
            await release_agent.wait()
            return {
                "final_response": "done",
                "completed": True,
                "messages": [],
            }, {"total_tokens": 1}
        finally:
            adapter._inflight_agent_runs -= 1

    monkeypatch.setattr(adapter, "_read_json_body", synchronized_body)
    monkeypatch.setattr(adapter, "_run_agent", blocking_run_agent)
    app = _session_app(adapter)
    async with TestClient(TestServer(app)) as client:
        first = asyncio.create_task(
            client.post(
                "/api/sessions/session-1/chat/stream",
                json={"message": "first"},
            )
        )
        second = asyncio.create_task(
            client.post(
                "/api/sessions/session-1/chat/stream",
                json={"message": "second"},
            )
        )
        responses = await asyncio.wait_for(
            asyncio.gather(first, second),
            timeout=2,
        )
        assert sorted(response.status for response in responses) == [200, 429]
        release_agent.set()
        for response in responses:
            await response.text()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("result_fields", "assistant_event", "run_event", "status"),
    [
        (
            {
                "completed": False,
                "partial": True,
                "interrupted": False,
                "failed": False,
                "turn_exit_reason": "max_iterations_reached(90/90)",
            },
            "assistant.partial",
            "run.partial",
            "partial",
        ),
        (
            {
                "completed": False,
                "partial": False,
                "interrupted": False,
                "failed": True,
                "turn_exit_reason": "provider_error",
                "error": "agent run failed",
            },
            "assistant.failed",
            "run.failed",
            "failed",
        ),
        (
            {
                "completed": False,
                "partial": False,
                "interrupted": True,
                "failed": False,
                "turn_exit_reason": "user_interrupt",
            },
            "assistant.partial",
            "run.partial",
            "interrupted",
        ),
    ],
)
async def test_session_stream_never_labels_incomplete_result_completed(
    tmp_path: Path,
    result_fields: dict,
    assistant_event: str,
    run_event: str,
    status: str,
) -> None:
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._session_db = SessionDB(tmp_path / "state.db")
    session_id = adapter._session_db.create_session("canary-stream", "api_server")
    transcript = [
        {"role": "user", "content": "complete the plan"},
        {"role": "assistant", "content": "finished step one"},
    ]

    async def fake_run(**_kwargs):
        return {
            "final_response": "finished step one",
            "session_id": session_id,
            "messages": transcript,
            **result_fields,
        }, {"total_tokens": 7}

    app = _session_app(adapter)
    with patch.object(adapter, "_run_agent", side_effect=fake_run):
        async with TestClient(TestServer(app)) as client:
            response = await client.post(
                f"/api/sessions/{session_id}/chat/stream",
                json={"message": "complete the plan"},
            )
            assert response.status == 200
            events = _event_payloads(await response.text())

    assert "assistant.completed" not in events
    assert "run.completed" not in events
    assert events[assistant_event]["status"] == status
    assert events[run_event]["status"] == status
    assert events[run_event]["completed"] is False
    assert events[run_event]["partial"] is bool(result_fields["partial"])
    assert events[run_event]["interrupted"] is bool(result_fields["interrupted"])
    assert events[run_event]["failed"] is bool(result_fields["failed"])
    assert events[run_event]["turn_exit_reason"] == result_fields["turn_exit_reason"]
    assert events[run_event]["messages"][0]["content"] == "finished step one"
    adapter._session_db.close()


def test_api_session_binding_has_fresh_writer_valid_epoch_per_run() -> None:
    observed = []
    for _ in range(2):
        tokens = APIServerAdapter._bind_api_server_session(
            chat_id="session-1",
            session_key="trusted-session-key",
            session_id="session-1",
        )
        try:
            envelope = trusted_runtime_envelope()
            observed.append(envelope)
            runtime = RuntimeContext(
                request_id="request-1",
                platform=envelope["platform"],
                session_key_sha256=envelope["session_key_sha256"],
                capability_epoch_sha256=envelope["capability_epoch_sha256"],
                chat_id=envelope["chat_id"],
            )
            _require_exact_runtime_epoch(runtime)
        finally:
            clear_session_vars(tokens)

    assert observed[0]["session_key_sha256"] == observed[1]["session_key_sha256"]
    assert observed[0]["capability_epoch_sha256"] != observed[1][
        "capability_epoch_sha256"
    ]
    assert trusted_runtime_envelope() == {}


@pytest.mark.asyncio
async def test_http_client_cannot_supply_epoch_and_revoke_precedes_clear(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._session_db = SessionDB(tmp_path / "state.db")
    session_id = adapter._session_db.create_session("epoch-stream", "api_server")
    during_agent = {}
    during_revoke = {}
    during_local_clear = {}
    cleanup_order = []

    class FakeAgent:
        session_prompt_tokens = 1
        session_completion_tokens = 1
        session_total_tokens = 2

        def __init__(self) -> None:
            self.session_id = session_id

        def run_conversation(self, **_kwargs):
            during_agent.update(trusted_runtime_envelope())
            return {
                "final_response": "done",
                "completed": True,
                "partial": False,
                "interrupted": False,
                "failed": False,
                "turn_exit_reason": "text_response(finish_reason=stop)",
                "messages": [
                    {"role": "user", "content": "finish"},
                    {"role": "assistant", "content": "done"},
                ],
            }

    monkeypatch.setattr(adapter, "_create_agent", lambda **_kwargs: FakeAgent())

    def fake_revoke(key: str, epoch: str) -> dict:
        assert key == session_id
        assert epoch == trusted_runtime_envelope()["capability_epoch_sha256"]
        cleanup_order.append("durable_revoke")
        during_revoke.update(trusted_runtime_envelope())
        return _local_revoke_receipt(key, epoch)

    real_local_clear = adapter._clear_api_server_run_local_authority

    def recording_local_clear(key: str, capability_epoch_sha256: str) -> None:
        assert key == session_id
        cleanup_order.append("local_clear")
        during_local_clear.update(trusted_runtime_envelope())
        assert (
            capability_epoch_sha256
            == during_local_clear["capability_epoch_sha256"]
        )
        real_local_clear(key, capability_epoch_sha256)

    monkeypatch.setattr(adapter, "_revoke_api_server_run_capabilities", fake_revoke)
    monkeypatch.setattr(
        adapter,
        "_clear_api_server_run_local_authority",
        recording_local_clear,
    )
    app = _session_app(adapter)
    client_epoch = "a" * 64
    async with TestClient(TestServer(app)) as client:
        response = await client.post(
            f"/api/sessions/{session_id}/chat/stream",
            json={
                "message": "finish",
                "capability_epoch_sha256": client_epoch,
            },
        )
        assert response.status == 200
        body = await response.text()

    assert "event: run.completed" in body
    assert cleanup_order == ["durable_revoke", "local_clear"]
    assert during_agent == during_revoke == during_local_clear
    assert during_agent["session_id"] == session_id
    assert during_agent["session_key_sha256"]
    assert during_agent["capability_epoch_sha256"] != client_epoch
    assert trusted_runtime_envelope() == {}
    adapter._session_db.close()


@pytest.mark.asyncio
async def test_run_local_authority_cannot_survive_stable_session_key_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import approval

    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    stable_key = "stable-api-authority-test"
    session_id = "stable-api-session"
    pattern_key = "dangerous-pattern"
    observed_epochs: list[str] = []
    call_count = 0
    approval.clear_session_local(stable_key)

    class FakeAgent:
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0

        def __init__(self) -> None:
            self.session_id = session_id

        def run_conversation(self, **_kwargs):
            nonlocal call_count
            envelope = trusted_runtime_envelope()
            observed_epochs.append(envelope["capability_epoch_sha256"])
            if call_count == 0:
                assert approval.approve_session(stable_key, pattern_key) is True
                assert approval.enable_session_yolo(stable_key) is True
                approval.submit_pending(stable_key, {"command": "dangerous"})
                with approval._lock:
                    approval._plan_capabilities[stable_key] = {
                        "plan-1": {"state": "granted"}
                    }
                assert approval.is_approved(stable_key, pattern_key) is True
                assert approval.is_session_yolo_enabled(stable_key) is True
            else:
                assert approval.is_approved(stable_key, pattern_key) is False
                assert approval.is_session_yolo_enabled(stable_key) is False
                with approval._lock:
                    assert stable_key not in approval._pending
                    assert stable_key not in approval._plan_capabilities
            call_count += 1
            return {"final_response": "done"}

    monkeypatch.setattr(adapter, "_create_agent", lambda **_kwargs: FakeAgent())
    monkeypatch.setattr(
        adapter,
        "_revoke_api_server_run_capabilities",
        _local_revoke_receipt,
    )

    try:
        await adapter._run_agent(
            user_message="first run",
            conversation_history=[],
            session_id=session_id,
            gateway_session_key=stable_key,
        )
        with approval._lock:
            assert stable_key not in approval._session_approved
            assert stable_key not in approval._session_yolo
            assert stable_key not in approval._pending
            assert stable_key not in approval._plan_capabilities
            assert (stable_key, observed_epochs[0]) in (
                approval._retired_session_capability_epochs
            )

        await adapter._run_agent(
            user_message="second run",
            conversation_history=[],
            session_id=session_id,
            gateway_session_key=stable_key,
        )
    finally:
        approval.clear_session_local(stable_key)

    assert call_count == 2
    assert len(observed_epochs) == 2
    assert observed_epochs[0] != observed_epochs[1]


@pytest.mark.asyncio
async def test_strict_capability_revoke_failure_retries_before_api_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = APIServerAdapter(PlatformConfig(enabled=True))

    class FakeAgent:
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0
        session_id = "strict-revoke-session"

        @staticmethod
        def run_conversation(**_kwargs):
            return {"final_response": "must not escape as success"}

    monkeypatch.setattr(adapter, "_create_agent", lambda **_kwargs: FakeAgent())

    attempts = 0
    states = []

    def fail_once_then_confirm(key: str, epoch: str) -> dict:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("writer revoke not confirmed")
        return _local_revoke_receipt(key, epoch)

    monkeypatch.setattr(
        adapter,
        "_revoke_api_server_run_capabilities",
        fail_once_then_confirm,
    )
    result, _usage = await adapter._run_agent(
        user_message="task",
        conversation_history=[],
        session_id="strict-revoke-session",
        cleanup_state_callback=states.append,
    )
    assert result["final_response"] == "must not escape as success"
    assert any(state["status"] == "cleanup_blocked" for state in states)
    assert states[-1]["durable_revoke_succeeded"] is True
    assert attempts >= 2


@pytest.mark.asyncio
async def test_strict_local_authority_clear_failure_fails_api_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = APIServerAdapter(PlatformConfig(enabled=True))

    class FakeAgent:
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0
        session_id = "strict-local-clear-session"

        @staticmethod
        def run_conversation(**_kwargs):
            return {"final_response": "must not escape as success"}

    monkeypatch.setattr(adapter, "_create_agent", lambda **_kwargs: FakeAgent())
    monkeypatch.setattr(
        adapter,
        "_revoke_api_server_run_capabilities",
        _local_revoke_receipt,
    )

    clear_attempts = 0

    def fail_once_local_clear(_key: str, _epoch: str) -> None:
        nonlocal clear_attempts
        clear_attempts += 1
        if clear_attempts == 1:
            raise RuntimeError("local authority fence not confirmed")

    monkeypatch.setattr(
        adapter,
        "_clear_api_server_run_local_authority",
        fail_once_local_clear,
    )
    result, _usage = await adapter._run_agent(
        user_message="task",
        conversation_history=[],
        session_id="strict-local-clear-session",
    )
    assert result["final_response"] == "must not escape as success"
    assert clear_attempts >= 2


@pytest.mark.asyncio
async def test_runs_bind_failure_preserves_original_error_and_skips_run_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    agent = MagicMock()
    agent.session_prompt_tokens = 0
    agent.session_completion_tokens = 0
    agent.session_total_tokens = 0
    monkeypatch.setattr(adapter, "_create_agent", lambda **_kwargs: agent)
    cleanup_calls: list[str] = []

    def fail_bind(**_kwargs):
        raise RuntimeError("session binding failed exactly")

    monkeypatch.setattr(adapter, "_bind_api_server_session", fail_bind)
    monkeypatch.setattr(
        adapter,
        "_revoke_api_server_run_capabilities",
        lambda _key, _epoch: cleanup_calls.append("revoke"),
    )
    monkeypatch.setattr(
        adapter,
        "_clear_api_server_run_local_authority",
        lambda _key, _epoch: cleanup_calls.append("local_clear"),
    )

    app = _session_app(adapter)
    async with TestClient(TestServer(app)) as client:
        started = await client.post("/v1/runs", json={"input": "finish"})
        run_id = (await started.json())["run_id"]
        events_response = await client.get(f"/v1/runs/{run_id}/events")
        events = _run_event_payloads(await events_response.text())
        status_response = await client.get(f"/v1/runs/{run_id}")
        status = await status_response.json()

    failure = next(event for event in events if event["event"] == "run.failed")
    assert "session binding failed exactly" in failure["error"]
    assert status["status"] == "failed"
    assert "session binding failed exactly" in status["error"]
    assert cleanup_calls == []
    agent.run_conversation.assert_not_called()
