"""Tests for /v1/runs endpoints: start, status, events, steer, and stop.

Covers:
- POST /v1/runs — start a run (202)
- GET /v1/runs/{run_id} — poll run status
- GET /v1/runs/{run_id}/events — SSE event stream
- POST /v1/runs/{run_id}/steer — inject guidance into a running agent
- POST /v1/runs/{run_id}/stop — interrupt a running agent
- Auth, error handling, and cleanup
"""

import asyncio
import hashlib
import json
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    _approval_event_choices,
    cors_middleware,
    security_headers_middleware,
)
from tools import approval as approval_mod


from tests.gateway.test_api_server_runs import (
    _create_runs_app,
    _make_adapter,
    _use_idempotency_db,
    auth_adapter,
)


@pytest.mark.asyncio
async def test_dead_owner_artifact_cleanup_is_retryable(tmp_path, monkeypatch):
        from gateway.hosted_room_artifacts import (
            RoomArtifactOutbox,
            RoomArtifactScope,
        )
        from gateway.platforms.api_server import RunIdempotencyStore

        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        room_scope = RoomArtifactScope.from_mapping({
            "room_id": "room-stale",
            "task_id": "task-stale",
            "execution_generation": 1,
            "member_id": "member-stale",
            "target_profile": "default",
            "home_install_id": "install-home",
            "target_install_id": "install-target",
            "authority_gateway_id": "gateway-home",
            "authority_epoch": 1,
        })
        output = tmp_path / "stale.md"
        output.write_text("stale\n", encoding="utf-8")
        outbox = RoomArtifactOutbox(home / "state.db")
        outbox.put_path(scope=room_scope, path=output)
        path = tmp_path / "idem.db"
        scope = hashlib.sha256(
            "default\0unauthenticated-test-listener".encode()
        ).hexdigest()
        store = RunIdempotencyStore(str(path))
        store.reserve(
            scope,
            "stale-run",
            "fingerprint",
            "run_stale",
            {
                "run_id": "run_stale",
                "status": "running",
                "room_artifact_scope": room_scope.as_mapping(),
            },
            owner_pid=999_999_999,
            owner_started=1,
        )
        store.close()

        restarted = _make_adapter()
        _use_idempotency_db(restarted, path)
        app = _create_runs_app(restarted)
        original_discard = RoomArtifactOutbox.discard
        attempts = 0

        def fail_once(instance, artifact_scope):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise OSError("temporary cleanup failure")
            return original_discard(instance, artifact_scope)

        monkeypatch.setattr(RoomArtifactOutbox, "discard", fail_once)
        async with TestClient(TestServer(app)) as cli:
            failed = await cli.get("/v1/runs/run_stale")
            assert failed.status == 500
            persisted = restarted._run_idempotency_store.status_for_run(
                scope,
                "run_stale",
            )
            assert persisted["status"]["status"] == "running"
            response = await cli.get("/v1/runs/run_stale")
            body = await response.json()
        assert response.status == 200
        assert body["status"] == "interrupted"
        assert body["last_event"] == "run.interrupted"
        assert outbox.list(room_scope) == []


@pytest.mark.asyncio
async def test_room_grant_refresh_does_not_expand_artifact_permissions(
    auth_adapter, monkeypatch
):
        from gateway import hosted_rooms
        from gateway.hosted_room_peer import decode_room_grant, issue_room_grant
        from gateway.hosted_rooms import local_authority_gateway_id

        old_grant = issue_room_grant(
            auth_adapter._room_grant_secret(),
            grant_id="grant-old",
            room_id="room-1",
            home_install_id="install-home",
            authority_gateway_id="install-home",
            authority_epoch=1,
            member_id="member-peer",
            target_install_id=local_authority_gateway_id(),
            target_profile="default",
            permissions=(
                "approve",
                "attachment.stage",
                "dispatch",
                "status",
                "stop",
            ),
            issued_at=100,
            ttl_seconds=300,
            status_expires_at=1000,
        )
        old_claims = decode_room_grant(
            auth_adapter._room_grant_secret(),
            old_grant,
            permission="status",
            now=100,
        )
        hosted_rooms.reserve_peer_room(
            hosted_rooms.default_db_path(),
            claims=old_claims,
            expires_at=1000,
            now=100,
        )
        monkeypatch.setattr("gateway.platforms.api_server.time.time", lambda: 200)
        app = _create_runs_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            refreshed = await cli.post(
                "/v1/room-members/grants/refresh",
                json={"ttl_seconds": 300},
                headers={"Authorization": f"HermesRoom {old_grant}"},
            )
            body = await refreshed.json()
        assert refreshed.status == 200
        assert body["grant"] != old_grant
        claims = decode_room_grant(
            auth_adapter._room_grant_secret(),
            body["grant"],
            permission="dispatch",
            now=200,
        )
        assert claims["room_id"] == "room-1"
        assert claims["home_install_id"] == "install-home"
        assert claims["status_expires_at"] == 1000
        assert set(claims["permissions"]) == set(old_claims["permissions"])
        assert {"artifact.ack", "artifact.read"}.isdisjoint(
            claims["permissions"]
        )

        status_only = issue_room_grant(
            auth_adapter._room_grant_secret(),
            grant_id="grant-status-only",
            room_id="room-1",
            home_install_id="install-home",
            authority_gateway_id="install-home",
            authority_epoch=1,
            member_id="member-peer",
            target_install_id=local_authority_gateway_id(),
            target_profile="default",
            permissions=("status",),
            issued_at=100,
            ttl_seconds=300,
            status_expires_at=1000,
        )
        status_claims = decode_room_grant(
            auth_adapter._room_grant_secret(),
            status_only,
            permission="status",
            now=100,
        )
        hosted_rooms.reserve_peer_room(
            hosted_rooms.default_db_path(),
            claims=status_claims,
            expires_at=1000,
            now=100,
        )
        app = _create_runs_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            status_refresh = await cli.post(
                "/v1/room-members/grants/refresh",
                json={"ttl_seconds": 300},
                headers={"Authorization": f"HermesRoom {status_only}"},
            )
            status_refresh_body = await status_refresh.json()
        assert status_refresh.status == 401
        assert status_refresh_body["error"]["code"] == "invalid_room_grant"

        fully_expired = issue_room_grant(
            auth_adapter._room_grant_secret(),
            grant_id="grant-expired",
            room_id="room-1",
            home_install_id="install-home",
            authority_gateway_id="install-home",
            authority_epoch=1,
            member_id="member-peer",
            target_install_id=local_authority_gateway_id(),
            target_profile="default",
            issued_at=100,
            ttl_seconds=10,
            status_expires_at=150,
        )
        app = _create_runs_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            denied = await cli.post(
                "/v1/room-members/grants/refresh",
                json={},
                headers={"Authorization": f"HermesRoom {fully_expired}"},
            )
            denied_body = await denied.json()
        assert denied.status == 401
        assert denied_body["error"]["code"] == "invalid_room_grant"


@pytest.mark.asyncio
async def test_room_run_binds_bot_room_source_and_emits_artifact_manifest(
    auth_adapter,
    tmp_path,
):
        from gateway.session_context import get_session_env
        from tools.hosted_room_artifact import share_group_file

        adapter = auth_adapter
        _use_idempotency_db(adapter, tmp_path / "idem.db")
        output = tmp_path / "handoff.md"
        output.write_text("# Hosted handoff\n", encoding="utf-8")
        captured = {}
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            invitation = await cli.post(
                "/v1/room-members/invitations",
                json={
                    "grant_id": "grant-room-artifact",
                    "room_id": "room-artifact",
                    "home_install_id": "install-home",
                    "authority_gateway_id": "gateway-home",
                    "authority_epoch": 1,
                    "member_id": "member-reviewer",
                    "ttl_seconds": 3600,
                },
                headers={"Authorization": "Bearer sk-secret"},
            )
            invitation_body = await invitation.json()
            assert invitation.status == 201
            grant = invitation_body["grant"]
            catalog = invitation_body["catalog"]
            prompt = "Prepare the hosted file handoff."
            dispatch = {
                "protocol_version": 2,
                "room_id": "room-artifact",
                "home_install_id": "install-home",
                "authority_gateway_id": "gateway-home",
                "authority_epoch": 1,
                "member_id": "member-reviewer",
                "target_install_id": catalog["installation_id"],
                "target_profile": "default",
                "task_id": "task-room-artifact",
                "execution_generation": 1,
                "source_event_seq": 1,
                "cancellation_scope_id": "cancel-room-artifact",
                "prompt": prompt,
                "prompt_digest": hashlib.sha256(prompt.encode()).hexdigest(),
                "capability_digest": catalog["catalog_digest"],
                "execution_policy_digest": catalog["execution_policy"][
                    "policy_digest"
                ],
                "trace_id": "trace-room-artifact",
            }
            with patch.object(adapter, "_create_agent") as create:
                agent = MagicMock()

                def run_room_turn(**_kwargs):
                    captured["source"] = get_session_env(
                        "HERMES_SESSION_SOURCE",
                        "",
                    )
                    captured["tool_result"] = json.loads(
                        share_group_file(str(output))
                    )
                    return {"final_response": "Hosted file ready."}

                agent.run_conversation.side_effect = run_room_turn
                agent.session_prompt_tokens = agent.session_completion_tokens = (
                    agent.session_total_tokens
                ) = 0
                create.return_value = agent
                started = await cli.post(
                    "/v1/runs",
                    json={"input": prompt, "hosted_room_dispatch": dispatch},
                    headers={
                        "Authorization": f"HermesRoom {grant}",
                        "Idempotency-Key": "room:task-room-artifact:1",
                    },
                )
                started_body = await started.json()
                assert started.status == 202
                run_id = started_body["run_id"]
                for _ in range(40):
                    status = await cli.get(
                        f"/v1/runs/{run_id}",
                        headers={"Authorization": f"HermesRoom {grant}"},
                    )
                    status_body = await status.json()
                    if status_body.get("status") == "completed":
                        break
                    await asyncio.sleep(0.05)

        assert captured["source"] == "bot_room"
        assert captured["tool_result"]["ok"] is True
        assert status_body["output"] == "Hosted file ready."
        assert status_body["artifacts"]["version"] == 1
        assert status_body["artifacts"]["items"][0]["name"] == "handoff.md"
        assert json.loads(share_group_file(str(output)))["ok"] is False


@pytest.mark.asyncio
async def test_room_run_exception_discards_private_artifacts(
    auth_adapter,
    tmp_path,
    monkeypatch,
):
        from gateway.hosted_room_artifacts import RoomArtifactOutbox, RoomArtifactScope
        from hermes_constants import get_hermes_home
        from tools.hosted_room_artifact import share_group_file

        adapter = auth_adapter
        _use_idempotency_db(adapter, tmp_path / "idem.db")
        output = tmp_path / "handoff.md"
        output.write_text("private draft\n", encoding="utf-8")
        original_discard = RoomArtifactOutbox.discard
        cleanup_attempts = 0

        def fail_once(instance, artifact_scope):
            nonlocal cleanup_attempts
            cleanup_attempts += 1
            if cleanup_attempts == 1:
                raise OSError("temporary cleanup fault")
            return original_discard(instance, artifact_scope)

        monkeypatch.setattr(RoomArtifactOutbox, "discard", fail_once)
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            invitation = await cli.post(
                "/v1/room-members/invitations",
                json={
                    "grant_id": "grant-room-artifact-error",
                    "room_id": "room-artifact-error",
                    "home_install_id": "install-home",
                    "authority_gateway_id": "gateway-home",
                    "authority_epoch": 1,
                    "member_id": "member-reviewer",
                    "ttl_seconds": 3600,
                },
                headers={"Authorization": "Bearer sk-secret"},
            )
            invitation_body = await invitation.json()
            grant = invitation_body["grant"]
            catalog = invitation_body["catalog"]
            prompt = "Prepare a file, then fail."
            dispatch = {
                "protocol_version": 2,
                "room_id": "room-artifact-error",
                "home_install_id": "install-home",
                "authority_gateway_id": "gateway-home",
                "authority_epoch": 1,
                "member_id": "member-reviewer",
                "target_install_id": catalog["installation_id"],
                "target_profile": "default",
                "task_id": "task-room-artifact-error",
                "execution_generation": 1,
                "source_event_seq": 1,
                "cancellation_scope_id": "cancel-room-artifact-error",
                "prompt": prompt,
                "prompt_digest": hashlib.sha256(prompt.encode()).hexdigest(),
                "capability_digest": catalog["catalog_digest"],
                "execution_policy_digest": catalog["execution_policy"][
                    "policy_digest"
                ],
                "trace_id": "trace-room-artifact-error",
            }
            with patch.object(adapter, "_create_agent") as create:
                agent = MagicMock()

                def fail_after_file(**_kwargs):
                    assert json.loads(share_group_file(str(output)))["ok"] is True
                    raise RuntimeError("provider exploded after tool output")

                agent.run_conversation.side_effect = fail_after_file
                agent.session_prompt_tokens = agent.session_completion_tokens = (
                    agent.session_total_tokens
                ) = 0
                create.return_value = agent
                started = await cli.post(
                    "/v1/runs",
                    json={"input": prompt, "hosted_room_dispatch": dispatch},
                    headers={
                        "Authorization": f"HermesRoom {grant}",
                        "Idempotency-Key": "room:task-room-artifact-error:1",
                    },
                )
                run_id = (await started.json())["run_id"]
                for _ in range(40):
                    status = await cli.get(
                        f"/v1/runs/{run_id}",
                        headers={"Authorization": f"HermesRoom {grant}"},
                    )
                    status_body = await status.json()
                    if status_body.get("status") == "failed":
                        break
                    await asyncio.sleep(0.05)

        scope = RoomArtifactScope.from_mapping({
            key: dispatch[key]
            for key in (
                "room_id",
                "task_id",
                "execution_generation",
                "member_id",
                "target_profile",
                "home_install_id",
                "target_install_id",
                "authority_gateway_id",
                "authority_epoch",
            )
        })
        assert status_body["status"] == "failed"
        assert cleanup_attempts == 1
        assert RoomArtifactOutbox(get_hermes_home() / "state.db").list(scope) == []
        assert cleanup_attempts == 2
