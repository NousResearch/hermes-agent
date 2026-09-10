"""Hosted RoomLink Runs: scoped admission, grants, and exact approval controls."""

import asyncio
import hashlib
import os
from unittest.mock import MagicMock, patch

import pytest
from aiohttp.test_utils import TestClient, TestServer

from tests.gateway.test_api_server_runs import (
    _create_runs_app,
    _use_idempotency_db,
    auth_adapter,  # noqa: F401
)
from tools import approval as approval_mod
from tools import approval_gateway_wait


class TestHostedRoomRuns:
    @pytest.mark.asyncio
    async def test_room_approval_requires_and_resolves_exact_request_id(
        self, auth_adapter, tmp_path
    ):
        from gateway.hosted_room_peer import decode_room_grant
        from gateway.platforms.api_server_run_scope import room_run_scope_key
        from gateway.status import get_process_start_time

        _use_idempotency_db(auth_adapter, tmp_path / "approval-runs.db")
        run_id = "run-room-approval"
        current = approval_gateway_wait._ApprovalEntry({
            "request_id": "approval-B",
            "command": "rm -rf build-B",
        })
        auth_adapter._run_approval_sessions[run_id] = run_id
        auth_adapter._run_statuses[run_id] = {
            "run_id": run_id,
            "status": "waiting_for_approval",
            "approval": dict(current.data),
        }
        with approval_mod._lock:
            approval_mod._gateway_queues[run_id] = [current]
        app = _create_runs_app(auth_adapter)
        try:
            async with TestClient(TestServer(app)) as cli:
                invitation = await cli.post(
                    "/v1/room-members/invitations",
                    json={
                        "room_id": "room-approval", "home_install_id": "install-home",
                        "authority_gateway_id": "gateway-home", "authority_epoch": 1,
                        "member_id": "member-reviewer",
                    },
                    headers={"Authorization": "Bearer sk-secret"},
                )
                assert invitation.status == 201
                grant = (await invitation.json())["grant"]
                claims = decode_room_grant(auth_adapter._room_grant_secret(), grant, permission="approve")
                scope = room_run_scope_key(claims)
                auth_adapter._run_idempotency_store.reserve(
                    scope, "approval-key", "approval-fingerprint", run_id,
                    auth_adapter._run_statuses[run_id], owner_pid=os.getpid(),
                    owner_started=int(get_process_start_time(os.getpid()) or 0),
                    retention_until=claims["status_expires_at"],
                )
                auth_adapter._run_idempotency_ids.add(run_id)
                auth_adapter._run_owners[run_id] = scope
                headers = {"Authorization": f"HermesRoom {grant}"}
                missing = await cli.post(
                    f"/v1/runs/{run_id}/approval",
                    json={"choice": "once"}, headers=headers,
                )
                stale = await cli.post(
                    f"/v1/runs/{run_id}/approval",
                    json={"choice": "once", "request_id": "approval-A"}, headers=headers,
                )
                exact = await cli.post(
                    f"/v1/runs/{run_id}/approval",
                    json={"choice": "once", "request_id": "approval-B"}, headers=headers,
                )
                missing_body = await missing.json()
                stale_body = await stale.json()
                exact_body = await exact.json()
        finally:
            approval_mod.unregister_gateway_notify(run_id)
            auth_adapter._run_idempotency_store.close()

        assert missing.status == 400
        assert missing_body["error"]["code"] == "approval_request_required"
        assert stale.status == 409
        assert stale_body["error"]["code"] == "approval_not_pending"
        assert exact.status == 200
        assert exact_body["request_id"] == "approval-B"
        assert current.result == "once"
        assert "approval" not in auth_adapter._run_statuses[run_id]

    @pytest.mark.asyncio
    async def test_room_grant_cannot_create_session_or_permanent_approval_policy(
        self, auth_adapter
    ):
        app = _create_runs_app(auth_adapter)
        with (
            patch.object(auth_adapter, "_check_run_auth", return_value=None),
            patch.object(auth_adapter, "_request_owns_run", return_value=True),
            patch.object(
                auth_adapter,
                "_durable_run_status",
                return_value={"status": "waiting_for_approval"},
            ),
            patch.object(
                auth_adapter, "_room_grant_token", return_value="scoped-grant"
            ),
        ):
            async with TestClient(TestServer(app)) as cli:
                permanent = await cli.post(
                    "/v1/runs/run-room/approval",
                    json={"choice": "always"},
                )
                resolve_all = await cli.post(
                    "/v1/runs/run-room/approval",
                    json={"choice": "once", "resolve_all": True},
                )
                permanent_body = await permanent.json()
                resolve_all_body = await resolve_all.json()

        assert permanent.status == 400
        assert permanent_body["error"]["code"] == "invalid_approval_choice"
        assert resolve_all.status == 400
        assert resolve_all_body["error"]["code"] == "invalid_approval_scope"

    @pytest.mark.asyncio
    async def test_invitation_uses_validated_app_managed_local_catalog(
        self, auth_adapter, monkeypatch
    ):
        monkeypatch.setenv("HERMES_DESKTOP", "1")
        monkeypatch.setenv(
            "HERMES_ROOM_LINK_URL", "https://peer.example.test/hermes"
        )
        app = _create_runs_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            invitation = await cli.post(
                "/v1/room-members/invitations",
                json={
                    "room_id": "room-1",
                    "home_install_id": "install-home",
                    "authority_gateway_id": "gateway-home",
                    "authority_epoch": 1,
                    "member_id": "member-reviewer",
                },
                headers={"Authorization": "Bearer sk-secret"},
            )
            body = await invitation.json()
        assert invitation.status == 201
        assert body["catalog"]["persistent_process"] is False
        assert body["catalog"]["link_modes"] == ["direct"]
        assert body["catalog"]["endpoint"] == {
            "available": True,
            "url": "https://peer.example.test/hermes",
            "transport_security": "tls",
        }
        assert body["expires_at"] == body["status_expires_at"]

    @pytest.mark.asyncio
    async def test_invitation_returns_operator_selected_status_horizon(
        self, auth_adapter
    ):
        app = _create_runs_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            invitation = await cli.post(
                "/v1/room-members/invitations",
                json={
                    "room_id": "room-horizon",
                    "home_install_id": "install-home",
                    "authority_gateway_id": "gateway-home",
                    "authority_epoch": 1,
                    "member_id": "member-reviewer",
                    "ttl_seconds": 600,
                    "status_ttl_seconds": 3600,
                },
                headers={"Authorization": "Bearer sk-secret"},
            )
            body = await invitation.json()

        assert invitation.status == 201
        assert body["status_expires_at"] - body["expires_at"] == 3000

    @pytest.mark.asyncio
    async def test_scoped_grant_refresh_requires_live_dispatch_authority(
        self, auth_adapter, monkeypatch
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
    async def test_scoped_grant_refresh_refuses_execution_policy_drift(
        self, auth_adapter, monkeypatch
    ):
        """Renewal must pause for reauthorization when the target's execution
        policy changed since the grant was issued — never silently mint a
        grant against the drifted policy (blocker 2, #97681 review)."""
        from gateway import hosted_rooms
        from gateway.hosted_room_peer import issue_room_grant, decode_room_grant
        from gateway.hosted_rooms import local_authority_gateway_id

        stale_digest = "c" * 64
        drifted = issue_room_grant(
            auth_adapter._room_grant_secret(),
            grant_id="grant-drifted",
            room_id="room-1",
            home_install_id="install-home",
            authority_gateway_id="install-home",
            authority_epoch=1,
            member_id="member-peer",
            target_install_id=local_authority_gateway_id(),
            target_profile="default",
            execution_policy_digest=stale_digest,
            issued_at=100,
            ttl_seconds=300,
            status_expires_at=1000,
        )
        drifted_claims = decode_room_grant(
            auth_adapter._room_grant_secret(),
            drifted,
            permission="status",
            now=100,
        )
        hosted_rooms.reserve_peer_room(
            hosted_rooms.default_db_path(),
            claims=drifted_claims,
            expires_at=1000,
            now=100,
        )
        monkeypatch.setattr("gateway.platforms.api_server.time.time", lambda: 200)
        app = _create_runs_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            refused = await cli.post(
                "/v1/room-members/grants/refresh",
                json={"ttl_seconds": 300},
                headers={"Authorization": f"HermesRoom {drifted}"},
            )
            refused_body = await refused.json()
        assert refused.status == 403
        assert refused_body["error"]["code"] == "room_reauthorization_required"

    @pytest.mark.asyncio
    async def test_scoped_grant_refresh_fails_after_secret_rotation(
        self, auth_adapter, monkeypatch
    ):
        from gateway import hosted_rooms
        from gateway.hosted_room_peer import decode_room_grant, issue_room_grant
        from gateway.hosted_rooms import local_authority_gateway_id

        monkeypatch.setattr("gateway.platforms.api_server.time.time", lambda: 200)
        revoked = issue_room_grant(
            b"x" * 32,
            grant_id="grant-revoked",
            room_id="room-1",
            home_install_id="install-home",
            authority_gateway_id="install-home",
            authority_epoch=1,
            member_id="member-peer",
            target_install_id=local_authority_gateway_id(),
            target_profile="default",
            issued_at=100,
            ttl_seconds=300,
            status_expires_at=1000,
        )
        app = _create_runs_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            denied = await cli.post(
                "/v1/room-members/grants/refresh",
                json={},
                headers={"Authorization": f"HermesRoom {revoked}"},
            )
            denied_body = await denied.json()
        assert denied.status == 401
        assert denied_body["error"]["code"] == "invalid_room_grant"

    def test_grant_refresh_keeps_idempotency_scope_but_member_change_does_not(
        self, auth_adapter
    ):
        from types import SimpleNamespace

        from gateway import hosted_rooms
        from gateway.hosted_room_peer import decode_room_grant, issue_room_grant
        from gateway.hosted_rooms import local_authority_gateway_id

        common = {
            "room_id": "room-1",
            "home_install_id": "install-home",
            "authority_gateway_id": "gateway-home",
            "authority_epoch": 1,
            "member_id": "member-reviewer",
            "target_install_id": local_authority_gateway_id(),
            "target_profile": "default",
        }
        first = issue_room_grant(
            auth_adapter._room_grant_secret(),
            grant_id="grant-first",
            **common,
        )
        refreshed = issue_room_grant(
            auth_adapter._room_grant_secret(),
            grant_id="grant-refreshed",
            **common,
        )
        other_member = issue_room_grant(
            auth_adapter._room_grant_secret(),
            grant_id="grant-other-member",
            **{**common, "member_id": "member-other"},
        )
        for grant in (first, other_member):
            claims = decode_room_grant(
                auth_adapter._room_grant_secret(),
                grant,
                permission="status",
            )
            hosted_rooms.reserve_peer_room(
                hosted_rooms.default_db_path(),
                claims=claims,
                expires_at=float(claims["status_expires_at"]),
            )

        def request(token):
            return SimpleNamespace(
                headers={"Authorization": f"HermesRoom {token}"},
                method="POST",
                path="/v1/runs",
            )

        first_scope = auth_adapter._run_idempotency_scope(request(first))
        assert auth_adapter._run_idempotency_scope(request(refreshed)) == first_scope
        assert auth_adapter._run_idempotency_scope(request(other_member)) != first_scope

    @pytest.mark.asyncio
    async def test_scoped_grant_revoke_is_idempotent_and_fences_prior_lineage(
        self, auth_adapter, monkeypatch
    ):
        from gateway import hosted_rooms
        from gateway.hosted_room_peer import decode_room_grant, issue_room_grant
        from gateway.hosted_rooms import local_authority_gateway_id

        for target in (
            "gateway.platforms.api_server.time.time",
            "gateway.hosted_rooms_common.time.time",
        ):
            monkeypatch.setattr(target, lambda: 200)
        claims = {
            "room_id": "room-1",
            "home_install_id": "install-home",
            "authority_gateway_id": "install-home",
            "authority_epoch": 1,
            "member_id": "member-peer",
            "target_install_id": local_authority_gateway_id(),
            "target_profile": "default",
        }
        old_grant = issue_room_grant(
            auth_adapter._room_grant_secret(),
            grant_id="grant-old",
            **claims,
            issued_at=100,
            ttl_seconds=300,
            status_expires_at=1000,
        )
        app = _create_runs_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            first = await cli.post(
                "/v1/room-members/grants/revoke",
                json={},
                headers={"Authorization": f"HermesRoom {old_grant}"},
            )
            repeated = await cli.post(
                "/v1/room-members/grants/revoke",
                json={},
                headers={"Authorization": f"HermesRoom {old_grant}"},
            )
            denied = await cli.get(
                "/v1/room-members/capabilities",
                headers={"Authorization": f"HermesRoom {old_grant}"},
            )
            denied_run = await cli.post(
                "/v1/runs",
                data="{never parsed",
                headers={
                    "Authorization": f"HermesRoom {old_grant}",
                    "Content-Type": "application/json",
                },
            )
            denied_body = await denied.json()
            denied_run_body = await denied_run.json()
            future_grant = issue_room_grant(
                auth_adapter._room_grant_secret(),
                grant_id="grant-repaired",
                **claims,
                issued_at=201,
                ttl_seconds=300,
                status_expires_at=1000,
            )
            future_claims = decode_room_grant(
                auth_adapter._room_grant_secret(),
                future_grant,
                permission="status",
                now=201,
            )
            hosted_rooms.reserve_peer_room(
                hosted_rooms.default_db_path(),
                claims=future_claims,
                expires_at=1000,
                now=201,
            )
            repaired = await cli.get(
                "/v1/room-members/capabilities",
                headers={"Authorization": f"HermesRoom {future_grant}"},
            )
        assert first.status == repeated.status == 200
        assert denied.status == 403
        assert denied_body["error"]["code"] == "room_reauthorization_required"
        assert denied_run.status == 403
        assert (
            denied_run_body["error"]["code"]
            == "room_reauthorization_required"
        )
        assert auth_adapter._pending_agent_requests == 0
        assert repaired.status == 200

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("method", "suffix"),
        [("GET", ""), ("POST", "/stop")],
    )
    async def test_room_grant_cannot_access_ownerless_compat_run(
        self, auth_adapter, tmp_path, method, suffix
    ):
        adapter = auth_adapter
        _use_idempotency_db(adapter, tmp_path / "idem.db")
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            invitation = await cli.post(
                "/v1/room-members/invitations",
                json={
                    "room_id": "room-1",
                    "home_install_id": "install-home",
                    "authority_gateway_id": "gateway-home",
                    "authority_epoch": 1,
                    "member_id": "member-reviewer",
                },
                headers={"Authorization": "Bearer sk-secret"},
            )
            grant = (await invitation.json())["grant"]
            adapter._run_statuses["run_ownerless"] = {
                "run_id": "run_ownerless",
                "status": "running",
            }
            response = await cli.request(
                method,
                f"/v1/runs/run_ownerless{suffix}",
                json={} if method == "POST" else None,
                headers={"Authorization": f"HermesRoom {grant}"},
            )
        assert response.status == 404

    @pytest.mark.asyncio
    async def test_scoped_grant_admits_group_session_run_without_peer_api_key(
        self, auth_adapter, tmp_path
    ):
        from gateway import hosted_rooms

        adapter = auth_adapter
        _use_idempotency_db(adapter, tmp_path / "idem.db")
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            invitation = await cli.post(
                "/v1/room-members/invitations",
                json={
                    "grant_id": "grant-room-1",
                    "room_id": "room-1",
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
            probe = await cli.get(
                "/v1/room-members/capabilities",
                headers={"Authorization": f"HermesRoom {grant}"},
            )
            probe_body = await probe.json()
            assert probe.status == 200
            assert probe_body["catalog"] == catalog
            prompt = "Review this room message."
            dispatch = {
                "protocol_version": 2,
                "room_id": "room-1",
                "home_install_id": "install-home",
                "authority_gateway_id": "gateway-home",
                "authority_epoch": 1,
                "member_id": "member-reviewer",
                "target_install_id": catalog["installation_id"],
                "target_profile": "default",
                "task_id": "task-room-1",
                "execution_generation": 1,
                "source_event_seq": 1,
                "cancellation_scope_id": "cancel-room-1",
                "prompt": prompt,
                "prompt_digest": hashlib.sha256(prompt.encode()).hexdigest(),
                "capability_digest": catalog["catalog_digest"],
                "execution_policy_digest": catalog["execution_policy"][
                    "policy_digest"
                ],
                "trace_id": "trace-room-1",
            }
            with patch.object(adapter, "_create_agent") as create:
                agent = MagicMock()
                agent.run_conversation.return_value = {
                    "final_response": "Scoped room reply."
                }
                agent.session_prompt_tokens = agent.session_completion_tokens = (
                    agent.session_total_tokens
                ) = 0
                create.return_value = agent
                started = await cli.post(
                    "/v1/runs",
                    json={"input": prompt, "hosted_room_dispatch": dispatch},
                    headers={
                        "Authorization": f"HermesRoom {grant}",
                        "Idempotency-Key": "room:task-room-1:1",
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
            assert status.status == 200
            assert status_body["output"] == "Scoped room reply."
            session_id = status_body["session_id"]
            db = await adapter._ensure_session_db_async()
            row = db.get_session(session_id)
            assert row["source"] == "bot_room"
            assert row["title"] == "Group: room-1"
            assert catalog["installation_id"] == (
                hosted_rooms.local_authority_gateway_id()
            )

    @pytest.mark.asyncio
    async def test_scoped_grant_rejects_capability_and_target_tampering(
        self, auth_adapter, tmp_path
    ):
        adapter = auth_adapter
        _use_idempotency_db(adapter, tmp_path / "idem.db")
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            invitation = await cli.post(
                "/v1/room-members/invitations",
                json={
                    "room_id": "room-1",
                    "home_install_id": "install-home",
                    "authority_gateway_id": "gateway-home",
                    "authority_epoch": 1,
                    "member_id": "member-reviewer",
                },
                headers={"Authorization": "Bearer sk-secret"},
            )
            invitation_body = await invitation.json()
            prompt = "Review."
            dispatch = {
                "protocol_version": 2,
                "room_id": "room-1",
                "home_install_id": "install-home",
                "authority_gateway_id": "gateway-home",
                "authority_epoch": 1,
                "member_id": "member-reviewer",
                "target_install_id": invitation_body["catalog"]["installation_id"],
                "target_profile": "default",
                "task_id": "task-room-1",
                "execution_generation": 1,
                "source_event_seq": 1,
                "cancellation_scope_id": "cancel-room-1",
                "prompt": prompt,
                "prompt_digest": hashlib.sha256(prompt.encode()).hexdigest(),
                "capability_digest": "f" * 64,
                "execution_policy_digest": invitation_body["catalog"][
                    "execution_policy"
                ]["policy_digest"],
                "trace_id": "trace-room-1",
            }
            with patch.object(adapter, "_create_agent") as create:
                rejected = await cli.post(
                    "/v1/runs",
                    json={"input": prompt, "hosted_room_dispatch": dispatch},
                    headers={
                        "Authorization": f"HermesRoom {invitation_body['grant']}",
                        "Idempotency-Key": "room:task-room-1:1",
                    },
                )
            assert rejected.status == 403
            create.assert_not_called()
