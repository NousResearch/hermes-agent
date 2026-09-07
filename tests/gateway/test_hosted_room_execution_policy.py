"""Target-issued execution-policy regressions for text-only RoomLink turns."""

from __future__ import annotations

import hashlib
import json

import pytest

from gateway.hosted_room_execution_policy import (
    MAX_POLICY_ITERATIONS,
    RoomExecutionPolicy,
    bind_room_execution_policy,
    execution_policy_mapping,
    reset_room_execution_policy,
)
from gateway.hosted_room_peer import (
    GatewayRoomCatalog,
    HostedMemberDispatch,
    HostedRoomGrantError,
    catalog_mapping,
    issue_room_grant,
    verify_room_grant,
)
from tools import approval
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError
from tui_gateway.hosted_room_service import _RouteStatusPeerClient


def _policy(*, approval_mode: str = "manual", max_turns: int = 12) -> dict:
    return execution_policy_mapping(
        target_profile="reviewer",
        config={
            "agent": {"max_turns": max_turns},
            "approvals": {"mode": approval_mode},
            "platform_toolsets": {"api_server": ["hermes-api-server", "web"]},
        },
    )


def _dispatch(policy: dict) -> HostedMemberDispatch:
    prompt = "Review the user's patch."
    catalog = GatewayRoomCatalog.from_mapping(
        catalog_mapping(
            installation_id="install-peer",
            persistent_process=True,
            target_profile="reviewer",
            execution_policy=policy,
        )
    )
    return HostedMemberDispatch.from_mapping({
        "protocol_version": 2,
        "room_id": "room-1",
        "home_install_id": "install-home",
        "authority_gateway_id": "gateway-home",
        "authority_epoch": 1,
        "member_id": "member-reviewer",
        "target_install_id": "install-peer",
        "target_profile": "reviewer",
        "task_id": "task-1",
        "execution_generation": 1,
        "source_event_seq": 1,
        "cancellation_scope_id": "cancel-1",
        "prompt": prompt,
        "prompt_digest": hashlib.sha256(prompt.encode()).hexdigest(),
        "capability_digest": catalog.catalog_digest,
        "execution_policy_digest": policy["policy_digest"],
        "trace_id": "trace-1",
    })


def test_execution_policy_digest_covers_tools_approvals_and_iteration_limit():
    value = _policy()
    checked = RoomExecutionPolicy.from_mapping(value)
    assert "bot_room" in checked.enabled_toolsets

    for field, replacement in (
        ("enabled_toolsets", ["bot_room"]),
        ("approval_mode", "off"),
        ("max_iterations", 99),
    ):
        with pytest.raises(ValueError, match="policy_digest"):
            RoomExecutionPolicy.from_mapping({**value, field: replacement})


def test_unlimited_policy_survives_the_catalog_json_round_trip_exactly():
    policy = _policy(max_turns=0)
    catalog = catalog_mapping(
        installation_id="install-peer",
        persistent_process=True,
        target_profile="reviewer",
        execution_policy=policy,
    )
    wire = json.loads(json.dumps(catalog))
    checked = GatewayRoomCatalog.from_mapping(wire)

    assert policy["max_iterations"] == MAX_POLICY_ITERATIONS
    assert int(float(policy["max_iterations"])) == MAX_POLICY_ITERATIONS
    assert checked.execution_policy.max_iterations == MAX_POLICY_ITERATIONS
    assert checked.execution_policy.as_mapping() == policy


def test_room_policy_overrides_broader_live_approval_config(monkeypatch):
    policy = RoomExecutionPolicy.from_mapping(_policy(approval_mode="manual"))
    monkeypatch.setattr(approval, "_get_approval_config", lambda: {"mode": "off"})
    token = bind_room_execution_policy(policy)
    try:
        assert approval._get_approval_mode() == "manual"
    finally:
        reset_room_execution_policy(token)


def test_room_catalog_fails_closed_when_remote_approvals_are_off():
    with pytest.raises(ValueError, match="requires manual or smart approvals"):
        catalog_mapping(
            installation_id="install-peer",
            persistent_process=True,
            target_profile="reviewer",
            execution_policy=_policy(approval_mode="off"),
        )


def test_grant_and_recipient_dispatch_bind_the_exact_policy_digest():
    policy = _policy(max_turns=7)
    dispatch = _dispatch(policy)
    token = issue_room_grant(
        b"s" * 32,
        grant_id="grant-1",
        room_id=dispatch.room_id,
        home_install_id=dispatch.home_install_id,
        authority_gateway_id=dispatch.authority_gateway_id,
        authority_epoch=dispatch.authority_epoch,
        member_id=dispatch.member_id,
        target_install_id=dispatch.target_install_id,
        target_profile=dispatch.target_profile,
        execution_policy_digest=policy["policy_digest"],
        issued_at=100,
        ttl_seconds=60,
    )
    assert (
        verify_room_grant(b"s" * 32, token, dispatch, now=120)[
            "execution_policy_digest"
        ]
        == policy["policy_digest"]
    )

    changed = _dispatch(_policy(max_turns=5))
    with pytest.raises(HostedRoomGrantError, match="scope does not match"):
        verify_room_grant(b"s" * 32, token, changed, now=120)


def test_room_agent_uses_target_policy_toolsets_and_turn_limit(monkeypatch):
    from gateway.platforms.api_server import APIServerAdapter
    from gateway.platforms.base import PlatformConfig

    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    policy = _policy(max_turns=7)
    monkeypatch.setattr("run_agent.AIAgent", FakeAgent)
    monkeypatch.setattr(
        "gateway.run._resolve_runtime_agent_kwargs",
        lambda: {"provider": "openai-codex", "base_url": "https://example.test/v1"},
    )
    monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda: "gpt-test")
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {})
    monkeypatch.setattr(
        "gateway.run.GatewayRunner._load_reasoning_config",
        staticmethod(lambda model="": {"enabled": True, "effort": "high"}),
    )
    monkeypatch.setattr(
        "gateway.run.GatewayRunner._load_fallback_model",
        staticmethod(lambda: None),
    )
    monkeypatch.setattr("gateway.run._current_max_iterations", lambda: 999)
    monkeypatch.setattr(
        "hermes_cli.tools_config._get_platform_tools",
        lambda *_: {"terminal", "file", "web"},
    )
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)

    adapter._create_agent(
        session_id="room-session",
        room_dispatch={"room_id": "room-1"},
        room_execution_policy=policy,
    )

    assert captured["enabled_toolsets"] == policy["enabled_toolsets"]
    assert captured["max_iterations"] == 7
    assert captured["reasoning_config"] == {"enabled": True, "effort": "high"}


def test_policy_drift_requires_reauthorization_without_retry():
    old_policy = _policy(max_turns=7)
    dispatch = _dispatch(old_policy)

    class DriftClient:
        def __init__(self):
            self.dispatches = []

        def dispatch(self, **kwargs):
            self.dispatches.append(kwargs)
            if len(self.dispatches) == 1:
                raise PeerRunsHTTPError(
                    "policy changed",
                    status_code=403,
                    error_code="room_execution_policy_changed",
                    not_admitted=True,
                )
            return {"status": "accepted"}

    refreshed = []
    reauthorization = []
    client = DriftClient()
    tracked = _RouteStatusPeerClient(
        client,
        on_ready=lambda: None,
        on_reauthorization=lambda: reauthorization.append(True),
        on_unavailable=lambda: None,
        on_refreshed=lambda grant, catalog=None: refreshed.append((grant, catalog)),
    )

    with pytest.raises(PeerRunsHTTPError) as caught:
        tracked.dispatch(dispatch=dispatch.as_mapping(), grant="grant-old")

    assert caught.value.needs_reauthorization is True
    assert len(client.dispatches) == 1
    assert refreshed == []
    assert reauthorization == [True]
