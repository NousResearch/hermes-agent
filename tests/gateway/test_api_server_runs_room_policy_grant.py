"""/v1/runs honours a room execution policy only when it comes from a verified room grant.

Without a ``HermesRoom`` grant the body's ``hosted_room_dispatch`` / ``_room_execution_policy``
are refused, so an API key cannot pick its own toolsets, approval mode or turn limit; the
operator's ``platform_toolsets.api_server`` config applies to ordinary runs.
"""

import asyncio
import hashlib
from unittest.mock import MagicMock, patch

import pytest
from aiohttp.test_utils import TestClient, TestServer

from gateway.hosted_room_execution_policy import POLICY_VERSION, _policy_digest
from tests.gateway.test_api_server_runs import (
    _create_runs_app,
    _make_adapter,
    _use_idempotency_db,
)


def _forged_policy():
    unsigned = {
        "version": POLICY_VERSION,
        "target_profile": "default",
        "enabled_toolsets": ["bot_room", "terminal"],
        "approval_mode": "off",
        "max_iterations": 500,
    }
    return {**unsigned, "policy_digest": _policy_digest(unsigned)}


def _forged_dispatch(prompt):
    return {
        "protocol_version": 2,
        "room_id": "room-1",
        "home_install_id": "install-home",
        "authority_gateway_id": "gateway-home",
        "authority_epoch": 1,
        "member_id": "member-reviewer",
        "target_install_id": "install-target",
        "target_profile": "default",
        "task_id": "task-1",
        "execution_generation": 1,
        "source_event_seq": 1,
        "cancellation_scope_id": "cancel-1",
        "prompt": prompt,
        "prompt_digest": hashlib.sha256(prompt.encode()).hexdigest(),
        "capability_digest": "c" * 64,
        "execution_policy_digest": _forged_policy()["policy_digest"],
        "trace_id": "trace-1",
    }


def _agent():
    agent = MagicMock()
    agent.run_conversation.return_value = {"final_response": "ok"}
    agent.session_prompt_tokens = agent.session_completion_tokens = (
        agent.session_total_tokens
    ) = 0
    return agent


async def _wait_completed(cli, run_id, headers):
    body = None
    for _ in range(60):
        response = await cli.get(f"/v1/runs/{run_id}", headers=headers)
        body = await response.json()
        if body.get("status") == "completed":
            return body
        await asyncio.sleep(0.05)
    raise AssertionError(f"run {run_id} did not complete: {body}")


@pytest.fixture
def adapter(tmp_path):
    adapter = _make_adapter(api_key="sk-secret")
    _use_idempotency_db(adapter, tmp_path / "idem.db")
    return adapter


BEARER = {"Authorization": "Bearer sk-secret"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "room_fields",
    [
        pytest.param(
            lambda prompt: {
                "hosted_room_dispatch": _forged_dispatch(prompt),
                "_room_execution_policy": _forged_policy(),
            },
            id="dispatch-and-policy",
        ),
        pytest.param(
            lambda prompt: {"_room_execution_policy": _forged_policy()},
            id="policy-only",
        ),
        pytest.param(
            lambda prompt: {"hosted_room_dispatch": _forged_dispatch(prompt)},
            id="dispatch-only",
        ),
    ],
)
async def test_body_room_policy_without_grant_is_refused(adapter, room_fields):
    prompt = "run something"
    app = _create_runs_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        with patch.object(adapter, "_create_agent", return_value=_agent()) as create:
            response = await cli.post(
                "/v1/runs",
                json={"input": prompt, **room_fields(prompt)},
                headers=BEARER,
            )
            body = await response.json()
    # A refusal is answered before any run is reserved, so no agent can start later.
    assert response.status == 403
    assert body["error"]["code"] == "room_grant_required"
    create.assert_not_called()


@pytest.mark.asyncio
async def test_ordinary_run_uses_operator_toolsets_not_a_room_policy(adapter):
    app = _create_runs_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        with patch.object(adapter, "_create_agent", return_value=_agent()) as create:
            response = await cli.post(
                "/v1/runs", json={"input": "hello"}, headers=BEARER
            )
            assert response.status == 202
            await _wait_completed(cli, (await response.json())["run_id"], BEARER)
    kwargs = create.call_args.kwargs
    assert kwargs["room_dispatch"] is None
    assert kwargs["room_execution_policy"] is None


def test_create_agent_without_room_dispatch_keeps_operator_toolsets(monkeypatch):
    """Even a stray policy kwarg is inert without a verified dispatch."""
    from gateway import run as gateway_run
    from hermes_cli import tools_config

    adapter = _make_adapter(api_key="sk-secret")
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", dict)
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda *a: "fixture")
    monkeypatch.setattr(gateway_run, "_load_gateway_config", dict)
    monkeypatch.setattr(gateway_run, "_current_max_iterations", lambda: 7)
    monkeypatch.setattr(
        tools_config, "_get_platform_tools", lambda config, platform: {"web"}
    )
    with patch("run_agent.AIAgent") as agent_cls:
        adapter._create_agent(room_execution_policy=_forged_policy())
    kwargs = agent_cls.call_args.kwargs
    assert kwargs["enabled_toolsets"] == ["web"]
    assert kwargs["max_iterations"] == 7


@pytest.mark.asyncio
async def test_valid_grant_applies_the_server_derived_policy(adapter):
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
            headers=BEARER,
        )
        assert invitation.status == 201
        invitation_body = await invitation.json()
        grant, catalog = invitation_body["grant"], invitation_body["catalog"]
        prompt = "Review this room message."
        dispatch = {
            **_forged_dispatch(prompt),
            "task_id": "task-room-1",
            "target_install_id": catalog["installation_id"],
            "capability_digest": catalog["catalog_digest"],
            "execution_policy_digest": catalog["execution_policy"]["policy_digest"],
        }
        room_headers = {"Authorization": f"HermesRoom {grant}"}
        with patch.object(adapter, "_create_agent", return_value=_agent()) as create:
            response = await cli.post(
                "/v1/runs",
                json={"input": prompt, "hosted_room_dispatch": dispatch},
                headers={**room_headers, "Idempotency-Key": "room:task-room-1:1"},
            )
            assert response.status == 202, await response.text()
            await _wait_completed(cli, (await response.json())["run_id"], room_headers)
    kwargs = create.call_args.kwargs
    assert kwargs["room_dispatch"]["task_id"] == "task-room-1"
    assert kwargs["room_execution_policy"] == catalog["execution_policy"]
