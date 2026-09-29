"""Plain and encrypted approval cards exercised by a separate persisted client."""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest
from testcontainers.core.container import DockerContainer
from testcontainers.core.network import Network

from tests.integration.matrix_live.conftest import (
    ApprovalGateway, LinuxNioObserver, LiveRoom, _create_live_room, _synapse_server,
)
from plugins.platforms.matrix.approval_cards import force_redact_command


@pytest.fixture(scope="module")
def synapse(docker_engine: None) -> Iterator[tuple[DockerContainer, str, Network]]:
    with _synapse_server(extra_config=(
        "rc_registration:\n  per_second: 100\n  burst_count: 100\n"
        "rc_login:\n  address:\n    per_second: 100\n    burst_count: 100\n"
    )) as server:
        yield server


@pytest.fixture
def live_room(synapse: tuple[DockerContainer, str, Network]) -> LiveRoom:
    return _create_live_room(synapse, unique_accounts=True)


@pytest.mark.parametrize("approval_gateway", [
    (encrypted, decision)
    for encrypted in (False, True)
    for decision in ("once", "deny", "expired", "summarized")
], indirect=True, ids=[f"{encrypted}-{decision}" for encrypted in (False, True) for decision in ("once", "deny", "expired", "summarized")])
def test_requester_controls_exact_cards_and_receives_terminal_replacements(
    approval_gateway: ApprovalGateway,
    live_room: LiveRoom,
    linux_nio_observer: LinuxNioObserver,
) -> None:
    gateway = approval_gateway
    other = gateway.other_user
    payload = {
        "room_id": live_room.room_id,
        "bot_device": live_room.bot.device_id,
        "bot_user": live_room.bot.user_id,
        "other_login": {"user_id": other.user_id, "device_id": other.device_id, "access_token": other.access_token},
        "decision": gateway.decision,
        "encrypted": gateway.encrypted,
    }
    code = (
        "import asyncio, json\n"
        "from approval_client import exercise\n"
        f"payload = json.loads({json.dumps(json.dumps(payload))})\n"
        "result = asyncio.run(exercise(**payload))\n"
        "print(json.dumps(result))\n"
    )
    compile(code, "matrix-approval-client", "exec")
    try:
        output = linux_nio_observer.run_python(code)
    except AssertionError as exc:
        pytest.fail(f"{exc}\nApproval gateway diagnostics:\n{gateway.diagnostics()}", pytrace=False)
    result = json.loads(output.strip().splitlines()[-1])
    assert result["cards"] == (2 if gateway.decision == "once" else 1)
    requests = gateway.model.main_requests()
    assert len(requests) == (4 if gateway.decision == "once" else 2)
    positions = [("first", 0, 3), ("second", 1, 2)] if gateway.decision == "once" else [("first", 0, 1)]
    for marker, initial, final in positions:
        first_messages = requests[initial]["messages"]
        assert any(message["role"] == "user" and result["anchors"][marker]["body"] in str(message["content"]) for message in first_messages)
        assert requests[final]["messages"][:len(first_messages)] == first_messages
        other_marker = "second" if marker == "first" else "first"
        assert f"[in:approval-{other_marker}]" not in json.dumps(requests[final]["messages"])
    results = [
        message for _, _, final in positions
        for message in requests[final]["messages"] if message["role"] == "tool"
    ]
    assert len(results) == result["cards"]
    assert len({message["tool_call_id"] for message in results}) == len(results)
    outputs = [json.loads(message["content"]) for message in results]
    tool_result_details = force_redact_command(json.dumps({"raw_tool_results": results, "parsed_tool_results": outputs}, indent=2))
    if gateway.decision in {"once", "summarized"}:
        executed = "approval-second-ran" if gateway.decision == "once" else "approval-first-ran"
        assert sum(executed in str(output.get("output", "")) for output in outputs) == 1, f"{tool_result_details}\nApproval gateway diagnostics:\n{gateway.diagnostics()}"
    else:
        assert all(output.get("exit_code") != 0 for output in outputs)
    assert len(gateway.model.aux_requests()) == (1 if gateway.decision == "summarized" else 0)
