"""Native ACP choices change actual requests to a recording loopback model endpoint."""

import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import sys

import pytest

from tests.acp_adapter.test_gateway_sessions import ACPPeer, editor
from tests.acp_adapter.model_metadata_fixtures import (
    CHOICE_A, CHOICE_B, CHOICE_OFF, MODEL_A, MODEL_B, MODEL_OFF, PROVIDER,
)


def _configure(home, base_url):
    config = {
        "gateway": {"multiplex_profiles": False},
        "model": {"provider": PROVIDER, "default": MODEL_A},
        "agent": {"reasoning_effort": "low", "api_max_retries": 1},
        "platform_toolsets": {"acp": [], "cli": []},
        "auxiliary": {"title_generation": {"enabled": False}},
        "providers": {
            "relay:one": {
                "name": "Relay one", "api": base_url, "api_key": "fixture-only", "discover_models": False,
                "models": {
                    MODEL_A: {"name": "First", "group": "source-a", "context_length": 64000,
                              "reasoning_efforts": ["low", "high"]},
                    MODEL_B: {"name": "Second", "group": "source-b", "context_length": 64000,
                              "reasoning_efforts": ["medium"]},
                    MODEL_OFF: {"name": "Plain", "group": "source-b", "context_length": 64000,
                                "reasoning_efforts": []},
                },
            },
        },
    }
    (home / "config.yaml").write_text(json.dumps(config))


def _selectors(response):
    assert "result" in response, response
    return {option["category"]: option for option in response["result"]["configOptions"]}


async def _turn(peer, session_id, marker):
    reply = await peer.rpc("session/prompt", sessionId=session_id, prompt=[{"type": "text", "text": marker}])
    assert reply.get("result", {}).get("stopReason") == "end_turn", reply


async def _exercise(peer, cwd):
    assert "result" in await peer.rpc("initialize", protocolVersion=1, clientCapabilities={})
    created = await peer.rpc("session/new", cwd=cwd, mcpServers=[])
    options = _selectors(created)
    session_id = created["result"]["sessionId"]
    assert created["result"]["models"]["currentModelId"] == options["model"]["currentValue"] == CHOICE_A
    groups = {group["name"]: group for group in options["model"]["options"]}
    assert groups["Relay one · source-a"]["options"][0]["value"] == CHOICE_A
    assert groups["Relay one · source-b"]["options"][0]["value"] == CHOICE_B
    assert groups["Relay one · source-a"]["group"] != groups["Relay one · source-b"]["group"]
    assert [choice["value"] for choice in options["thought_level"]["options"]] == ["low", "high"]

    picked = await peer.rpc("session/set_config_option", sessionId=session_id,
                            configId="reasoning_effort", value="high")
    assert _selectors(picked)["thought_level"]["currentValue"] == "high"
    refused = await peer.rpc("session/set_config_option", sessionId=session_id,
                             configId="reasoning_effort", value="medium")
    assert "error" in refused, refused
    await _turn(peer, session_id, "WS_SHARED_METADATA_FIRST")

    forked = await peer.rpc("session/fork", sessionId=session_id, cwd=cwd, mcpServers=[])
    assert _selectors(forked)["thought_level"]["currentValue"] == "high"
    await _turn(peer, forked["result"]["sessionId"], "WS_SHARED_METADATA_FORK")

    picked = await peer.rpc("session/set_config_option", sessionId=session_id, configId="model", value=CHOICE_B)
    options = _selectors(picked)
    assert options["model"]["currentValue"] == CHOICE_B
    assert options["thought_level"]["currentValue"] == "medium"
    assert [choice["value"] for choice in options["thought_level"]["options"]] == ["medium"]
    await _turn(peer, session_id, "WS_SHARED_METADATA_SECOND")

    picked = await peer.rpc("session/set_config_option", sessionId=session_id, configId="model", value=CHOICE_OFF)
    assert "thought_level" not in _selectors(picked)
    refused = await peer.rpc("session/set_config_option", sessionId=session_id,
                             configId="reasoning_effort", value="high")
    assert "error" in refused, refused
    await _turn(peer, session_id, "WS_SHARED_METADATA_OFF")

    await peer.rpc("session/set_config_option", sessionId=session_id, configId="model", value=CHOICE_A)
    await peer.rpc("session/set_config_option", sessionId=session_id, configId="reasoning_effort", value="high")
    updates = [frame["params"]["update"] for frame in peer.frames if frame.get("method") == "session/update"
               and frame["params"]["update"].get("sessionUpdate") == "config_option_update"]
    assert any(next(option for option in update["configOptions"] if option["category"] == "model")["currentValue"]
               == CHOICE_B for update in updates), updates
    return session_id


def _assert_requests(requests):
    expected = [
        ("WS_SHARED_METADATA_FIRST", MODEL_A, "high"),
        ("WS_SHARED_METADATA_FORK", MODEL_A, "high"),
        ("WS_SHARED_METADATA_SECOND", MODEL_B, "medium"),
        ("WS_SHARED_METADATA_OFF", MODEL_OFF, "none"),
        ("WS_SHARED_METADATA_REATTACH", MODEL_A, "high"),
    ]
    for marker, model, effort in expected:
        matches = [body for body in requests if marker in json.dumps(next(
            (message.get("content") for message in reversed(body.get("messages", [])) if message.get("role") == "user"),
            None))]
        assert matches, (marker, requests)
        assert (matches[-1]["model"], matches[-1].get("reasoning_effort")) == (model, effort), matches[-1]


def _evidence(surface, frames, requests):
    root = os.environ.get("HERMES_E2E_ARTIFACTS")
    if root:
        path = Path(root)
        path.mkdir(parents=True, exist_ok=True)
        (path / f"acp-model-metadata-{surface}.json").write_text(
            json.dumps({"surface": surface, "frames": frames, "requests": requests}, indent=2))


@asynccontextmanager
async def _inprocess_editor(home, user, tmp_path):
    root = Path(__file__).resolve().parents[2]
    env = {key: os.environ[key] for key in ("PATH", "LANG", "TZ") if key in os.environ}
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONUNBUFFERED="1")
    script = (
        "import asyncio,acp;from acp_adapter.server import HermesACPAgent;"
        "from acp_adapter.session import SessionManager;"
        "asyncio.run(acp.run_agent(HermesACPAgent(session_manager=SessionManager()),use_unstable_protocol=True))"
    )
    with (tmp_path / "inprocess-acp.log").open("a") as log:
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-c", script, cwd=root, env=env,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=log)
        peer = ACPPeer(process)
        try:
            yield peer
        finally:
            process.stdin.close()
            try:
                await asyncio.wait_for(process.wait(), 10)
            except TimeoutError:
                process.kill()
                await process.wait()
            await peer.reader


@pytest.mark.asyncio
async def test_inprocess_stdio_metadata_controls_reach_the_recording_endpoint(tmp_path, model_peer):
    home, user = tmp_path / "home", tmp_path / "user"
    home.mkdir()
    user.mkdir()
    _configure(home, f"http://127.0.0.1:{model_peer.server_port}/v1")
    frames = []
    async with _inprocess_editor(home, user, tmp_path) as peer:
        session_id = await _exercise(peer, str(tmp_path))
        frames.extend(peer.frames)
    async with _inprocess_editor(home, user, tmp_path) as peer:
        await peer.rpc("initialize", protocolVersion=1, clientCapabilities={})
        loaded = await peer.rpc("session/load", sessionId=session_id, cwd=str(tmp_path), mcpServers=[])
        assert _selectors(loaded)["thought_level"]["currentValue"] == "high"
        await _turn(peer, session_id, "WS_SHARED_METADATA_REATTACH")
        frames.extend(peer.frames)
    _assert_requests(model_peer.requests)
    _evidence("inprocess", frames, model_peer.requests)


@pytest.mark.asyncio
async def test_gateway_stdio_metadata_controls_reach_the_recording_endpoint(daemon, tmp_path, model_peer):
    home = daemon[0]
    _configure(home, f"http://127.0.0.1:{model_peer.server_port}/v1")
    frames = []
    async with editor(daemon, tmp_path) as peer:
        session_id = await _exercise(peer, str(tmp_path))
        frames.extend(peer.frames)
    async with editor(daemon, tmp_path) as peer:
        await peer.rpc("initialize", protocolVersion=1, clientCapabilities={})
        loaded = await peer.rpc("session/load", sessionId=session_id, cwd=str(tmp_path), mcpServers=[])
        assert _selectors(loaded)["thought_level"]["currentValue"] == "high"
        await _turn(peer, session_id, "WS_SHARED_METADATA_REATTACH")
        frames.extend(peer.frames)
    _assert_requests(model_peer.requests)
    _evidence("gateway", frames, model_peer.requests)


@pytest.mark.asyncio
async def test_a_gateway_session_starting_on_an_empty_effort_list_disables_reasoning(daemon, tmp_path, model_peer):
    home = daemon[0]
    _configure(home, f"http://127.0.0.1:{model_peer.server_port}/v1")
    config = json.loads((home / "config.yaml").read_text())
    config["model"]["default"] = MODEL_OFF
    config["agent"]["reasoning_effort"] = "high"
    (home / "config.yaml").write_text(json.dumps(config))
    async with editor(daemon, tmp_path) as peer:
        await peer.rpc("initialize", protocolVersion=1, clientCapabilities={})
        created = await peer.rpc("session/new", cwd=str(tmp_path), mcpServers=[])
        assert "thought_level" not in _selectors(created)
        await _turn(peer, created["result"]["sessionId"], "WS_SHARED_EMPTY_DEFAULT")
        body = model_peer.requests[-1]
        assert (body["model"], body.get("reasoning_effort")) == (MODEL_OFF, "none"), body
        _evidence("gateway-empty-default", peer.frames, model_peer.requests)
