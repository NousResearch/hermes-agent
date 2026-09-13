"""Client launch must not silently downgrade policy or own execution."""
import argparse

import pytest


def test_unsupported_launch_options_fail_before_connection(monkeypatch, capsys):
    from hermes_cli import gateway_chat
    calls = []
    monkeypatch.setattr(gateway_chat, "connect_gateway", lambda: calls.append(True))
    for option in ("yolo", "worktree", "usage_file", "run_budget"):
        args = argparse.Namespace(**{option: True})
        assert gateway_chat.launch_from_args(args) == 2
        assert option.replace("_", "-") in capsys.readouterr().err
    # Bare -c (breadcrumb/MRU) and a nameless --create-if-missing stay refused; only a
    # titled -c is a gateway-resolvable selector.
    assert gateway_chat.launch_from_args(argparse.Namespace(continue_last=True)) == 2
    assert "--continue" in capsys.readouterr().err
    assert gateway_chat.launch_from_args(argparse.Namespace(create_if_missing=True)) == 2
    assert "create-if-missing" in capsys.readouterr().err
    assert gateway_chat.launch_from_args(argparse.Namespace(continue_last="named", model="m", query="x")) == 2
    assert calls == []
    assert gateway_chat.launch_from_args(argparse.Namespace(resume="stored", source="tui", query="x")) == 2
    # Bypass launches read no profile default model, so one must be explicit.
    assert gateway_chat.launch_from_args(argparse.Namespace(safe_mode=True, query="x")) == 1
    assert "--model" in capsys.readouterr().err
    assert calls == []


def test_refusals_name_the_replacement_and_a_runnable_safe_mode_example(monkeypatch, capsys):
    """An exit-2 refusal tells the user where every refused flag went; the safe-mode
    refusal prints a command they can run as-is."""
    from hermes_cli import gateway_chat
    monkeypatch.setattr(gateway_chat, "connect_gateway", lambda: pytest.fail("connected"))
    assert gateway_chat.launch_from_args(argparse.Namespace(yolo=True, run_budget=30.0)) == 2
    err = capsys.readouterr().err
    assert "--yolo: use" in err and "approvals.mode" in err
    assert "--run-budget: use" in err and "run_budget_seconds" in err
    for name in gateway_chat._UNSUPPORTED:
        assert name in gateway_chat._RELOCATED, f"{name} refused without saying where it went"
    assert gateway_chat.launch_from_args(argparse.Namespace(safe_mode=True, query="x")) == 1
    err = capsys.readouterr().err
    example = next(line.split("Example: ", 1)[1] for line in err.splitlines() if "Example: " in line)
    import shlex
    from hermes_cli._parser import build_top_level_parser
    parsed = build_top_level_parser()[0].parse_args(shlex.split(example)[1:])
    assert parsed.safe_mode and parsed.model and parsed.provider and parsed.query


@pytest.mark.asyncio
async def test_creation_preserves_advertised_cwd_model_and_toolsets(monkeypatch, tmp_path):
    from contextlib import asynccontextmanager
    from hermes_cli import gateway_chat
    from hermes_cli.gateway_chat_view import GatewayChatView
    calls = []

    class Peer:
        async def rpc(self, method, **params):
            calls.append((method, params))
            if method == "runtime.describe":
                return {"session_create": {"sources": ["cli"], "parameters": ["cwd", "model", "toolsets", "request_id", "source"]}}
            return {"stored_session_id": "stored"}

    @asynccontextmanager
    async def connected():
        yield Peer()

    async def rendered(self, query, *, oneshot):
        assert query == "literal"
        return 0

    monkeypatch.setattr(gateway_chat, "connect_gateway", connected)
    monkeypatch.setattr(GatewayChatView, "run", rendered)
    monkeypatch.chdir(tmp_path)
    args = argparse.Namespace(query="literal", model="explicit-model", toolsets="terminal, file", quiet=True)
    assert await gateway_chat.run_gateway_chat(args) == 0
    create = calls[1][1]
    assert create["cwd"] == str(tmp_path)
    assert create["model"] == "explicit-model"
    assert create["toolsets"] == ["terminal", "file"]
    assert create["source"] == "cli" and create["request_id"]


@pytest.mark.asyncio
async def test_rpc_preserves_notifications_and_errors():
    import asyncio
    import json
    from websockets.asyncio.server import serve
    from websockets.asyncio.client import connect
    from hermes_cli.gateway_client import GatewayClient, GatewayClientError

    async def peer(ws):
        async for raw in ws:
            request = json.loads(raw)
            await ws.send(json.dumps({"method": "message.complete", "params": {"text": "reply"}}))
            await ws.send(json.dumps({"id": request["id"], "error": {"message": "stale_generation"}}))

    async with serve(peer, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        async with connect(f"ws://127.0.0.1:{port}") as ws:
            async with GatewayClient(ws) as client:
                with pytest.raises(GatewayClientError, match="stale_generation"):
                    await client.rpc("session.interrupt", session_id="stored", execution_generation=4)
                event = await asyncio.wait_for(client.events.get(), 2)
                assert event["params"]["text"] == "reply"
