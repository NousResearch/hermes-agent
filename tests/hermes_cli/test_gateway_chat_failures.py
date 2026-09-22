"""Failure-path and terminal-receipt contracts for the transport-only CLI."""
import argparse
import asyncio
import socket

import pytest


def test_direct_query_alias_survives_noninteractive_launch(monkeypatch):
    from hermes_cli import gateway_chat
    from hermes_cli import gateway_chat_startup
    seen = []

    async def run(args, emitter=None):
        seen.append(args.q)
        return 0

    # The alias contract is independent of whether this machine has a provider configured.
    monkeypatch.setattr(gateway_chat_startup, "ensure_launch_provider", lambda args: True)
    monkeypatch.setattr(gateway_chat, "run_gateway_chat", run)
    assert gateway_chat.launch_from_kwargs({"q": "literal"}) == 0
    assert seen == ["literal"]


@pytest.mark.asyncio
async def test_explicit_remote_failure_never_ensures_local(monkeypatch):
    from hermes_cli.gateway_client import connect_gateway, GatewayClientError
    from hermes_cli import gateway_runtime

    def forbidden(*args, **kwargs):
        raise AssertionError("Remote failure invoked local lifecycle")

    monkeypatch.setattr(gateway_runtime, "ensure_gateway_runtime", forbidden)
    with socket.socket() as unavailable:
        unavailable.bind(("127.0.0.1", 0))
        monkeypatch.setenv("HERMES_TUI_GATEWAY_URL", f"ws://127.0.0.1:{unavailable.getsockname()[1]}/api/ws?token=private-test-value")
        with pytest.raises(GatewayClientError, match="no local fallback") as caught:
            async with connect_gateway():
                raise AssertionError("Unavailable remote unexpectedly connected")
        assert "private-test-value" not in str(caught.value)


@pytest.mark.asyncio
async def test_oneshot_matches_own_terminal_receipt_not_neighbor(capsys):
    from hermes_cli.gateway_chat_view import GatewayChatView

    class Peer:
        events = asyncio.Queue()
        async def rpc(self, method, **params):
            assert method == "prompt.submit"
            for admission, outcome in (("neighbor", "completed"), ("mine", "failed")):
                self.events.put_nowait({"method": "event", "params": {
                    "type": "message.complete", "session_id": "stored", "admission_id": admission,
                    "payload": {"text": admission, "outcome": outcome}}})
            return {"admission_id": "mine"}

    view = GatewayChatView(Peer(), {"stored_session_id": "stored"}, quiet=True)
    assert await asyncio.wait_for(view.run("query", oneshot=True), 2) == 1
    assert capsys.readouterr().out == "mine\n"


@pytest.mark.asyncio
async def test_oneshot_refuses_unknown_blocked_session_before_submitting(capsys):
    """After a SIGKILL mid-turn the head admission is ``unknown``; a new -q admission would queue
    behind it forever. One-shot refuses BEFORE submitting (exit 3) and names the discard remedy."""
    from hermes_cli.gateway_chat_view import GatewayChatView

    class Peer:
        events = asyncio.Queue()
        async def rpc(self, method, **params):
            raise AssertionError(f"nothing may be submitted behind an unknown row: {method}")

    lost = "admission-unknown-0123456789abcdef"
    snapshot = {"stored_session_id": "stored", "pending": [{"admission_id": lost, "status": "unknown", "execution_generation": 4}]}
    assert await asyncio.wait_for(GatewayChatView(Peer(), snapshot, quiet=True).run("pong", oneshot=True), 2) == 3
    err = capsys.readouterr().err
    assert f"/discard {lost}" in err and "prompt.resolve_unknown" in err and "nothing was submitted" in err


@pytest.mark.asyncio
@pytest.mark.parametrize("blocker", ["unknown_row", "approval_prompt"])
async def test_stream_json_oneshot_always_closes_with_a_result_record(capsys, blocker):
    """``--format stream-json`` consumers parse stdout: every exit-3 detach (unknown row refused,
    approval needed) must end the JSONL with a failed ``result`` record carrying exit_code 3."""
    import json
    from hermes_cli.gateway_chat_view import GatewayChatView
    from hermes_cli.stream_json import StreamJsonEmitter

    class Peer:
        events = asyncio.Queue()
        async def rpc(self, method, **params):
            assert blocker == "approval_prompt" and method == "prompt.submit"
            self.events.put_nowait({"method": "event", "params": {
                "type": "approval.request", "session_id": "stored", "admission_id": "mine",
                "payload": {"prompt_id": "p1", "kind": "approval", "command": "rm -rf /", "choices": ["yes", "no"],
                            "execution_generation": 1}}})
            return {"admission_id": "mine"}

    pending = [{"admission_id": "lost", "status": "unknown", "execution_generation": 4}] if blocker == "unknown_row" else []
    emitter = StreamJsonEmitter(model="m", session_id="stored")
    view = GatewayChatView(Peer(), {"stored_session_id": "stored", "pending": pending}, emitter=emitter)
    assert await asyncio.wait_for(view.run("pong", oneshot=True), 2) == 3
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert records[0]["type"] == "system" and records[-1]["type"] == "result", records
    assert records[-1]["exit_code"] == 3 and records[-1]["error"], records[-1]
    assert sum(r["type"] == "result" for r in records) == 1
