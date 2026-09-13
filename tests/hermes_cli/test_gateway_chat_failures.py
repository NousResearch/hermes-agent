"""Failure-path and terminal-receipt contracts for the transport-only CLI."""
import argparse
import asyncio
import socket

import pytest


def test_direct_query_alias_survives_noninteractive_launch(monkeypatch):
    from hermes_cli import gateway_chat
    from hermes_cli import gateway_chat_startup
    seen = []

    async def run(args):
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
