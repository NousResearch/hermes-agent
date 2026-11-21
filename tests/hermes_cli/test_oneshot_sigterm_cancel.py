"""Regression tests for client-side SIGTERM / SIGINT cancellation on one-shot runs."""

import argparse
import asyncio
from contextlib import asynccontextmanager
import json
import os
import signal

import pytest

from hermes_cli import gateway_chat
import hermes_cli.gateway_chat_startup as startup
from hermes_cli.gateway_chat_view import GatewayChatView


@pytest.mark.skipif(not hasattr(signal, "SIGTERM"), reason="POSIX SIGTERM required")
def test_oneshot_sigterm_cancels_inflight_admission_detaches_and_writes_usage_file(tmp_path, capsys):
    calls = []
    usage_path = tmp_path / "usage-sigterm.json"

    async def scenario():
        submitted = asyncio.Event()

        class Peer:
            def __init__(self):
                self.events = asyncio.Queue()

            async def rpc(self, method, **params):
                calls.append((method, params))
                if method == "prompt.submit":
                    submitted.set()
                    return {"admission_id": "adm-sigterm"}
                if method == "prompt.cancel":
                    return {"status": "terminal", "outcome": "cancelled"}
                if method == "session.detach":
                    return {"detached": True}
                return {}

        snapshot = {
            "stored_session_id": "local-sigterm-1",
            "subscription_id": "sub-sigterm-1",
            "info": {"model": "glm-5.3-flash", "source": "oneshot"},
        }
        view = GatewayChatView(Peer(), snapshot, quiet=True, usage_file=str(usage_path))
        view.unattended = True

        async def send_sigterm():
            await submitted.wait()
            await asyncio.sleep(0.02)
            os.kill(os.getpid(), signal.SIGTERM)

        killer = asyncio.create_task(send_sigterm())
        exit_code = await asyncio.wait_for(view.run("long running task", oneshot=True), timeout=5)
        await killer
        return exit_code

    exit_code = asyncio.run(scenario())

    assert exit_code == 128 + int(signal.SIGTERM)
    methods = [m for m, _ in calls]
    assert methods == ["prompt.submit", "prompt.cancel", "session.detach"]
    assert calls[1][1] == {"session_id": "local-sigterm-1", "admission_id": "adm-sigterm", "_timeout": 2.0}
    assert calls[2][1] == {
        "session_id": "local-sigterm-1",
        "subscription_id": "sub-sigterm-1",
        "end_reason": "sigterm",
        "_timeout": 2.0,
    }

    report = json.loads(usage_path.read_text(encoding="utf-8"))
    assert report["session_id"] == "local-sigterm-1"
    assert report["completed"] is False
    assert report["interrupted"] is True
    assert report["failed"] is True
    assert report["turn_exit_reason"] == "sigterm"
    assert "Terminated (sigterm)" in capsys.readouterr().err


def test_oneshot_sigint_cancels_inflight_admission_and_writes_usage_file(monkeypatch, tmp_path, capsys):
    calls = []

    class Peer:
        def __init__(self):
            self.events = asyncio.Queue()

        async def rpc(self, method, **params):
            calls.append((method, params))
            if method == "runtime.describe":
                return {"session_create": {"sources": ["cli", "oneshot"], "parameters": ["cwd", "request_id", "source"]}}
            if method == "session.create":
                return {
                    "stored_session_id": "local-sigint-1",
                    "subscription_id": "sub-sigint-1",
                    "info": {"model": "glm-5.3-flash", "source": "oneshot"},
                }
            if method == "prompt.submit":
                async def raise_interrupt():
                    await asyncio.sleep(0.02)
                    for task in asyncio.all_tasks():
                        if task is not asyncio.current_task():
                            task.cancel()
                asyncio.create_task(raise_interrupt())
                return {"admission_id": "adm-sigint"}
            return {}

    @asynccontextmanager
    async def connected():
        yield Peer()

    monkeypatch.setattr(startup, "ensure_launch_provider", lambda _args: True)
    monkeypatch.setattr(gateway_chat, "_register_terminal_process", lambda: None)
    monkeypatch.setattr(gateway_chat, "connect_gateway", connected)

    # Simulate asyncio.run raising KeyboardInterrupt after task cancellation unwinds.
    real_run = asyncio.run

    def run_converting_cancel(coro):
        try:
            return real_run(coro)
        except asyncio.CancelledError as exc:
            raise KeyboardInterrupt from exc

    monkeypatch.setattr(gateway_chat.asyncio, "run", run_converting_cancel)

    usage_path = tmp_path / "usage-sigint.json"
    args = argparse.Namespace(oneshot="do work", usage_file=str(usage_path), output_format="text", model=None)
    code = gateway_chat.launch_from_args(args)

    assert code == 130
    methods = [m for m, _ in calls]
    assert "prompt.cancel" in methods
    assert "session.detach" in methods
    report = json.loads(usage_path.read_text(encoding="utf-8"))
    assert report["session_id"] == "local-sigint-1"
    assert report["interrupted"] is True
    assert report["completed"] is False
    assert report["turn_exit_reason"] == "sigint"
    assert "Cancelled one-shot run." in capsys.readouterr().err
