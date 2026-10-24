"""Finite gateway chat tracks session-blocking controls without leaking foreign output."""

import asyncio

import pytest

from hermes_cli.gateway_chat_view import GatewayChatView


def _control_event(kind, *, admission="neighbor", prompt_id="p1"):
    control = "approval" if kind.startswith("approval") else "clarify"
    payload = {
        "prompt_id": prompt_id,
        "kind": control,
        "execution_generation": 7,
    }
    if kind.endswith(".request"):
        if control == "approval":
            payload.update(command="secret command", choices=["yes", "no"])
        else:
            payload.update(question="secret question")
    return {
        "method": "event",
        "params": {
            "type": kind,
            "session_id": "stored",
            "admission_id": admission,
            "payload": payload,
        },
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["approval.request", "clarify.request"])
@pytest.mark.parametrize("timing", ["before_receipt", "after_receipt"])
async def test_oneshot_tracks_foreign_blocking_control_and_detaches(
    capsys, kind, timing
):
    class Peer:
        events = asyncio.Queue()

        async def _after_receipt(self):
            await asyncio.sleep(0)
            self.events.put_nowait({
                "method": "event",
                "params": {
                    "type": "message.delta",
                    "session_id": "stored",
                    "admission_id": "neighbor",
                    "payload": {"text": "neighbor output"},
                },
            })
            self.events.put_nowait(_control_event(kind))

        async def rpc(self, method, **params):
            assert method == "prompt.submit"
            if timing == "before_receipt":
                self.events.put_nowait({
                    "method": "event",
                    "params": {
                        "type": "message.delta",
                        "session_id": "stored",
                        "admission_id": "neighbor",
                        "payload": {"text": "neighbor output"},
                    },
                })
                self.events.put_nowait(_control_event(kind))
                await asyncio.sleep(0)
            else:
                asyncio.create_task(self._after_receipt())
            return {"admission_id": "mine"}

    view = GatewayChatView(Peer(), {"stored_session_id": "stored"}, quiet=True)
    assert await asyncio.wait_for(view.run("query", oneshot=True), 2) == 3

    captured = capsys.readouterr()
    assert "neighbor output" not in captured.out
    assert "secret command" not in captured.err
    assert "secret question" not in captured.err
    assert "Input required; detached without cancelling" in captured.err


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["approval.settled", "clarify.settled"])
async def test_oneshot_applies_foreign_control_settlement_before_receipt(kind):
    control = "approval" if kind.startswith("approval") else "clarify"
    pending = {
        "prompt_id": "p1",
        "kind": control,
        "execution_generation": 7,
        **({"command": "old command", "choices": ["yes", "no"]}
           if control == "approval" else {"question": "old question"}),
    }

    class Peer:
        events = asyncio.Queue()

        async def rpc(self, method, **params):
            if method == "prompt.receipt":
                return {"status": "terminal", "result": {"completed": True, "final_response": "done"}}
            assert method == "prompt.submit"
            self.events.put_nowait(_control_event(kind))
            self.events.put_nowait({
                "method": "event",
                "params": {
                    "type": "message.complete",
                    "session_id": "stored",
                    "admission_id": "mine",
                    "payload": {"text": "done", "outcome": "completed"},
                },
            })
            await asyncio.sleep(0)
            return {"admission_id": "mine"}

    snapshot = {"stored_session_id": "stored", "prompts": [pending]}
    view = GatewayChatView(Peer(), snapshot, quiet=True)
    assert await asyncio.wait_for(view.run("query", oneshot=True), 2) == 0
    assert "p1" not in view.prompts


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["approval.settled", "clarify.settled"])
async def test_finite_renderer_applies_foreign_settlement_after_receipt(kind):
    control = "approval" if kind.startswith("approval") else "clarify"

    class Peer:
        events = asyncio.Queue()

    view = GatewayChatView(
        Peer(),
        {
            "stored_session_id": "stored",
            "prompts": [{
                "prompt_id": "p1",
                "kind": control,
                "execution_generation": 7,
            }],
        },
        quiet=True,
    )
    view.finite = True
    view.finite_admission = "mine"
    renderer = asyncio.create_task(view.render())
    try:
        view.client.events.put_nowait(_control_event(kind))
        await asyncio.wait_for(view.changed.wait(), 2)
        assert "p1" not in view.prompts
    finally:
        renderer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await renderer
