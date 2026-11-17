"""One-shot scripting contract: ``--on-approval``, ``--timeout`` and ``--format json`` on
``hermes -z`` / ``hermes chat -q`` (exit codes in ``hermes_cli.turn_exit.EXIT_CODES``)."""
import asyncio
import json

import pytest

from hermes_cli._parser import build_top_level_parser
from hermes_cli.gateway_chat_view import GatewayChatView
from hermes_cli.oneshot_contract import RunPolicy
from hermes_cli.stream_json import EMITTERS


class _Owner:
    """Owner fake: the submitted turn raises one approval; a deny or a cancel settles it cancelled."""

    def __init__(self, *, approval=True):
        self.events, self.calls, self.approval = asyncio.Queue(), [], approval

    def _event(self, kind, payload):
        self.events.put_nowait({"method": "event", "params": {
            "type": kind, "session_id": "stored", "admission_id": "mine", "payload": payload}})

    async def rpc(self, method, **params):
        self.calls.append((method, params))
        if method == "prompt.submit":
            if self.approval:
                self._event("approval.request", {
                    "kind": "approval", "prompt_id": "p1", "execution_generation": 4, "choices": ["once", "deny"],
                    "command": "rm -rf build/", "description": "recursive delete"})
            return {"admission_id": "mine"}
        if method == "prompt.cancel":
            self._event("message.complete", {"text": "", "outcome": "cancelled"})
            return {"status": "terminal"}
        if method == "prompt.receipt":
            return {"status": "terminal", "outcome": "cancelled", "result": {"interrupted": True, "total_tokens": 9}}
        return {}


def _view(owner, argv, fmt="json"):
    args = build_top_level_parser()[0].parse_args(argv)
    view = GatewayChatView(owner, {"stored_session_id": "stored"}, emitter=EMITTERS[fmt](model="m"))
    view.policy = RunPolicy.from_args(args)
    view.unattended = argv[0] == "-z"
    return view


@pytest.mark.asyncio
@pytest.mark.parametrize("argv", [["-z", "clean up", "--on-approval", "fail", "--format", "json"],
                                  ["chat", "-q", "clean up", "--format", "json", "--on-approval", "fail"]])
async def test_on_approval_fail_denies_stops_and_exits_5_with_the_pending_approval(capsys, argv):
    owner = _Owner()
    view = _view(owner, argv)
    assert await asyncio.wait_for(view.run("clean up", oneshot=True), 2) == 5

    submit = dict(owner.calls[0][1])
    assert submit["finite"] is True and submit["on_approval"] == "ask" and "unattended" not in submit
    assert ("approval.respond", {"session_id": "stored", "execution_generation": 4, "prompt_id": "p1",
                                 "choice": "deny"}) in owner.calls
    assert ("prompt.cancel", {"session_id": "stored", "admission_id": "mine"}) in owner.calls
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 1, out  # --format json: one result object, nothing else on stdout
    result = json.loads(out[0])
    assert result["type"] == "result" and result["exit_code"] == 5
    assert result["pending_approval"] == {"prompt_id": "p1", "command": "rm -rf build/",
                                          "description": "recursive delete"}


@pytest.mark.asyncio
async def test_timeout_interrupts_the_turn_and_exits_124(capsys):
    owner = _Owner(approval=False)
    view = _view(owner, ["chat", "-q", "slow", "--timeout", "0.05"], fmt="stream-json")
    assert await asyncio.wait_for(view.run("slow", oneshot=True), 2) == 124
    assert ("prompt.cancel", {"session_id": "stored", "admission_id": "mine"}) in owner.calls
    record = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert record["exit_code"] == 124 and record["timed_out"] is True and record["tokens"]["total"] == 9


def test_default_submissions_keep_their_legacy_approval_fields():
    """No flag: `-z` still auto-approves (``unattended``), `chat -q` sends nothing (config decides)."""
    assert RunPolicy().submit_fields(oneshot_z=True) == {"unattended": True}
    assert RunPolicy().submit_fields(oneshot_z=False) == {}
    assert RunPolicy(on_approval="deny").submit_fields(oneshot_z=True) == {"on_approval": "deny"}
