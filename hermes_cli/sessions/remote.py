"""``hermes sessions send | wait | watch``: drive a gateway-hosted session from a script.

A supervisor (a shell loop, CI, another agent's terminal tool) that holds a session id can add a
prompt to it, block until it settles, or follow its output, without opening a REPL. Each verb is a
thin client of the owner's existing controls (``session.resume``, ``prompt.submit``, the session's
event stream); closing the client never stops the turn.

Exit codes: ``send --wait`` uses the finite chat contract (0 completed, 1 failed, 130 interrupted,
3 detached on a blocking prompt or paused session); ``wait`` and ``watch`` exit 0 once the sessions
are idle; every ``--timeout`` exits 124 with the turn still running; 1 is a gateway or lookup error.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid

from hermes_cli.sessions.target import (
    TIMEOUT_EXIT, attach, busy, paused, run_gateway_verb, session_argument, stored_id,
)

_PAUSED_HINT = ("paused behind a turn whose ending Hermes couldn't confirm; "
                "run `hermes sessions discard {sid} --yes` to let it continue")


def _timed_out(what, seconds):
    print(f"Timed out after {seconds:g}s {what}; nothing was stopped.", file=sys.stderr)
    return TIMEOUT_EXIT


async def _within(seconds, what, work):
    """Await *work*; past *seconds* (None: no deadline) exit 124 and leave the session running."""
    deadline = asyncio.timeout(seconds)
    try:
        async with deadline:
            return await work
    except TimeoutError:
        if deadline.expired():
            return _timed_out(what, seconds)
        raise


async def _send_and_wait(client, args):
    from hermes_cli.gateway_chat_view import GatewayChatView
    snapshot = await attach(client, args.session, omit_messages=True)
    # The finite chat path: the owner treats the turn as single-query (nobody answers a clarify),
    # and the reply, exit code and stderr ``session_id:`` line match `hermes chat --resume <id> -q`.
    view = GatewayChatView(client, snapshot, quiet=True)
    return await view.run(" ".join(args.text), oneshot=True)


async def _send(client, args):
    if args.wait:
        return await _within(args.timeout, "waiting for the reply", _send_and_wait(client, args))
    snapshot = await attach(client, args.session, omit_messages=True)
    sid = stored_id(snapshot)
    if paused(snapshot.get("pending")):
        print("Not sent: session " + _PAUSED_HINT.format(sid=sid) + ".", file=sys.stderr)
        return 3
    receipt = await client.rpc("prompt.submit", session_id=sid, input_id=uuid.uuid4().hex, text=" ".join(args.text))
    if receipt.get("voice_stopped") is True:
        # A bare stop phrase typed into the session's live voice chat ended that chat; no turn was queued.
        payload = {"session_id": sid, "admission_id": None, "status": "voice_stopped"}
    else:
        payload = {"session_id": sid, "admission_id": receipt["admission_id"], "status": receipt.get("status")}
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    elif payload["admission_id"] is None:
        print(f"Not queued: the text ended the voice chat of {sid}.")
    else:
        print(f"Sent to {sid} ({payload['status']}): admission {payload['admission_id']}")
    return 0


class _Waiter:
    """The pending rows of each attached session, kept current from its ``session.info`` events."""

    def __init__(self, client, snapshots):
        self.client = client
        self.pending = {stored_id(s): s.get("pending") or [] for s in snapshots}
        self.changed = asyncio.Event()

    def idle(self):
        return [sid for sid, rows in self.pending.items() if not busy(rows)]

    async def _apply(self, event):
        if isinstance(event, Exception):
            raise event
        params = event.get("params", {})
        sid = params.get("session_id")
        if sid not in self.pending:
            return
        if params.get("type") == "session.info" and "pending" in (params.get("payload") or {}):
            self.pending[sid] = params["payload"]["pending"]
        elif params.get("type") == "session.replay_gap":
            self.pending[sid] = (await self.client.rpc("session.resume", session_id=sid,
                                                       omit_messages=True)).get("pending") or []
        else:
            return
        self.changed.set()

    async def follow(self):
        while True:
            await self._apply(await self.client.events.get())

    async def until(self, enough):
        # Updates that arrived while later sessions were attaching are newer than the earlier
        # snapshots: apply them before the first verdict, so a turn that started meanwhile counts.
        while not self.client.events.empty():
            await self._apply(self.client.events.get_nowait())
        reader = asyncio.create_task(self.follow())
        try:
            while not enough(self.idle()):
                self.changed.clear()
                waiter = asyncio.create_task(self.changed.wait())
                await asyncio.wait({reader, waiter}, return_when=asyncio.FIRST_COMPLETED)
                waiter.cancel()
                if reader.done():
                    reader.result()  # the stream failed: raise it
        finally:
            reader.cancel()


async def _settle(client, args):
    snapshots = [await attach(client, name, omit_messages=True) for name in args.sessions]
    waiter = _Waiter(client, snapshots)
    total = len(waiter.pending)
    await waiter.until((lambda idle: bool(idle)) if args.any else (lambda idle: len(idle) == total))
    for sid in waiter.idle():
        state = _PAUSED_HINT.format(sid=sid) if paused(waiter.pending[sid]) else "idle"
        print(f"{sid}: {state}")
    return 0


async def _wait(client, args):
    # The deadline covers resolving and attaching too: `--timeout 1` never waits out an RPC budget.
    return await _within(args.timeout, "waiting for the sessions to settle", _settle(client, args))


async def _watch(client, args):
    from hermes_cli.gateway_chat_view import GatewayChatView
    snapshot = await attach(client, args.session, omit_messages=True)
    view = GatewayChatView(client, snapshot)
    view.show_pending()
    for prompt in view.prompts.values():
        view.show_prompt(prompt)
    renderer = asyncio.create_task(view.render())
    try:
        while args.follow or busy(view.pending):
            view.changed.clear()
            if view.failure:
                raise view.failure
            await view.changed.wait()
    finally:
        renderer.cancel()
    print(f"\n{view.session_id}: {'paused' if paused(view.pending) else 'idle'}", file=sys.stderr)
    return 0


def _seconds(raw):
    value = float(raw)
    if not value > 0:
        raise argparse.ArgumentTypeError("must be a positive number of seconds")
    return value


def add_remote_parsers(sessions_subparsers) -> None:
    send = sessions_subparsers.add_parser(
        "send", help="Send a prompt to a gateway-hosted session (queued behind a running turn)",
        description="Submit TEXT as the next input of an existing session, as if typed in its chat. "
            "Without --wait it returns once the gateway accepted it. --wait blocks until the turn "
            "settles and prints the reply, with the exit code of `hermes chat --resume <id> -q`.")
    session_argument(send)
    send.add_argument("text", nargs="+", help="The prompt to send")
    send.add_argument("--wait", action="store_true", help="Wait for the reply and print it")
    send.add_argument("--timeout", type=_seconds, metavar="SECS",
                      help="With --wait: give up after SECS (exit 124; the turn keeps running)")
    send.add_argument("--json", action="store_true", help="Without --wait: print the receipt as one JSON line")
    send.set_defaults(sessions_action="send")

    wait = sessions_subparsers.add_parser(
        "wait", help="Block until gateway-hosted sessions have no running or queued turn",
        description="Exit 0 once every named session (or, with --any, the first) is idle; "
            "exit 124 when --timeout runs out first. Nothing is interrupted.")
    wait.add_argument("sessions", nargs="+", metavar="session",
                      help="Session ID, unique ID prefix, title, or '-' for the last session in this directory")
    mode = wait.add_mutually_exclusive_group()
    mode.add_argument("--all", action="store_true", help="Wait for every session (default)")
    mode.add_argument("--any", action="store_true", help="Return as soon as one session is idle")
    wait.add_argument("--timeout", type=_seconds, metavar="SECS", help="Give up after SECS (exit 124)")
    wait.set_defaults(sessions_action="wait")

    watch = sessions_subparsers.add_parser(
        "watch", help="Stream a gateway-hosted session's replies and tool calls until it is idle",
        description="Print the live output of the session's running and queued turns, then exit "
            "once it is idle. --follow keeps watching later turns until Ctrl-C.")
    session_argument(watch)
    watch.add_argument("--follow", "-f", action="store_true", help="Keep watching after the session is idle")
    watch.set_defaults(sessions_action="watch")


_VERBS = {"send": _send, "wait": _wait, "watch": _watch}


def cmd_remote(args) -> int:
    return run_gateway_verb(_VERBS[args.sessions_action], args)
