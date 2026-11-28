"""Opt-in, bounded signal cancellation for a one-shot gateway viewer."""
from __future__ import annotations

import asyncio
import signal
import sys

from hermes_cli.gateway_client import GatewayClientError

# Leave watchdogs time to reap the client even when the owner cannot answer.
CANCEL_TIMEOUT = 3.0
_CANCEL_FAILED = "Cancellation could not be confirmed; work may continue at the gateway."


async def _cancel_own_turn(view, running) -> bool:
    """Use this invocation's admission receipt, never the session's latest generation."""
    submitted = asyncio.create_task(view.submission_ready.wait())
    try:
        if view.finite_admission is None:
            await asyncio.wait({running, submitted}, return_when=asyncio.FIRST_COMPLETED)
        if running.done():
            return False  # a completed view already wrote its terminal result
        admission = view.finite_admission
        # Stop rendering before an awaited RPC lets a completion emit a second terminal result.
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)
        params = {"session_id": view.session_id, "admission_id": admission}
        receipt = await view.client.rpc("prompt.receipt", **params)
        if receipt.get("status") == "queued":
            try:
                # The owner also handles a queue->claim race inside its cancel commit.
                await view.client.rpc("prompt.cancel", **params)
                return True
            except GatewayClientError as exc:
                if str(exc) != "stale_generation":
                    raise
                # Claimed before prompt.cancel reached the owner: read only our own row again.
                receipt = await view.client.rpc("prompt.receipt", **params)
        status = receipt.get("status")
        if status == "terminal":
            return True
        if status != "started" or not isinstance(receipt.get("execution_generation"), int):
            raise GatewayClientError("execution_outcome_unknown")
        try:
            await view.client.rpc("session.interrupt", session_id=view.session_id,
                                  execution_generation=receipt["execution_generation"])
        except GatewayClientError as exc:
            # Completion can win the race. Never retry against a successor generation.
            if str(exc) != "stale_generation" or (await view.client.rpc("prompt.receipt", **params)).get("status") != "terminal":
                raise
        return True
    finally:
        submitted.cancel()
        await asyncio.gather(submitted, return_exceptions=True)


async def run_with_exit_policy(view, query, *, oneshot, interrupt_on_exit):
    if interrupt_on_exit:
        return await run_interruptible(view, query)
    return await view.run(query, oneshot=oneshot)


async def run_interruptible(view, query):
    """Keep the RPC connection alive until our own Stop is acknowledged or the budget expires."""
    from websockets.exceptions import WebSocketException

    loop = asyncio.get_running_loop()
    signalled = asyncio.Event()
    received = []
    previous = {}
    loop_signals = set()

    def on_signal(signum, _frame=None):
        if not received:
            received.append(signum)
            loop.call_soon_threadsafe(signalled.set)

    running = waiter = None
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.getsignal(sig)
            try:
                loop.add_signal_handler(sig, on_signal, sig)  # windows-footgun: ok — NotImplementedError fallback below
                loop_signals.add(sig)
            except NotImplementedError:  # Windows event loops have no signal-handler API.
                signal.signal(sig, on_signal)
        running = asyncio.create_task(view.run(query, oneshot=True))
        waiter = asyncio.create_task(signalled.wait())
        done, _ = await asyncio.wait({running, waiter}, return_when=asyncio.FIRST_COMPLETED)
        if running in done:
            return running.result()
        code, result = 128 + received[0], {"interrupted": True}
        try:
            async with asyncio.timeout(CANCEL_TIMEOUT):
                if not await _cancel_own_turn(view, running):
                    return running.result()
            print("Cancellation requested for this client's turn.", file=sys.stderr)
        except (GatewayClientError, OSError, TimeoutError, WebSocketException):
            print(_CANCEL_FAILED, file=sys.stderr)
            code, result = 1, {"failed": True, "error": _CANCEL_FAILED}
        if view.emitter is not None:
            return view.emitter.emit_result(result, session_id=view.session_id, exit_code=code)
        return code
    finally:
        for task in (running, waiter):
            if task is not None:
                task.cancel()
        await asyncio.gather(*(task for task in (running, waiter) if task is not None), return_exceptions=True)
        for sig, handler in previous.items():
            if sig in loop_signals:
                loop.remove_signal_handler(sig)
            signal.signal(sig, handler)
