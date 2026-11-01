"""``restart-interrupt`` control verb: the restart ``hermes gateway restart`` asks for once the operator
agreed to interrupt the live work it named.

SIGUSR1 carries no arguments, so the in-band restart it starts always waits for in-flight work
(``agent.restart_after_turn_timeout``, 30 min by default) before ``stop()``. After the CLI's live-work
prompt was answered yes (or ``--yes``/``--interrupt`` was passed) that wait broke the promise the
prompt made. This verb starts the same restart with the after-turn wait skipped, or cuts short a wait
already running; ``stop()``'s drain still interrupts the work and parks the sessions for resume.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Callable

RESTART_INTERRUPT_VERB = "restart-interrupt"


def restart_interrupt_verb(runner, loop) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Handler for the control socket: runs on its executor thread, requests the restart on the loop.

    ``params["pid"]`` names the gateway the CLI is about to wait on; any other process declines, so a
    consent never reaches a gateway the operator was not shown."""

    def _handler(params: dict[str, Any]) -> dict[str, Any]:
        pid = os.getpid()
        if params.get("pid") != pid:
            return {"interrupting": False, "pid": pid}
        started: list[bool] = []
        done = threading.Event()

        def _request() -> None:
            try:
                # Before request_restart: a restart already in its after-turn wait reads this and stops waiting.
                runner._restart_interrupt_work = True
                started.append(runner.request_restart(detached=False, via_service=True))
            finally:
                done.set()

        loop.call_soon_threadsafe(_request)
        if not done.wait(timeout=5.0) or not started:
            raise RuntimeError("restart-interrupt: the gateway loop did not answer")
        return {"interrupting": True, "already_restarting": not started[0], "pid": pid}

    return _handler
