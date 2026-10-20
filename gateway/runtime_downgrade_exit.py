"""Stop a gateway whose checkout was moved back to a release that predates the gateway runtime.

Downgrading is unsupported, but a manual ``git checkout <older release>`` under a running gateway
leaves this process serving every local session from code the tree no longer has. Older clients
cannot attach to it, and older ``hermes gateway run`` refuses because this process still holds the
profile. Nothing on the older side knows how to stop it. So the gateway watches its own tree, using the boot
revision ``gateway.code_skew`` records. Once the checkout has provably moved (two consecutive
reads, never while an update holds the tree) to code whose state-store modules lack the runtime
ledger, it logs why and stops with the normal drain (a hard exit if that stop fails on the older
tree). It never asks for a restart, so a service manager that relaunches it starts the older
release's own gateway. Every uncertain read keeps it up.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

POLL_S = 30.0
_CONFIRMATIONS = 2
_RUNTIME_SCHEMA_MARKER = "CREATE TABLE IF NOT EXISTS session_admissions"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def tree_predates_runtime(root: Path = _PROJECT_ROOT) -> bool:
    """True only when the checkout's state-store modules are readable and none defines the ledger."""
    try:
        sources = [path.read_text(encoding="utf-8-sig", errors="replace") for path in root.glob("hermes_state*.py")]
    except OSError:
        return False  # mid-checkout or unreadable: proves nothing
    return bool(sources) and not any(_RUNTIME_SCHEMA_MARKER in text for text in sources)


def downgrade_reason(skew, *, update_in_progress: bool, predates: Callable[[], bool]) -> Optional[str]:
    """The log line when this gateway must stop for a downgraded tree, else None."""
    if not skew or update_in_progress or not predates():
        return None
    return ("This gateway loaded {}, but its checkout is now at {}, a release that predates the gateway "
            "runtime. Downgrading is unsupported; stopping (no restart) so the older release can run its "
            "own gateway.".format(*tuple(skew)))


async def _stop_or_exit(runner, *, hard_exit: Callable[[int], None] = os._exit) -> None:
    """The normal drain, and a hard exit if it raises. The stop path lazily imports modules that
    the older tree on disk no longer has; a stop that raised never sets the shutdown event and has
    already disarmed its own watchdog, so without this the process idles on forever."""
    try:
        await runner.stop()
    except Exception:  # health: allow BLE001 -- any failure of the stop on a downgraded tree ends the same way
        logger.exception("Gateway stop failed on the downgraded checkout; exiting now")
        hard_exit(1)


async def runtime_downgrade_watcher(runner, *, poll_s: float = POLL_S, skew_fn=None, update_probe=None,
                                    predates=tree_predates_runtime, max_polls: Optional[int] = None) -> None:
    """Supervised gateway watcher: ``runner.stop()`` after ``_CONFIRMATIONS`` consecutive downgrade reads.

    The probes are bound here, at gateway start: a lazy import after the tree moved would load the
    older release's module into this process."""
    if skew_fn is None:
        from gateway.code_skew import detect_code_skew as skew_fn
    if update_probe is None:
        from hermes_cli.update_lock import update_in_progress as update_probe
    observe_skew: Callable = skew_fn
    observe_update: Callable[[], bool] = update_probe

    def _read() -> Optional[str]:
        try:
            busy = observe_update()
        except Exception:  # health: allow BLE001 -- an unreadable update lock cannot prove the swap is over
            busy = True
        return downgrade_reason(observe_skew(), update_in_progress=busy, predates=predates)

    seen = polls = 0
    while getattr(runner, "_running", False) and (max_polls is None or polls < max_polls):
        polls += 1
        reason = await asyncio.to_thread(_read)
        seen = seen + 1 if reason else 0
        if seen >= _CONFIRMATIONS:
            logger.error("%s", reason)
            runner._exit_reason = reason
            # Detached, strongly held: stop() cancels every background task, this watcher included,
            # and a stop awaited from here would be cancelled with it (the #12875 class).
            runner._runtime_downgrade_stop = asyncio.get_running_loop().create_task(_stop_or_exit(runner))
            return
        await asyncio.sleep(poll_s)
