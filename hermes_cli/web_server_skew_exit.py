"""Code-skew retirement for Desktop-owned ``hermes serve --isolated`` backends reached over SSH.

Such a backend belongs to a Desktop on another machine: the host's updater never restarts it (only
the remote client holds its token and owner nonce), so after ``hermes update`` it keeps serving the
code it imported at startup against a tree that has moved on, and lazily imports new modules into
an old process. The idle watchdog (``web_server_idle_exit``) only fires once every client has left,
which a connected Desktop never does.

This watchdog polls ``gateway.code_skew`` (the boot revision the serve lifespan records, compared
with the checkout on disk) and, once no update is in flight and the retirement fence proves the
backend idle (and closes admission so no turn can start mid-teardown), exits cleanly. The client's
reconnect path respawns it from the new code. Every uncertain read (unknown revision, busy or
indeterminate ledgers, a live update) keeps the process up.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

_log = logging.getLogger(__name__)

DEFAULT_SKEW_POLL_S = 30.0
# Two consecutive observations: a checkout mid-rewrite (ref file replaced, packed-refs repacked) can
# briefly read differently; a real update reads the new revision on every poll after it finishes.
_CONFIRMATIONS = 2

# ``(boot_rev, disk_rev)`` when the checkout drifted since boot, else None (``detect_code_skew``).
Skew = Optional[tuple[str, str]]


def should_retire_for_skew(*, skew: Skew, update_in_progress: bool) -> bool:
    """Retire only on a proven code change, and never while the tree may still be mid-swap."""
    return bool(skew) and not update_in_progress


def _update_in_progress() -> bool:
    try:
        from hermes_cli.update_lock import read_live_update

        return read_live_update() is not None
    except Exception:
        return True  # cannot prove the swap is over: wait


def start_code_skew_watchdog(server, *, skew_fn: Optional[Callable[[], Skew]] = None,
                             update_probe: Callable[[], bool] = _update_in_progress,
                             fence=None, poll_s: float = DEFAULT_SKEW_POLL_S,
                             max_polls: Optional[int] = None) -> threading.Thread:
    """Daemon thread that sets ``server.should_exit`` once the loaded code is provably stale and
    the backend is provably idle. ``max_polls`` bounds the loop for tests only."""
    if skew_fn is None:
        from gateway.code_skew import detect_code_skew

        skew_fn = detect_code_skew
    if fence is None:
        from hermes_cli.backend_retirement import retirement

        fence = retirement
    read_skew: Callable[[], Skew] = skew_fn

    def _loop() -> None:
        seen = polls = 0
        while not getattr(server, "should_exit", False) and (max_polls is None or polls < max_polls):
            polls += 1
            skew = read_skew()
            if should_retire_for_skew(skew=skew, update_in_progress=update_probe()):
                seen += 1
            else:
                seen = 0
            if seen >= _CONFIRMATIONS and skew:
                permit = fence.prepare()
                if permit.get("ok") and fence.commit(permit.get("token")).get("ok"):
                    _log.warning("SSH-isolated backend loaded %s but the install is now at %s; "
                                 "retiring so its client respawns it on the new code.", *skew)
                    server.should_exit = True
                    return
            time.sleep(poll_s)

    thread = threading.Thread(target=_loop, daemon=True, name="ssh-isolated-skew-watchdog")
    thread.start()
    return thread
