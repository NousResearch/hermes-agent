"""Diagnostic watchdog for post-response CPU stalls (#32079).

Issue #32079 sampled a stalled Hermes gateway whose entire active stack
was inside Python's ``re`` engine after a Codex Responses API call had
already returned successfully.  The reporter saw no follow-up tool
execution log and no ``Turn ended`` log; the gateway became silent
until the process was restarted.  Other Python threads in the same
process were sampled blocked on ``take_gil``, which is consistent with
a CPU-bound regex path holding the GIL.

The post-response phase between API success and tool dispatch runs
several regex-heavy passes on assistant content that the agent does
not control: leaked tool-call detection, MEDIA tag extraction,
truncation heuristics, and downstream gateway delivery rewriters.
Any one of those, given a sufficiently adversarial input, can take
long enough to look like a hang to the user with no diagnostic.

This module installs a one-shot watchdog around such phases.  When
the wrapped block runs longer than ``timeout_s``, a daemon thread
dumps the tracebacks of *every* Python thread to ``log_path`` and
emits a structured warning naming the phase.  ``faulthandler``'s
``dump_traceback`` walks frames without acquiring the GIL — by
design, so it works exactly when the main thread is wedged in a
C extension — which is what makes it the right primitive here.

Usage::

    with post_response_watchdog(label="codex.normalize", timeout_s=30):
        normalized = normalize_response(...)

The watchdog is opt-in via a context manager so callers stay free
to decide which phases need the diagnostic.  When the timer fires
it neither cancels nor interrupts the block — it only emits a
diagnostic so the next user report can point at the stuck phase
instead of leaving operators to read ``sample`` output.

Environment knobs:

* ``HERMES_POST_RESPONSE_WATCHDOG_DISABLED=1`` — turn the watchdog
  off entirely (legacy / debugging escape hatch).
* ``HERMES_POST_RESPONSE_DUMP_PATH`` — override the default dump
  destination (``$HERMES_HOME/logs/post-response-stall.log`` when
  the env var is unset, falling back to ``/tmp`` if the resolved
  log directory cannot be created).
"""

from __future__ import annotations

import faulthandler
import logging
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_S = 30.0
_DUMP_DEDUPE_WINDOW_S = 60.0


def _resolve_dump_path() -> Path:
    """Return the path the watchdog should append stack dumps to.

    Order:
      1. ``HERMES_POST_RESPONSE_DUMP_PATH`` env var.
      2. ``$HERMES_HOME/logs/post-response-stall.log`` when
         ``HERMES_HOME`` is set and writable.
      3. ``/tmp/hermes-post-response-stall.log`` as a last-resort
         fallback that always works.
    """
    explicit = os.environ.get("HERMES_POST_RESPONSE_DUMP_PATH")
    if explicit:
        return Path(explicit)
    hermes_home = os.environ.get("HERMES_HOME")
    if hermes_home:
        candidate = Path(hermes_home) / "logs" / "post-response-stall.log"
        try:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            return candidate
        except OSError:
            pass
    return Path("/tmp/hermes-post-response-stall.log")


def _is_disabled() -> bool:
    return os.environ.get("HERMES_POST_RESPONSE_WATCHDOG_DISABLED", "").strip() in {
        "1", "true", "yes", "on",
    }


# Dedupe identical (label, dump_path) firings so a real stall doesn't
# spam the log with hundreds of identical dumps if the wedged thread
# eventually unsticks while many sibling watchdogs are still armed.
_LAST_DUMP_AT: dict[tuple[str, str], float] = {}
_LAST_DUMP_LOCK = threading.Lock()


def _should_emit_dump(label: str, dump_path: Path) -> bool:
    key = (label, str(dump_path))
    now = time.monotonic()
    with _LAST_DUMP_LOCK:
        last = _LAST_DUMP_AT.get(key, 0.0)
        if now - last < _DUMP_DEDUPE_WINDOW_S:
            return False
        _LAST_DUMP_AT[key] = now
    return True


def _emit_stall_dump(label: str, started_at: float, timeout_s: float, dump_path: Path) -> None:
    """Append a labelled stack dump for every thread to ``dump_path``."""
    if not _should_emit_dump(label, dump_path):
        logger.warning(
            "[post-response-watchdog] %s exceeded %.1fs (dedup'd, see %s)",
            label, timeout_s, dump_path,
        )
        return
    elapsed = time.monotonic() - started_at
    header = (
        f"\n=== POST-RESPONSE STALL ({label}) "
        f"elapsed={elapsed:.1f}s timeout={timeout_s:.1f}s "
        f"pid={os.getpid()} ts={time.time():.0f} ===\n"
    )
    try:
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dump_path, "a", encoding="utf-8") as fp:
            fp.write(header)
            try:
                # all_threads=True so the dump captures the thread that
                # actually holds the GIL, not just the watchdog's own
                # frames.  ``dump_traceback`` does not acquire the GIL
                # and is safe to call from a background thread when the
                # interpreter is blocked.
                faulthandler.dump_traceback(file=fp, all_threads=True)
            except Exception as exc:  # pragma: no cover — diagnostic best-effort
                fp.write(f"<faulthandler.dump_traceback failed: {exc!r}>\n")
            fp.flush()
    except OSError as exc:
        logger.warning(
            "[post-response-watchdog] %s exceeded %.1fs but stall dump to %s failed: %s",
            label, timeout_s, dump_path, exc,
        )
        return
    logger.warning(
        "[post-response-watchdog] %s exceeded %.1fs — stack dump written to %s "
        "(issue #32079: Codex post-response regex stall diagnostic)",
        label, timeout_s, dump_path,
    )


@contextmanager
def post_response_watchdog(
    label: str,
    *,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    dump_path: Optional[Path] = None,
) -> Iterator[None]:
    """Arm a one-shot stack-dump timer around a CPU-bound block.

    Emits a single diagnostic dump if the wrapped block runs longer
    than ``timeout_s`` seconds.  The block itself is *not* cancelled —
    the watchdog only produces evidence the operator can attach to a
    bug report.

    Safe to nest, safe across threads (each call gets its own timer).
    Cheap when the block completes promptly: the timer is cancelled
    on exit and no dump is written.
    """
    if _is_disabled() or timeout_s <= 0:
        yield
        return

    resolved_dump = dump_path or _resolve_dump_path()
    started_at = time.monotonic()

    def _on_timeout() -> None:
        _emit_stall_dump(label, started_at, timeout_s, resolved_dump)

    timer = threading.Timer(timeout_s, _on_timeout)
    timer.daemon = True
    timer.start()
    try:
        yield
    finally:
        timer.cancel()


__all__ = ["post_response_watchdog"]
