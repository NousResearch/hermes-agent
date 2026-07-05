"""Asyncio-loop-level reproduction of the gateway.log blocking-write freeze.

Kanban task: ``t_68124348`` — "Reproduce el bloqueo del gateway por stall de log".

Body acceptance criterion (verbatim):
    "una reproducción determinista o semi-determinista que falle con la
    implementación actual y evidencie que el bloqueo del logging afecta la
    responsiveness del gateway."

This test fills the gap between two existing artifacts in the repo:

* ``tests/gateway/test_nonblocking_log_handler.py`` — proves ``logger.info()``
  returns quickly on the calling thread even when the underlying fd is wedged.
  Good, but it does NOT prove the *asyncio event loop* keeps ticking. The
  caller is just the test's main thread, not a coroutine in the gateway.

* ``tests/gateway/test_filehandler_blocking_repro.py`` (task ``t_c992d0e2``) —
  reproduces the *thread-lock* serialization between the inbound and clarify
  paths. Different bug surface: that one is about the stdlib ``Handler.lock``
  serializing two Python threads; THIS one is about the asyncio event loop
  itself going unresponsive while a synchronous ``emit()`` blocks inside
  ``flush()``.

The repro installs ``setup_logging(mode='gateway')`` (either with
``HERMES_LOG_BLOCKING=1`` for the synchronous baseline, or with the default
``_NonBlockingRotatingFileHandler`` for the post-fix variant), swaps the
handler's underlying stream for a hanging fd, then runs an asyncio loop that
hosts:

  1. A periodic heartbeat (``asyncio.sleep`` + counter) — the canary.
  2. A coroutine that calls ``logger.info(...)`` on ``gateway.run`` while the
     fd is wedged — the gateway's critical path.

Two parallel test functions:

  ``test_sync_handler_freezes_asyncio_loop`` (NEGATIVE CONTROL)
      With ``HERMES_LOG_BLOCKING=1`` the handler is the synchronous
      ``_ManagedRotatingFileHandler``. The ``logger.info()`` call inside the
      coroutine blocks for the full 5 seconds (the simulated wedged disk).
      During that window the heartbeat coroutine fires ZERO times — the
      asyncio loop is fully unresponsive.

  ``test_nonblocking_handler_keeps_loop_responsive`` (POSITIVE / REGRESSION)
      Same setup, but using the production ``_NonBlockingRotatingFileHandler``
      (do NOT set ``HERMES_LOG_BLOCKING``). The ``logger.info()`` call returns
      within ``HERMES_LOG_EMIT_TIMEOUT_S + slack``. The heartbeat keeps firing
      ~40–50 times during the 5-second hang window. The asyncio loop is
      healthy even though the underlying fd is still wedged.

Run::

    cd /Users/pones/.hermes/hermes-agent
    pytest tests/gateway/test_gateway_log_asyncio_freeze.py -v -s

Both tests run in < 10 s total on a normal machine. The fixtures are hermetic
(tmp ``HERMES_HOME`` + per-test handler install); root logger is restored in
teardown so sibling tests are not polluted.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from pathlib import Path
from typing import Iterator, List, Tuple

import pytest

import hermes_logging
from hermes_logging import (
    _ManagedRotatingFileHandler,
    _NonBlockingRotatingFileHandler,
    setup_logging,
)


# ----------------------------------------------------------------------
# Synthetic wedged fd — same shape as the sibling regression test, kept
# here as a private copy so this file is fully self-contained.
# ----------------------------------------------------------------------


class _HangingWriteFile:
    """File-like object whose ``write()`` and ``flush()`` block until released.

    Models a wedged disk write: full volume, stalled NFS server, AV scanner
    holding the fd, USB disk glitch.  Threads block on ``write()`` /
    ``flush()``; they are released by calling ``release_writes()``.

    The blocking uses ``threading.Event.wait()`` with a configurable
    timeout, so even if ``release_writes()`` is never called the test does
    not hang past ``release_timeout_s`` — important because the executor
    workers (in the non-blocking path) are daemon threads that survive
    past the test fn and would otherwise pollute pytest's stderr.

    Implements the bare minimum of the file-like protocol to satisfy
    ``RotatingFileHandler.shouldRollover`` (``seek`` / ``tell``) without
    raising — otherwise the rollover check would fast-fail before the
    blocking write is even attempted, and the worker would log an
    AttributeError to stderr instead of actually stalling on the fd.
    """

    def __init__(self, release_timeout_s: float = 5.0) -> None:
        self._released = threading.Event()
        self.writes_attempted = 0
        self._lock = threading.Lock()
        self._release_timeout_s = release_timeout_s
        # A synthetic file-position counter so ``shouldRollover``'s
        # ``seek(0, 2) + tell()`` returns a stable value.  This is the
        # only file-position the simulated disk knows about — the real
        # disk would return the actual byte offset.
        self._pos = 0

    # The handler calls ``self.stream.write(s)`` then ``self.flush()``.
    # Both must block for the simulation to be realistic.
    def write(self, s: str) -> int:  # type: ignore[override]
        with self._lock:
            self.writes_attempted += 1
        self._released.wait(timeout=self._release_timeout_s)
        # Simulate the write completing and growing the file position.
        n = len(s)
        with self._lock:
            self._pos += n
        return n

    def flush(self) -> None:  # type: ignore[override]
        self._released.wait(timeout=self._release_timeout_s)

    def seek(self, offset: int, whence: int = 0) -> int:  # type: ignore[override]
        # Stdlib ``RotatingFileHandler.shouldRollover`` calls
        # ``self.stream.seek(0, 2)`` to position at end-of-file.  We
        # return ``0`` for any seek call — shouldRollover only compares
        # against the ``maxBytes`` threshold; an offset of 0 is fine for
        # our purposes because the gateway.log is created at 0 bytes by
        # the test fixture and we don't actually rotate.
        return 0

    def tell(self) -> int:  # type: ignore[override]
        return 0

    def release_writes(self) -> None:
        self._released.set()


# ----------------------------------------------------------------------
# Fixtures — install gateway logging in an isolated HERMES_HOME and
# replace the gateway.log stream with a hanging fd.
# ----------------------------------------------------------------------


@pytest.fixture
def installed_gateway_logging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> Iterator[Tuple[logging.Logger, _HangingWriteFile, List[logging.Handler]]]:
    """Install gateway logging with a hanging fd; yield ``(logger, hang, handlers)``.

    The fixture honors a per-test marker::

        @pytest.mark.blocking          # force synchronous handler (negative control)
        @pytest.mark.nonblocking       # force the production handler (positive case)

    By default (no marker) we run the *blocking* case because that is the
    scenario the diagnostic flagged as the bug surface.  Tests that exercise
    the post-fix path should mark themselves explicitly.

    The fixture also installs a FRESH process-wide emit executor so this
    test doesn't inherit a stuck or busy executor pool from a previous
    test (notably ``test_filehandler_blocking_repro.py``'s deadlock test,
    which submits ~6 emits that take ~1s each on the executor's worker
    threads).  Without this reset, the very first submit() in the
    non-blocking path would queue behind stuck workers and miss its
    0.5s timeout window.
    """
    # Install a fresh executor for this test, replacing the module-level
    # reference.  The lazy re-creation logic in ``_get_log_emit_executor``
    # uses a lock + double-checked read, so simply setting the module
    # attribute to a fresh pool is enough — the next emit() will use it.
    from concurrent.futures import ThreadPoolExecutor
    fresh_executor = ThreadPoolExecutor(
        max_workers=2,
        thread_name_prefix=f"hermes-log-emit-test-{request.node.name}",
    )
    import hermes_logging as _hermes_logging_mod
    _hermes_logging_mod._log_emit_executor = fresh_executor

    blocking = getattr(request, "param", None) == "blocking" or not getattr(
        request, "param", None
    )
    # When param == "nonblocking" we explicitly do NOT set HERMES_LOG_BLOCKING.
    if blocking:
        monkeypatch.setenv("HERMES_LOG_BLOCKING", "1")
    else:
        monkeypatch.delenv("HERMES_LOG_BLOCKING", raising=False)

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))

    # Reset the module-level init guard so ``setup_logging(force=True)`` actually
    # re-runs (otherwise the second test in a session reuses the first setup).
    hermes_logging._logging_initialized = False
    # Clear all root handlers from prior tests so we don't inherit handlers
    # whose workers are still busy in a module-level executor.  The previous
    # test's setup_logging() calls are idempotent on handler paths but
    # NOT on root.handler list (they skip if same path), so different
    # tmp_path instances leave multiple handlers attached.
    root = logging.getLogger()
    for h in list(root.handlers):
        try:
            h.close()
        except Exception:
            pass
        root.removeHandler(h)
    setup_logging(hermes_home=tmp_path / "hermes_home", mode="gateway", force=True)
    root = logging.getLogger()

    # Pick the right handler class based on the env var we set above.
    if blocking:
        handlers = [
            h
            for h in root.handlers
            if isinstance(h, _ManagedRotatingFileHandler)
            and not isinstance(h, _NonBlockingRotatingFileHandler)
        ]
    else:
        handlers = [
            h
            for h in root.handlers
            if isinstance(h, _NonBlockingRotatingFileHandler)
        ]
    assert handlers, (
        f"Expected at least one {'synchronous' if blocking else 'non-blocking'} "
        f"handler on root; HERMES_LOG_BLOCKING={'1' if blocking else '<unset>'}"
    )

    # Swap the underlying stream for a hanging fd.  This is what the production
    # handler will see on every emit; once the test starts the hang, NO emit
    # can succeed until ``hanging.release_writes()`` is called.
    hanging = _HangingWriteFile()
    for h in handlers:
        h.stream = hanging  # type: ignore[attr-defined]

    gateway_logger = logging.getLogger("gateway.run")

    # Snapshot the root handler list so teardown restores it cleanly.
    original_handlers = list(root.handlers)
    yield gateway_logger, hanging, handlers

    # --- teardown ------------------------------------------------------
    hanging.release_writes()  # let any straggler emits finish
    # Shut down the fresh executor we installed at the top of this fixture.
    # ``wait=True`` blocks until any in-flight worker emits finish; the
    # release above should have unblocked them. ``cancel_futures=True``
    # drops anything still queued (shouldn't be any, but defensive).
    try:
        fresh_executor.shutdown(wait=True, cancel_futures=True)
    except Exception:
        pass
    # Null the module reference so a future test that doesn't install a
    # fresh executor gets one via the lazy path in ``_get_log_emit_executor``.
    _hermes_logging_mod._log_emit_executor = None
    for h in list(root.handlers):
        try:
            h.close()
        except Exception:
            pass
        root.removeHandler(h)
    for h in original_handlers:
        if h not in root.handlers:
            root.addHandler(h)
    hermes_logging._logging_initialized = False


# ----------------------------------------------------------------------
# The asyncio loop probe — runs heartbeats while a coroutine logs.
# ----------------------------------------------------------------------


def run_probe(
    logger: logging.Logger,
    hang_duration_s: float = 5.0,
    heartbeat_period_s: float = 0.05,
    observer_poll_s: float = 0.1,
) -> dict:
    """Run the probe on a background thread; observe from the main thread.

    The probe's asyncio loop runs in a child thread (so the main thread
    can keep ticking).  The main thread polls the probe's state dict
    every ``observer_poll_s`` seconds and stops the probe after
    ``hang_duration_s + 2.0`` seconds (giving it some slack to observe
    the freeze).  The observation timestamps come from the main thread,
    so they are independent of whether the probe's loop is frozen.

    Returns a dict with::

        observer_wall_s        — wall time of the observation window
        heartbeats             — total heartbeats the probe fired
        heartbeats_at_log_start — heartbeats count when do_log started
        log_start              — wall time when do_log started (or None)
        log_end                — wall time when do_log returned (or None)
        log_returned           — True iff do_log returned within the window
        snapshots              — list of (t_s, heartbeats) for the timeline
        probe_thread_alive_at_end — True if the probe is still stuck at end
    """
    state_box: dict = {}
    state_lock = threading.Lock()
    probe_error: dict = {}

    def probe_body() -> None:
        try:
            asyncio.run(
                _asyncio_loop_probe_body(
                    logger, state_box, state_lock,
                    heartbeat_period_s=heartbeat_period_s,
                )
            )
        except BaseException as exc:  # pragma: no cover — defensive
            probe_error["exc"] = exc

    t = threading.Thread(target=probe_body, name="asyncio-loop-probe", daemon=True)
    t0 = time.monotonic()
    t.start()
    # Observe the probe for hang_duration_s + 2.0 seconds.  The main thread
    # is the clock here — even if the probe thread is wedged in a sync
    # handler, our poll loop will keep running.
    total_s = hang_duration_s + 2.0
    deadline = t0 + total_s
    snapshots: List[Tuple[float, int]] = []
    while time.monotonic() < deadline:
        time.sleep(observer_poll_s)
        with state_lock:
            snapshots.append((time.monotonic() - t0, state_box.get("heartbeats", 0)))
    # Give the probe a moment to wrap up.  If the probe thread is wedged in
    # a synchronous emit(), the join will time out and the test still passes
    # (we want to surface the symptom, not kill the test).
    t.join(timeout=2.0)
    with state_lock:
        result = dict(state_box)
    result["snapshots"] = snapshots
    result["probe_thread_alive_at_end"] = t.is_alive()
    result["probe_error"] = probe_error.get("exc")
    result["observer_wall_s"] = time.monotonic() - t0
    return result


async def _asyncio_loop_probe_body(
    logger: logging.Logger,
    state_box: dict,
    state_lock: threading.Lock,
    heartbeat_period_s: float = 0.05,
) -> None:
    """Body of the probe, sharing ``state_box`` with the observer thread.

    Sets up the heartbeat task and the do_log task, then returns control
    to the asyncio scheduler.  The do_log task is what blocks under a
    wedged fd + sync handler; the observer thread measures the freeze.
    """
    state_box.update(
        {
            "heartbeats": 0,
            "heartbeats_at_log_start": 0,
            "heartbeats_at_log_end": 0,
            "log_start": None,
            "log_end": None,
            "log_returned": False,
        }
    )

    async def heartbeat() -> None:
        try:
            while True:
                await asyncio.sleep(heartbeat_period_s)
                with state_lock:
                    state_box["heartbeats"] += 1
        except asyncio.CancelledError:
            return

    async def do_log() -> None:
        with state_lock:
            state_box["log_start"] = time.monotonic()
            state_box["heartbeats_at_log_start"] = state_box["heartbeats"]
        # CRITICAL: this is the gateway's critical-path log call.
        logger.info("inbound message: platform=telegram msg='probe'")
        with state_lock:
            state_box["log_end"] = time.monotonic()
            state_box["heartbeats_at_log_end"] = state_box["heartbeats"]
            state_box["log_returned"] = True

    hb = asyncio.create_task(heartbeat())
    log_task = asyncio.create_task(do_log())
    # Yield a few times so both tasks get scheduled before we park.  We
    # do NOT await log_task here — under a wedged fd + sync handler, that
    # would freeze the loop forever.  Instead we just let the scheduler
    # run for a beat and then park until the observer shuts the loop down.
    for _ in range(3):
        await asyncio.sleep(0)
    # Park until the heartbeat runs long enough for the observer to do
    # its measurement.  The observer is on a different thread and will
    # measure the heartbeat count even if the asyncio loop is frozen, so
    # this park serves the non-blocking case (where the loop is healthy
    # and the heartbeat fires as expected).  Under a wedged sync handler
    # this park is the LAST thing that runs before the loop freezes.
    # ``asyncio.run`` will cancel this sleep when the observer joins on
    # the probe thread and tears the loop down.
    try:
        await asyncio.sleep(60.0)
    except asyncio.CancelledError:
        return


# ----------------------------------------------------------------------
# Test 1: NEGATIVE CONTROL — synchronous handler freezes the loop.
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "installed_gateway_logging", ["blocking"], indirect=True
)
def test_sync_handler_freezes_asyncio_loop(
    installed_gateway_logging, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wedged ``gateway.log`` write on the synchronous handler stops the loop.

    Acceptance criterion: with ``HERMES_LOG_BLOCKING=1`` (the pre-fix code
    path), calling ``logger.info(...)`` from inside a coroutine while the
    underlying fd is wedged must:

      - block the calling coroutine for the full hang duration (≥ 4 s);
      - stop the heartbeat coroutine from firing (0 heartbeats during the log call);
      - keep the process alive (``kill -0`` succeeds) — the gateway
        is "alive but unresponsive", the exact symptom the body asks us
        to evidence.
    """
    gateway_logger, hanging, _handlers = installed_gateway_logging
    hang_duration_s = 5.0

    # Snapshot for after-the-fact assertion that the process IS still alive.
    pid_self = os.getpid()
    assert os.kill(pid_self, 0) is None, "Process must be alive before the test runs"

    # Run the probe on a background thread; observe from the main thread.
    # This is the only correct way to measure a frozen loop: an observer
    # OUTSIDE the loop is needed because the loop is the thing that's
    # broken.
    result = run_probe(gateway_logger, hang_duration_s=hang_duration_s)
    wall = result["observer_wall_s"]

    # Release the hang so the probe thread can wrap up and pytest can exit.
    hanging.release_writes()

    # Print the diagnostic so -s mode makes the symptom obvious.
    print(
        f"\n[SYNC HANDLER PROBE] observer_wall={wall:.2f}s "
        f"heartbeats_total={result['heartbeats']} "
        f"log_returned={result['log_returned']} "
        f"log_start_set={result['log_start'] is not None} "
        f"log_end_set={result['log_end'] is not None} "
        f"loop_responsive={result['log_end'] is not None} "
        f"process_alive={os.kill(pid_self, 0) is None} "
        f"first_5_snapshots={result['snapshots'][:5]} "
        f"last_5_snapshots={result['snapshots'][-5:]}"
    )

    # --- Assertions: the asyncio loop MUST have been frozen. ---------------
    log_start = result["log_start"]
    log_end = result["log_end"]
    heartbeats_total = result["heartbeats"]
    heartbeats_at_start = result["heartbeats_at_log_start"]

    assert log_start is not None, (
        "do_log never recorded its start — the probe didn't actually run."
    )
    assert log_end is None, (
        f"do_log unexpectedly returned at t={log_end:.2f}s (log_start={log_start:.2f}s). "
        f"The sync handler was supposed to block it for ~{hang_duration_s}s. "
        f"If this trips, the negative control has lost its teeth."
    )
    # Heartbeat count between log_start and the end of the observation
    # window: should be 0 (or near-zero) because the loop is frozen.
    heartbeats_after_log_start = heartbeats_total - heartbeats_at_start
    # Allow a small slack: the heartbeat might fire once or twice if the
    # log call has just started and the sleep is already scheduled. But
    # over a 5s window with 0.05s period we expect ~100 ticks; we should
    # see at most 1 or 2.
    assert heartbeats_after_log_start <= 2, (
        f"asyncio loop fired {heartbeats_after_log_start} heartbeats during a "
        f"5s wedged write — the event loop was responsive despite the hang. "
        f"Expected 0-2. The bug surface has changed shape; the repro needs a "
        f"redesign."
    )

    # --- Process alive check: the gateway process would still respond to
    # ``kill -0`` in this state — that's the "alive but unresponsive" symptom
    # the body asks us to evidence.
    assert os.kill(pid_self, 0) is None, (
        "Process should still be alive after a wedged write (the hang is in "
        "the kernel, not user space). If this trips, the test killed itself."
    )


# ----------------------------------------------------------------------
# Test 2: POSITIVE / REGRESSION — the production handler keeps the loop alive.
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "installed_gateway_logging", ["nonblocking"], indirect=True
)
def test_nonblocking_handler_keeps_loop_responsive(
    installed_gateway_logging, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ``_NonBlockingRotatingFileHandler`` keeps the asyncio loop alive.

    Same setup as the negative control but WITHOUT ``HERMES_LOG_BLOCKING=1``
    — the production ``_NonBlockingRotatingFileHandler`` is active. Its
    ``emit()`` submits the actual write to a background thread with a
    deadline (default 0.5 s). The caller returns immediately, the heartbeat
    coroutine keeps firing through the hang, and the asyncio loop stays
    responsive.

    Acceptance: heartbeats_during_log >= 30 (i.e. ≥ 1.5 s of heartbeats at
    0.05 s period) and log_call_elapsed < 1.5 s (the timeout + slack).
    """
    gateway_logger, hanging, _handlers = installed_gateway_logging
    hang_duration_s = 5.0
    timeout_budget_s = float(os.environ.get("HERMES_LOG_EMIT_TIMEOUT_S", "0.5")) + 1.0

    pid_self = os.getpid()

    result = run_probe(gateway_logger, hang_duration_s=hang_duration_s)
    wall = result["observer_wall_s"]

    hanging.release_writes()

    print(
        f"\n[NONBLOCKING HANDLER PROBE] observer_wall={wall:.2f}s "
        f"heartbeats_total={result['heartbeats']} "
        f"log_returned={result['log_returned']} "
        f"log_start_set={result['log_start'] is not None} "
        f"log_end_set={result['log_end'] is not None} "
        f"loop_responsive={result['log_end'] is not None} "
        f"process_alive={os.kill(pid_self, 0) is None} "
        f"first_5_snapshots={result['snapshots'][:5]} "
        f"last_5_snapshots={result['snapshots'][-5:]}"
    )

    # --- Assertions: the asyncio loop MUST have stayed alive. --------------
    log_start = result["log_start"]
    log_end = result["log_end"]
    heartbeats_total = result["heartbeats"]
    heartbeats_at_start = result["heartbeats_at_log_start"]

    assert log_start is not None, "do_log never recorded its start"
    # Note: with the non-blocking handler, do_log returns within the emit
    # timeout (~0.5s) because the future is cancelled. After that the loop
    # is fully unblocked and the heartbeat fires freely.
    assert log_end is not None, (
        f"do_log never returned in {wall:.2f}s — the non-blocking handler is "
        f"not honoring HERMES_LOG_EMIT_TIMEOUT_S ({timeout_budget_s:.2f}s)."
    )
    log_call_elapsed = log_end - log_start
    assert log_call_elapsed < timeout_budget_s, (
        f"do_log took {log_call_elapsed:.2f}s — the non-blocking handler is "
        f"not honoring its timeout."
    )
    # After the log call returned, the loop should have fired many heartbeats.
    # During a 5-second window with the heartbeat firing every 0.05 s, we'd
    # expect ~100 ticks if the loop were entirely free.
    heartbeats_after_log_start = heartbeats_total - heartbeats_at_start
    assert heartbeats_after_log_start >= 30, (
        f"Only {heartbeats_after_log_start} heartbeats during a 5s window — "
        f"the asyncio loop was unresponsive even with the non-blocking handler."
    )


# ----------------------------------------------------------------------
# Test 3: GUARD-RAIL — production setup_logging() must install a non-blocking
# gateway.log handler. Pure structural assertion; no hang, no threads, < 1 s.
# ----------------------------------------------------------------------


def test_setup_logging_installs_nonblocking_gateway_handler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Production ``setup_logging(mode='gateway')`` must NOT install a plain
    ``_ManagedRotatingFileHandler`` for ``gateway.log``.

    Body acceptance criterion (verbatim): "el test falla contra la
    implementación antigua".  This is the cheap, structural half of that
    promise — it runs in < 1 s, doesn't touch asyncio, and trips the
    moment someone reverts ``_add_rotating_handler`` back to a vanilla
    ``RotatingFileHandler`` (or otherwise drops the non-blocking wrapper).

    We assert the *positive* shape: at least one
    ``_NonBlockingRotatingFileHandler`` on the root logger after a default
    ``setup_logging()``.  If only the synchronous handler is present (the
    pre-fix shape), the test fails with a message that names the commit
    that introduced the fix so the failure is actionable.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
    # Explicitly do NOT set HERMES_LOG_BLOCKING — we want the default
    # production setup.  If the env var leaks from a parent session, the
    # test's own assertion below catches it.
    monkeypatch.delenv("HERMES_LOG_BLOCKING", raising=False)
    hermes_logging._logging_initialized = False
    # Drop any handlers inherited from a sibling test (same process).
    root = logging.getLogger()
    for h in list(root.handlers):
        try:
            h.close()
        except Exception:
            pass
        root.removeHandler(h)

    setup_logging(hermes_home=tmp_path / "hermes_home", mode="gateway", force=True)
    root = logging.getLogger()

    nonblocking_handlers = [
        h for h in root.handlers if isinstance(h, _NonBlockingRotatingFileHandler)
    ]
    synchronous_handlers = [
        h
        for h in root.handlers
        if isinstance(h, _ManagedRotatingFileHandler)
        and not isinstance(h, _NonBlockingRotatingFileHandler)
    ]

    # The diagnostic: print what got installed so the failure mode is
    # immediately obvious in -v mode.
    print(
        f"\n[GUARD-RAIL PROBE] "
        f"nonblocking={len(nonblocking_handlers)} synchronous={len(synchronous_handlers)} "
        f"HERMES_LOG_BLOCKING={os.environ.get('HERMES_LOG_BLOCKING', '<unset>')!r}"
    )

    assert nonblocking_handlers, (
        "REGRESSION: setup_logging(mode='gateway') did not install any "
        "_NonBlockingRotatingFileHandler on the root logger. "
        f"Found {len(synchronous_handlers)} synchronous handler(s) instead. "
        "This means a wedged gateway.log write would freeze the asyncio "
        "event loop — exactly the bug that a835f97 ('non-blocking FileHandler "
        "emit prevents asyncio freeze') fixed. Revert the revert, or check "
        "_add_rotating_handler() in hermes_logging.py."
    )
    # Belt-and-braces: the sync path must NOT be the default.  If both are
    # installed (unlikely but possible if a refactor accidentally leaves
    # the old handler attached alongside the new one), we still flag it.
    assert not synchronous_handlers, (
        f"REGRESSION: setup_logging(mode='gateway') installed "
        f"{len(synchronous_handlers)} synchronous handler(s) alongside the "
        "non-blocking one. The sync handler will receive emits on the "
        "main asyncio thread and freeze the loop on a wedged disk."
    )

    # Teardown: keep sibling tests hermetic.
    for h in list(root.handlers):
        try:
            h.close()
        except Exception:
            pass
        root.removeHandler(h)
    hermes_logging._logging_initialized = False


# ----------------------------------------------------------------------
# Standalone entry point — runs both probes without pytest so an operator
# can copy/paste the command from a runbook.
# ----------------------------------------------------------------------


def _standalone_probe(blocking: bool, hang_s: float = 5.0) -> dict:
    """Run the asyncio loop probe outside pytest. Returns a dict of metrics."""
    import tempfile
    import shutil

    tmp = Path(tempfile.mkdtemp(prefix="gateway-log-freeze-"))
    try:
        # Wire logging the same way the fixture does.
        os.environ["HERMES_HOME"] = str(tmp / "hermes_home")
        if blocking:
            os.environ["HERMES_LOG_BLOCKING"] = "1"
        else:
            os.environ.pop("HERMES_LOG_BLOCKING", None)
        hermes_logging._logging_initialized = False
        setup_logging(hermes_home=tmp / "hermes_home", mode="gateway", force=True)
        root = logging.getLogger()
        if blocking:
            handlers = [
                h for h in root.handlers
                if isinstance(h, _ManagedRotatingFileHandler)
                and not isinstance(h, _NonBlockingRotatingFileHandler)
            ]
        else:
            handlers = [
                h for h in root.handlers
                if isinstance(h, _NonBlockingRotatingFileHandler)
            ]
        hanging = _HangingWriteFile()
        for h in handlers:
            h.stream = hanging  # type: ignore[attr-defined]
        gateway_logger = logging.getLogger("gateway.run")

        # Run the probe via the threaded observer — same path the pytest
        # tests use, so the standalone main() and the pytest tests exercise
        # the same code.
        result = run_probe(gateway_logger, hang_duration_s=hang_s)
        hanging.release_writes()

        result["blocking"] = blocking
        result["process_alive"] = os.kill(os.getpid(), 0) is None
        return result
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    """Run both probes from the command line and print a summary.

    Usage::

        python tests/gateway/test_gateway_log_asyncio_freeze.py

    Exit code 0 iff both probes matched expectations.
    """
    print("=" * 78)
    print("Gateway.log asyncio-loop freeze reproduction (kanban t_68124348)")
    print("=" * 78)

    sync = _standalone_probe(blocking=True, hang_s=5.0)
    print()
    print("[1/2] SYNCHRONOUS handler (HERMES_LOG_BLOCKING=1, pre-fix code path):")
    print(
        f"      observer_wall={sync['observer_wall_s']:.2f}s  "
        f"heartbeats_total={sync['heartbeats']}"
    )
    print(
        f"      do_log_returned={sync['log_returned']}  "
        f"do_log_start_at={sync['log_start']}  "
        f"do_log_end_at={sync['log_end']}"
    )
    print(
        f"      loop_responsive={sync['log_end'] is not None}  "
        f"process_alive={sync['process_alive']}"
    )

    nonblock = _standalone_probe(blocking=False, hang_s=5.0)
    print()
    print("[2/2] NON-BLOCKING handler (production _NonBlockingRotatingFileHandler):")
    print(
        f"      observer_wall={nonblock['observer_wall_s']:.2f}s  "
        f"heartbeats_total={nonblock['heartbeats']}"
    )
    print(
        f"      do_log_returned={nonblock['log_returned']}  "
        f"do_log_start_at={nonblock['log_start']}  "
        f"do_log_end_at={nonblock['log_end']}"
    )
    print(
        f"      loop_responsive={nonblock['log_end'] is not None}  "
        f"process_alive={nonblock['process_alive']}"
    )

    print()
    print("=" * 78)
    print("Verdict:")
    sync_loop_froze = (sync["log_end"] is None) and (
        sync["heartbeats"] - sync["heartbeats_at_log_start"] <= 2
    )
    nonblock_loop_alive = (
        nonblock["log_end"] is not None
        and (nonblock["heartbeats"] - nonblock["heartbeats_at_log_start"]) >= 30
    )
    print(f"  sync handler froze the asyncio loop: {sync_loop_froze}")
    print(f"  non-blocking handler kept loop alive: {nonblock_loop_alive}")
    print(
        f"  process stayed alive in both cases: "
        f"{sync['process_alive'] and nonblock['process_alive']}"
    )
    print("=" * 78)

    ok = (
        sync_loop_froze
        and sync["process_alive"]
        and nonblock_loop_alive
        and nonblock["process_alive"]
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())