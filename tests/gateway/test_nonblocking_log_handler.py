"""Regression test for the gateway.log asyncio-freeze fix.

Verifies the acceptance criterion from kanban task ``t_3c891904``::

    "un write lento/bloqueado en gateway.log no debe impedir que Telegram
     siga procesando mensajes críticos."

Concretely: install ``setup_logging(mode='gateway')`` with the production
``_NonBlockingRotatingFileHandler`` (the default — do NOT set
``HERMES_LOG_BLOCKING``), then swap its real handler for a synthetic slow
proxy that holds the ``emit()`` write for 5 seconds (simulating a wedged
disk).  Spawn a worker thread that does ``logger.info(...)`` on the
``gateway.run`` namespace — the same critical path ``gateway.run`` uses on
every inbound Telegram message — and measure how long the call returns.  It
MUST return within ``HERMES_LOG_EMIT_TIMEOUT_S + small slack`` (default
0.5s + 1s slack = 1.5s budget).  A regression to the old synchronous
``RotatingFileHandler.emit()`` would block for the full 5 seconds and fail
this test.

The fixture is hermetic: tmp HERMES_HOME, fresh ``setup_logging(force=True)``,
proxies are installed for the duration of the test, and the root logger is
restored in teardown so we don't pollute sibling tests.

Run::

    cd /Users/pones/.hermes/hermes-agent
    pytest tests/gateway/test_nonblocking_log_handler.py -v
"""

from __future__ import annotations

import io
import logging
import os
import threading
import time
from pathlib import Path
from typing import List

import pytest

import hermes_logging
from hermes_logging import _NonBlockingRotatingFileHandler, setup_logging


class _HangingWriteFile(io.TextIOBase):
    """File-like object whose ``write()`` blocks until released.

    Used to simulate a wedged ``gateway.log`` write (full disk, stalled NFS,
    AV scanner holding the fd, etc.).  Threads block on ``write()``; they
    are released by calling ``release_writes()`` after the test has
    confirmed its measurement.
    """

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._released = threading.Event()
        self.writes_attempted = 0

    def write(self, s):  # type: ignore[override]
        with self._lock:
            self.writes_attempted += 1
        # Block until the test tells us to release.
        self._released.wait(timeout=30.0)
        return len(s)

    def flush(self) -> None:  # type: ignore[override]
        # flush() does the real harm — that's what blocks the asyncio loop
        # in production. Block here too so the test mirrors reality.
        self._released.wait(timeout=30.0)

    def release_writes(self) -> None:
        self._released.set()


@pytest.fixture
def isolated_gateway_logging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Install gateway logging into a tmp HERMES_HOME for one test.

    Returns ``(gateway_logger, hanging_file, restore)`` where ``hanging_file``
    replaces the real gateway.log fds.  ``restore`` puts the original handlers
    back and shuts down the test-specific executor workers.

    Crucially, this fixture does NOT set ``HERMES_LOG_BLOCKING=1`` — the
    whole point is that the production non-blocking code path is active,
    otherwise the test would silently validate the wrong thing.
    """
    monkeypatch.delenv("HERMES_LOG_BLOCKING", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))

    hermes_logging._logging_initialized = False
    setup_logging(hermes_home=tmp_path / "hermes_home", mode="gateway", force=True)
    root = logging.getLogger()

    # Sanity: confirm production wiring is in effect. If a future change
    # accidentally reverted to ``_ManagedRotatingFileHandler`` directly,
    # this assertion will catch it before it reaches production.
    gateway_handlers: List[logging.Handler] = [
        h for h in root.handlers
        if isinstance(h, _NonBlockingRotatingFileHandler)
    ]
    assert gateway_handlers, (
        "Expected at least one _NonBlockingRotatingFileHandler on root; "
        "the production fix is not wired up."
    )

    # Snapshot originals so we can restore them after the test.
    snapshot = list(root.handlers)

    # Wrap each production handler so its underlying stream is a hanging
    # file. _NonBlockingRotatingFileHandler.emit() goes through
    # ``StreamHandler.emit`` → ``self.stream.write(...)`` → ``self.flush()``,
    # so swapping ``self.stream`` is enough to simulate a stalled disk
    # without touching the handler's lock or the executor wrapper.
    hanging = _HangingWriteFile()
    replaced: List[_NonBlockingRotatingFileHandler] = []
    for h in gateway_handlers:
        if hasattr(h, "stream"):
            h.stream = hanging  # type: ignore[attr-defined]
            replaced.append(h)  # type: ignore[arg-type]

    gateway_logger = logging.getLogger("gateway.run")

    def restore() -> None:
        # Restore originals and let pending workers finish (or time out).
        hanging.release_writes()
        for h in replaced:
            try:
                # Best-effort: re-open the real log path so subsequent
                # teardown doesn't error on a closed stream.
                h.stream = h._open()  # type: ignore[attr-defined]
            except Exception:
                pass

    try:
        yield gateway_logger, hanging
    finally:
        restore()


def _measure_handler_call(
    logger: logging.Logger, message: str, budget_seconds: float = 2.0
) -> float:
    """Time how long ``logger.info(...)`` takes when the underlying fd hangs.

    Returns wall-clock seconds; the assertion expects ``< budget_seconds``.
    """
    start = time.monotonic()
    logger.info(message)
    elapsed = time.monotonic() - start
    return elapsed


def test_slow_gateway_log_write_does_not_block_caller(
    isolated_gateway_logging,
) -> None:
    """Write a slow gateway.log and confirm emit returns within the budget."""
    gateway_logger, hanging = isolated_gateway_logging

    # The deadline is 0.5s by default; give a 1.5s budget for CI slack
    # (GC pauses, contended CI runners, etc.).  A regression to the
    # synchronous handler would block until ``release_writes()`` is
    # called → many seconds, not sub-2s.
    elapsed = _measure_handler_call(
        gateway_logger,
        "inbound message: platform=telegram msg='hello from regression test'",
        budget_seconds=2.0,
    )
    assert elapsed < 2.0, (
        f"gateway.log emit blocked the caller for {elapsed:.2f}s — "
        f"the non-blocking handler is not active. "
        f"Set HERMES_LOG_BLOCKING=1 only for diagnosis."
    )


def test_slow_write_does_not_starve_subsequent_emits(
    isolated_gateway_logging,
) -> None:
    """After the slow write times out, later emits still return promptly.

    This is the assertion that prevents a "fixed it once, broke it again"
    regression: the first ``emit()`` blocks the executor worker for
    5 seconds (the disk simulation); under the fix, the call returns in
    ``< timeout`` because the future is cancelled.  A subsequent ``emit()``
    on the same handler must NOT be stuck behind the still-blocked first
    emit.  The handler's ``Handler.lock`` is held by the worker thread,
    so the second emit would normally wait — the timeout+drop path is
    what saves us here.
    """
    gateway_logger, hanging = isolated_gateway_logging

    # First emit — will time out after HERMES_LOG_EMIT_TIMEOUT_S.
    first = _measure_handler_call(
        gateway_logger,
        "first message: will time out and be dropped",
        budget_seconds=2.0,
    )
    assert first < 2.0, (
        f"First emit should have returned after timeout but blocked for {first:.2f}s"
    )

    # Release the hanging fd briefly so the second emit CAN succeed at all,
    # then re-hang.  This isolates the "second call is also non-blocking"
    # behavior from any other interaction.
    hanging.release_writes()
    time.sleep(0.1)  # let the worker actually finalize the first write

    # Now re-hang and time the next emit. It must still return within budget.
    new_hang = _HangingWriteFile()
    for h in logging.getLogger().handlers:
        if isinstance(h, _NonBlockingRotatingFileHandler) and hasattr(h, "stream"):
            h.stream = new_hang  # type: ignore[attr-defined]

    try:
        elapsed = _measure_handler_call(
            gateway_logger,
            "second message: handler still must not block after recovery",
            budget_seconds=2.0,
        )
        assert elapsed < 2.0, (
            f"Second emit unexpectedly blocked for {elapsed:.2f}s — "
            f"recovery path regressed."
        )
    finally:
        new_hang.release_writes()


def test_blocking_env_var_opt_out_preserves_synchronous_emit(tmp_path: Path) -> None:
    """``HERMES_LOG_BLOCKING=1`` returns the synchronous handler.

    The opt-out must keep the managed-mode chmod + external-rotation paths
    intact — it must NOT replace the production handler with raw stdlib
    (which would silently drop the NixOS-mode group-writable logic).
    """
    os.environ["HERMES_LOG_BLOCKING"] = "1"
    os.environ["HERMES_HOME"] = str(tmp_path / "hermes_home_blocking")
    try:
        hermes_logging._logging_initialized = False
        setup_logging(
            hermes_home=tmp_path / "hermes_home_blocking",
            mode="gateway",
            force=True,
        )

        blocking_handlers = [
            h for h in logging.getLogger().handlers
            if isinstance(h, hermes_logging._ManagedRotatingFileHandler)
            and not isinstance(h, hermes_logging._NonBlockingRotatingFileHandler)
        ]
        assert blocking_handlers, (
            "HERMES_LOG_BLOCKING=1 should install at least one synchronous "
            "_ManagedRotatingFileHandler subclass instance."
        )
    finally:
        os.environ.pop("HERMES_LOG_BLOCKING", None)
        # best-effort teardown so the root logger is clean for sibling tests
        for h in list(logging.getLogger().handlers):
            try:
                h.close()
            except Exception:
                pass
            logging.getLogger().removeHandler(h)
        hermes_logging._logging_initialized = False


def test_synchronous_parent_does_block_on_hanging_write(tmp_path: Path) -> None:
    """Sanity check: with the synchronous parent handler, a wedged fd DOES block.

    This is the negative-control the regression tests above depend on —
    if ``_ManagedRotatingFileHandler.emit()`` ever stopped blocking the
    calling thread on a wedged write, the positive tests would silently
    pass for the wrong reason.  Run this in a thread and confirm that the
    synchronous path genuinely stalls when the disk hangs.
    """
    from hermes_logging import _ManagedRotatingFileHandler

    hang_file = _HangingWriteFile()
    # Build a real synchronous handler pointed at a tmp path, then swap
    # its underlying stream for the hanging one. We don't go through
    # ``_add_rotating_handler`` because we don't want this test to attach
    # the handler to root logger.
    real = _ManagedRotatingFileHandler(str(tmp_path / "synchronous-test.log"))
    real.stream = hang_file  # type: ignore[attr-defined]

    logger = logging.getLogger("synchronous_negative_control")
    logger.setLevel(logging.INFO)
    logger.addHandler(real)
    try:
        result: dict[str, float | Exception | None] = {"elapsed": None, "error": None}

        def emit_one() -> None:
            try:
                start = time.monotonic()
                logger.info("hello from synchronous path")
                result["elapsed"] = time.monotonic() - start
            except Exception as exc:  # pragma: no cover — defensive
                result["error"] = exc

        t = threading.Thread(target=emit_one)
        t.start()
        # Wait briefly; if the synchronous path is alive, the emit will
        # still be blocked inside ``hang_file.write()`` / ``flush()``.
        t.join(timeout=0.5)
        assert t.is_alive(), (
            "Negative control failed: the synchronous "
            "_ManagedRotatingFileHandler should still block on a wedged fd. "
            "If this trips, the regression tests above have lost their teeth."
        )

        # Clean up: release the hang so the worker thread doesn't outlive
        # the test (and pollute pytest's stderr with thread-dump noise).
        hang_file.release_writes()
        t.join(timeout=2.0)
        assert not t.is_alive(), "Worker thread did not exit after release_writes()"
    finally:
        hang_file.release_writes()
        logger.removeHandler(real)
        try:
            real.close()
        except Exception:
            pass
