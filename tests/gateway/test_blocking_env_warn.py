"""Surface the HERMES_LOG_BLOCKING=1 escape hatch at startup.

Regression coverage for kanban task t_747076e8 (H-004 of the
gateway-logging audit).  The env var reverts the gateway to the
synchronous _ManagedRotatingFileHandler -- reintroducing the
asyncio-freeze risk that the a835f97 fix removed.  An operator who
accidentally leaves the var set in production (a launchd plist override,
a deploy wrapper, etc.) would see logs "work" until the disk wedges, at
which point the gateway would die exactly as in 2026-07-04.

Run::

    cd /Users/pones/.hermes/hermes-agent
    pytest tests/gateway/test_blocking_env_warn.py -v
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

import hermes_logging
from hermes_logging import setup_logging


EXPECTED_WARNING = (
    "[hermes-log] WARNING: HERMES_LOG_BLOCKING=1 -- gateway.log emit() is "
    "SYNCHRONOUS; a wedged disk will freeze the asyncio loop. "
    "Unset this env var to restore non-blocking behavior."
)


def _restore_root_logger() -> None:
    """Drop our handlers so sibling tests do not inherit them."""
    root = logging.getLogger()
    for h in list(root.handlers):
        try:
            h.close()
        except Exception:
            pass
        root.removeHandler(h)
    hermes_logging._logging_initialized = False


def _capture_stderr(capsys):
    """Return everything written to stderr around the test body."""
    return capsys.readouterr().err


@pytest.fixture
def fresh_blocking_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """Hermetic setup: own HERMES_HOME, fresh warn-state flag.

    The _blocking_warn_emitted flag lives on the module so repeated
    setup_logging() calls in the same process are idempotent.  We reset
    it explicitly here -- otherwise one passing test would silently
    satisfy the next.
    """
    hermes_logging._blocking_warn_emitted = False  # type: ignore[attr-defined]
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
    yield


def test_blocking_env_var_emits_warning_to_stderr(
    fresh_blocking_state,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """HERMES_LOG_BLOCKING=1 writes the warning to stderr verbatim."""
    monkeypatch.setenv("HERMES_LOG_BLOCKING", "1")
    hermes_home = Path(os.environ["HERMES_HOME"])

    setup_logging(hermes_home=hermes_home, mode="gateway", force=True)

    stderr = _capture_stderr(capsys)
    assert EXPECTED_WARNING in stderr, (
        "setup_logging() did not surface the HERMES_LOG_BLOCKING=1 warning. "
        f"Stderr captured:\n{stderr!r}"
    )


def test_no_warning_when_blocking_env_var_unset(
    fresh_blocking_state,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The default (non-blocking) path is silent -- no warning noise."""
    monkeypatch.delenv("HERMES_LOG_BLOCKING", raising=False)
    hermes_home = Path(os.environ["HERMES_HOME"])

    setup_logging(hermes_home=hermes_home, mode="gateway", force=True)

    stderr = _capture_stderr(capsys)
    assert "HERMES_LOG_BLOCKING=1" not in stderr, (
        f"Default (non-blocking) path emitted the warning unexpectedly:\n{stderr!r}"
    )


def test_blocking_warning_is_idempotent(
    fresh_blocking_state,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Multiple setup_logging() calls -> exactly one warning per process.

    An operator who restarts a sub-component, or a cron tick that
    re-imports the module, must NOT see the warning spam stderr on every
    call.  Acceptance criterion from the task body: "se imprime UNA vez
    por proceso (idempotente)".
    """
    monkeypatch.setenv("HERMES_LOG_BLOCKING", "1")
    hermes_home = Path(os.environ["HERMES_HOME"])

    setup_logging(hermes_home=hermes_home, mode="gateway", force=True)
    setup_logging(hermes_home=hermes_home, mode="gateway", force=True)
    setup_logging(hermes_home=hermes_home, mode="gateway", force=True)

    stderr = _capture_stderr(capsys)
    occurrences = stderr.count("HERMES_LOG_BLOCKING=1 -- gateway.log emit()")
    assert occurrences == 1, (
        f"Expected exactly 1 warning emission across 3 setup_logging() calls, "
        f"got {occurrences}. Stderr:\n{stderr!r}"
    )


def teardown_function() -> None:
    """Drop any handlers tests may have left on the root logger."""
    _restore_root_logger()
