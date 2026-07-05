"""Negative-control sanity check that the guard-rail regression test fails
when HERMES_LOG_BLOCKING=1 is set as the production default.

We run a stripped-down copy of the guard-rail test body WITHOUT
monkeypatch.delenv(), simulating the scenario where someone accidentally
left HERMES_LOG_BLOCKING=1 in production. The test should fail.
"""
import logging
import os
import tempfile
from pathlib import Path

import pytest

import hermes_logging


def test_guard_rail_negative_control_fails_with_blocking_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If HERMES_LOG_BLOCKING=1 is the active setup, the guard-rail test
    MUST fail — proving that the test correctly catches the regression.

    This file is for manual verification only and is NOT intended to be
    part of the regular CI suite. Run::

        cd /Users/pones/.hermes/hermes-agent
        HERMES_LOG_BLOCKING=1 pytest tests/gateway/_neg_ctrl_guard_rail.py -v

    The expected result is: test FAILS with the REGRESSION message.
    """
    monkeypatch.setenv("HERMES_LOG_BLOCKING", "1")
    tmp = Path(tempfile.mkdtemp(prefix="neg-ctrl-"))
    monkeypatch.setenv("HERMES_HOME", str(tmp / "hermes_home"))

    hermes_logging._logging_initialized = False
    root = logging.getLogger()
    for h in list(root.handlers):
        try:
            h.close()
        except Exception:
            pass
        root.removeHandler(h)

    hermes_logging.setup_logging(hermes_home=tmp / "hermes_home", mode="gateway", force=True)
    root = logging.getLogger()

    nb = [h for h in root.handlers if isinstance(h, hermes_logging._NonBlockingRotatingFileHandler)]
    sync = [
        h for h in root.handlers
        if isinstance(h, hermes_logging._ManagedRotatingFileHandler)
        and not isinstance(h, hermes_logging._NonBlockingRotatingFileHandler)
    ]

    assert nb, (
        f"REGRESSION: setup_logging(mode='gateway') under HERMES_LOG_BLOCKING=1 "
        f"did not install any non-blocking handler. Found {len(sync)} sync "
        f"handler(s) — these will freeze the asyncio loop on a wedged disk."
    )