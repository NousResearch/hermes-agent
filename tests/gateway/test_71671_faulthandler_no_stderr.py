"""Regression: #71671 — gateway must survive faulthandler.enable() with sys.stderr=None."""

from __future__ import annotations

import faulthandler
import os
import sys

import pytest


def test_faulthandler_enable_falls_back_when_stderr_is_none(tmp_path):
    was_enabled = faulthandler.is_enabled()
    if was_enabled:
        faulthandler.disable()

    real_stderr = sys.stderr
    sys.stderr = None
    try:
        with pytest.raises(RuntimeError, match="sys.stderr is None"):
            faulthandler.enable()

        log_path = tmp_path / "gateway_faulthandler.log"
        fh = open(log_path, "a", encoding="utf-8")
        try:
            faulthandler.enable(file=fh, all_threads=True)
            assert faulthandler.is_enabled()
            assert log_path.exists()
        finally:
            faulthandler.disable()
            fh.close()
    finally:
        sys.stderr = real_stderr
        if was_enabled:
            faulthandler.enable()


def test_gateway_run_keeps_faulthandler_guard():
    src = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "gateway", "run.py",
    )
    with open(src, encoding="utf-8") as f:
        text = f.read()
    assert "try:\n            faulthandler.enable()\n        except (RuntimeError, ValueError, OSError):" in text, (
        "gateway/run.py must keep the try/except guard around faulthandler.enable() (see #71671)"
    )
