"""Real Pipecat import smoke (replaces the prior import-skipped stub).

Where the simplex-streaming extra is installed, asserts pipecat imports and the
probe reports 1.3.0. Skips cleanly where the extra is absent (local dev); the
absence-is-safe path is covered deterministically in test_pipecat_runtime.py.
"""
from __future__ import annotations

import pytest

from gateway.calls.native.streaming.pipecat_runtime import pipecat_available, pipecat_version

pytestmark = pytest.mark.skipif(
    not pipecat_available(), reason="simplex-streaming extra not installed"
)


def test_pipecat_imports_and_version():
    import pipecat  # noqa: F401

    assert pipecat_version() == "1.3.0"
