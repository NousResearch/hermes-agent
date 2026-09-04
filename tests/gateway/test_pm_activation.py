"""Gateway main() must run the pm startup contract (gateway/run.py).

The gateway daemon boots via its own entrypoint and never passes through the
CLI dispatch in hermes_cli/main.py — without an explicit call here the
gateway's os.environ PATH would lack the pm store's tool dirs (git/bash/
ffmpeg/...), so every reactive ``shutil.which`` inside the daemon resolves
system binaries (or nothing) instead of the pinned store ones.

Behavior contract (mirrors the CLI-dispatch block, hermes_cli/main.py):
- healthy store → ``pm.activate()`` provisions PATH
- out-of-sync store → warning, boot continues (never blocks)
- pm import/lookup failure → debug-logged, boot continues
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest


@pytest.fixture
def gateway_main(monkeypatch, tmp_path):
    """Import gateway.run.main with the heavy boot machinery neutralized."""
    # The real main() advertises env vars, registers process identity, runs
    # boot bootstrap, and FORCE-EXITS via _exit_after_graceful_shutdown
    # (os._exit — wedge-proof by design, #53107) — none of it is what we
    # assert on. Neutralize the exit + the async boot; keep the pm block's
    # imports real.
    import gateway.run as gr

    async def _no_gateway(_config=None, **_kwargs):
        return True

    exited: list[int] = []

    def _fake_exit(code):
        exited.append(code)

    monkeypatch.setattr(gr, "start_gateway", _no_gateway)
    monkeypatch.setattr(gr, "_exit_after_graceful_shutdown", _fake_exit)
    monkeypatch.setattr(
        "hermes_cli.boot_bootstrap.maybe_run_boot_bootstrap", lambda _root: None
    )

    def _main():
        gr.main()
        return exited

    return _main


def test_gateway_main_activates_pm_store(gateway_main, tmp_path, monkeypatch):
    """A healthy store: main() provisions PATH before starting adapters."""
    calls: list[str] = []

    class _FakePm:
        @staticmethod
        def adopt():
            calls.append("adopt")

        @staticmethod
        def check():
            calls.append("check")
            return []  # healthy

        @staticmethod
        def activate():
            calls.append("activate")

    import sys

    monkeypatch.setitem(sys.modules, "pm", _FakePm)
    with patch("sys.argv", ["gateway"]):
        gateway_main()
    assert calls == ["adopt", "check", "activate"]


def test_gateway_main_warns_but_boots_when_store_out_of_sync(
    gateway_main, monkeypatch, caplog
):
    """An out-of-sync store surfaces a warning and NEVER blocks the boot."""

    class _FakePm:
        @staticmethod
        def adopt():
            pass

        @staticmethod
        def check():
            return ["uv not installed", "ffmpeg outdated"]

        @staticmethod
        def activate():
            raise AssertionError("activate must not run on a broken store")

    import sys

    monkeypatch.setitem(sys.modules, "pm", _FakePm)
    with patch("sys.argv", ["gateway"]):
        gateway_main()
    assert any(
        "install out of sync" in rec.message for rec in caplog.records
    ), "the out-of-sync warning must be logged"


def test_gateway_main_survives_pm_failure(gateway_main, monkeypatch):
    """A broken pm import is debug-logged; the gateway still boots."""

    class _ExplodingPm:
        @staticmethod
        def adopt():
            raise RuntimeError("pm exploded")

    import sys

    monkeypatch.setitem(sys.modules, "pm", _ExplodingPm)
    with patch("sys.argv", ["gateway"]):
        gateway_main()  # must not raise
