"""Regression tests for #21915.

When the duplicate-instance guard fires under systemd (Restart=always), the
gateway must NOT exit with a non-zero status — that triggers an immediate
respawn that detects the same duplicate and exits non-zero again, producing
a flap loop that eventually trips StartLimitBurst and leaves the unit in
``failed`` state.

The fix: when the parent is systemd (``INVOCATION_ID`` env var is set), the
duplicate-detected branch returns True (clean exit 0) so the unit settles in
``active (exited)``.  When invoked from a user shell (no ``INVOCATION_ID``),
behavior is unchanged — return False so the shell sees a non-zero exit code
and the user knows the start was rejected.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from gateway.config import GatewayConfig


@pytest.mark.asyncio
async def test_duplicate_under_systemd_returns_true_for_clean_exit(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("INVOCATION_ID", "deadbeef-cafe-1234-5678-aabbccddeeff")

    monkeypatch.setattr("gateway.status.get_running_pid", lambda: 42)
    monkeypatch.setattr("gateway.run.os.getpid", lambda: 100)

    from gateway.run import start_gateway

    ok = await start_gateway(config=GatewayConfig(), replace=False, verbosity=None)

    assert ok is True


@pytest.mark.asyncio
async def test_duplicate_outside_systemd_returns_false(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("INVOCATION_ID", raising=False)

    monkeypatch.setattr("gateway.status.get_running_pid", lambda: 42)
    monkeypatch.setattr("gateway.run.os.getpid", lambda: 100)

    from gateway.run import start_gateway

    ok = await start_gateway(config=GatewayConfig(), replace=False, verbosity=None)

    assert ok is False


@pytest.mark.asyncio
async def test_duplicate_under_systemd_logs_clean_exit_reason(
    monkeypatch, tmp_path, caplog
):
    """The clean-exit path emits an INFO log explaining why we returned 0,
    so an operator inspecting ``journalctl`` can see the unit-flap-prevention
    behavior rather than just an unexplained ``active (exited)``.
    """
    import logging

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("INVOCATION_ID", "deadbeef-cafe-1234-5678-aabbccddeeff")

    monkeypatch.setattr("gateway.status.get_running_pid", lambda: 42)
    monkeypatch.setattr("gateway.run.os.getpid", lambda: 100)

    from gateway.run import start_gateway

    with caplog.at_level(logging.INFO, logger="gateway.run"):
        ok = await start_gateway(
            config=GatewayConfig(), replace=False, verbosity=None
        )

    assert ok is True
    assert any(
        "systemd" in rec.getMessage().lower()
        and "exiting cleanly" in rec.getMessage().lower()
        for rec in caplog.records
    ), f"expected clean-exit log, got: {[r.getMessage() for r in caplog.records]}"


def test_source_guards_invocation_id_check_in_duplicate_branch():
    """Source-level guard: regardless of refactor, the duplicate-detected
    branch must consult ``INVOCATION_ID`` and return True under systemd —
    otherwise #21915 silently regresses.
    """
    src = Path(__file__).resolve().parents[2] / "gateway" / "run.py"
    text = src.read_text(encoding="utf-8")

    assert "INVOCATION_ID" in text, (
        "gateway/run.py no longer references INVOCATION_ID — the systemd "
        "clean-exit guard for #21915 has likely been removed."
    )
    assert "#21915" in text, (
        "gateway/run.py no longer references #21915 — the comment anchoring "
        "the systemd clean-exit guard has been dropped."
    )
