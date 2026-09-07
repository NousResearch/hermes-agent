"""Unit properties retain systemd semantics without overflowing subprocess polling."""

import os
import selectors
import subprocess

import pytest

from gateway.shutdown_forensics import parse_systemd_duration_to_us
from hermes_cli import update_cmd_fleet as fleet


@pytest.mark.parametrize("stop, expected", [
    ("70000000", 175),
    (" 70000000 ", 175),
    ("1min 10s", 175),
    ("1d 6h", 30 * 3600 + 105),
    ("1w", 7 * 86400 + 105),
    ("0", 195),
])
def test_property_representations(monkeypatch, stop, expected):
    monkeypatch.setattr(fleet, "_systemctl", lambda cmd, **kw: subprocess.CompletedProcess(
        cmd, 0, f"TimeoutStopUSec={stop}\nTimeoutStartUSec=90s\n", ""))
    assert fleet._systemd_restart_timeout(["systemctl", "--user"], "hermes-serve-test") == expected


@pytest.mark.parametrize("raw, seconds", [
    ("1d", 86400), ("1w", 7 * 86400),
    ("1month", 2629800), ("1y", 31557600),
])
def test_systemd_emitted_large_units(raw, seconds):
    assert parse_systemd_duration_to_us(raw) == seconds * 1_000_000


@pytest.mark.skipif(not hasattr(selectors, "PollSelector"), reason="systemd uses POSIX polling")
@pytest.mark.parametrize("raw", ["1y", "18446744073709551614", "9" * 300 + "s"])
def test_combined_timeout_fits_real_poll(monkeypatch, raw):
    monkeypatch.setattr(fleet, "_systemctl", lambda cmd, **kw: subprocess.CompletedProcess(
        cmd, 0, f"TimeoutStopUSec={raw}\nTimeoutStartUSec={raw}\n", ""))
    budget = fleet._systemd_restart_timeout(["systemctl"], "hermes-serve-test")
    assert budget >= 30 * 3600  # must not regress legitimate long budgets to minutes
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, b"ready")
        with selectors.PollSelector() as poller:
            poller.register(read_fd, selectors.EVENT_READ)
            # Readiness means no sleeping; the native API still validates timeout.
            assert poller.select(budget)
    finally:
        os.close(read_fd)
        os.close(write_fd)
