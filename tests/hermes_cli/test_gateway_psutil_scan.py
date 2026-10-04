"""Tests for the psutil-first Windows gateway PID scan.

``_scan_gateway_pids()`` prefers ``_psutil_windows_processes()`` (~7ms) over the
wmic / ``Get-CimInstance`` spawn (~1.4s), but psutil cannot read the cmdline of
processes owned by another user, so a psutil scan that saw an unreadable process
must not be trusted to report "no gateway is running" — otherwise an elevated
gateway would look stopped to an unelevated CLI and stop/update would leave it
alive. These tests pin that contract; the platform arm is selected by patching
``is_windows`` so they run everywhere.
"""

import pytest

import hermes_cli.gateway as gateway_mod

_GATEWAY_CMD = "python -m hermes_cli.main gateway run"
_OTHER_CMD = "python -m some_other_thing"
_GATEWAY_PID = 4242


@pytest.fixture
def windows_scan(monkeypatch):
    """Windows arm of ``_scan_gateway_pids`` with a counting wmic/PowerShell listing."""
    monkeypatch.setattr(gateway_mod, "is_windows", lambda: True)
    monkeypatch.setattr(gateway_mod, "_get_ancestor_pids", set)
    calls = {"listing": 0}

    def _listing():
        calls["listing"] += 1
        return f"CommandLine={_GATEWAY_CMD}\nProcessId={_GATEWAY_PID}\n"

    monkeypatch.setattr(gateway_mod, "_windows_process_listing", _listing)
    return calls


def _set_psutil_scan(monkeypatch, value):
    monkeypatch.setattr(gateway_mod, "_psutil_windows_processes", lambda: value)


def test_psutil_match_skips_the_subprocess_listing(windows_scan, monkeypatch):
    _set_psutil_scan(monkeypatch, ([(_GATEWAY_PID, _GATEWAY_CMD)], True))

    assert gateway_mod._scan_gateway_pids(set(), all_profiles=True) == [_GATEWAY_PID]
    assert windows_scan["listing"] == 0


def test_complete_psutil_scan_without_match_skips_the_listing(windows_scan, monkeypatch):
    """Nothing matched and every cmdline was readable: the answer is trustworthy."""
    _set_psutil_scan(monkeypatch, ([(1, _OTHER_CMD)], True))

    assert gateway_mod._scan_gateway_pids(set(), all_profiles=True) == []
    assert windows_scan["listing"] == 0


def test_incomplete_psutil_scan_without_match_falls_back(windows_scan, monkeypatch):
    """An AccessDenied cmdline may have hidden an elevated gateway — re-scan via WMI."""
    _set_psutil_scan(monkeypatch, ([(1, _OTHER_CMD)], False))

    assert gateway_mod._scan_gateway_pids(set(), all_profiles=True) == [_GATEWAY_PID]
    assert windows_scan["listing"] == 1


def test_unusable_psutil_falls_back(windows_scan, monkeypatch):
    """psutil missing or enumeration failed."""
    _set_psutil_scan(monkeypatch, None)

    assert gateway_mod._scan_gateway_pids(set(), all_profiles=True) == [_GATEWAY_PID]
    assert windows_scan["listing"] == 1


def test_incomplete_scan_with_match_is_still_fast(windows_scan, monkeypatch):
    """A hit is a hit: an unreadable process elsewhere cannot invalidate it."""
    _set_psutil_scan(monkeypatch, ([(_GATEWAY_PID, _GATEWAY_CMD)], False))

    assert gateway_mod._scan_gateway_pids(set(), all_profiles=True) == [_GATEWAY_PID]
    assert windows_scan["listing"] == 0


@pytest.mark.windows_only
def test_psutil_scan_reports_real_processes():
    """The real enumeration returns rows and flags its own completeness."""
    scan = gateway_mod._psutil_windows_processes()

    assert scan is not None
    rows, complete = scan
    assert rows and all(isinstance(pid, int) and isinstance(cmd, str) for pid, cmd in rows)
    assert isinstance(complete, bool)
