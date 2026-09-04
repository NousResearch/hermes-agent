"""Tests: the Windows SCM service frontend (gateway/windows_service.py).

Hermetic — pywin32 pieces are probed/stubbed, the child spawn is faked,
the planned-stop marker write is asserted by name. The frontend's
CONTRACTS are what's tested here; the SCM protocol itself is pywin32's
and gets its live proof in the wine2e lane (plan Task 1/W5.6).
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

import gateway.windows_service as ws


def test_service_name_matches_manifest_key():
    # The manifest names this exact service; the CLI verbs key off it too.
    assert ws.SERVICE_NAME == "HermesGateway"


def test_drain_window_default_below_scm_window():
    # The frontend's own deadline must sit BELOW the SCM's
    # WaitToKillServiceTimeout with margin — else the SCM force-kill
    # races the graceful drain (Task 3's whole contract).
    window = ws._drain_window_s()
    assert window <= ws._SCM_WINDOW_FALLBACK_S - ws._STOP_MARGIN_S


def test_drain_window_env_override_capped():
    with patch.dict(ws.os.environ, {"HERMES_RESTART_DRAIN_TIMEOUT": "600"}):
        assert ws._drain_window_s() == float(ws._SCM_WINDOW_FALLBACK_S - ws._STOP_MARGIN_S)


def test_drain_window_env_override_respected_when_smaller():
    with patch.dict(ws.os.environ, {"HERMES_RESTART_DRAIN_TIMEOUT": "8"}):
        assert ws._drain_window_s() == 8.0


def test_stop_marker_write_uses_planned_stop_path(monkeypatch):
    """The graceful nudge MUST be the existing planned-stop marker — the
    same path `hermes gateway stop` uses; no second stop protocol."""
    calls = {}

    import gateway.status as status_mod

    monkeypatch.setattr(
        status_mod, "write_planned_stop_marker", lambda pid: calls.setdefault("pid", pid)
    )
    ws._write_stop_marker(4242)
    assert calls == {"pid": 4242}


def test_stop_marker_failure_falls_back_to_terminate(monkeypatch):
    """Marker machinery unavailable (partial payload): terminate the pid
    rather than leave an unstoppable child."""
    terminated = {}

    def boom(pid):
        raise RuntimeError("no gateway.status here")

    import gateway.status as status_mod

    monkeypatch.setattr(status_mod, "write_planned_stop_marker", boom)
    monkeypatch.setattr(ws.os, "kill", lambda pid, sig: terminated.update(pid=pid, sig=sig))
    ws._write_stop_marker(99)
    assert terminated == {"pid": 99, "sig": 15}


@pytest.mark.platforms("windows")
def test_no_pywin32_means_friendly_error(tmp_path, capfd):
    """A user running --service BY HAND (not via SCM): pywin32 is present
    but Initialize/dispatcher refuse without the SCM — the frontend must
    print guidance, never hang. (Simulated: ImportError shape.)"""
    with patch.dict(sys.modules, {"win32serviceutil": None, "servicemanager": None}):
        rc = ws.run_as_service()
        assert rc == 2
        out = capfd.readouterr().err
        assert "foreground gateway run" in out
