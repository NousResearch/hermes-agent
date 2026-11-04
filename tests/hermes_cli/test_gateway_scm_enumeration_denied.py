"""A token the SCM refuses to enumerate for has no service gateways to coordinate (#148204).

``hermes update`` must carry on with process discovery rather than abort before touching the checkout, while
any other enumeration failure stays indeterminate and keeps aborting.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

import psutil
import pytest

from hermes_cli.gateway import find_windows_gateway_services

pytestmark = pytest.mark.platforms("windows")


def _psutil_whose_enumeration_raises(exc: Exception) -> SimpleNamespace:
    def win_service_iter():
        # psutil's own iterator is a generator: OpenSCManager fails on the first step, not on the call.
        raise exc
        yield

    return SimpleNamespace(win_service_iter=win_service_iter, AccessDenied=psutil.AccessDenied)


@pytest.mark.parametrize("denial", [
    PermissionError(13, "Access is denied", "(originated from OpenSCManager)"),
    psutil.AccessDenied(),
])
def test_a_denied_enumeration_means_no_service_gateways(denial, caplog):
    fake = _psutil_whose_enumeration_raises(denial)

    assert find_windows_gateway_services(psutil_module=fake, profile_processes=[]) == []
    assert "denied" in caplog.text.lower()


def test_any_other_enumeration_failure_stays_indeterminate():
    fake = _psutil_whose_enumeration_raises(OSError("RPC server unavailable"))

    with pytest.raises(RuntimeError, match="SCM service enumeration failed"):
        find_windows_gateway_services(psutil_module=fake, profile_processes=[])


def test_update_discovery_survives_a_denied_enumeration(monkeypatch):
    from hermes_cli import gateway, update_cmd_windows

    fake = _psutil_whose_enumeration_raises(PermissionError(13, "Access is denied"))
    monkeypatch.setitem(sys.modules, "psutil", fake)
    monkeypatch.setattr(gateway, "find_profile_gateway_processes", lambda **_kw: [])
    monkeypatch.setattr(gateway, "find_gateway_pids", lambda **_kw: [])

    _profiles, service_gateways, service_pids, _running = update_cmd_windows._discover_windows_gateways()

    assert service_gateways == []
    assert service_pids == set()
