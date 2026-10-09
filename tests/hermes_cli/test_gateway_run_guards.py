"""The legacy ``python cli.py --gateway`` entry must refuse exactly where ``hermes gateway run`` does.

It used to call ``gateway.run.start_gateway()`` directly, so a second dispatcher could start next
to a supervised or already-running gateway (or as root in the official image).
"""

from __future__ import annotations

import sys
import types

import pytest

GUARDS = (
    "_guard_official_docker_root_gateway",
    "_attach_to_host_gateway_or_guard",
    "_guard_supervised_gateway_conflict",
    "_guard_existing_gateway_process_conflict",
)


@pytest.fixture
def started(monkeypatch):
    """Replace the gateway runtime with a recorder; the list is non-empty once it is started."""
    calls = []

    async def _start_gateway(*args, **kwargs):
        calls.append(True)
        return True

    monkeypatch.setenv("HERMES_STARTUP_WATCHDOG", "0")
    monkeypatch.setitem(sys.modules, "gateway.run", types.SimpleNamespace(start_gateway=_start_gateway))
    return calls


def _stub_guards(monkeypatch, keep=()):
    import hermes_cli.gateway as gw

    for name in GUARDS:
        if name not in keep:
            monkeypatch.setattr(gw, name, lambda *a, **k: None)
    return gw


def test_legacy_gateway_entry_starts_when_every_guard_passes(monkeypatch, started):
    import cli

    _stub_guards(monkeypatch)
    cli._run_legacy_gateway()
    assert started == [True]


@pytest.mark.parametrize("guard", GUARDS)
def test_legacy_gateway_entry_stops_when_any_run_guard_refuses(monkeypatch, started, guard):
    import cli

    gw = _stub_guards(monkeypatch)
    monkeypatch.setattr(gw, guard, lambda *a, **k: sys.exit(1))
    with pytest.raises(SystemExit) as exc:
        cli._run_legacy_gateway()
    assert exc.value.code == 1
    assert started == []


def test_legacy_gateway_entry_refuses_next_to_a_running_gateway(monkeypatch, started):
    """Real duplicate-process guard with strict defaults: the legacy entry has no --replace."""
    import cli
    import gateway.status

    gw = _stub_guards(monkeypatch, keep=("_guard_existing_gateway_process_conflict",))
    monkeypatch.setattr(gw, "_running_under_gateway_supervisor", lambda: False)
    monkeypatch.setattr(gateway.status, "get_running_pid", lambda *a, **k: 4242)
    with pytest.raises(SystemExit) as exc:
        cli._run_legacy_gateway()
    assert exc.value.code == 1
    assert started == []
