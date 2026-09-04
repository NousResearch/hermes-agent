"""Regression test for #102915: ``hermes gateway start`` inside a container
without s6 must NOT exit 0 — an unstarted gateway is not a successful start.

The no-s6 container fallthrough printed Docker lifecycle guidance and exited
0, so programmatic callers (e.g. WebUI lifecycle buttons via
``subprocess.run``) reported a green success while no gateway was running.
The ``("start", "container")`` entry in ``_NO_BACKEND_MESSAGES`` now exits 1
and emits a machine-readable ``gateway_start: not_applicable`` line.
"""

import pytest

import hermes_cli.gateway as gateway_mod
from hermes_cli.gateway import _cmd_start


def _container_args():
    return type("Args", (), {"gateway_command": "start", "system": False, "all": False})()


def _make_container_no_s6(monkeypatch):
    monkeypatch.setattr(gateway_mod, "is_termux", lambda: False)
    monkeypatch.setattr(gateway_mod, "_service_backend", lambda: None)
    monkeypatch.setattr(gateway_mod, "is_wsl", lambda: False)
    monkeypatch.setattr(gateway_mod, "is_container", lambda: True)
    monkeypatch.setattr(gateway_mod, "_running_under_s6", lambda: False)
    monkeypatch.setattr(
        gateway_mod, "_dispatch_via_service_manager_if_s6", lambda _action: False
    )


def test_start_container_without_s6_exits_nonzero_with_machine_readable_outcome(
    monkeypatch, capsys
):
    _make_container_no_s6(monkeypatch)

    with pytest.raises(SystemExit) as exc_info:
        _cmd_start(_container_args())

    assert exc_info.value.code != 0, (
        "gateway start that did not start (or delegate) a gateway must not "
        "exit 0 — callers treat rc==0 as a successful start (#102915)"
    )
    out = capsys.readouterr().out
    assert "gateway_start: not_applicable" in out, (
        "machine-readable terminal outcome lets programmatic callers tell "
        "not_applicable apart from started/delegated"
    )
    # Docker lifecycle guidance is still printed for humans.
    assert "docker start" in out


def test_s6_container_start_dispatches_and_returns_cleanly(monkeypatch):
    """Control: inside an s6 container the start is delegated to the service
    slot and returns without the not-applicable exit."""
    monkeypatch.setattr(gateway_mod, "is_termux", lambda: False)
    monkeypatch.setattr(gateway_mod, "_dispatch_via_service_manager_if_s6", lambda _a: True)

    # Must return None (no SystemExit) — delegated, not not_applicable.
    assert _cmd_start(_container_args()) is None
