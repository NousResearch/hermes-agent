"""``gateway restart --all`` owes the same "Gateway online" notice as ``gateway restart``.

The verb stops the host multiplexer through a service call or a SIGTERM, then starts it again, so
the old process never runs the in-band restart path that stamps ``.restart_pending.json``. Without a
marker the next boot reads a plain stop and every home channel stays silent — the same gap #142515
describes for the Windows CLI restart, on every OS.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gateway import run as gateway_run


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    return tmp_path


def test_restart_all_stamps_the_marker_after_the_host_is_stopped(hermes_home, monkeypatch):
    from hermes_cli import gateway as gw

    marker = hermes_home / ".restart_pending.json"
    seen: dict[str, bool] = {}
    monkeypatch.setattr(
        gw, "_stop_installed_service", lambda system: seen.setdefault("marker_at_stop", marker.exists()) or False
    )
    monkeypatch.setattr(gw, "_stop_host_multiplexer", lambda owner: 1)
    monkeypatch.setattr(gw, "_print_unfolded_gateway_note", lambda owner: None)
    monkeypatch.setattr(gw, "_wait_for_gateway_exit", lambda **kw: None)
    monkeypatch.setattr(gw, "_wait_for_api_server_port_free", lambda: None)
    monkeypatch.setattr(gw, "_discard_dead_host_record", lambda: None)
    monkeypatch.setattr(gw, "_forget_host_gateway_probe", lambda: None)
    monkeypatch.setattr(gw, "_installed_service_kind_for", lambda windows: "systemd")
    monkeypatch.setattr(gw, "_service_call", lambda kind, verb, system: seen.setdefault("marker_at_start", marker.exists()))
    monkeypatch.setattr(gw, "is_windows", False)

    owner = SimpleNamespace(pid=4242, profiles=["default"], home=hermes_home, describe=lambda: "host")
    monkeypatch.setattr(gw, "_host_multiplexer_for_all_verb", lambda: owner)
    monkeypatch.setattr(gw, "_host_multiplexer_is_ours", lambda o: True)

    # The verb itself, exactly as the CLI dispatches it.
    gw._restart_all(system=False)

    assert seen["marker_at_stop"] is False, "the still-draining old host must not see the marker"
    assert seen["marker_at_start"] is True, "the restarted host's boot pass reads this marker"
