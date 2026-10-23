"""A failed service pause keeps its record exactly when its rollback left gateways down.

``_pause_windows_gateways_for_update`` decides whether to abandon the durable pause record from the
service-pause failure. That decision is read from a typed ``rollback_failures`` list, never from the
error message's wording: a reworded message must not delete the record while gateways stay stopped.

Real pause producer and durable record; seams are the Windows-only discovery, the SCM stop/restore
calls and the ordinary-gateway stop (nothing is signalled).
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from hermes_cli import gateway, update_cmd, update_cmd_windows
from hermes_cli import update_pause_record as pause_record
from hermes_cli.update_cmd_windows import ServicePauseFailed, _pause_windows_gateways_for_update


@pytest.mark.parametrize("restore_fails", [False, True])
def test_the_pause_record_survives_exactly_when_the_service_rollback_failed(monkeypatch, restore_fails):
    from hermes_cli import main
    pid = os.getpid()  # a live identity for the service's gateway; never signalled
    service = SimpleNamespace(name="hermes-gw", profile="default", gateway_pid=pid, gateway_create_time=1.0,
                              service_pid=pid, service_create_time=1.0, descendant_identities=())
    monkeypatch.setattr(main, "_is_windows", lambda: True)
    monkeypatch.setattr(gateway, "find_gateway_pids", lambda all_profiles=False: [])
    monkeypatch.setattr(gateway, "find_profile_gateway_processes", lambda strict=False: [])
    monkeypatch.setattr(gateway, "find_windows_gateway_services", lambda profile_processes=(): [service])
    monkeypatch.setattr(update_cmd_windows, "_stop_windows_gateways", lambda *a, **k: {})
    monkeypatch.setattr(update_cmd_windows, "_record_attested_cold_start_profiles", lambda *a: None)

    def stop(*_a, **_k):
        raise OSError("access denied")

    def restore(_name):
        if restore_fails:
            raise OSError("service will not start")

    monkeypatch.setattr(update_cmd, "_stop_windows_gateway_service", stop)
    monkeypatch.setattr(update_cmd, "_restore_windows_gateway_service", restore)

    with pytest.raises(ServicePauseFailed) as failed:
        _pause_windows_gateways_for_update()
    assert bool(failed.value.rollback_failures) is restore_fails
    saved = pause_record.read(pause_record.record_path())
    assert (saved is not None) is restore_fails, \
        "a rolled-back pause kept its record" if saved else "the record of still-stopped gateways was deleted"
