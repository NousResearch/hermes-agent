"""``ensure`` never stops the gateway that spawned the caller; it names that case so a scoped cron
worker can wait for the update to replace its parent instead of failing (``parent_outdated``)."""
import os
import time

from hermes_cli import gateway_runtime
from hermes_cli import gateway_runtime_replace as replace


def test_an_old_owner_that_is_the_callers_parent_is_never_stopped(tmp_path, monkeypatch):
    from gateway.status import _GATEWAY_KIND
    home = tmp_path.resolve()
    payload = {"kind": _GATEWAY_KIND, "hermes_home": str(home), "pid": os.getppid(), "start_time": 1}
    monkeypatch.setattr(replace, "_install_root", lambda: home)
    monkeypatch.setattr(replace, "_identify", lambda h, timeout: (home, payload))
    stopped = []
    monkeypatch.setattr("gateway.status.terminate_pid", lambda *a, **k: stopped.append(a))
    assert replace.replace_incompatible_owner(home, deadline=time.monotonic() + 5) == "owner_is_parent"
    assert stopped == []


def test_ensure_reports_a_parent_owner_as_parent_outdated(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway_runtime, "discover_gateway_endpoint", lambda home, timeout: gateway_runtime.GatewayDiscovery(
        "incompatible", reason_code="runtime_protocol"))
    monkeypatch.setattr(replace, "replace_incompatible_owner", lambda home, deadline: "owner_is_parent")
    observed = gateway_runtime.ensure_gateway_runtime(tmp_path, timeout=5)
    assert (observed.state, observed.reason_code) == ("incompatible", "parent_outdated")
    assert observed.detail == replace.REFUSAL_DETAIL["owner_is_parent"]