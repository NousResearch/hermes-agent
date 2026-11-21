"""The post-update POSIX resume must reconcile with the supervisor, not guess.

The supervisor restarts what the pause stopped — systemd units carry ``Restart=always`` and launchd
jobs ``KeepAlive`` — so the runtime is often already back on the new code before the resume runs.
These tests pin what the resume does with that, where it starts a job that did NOT come back, and
that a handled ``launchctl bootstrap`` EIO stops reading like a fatal update error.
"""

from __future__ import annotations

import subprocess

import pytest

from hermes_cli import update_cmd_posix_pause as pause

_PLIST = "/tmp/ai.hermes.gateway.plist"


@pytest.fixture(autouse=True)
def _clean_restarted_ledger():
    for seen in pause._RESTARTED.values():
        seen.clear()
    yield
    for seen in pause._RESTARTED.values():
        seen.clear()


def _token(**entry) -> dict:
    unit = {"kind": "systemd", "scope": "user", "unit": "hermes-gateway.service", "pid": 4242, **entry}
    return {"platform": "posix", "resume_needed": True, "posix_stopped": True, "posix_units": [unit]}


def test_a_unit_the_supervisor_already_restarted_is_discharged_not_started_again(monkeypatch):
    """#145375: systemd brought the unit back with a new MainPID between the stop and the resume, so
    nothing was owed — yet the resume could not start it (no passwordless sudo) and kept the entry,
    which every later command retried forever."""
    token = _token()
    monkeypatch.setattr(pause, "_supervised_pid", lambda entry: 5150)

    def never_start(_entry):
        raise AssertionError("a runtime already running on the new code was started again")

    monkeypatch.setattr(pause, "_start_unit", never_start)

    pause.resume_paused_set(token)

    assert token["posix_units"] == [], "the satisfied obligation stayed owed"
    assert token["resume_needed"] is False
    assert token["restarted_services"] == ["hermes-gateway.service"]
    assert pause.already_restarted()["units"] == {"user/hermes-gateway.service"}
    assert pause.already_restarted()["pids"] == {5150}


def test_a_unit_still_running_as_the_paused_pid_is_started(monkeypatch):
    """Same MainPID means the process the pause stopped never left: the start is still the only way
    back, so reconciliation must not skip it."""
    token = _token(pid=4242)
    monkeypatch.setattr(pause, "_supervised_pid", lambda entry: 4242)
    started: list[dict] = []
    monkeypatch.setattr(pause, "_start_unit", lambda entry: started.append(entry) or 9001)

    pause.resume_paused_set(token)

    assert len(started) == 1
    assert token["posix_units"] == []
    assert pause.already_restarted()["pids"] == {9001}


def test_an_unreadable_supervisor_still_tries_the_start(monkeypatch):
    """Unknown state is not "restarted": the resume keeps the behaviour it had before reconciling."""
    token = _token()
    monkeypatch.setattr(pause, "_supervised_pid", lambda entry: None)
    started: list[dict] = []
    monkeypatch.setattr(pause, "_start_unit", lambda entry: started.append(entry) or 1)

    pause.resume_paused_set(token)

    assert len(started) == 1


def _launchd_job(domain: str) -> dict:
    return {"kind": "launchd", "label": "ai.hermes.gateway", "domain": domain,
            "plist": _PLIST, "pid": 4242, "home": "/tmp/home"}


def test_a_launchd_job_starts_in_the_domain_that_holds_it_now(monkeypatch):
    """The plist refresh loads into the domain launchd resolves NOW (gui/<uid> under Aqua) while the
    pause recorded the domain the job lived in then (user/<uid>). Bootstrapping the recorded domain
    afterwards answers EIO 5 — "label already registered" — which is what turned a healthy restart
    into a failed resume and left the record owed."""
    from hermes_cli import gateway as gw

    calls: list[tuple] = []
    monkeypatch.setattr(gw, "get_launchd_label", lambda: "ai.hermes.gateway")
    monkeypatch.setattr(gw, "refresh_launchd_plist_if_needed", lambda: True)
    monkeypatch.setattr(gw, "_locate_launchd_gateway_service", lambda label: ("gui/501", 777))
    monkeypatch.setattr(gw, "_launchctl_bootstrap", lambda *a, **k: calls.append(("bootstrap", a)))
    monkeypatch.setattr(gw, "_wait_for_launchd_service_pid", lambda *a, **k: True)
    monkeypatch.setattr(pause.subprocess, "run",
                        lambda argv, **k: calls.append(("run", argv)) or subprocess.CompletedProcess(argv, 0))

    pause._start_job(_launchd_job("user/501"))

    assert not [c for c in calls if c[0] == "bootstrap"], "bootstrapped a domain the job had left"
    assert ("run", ["launchctl", "kickstart", "gui/501/ai.hermes.gateway"]) in calls


def test_an_unregistered_launchd_job_is_bootstrapped_in_the_recorded_domain(monkeypatch):
    from hermes_cli import gateway as gw

    bootstrapped: list[tuple] = []
    monkeypatch.setattr(gw, "get_launchd_label", lambda: "ai.hermes.gateway")
    monkeypatch.setattr(gw, "refresh_launchd_plist_if_needed", lambda: False)
    monkeypatch.setattr(gw, "_locate_launchd_gateway_service", lambda label: (None, None))
    monkeypatch.setattr(gw, "_launchctl_bootstrap",
                        lambda domain, plist, label: bootstrapped.append((domain, plist, label)))
    monkeypatch.setattr(gw, "_wait_for_launchd_service_pid", lambda *a, **k: True)
    monkeypatch.setattr(pause.subprocess, "run", lambda argv, **k: subprocess.CompletedProcess(argv, 0))

    pause._start_job(_launchd_job("user/501"))

    assert bootstrapped == [("user/501", _PLIST, "ai.hermes.gateway")]


def test_a_handled_bootstrap_eio_does_not_print_launchctls_complaint(monkeypatch):
    """EIO is this call's expected "already registered" answer, recovered by bootout + retry two
    lines later; launchd's two stderr lines about it must not reach the update's terminal."""
    from hermes_cli import gateway_launchd as gl

    seen: list[tuple] = []

    def fake_run(argv, **kwargs):
        seen.append((argv, kwargs))
        if len(seen) == 1:
            raise subprocess.CalledProcessError(5, argv)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(gl.subprocess, "run", fake_run)

    gl._launchctl_bootstrap("user/501", _PLIST, "ai.hermes.gateway")

    first_argv, first_kwargs = seen[0]
    assert first_argv == ["launchctl", "bootstrap", "user/501", _PLIST]
    assert first_kwargs.get("capture_output") is True, "the recoverable EIO still writes to the terminal"
    assert first_kwargs.get("text") is True
    assert seen[1][0][:2] == ["launchctl", "bootout"]
    assert seen[2][0] == ["launchctl", "bootstrap", "user/501", _PLIST]


def test_the_resume_failure_prefix_is_platform_neutral_and_still_classified():
    """The message lost its Windows-only wording; receipts that a released version already parked
    with the old prefix must keep classifying the same way."""
    from hermes_cli.observability.shared_metrics_update import _STOP_REASON_PREFIX_CLASSES
    from hermes_cli.update_receipt import _PARKED_REASON_LABEL

    classes = dict(_STOP_REASON_PREFIX_CLASSES)
    assert classes["gateway recovery failed"] == "restart_failed"
    assert classes["Windows gateway recovery failed"] == "restart_failed"
    parked = _PARKED_REASON_LABEL.match("gateway recovery failed: CalledProcessError: x")
    assert parked is not None and parked.group(0) == "gateway recovery failed"
    legacy = _PARKED_REASON_LABEL.match("Windows gateway recovery failed: x")
    assert legacy is not None and legacy.group(0) == "Windows gateway recovery failed"
