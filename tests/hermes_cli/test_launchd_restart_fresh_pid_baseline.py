"""``launchd_restart`` must measure "a fresh service pid" against launchd's own report.

``get_running_pid()`` validates liveness, not identity: a stale lock record or a
recycled pid passes its check. When that unverified pidfile pid is the baseline,
the replacement-pid verdict after a graceful exit can be satisfied by the
*incumbent* gateway — the launchd-supervised process never changed, yet
``✓ Service restart requested`` printed and the operator believed the service
cycled (#137228: two restarts, two false successes, the real gateway PID never
moved). The self-restart hand-off branch had the same hole one level earlier:
it printed success off a signal-send alone.

The contract pinned here: every "restart succeeded" verdict in ``launchd_restart``
compares against the pid launchd reported *before* the restart attempt, and the
signal-send branch also has to see a fresh supervised pid before it may claim
success.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import hermes_cli.gateway as gateway_cli

STALE_PIDFILE_PID = 60047  # recycled pid — alive, but NOT the gateway
INCUMBENT_PID = 66398  # the gateway launchd actually supervises (PPID 1)


@pytest.fixture
def launchd_seam(monkeypatch, tmp_path):
    """Neutralize process-side effects; a launchd that always reports the incumbent."""
    calls = []
    seams = {}
    plist_path = tmp_path / "ai.hermes.gateway.plist"
    plist_path.write_text("<plist>whatever</plist>", encoding="utf-8")

    monkeypatch.setattr(gateway_cli, "get_launchd_label", lambda: "ai.hermes.gateway")
    monkeypatch.setattr(gateway_cli, "_launchd_domain", lambda: "gui/501")
    monkeypatch.setattr(gateway_cli, "get_launchd_plist_path", lambda: plist_path)
    monkeypatch.setattr(gateway_cli, "launchd_plist_is_current", lambda: True)
    monkeypatch.setattr(gateway_cli, "refresh_launchd_plist_if_needed", lambda: True)
    monkeypatch.setattr(gateway_cli, "_clear_launchd_unsupported_marker", lambda: None)
    monkeypatch.setattr(
        gateway_cli, "_launchd_unsupported_marker_exists", lambda: False
    )
    monkeypatch.setattr(gateway_cli, "_get_restart_exit_wait_budget", lambda: 5.0)
    monkeypatch.setattr(gateway_cli, "_wait_for_api_server_port_free", lambda: None)
    monkeypatch.setattr(gateway_cli, "_request_gateway_self_restart", lambda pid: False)
    monkeypatch.setattr(
        gateway_cli, "_graceful_restart_via_sigusr1", lambda pid, timeout, **_: False
    )
    # Not wedged for any probed pid.
    monkeypatch.setattr(
        gateway_cli, "probe_gateway_loop_liveness", lambda pid: "loop-alive-witness"
    )
    monkeypatch.setattr(gateway_cli, "GATEWAY_LOOP_WEDGED", object())

    # launchd's own report: the incumbent, before and (unless a test says otherwise) after.
    launchd_reported_pid = INCUMBENT_PID

    def fake_print_service_pid(domain, label):
        return (True, launchd_reported_pid)

    monkeypatch.setattr(
        gateway_cli, "_launchd_print_service_pid", fake_print_service_pid
    )

    def fake_supervision_wait(*, label=None, old_pid=None, **_):
        return old_pid is not None and launchd_reported_pid != old_pid

    monkeypatch.setattr(
        gateway_cli, "wait_for_launchd_gateway_supervision", fake_supervision_wait
    )

    def fake_wait_for_service_pid(label, old_pid, timeout=10.0, *, domain):
        seams["fresh_pid_baseline"] = old_pid
        return old_pid is not None and launchd_reported_pid != old_pid

    monkeypatch.setattr(
        gateway_cli, "_wait_for_launchd_service_pid", fake_wait_for_service_pid
    )

    def fake_run(cmd, check=False, timeout=None, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)
    seams["calls"] = calls

    def set_launchd_pid(pid):
        nonlocal launchd_reported_pid
        launchd_reported_pid = pid

    seams["set_launchd_pid"] = set_launchd_pid
    return seams


def test_stale_pidfile_baseline_cannot_satisfy_the_fresh_pid_verdict(
    monkeypatch, capsys, launchd_seam
):
    """The reporter's exact shape: SIGUSR1 "succeeds" against a recycled pid.

    The stale pidfile pid dies (default SIGUSR1 disposition), the drain wait
    returns True — and pre-fix, the incumbent differed from that stale pid, so
    the fresh-pid check passed and printed success while the real gateway never
    moved. Post-fix the verdict is measured against launchd's own report (the
    incumbent), fails, and the bounded kickstart forcing path runs instead.
    """
    monkeypatch.setattr(
        "gateway.status.get_running_pid", lambda *a, **k: STALE_PIDFILE_PID
    )
    # The recycled pid "exits" under the graceful SIGUSR1 wait.
    monkeypatch.setattr(
        gateway_cli, "_graceful_restart_via_sigusr1", lambda pid, timeout, **_: True
    )

    gateway_cli.launchd_restart()

    out = capsys.readouterr().out
    assert "✓ Service restart requested" not in out, (
        "the incumbent gateway satisfied the check — a restart that changed nothing "
        "must not print success"
    )
    assert launchd_seam["fresh_pid_baseline"] == INCUMBENT_PID, (
        "the fresh-pid verdict must be measured against launchd's own pre-restart report, "
        "not the identity-unverified pidfile pid"
    )
    kickstart_k = [c for c in launchd_seam["calls"] if "kickstart" in c and "-k" in c]
    assert kickstart_k, (
        "a failed fresh-pid verdict must fall through to the bounded forcing path"
    )


def test_graceful_exit_with_a_real_replacement_pid_still_prints_success(
    monkeypatch, capsys, launchd_seam
):
    """The honest restart keeps its success line: exit, KeepAlive revives, pid changes."""
    monkeypatch.setattr("gateway.status.get_running_pid", lambda *a, **k: INCUMBENT_PID)
    monkeypatch.setattr(
        gateway_cli, "_graceful_restart_via_sigusr1", lambda pid, timeout, **_: True
    )

    # The drain exits the incumbent; KeepAlive revives a fresh pid.
    def sigusr1_then_revive(pid, timeout, **_):
        launchd_seam["set_launchd_pid"](INCUMBENT_PID + 1)
        return True

    monkeypatch.setattr(
        gateway_cli, "_graceful_restart_via_sigusr1", sigusr1_then_revive
    )

    gateway_cli.launchd_restart()

    out = capsys.readouterr().out
    assert "✓ Service restart requested" in out
    assert launchd_seam["fresh_pid_baseline"] == INCUMBENT_PID
    assert not [c for c in launchd_seam["calls"] if "kickstart" in c and "-k" in c], (
        "a verified fresh pid must not be killed by a forcing kickstart"
    )


def test_self_restart_handoff_needs_a_fresh_supervised_pid(
    monkeypatch, capsys, launchd_seam
):
    """A signal-send alone restarts nothing: the hand-off is asynchronous.

    When launchd still reports the same supervised pid after the wait window,
    the branch must warn instead of printing success.
    """
    monkeypatch.setattr("gateway.status.get_running_pid", lambda *a, **k: INCUMBENT_PID)
    monkeypatch.setattr(gateway_cli, "_request_gateway_self_restart", lambda pid: True)

    gateway_cli.launchd_restart()

    out = capsys.readouterr().out
    assert "✓ Service restart requested" not in out
    assert "⚠" in out and "no fresh service pid" in out


def test_self_restart_handoff_with_fresh_pid_prints_success(
    monkeypatch, capsys, launchd_seam
):
    """Hand-off that demonstrably replaces the supervised process keeps its success line."""
    monkeypatch.setattr("gateway.status.get_running_pid", lambda *a, **k: INCUMBENT_PID)

    def handoff_then_revive(pid):
        launchd_seam["set_launchd_pid"](INCUMBENT_PID + 1)
        return True

    monkeypatch.setattr(
        gateway_cli, "_request_gateway_self_restart", handoff_then_revive
    )

    gateway_cli.launchd_restart()

    out = capsys.readouterr().out
    assert "✓ Service restart requested" in out


def test_launchd_restart_uses_sigusr1_and_exit_wait_budget(monkeypatch, capsys):
    """launchd_restart must take the same graceful path as systemd_restart.

    Regression: it previously sent a bare SIGTERM and waited
    ``_get_restart_drain_timeout()`` (default 0), so the wait could never
    succeed and every restart fell through to ``kickstart -k``. A bare
    SIGTERM leaves ``restart_requested`` False, so the gateway exits 1
    instead of 75 and announces itself as "shutting down" rather than
    "restarting", dropping the resume_pending handoff.

    Moved here from ``test_gateway_service.py`` (FILE_LINES ratchet) when
    #137228 added the pre-restart launchd baseline probe to this path.
    """
    calls = []

    monkeypatch.setattr(gateway_cli, "get_launchd_label", lambda: "ai.hermes.gateway")
    monkeypatch.setattr(gateway_cli, "_launchd_domain", lambda: "gui/501")
    monkeypatch.setattr("gateway.status.get_running_pid", lambda *a, **k: 654)
    monkeypatch.setattr(gateway_cli, "_request_gateway_self_restart", lambda pid: False)
    monkeypatch.setattr(
        gateway_cli,
        "probe_gateway_loop_liveness",
        lambda pid, **kw: gateway_cli.GATEWAY_LOOP_ALIVE,
    )
    # Wait budget covers after-turn deferral + drain + headroom (#77184);
    # the raw drain timeout (0 by default) must not be used here.
    monkeypatch.setattr(gateway_cli, "_get_restart_drain_timeout", lambda: 0.0)
    monkeypatch.setattr(gateway_cli, "_get_restart_exit_wait_budget", lambda: 27.0)
    monkeypatch.setattr(
        gateway_cli,
        "_graceful_restart_via_sigusr1",
        lambda pid, timeout, **_: calls.append(("graceful", pid, timeout)) or True,
    )
    monkeypatch.setattr(
        gateway_cli,
        "terminate_pid",
        lambda pid, force=False: calls.append(("sigterm", pid)),
    )
    monkeypatch.setattr(
        gateway_cli.subprocess,
        "run",
        lambda *a, **k: (
            calls.append(("kickstart", a[0]))
            or SimpleNamespace(returncode=0, stdout="", stderr="")
        ),
    )
    monkeypatch.setattr(gateway_cli, "_clear_launchd_unsupported_marker", lambda: None)
    # launchd reports the same pid the pidfile names (#137228): the pre-restart
    # baseline probe must not reach a real launchctl from a unit test.
    monkeypatch.setattr(
        gateway_cli, "_launchd_print_service_pid", lambda domain, label: (True, 654)
    )
    # KeepAlive revives the label on a fresh PID — replacement observed.
    monkeypatch.setattr(
        gateway_cli,
        "_wait_for_launchd_service_pid",
        lambda label, old_pid, timeout=10.0, *, domain: (
            calls.append(("observe", label, old_pid, domain)) or True
        ),
    )

    gateway_cli.launchd_restart()

    assert ("graceful", 654, 27.0) in calls
    # A bare SIGTERM would strand the gateway on the unplanned-shutdown path.
    assert not any(call[0] == "sigterm" for call in calls)
    # ``-k`` after a successful graceful exit would kill the replacement.
    assert not any(call[0] == "kickstart" for call in calls)
    # The success message must follow an observed replacement PID.
    assert ("observe", "ai.hermes.gateway", 654, "gui/501") in calls
    out = capsys.readouterr().out
    assert "up to 27s" in out
    assert "up to 0s" not in out
