"""Regression for #88848 - a launchd restart the update never verified.

``hermes update`` on macOS printed ``Update complete!`` and exited 0 while the
``ai.hermes.gateway`` LaunchAgent sat deregistered for 36 minutes.  The restart
phase treated "``launchd_restart()`` returned without raising" as success and
appended the label to ``restarted_services``.  Both of launchd_restart's normal
outcomes are asynchronous - the ``_request_gateway_self_restart`` branch returns
the instant the running gateway is *asked* to restart, and a plist reload is
handed to a detached helper - so a helper that died before its first bootstrap
was invisible to the caller.

The systemd branch of the same phase has never drawn that inference: it polls
``_wait_for_service_active`` before recording the unit and routes a unit that
never came back into ``failed_or_stale_units``, which is what makes the update
exit non-zero.  These tests pin the same contract for launchd.

No macOS hardware is involved: every case drives the seam through mocked
``launchctl`` outcomes.
"""

from __future__ import annotations

import subprocess

import pytest

import hermes_cli.gateway as gateway_cli
import hermes_cli.update_cmd as update_cmd
from hermes_cli.update_cmd import _warn_incomplete_gateway_fleet_restart

LABEL = "ai.hermes.gateway"


class _FakeClock:
    """Monotonic clock that only advances when the code under test sleeps.

    Keeps the poll loop's wall-clock budget honest without spending it: a real
    20s verification timeout would otherwise make this file the slowest in the
    suite, and shortening the timeout would stop testing the throttle window.
    """

    def __init__(self):
        self.now = 1000.0
        self.slept = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    fake = _FakeClock()
    monkeypatch.setattr(gateway_cli.time, "monotonic", fake.monotonic)
    monkeypatch.setattr(gateway_cli.time, "sleep", fake.sleep)
    return fake


@pytest.fixture(autouse=True)
def _no_detached_fallback(monkeypatch):
    """Default every test to "launchd can manage this domain"."""
    monkeypatch.setattr(
        gateway_cli, "_launchd_unsupported_marker_exists", lambda: False
    )


def _supervision_returning(*results):
    """Fake ``_launchctl_label_supervising_process`` yielding ``results`` in order.

    The final value repeats, so a test can say "False twenty times, then True
    from then on".
    """
    seq = list(results)
    calls = []

    def probe(label):
        calls.append(label)
        return seq[min(len(calls) - 1, len(seq) - 1)]

    probe.calls = calls
    return probe


class TestWaitForLaunchdGatewaySupervision:
    def test_returns_true_when_already_supervised(self, monkeypatch, clock):
        """The common case must not cost a single sleep."""
        monkeypatch.setattr(
            gateway_cli,
            "_launchctl_label_supervising_process",
            _supervision_returning(True),
        )

        assert gateway_cli.wait_for_launchd_gateway_supervision(label=LABEL) is True
        assert clock.slept == []

    def test_waits_out_the_launchd_respawn_throttle(self, monkeypatch, clock):
        """A pid that only appears after ~10s is a SUCCESS, not a failure.

        launchd will not relaunch a KeepAlive job more than about once per 10
        seconds, so a gateway that exits promptly leaves the label registered
        with no pid for most of that window.  A one-shot check - or any budget
        shorter than the throttle - would report a perfectly healthy restart as
        a silent failure, which is a worse bug than the one being fixed.
        """
        # 0.5s poll interval: 20 misses is ~10s of throttle, then the pid lands.
        probe = _supervision_returning(*([False] * 20 + [True]))
        monkeypatch.setattr(
            gateway_cli, "_launchctl_label_supervising_process", probe
        )

        assert gateway_cli.wait_for_launchd_gateway_supervision(label=LABEL) is True
        assert sum(clock.slept) == pytest.approx(10.0)

    def test_gives_up_at_the_deadline(self, monkeypatch, clock):
        """A job that never comes back must fail, and must fail bounded."""
        probe = _supervision_returning(False)
        monkeypatch.setattr(
            gateway_cli, "_launchctl_label_supervising_process", probe
        )

        assert (
            gateway_cli.wait_for_launchd_gateway_supervision(
                label=LABEL, timeout=20.0
            )
            is False
        )
        assert sum(clock.slept) <= 20.0
        # The deadline is enforced by wall clock, not by a probe count.
        assert len(probe.calls) == 41

    def test_detached_fallback_is_not_a_failure(self, monkeypatch, clock):
        """On a host where launchd cannot manage the domain, no pid is correct.

        ``_launchd_fallback_to_detached`` is a legitimate outcome (macOS 26+
        unmanageable domains); the gateway runs unsupervised by design there.
        Reporting that as an incomplete update would fail every update on those
        hosts.
        """
        monkeypatch.setattr(
            gateway_cli, "_launchd_unsupported_marker_exists", lambda: True
        )
        probe = _supervision_returning(False)
        monkeypatch.setattr(
            gateway_cli, "_launchctl_label_supervising_process", probe
        )

        assert gateway_cli.wait_for_launchd_gateway_supervision(label=LABEL) is True
        assert probe.calls == []


def _patch_launchd_env(
    monkeypatch,
    *,
    plist_exists=True,
    registered=True,
    restart=None,
    supervised=True,
):
    """Drive ``_restart_macos_launchd_gateways`` through the invoking profile only.

    ``launchd_gateway_labels_for_install`` is pinned to the current label so the
    sibling loop is a no-op: this file is about the invoking profile, which is
    the one branch that was never verified.
    """

    class _Plist:
        def exists(self):
            return plist_exists

    monkeypatch.setattr(gateway_cli, "get_launchd_plist_path", lambda: _Plist())
    monkeypatch.setattr(gateway_cli, "get_launchd_label", lambda: LABEL)
    monkeypatch.setattr(
        gateway_cli, "_launchd_service_registered", lambda label: registered
    )
    monkeypatch.setattr(
        gateway_cli, "launchd_gateway_labels_for_install", lambda: [LABEL]
    )

    calls = {"restart": 0, "verify": 0, "label": None}

    def _restart():
        calls["restart"] += 1
        if restart is not None:
            raise restart

    monkeypatch.setattr(gateway_cli, "launchd_restart", _restart)

    def _verify(*, label=None, **_kw):
        calls["verify"] += 1
        calls["label"] = label
        return supervised

    monkeypatch.setattr(
        gateway_cli, "wait_for_launchd_gateway_supervision", _verify
    )
    return calls


def _run_fleet_restart():
    """Run the real update-path helper and return its two accounting lists."""
    restarted: list = []
    failed_or_stale: list = []
    update_cmd._restart_macos_launchd_gateways(restarted, failed_or_stale, 5.0)
    return restarted, failed_or_stale


class TestInvokingProfileIsVerifiedLikeItsSiblings:
    """The sibling loop already polls for a fresh supervised pid before
    counting a label as restarted.  The invoking profile did not, so the two
    halves of the same function disagreed about what "restarted" means."""

    def test_reports_restarted_only_after_supervision_is_confirmed(
        self, monkeypatch
    ):
        calls = _patch_launchd_env(monkeypatch, supervised=True)

        restarted, failed_or_stale = _run_fleet_restart()

        assert restarted == [LABEL]
        assert failed_or_stale == []
        assert calls["restart"] == 1
        assert calls["verify"] == 1
        assert calls["label"] == LABEL

    def test_unverified_restart_is_not_reported_as_restarted(
        self, monkeypatch, capsys
    ):
        """THE regression (#88848).

        ``launchd_restart()`` returns normally - that is the reported failure's
        exact shape, since the ``_request_gateway_self_restart`` branch returns
        without raising while the reload helper dies afterwards.  Before the
        fix this appended the label to ``restarted_services`` and the update
        reported the gateway as restarted, and exited 0, over a job that was
        deregistered from launchd.
        """
        _patch_launchd_env(monkeypatch, supervised=False)

        restarted, failed_or_stale = _run_fleet_restart()

        assert restarted == []
        # Routed into failed_or_stale_units, which sets
        # gateway_fleet_restart_incomplete and makes the update exit 1.
        assert failed_or_stale == [LABEL]
        assert LABEL in capsys.readouterr().out

    def test_verification_budget_clears_the_respawn_throttle(self):
        """A budget under launchd's ~10s respawn throttle would false-alarm.

        The call site takes the helper's default, so the default is the
        contract that has to stay above the throttle.
        """
        assert gateway_cli.LAUNCHD_SUPERVISION_VERIFY_TIMEOUT >= 15.0

    def test_raised_restart_failure_is_not_verified_and_is_reported(
        self, monkeypatch, capsys
    ):
        """A raised restart is a failed restart, and must not be verified.

        Every path in ``launchd_restart`` that can still leave a working
        gateway - the detached fallback on an unmanageable domain - returns
        rather than raising, so reaching the handler means neither kickstart
        nor bootstrap brought the service back.

        This matters for composition with the open PRs that add verification
        *inside* ``launchd_restart`` (#63304, #72752): they convert this exact
        failure from silent to raised, so the caller must keep routing it into
        ``failed_or_stale_units`` rather than falling through to the verifier.
        """
        exc = subprocess.CalledProcessError(1, ["launchctl"], stderr="boom")
        calls = _patch_launchd_env(monkeypatch, restart=exc)

        restarted, failed_or_stale = _run_fleet_restart()

        assert restarted == []
        assert failed_or_stale == [LABEL]
        assert calls["verify"] == 0
        assert "boom" in capsys.readouterr().out

    def test_no_plist_means_the_gateway_is_not_launchd_managed(self, monkeypatch):
        calls = _patch_launchd_env(monkeypatch, plist_exists=False)

        assert _run_fleet_restart() == ([], [])
        assert calls["restart"] == 0
        assert calls["verify"] == 0

    def test_unregistered_label_is_restarted_not_skipped(self, monkeypatch):
        """A booted-out job (plist present, deregistered) must be RESTARTED.

        FLIPPED by the #74973 fix (salvage #75021): this test used to pin
        'registered=False → nothing to restart', which was precisely the
        silent-skip bug — launchctl list is session-scoped and non-zero
        for booted-out jobs whose plist very much still wants a gateway;
        launchd_restart() owns the bootout/bootstrap ladder for that state.
        """
        calls = _patch_launchd_env(monkeypatch, registered=False)

        assert _run_fleet_restart() == ([LABEL], [])
        assert calls["restart"] == 1
        assert calls["verify"] == 1


class TestIncompleteFleetWarningIsPlatformCorrect:
    def test_macos_recovery_instructions_are_launchctl(self, monkeypatch, capsys):
        """A launchd label must not be handed systemctl commands."""
        monkeypatch.setattr(gateway_cli, "is_macos", lambda: True)

        _warn_incomplete_gateway_fleet_restart([LABEL])

        out = capsys.readouterr().out
        assert "launchctl bootstrap" in out
        assert "systemctl" not in out

    def test_linux_recovery_instructions_are_unchanged(self, monkeypatch, capsys):
        monkeypatch.setattr(gateway_cli, "is_macos", lambda: False)

        _warn_incomplete_gateway_fleet_restart(["hermes-gateway.service"])

        out = capsys.readouterr().out
        assert "systemctl --user restart <unit>" in out
        assert "launchctl" not in out
