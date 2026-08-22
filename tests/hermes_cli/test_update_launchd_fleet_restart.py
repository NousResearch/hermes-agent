"""Regression for #41403 — ``hermes update`` must restart ALL macOS launchd gateways.

The macOS branch of the update's fleet-restart step only restarted the
invoking profile's LaunchAgent (``get_launchd_label()`` is profile-scoped).
Sibling ``ai.hermes.gateway-<profile>`` services kept running pre-update
modules cached in ``sys.modules`` and died on their next agent turn once the
new code lazily imported a symbol the old module generation didn't have
(``ImportError: cannot import name ...`` — or, with a wider version gap,
``TypeError``/``AttributeError`` on changed call signatures with garbled
tracebacks, because the source files on disk no longer match the loaded
code objects).

Also covers the launchd-domain review feedback on PR #41403: every sibling
interaction (liveness discovery, kickstart, fresh-PID verification) must be
domain-explicit — ``_launchd_domain()`` caches the *current* profile's
domain, and a sibling bootstrapped in the other supported domain
(``gui/<uid>`` vs ``user/<uid>``) would otherwise be probed or kickstarted
in a domain it does not live in.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

import hermes_cli.gateway as gw
import hermes_cli.profiles
from hermes_cli.gateway import (
    _locate_launchd_gateway_service,
    _parse_launchd_pid_from_print_output,
    _probe_launchd_domain_for_label,
    launchd_gateway_labels_for_install,
)
from hermes_cli.update_cmd import (
    _restart_macos_launchd_gateways,
    _warn_incomplete_gateway_fleet_restart,
)


pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="launchd fleet restart is macOS-only; helpers use POSIX os.getuid",
)

UID = 501

PRINT_RUNNING = (
    "system/com.example = {\n"
    "\tactive count = 1\n"
    "\tstate = running\n"
    "\tpid = 4242\n"
    "\tprogram = /usr/bin/true\n"
    "}\n"
)
PRINT_LOADED_NOT_RUNNING = (
    "system/com.example = {\n"
    "\tactive count = 0\n"
    "\tstate = not running\n"
    "\tprogram = /usr/bin/true\n"
    "}\n"
)


@pytest.fixture(autouse=True)
def _fixed_uid(monkeypatch):
    monkeypatch.setattr(gw.os, "getuid", lambda: UID)


def _completed(returncode: int = 0, stdout: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=""
    )


class _Profile:
    def __init__(self, name, is_default=False):
        self.name = name
        self.is_default = is_default


class TestLaunchdGatewayLabelsForInstall:
    def test_labels_derive_from_this_installs_profiles(self, monkeypatch):
        """The fleet is THIS install's profiles, root first — never a glob of
        the shared per-user LaunchAgents dir. A sandboxed HERMES_HOME (tests,
        side-by-side installs) must not enumerate — and restart — another
        install's services, and the hermetic test suite must not see the dev
        machine's real fleet."""
        monkeypatch.setattr(
            hermes_cli.profiles,
            "list_profiles",
            lambda: [
                _Profile("tfl-wiki"),
                _Profile("default", is_default=True),
                _Profile("merit-ops"),
                _Profile("Bad Name!"),  # cannot map to a service suffix — skipped
            ],
        )
        assert launchd_gateway_labels_for_install() == [
            "ai.hermes.gateway",
            "ai.hermes.gateway-merit-ops",
            "ai.hermes.gateway-tfl-wiki",
        ]

    def test_no_profiles_means_no_fleet(self, monkeypatch):
        monkeypatch.setattr(hermes_cli.profiles, "list_profiles", lambda: [])
        assert launchd_gateway_labels_for_install() == []


class TestParseLaunchdPidFromPrintOutput:
    def test_running_service_pid(self):
        assert _parse_launchd_pid_from_print_output(PRINT_RUNNING) == 4242

    def test_loaded_but_not_running_has_no_pid(self):
        assert _parse_launchd_pid_from_print_output(PRINT_LOADED_NOT_RUNNING) is None


class TestLocateLaunchdGatewayService:
    def test_domains_resolve_per_label_not_from_cache(self, monkeypatch):
        """The #41403 review defect: sibling domains are independent."""
        gui_loaded = {"ai.hermes.gateway-a"}

        def fake_run(cmd, **kwargs):
            assert cmd[:2] == ["launchctl", "print"]
            domain, _, label = cmd[2].rpartition("/")
            in_gui = domain == f"gui/{UID}" and label in gui_loaded
            in_user = domain == f"user/{UID}" and label not in gui_loaded
            if in_gui or in_user:
                return _completed(0, PRINT_RUNNING)
            return _completed(113)

        monkeypatch.setattr(gw.subprocess, "run", fake_run)
        # Simulate a prior current-profile resolution having populated the
        # process-wide cache — per-label lookups must not consult it.
        monkeypatch.setattr(gw, "_resolved_launchd_domain", f"gui/{UID}")

        assert _locate_launchd_gateway_service("ai.hermes.gateway-a") == (
            f"gui/{UID}",
            4242,
        )
        assert _locate_launchd_gateway_service("ai.hermes.gateway-b") == (
            f"user/{UID}",
            4242,
        )

    def test_loaded_without_live_process(self, monkeypatch):
        monkeypatch.setattr(
            gw.subprocess,
            "run",
            lambda *a, **k: _completed(0, PRINT_LOADED_NOT_RUNNING),
        )
        assert _locate_launchd_gateway_service("ai.hermes.gateway-x") == (
            f"gui/{UID}",
            None,
        )

    def test_not_loaded_in_either_domain(self, monkeypatch):
        monkeypatch.setattr(gw.subprocess, "run", lambda *a, **k: _completed(113))
        assert _locate_launchd_gateway_service("ai.hermes.gateway-x") == (None, None)

    def test_timeout_propagates_to_caller(self, monkeypatch):
        """A wedged launchctl must surface as a failure, not read as
        'unloaded' — the update path owns per-label failure accounting."""

        def fake_run(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=5)

        monkeypatch.setattr(gw.subprocess, "run", fake_run)
        with pytest.raises(subprocess.TimeoutExpired):
            _locate_launchd_gateway_service("ai.hermes.gateway-x")


class TestProbeLaunchdDomainForLabel:
    def test_unloaded_label_falls_back_to_managername(self, monkeypatch):
        def fake_run(cmd, **kwargs):
            if cmd[:2] == ["launchctl", "print"]:
                raise subprocess.CalledProcessError(113, cmd)
            if cmd == ["launchctl", "managername"]:
                return _completed(0, "Aqua\n")
            raise AssertionError(f"unexpected command {cmd}")

        monkeypatch.setattr(gw.subprocess, "run", fake_run)
        assert _probe_launchd_domain_for_label("ai.hermes.gateway-x") == f"gui/{UID}"

    def test_unloaded_label_defaults_to_user_domain(self, monkeypatch):
        def fake_run(cmd, **kwargs):
            if cmd[:2] == ["launchctl", "print"]:
                raise subprocess.CalledProcessError(113, cmd)
            if cmd == ["launchctl", "managername"]:
                return _completed(0, "Background\n")
            raise AssertionError(f"unexpected command {cmd}")

        monkeypatch.setattr(gw.subprocess, "run", fake_run)
        assert _probe_launchd_domain_for_label("ai.hermes.gateway-x") == f"user/{UID}"


class TestGetServicePidsScoping:
    def _wire(self, monkeypatch):
        monkeypatch.setattr(gw, "is_macos", lambda: True)
        monkeypatch.setattr(gw, "supports_systemd_services", lambda: False)
        monkeypatch.setattr(gw, "get_launchd_label", lambda: "ai.hermes.gateway")
        monkeypatch.setattr(
            gw,
            "launchd_gateway_labels_for_install",
            lambda: ["ai.hermes.gateway", "ai.hermes.gateway-a", "ai.hermes.gateway-b"],
        )
        located = {
            "ai.hermes.gateway": (f"gui/{UID}", 100),
            "ai.hermes.gateway-a": (f"gui/{UID}", 200),
            "ai.hermes.gateway-b": (None, None),  # not bootstrapped
        }
        monkeypatch.setattr(
            gw, "_locate_launchd_gateway_service", lambda label: located[label]
        )

    def test_all_profiles_returns_every_gateway_service_pid(self, monkeypatch):
        """The update sweep's exclude-set must protect ALL freshly-restarted
        services, not only the invoking profile's (else the sweep SIGTERMs
        gateways launchd just respawned)."""
        self._wire(monkeypatch)
        assert gw._get_service_pids(all_profiles=True) == {100, 200}

    def test_default_stays_scoped_to_current_profile(self, monkeypatch):
        """Regression guard: default-scope callers (gateway status, cron,
        stop_profile_gateway's orphan reaper) must NOT start seeing sibling
        service PIDs — the reaper SIGTERM/SIGKILLs what they feed it."""
        self._wire(monkeypatch)
        assert gw._get_service_pids() == {100}

    def test_find_gateway_pids_passes_profile_scope_through(self, monkeypatch):
        calls: list[bool] = []
        monkeypatch.setattr(
            gw,
            "_get_service_pids",
            lambda all_profiles=False: (calls.append(all_profiles), set())[1],
        )
        monkeypatch.setattr(gw, "_scan_gateway_pids", lambda *a, **k: [])
        monkeypatch.setattr(gw, "supports_systemd_services", lambda: True)

        gw.find_gateway_pids(all_profiles=False)
        gw.find_gateway_pids(all_profiles=True)
        assert calls == [False, True]


def _fleet(monkeypatch, tmp_path, *, current, labels, located,
           registered=None, plist_exists=True,
           drain_results=None, kick_errors=None, wait_results=None):
    """Wire a fake launchd fleet through hermes_cli.gateway seams.

    ``located`` maps label -> (domain, pid) as ``_locate_launchd_gateway_service``
    would return it (values may also be exceptions to raise). ``registered``
    maps label -> bool for the current-profile ``launchctl list`` gate and
    defaults to "located in some domain". Returns a SimpleNamespace of
    recorder lists: rec.kickstarts, rec.drains, rec.current_restarts, rec.waits, locates,
    registered_checks.
    """
    from types import SimpleNamespace

    rec = SimpleNamespace(
        kickstarts=[], drains=[], current_restarts=[], waits=[],
        locates=[], registered_checks=[],
    )

    plist = tmp_path / f"{current}.plist"
    if plist_exists:
        plist.write_text("<plist/>")

    def fake_locate(label):
        rec.locates.append(label)
        value = located[label]
        if isinstance(value, Exception):
            raise value
        return value

    def fake_registered(label):
        rec.registered_checks.append(label)
        if registered is not None:
            return registered[label]
        value = located.get(label)
        return (
            value is not None
            and not isinstance(value, Exception)
            and value[0] is not None
        )

    monkeypatch.setattr(gw, "get_launchd_label", lambda: current)
    monkeypatch.setattr(gw, "get_launchd_plist_path", lambda: plist)
    monkeypatch.setattr(gw, "launchd_gateway_labels_for_install", lambda: list(labels))
    monkeypatch.setattr(gw, "_locate_launchd_gateway_service", fake_locate)
    monkeypatch.setattr(gw, "_launchd_service_registered", fake_registered)
    monkeypatch.setattr(
        gw,
        "_graceful_restart_via_sigusr1",
        lambda pid, drain_timeout: (rec.drains.append(pid), (drain_results or {}).get(pid, False))[1],
    )

    def fake_kickstart(label, domain):
        err = (kick_errors or {}).get(label)
        if err is not None:
            raise err
        rec.kickstarts.append(f"{domain}/{label}")

    monkeypatch.setattr(gw, "_launchd_kickstart", fake_kickstart)

    def fake_wait(label, old_pid, timeout, domain):
        rec.waits.append(f"{domain}/{label}")
        return (wait_results or {}).get(label, True)

    monkeypatch.setattr(gw, "_wait_for_launchd_service_pid", fake_wait)
    monkeypatch.setattr(
        gw, "launchd_restart", lambda: rec.current_restarts.append(current)
    )
    return rec


class TestRestartMacosLaunchdGateways:
    def test_current_delegates_and_siblings_kickstart_in_own_domains(
        self, monkeypatch, tmp_path
    ):
        """Current profile keeps launchd_restart(); every sibling (including
        the root gateway when a named profile invokes the update) is
        kickstarted — and verified — in the domain IT was located in."""
        current = "ai.hermes.gateway-merit-ops"
        rec = _fleet(
            monkeypatch,
            tmp_path,
            current=current,
            labels=["ai.hermes.gateway", current, "ai.hermes.gateway-user-scoped"],
            located={
                "ai.hermes.gateway": (f"gui/{UID}", 100),
                current: (f"gui/{UID}", 200),
                "ai.hermes.gateway-user-scoped": (f"user/{UID}", 300),
            },
        )
        restarted: list[str] = []
        failed: list[str] = []

        _restart_macos_launchd_gateways(restarted, failed, drain_budget=0.0)

        assert rec.current_restarts == [current]
        assert rec.kickstarts == [
            f"gui/{UID}/ai.hermes.gateway",
            f"user/{UID}/ai.hermes.gateway-user-scoped",
        ]
        assert rec.waits == [
            f"gui/{UID}/ai.hermes.gateway",
            f"user/{UID}/ai.hermes.gateway-user-scoped",
        ]
        assert restarted == [
            current,
            "ai.hermes.gateway",
            "ai.hermes.gateway-user-scoped",
        ]
        assert failed == []
        # Siblings were drained before the hard kickstart.
        assert set(rec.drains) == {100, 300}

    def test_current_profile_without_plist_makes_no_launchctl_calls(
        self, monkeypatch, tmp_path
    ):
        """Upstream gate order preserved: no plist → the current profile is
        skipped without ANY launchctl interaction (no registered probe, no
        locate) — and definitely without inventing a failure. Siblings are
        still processed."""
        current = "ai.hermes.gateway"
        rec = _fleet(
            monkeypatch,
            tmp_path,
            current=current,
            labels=[current, "ai.hermes.gateway-a"],
            located={"ai.hermes.gateway-a": (f"gui/{UID}", 200)},
            plist_exists=False,
        )
        restarted: list[str] = []
        failed: list[str] = []

        _restart_macos_launchd_gateways(restarted, failed, drain_budget=0.0)

        assert rec.current_restarts == []
        assert current not in rec.registered_checks
        assert current not in rec.locates
        assert restarted == ["ai.hermes.gateway-a"]
        assert failed == []

    def test_current_profile_registered_but_unlocatable_still_restarts(
        self, monkeypatch, tmp_path
    ):
        """macOS-26 quirk: a label can be `launchctl list`-registered while
        both explicit gui/user `launchctl print` probes fail (domain doesn't
        support service management). The gate must use the registered
        predicate and hand off to launchd_restart(), which owns the
        domain-unsupported fallback — locate is for siblings only."""
        current = "ai.hermes.gateway"
        rec = _fleet(
            monkeypatch,
            tmp_path,
            current=current,
            labels=[current],
            located={current: (None, None)},
            registered={current: True},
        )
        restarted: list[str] = []
        failed: list[str] = []

        _restart_macos_launchd_gateways(restarted, failed, drain_budget=0.0)

        assert rec.current_restarts == [current]
        assert current not in rec.locates
        assert restarted == [current]
        assert failed == []

    def test_unbootstrapped_sibling_is_skipped_not_failed(
        self, monkeypatch, tmp_path
    ):
        rec = _fleet(
            monkeypatch,
            tmp_path,
            current="ai.hermes.gateway",
            labels=["ai.hermes.gateway", "ai.hermes.gateway-idle"],
            located={
                "ai.hermes.gateway": (f"gui/{UID}", 100),
                "ai.hermes.gateway-idle": (None, None),
            },
        )
        restarted: list[str] = []
        failed: list[str] = []

        _restart_macos_launchd_gateways(restarted, failed, drain_budget=0.0)

        assert rec.kickstarts == []
        assert restarted == ["ai.hermes.gateway"]
        assert failed == []

    def test_loaded_but_not_running_sibling_is_kickstarted(
        self, monkeypatch, tmp_path
    ):
        """A bootstrapped service with no live process still holds the old
        code path for its next launch trigger — kickstart it (no drain)."""
        rec = _fleet(
            monkeypatch,
            tmp_path,
            current="ai.hermes.gateway",
            labels=["ai.hermes.gateway", "ai.hermes.gateway-dormant"],
            located={
                "ai.hermes.gateway": (f"gui/{UID}", 100),
                "ai.hermes.gateway-dormant": (f"gui/{UID}", None),
            },
        )
        restarted: list[str] = []
        failed: list[str] = []

        _restart_macos_launchd_gateways(restarted, failed, drain_budget=0.0)

        assert rec.drains == []
        assert rec.kickstarts == [f"gui/{UID}/ai.hermes.gateway-dormant"]
        assert restarted == ["ai.hermes.gateway", "ai.hermes.gateway-dormant"]
        assert failed == []

    def test_graceful_drain_with_keepalive_respawn_skips_kickstart(
        self, monkeypatch, tmp_path
    ):
        """When SIGUSR1 rec.drains the sibling and KeepAlive already respawned it
        on a fresh PID, a second hard kickstart would kill the new process."""
        rec = _fleet(
            monkeypatch,
            tmp_path,
            current="ai.hermes.gateway",
            labels=["ai.hermes.gateway", "ai.hermes.gateway-a"],
            located={
                "ai.hermes.gateway": (f"gui/{UID}", 100),
                "ai.hermes.gateway-a": (f"gui/{UID}", 200),
            },
            drain_results={200: True},
        )
        restarted: list[str] = []
        failed: list[str] = []

        _restart_macos_launchd_gateways(restarted, failed, drain_budget=0.0)

        assert rec.drains == [200]
        assert rec.kickstarts == []
        assert rec.waits == [f"gui/{UID}/ai.hermes.gateway-a"]
        assert restarted == ["ai.hermes.gateway", "ai.hermes.gateway-a"]
        assert failed == []

    def test_kickstart_failure_is_recorded_and_rest_continue(
        self, monkeypatch, tmp_path
    ):
        rec = _fleet(
            monkeypatch,
            tmp_path,
            current="ai.hermes.gateway",
            labels=[
                "ai.hermes.gateway",
                "ai.hermes.gateway-bad",
                "ai.hermes.gateway-good",
            ],
            located={
                "ai.hermes.gateway": (f"gui/{UID}", 100),
                "ai.hermes.gateway-bad": (f"gui/{UID}", 200),
                "ai.hermes.gateway-good": (f"gui/{UID}", 300),
            },
            kick_errors={
                "ai.hermes.gateway-bad": subprocess.CalledProcessError(
                    5, ["launchctl", "kickstart"]
                )
            },
        )
        restarted: list[str] = []
        failed: list[str] = []

        _restart_macos_launchd_gateways(restarted, failed, drain_budget=0.0)

        assert failed == ["ai.hermes.gateway-bad"]
        assert rec.kickstarts == [f"gui/{UID}/ai.hermes.gateway-good"]
        assert restarted == ["ai.hermes.gateway", "ai.hermes.gateway-good"]

    def test_timeout_during_discovery_is_failed_and_rest_continue(
        self, monkeypatch, tmp_path
    ):
        """A wedged launchctl during liveness discovery must be accounted as
        a failure (the sibling may still be on old code), not silently
        skipped — and must not abort the remaining fleet (#68523 parity)."""
        rec = _fleet(
            monkeypatch,
            tmp_path,
            current="ai.hermes.gateway",
            labels=[
                "ai.hermes.gateway",
                "ai.hermes.gateway-wedged",
                "ai.hermes.gateway-after",
            ],
            located={
                "ai.hermes.gateway": (f"gui/{UID}", 100),
                "ai.hermes.gateway-wedged": subprocess.TimeoutExpired(
                    cmd=["launchctl", "print"], timeout=5
                ),
                "ai.hermes.gateway-after": (f"gui/{UID}", 300),
            },
        )
        restarted: list[str] = []
        failed: list[str] = []

        _restart_macos_launchd_gateways(restarted, failed, drain_budget=0.0)

        assert failed == ["ai.hermes.gateway-wedged"]
        assert rec.kickstarts == [f"gui/{UID}/ai.hermes.gateway-after"]
        assert restarted == ["ai.hermes.gateway", "ai.hermes.gateway-after"]

    def test_timeout_during_kickstart_is_failed_and_rest_continue(
        self, monkeypatch, tmp_path
    ):
        rec = _fleet(
            monkeypatch,
            tmp_path,
            current="ai.hermes.gateway",
            labels=[
                "ai.hermes.gateway",
                "ai.hermes.gateway-wedged",
                "ai.hermes.gateway-after",
            ],
            located={
                "ai.hermes.gateway": (f"gui/{UID}", 100),
                "ai.hermes.gateway-wedged": (f"gui/{UID}", 200),
                "ai.hermes.gateway-after": (f"gui/{UID}", 300),
            },
            kick_errors={
                "ai.hermes.gateway-wedged": subprocess.TimeoutExpired(
                    cmd=["launchctl", "kickstart"], timeout=90
                )
            },
        )
        restarted: list[str] = []
        failed: list[str] = []

        _restart_macos_launchd_gateways(restarted, failed, drain_budget=0.0)

        assert failed == ["ai.hermes.gateway-wedged"]
        assert rec.kickstarts == [f"gui/{UID}/ai.hermes.gateway-after"]
        assert restarted == ["ai.hermes.gateway", "ai.hermes.gateway-after"]

    def test_sibling_that_never_comes_back_is_failed(self, monkeypatch, tmp_path):
        rec = _fleet(
            monkeypatch,
            tmp_path,
            current="ai.hermes.gateway",
            labels=["ai.hermes.gateway", "ai.hermes.gateway-zombie"],
            located={
                "ai.hermes.gateway": (f"gui/{UID}", 100),
                "ai.hermes.gateway-zombie": (f"gui/{UID}", 200),
            },
            wait_results={"ai.hermes.gateway-zombie": False},
        )
        restarted: list[str] = []
        failed: list[str] = []

        _restart_macos_launchd_gateways(restarted, failed, drain_budget=0.0)

        assert restarted == ["ai.hermes.gateway"]
        assert failed == ["ai.hermes.gateway-zombie"]


class TestWaitForLaunchdServicePid:
    def test_returns_true_once_pid_changes(self, monkeypatch):
        pids = iter([200, 200, 4242])
        monkeypatch.setattr(
            gw,
            "_launchd_print_service_pid",
            lambda domain, label: (True, next(pids)),
        )
        monkeypatch.setattr(gw.time, "sleep", lambda _s: None)
        assert gw._wait_for_launchd_service_pid(
            "ai.hermes.gateway-x", old_pid=200, timeout=5.0, domain=f"gui/{UID}"
        )

    def test_returns_false_when_pid_never_changes(self, monkeypatch):
        clock = iter(float(i) for i in range(100))
        monkeypatch.setattr(gw.time, "monotonic", lambda: next(clock))
        monkeypatch.setattr(gw.time, "sleep", lambda _s: None)
        monkeypatch.setattr(
            gw,
            "_launchd_print_service_pid",
            lambda domain, label: (True, 200),
        )
        assert not gw._wait_for_launchd_service_pid(
            "ai.hermes.gateway-x", old_pid=200, timeout=3.0, domain=f"gui/{UID}"
        )


class TestIncompleteWarningMentionsLaunchctl:
    def test_launchd_labels_get_launchctl_hint(self, capsys):
        _warn_incomplete_gateway_fleet_restart(["ai.hermes.gateway-merit-ops"])
        out = capsys.readouterr().out
        assert "Update incomplete" in out
        assert "launchctl kickstart -k" in out

    def test_systemd_units_keep_systemctl_hint(self, capsys):
        _warn_incomplete_gateway_fleet_restart(["hermes-gateway-coder"])
        out = capsys.readouterr().out
        assert "systemctl" in out
        assert "launchctl" not in out
