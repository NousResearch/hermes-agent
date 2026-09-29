"""A launchd 'disabled' override is a permanent EIO that only `launchctl enable` clears.

`launchctl disable` — and legacy `launchctl unload`, which the macbook-maintenance skill still
recommends — writes a persistent per-user override row. From then on `launchctl bootstrap` on that
label fails `5: Input/output error` (EIO) *forever*: the same exit code a stale registration
returns, so the existing bootout+retry cannot clear it. Every caller then misreads the host as
"launchd cannot manage this macOS version", degrades to a detached unsupervised gateway, and
reports the fleet DOWN on the next update. Verified live on macOS 26.5.2 (uid 501):
disable -> bootstrap = EIO 5, enable -> bootstrap = rc 0.

These tests pin the *behavior contract* through the facade seam, never the source text.
"""
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import gateway as gateway_cli


PLIST = "/tmp/ai.hermes.gateway.plist"
DOMAIN = "gui/501"
LABEL = "ai.hermes.gateway"


@pytest.fixture
def launchctl(monkeypatch):
    """Fake launchctl. `state` drives which labels the override table has disabled, and whether a
    bootstrap against a disabled label fails EIO the way the real one does. The label is derived
    from the plist basename, as launchd derives it from the plist it is handed."""
    state = {"disabled": set(), "calls": [], "enables": 0}

    def fake_run(cmd, check=True, **kwargs):
        args = list(cmd)
        state["calls"].append(args)
        verb = args[1] if len(args) > 1 else ""
        if verb == "print-disabled":
            rows = "".join(f'\t"{lab}" => disabled\n' for lab in sorted(state["disabled"]))
            return SimpleNamespace(returncode=0, stdout="{\n" + rows + "}\n", stderr="")
        if verb == "enable":
            state["enables"] += 1
            state["disabled"].discard(args[2].rsplit("/", 1)[-1])
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if verb == "bootstrap":
            plist_label = Path(args[-1]).stem
            if plist_label in state["disabled"]:
                raise subprocess.CalledProcessError(5, args)
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)
    return state


class TestDisabledOverrideIsDetected:
    def test_reads_the_override_table_for_this_domain(self, launchctl):
        assert gateway_cli._launchctl_label_is_disabled(DOMAIN, LABEL) is False
        assert ["launchctl", "print-disabled", DOMAIN] in launchctl["calls"]

    def test_a_disabled_label_is_reported(self, launchctl):
        launchctl["disabled"].add(LABEL)
        assert gateway_cli._launchctl_label_is_disabled(DOMAIN, LABEL) is True

    def test_a_sibling_profile_label_never_matches_another_labels_row(self, launchctl):
        # `ai.hermes.gateway-alpha` disabled must not read as `ai.hermes.gateway` disabled.
        launchctl["disabled"].add("ai.hermes.gateway-alpha")
        assert gateway_cli._launchctl_label_is_disabled(DOMAIN, LABEL) is False

    def test_unreadable_table_answers_false_rather_than_raising(self, monkeypatch):
        def fake_run(cmd, check=True, **kwargs):
            raise subprocess.TimeoutExpired(cmd, 15)

        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)
        assert gateway_cli._launchctl_label_is_disabled(DOMAIN, LABEL) is False


class TestBootstrapClearsTheDisabledOverride:
    def test_disabled_label_is_enabled_then_bootstrapped(self, launchctl):
        launchctl["disabled"].add(LABEL)
        gateway_cli._launchctl_bootstrap(DOMAIN, PLIST, LABEL)
        verbs = [c[1] for c in launchctl["calls"]]
        # enable BEFORE the retrying bootstrap, or the retry is the same doomed call.
        assert verbs.index("enable") < len(verbs) - 1
        assert launchctl["enables"] == 1
        assert verbs[-1] == "bootstrap"

    def test_a_stale_registration_does_not_issue_a_pointless_enable(self, monkeypatch):
        # EIO from a stale label (not disabled) must keep the old minimal bootout+retry shape.
        calls = []

        def fake_run(cmd, check=True, **kwargs):
            args = list(cmd)
            calls.append(args)
            if args[1] == "bootstrap" and len([c for c in calls if c[1] == "bootstrap"]) == 1:
                raise subprocess.CalledProcessError(5, args)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)
        gateway_cli._launchctl_bootstrap(DOMAIN, PLIST, LABEL)
        assert [c[1] for c in calls] == ["bootstrap", "print-disabled", "bootout", "bootstrap"]

    def test_persistent_eio_still_raises_for_the_domain_fallback(self, monkeypatch):
        def always_eio(cmd, check=True, **kwargs):
            if cmd[1] == "bootstrap":
                raise subprocess.CalledProcessError(5, list(cmd))
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(gateway_cli.subprocess, "run", always_eio)
        with pytest.raises(subprocess.CalledProcessError):
            gateway_cli._launchctl_bootstrap(DOMAIN, PLIST, LABEL)


class TestRetryLoopClearsTheOverrideInsteadOfStorming:
    def test_retry_clears_the_override_and_then_registers(self, launchctl, monkeypatch):
        import time

        launchctl["disabled"].add(LABEL)
        # A supervised pid only appears once the override has been cleared and bootstrap lands.
        monkeypatch.setattr(
            gateway_cli,
            "_launchctl_label_supervising_process",
            lambda label: launchctl["enables"] > 0,
        )
        ok = gateway_cli._retry_launchctl_bootstrap_until_registered(
            DOMAIN, PLIST, LABEL, deadline=time.monotonic() + 5
        )
        assert ok is True
        assert launchctl["enables"] >= 1
