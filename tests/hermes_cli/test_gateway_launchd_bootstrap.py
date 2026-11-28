"""launchd bootstrap/status regressions: a disabled label shares EIO with a stale load.

``launchctl bootstrap`` returns ``5: Input/output error`` both for a label still registered in the
domain and for one disabled in launchd's override DB (System Settings → Login Items, a migration
that disabled the old backend, an uninstall). Treating both as "launchd cannot manage this macOS
version" degraded the gateway to a detached process that ``hermes update`` could never restart
(#23387), so every update ended by asking the user to run ``hermes gateway restart`` by hand.
"""

import subprocess
from types import SimpleNamespace

import pytest

import hermes_cli.gateway as gateway_cli


class TestLaunchctlBootstrapEioRetry:
    PLIST = "/tmp/ai.hermes.gateway.plist"
    DOMAIN = "gui/501"
    LABEL = "ai.hermes.gateway"

    def test_eio_triggers_bootout_then_retry(self, monkeypatch):
        calls = []

        def fake_run(cmd, check=True, **kwargs):
            calls.append(cmd)
            bootstrap_calls = [c for c in calls if c[1] == "bootstrap"]
            # First bootstrap hits EIO; enable + bootout clear it; retry succeeds.
            if cmd[1] == "bootstrap" and len(bootstrap_calls) == 1:
                raise subprocess.CalledProcessError(5, cmd)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)

        gateway_cli._launchctl_bootstrap(self.DOMAIN, self.PLIST, self.LABEL)

        assert calls == [
            ["launchctl", "bootstrap", self.DOMAIN, self.PLIST],
            ["launchctl", "enable", f"{self.DOMAIN}/{self.LABEL}"],
            ["launchctl", "bootout", f"{self.DOMAIN}/{self.LABEL}"],
            ["launchctl", "bootstrap", self.DOMAIN, self.PLIST],
        ]

    def test_disabled_label_is_enabled_before_retry(self, monkeypatch):
        # The retry below succeeds only because `launchctl enable` ran first; without it both
        # attempts fail and the caller degrades to a detached gateway no update can restart.
        state = {"enabled": False}

        def fake_run(cmd, check=True, **kwargs):
            if cmd[1] == "enable":
                state["enabled"] = True
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            if cmd[1] == "bootstrap" and not state["enabled"]:
                raise subprocess.CalledProcessError(5, cmd)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)

        gateway_cli._launchctl_bootstrap(self.DOMAIN, self.PLIST, self.LABEL)

        assert state["enabled"] is True

    def test_persistent_eio_reraises_for_domain_fallback(self, monkeypatch):
        # When the retry also fails, the error must propagate so callers apply
        # their _launchctl_domain_unsupported fallback (degrade to detached).
        def fake_run(cmd, check=True, **kwargs):
            if cmd[1] == "bootstrap":
                raise subprocess.CalledProcessError(5, cmd)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)

        with pytest.raises(subprocess.CalledProcessError) as excinfo:
            gateway_cli._launchctl_bootstrap(self.DOMAIN, self.PLIST, self.LABEL)
        assert excinfo.value.returncode == 5


class TestLaunchdStatusDiagnostics:
    def test_names_a_disabled_label(self, tmp_path, monkeypatch, capsys):
        """A disabled label is not "unloaded": name the real cause so the fix is obvious."""
        plist_path = tmp_path / "ai.hermes.gateway.plist"
        plist_path.write_text("<plist/>", encoding="utf-8")
        monkeypatch.setattr(gateway_cli, "get_launchd_plist_path", lambda: plist_path)
        monkeypatch.setattr(gateway_cli, "launchd_plist_is_current", lambda: True)
        monkeypatch.setattr(gateway_cli, "get_hermes_home", lambda: tmp_path)
        monkeypatch.setattr(gateway_cli, "_launchd_domain", lambda: "gui/501")

        label = gateway_cli.get_launchd_label()

        def fake_run(cmd, **kwargs):
            if cmd[:2] == ["launchctl", "list"]:
                return SimpleNamespace(returncode=113, stdout="", stderr="")
            if cmd[:2] == ["launchctl", "print-disabled"]:
                return SimpleNamespace(
                    returncode=0,
                    stdout=f'disabled services = {{\n\t"{label}" => disabled\n}}',
                    stderr="",
                )
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)
        monkeypatch.setattr("gateway.status.get_running_pid", lambda cleanup_stale=False: None)

        gateway_cli.launchd_status()

        out = capsys.readouterr().out
        assert "disabled" in out.lower()
        assert "hermes gateway start" in out
