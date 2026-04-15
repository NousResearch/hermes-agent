from types import SimpleNamespace

import hermes_cli.gateway as gateway


class TestLaunchdDomainSelection:
    def test_falls_back_to_user_domain_when_gui_domain_is_unavailable(self, monkeypatch):
        monkeypatch.setattr(gateway.os, "getuid", lambda: 501)

        calls = []

        def _fake_run(cmd, capture_output=False, text=False, timeout=None, check=False):
            calls.append(cmd)
            domain = cmd[-1]
            if domain == "gui/501":
                return SimpleNamespace(returncode=125, stdout="", stderr="Domain does not support specified action")
            if domain == "user/501":
                return SimpleNamespace(returncode=0, stdout="ok", stderr="")
            raise AssertionError(f"unexpected command: {cmd}")

        monkeypatch.setattr(gateway.subprocess, "run", _fake_run)

        assert gateway._launchd_domain() == "user/501"
        assert calls == [
            ["launchctl", "print", "gui/501"],
            ["launchctl", "print", "user/501"],
        ]

    def test_keeps_gui_domain_as_final_fallback_if_probes_fail(self, monkeypatch):
        monkeypatch.setattr(gateway.os, "getuid", lambda: 501)

        def _fake_run(cmd, capture_output=False, text=False, timeout=None, check=False):
            return SimpleNamespace(returncode=125, stdout="", stderr="nope")

        monkeypatch.setattr(gateway.subprocess, "run", _fake_run)

        assert gateway._launchd_domain() == "gui/501"
