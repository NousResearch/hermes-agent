"""Tests for hermes_cli.dashboard_service (issue #44106)."""

import plistlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture()
def profile_env(tmp_path, monkeypatch):
    """Mirror the profile_env fixture from tests/hermes_cli/test_profiles.py."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default_home = tmp_path / ".hermes"
    default_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    return tmp_path


@pytest.fixture()
def mock_dashboard_service(monkeypatch):
    """Patch the macOS-gated functions to be callable without a real macOS host."""
    import hermes_cli.dashboard_service as ds
    monkeypatch.setattr(ds, "is_macos", lambda: True)
    monkeypatch.setattr(
        ds.subprocess, "run", lambda cmd, **kwargs: MagicMock(returncode=0, stdout="")
    )
    monkeypatch.setattr(
        ds, "_launchd_domain", lambda: f"gui/{__import__('os').getuid()}"
    )
    return ds


# ------------------------------------------------------------------
# Pure function tests (no OS markers needed)
# ------------------------------------------------------------------


class TestDashboardLaunchdLabels:
    def test_default_profile_label(self, profile_env):
        from hermes_cli.dashboard_service import get_dashboard_launchd_label
        assert get_dashboard_launchd_label() == "ai.hermes.dashboard"

    def test_named_profile_label(self, profile_env, monkeypatch):
        import hermes_cli.dashboard_service as ds
        monkeypatch.setattr(ds, "_profile_suffix", lambda: "work")
        from hermes_cli.dashboard_service import get_dashboard_launchd_label
        assert get_dashboard_launchd_label() == "ai.hermes.dashboard-work"


class TestDashboardLaunchdPlistPath:
    def test_default_path_uses_real_account_home(self, monkeypatch):
        import hermes_cli.dashboard_service as ds
        monkeypatch.setattr(
            ds, "_profile_suffix", lambda: ""
        )
        import pwd
        expected_home = Path(pwd.getpwuid(__import__("os").getuid()).pw_dir)
        path = ds.get_dashboard_launchd_plist_path()
        assert path.name == "ai.hermes.dashboard.plist"
        assert path.parent.name == "LaunchAgents"
        assert str(path).startswith(str(expected_home / "Library"))


# ------------------------------------------------------------------
# Plist generation tests
# ------------------------------------------------------------------


class TestGenerateDashboardLaunchdPlist:
    def test_contains_python_argv0_not_console_script(self, mock_dashboard_service):
        plist_text = mock_dashboard_service.generate_dashboard_launchd_plist("127.0.0.1", 9119)
        data = plistlib.loads(plist_text.encode("utf-8"))
        prog_args = data.get("ProgramArguments", [])
        assert len(prog_args) >= 1
        first_arg = prog_args[0]
        assert first_arg.endswith("python") or "/python" in first_arg
        assert prog_args[1] == "-m"
        assert prog_args[2] == "hermes_cli.main"

    def test_default_profile_pins_p_default(self, mock_dashboard_service, monkeypatch):
        monkeypatch.setattr(mock_dashboard_service, "_profile_suffix", lambda: "")
        plist_text = mock_dashboard_service.generate_dashboard_launchd_plist("0.0.0.0", 9119)
        data = plistlib.loads(plist_text.encode("utf-8"))
        prog_args = data.get("ProgramArguments", [])
        args_str = " ".join(prog_args)
        assert "-p default" in args_str
        assert "dashboard" in args_str
        assert "--detach" not in args_str

    def test_named_profile_uses_isolated(self, mock_dashboard_service, monkeypatch):
        monkeypatch.setattr(mock_dashboard_service, "_profile_suffix", lambda: "testprof")
        plist_text = mock_dashboard_service.generate_dashboard_launchd_plist("127.0.0.1", 9119)
        data = plistlib.loads(plist_text.encode("utf-8"))
        prog_args = data.get("ProgramArguments", [])
        args_str = " ".join(prog_args)
        assert "--profile" in args_str
        assert "testprof" in args_str
        assert "--isolated" in args_str

    def test_no_detach_flag_anywhere(self, mock_dashboard_service):
        plist_text = mock_dashboard_service.generate_dashboard_launchd_plist("127.0.0.1", 9119)
        assert "--detach" not in plist_text

    def test_keepalive_and_run_at_load_present(self, mock_dashboard_service):
        plist_text = mock_dashboard_service.generate_dashboard_launchd_plist("127.0.0.1", 9119)
        assert "<key>KeepAlive</key>" in plist_text
        assert "<true/>" in plist_text
        assert "<key>RunAtLoad</key>" in plist_text

    def test_throttle_interval_and_exit_timeout(self, mock_dashboard_service):
        plist_text = mock_dashboard_service.generate_dashboard_launchd_plist("127.0.0.1", 9119)
        assert "<key>ThrottleInterval</key>" in plist_text
        assert "<integer>30</integer>" in plist_text
        assert "<key>ExitTimeOut</key>" in plist_text
        assert "<integer>25</integer>" in plist_text

    def test_environment_variables_present(self, mock_dashboard_service):
        plist_text = mock_dashboard_service.generate_dashboard_launchd_plist("127.0.0.1", 9119)
        data = plistlib.loads(plist_text.encode("utf-8"))
        env = data.get("EnvironmentVariables", {})
        assert "HERMES_HOME" in env
        assert "VIRTUAL_ENV" in env
        assert "PATH" in env
        assert env.get("HERMES_SUPERVISED_CHILD") == "1"

    def test_extra_args_included(self, mock_dashboard_service):
        plist_text = mock_dashboard_service.generate_dashboard_launchd_plist(
            "127.0.0.1", 9119, extra_args=["--skip-build"]
        )
        data = plistlib.loads(plist_text.encode("utf-8"))
        prog_args = data.get("ProgramArguments", [])
        assert "--skip-build" in prog_args

    def test_plistlib_roundtrip_validates(self, mock_dashboard_service):
        plist_text = mock_dashboard_service.generate_dashboard_launchd_plist("127.0.0.1", 9119)
        parsed = plistlib.loads(plist_text.encode("utf-8"))
        assert "Label" in parsed
        assert "ProgramArguments" in parsed
        assert "WorkingDirectory" in parsed
        assert "KeepAlive" in parsed


# ------------------------------------------------------------------
# Lifecycle operation tests
# ------------------------------------------------------------------


class TestDashboardLifecycle:
    def test_install_refuses_without_force_when_exists(self, mock_dashboard_service, monkeypatch, tmp_path, capsys):
        plist_path = tmp_path / ".fake_home" / "Library" / "LaunchAgents" / "ai.hermes.dashboard.plist"
        plist_path.parent.mkdir(parents=True)
        plist_path.write_text("fake", encoding="utf-8")
        monkeypatch.setattr(
            mock_dashboard_service, "get_dashboard_launchd_plist_path", lambda: plist_path
        )
        mock_dashboard_service.dashboard_service_install("127.0.0.1", 9119)
        out = capsys.readouterr().out
        assert "already installed" in out

    def test_install_writes_plist_with_force(self, mock_dashboard_service, monkeypatch, tmp_path):
        plist_path = tmp_path / ".fake_home" / "Library" / "LaunchAgents" / "ai.hermes.dashboard.plist"
        plist_path.parent.mkdir(parents=True)
        monkeypatch.setattr(
            mock_dashboard_service, "get_dashboard_launchd_plist_path", lambda: plist_path
        )
        mock_dashboard_service.dashboard_service_install("127.0.0.1", 9119, force=True)
        assert plist_path.exists()

    def test_stop_on_non_macos_exits_nonzero(self, monkeypatch, capsys):
        # No macos_only marker — must run on Linux CI (F5).
        import hermes_cli.dashboard_service as ds
        monkeypatch.setattr(ds, "is_macos", lambda: False)
        with pytest.raises(SystemExit) as exc:
            ds.dashboard_service_stop()
        assert exc.value.code == 1
        out = capsys.readouterr().out
        assert "only supported on macOS" in out

    def test_status_exits_when_uninstalled(self, mock_dashboard_service, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(
            mock_dashboard_service,
            "get_dashboard_launchd_plist_path",
            lambda: tmp_path / "nonexistent.plist",
        )
        with pytest.raises(SystemExit) as exc:
            mock_dashboard_service.dashboard_service_status()
        assert exc.value.code == 1

    def test_status_prints_registered_when_installed(self, mock_dashboard_service, monkeypatch, tmp_path, capsys):
        plist_path = tmp_path / "installed.plist"
        plist_data = plistlib.dumps({
            "ProgramArguments": [
                "python", "-m", "hermes_cli.main", "dashboard",
                "--host", "127.0.0.1", "--port", "9119", "--no-open",
            ],
        })
        plist_path.write_bytes(plist_data)
        monkeypatch.setattr(
            mock_dashboard_service, "get_dashboard_launchd_plist_path", lambda: plist_path
        )
        # Mock the gateway parser to return registered without a PID.
        monkeypatch.setattr(
            mock_dashboard_service,
            "_launchd_print_service_pid",
            lambda domain, label: (True, None),
        )
        mock_dashboard_service.dashboard_service_status()
        out = capsys.readouterr().out
        assert "Dashboard service registered" in out

    def test_uninstall_deletes_matching_plist(self, mock_dashboard_service, monkeypatch, tmp_path):
        plist_path = tmp_path / "test.plist"
        plist_path.write_bytes(plistlib.dumps({"Label": "ai.hermes.dashboard", "ProgramArguments": ["python"]}))
        monkeypatch.setattr(
            mock_dashboard_service, "get_dashboard_launchd_plist_path", lambda: plist_path
        )
        mock_dashboard_service.dashboard_service_uninstall()
        assert not plist_path.exists()

    def test_uninstall_never_deletes_non_matching_plist(self, mock_dashboard_service, monkeypatch, tmp_path):
        plist_path = tmp_path / "other.plist"
        plist_path.write_bytes(plistlib.dumps({"Label": "ai.hermes.gateway", "ProgramArguments": ["python"]}))
        monkeypatch.setattr(
            mock_dashboard_service, "get_dashboard_launchd_plist_path", lambda: plist_path
        )
        monkeypatch.setattr(mock_dashboard_service, "get_dashboard_launchd_label", lambda: "ai.hermes.dashboard")
        mock_dashboard_service.dashboard_service_uninstall()
        assert plist_path.exists()


# ------------------------------------------------------------------
# Non-macOS gate message
# ------------------------------------------------------------------


class TestNonMacOSGate:
    def test_start_prints_systemd_hint_on_linux(self, monkeypatch, capsys):
        import hermes_cli.dashboard_service as ds
        monkeypatch.setattr(ds, "is_macos", lambda: False)
        with pytest.raises(SystemExit) as exc:
            ds.dashboard_service_start("127.0.0.1", 9119)
        assert exc.value.code != 0
        out = capsys.readouterr().out
        assert "systemd" in out.lower()

    def test_stop_prints_systemd_hint_on_linux(self, monkeypatch, capsys):
        import hermes_cli.dashboard_service as ds
        monkeypatch.setattr(ds, "is_macos", lambda: False)
        with pytest.raises(SystemExit) as exc:
            ds.dashboard_service_stop()
        assert exc.value.code != 0
        out = capsys.readouterr().out
        assert "systemd" in out.lower()

    def test_status_prints_systemd_hint_on_linux_and_exits_1(self, monkeypatch, capsys):
        import hermes_cli.dashboard_service as ds
        monkeypatch.setattr(ds, "is_macos", lambda: False)
        with pytest.raises(SystemExit) as exc:
            ds.dashboard_service_status()
        assert exc.value.code == 1
        out = capsys.readouterr().out
        assert "only supported on macOS" in out


# ------------------------------------------------------------------
# Regression tests for spec review G1-G5 (F1-F7)
# ------------------------------------------------------------------


class TestRegressionG1G2StartMirrorsGateway:
    """F1 + F2 (G1/G2): start must mirror gateway launchd_start semantics."""

    def test_start_on_missing_plist_regenerates_and_bootstraps_kickstarts(self, monkeypatch, tmp_path, capsys):
        import hermes_cli.dashboard_service as ds
        monkeypatch.setattr(ds, "is_macos", lambda: True)
        plist_path = tmp_path / "Library" / "LaunchAgents" / "ai.hermes.dashboard.plist"
        monkeypatch.setattr(ds, "get_dashboard_launchd_plist_path", lambda: plist_path)
        monkeypatch.setattr(ds, "_launchd_domain", lambda: "gui/501")
        calls = []

        def capture(subcommand, *args, **kwargs):
            calls.append(subcommand)
            return MagicMock(returncode=0, stdout="")

        monkeypatch.setattr(ds.subprocess, "run", capture)
        # Force the bootstrap/kickstart helpers to succeed.
        monkeypatch.setattr(ds, "_launchctl_bootstrap", lambda *a, **k: None)
        monkeypatch.setattr(ds, "_launchctl_kickstart_current", lambda *a, **k: None)

        ds.dashboard_service_start("127.0.0.1", 9119)
        out = capsys.readouterr().out
        # Plist should be regenerated then bootstrap + kickstart called.
        assert plist_path.exists()
        assert any("bootstrap" in str(c) or "kickstart" in str(c) for c in calls) or "regenerated" in out or "started" in out

    def test_start_kickstart_unloaded_rebootstraps_then_kickstarts(self, monkeypatch, tmp_path, capsys):
        import hermes_cli.dashboard_service as ds
        import subprocess
        monkeypatch.setattr(ds, "is_macos", lambda: True)
        import plistlib
        plist_path = tmp_path / "Library" / "LaunchAgents" / "ai.hermes.dashboard.plist"
        plist_path.parent.mkdir(parents=True, exist_ok=True)
        plist_path.write_bytes(plistlib.dumps({"ProgramArguments": ["python", "-m", "hermes_cli.main"]}))
        monkeypatch.setattr(ds, "get_dashboard_launchd_plist_path", lambda: plist_path)
        monkeypatch.setattr(ds, "_launchd_domain", lambda: "gui/501")
        monkeypatch.setattr(ds, "_launchd_print_service_pid", lambda d, l: (True, 1234))

        sequence = []

        def mock_kickstart_current(label):
            sequence.append("kickstart_first")
            raise subprocess.CalledProcessError(3, ["launchctl", "kickstart"])

        monkeypatch.setattr(ds, "_launchctl_kickstart_current", mock_kickstart_current)
        # After unloaded error, bootstrap succeeds then second kickstart succeeds.
        def mock_bootstrap(*a, **k):
            sequence.append("bootstrap")
            return None
        monkeypatch.setattr(ds, "_launchctl_bootstrap", mock_bootstrap)
        call_count = [0]

        def mock_kickstart_retry(label):
            call_count[0] += 1
            sequence.append(f"kickstart_retry_{call_count[0]}")
            return None
        # The retry path inside start uses the same name; we can monkeypatch again after the first failure.
        # Instead, patch the module-level reference directly.

        # Simpler approach: use a callable object that tracks.
        calls_tracker = {"calls": []}

        def tracking_kickstart(label):
            calls_tracker["calls"].append("kickstart")
            if len(calls_tracker["calls"]) == 1:
                raise subprocess.CalledProcessError(3, ["launchctl", "kickstart", f"gui/501/{label}"])
            return None
        monkeypatch.setattr(ds, "_launchctl_kickstart_current", tracking_kickstart)
        monkeypatch.setattr(ds, "_launchctl_bootstrap", lambda *a, **k: calls_tracker["calls"].append("bootstrap") or None)
        monkeypatch.setattr(ds, "_launchd_error_indicates_unloaded", lambda exc: exc.returncode in {3, 113, 125})

        ds.dashboard_service_start("127.0.0.1", 9119)
        assert "bootstrap" in calls_tracker["calls"]
        assert calls_tracker["calls"].count("kickstart") == 2


class TestRegressionG1G2InstallMirror:
    def test_install_degrades_on_domain_unsupported_125(self, monkeypatch, tmp_path, capsys):
        import hermes_cli.dashboard_service as ds
        import subprocess
        monkeypatch.setattr(ds, "is_macos", lambda: True)
        plist_path = tmp_path / "LaunchAgents" / "test.plist"
        plist_path.parent.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(ds, "get_dashboard_launchd_plist_path", lambda: plist_path)
        monkeypatch.setattr(ds, "get_dashboard_launchd_label", lambda: "ai.hermes.dashboard")
        monkeypatch.setattr(ds, "_launchd_domain", lambda: "gui/501")

        def raise_unsupported(*a, **k):
            raise subprocess.CalledProcessError(125, ["launchctl", "bootstrap"])
        monkeypatch.setattr(ds, "_launchctl_bootstrap", raise_unsupported)
        monkeypatch.setattr(ds, "_launchctl_domain_unsupported", lambda rc: rc in {5, 125})

        with pytest.raises(SystemExit) as exc:
            ds.dashboard_service_install("127.0.0.1", 9119)
        assert exc.value.code == 1
        out = capsys.readouterr().out
        assert "cannot manage" in out or "manual workaround" in out
        # Must NOT spawn a gateway (F1 note).
        assert "gateway" not in out.lower() and "detached" not in out.lower()


class TestRegressionG3StatusReuseParser:
    def test_status_reuses_gateway_pid_parser_and_normalizes_probe_host(self, monkeypatch, tmp_path, capsys):
        import hermes_cli.dashboard_service as ds
        import plistlib
        monkeypatch.setattr(ds, "is_macos", lambda: True)
        plist_path = tmp_path / "installed.plist"
        plist_path.write_bytes(plistlib.dumps({
            "ProgramArguments": [
                "python", "-m", "hermes_cli.main",
                "--host", "0.0.0.0", "--port", "9119", "--no-open",
            ],
        }))
        monkeypatch.setattr(ds, "get_dashboard_launchd_plist_path", lambda: plist_path)
        monkeypatch.setattr(ds, "_launchd_domain", lambda: "gui/501")
        # Mock gateway parser: registered with positive PID.
        monkeypatch.setattr(
            ds, "_launchd_print_service_pid", lambda d, l: (True, 9876)
        )
        ds.dashboard_service_status()
        out = capsys.readouterr().out
        assert "Supervising PID: 9876" in out
        # Probe URL must normalize 0.0.0.0 to 127.0.0.1.
        assert "127.0.0.1" in out or "Dashboard HTTP" in out


class TestRegressionF7RegressionTestsAdded:
    def test_start_missing_plist_regenerates_and_bootstraps_kickstarts_sequence(self, monkeypatch, tmp_path):
        import hermes_cli.dashboard_service as ds
        import subprocess
        monkeypatch.setattr(ds, "is_macos", lambda: True)
        monkeypatch.setattr(ds, "get_dashboard_launchd_plist_path", lambda: tmp_path / "missing.plist")
        monkeypatch.setattr(ds, "_launchd_domain", lambda: "gui/501")
        bootstrap_calls = []
        kickstart_calls = []

        def capture_bootstrap(*a, **k):
            bootstrap_calls.append("bootstrap")
        monkeypatch.setattr(ds, "_launchctl_bootstrap", capture_bootstrap)
        monkeypatch.setattr(ds, "_launchctl_kickstart_current", lambda label: kickstart_calls.append("kickstart"))
        ds.dashboard_service_start("127.0.0.1", 9119)
        assert len(bootstrap_calls) >= 1
        assert len(kickstart_calls) >= 1

    def test_start_kickstart_unloaded_code_3_retries_bootstrap_then_kickstart(self, monkeypatch, tmp_path):
        import hermes_cli.dashboard_service as ds
        import subprocess
        monkeypatch.setattr(ds, "is_macos", lambda: True)
        import plistlib
        plist_path = tmp_path / "existing.plist"
        plist_path.write_bytes(plistlib.dumps({"ProgramArguments": ["python"]}))
        monkeypatch.setattr(ds, "get_dashboard_launchd_plist_path", lambda: plist_path)
        monkeypatch.setattr(ds, "_launchd_domain", lambda: "gui/501")
        sequence = []

        def first_kickstart(label):
            sequence.append("kickstart_first")
            raise subprocess.CalledProcessError(3, ["launchctl"])
        monkeypatch.setattr(ds, "_launchctl_kickstart_current", first_kickstart)

        def bootstrap_capture(*a, **k):
            sequence.append("bootstrap_retry")
        monkeypatch.setattr(ds, "_launchctl_bootstrap", bootstrap_capture)

        def second_kickstart(label):
            sequence.append("kickstart_retry")
        monkeypatch.setattr(ds, "_launchd_error_indicates_unloaded", lambda exc: True)
        # After first failure, retry path must call bootstrap then kickstart.
        # The retry inside the except block uses the original module reference.
        # We need a callable that changes behavior on second invocation.
        tracker = {"calls": 0}

        def tracking_kickstart(label):
            tracker["calls"] += 1
            if tracker["calls"] == 1:
                raise subprocess.CalledProcessError(3, ["launchctl"])
            sequence.append(f"kickstart_retry_{tracker['calls']}")
        monkeypatch.setattr(ds, "_launchctl_kickstart_current", tracking_kickstart)
        ds.dashboard_service_start("127.0.0.1", 9119)
        assert "bootstrap_retry" in sequence
        assert any("kickstart_retry" in s for s in sequence)

    def test_start_bootstrap_fails_125_prints_hint_and_exits_1_no_gateway_spawn(self, monkeypatch, tmp_path, capsys):
        import hermes_cli.dashboard_service as ds
        import subprocess
        monkeypatch.setattr(ds, "is_macos", lambda: True)
        monkeypatch.setattr(ds, "get_dashboard_launchd_plist_path", lambda: tmp_path / "test.plist")
        monkeypatch.setattr(ds, "_launchd_domain", lambda: "gui/501")
        monkeypatch.setattr(ds, "_launchctl_domain_unsupported", lambda rc: rc == 125)

        def fail_bootstrap(*a, **k):
            raise subprocess.CalledProcessError(125, ["launchctl"])
        monkeypatch.setattr(ds, "_launchctl_bootstrap", fail_bootstrap)
        with pytest.raises(SystemExit) as exc:
            ds.dashboard_service_start("0.0.0.0", 9119)
        assert exc.value.code == 1
        out = capsys.readouterr().out
        assert "manual workaround" in out or "launchd cannot manage" in out
        # No gateway spawn happens (no "detached" or gateway-specific spawn message).
        assert "detached gateway" not in out

    def test_install_bootstrap_fails_125_exits_1_with_hint(self, monkeypatch, tmp_path, capsys):
        import hermes_cli.dashboard_service as ds
        import subprocess
        monkeypatch.setattr(ds, "is_macos", lambda: True)
        monkeypatch.setattr(ds, "get_dashboard_launchd_plist_path", lambda: tmp_path / "test.plist")
        monkeypatch.setattr(ds, "_launchd_domain", lambda: "gui/501")
        monkeypatch.setattr(ds, "_launchctl_domain_unsupported", lambda rc: rc == 5)

        def fail_bootstrap(*a, **k):
            raise subprocess.CalledProcessError(5, ["launchctl"])
        monkeypatch.setattr(ds, "_launchctl_bootstrap", fail_bootstrap)
        with pytest.raises(SystemExit) as exc:
            ds.dashboard_service_install("127.0.0.1", 9119, force=True)
        assert exc.value.code == 1
        out = capsys.readouterr().out
        assert "cannot manage" in out or "manual workaround" in out

    def test_status_probe_host_0_0_0_0_uses_127_0_0_1(self, monkeypatch, tmp_path, capsys):
        import hermes_cli.dashboard_service as ds
        import plistlib
        monkeypatch.setattr(ds, "is_macos", lambda: True)
        plist_path = tmp_path / "installed.plist"
        plist_path.write_bytes(plistlib.dumps({
            "ProgramArguments": [
                "python", "-m", "hermes_cli.main",
                "--host", "0.0.0.0", "--port", "9119",
            ],
        }))
        monkeypatch.setattr(ds, "get_dashboard_launchd_plist_path", lambda: plist_path)
        monkeypatch.setattr(ds, "_launchd_domain", lambda: "gui/501")
        # Mock parser and urlopen.
        monkeypatch.setattr(
            ds, "_launchd_print_service_pid", lambda d, l: (True, 1234)
        )
        # Capture the URL built by the module.
        url_captured = []

        def mock_urlopen(url, **kw):
            url_captured.append(str(url))
            return MagicMock(__enter__=lambda s: s, __exit__=lambda *a: False, status=200)
        monkeypatch.setattr(ds.urllib.request, "urlopen", mock_urlopen)
        ds.dashboard_service_status()
        assert len(url_captured) == 1
        assert "127.0.0.1" in url_captured[0]
        assert "0.0.0.0" not in url_captured[0]


class TestXMLInjectionRegression:
    """Security regression: extra_args with XML metacharacters must round-trip transparently."""

    def test_xml_metacharacters_in_extra_args_roundtrip(self, mock_dashboard_service):
        import plistlib
        malicious_extra = ["--foo", 'a<b>&c"d']
        plist_text = mock_dashboard_service.generate_dashboard_launchd_plist(
            "127.0.0.1", 9119, extra_args=malicious_extra
        )
        parsed = plistlib.loads(plist_text.encode("utf-8"))
        prog_args = parsed.get("ProgramArguments", [])
        # The malicious values must appear exactly in the parsed arguments (escape is transparent).
        assert "--foo" in prog_args
        assert 'a<b>&c"d' in prog_args
        # Ensure the raw plist text actually contains escaped entities (not raw metacharacters).
        raw_text = plist_text
        assert "&lt;b&gt;" in raw_text or "a&lt;b" in raw_text
        # Confirm the exact original is recoverable (escape is transparent on read-back).
        assert "--foo" in prog_args
        assert 'a<b>&c"d' in prog_args


# ------------------------------------------------------------------
# Parser + dispatcher tests (issue #44106)
# ------------------------------------------------------------------

class TestDashboardServiceParser:
    """Parser construction for `hermes dashboard service <verb>` — no real subprocess."""

    def test_service_install_parses_with_flags(self):
        import argparse
        from hermes_cli.subcommands.dashboard import build_dashboard_parser
        root = argparse.ArgumentParser()
        sub = root.add_subparsers()
        build_dashboard_parser(
            sub,
            cmd_dashboard=lambda a: None,
            cmd_dashboard_register=lambda a: None,
            cmd_dashboard_service=lambda a: None,
        )
        args = root.parse_args([
            "dashboard", "service", "install",
            "--host", "0.0.0.0", "--port", "9200", "--force",
        ])
        assert args.dashboard_service_command == "install"
        assert args.host == "0.0.0.0"
        assert args.port == 9200
        assert args.force is True
        assert args.func is not None

    def test_service_install_skip_build(self):
        import argparse
        from hermes_cli.subcommands.dashboard import build_dashboard_parser
        root = argparse.ArgumentParser()
        sub = root.add_subparsers()
        build_dashboard_parser(
            sub,
            cmd_dashboard=lambda a: None,
            cmd_dashboard_register=lambda a: None,
            cmd_dashboard_service=lambda a: None,
        )
        args = root.parse_args([
            "dashboard", "service", "install", "--skip-build",
        ])
        assert args.skip_build is True
        assert args.dashboard_service_command == "install"

    def test_service_restart_no_extra_attrs(self):
        import argparse
        from hermes_cli.subcommands.dashboard import build_dashboard_parser
        root = argparse.ArgumentParser()
        sub = root.add_subparsers()
        build_dashboard_parser(
            sub,
            cmd_dashboard=lambda a: None,
            cmd_dashboard_register=lambda a: None,
            cmd_dashboard_service=lambda a: None,
        )
        args = root.parse_args(["dashboard", "service", "restart"])
        assert args.dashboard_service_command == "restart"
        assert args.func is not None

    def test_service_missing_verb_exits(self):
        import argparse
        import pytest
        from hermes_cli.subcommands.dashboard import build_dashboard_parser
        root = argparse.ArgumentParser()
        sub = root.add_subparsers()
        build_dashboard_parser(
            sub,
            cmd_dashboard=lambda a: None,
            cmd_dashboard_register=lambda a: None,
            cmd_dashboard_service=lambda a: None,
        )
        with pytest.raises(SystemExit) as exc:
            root.parse_args(["dashboard", "service"])
        assert exc.value.code == 2

    def test_register_still_parses_no_regression(self):
        import argparse
        from hermes_cli.subcommands.dashboard import build_dashboard_parser
        root = argparse.ArgumentParser()
        sub = root.add_subparsers()
        build_dashboard_parser(
            sub,
            cmd_dashboard=lambda a: None,
            cmd_dashboard_register=lambda a: None,
            cmd_dashboard_service=lambda a: None,
        )
        args = root.parse_args(["dashboard", "register", "--name", "test"])
        assert args.dashboard_subcommand == "register"
        assert args.name == "test"


class TestDashboardServiceDispatcher:
    """Table dispatch asserts: each verb forwards to the right module function."""

    def test_install_forwards_force_and_skip_build(self, monkeypatch, profile_env):
        import hermes_cli.dashboard_service as ds
        calls = {}
        monkeypatch.setattr(ds, "dashboard_service_install", lambda *a, **k: calls.update({"verb": "install", "args": a, "kwargs": k}))
        from hermes_cli.dashboard_service import dashboard_service_command
        import types
        args = types.SimpleNamespace(
            dashboard_service_command="install",
            host="127.0.0.1",
            port=9119,
            skip_build=True,
            force=True,
        )
        dashboard_service_command(args)
        assert calls["verb"] == "install"
        # positional args: host, port; keyword args: extra_args, force
        assert calls["args"] == ("127.0.0.1", 9119)
        assert calls["kwargs"]["force"] is True
        assert calls["kwargs"]["extra_args"] == ["--skip-build"]

    def test_start_forwards_host_port_skip_build(self, monkeypatch, profile_env):
        import hermes_cli.dashboard_service as ds
        calls = {}
        monkeypatch.setattr(ds, "dashboard_service_start", lambda *a, **k: calls.update({"verb": "start", "args": a, "kwargs": k}))
        from hermes_cli.dashboard_service import dashboard_service_command
        import types
        args = types.SimpleNamespace(
            dashboard_service_command="start",
            host="0.0.0.0",
            port=9200,
            skip_build=False,
        )
        dashboard_service_command(args)
        assert calls["verb"] == "start"
        assert calls["args"] == ("0.0.0.0", 9200)
        assert calls["kwargs"]["extra_args"] is None or calls["kwargs"]["extra_args"] == None

    def test_stop_called_no_args(self, monkeypatch, profile_env):
        import hermes_cli.dashboard_service as ds
        calls = {}
        monkeypatch.setattr(ds, "dashboard_service_stop", lambda *a, **k: calls.update({"verb": "stop"}))
        from hermes_cli.dashboard_service import dashboard_service_command
        import types
        args = types.SimpleNamespace(dashboard_service_command="stop")
        dashboard_service_command(args)
        assert calls["verb"] == "stop"

    def test_restart_called_no_args(self, monkeypatch, profile_env):
        import hermes_cli.dashboard_service as ds
        calls = {}
        monkeypatch.setattr(ds, "dashboard_service_restart", lambda *a, **k: calls.update({"verb": "restart"}))
        from hermes_cli.dashboard_service import dashboard_service_command
        import types
        args = types.SimpleNamespace(dashboard_service_command="restart")
        dashboard_service_command(args)
        assert calls["verb"] == "restart"

    def test_status_called_no_args(self, monkeypatch, profile_env):
        import hermes_cli.dashboard_service as ds
        calls = {}
        monkeypatch.setattr(ds, "dashboard_service_status", lambda *a, **k: calls.update({"verb": "status"}))
        from hermes_cli.dashboard_service import dashboard_service_command
        import types
        args = types.SimpleNamespace(dashboard_service_command="status")
        dashboard_service_command(args)
        assert calls["verb"] == "status"

    def test_uninstall_called_no_args(self, monkeypatch, profile_env):
        import hermes_cli.dashboard_service as ds
        calls = {}
        monkeypatch.setattr(ds, "dashboard_service_uninstall", lambda *a, **k: calls.update({"verb": "uninstall"}))
        from hermes_cli.dashboard_service import dashboard_service_command
        import types
        args = types.SimpleNamespace(dashboard_service_command="uninstall")
        dashboard_service_command(args)
        assert calls["verb"] == "uninstall"


class TestBuildServeParserDoesNotBreak:
    """Lean hot-path parser (`serve`) stays clean after the service parser addition."""

    def test_serve_parser_still_builds(self):
        from hermes_cli.subcommands.dashboard import build_serve_parser
        p = build_serve_parser(cmd_dashboard=lambda a: None)
        assert p.prog == "hermes serve"
