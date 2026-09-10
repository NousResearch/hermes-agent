"""Tests for the unified profile→machine dashboard launch routing.

`<profile> dashboard` routes to ONE machine-level dashboard instead of
spawning a per-profile server: attach (open browser at ?profile=) when one
is already listening, else re-exec as the machine dashboard with the
launching profile preselected. `--isolated` opts out.
"""
import sys
import types
import pytest
from hermes_cli import main_dashboard


@pytest.fixture
def main_mod():
    import hermes_cli.main as main_mod
    return main_mod


@pytest.fixture(autouse=True)
def _block_real_dashboard_spawn(monkeypatch):
    """Never let these tests spawn a real dashboard process.

    cmd_dashboard's profile reroute re-execs via ``os.execvpe`` on POSIX but
    via ``subprocess.Popen`` on Windows (execvpe does not truly replace the
    process there — see main.py). Tests below monkeypatch only ``execvpe``,
    so on Windows the reroute escaped the mock and launched a REAL machine
    dashboard pinned to the REAL machine root. That child's bootstrap then
    quarantined the live venv's hermes.exe/hermes-gateway.exe entry-point
    shims, deleting the developer's working ``hermes`` command — observed
    on a full-suite run on Windows (2026-07-02).
    """
    import hermes_cli.main as main_mod

    def _no_real_spawn(*args, **kwargs):
        raise AssertionError(
            "test attempted to spawn a real process via subprocess.Popen — "
            "the Windows reexec branch must be mocked, not executed"
        )

    monkeypatch.setattr(main_mod.subprocess, "Popen", _no_real_spawn)


def _args(**kw):
    defaults = dict(
        status=False, stop=False, host="127.0.0.1", port=9119,
        no_open=True, insecure=False, skip_build=False,
        isolated=False, open_profile="",
    )
    defaults.update(kw)
    return types.SimpleNamespace(**defaults)


class TestUnifiedDashboardRouting:


    def test_profile_launch_reexecs_machine_dashboard(self, main_mod, monkeypatch):
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.setattr(
            "hermes_cli.profiles.get_active_profile_name", lambda: "worker_x"
        )
        monkeypatch.setattr(main_dashboard, "_dashboard_listening", lambda host, port: False)
        execs = []

        def fake_exec(exe, argv, env):
            execs.append((exe, argv, env))
            raise SystemExit(0)  # execvpe never returns

        monkeypatch.setattr(main_mod.os, "execvpe", fake_exec)

        # Windows takes the subprocess.Popen reexec branch instead of
        # execvpe — record it into the same list so the assertions below
        # hold on every platform (and no real process is ever spawned).
        def fake_popen(argv, env=None, **kwargs):
            execs.append((argv[0], argv, env))
            return types.SimpleNamespace(wait=lambda: 0)

        monkeypatch.setattr(main_mod.subprocess, "Popen", fake_popen)

        with pytest.raises(SystemExit):
            main_mod.cmd_dashboard(_args())

        assert len(execs) == 1
        exe, argv, env = execs[0]
        assert exe == sys.executable
        # Pinned to the default profile + launching profile preselected.
        assert "-p" in argv and argv[argv.index("-p") + 1] == "default"
        assert "--open-profile" in argv
        assert argv[argv.index("--open-profile") + 1] == "worker_x"
        # The child is pinned to the machine ROOT, not the launching profile's
        # HERMES_HOME.  For a standard install (HERMES_HOME unset) that root is
        # the platform-native default (~/.hermes), NOT dropped — see the Docker
        # test below for why we resolve explicitly instead of popping.
        from hermes_constants import get_default_hermes_root
        assert env.get("HERMES_HOME") == str(get_default_hermes_root())

    def test_reexec_pins_docker_machine_root(self, main_mod, monkeypatch):
        """In the Docker layout (HERMES_HOME=/opt/data, profiles under
        /opt/data/profiles/<name>) the reroute must pin the child to the
        machine root /opt/data — NOT drop HERMES_HOME.

        Dropping it makes the child fall back to $HOME/.hermes
        (= /opt/data/.hermes), an empty auto-seeded home, so the dashboard
        shows only the default profile and the .install_method stamp is
        missing (which also misfires the Docker update-button guard).
        Regression test for the support report.
        """
        monkeypatch.setenv("HERMES_HOME", "/opt/data/profiles/oracle")
        monkeypatch.setattr(
            "hermes_cli.profiles.get_active_profile_name", lambda: "oracle"
        )
        monkeypatch.setattr(main_mod, "_dashboard_listening", lambda host, port: False)
        execs = []

        def fake_exec(exe, argv, env):
            execs.append((exe, argv, env))
            raise SystemExit(0)

        monkeypatch.setattr(main_mod.os, "execvpe", fake_exec)

        def fake_popen(argv, env=None, **kwargs):
            execs.append((argv[0], argv, env))
            return types.SimpleNamespace(wait=lambda: 0)

        monkeypatch.setattr(main_mod.subprocess, "Popen", fake_popen)

        with pytest.raises(SystemExit):
            main_mod.cmd_dashboard(_args())

        assert len(execs) == 1
        _exe, _argv, env = execs[0]
        # get_default_hermes_root() strips the trailing profiles/<name>, so the
        # child binds /opt/data — where the real default/oracle/saga profiles
        # and the .install_method stamp actually live. Compare via Path so the
        # separator rendering matches on Windows dev machines too (the Docker
        # layout itself is POSIX-only, but the stripping semantics are not).
        from pathlib import Path
        assert env.get("HERMES_HOME") == str(Path("/opt/data"))

    def test_desktop_profile_backend_skips_machine_dashboard_reroute(self, main_mod, monkeypatch):
        """A desktop-spawned named-profile backend (HERMES_DESKTOP=1) must NOT
        reroute into the machine dashboard. The reroute re-execs as the default
        profile and exits, so the desktop never sees a ready backend → boot
        loop. The guard keeps desktop pool backends per-profile."""
        monkeypatch.setenv("HERMES_DESKTOP", "1")
        monkeypatch.setattr(
            "hermes_cli.profiles.get_active_profile_name", lambda: "worker_x"
        )
        listening_calls = []
        monkeypatch.setattr(main_dashboard, "_dashboard_listening",
            lambda host, port: listening_calls.append(1) or False,
        )
        execs = []
        monkeypatch.setattr(main_mod.os, "execvpe", lambda *a, **k: execs.append(a))
        monkeypatch.setitem(sys.modules, "fastapi", None)

        with pytest.raises((SystemExit, AttributeError, ImportError, TypeError)):
            main_mod.cmd_dashboard(_args())
        assert listening_calls == []
        assert execs == []


class TestInteractiveDashboardAuthSetup:

    def test_loopback_proxy_public_url_offers_auth_setup(
        self, main_mod, monkeypatch, capsys
    ):
        """A TTY operator is prompted when public_url gates a loopback bind."""
        from hermes_cli.dashboard_auth import clear_providers

        monkeypatch.setenv(
            "HERMES_DASHBOARD_PUBLIC_URL",
            "https://dashboard.example.test:9443",
        )
        clear_providers()
        monkeypatch.setattr(main_mod.sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr(main_mod.sys.stdout, "isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda _prompt: "3")

        with pytest.raises(SystemExit) as exc:
            main_mod._maybe_setup_dashboard_auth_interactively(_args())

        assert exc.value.code == 1
        output = capsys.readouterr().out
        assert "configured external dashboard.public_url" in output




