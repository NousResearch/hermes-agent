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


def _args(**kw):
    defaults = dict(
        status=False, stop=False, host="127.0.0.1", port=9119,
        no_open=True, insecure=False, skip_build=False,
        isolated=False, open_profile="", ui_surface="dashboard",
    )
    defaults.update(kw)
    return types.SimpleNamespace(**defaults)


def _host_owner(monkeypatch, surface):
    from gateway import host_rendezvous as hr

    record = types.SimpleNamespace(pid=123, host="127.0.0.1", port=9119, role="serve", profiles=())
    monkeypatch.setattr(main_dashboard, "_host_backend_attachment", lambda: record)
    monkeypatch.setattr(hr, "probe_owner", lambda _: {"servesSpa": True, "ui_surface": surface})
    monkeypatch.setattr(main_dashboard, "_explicit_endpoint_flags", lambda: set())


class TestUnifiedDashboardRouting:

    def test_profile_launch_reexecs_machine_dashboard(self, main_mod, monkeypatch):
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.setattr(
            "hermes_cli.profiles.get_active_profile_name", lambda: "worker_x"
        )
        monkeypatch.setattr(main_dashboard, "_dashboard_listening", lambda host, port: False)
        monkeypatch.setattr(main_dashboard, "_host_backend_attachment", lambda: None)
        execs = []

        if sys.platform == "win32":
            # Windows cannot truly replace the process, so cmd_dashboard
            # re-execs via subprocess.Popen + sys.exit(code) instead of
            # os.execvpe (which doesn't exist on Windows).
            spawns = []

            class _Done:
                def wait(self):
                    return 0

            monkeypatch.setattr(
                main_mod.subprocess,
                "Popen",
                lambda argv, env=None, **kw: spawns.append((argv, env)) or _Done(),
            )
            with pytest.raises(SystemExit):
                main_mod.cmd_dashboard(_args())
            assert len(spawns) == 1
            argv, env = spawns[0]
        else:
            execs = []

            def fake_exec(exe, argv, env):
                execs.append((exe, argv, env))
                raise SystemExit(0)  # execvpe never returns

            monkeypatch.setattr(main_mod.os, "execvpe", fake_exec)

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

    def test_named_webapp_refuses_to_attach_to_dashboard_surface(self, main_mod, monkeypatch):
        monkeypatch.delenv("HERMES_DESKTOP", raising=False)
        monkeypatch.delenv("HERMES_WEB_DIST", raising=False)
        monkeypatch.setattr(
            "hermes_cli.profiles.get_active_profile_name", lambda: "worker_x"
        )
        _host_owner(monkeypatch, "dashboard")
        opened = []
        monkeypatch.setitem(
            sys.modules,
            "webbrowser",
            types.SimpleNamespace(open=lambda url: opened.append(url)),
        )

        with pytest.raises(SystemExit) as exc:
            main_mod.cmd_dashboard(
                _args(no_open=False, skip_build=True, ui_surface="webapp")
            )

        assert exc.value.code == 1
        assert opened == []

    def test_named_webapp_prints_route_without_opening_unauthorized_tab(self, main_mod, monkeypatch, capsys):
        monkeypatch.delenv("HERMES_DESKTOP", raising=False)
        monkeypatch.delenv("HERMES_WEB_DIST", raising=False)
        monkeypatch.setattr(
            "hermes_cli.profiles.get_active_profile_name", lambda: "worker_x"
        )
        _host_owner(monkeypatch, "webapp")
        opened = []
        monkeypatch.setitem(
            sys.modules,
            "webbrowser",
            types.SimpleNamespace(open=lambda url: opened.append(url)),
        )

        with pytest.raises(SystemExit) as exc:
            main_mod.cmd_dashboard(
                _args(no_open=False, skip_build=True, ui_surface="webapp")
            )

        assert exc.value.code == 0
        assert opened == []
        output = capsys.readouterr().out
        assert "http://127.0.0.1:9119/?profile=worker_x" in output
        assert "private launch link" in output


    def test_desktop_profile_backend_skips_machine_dashboard_reroute(self, main_mod, monkeypatch):
        """A desktop-spawned named-profile backend (HERMES_DESKTOP=1) must NOT
        reroute into the machine dashboard. The reroute re-execs as the default
        profile and exits, so the desktop never sees a ready backend → boot
        loop. The guard keeps desktop pool backends per-profile."""
        monkeypatch.setenv("HERMES_DESKTOP", "1")
        monkeypatch.setenv("HERMES_DASHBOARD_SESSION_TOKEN", "desktop-spawn-token")
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


