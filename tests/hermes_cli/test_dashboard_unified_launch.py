"""Tests for the unified profile→machine dashboard launch routing.

`<profile> dashboard` routes to ONE machine-level dashboard instead of
spawning a per-profile server: attach (open browser at ?profile=) when one
is already listening, else re-exec as the machine dashboard with the
launching profile preselected. `--isolated` opts out.
"""
import sys
import types
import pytest


@pytest.fixture
def main_mod():
    import hermes_cli.main as main_mod
    return main_mod


def _args(**kw):
    defaults = dict(
        status=False, stop=False, host="127.0.0.1", port=9119,
        no_open=True, insecure=False, skip_build=False,
        isolated=False, open_profile="",
    )
    defaults.update(kw)
    return types.SimpleNamespace(**defaults)


class TestUnifiedDashboardRouting:
    def test_profile_launch_attaches_to_running_dashboard(self, main_mod, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.profiles.get_active_profile_name", lambda: "worker_x"
        )
        monkeypatch.setattr(main_mod, "_dashboard_listening", lambda host, port: True)
        execs = []
        monkeypatch.setattr(main_mod.os, "execvpe", lambda *a, **k: execs.append(a))

        with pytest.raises(SystemExit) as exc:
            main_mod.cmd_dashboard(_args())
        assert exc.value.code == 0
        assert execs == []  # attached, never re-exec'd

    def test_profile_launch_attach_opens_scoped_url(self, main_mod, monkeypatch):
        """The attach path must open the browser at ?profile=<name> — that
        URL is the entire point of attaching (preselects the switcher)."""
        monkeypatch.setattr(
            "hermes_cli.profiles.get_active_profile_name", lambda: "worker_x"
        )
        monkeypatch.setattr(main_mod, "_dashboard_listening", lambda host, port: True)
        opened = []
        import webbrowser
        monkeypatch.setattr(webbrowser, "open", lambda url: opened.append(url))

        with pytest.raises(SystemExit) as exc:
            main_mod.cmd_dashboard(_args(no_open=False))
        assert exc.value.code == 0
        assert opened == ["http://127.0.0.1:9119/?profile=worker_x"]

    def test_profile_launch_reexecs_machine_dashboard(self, main_mod, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.profiles.get_active_profile_name", lambda: "worker_x"
        )
        monkeypatch.setattr(main_mod, "_dashboard_listening", lambda host, port: False)
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
        # Profile HERMES_HOME dropped so the child binds the machine root.
        assert "HERMES_HOME" not in env

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
        monkeypatch.setattr(
            main_mod, "_dashboard_listening",
            lambda host, port: listening_calls.append(1) or False,
        )
        execs = []
        monkeypatch.setattr(main_mod.os, "execvpe", lambda *a, **k: execs.append(a))
        monkeypatch.setitem(sys.modules, "fastapi", None)

        with pytest.raises((SystemExit, AttributeError, ImportError, TypeError)):
            main_mod.cmd_dashboard(_args())
        assert listening_calls == []
        assert execs == []

    def test_isolated_flag_skips_routing(self, main_mod, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.profiles.get_active_profile_name", lambda: "worker_x"
        )
        listening_calls = []
        monkeypatch.setattr(
            main_mod, "_dashboard_listening",
            lambda host, port: listening_calls.append(1) or True,
        )
        # With --isolated the routing block is skipped entirely; the command
        # proceeds to dependency checks. Make the first post-routing step
        # bail so the test doesn't actually start a server.
        monkeypatch.setitem(sys.modules, "fastapi", None)

        with pytest.raises((SystemExit, AttributeError, ImportError, TypeError)):
            main_mod.cmd_dashboard(_args(isolated=True))
        assert listening_calls == []

    def test_default_profile_launch_skips_routing(self, main_mod, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.profiles.get_active_profile_name", lambda: "default"
        )
        listening_calls = []
        monkeypatch.setattr(
            main_mod, "_dashboard_listening",
            lambda host, port: listening_calls.append(1) or True,
        )
        monkeypatch.setitem(sys.modules, "fastapi", None)

        with pytest.raises((SystemExit, AttributeError, ImportError, TypeError)):
            main_mod.cmd_dashboard(_args())
        assert listening_calls == []

    def test_reexec_child_does_not_reroute(self, main_mod, monkeypatch):
        """The re-exec'd child carries --open-profile; the guard must treat
        that as 'already routed' and never re-exec again (no exec loop)."""
        monkeypatch.setattr(
            "hermes_cli.profiles.get_active_profile_name", lambda: "worker_x"
        )
        execs = []
        monkeypatch.setattr(main_mod.os, "execvpe", lambda *a, **k: execs.append(a))
        monkeypatch.setitem(sys.modules, "fastapi", None)

        with pytest.raises((SystemExit, AttributeError, ImportError, TypeError)):
            main_mod.cmd_dashboard(_args(open_profile="worker_x"))
        assert execs == []

    def test_dashboard_starts_mcp_discovery_for_ws_backend(self, main_mod, monkeypatch):
        """The dashboard process serves the /api/ws gateway but never runs
        tui_gateway/entry.py, so it must kick off MCP discovery itself or
        desktop sessions never see a profile's MCP tools."""
        monkeypatch.setattr(
            "hermes_cli.profiles.get_active_profile_name", lambda: "default"
        )
        monkeypatch.delenv("HERMES_WEB_DIST", raising=False)
        monkeypatch.setattr(main_mod, "_sync_bundled_skills_quietly", lambda: None)
        monkeypatch.setattr(main_mod, "_build_web_ui", lambda *_a, **_k: True)
        monkeypatch.setitem(sys.modules, "fastapi", types.SimpleNamespace())
        monkeypatch.setitem(sys.modules, "uvicorn", types.SimpleNamespace())
        monkeypatch.setitem(
            sys.modules,
            "hermes_logging",
            types.SimpleNamespace(setup_logging=lambda **_k: None),
        )
        monkeypatch.setitem(
            sys.modules,
            "hermes_cli.plugins",
            types.SimpleNamespace(discover_plugins=lambda: None),
        )
        calls = []
        monkeypatch.setattr(
            "hermes_cli.mcp_startup.start_background_mcp_discovery",
            lambda **kwargs: calls.append(kwargs),
        )
        monkeypatch.setitem(
            sys.modules,
            "hermes_cli.web_server",
            types.SimpleNamespace(start_server=lambda **_kwargs: None),
        )

        main_mod.cmd_dashboard(_args())

        assert calls == [
            {
                "logger": main_mod.logger,
                "thread_name": "dashboard-mcp-discovery",
            }
        ]
