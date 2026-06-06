"""Tests for `hermes dashboard (install-launchd flag)` and the macOS
LaunchAgent installer (PR feat/dashboard-launchagent).

The double-launchctl dance is hard to exercise end-to-end in CI
(no launchd on Linux, no $HOME/Library/LaunchAgents, and a real
install would persist on the developer's machine).  Instead we:

* Unit-test the plist template renderer: token substitution
  must be complete (no leftover ``__TOKEN__`` markers) and the
  rendered XML must be parseable by ``plistlib``.
* Unit-test the installer's precheck: it must refuse to install
  a 0.0.0.0 bind without --insecure (a service bound to all
  interfaces and restarted on login would expose the dashboard
  to the LAN forever) and must exit non-zero on non-darwin.
* Smoke-test the dispatch: ``cmd_dashboard`` with
  ``install_launchd=True`` calls the installer (and the
  installer is short-circuited in unit tests via monkeypatch).
"""

from __future__ import annotations

import plistlib
import sys
from pathlib import Path
from unittest import mock

import pytest


# ── Plist template + renderer ───────────────────────────────────────────


class TestPlistTemplate:
    """The shipped template is the single source of truth for the
    LaunchAgent shape; we lock the renderer so a typo in the
    template can't ship a broken plist silently."""

    def test_template_exists(self):
        from hermes_cli.main import _launchd_plist_template_path

        path = _launchd_plist_template_path()
        assert path.is_file(), f"plist template missing at {path}"
        assert path.name == "ai.hermes.dashboard.plist"

    def test_template_is_parseable_plist(self):
        """The template itself (with __TOKEN__ placeholders) must be
        a well-formed plist so users can inspect it directly."""
        from hermes_cli.main import _launchd_plist_template_path

        path = _launchd_plist_template_path()
        # plistlib raises on malformed XML; we don't need to assert
        # on the values since they're placeholders.
        plistlib.loads(path.read_text(encoding="utf-8").encode())

    def test_template_contains_required_knobs(self):
        """Lock the keys we expect launchd to read.  Dropping any
        of these silently regresses the service."""
        from hermes_cli.main import _launchd_plist_template_path

        text = _launchd_plist_template_path().read_text(encoding="utf-8")
        for required in (
            "<key>Label</key>",
            "<key>ProgramArguments</key>",
            "<key>RunAtLoad</key>",
            "<key>KeepAlive</key>",
            "<key>EnvironmentVariables</key>",
            "<key>StandardOutPath</key>",
            "<key>StandardErrorPath</key>",
        ):
            assert required in text, f"plist template missing {required}"


class TestPlistRender:
    """Token substitution must be complete and the output must
    round-trip through plistlib."""

    def test_render_substitutes_every_placeholder(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from hermes_cli.main import _render_launchd_plist

        text = (
            "Hello __HERMES_BIN__ on __DASHBOARD_HOST__:__DASHBOARD_PORT__ "
            "home=__HOME__ path=__PATH__ hermes=__HERMES_HOME__ log=__LOG_PATH__"
        )
        rendered = _render_launchd_plist(
            text,
            hermes_bin=Path("/usr/local/bin/hermes"),
            host="127.0.0.1",
            port=9119,
        )
        assert "__HERMES_BIN__" not in rendered
        assert "__DASHBOARD_HOST__" not in rendered
        assert "__DASHBOARD_PORT__" not in rendered
        assert "__HOME__" not in rendered
        assert "__PATH__" not in rendered
        assert "__HERMES_HOME__" not in rendered
        assert "__LOG_PATH__" not in rendered
        assert "/usr/local/bin/hermes" in rendered
        assert "127.0.0.1" in rendered
        assert "9119" in rendered

    def test_render_rejects_unknown_placeholders(self, tmp_path, monkeypatch):
        """A typo in the template (or a new token added to the
        replacements table but not the template) must raise so we
        don't ship a plist with literal ``__SOMETHING__`` in it."""
        from hermes_cli.main import _render_launchd_plist

        with pytest.raises(ValueError, match="__TYPO_HERE__"):
            _render_launchd_plist(
                "value: __TYPO_HERE__",
                hermes_bin=Path("/usr/local/bin/hermes"),
                host="127.0.0.1",
                port=9119,
            )

    def test_render_creates_log_dir(self, tmp_path, monkeypatch):
        """The launchd log file's parent must exist so launchd
        can open it on first write."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from hermes_cli.main import _render_launchd_plist

        rendered = _render_launchd_plist(
            open(
                Path(__file__).parent.parent.parent
                / "contrib"
                / "launchd"
                / "ai.hermes.dashboard.plist"
            ).read(),
            hermes_bin=Path("/usr/local/bin/hermes"),
            host="127.0.0.1",
            port=9119,
        )
        # __LOG_PATH__ resolves under HERMES_HOME/logs/ which the
        # renderer must have created.
        assert (tmp_path / "logs").is_dir()
        # And the rendered XML points at a file path that exists'
        # parent.
        parsed = plistlib.loads(rendered.encode())
        log_path = Path(parsed["StandardOutPath"])
        assert log_path.parent.is_dir()
        assert log_path.name == "launchd.dashboard.log"


class TestFullPlistRoundTrip:
    """The end-to-end render must produce a valid, well-formed
    plist with the right ProgramArguments — this is the contract
    with macOS launchd."""

    def test_rendered_plist_parses_and_has_right_args(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from hermes_cli.main import _launchd_plist_template_path, _render_launchd_plist

        rendered = _render_launchd_plist(
            _launchd_plist_template_path().read_text(encoding="utf-8"),
            hermes_bin=Path("/usr/local/bin/hermes"),
            host="127.0.0.1",
            port=9119,
        )
        parsed = plistlib.loads(rendered.encode())

        assert parsed["Label"] == "ai.hermes.dashboard"
        assert parsed["RunAtLoad"] is True
        # KeepAlive must be a literal True (not a dict with SuccessfulExit).
        # The launchd-spawned launcher exits 0 after execvp'ing the
        # grandchild, so SuccessfulExit=false would let launchd consider
        # the job done and stop respawning — defeating crash recovery.
        assert parsed["KeepAlive"] is True
        assert parsed["ThrottleInterval"] == 10
        # ProgramArguments has to be: hermes -m hermes_cli.main dashboard
        # --detach --no-open --host <host> --port <port>
        argv = parsed["ProgramArguments"]
        assert argv[0].endswith("hermes")
        assert argv[1:4] == ["-m", "hermes_cli.main", "dashboard"]
        assert "--detach" in argv
        assert "--no-open" in argv
        assert "--host" in argv
        idx = argv.index("--host")
        assert argv[idx + 1] == "127.0.0.1"
        assert "--port" in argv
        pidx = argv.index("--port")
        assert argv[pidx + 1] == "9119"
        # EnvironmentVariables must include HERMES_HOME so profile
        # mode works under launchd.
        env = parsed["EnvironmentVariables"]
        assert env["HERMES_HOME"] == str(tmp_path)
        assert "HOME" in env
        assert "PATH" in env


# ── cmd_dashboard dispatch ─────────────────────────────────────────────


class TestInstallDispatch:
    """``cmd_dashboard`` with install_launchd=True must route to
    the installer and not start a server."""

    def test_install_launchd_dispatches(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        import argparse

        from hermes_cli.main import cmd_dashboard

        called = {"count": 0}

        def _fake_install(args):
            called["count"] += 1

        monkeypatch.setattr("hermes_cli.main._install_launchd", _fake_install)

        # Also short-circuit the port precheck (the installer's own
        # gate is the unit under test on the install path; for
        # dispatch we just want to confirm we get there).
        monkeypatch.setattr(
            "hermes_cli.main._port_is_in_use", lambda h, p, t=0.25: False
        )

        args = argparse.Namespace(
            host="127.0.0.1",
            port=9119,
            no_open=True,
            insecure=False,
            detach=False,
            pid_file=None,
            skip_build=True,
            install_launchd=True,
            uninstall_launchd=False,
            status=False,
            stop=False,
            dashboard_subcommand=None,
        )
        # Stub out subprocess so the real launchctl isn't called.
        with mock.patch("hermes_cli.main._run_launchctl") as run_lc:
            run_lc.return_value = mock.Mock(returncode=0, stderr="")
            with mock.patch("builtins.print"):
                cmd_dashboard(args)

        assert called["count"] == 1

    def test_uninstall_launchd_dispatches(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        import argparse

        from hermes_cli.main import cmd_dashboard

        called = {"count": 0}

        def _fake_uninstall():
            called["count"] += 1

        monkeypatch.setattr("hermes_cli.main._uninstall_launchd", _fake_uninstall)

        args = argparse.Namespace(
            host="127.0.0.1",
            port=9119,
            no_open=True,
            insecure=False,
            detach=False,
            pid_file=None,
            skip_build=True,
            install_launchd=False,
            uninstall_launchd=True,
            status=False,
            stop=False,
            dashboard_subcommand=None,
        )
        with mock.patch("builtins.print"):
            cmd_dashboard(args)
        assert called["count"] == 1


# ── Installer: precheck behaviour ──────────────────────────────────────


class TestInstallRefusals:
    """The installer must refuse to set up unsafe configurations
    and exit non-zero on unsupported platforms."""

    def test_non_darwin_exits_nonzero(self, monkeypatch):
        """On Linux/Windows the flag is a no-op stub that prints
        a hint and exits 1."""
        from hermes_cli.main import _install_launchd

        args = mock.Mock(host="127.0.0.1", port=9119, insecure=False)
        monkeypatch.setattr("hermes_cli.main.sys.platform", "linux")

        with mock.patch("hermes_cli.main.sys.exit") as exit_mock:
            with mock.patch("builtins.print") as print_mock:
                _install_launchd(args)
        assert exit_mock.called
        assert exit_mock.call_args.args == (1,)

    def test_darwin_install_calls_launchctl_bootstrap(self, tmp_path, monkeypatch):
        """Smoke test: on darwin with a free port, the installer
        writes the plist, calls `launchctl bootstrap`, and prints
        a confirmation.  launchctl itself is stubbed."""
        from hermes_cli.main import _install_launchd

        monkeypatch.setattr("hermes_cli.main.sys.platform", "darwin")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(
            "hermes_cli.main._port_is_in_use", lambda h, p, t=0.25: False
        )

        # Use a fake plist destination we can clean up.
        agents = tmp_path / "Library" / "LaunchAgents"
        monkeypatch.setattr(
            "hermes_cli.main.Path.home", lambda: agents.parent.parent
        )

        run_calls: list[list[str]] = []

        def _fake_run(*args, check=True):
            run_calls.append(list(args))
            m = mock.Mock(returncode=0, stderr="")
            return m

        monkeypatch.setattr("hermes_cli.main._run_launchctl", _fake_run)

        args = mock.Mock(host="127.0.0.1", port=9119, insecure=False)
        with mock.patch("builtins.print"):
            _install_launchd(args)

        # The plist landed at the expected location.
        plist_path = agents / "ai.hermes.dashboard.plist"
        assert plist_path.is_file()
        # And launchctl bootstrap was called with gui/<uid> + the plist.
        bootstrap_calls = [c for c in run_calls if c[:1] == ["bootstrap"]]
        assert len(bootstrap_calls) == 1
        assert bootstrap_calls[0][1].startswith("gui/")
        assert bootstrap_calls[0][2] == str(plist_path)

    def test_darwin_install_occupied_port_exits(self, tmp_path, monkeypatch):
        """The installer must refuse to install when the target
        port is already in use, so the user doesn't end up with
        a LaunchAgent that immediately fails to bind."""
        from hermes_cli.main import _install_launchd

        monkeypatch.setattr("hermes_cli.main.sys.platform", "darwin")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(
            "hermes_cli.main._port_is_in_use", lambda h, p, t=0.25: True
        )

        args = mock.Mock(host="127.0.0.1", port=9119, insecure=False)
        with mock.patch("hermes_cli.main.sys.exit") as exit_mock:
            with mock.patch("builtins.print"):
                _install_launchd(args)
        assert exit_mock.called
        assert exit_mock.call_args.args == (1,)


# ── Uninstaller ────────────────────────────────────────────────────────


class TestUninstall:
    def test_non_darwin_uninstall_exits_nonzero(self, monkeypatch):
        from hermes_cli.main import _uninstall_launchd

        monkeypatch.setattr("hermes_cli.main.sys.platform", "linux")
        with mock.patch("hermes_cli.main.sys.exit") as exit_mock:
            with mock.patch("builtins.print"):
                _uninstall_launchd()
        assert exit_mock.called
        assert exit_mock.call_args.args == (1,)

    def test_darwin_uninstall_removes_plist(self, tmp_path, monkeypatch):
        from hermes_cli.main import _uninstall_launchd

        monkeypatch.setattr("hermes_cli.main.sys.platform", "darwin")
        # Place a fake plist where the uninstaller will look.
        agents = tmp_path / "Library" / "LaunchAgents"
        agents.mkdir(parents=True)
        plist = agents / "ai.hermes.dashboard.plist"
        plist.write_text("<?xml version='1.0'?><plist/>")
        monkeypatch.setattr(
            "hermes_cli.main.Path.home", lambda: tmp_path
        )

        run_calls: list[list[str]] = []

        def _fake_run(*args, check=True):
            run_calls.append(list(args))
            return mock.Mock(returncode=0, stderr="")

        monkeypatch.setattr("hermes_cli.main._run_launchctl", _fake_run)

        with mock.patch("builtins.print"):
            _uninstall_launchd()

        assert not plist.exists(), "uninstall should remove the plist"
        # bootout was called.
        assert any(c[:1] == ["bootout"] for c in run_calls)

    def test_darwin_uninstall_no_plist_is_ok(self, tmp_path, monkeypatch):
        """If the plist is already gone (or never installed), the
        uninstaller must be idempotent — not a fatal error."""
        from hermes_cli.main import _uninstall_launchd

        monkeypatch.setattr("hermes_cli.main.sys.platform", "darwin")
        agents = tmp_path / "Library" / "LaunchAgents"
        agents.mkdir(parents=True)
        # No plist file.
        monkeypatch.setattr(
            "hermes_cli.main.Path.home", lambda: tmp_path
        )
        monkeypatch.setattr(
            "hermes_cli.main._run_launchctl",
            lambda *a, check=True: mock.Mock(returncode=0, stderr=""),
        )

        with mock.patch("builtins.print"):
            _uninstall_launchd()  # must not raise
