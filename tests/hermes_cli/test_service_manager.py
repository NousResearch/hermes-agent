"""Unit tests for ``hermes_cli.service_manager``.

``service_manager.py`` (added in PR #73, 1,123 lines) shipped with zero
test coverage despite sitting on the gateway lifecycle critical path.

Everything that shells out is mocked: no test invokes a real ``s6-svc``,
``s6-svstat``, ``s6-svscanctl``, ``systemctl``, ``launchctl`` or touches a
real supervised service. ``subprocess.run`` is patched (the module does a
local ``import subprocess`` inside each method, so patching
``subprocess.run`` in the shared ``subprocess`` module intercepts it),
and the host backends' delegate functions are patched on
``hermes_cli.gateway``.

Covered:
  * ``detect_service_manager`` / ``get_service_manager`` backend selection
    for every platform plus the fallback "none" case.
  * ``S6ServiceManager`` start / stop / restart / is_running, error
    translation (GatewayNotRegisteredError / S6CommandError),
    register / unregister / list_profile_gateways.
  * ``SystemdServiceManager`` / ``LaunchdServiceManager`` restart + the
    registration-unsupported contract.
  * ``_dispatch_via_service_manager_if_s6`` and the ``_all_`` variant.
  * ``write_platform_config_field`` round-trip against a tmp config.
  * ``validate_profile_name``.
"""
from __future__ import annotations

import os
import subprocess
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import service_manager as sm
from hermes_cli.service_manager import (
    GatewayNotRegisteredError,
    LaunchdServiceManager,
    S6CommandError,
    S6ServiceManager,
    SystemdServiceManager,
    WindowsServiceManager,
    detect_service_manager,
    get_service_manager,
    validate_profile_name,
)

S6_BIN = "/command"


def _completed(returncode=0, stdout="", stderr=""):
    """Stand-in for ``subprocess.CompletedProcess`` with concrete attrs.

    Using a real object (not a bare MagicMock) keeps ``returncode != 0``
    and string ``.stdout`` comparisons behaving like the production code
    expects.
    """
    return types.SimpleNamespace(
        returncode=returncode, stdout=stdout, stderr=stderr
    )


# ---------------------------------------------------------------------------
# validate_profile_name
# ---------------------------------------------------------------------------


class TestValidateProfileName:
    def test_accepts_valid_names(self):
        for name in ("default", "a", "team-1", "team_1", "abc123"):
            validate_profile_name(name)  # no raise

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            validate_profile_name("")

    def test_rejects_uppercase(self):
        with pytest.raises(ValueError):
            validate_profile_name("Team")

    def test_rejects_path_traversal(self):
        with pytest.raises(ValueError):
            validate_profile_name("../etc")

    def test_rejects_leading_dash(self):
        with pytest.raises(ValueError):
            validate_profile_name("-foo")

    def test_rejects_too_long(self):
        with pytest.raises(ValueError):
            validate_profile_name("a" * 252)


# ---------------------------------------------------------------------------
# detect_service_manager / get_service_manager
# ---------------------------------------------------------------------------


class TestDetectServiceManager:
    """``detect_service_manager`` keys on ``_s6_running`` then, in order,
    the gateway helpers ``is_windows`` / ``is_macos`` /
    ``supports_systemd_services`` (imported lazily from
    ``hermes_cli.gateway``)."""

    def _patch_gateway(self, monkeypatch, *, windows=False, macos=False,
                       systemd=False):
        from hermes_cli import gateway
        monkeypatch.setattr(gateway, "is_windows", lambda: windows)
        monkeypatch.setattr(gateway, "is_macos", lambda: macos)
        monkeypatch.setattr(
            gateway, "supports_systemd_services", lambda: systemd
        )

    def test_s6_wins_over_everything(self, monkeypatch):
        monkeypatch.setattr(sm, "_s6_running", lambda: True)
        # Even with a windows host underneath, s6 short-circuits first.
        self._patch_gateway(monkeypatch, windows=True)
        assert detect_service_manager() == "s6"

    def test_windows(self, monkeypatch):
        monkeypatch.setattr(sm, "_s6_running", lambda: False)
        self._patch_gateway(monkeypatch, windows=True)
        assert detect_service_manager() == "windows"

    def test_launchd(self, monkeypatch):
        monkeypatch.setattr(sm, "_s6_running", lambda: False)
        self._patch_gateway(monkeypatch, macos=True)
        assert detect_service_manager() == "launchd"

    def test_systemd(self, monkeypatch):
        monkeypatch.setattr(sm, "_s6_running", lambda: False)
        self._patch_gateway(monkeypatch, systemd=True)
        assert detect_service_manager() == "systemd"

    def test_none_fallback(self, monkeypatch):
        monkeypatch.setattr(sm, "_s6_running", lambda: False)
        self._patch_gateway(monkeypatch)  # all False
        assert detect_service_manager() == "none"


class TestS6Running:
    def test_false_when_comm_unreadable(self, monkeypatch):
        def _raise(*_a, **_k):
            raise OSError("nope")
        monkeypatch.setattr(Path, "read_text", _raise)
        assert sm._s6_running() is False

    def test_false_when_comm_not_s6(self, monkeypatch):
        monkeypatch.setattr(Path, "read_text", lambda *_a, **_k: "bash\n")
        assert sm._s6_running() is False

    def test_true_when_comm_and_basedir(self, monkeypatch):
        monkeypatch.setattr(
            Path, "read_text", lambda *_a, **_k: "s6-svscan\n"
        )
        monkeypatch.setattr(Path, "is_dir", lambda *_a, **_k: True)
        assert sm._s6_running() is True

    def test_false_when_basedir_missing(self, monkeypatch):
        monkeypatch.setattr(
            Path, "read_text", lambda *_a, **_k: "s6-svscan\n"
        )
        monkeypatch.setattr(Path, "is_dir", lambda *_a, **_k: False)
        assert sm._s6_running() is False


class TestGetServiceManager:
    def test_returns_backend_per_kind(self, monkeypatch):
        mapping = {
            "systemd": SystemdServiceManager,
            "launchd": LaunchdServiceManager,
            "windows": WindowsServiceManager,
            "s6": S6ServiceManager,
        }
        for kind, cls in mapping.items():
            monkeypatch.setattr(sm, "detect_service_manager", lambda k=kind: k)
            mgr = get_service_manager()
            assert isinstance(mgr, cls)
            assert mgr.kind == kind

    def test_none_raises_runtimeerror(self, monkeypatch):
        monkeypatch.setattr(sm, "detect_service_manager", lambda: "none")
        with pytest.raises(RuntimeError):
            get_service_manager()


# ---------------------------------------------------------------------------
# S6ServiceManager — lifecycle
# ---------------------------------------------------------------------------


@pytest.fixture
def s6(tmp_path, monkeypatch):
    """An S6ServiceManager rooted at a tmp scandir, with HERMES_HOME set
    to tmp so the best-effort desired-state writes never touch a real
    profile dir."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    scandir = tmp_path / "service"
    scandir.mkdir()
    return S6ServiceManager(scandir=scandir)


class TestS6Lifecycle:
    def test_start_issues_svc_up(self, s6):
        (s6.scandir / "gateway-foo").mkdir()
        with patch("subprocess.run", return_value=_completed()) as run:
            s6.start("gateway-foo")
        run.assert_called_once()
        args, kwargs = run.call_args
        assert args[0] == [
            f"{S6_BIN}/s6-svc", "-u", str(s6.scandir / "gateway-foo")
        ]
        assert kwargs["check"] is True
        assert kwargs["timeout"] == 5

    def test_restart_issues_svc_t(self, s6):
        (s6.scandir / "gateway-foo").mkdir()
        with patch("subprocess.run", return_value=_completed()) as run:
            s6.restart("gateway-foo")
        assert run.call_args.args[0] == [
            f"{S6_BIN}/s6-svc", "-t", str(s6.scandir / "gateway-foo")
        ]

    def test_stop_issues_svc_down(self, s6):
        (s6.scandir / "gateway-foo").mkdir()

        def fake_run(cmd, *a, **k):
            # s6-svstat probe in _supervised_pid: report no pid so the
            # planned-stop marker path is skipped.
            if cmd[0].endswith("s6-svstat"):
                return _completed(returncode=1)
            return _completed()

        with patch("subprocess.run", side_effect=fake_run) as run:
            s6.stop("gateway-foo")
        svc_calls = [c for c in run.call_args_list
                     if c.args[0][0].endswith("s6-svc")]
        assert svc_calls[0].args[0] == [
            f"{S6_BIN}/s6-svc", "-d", str(s6.scandir / "gateway-foo")
        ]

    def test_stop_writes_planned_marker_when_pid_present(self, s6):
        (s6.scandir / "gateway-foo").mkdir()

        def fake_run(cmd, *a, **k):
            if cmd[0].endswith("s6-svstat"):
                return _completed(returncode=0, stdout="up (pid 4321) 5s\n")
            return _completed()

        marker = MagicMock()
        with patch("subprocess.run", side_effect=fake_run), \
                patch("gateway.status.write_planned_stop_marker", marker):
            s6.stop("gateway-foo")
        marker.assert_called_once_with(4321)

    def test_missing_service_dir_raises_not_registered(self, s6):
        with patch("subprocess.run") as run:
            with pytest.raises(GatewayNotRegisteredError) as ei:
                s6.start("gateway-ghost")
        # s6-svc is never invoked when the slot doesn't exist.
        run.assert_not_called()
        assert ei.value.profile == "ghost"
        assert ei.value.service == "gateway-ghost"

    def test_svc_nonzero_raises_command_error(self, s6):
        (s6.scandir / "gateway-foo").mkdir()
        err = subprocess.CalledProcessError(
            returncode=100, cmd=["s6-svc"], output="", stderr="EACCES fifo"
        )
        with patch("subprocess.run", side_effect=err):
            with pytest.raises(S6CommandError) as ei:
                s6.start("gateway-foo")
        assert ei.value.returncode == 100
        assert ei.value.action == "start"
        assert "EACCES fifo" in str(ei.value)

    def test_is_running_true(self, s6):
        with patch("subprocess.run",
                   return_value=_completed(returncode=0, stdout="up (pid 5) 1s")):
            assert s6.is_running("gateway-foo") is True

    def test_is_running_false_when_down(self, s6):
        with patch("subprocess.run",
                   return_value=_completed(returncode=0, stdout="down 1s")):
            assert s6.is_running("gateway-foo") is False

    def test_is_running_false_on_nonzero(self, s6):
        with patch("subprocess.run",
                   return_value=_completed(returncode=1, stdout="")):
            assert s6.is_running("gateway-foo") is False

    def test_supports_runtime_registration(self, s6):
        assert s6.supports_runtime_registration() is True


# ---------------------------------------------------------------------------
# S6ServiceManager — registration
# ---------------------------------------------------------------------------


class TestS6Registration:
    def test_register_builds_service_dir(self, s6):
        with patch("subprocess.run", return_value=_completed()) as run:
            s6.register_profile_gateway("foo")

        svc = s6.scandir / "gateway-foo"
        assert svc.is_dir()
        assert (svc / "type").read_text() == "longrun\n"
        assert (svc / "run").exists()
        assert (svc / "finish").exists()
        assert (svc / "log" / "run").exists()
        assert not (svc / "down").exists()  # start_now default True
        # rescan triggered.
        assert run.call_args.args[0] == [
            f"{S6_BIN}/s6-svscanctl", "-a", str(s6.scandir)
        ]
        # run script targets the profile, drops privileges.
        run_body = (svc / "run").read_text()
        assert "hermes -p foo gateway run --replace" in run_body
        assert "s6-setuidgid hermes" in run_body

    def test_register_default_profile_omits_p_flag(self, s6):
        with patch("subprocess.run", return_value=_completed()):
            s6.register_profile_gateway("default")
        run_body = (s6.scandir / "gateway-default" / "run").read_text()
        assert "hermes gateway run --replace" in run_body
        assert " -p " not in run_body

    def test_register_start_now_false_writes_down_marker(self, s6):
        with patch("subprocess.run", return_value=_completed()):
            s6.register_profile_gateway("foo", start_now=False)
        assert (s6.scandir / "gateway-foo" / "down").exists()

    def test_register_existing_raises_valueerror(self, s6):
        (s6.scandir / "gateway-foo").mkdir()
        with patch("subprocess.run", return_value=_completed()):
            with pytest.raises(ValueError):
                s6.register_profile_gateway("foo")

    def test_register_rescan_failure_cleans_up_and_raises(self, s6):
        with patch(
            "subprocess.run",
            return_value=_completed(returncode=1, stderr="scan boom"),
        ):
            with pytest.raises(RuntimeError, match="s6-svscanctl"):
                s6.register_profile_gateway("foo")
        # The half-built slot is torn down on rescan failure.
        assert not (s6.scandir / "gateway-foo").exists()

    def test_register_invalid_profile_rejected(self, s6):
        with patch("subprocess.run", return_value=_completed()):
            with pytest.raises(ValueError):
                s6.register_profile_gateway("../evil")

    def test_unregister_stops_and_removes(self, s6):
        svc = s6.scandir / "gateway-foo"
        svc.mkdir()
        with patch("subprocess.run", return_value=_completed()) as run, \
                patch("time.sleep"):
            s6.unregister_profile_gateway("foo")
        assert not svc.exists()
        cmds = [c.args[0][1] if len(c.args[0]) > 1 else c.args[0][0]
                for c in run.call_args_list]
        # Ordering: stop (-d), wait-for-down (-D), reap (-an).
        joined = [" ".join(c.args[0]) for c in run.call_args_list]
        assert any("s6-svc -d" in j for j in joined)
        assert any("s6-svwait" in j for j in joined)
        assert any("s6-svscanctl -an" in j for j in joined)

    def test_unregister_absent_is_noop(self, s6):
        with patch("subprocess.run") as run, patch("time.sleep"):
            s6.unregister_profile_gateway("nope")
        run.assert_not_called()

    def test_list_profile_gateways(self, s6):
        (s6.scandir / "gateway-alpha").mkdir()
        (s6.scandir / "gateway-beta").mkdir()
        (s6.scandir / ".hidden-tmp").mkdir()  # skipped (dot-prefixed)
        (s6.scandir / "s6-linux-init").mkdir()  # skipped (no prefix)
        (s6.scandir / "gateway-file").write_text("x")  # skipped (not dir)
        assert set(s6.list_profile_gateways()) == {"alpha", "beta"}

    def test_list_profile_gateways_missing_scandir(self, tmp_path):
        mgr = S6ServiceManager(scandir=tmp_path / "does-not-exist")
        assert mgr.list_profile_gateways() == []

    def test_render_finish_script_defines_fatal_config_exit_code(self):
        """Regression test for a real bug: _render_finish_script imported
        GATEWAY_FATAL_CONFIG_EXIT_CODE from gateway.restart, but that
        symbol was never defined there — only GATEWAY_SERVICE_RESTART_EXIT_CODE
        existed. That raised ImportError on the real register_profile_gateway
        path, breaking s6 profile-gateway registration in production.

        Assert the import now resolves and the rendered finish script
        checks for the correct (EX_CONFIG) exit code, distinct from the
        restart exit code.
        """
        from gateway.restart import (
            GATEWAY_FATAL_CONFIG_EXIT_CODE,
            GATEWAY_SERVICE_RESTART_EXIT_CODE,
        )

        assert GATEWAY_FATAL_CONFIG_EXIT_CODE == 78
        assert GATEWAY_FATAL_CONFIG_EXIT_CODE != GATEWAY_SERVICE_RESTART_EXIT_CODE

        script = S6ServiceManager._render_finish_script()
        assert f'if [ "$1" = "{GATEWAY_FATAL_CONFIG_EXIT_CODE}" ]; then' in script
        assert "exit 125" in script


# ---------------------------------------------------------------------------
# Host backends — systemd / launchd
# ---------------------------------------------------------------------------


class TestHostBackends:
    def test_systemd_restart_delegates(self, monkeypatch):
        from hermes_cli import gateway
        called = MagicMock()
        monkeypatch.setattr(gateway, "systemd_restart", called)
        SystemdServiceManager().restart("ignored")
        called.assert_called_once_with()

    def test_systemd_start_stop_delegate(self, monkeypatch):
        from hermes_cli import gateway
        start, stop = MagicMock(), MagicMock()
        monkeypatch.setattr(gateway, "systemd_start", start)
        monkeypatch.setattr(gateway, "systemd_stop", stop)
        mgr = SystemdServiceManager()
        mgr.start("x")
        mgr.stop("x")
        start.assert_called_once_with()
        stop.assert_called_once_with()

    def test_launchd_restart_delegates(self, monkeypatch):
        from hermes_cli import gateway
        called = MagicMock()
        monkeypatch.setattr(gateway, "launchd_restart", called)
        LaunchdServiceManager().restart("ignored")
        called.assert_called_once_with()

    def test_host_backends_reject_registration(self):
        for cls in (SystemdServiceManager, LaunchdServiceManager,
                    WindowsServiceManager):
            mgr = cls()
            assert mgr.supports_runtime_registration() is False
            assert mgr.list_profile_gateways() == []
            with pytest.raises(NotImplementedError):
                mgr.register_profile_gateway("foo")
            with pytest.raises(NotImplementedError):
                mgr.unregister_profile_gateway("foo")


# ---------------------------------------------------------------------------
# CLI wiring — _dispatch_via_service_manager_if_s6 (+ _all_ variant)
# ---------------------------------------------------------------------------


class TestDispatchViaServiceManager:
    def test_non_s6_falls_through(self, monkeypatch):
        from hermes_cli import gateway
        monkeypatch.setattr(sm, "detect_service_manager", lambda: "systemd")
        got_mgr = MagicMock()
        monkeypatch.setattr(sm, "get_service_manager", got_mgr)
        assert gateway._dispatch_via_service_manager_if_s6(
            "restart", profile="foo"
        ) is False
        got_mgr.assert_not_called()

    def test_s6_restart_dispatches(self, monkeypatch):
        from hermes_cli import gateway
        monkeypatch.setattr(sm, "detect_service_manager", lambda: "s6")
        mgr = MagicMock()
        monkeypatch.setattr(sm, "get_service_manager", lambda: mgr)
        assert gateway._dispatch_via_service_manager_if_s6(
            "restart", profile="foo"
        ) is True
        mgr.restart.assert_called_once_with("gateway-foo")

    def test_s6_start_and_stop_dispatch(self, monkeypatch):
        from hermes_cli import gateway
        monkeypatch.setattr(sm, "detect_service_manager", lambda: "s6")
        mgr = MagicMock()
        monkeypatch.setattr(sm, "get_service_manager", lambda: mgr)
        gateway._dispatch_via_service_manager_if_s6("start", profile="p")
        gateway._dispatch_via_service_manager_if_s6("stop", profile="p")
        mgr.start.assert_called_once_with("gateway-p")
        mgr.stop.assert_called_once_with("gateway-p")

    def test_s6_unknown_action_returns_false(self, monkeypatch):
        from hermes_cli import gateway
        monkeypatch.setattr(sm, "detect_service_manager", lambda: "s6")
        mgr = MagicMock()
        monkeypatch.setattr(sm, "get_service_manager", lambda: mgr)
        assert gateway._dispatch_via_service_manager_if_s6(
            "bogus", profile="p"
        ) is False

    def test_not_registered_exits(self, monkeypatch, capsys):
        from hermes_cli import gateway
        monkeypatch.setattr(sm, "detect_service_manager", lambda: "s6")
        mgr = MagicMock()
        mgr.restart.side_effect = GatewayNotRegisteredError("typo")
        monkeypatch.setattr(sm, "get_service_manager", lambda: mgr)
        with pytest.raises(SystemExit) as ei:
            gateway._dispatch_via_service_manager_if_s6(
                "restart", profile="typo"
            )
        assert ei.value.code == 1
        assert "typo" in capsys.readouterr().out

    def test_command_error_exits(self, monkeypatch, capsys):
        from hermes_cli import gateway
        monkeypatch.setattr(sm, "detect_service_manager", lambda: "s6")
        mgr = MagicMock()
        mgr.stop.side_effect = S6CommandError(
            service="gateway-p", action="stop", returncode=1, stderr="boom"
        )
        monkeypatch.setattr(sm, "get_service_manager", lambda: mgr)
        with pytest.raises(SystemExit):
            gateway._dispatch_via_service_manager_if_s6("stop", profile="p")
        assert "boom" in capsys.readouterr().out


class TestDispatchAllViaServiceManager:
    def test_non_s6_falls_through(self, monkeypatch):
        from hermes_cli import gateway
        monkeypatch.setattr(sm, "detect_service_manager", lambda: "systemd")
        assert gateway._dispatch_all_via_service_manager_if_s6(
            "restart"
        ) is False

    def test_unsupported_action_returns_false(self, monkeypatch):
        from hermes_cli import gateway
        monkeypatch.setattr(sm, "detect_service_manager", lambda: "s6")
        monkeypatch.setattr(sm, "get_service_manager", lambda: MagicMock())
        assert gateway._dispatch_all_via_service_manager_if_s6(
            "start"
        ) is False

    def test_restart_all_iterates_every_gateway(self, monkeypatch, capsys):
        from hermes_cli import gateway
        monkeypatch.setattr(sm, "detect_service_manager", lambda: "s6")
        mgr = MagicMock()
        mgr.list_profile_gateways.return_value = ["a", "b"]
        monkeypatch.setattr(sm, "get_service_manager", lambda: mgr)
        assert gateway._dispatch_all_via_service_manager_if_s6(
            "restart"
        ) is True
        mgr.restart.assert_any_call("gateway-a")
        mgr.restart.assert_any_call("gateway-b")
        assert mgr.restart.call_count == 2

    def test_stop_all_reports_errors_and_continues(self, monkeypatch, capsys):
        from hermes_cli import gateway
        monkeypatch.setattr(sm, "detect_service_manager", lambda: "s6")
        mgr = MagicMock()
        mgr.list_profile_gateways.return_value = ["a", "b"]

        def stop(name):
            if name == "gateway-a":
                raise S6CommandError(
                    service=name, action="stop", returncode=1, stderr="x"
                )
        mgr.stop.side_effect = stop
        monkeypatch.setattr(sm, "get_service_manager", lambda: mgr)
        assert gateway._dispatch_all_via_service_manager_if_s6("stop") is True
        out = capsys.readouterr().out
        assert "gateway-a" in out  # error line for the failing one
        assert mgr.stop.call_count == 2  # continued past the failure

    def test_no_gateways_registered(self, monkeypatch, capsys):
        from hermes_cli import gateway
        monkeypatch.setattr(sm, "detect_service_manager", lambda: "s6")
        mgr = MagicMock()
        mgr.list_profile_gateways.return_value = []
        monkeypatch.setattr(sm, "get_service_manager", lambda: mgr)
        assert gateway._dispatch_all_via_service_manager_if_s6(
            "restart"
        ) is True
        assert "No profile gateways" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# write_platform_config_field
# ---------------------------------------------------------------------------


class TestWritePlatformConfigField:
    def test_writes_and_roundtrips(self, tmp_path, monkeypatch):
        from hermes_cli import config as cfg
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        cfg.write_platform_config_field("telegram", "bot_token", "abc123")
        reloaded = cfg.load_config()
        assert reloaded["platforms"]["telegram"]["bot_token"] == "abc123"

    def test_updates_existing_field(self, tmp_path, monkeypatch):
        from hermes_cli import config as cfg
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        cfg.write_platform_config_field("telegram", "bot_token", "old")
        cfg.write_platform_config_field("telegram", "bot_token", "new")
        reloaded = cfg.load_config()
        assert reloaded["platforms"]["telegram"]["bot_token"] == "new"

    def test_preserves_sibling_platforms(self, tmp_path, monkeypatch):
        from hermes_cli import config as cfg
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        cfg.write_platform_config_field("telegram", "bot_token", "tg")
        cfg.write_platform_config_field("discord", "bot_token", "dc")
        reloaded = cfg.load_config()
        assert reloaded["platforms"]["telegram"]["bot_token"] == "tg"
        assert reloaded["platforms"]["discord"]["bot_token"] == "dc"

    def test_raw_writes_to_raw_config(self, tmp_path, monkeypatch):
        import yaml
        from hermes_cli import config as cfg
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        cfg.write_platform_config_field(
            "telegram", "bot_token", "raw-token", raw=True
        )
        on_disk = yaml.safe_load((tmp_path / "config.yaml").read_text())
        assert on_disk["platforms"]["telegram"]["bot_token"] == "raw-token"
