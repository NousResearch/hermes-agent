"""Supervisor detection must not misread desktop-session INVOCATION_ID.

On a GNOME/ptyxis desktop login every process carries ``INVOCATION_ID``
(gnome-session and the terminal's transient ``*-spawn-*.scope`` are systemd
units), so the env marker alone cannot distinguish a supervised gateway
service from a CLI running in a desktop terminal.  A desktop-terminal
gateway that is treated as supervised routes ``/restart`` to the exit-75
service path, where no service manager exists to restart it — the gateway
just dies.

The fix vetoes INVOCATION_ID only when the process's innermost systemd
cgroup unit is a transient ``.scope`` (desktop/terminal shape).  Any
``.service`` ownership — standard, profile-suffixed, legacy, or
user-custom names — and any platform where the cgroup file is unreadable
keep the historical INVOCATION_ID meaning.
"""

import pytest

from gateway.restart import (
    _innermost_systemd_unit_kinds,
    is_gateway_supervisor_process,
)


def _write_cgroup(tmp_path, content):
    path = tmp_path / "cgroup"
    path.write_text(content, encoding="utf-8")
    return str(path)


ENV_INVOCATION = {"INVOCATION_ID": "abc123", "XPC_SERVICE_NAME": "0"}
ENV_EMPTY = {"XPC_SERVICE_NAME": "0"}


class TestInnermostUnitKinds:
    def test_service_leaf(self, tmp_path):
        p = _write_cgroup(
            tmp_path,
            "0::/system.slice/hermes-gateway.service\n",
        )
        assert _innermost_systemd_unit_kinds(p) == {"service"}

    def test_desktop_scope_under_user_manager(self, tmp_path):
        """user@N.service encloses the scope but the SCOPE is innermost."""
        p = _write_cgroup(
            tmp_path,
            "0::/user.slice/user-1000.slice/user@1000.service/app.slice/"
            "ptyxis-spawn-42.scope\n",
        )
        assert _innermost_systemd_unit_kinds(p) == {"scope"}

    def test_delegated_sub_cgroup_still_service(self, tmp_path):
        """Delegate=yes nests sub-cgroups under the service — still a service."""
        p = _write_cgroup(
            tmp_path,
            "0::/system.slice/hermes-gateway.service/worker\n",
        )
        assert _innermost_systemd_unit_kinds(p) == {"service"}

    def test_user_service_leaf_under_user_manager(self, tmp_path):
        """systemctl --user unit: .service leaf under user@N.service → service.

        This is the shape where a naive "any .service in the path" and the
        leaf-inward scan agree, but a naive "last component only" parse and
        an outer-first scan would disagree — pin it explicitly.
        """
        p = _write_cgroup(
            tmp_path,
            "0::/user.slice/user-1000.slice/user@1000.service/app.slice/"
            "hermes-gateway.service\n",
        )
        assert _innermost_systemd_unit_kinds(p) == {"service"}

    def test_cgroup_v1_multiline(self, tmp_path):
        """v1 exposes many hierarchies; the systemd one carries the unit."""
        p = _write_cgroup(
            tmp_path,
            "12:cpu,cpuacct:/\n"
            "3:memory:/user.slice\n"
            "1:name=systemd:/user.slice/user-1000.slice/user@1000.service/"
            "app.slice/ptyxis-spawn-7.scope\n",
        )
        assert _innermost_systemd_unit_kinds(p) == {"scope"}

    def test_unreadable_returns_empty(self, tmp_path):
        assert _innermost_systemd_unit_kinds(str(tmp_path / "missing")) == set()


class TestSupervisorDetection:
    @pytest.mark.parametrize(
        "unit",
        [
            "hermes-gateway.service",
            "hermes-gateway-coder.service",  # profile-suffixed
            "hermes.service",  # legacy name (_LEGACY_SERVICE_NAMES)
            "my-custom-hermes.service",  # user-custom unit
        ],
    )
    def test_service_units_remain_supervised(self, tmp_path, unit):
        p = _write_cgroup(tmp_path, f"0::/system.slice/{unit}\n")
        assert is_gateway_supervisor_process(ENV_INVOCATION, cgroup_path=p) is True

    def test_desktop_terminal_scope_is_not_supervised(self, tmp_path):
        """The GNOME/ptyxis false positive: INVOCATION_ID + innermost scope."""
        p = _write_cgroup(
            tmp_path,
            "0::/user.slice/user-1000.slice/user@1000.service/app.slice/"
            "ptyxis-spawn-42.scope\n",
        )
        assert is_gateway_supervisor_process(ENV_INVOCATION, cgroup_path=p) is False

    def test_session_scope_is_not_supervised(self, tmp_path):
        p = _write_cgroup(
            tmp_path,
            "0::/user.slice/user-1000.slice/user@1000.service/session-3.scope\n",
        )
        assert is_gateway_supervisor_process(ENV_INVOCATION, cgroup_path=p) is False

    def test_unreadable_cgroup_preserves_invocation_id(self, tmp_path):
        """macOS / no-procfs: INVOCATION_ID keeps its historical meaning."""
        missing = str(tmp_path / "missing")
        assert (
            is_gateway_supervisor_process(ENV_INVOCATION, cgroup_path=missing)
            is True
        )

    def test_no_markers_at_all(self, tmp_path):
        p = _write_cgroup(
            tmp_path,
            "0::/user.slice/user-1000.slice/user@1000.service/app.slice/"
            "ptyxis-spawn-42.scope\n",
        )
        assert is_gateway_supervisor_process(ENV_EMPTY, cgroup_path=p) is False

    def test_scope_veto_does_not_mask_other_markers(self, tmp_path):
        """s6 / launchd / explicit external markers bypass the cgroup veto."""
        p = _write_cgroup(
            tmp_path,
            "0::/user.slice/user-1000.slice/user@1000.service/app.slice/"
            "ptyxis-spawn-42.scope\n",
        )
        s6 = dict(ENV_INVOCATION, HERMES_S6_SUPERVISED_CHILD="1")
        assert is_gateway_supervisor_process(s6, cgroup_path=p) is True
        launchd = dict(ENV_INVOCATION, XPC_SERVICE_NAME="ai.hermes.gateway")
        assert is_gateway_supervisor_process(launchd, cgroup_path=p) is True
        external = dict(ENV_INVOCATION, HERMES_GATEWAY_EXTERNAL_SUPERVISOR="1")
        assert is_gateway_supervisor_process(external, cgroup_path=p) is True

    def test_default_env_and_default_cgroup_path_still_work(self, monkeypatch):
        """Existing callers pass nothing — env-based behavior is intact."""
        monkeypatch.delenv("INVOCATION_ID", raising=False)
        monkeypatch.delenv("HERMES_S6_SUPERVISED_CHILD", raising=False)
        monkeypatch.setenv("XPC_SERVICE_NAME", "0")
        monkeypatch.delenv("HERMES_GATEWAY_EXTERNAL_SUPERVISOR", raising=False)
        assert is_gateway_supervisor_process() is False
