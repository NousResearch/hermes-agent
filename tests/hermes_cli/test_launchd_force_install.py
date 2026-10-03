"""Force-install launchd behavior under drain EIO and registration outcomes (#120629)."""
import subprocess

import pytest
from unittest.mock import MagicMock

from hermes_cli import gateway as gw
from hermes_cli.gateway_command_errors import explain_service_failure


@pytest.mark.parametrize("registered", [True, False])
def test_launchd_force_install_bootout_wait_retry_order(tmp_path, monkeypatch, capsys, registered):
    """force=True performs bootout→wait→retry; reports success only on actual registration."""
    plist_path = tmp_path / "com.hermes.plist"
    plist_path.write_text("<old/>", encoding="utf-8")
    monkeypatch.setattr(gw, "get_launchd_plist_path", lambda: plist_path)
    monkeypatch.setattr(gw, "generate_launchd_plist", lambda: "<new/>")
    monkeypatch.setattr(gw, "_refuse_temp_home_service_write", lambda *a: False)
    monkeypatch.setattr(gw, "get_launchd_label", lambda: "com.hermes.agent")
    monkeypatch.setattr(gw, "_launchd_domain", lambda: "gui/501")
    monkeypatch.setattr(gw, "_launchctl_supervised_pid", lambda label: 417)
    calls = []

    def _mock_wait(pid, budget):
        calls.append(("wait", pid))
        return True

    monkeypatch.setattr(gw, "_wait_for_pid_exit", _mock_wait)

    def _mock_retry(*a, **k):
        calls.append(("retry", gw.get_launchd_label()))
        return registered

    monkeypatch.setattr(gw, "_retry_launchctl_bootstrap_until_registered", _mock_retry)

    def _mock_run(cmd, **kwargs):
        subcmd = cmd[1] if isinstance(cmd, (list, tuple)) and len(cmd) > 1 else str(cmd)
        calls.append((subcmd, cmd))
        if subcmd == "bootstrap":
            raise subprocess.CalledProcessError(5, cmd)
        return MagicMock(returncode=0)

    monkeypatch.setattr(subprocess, "run", _mock_run)

    if registered:
        gw.launchd_install(force=True)
    else:
        with pytest.raises(SystemExit) as exc:
            gw.launchd_install(force=True)
        assert exc.value.code == 1

    out = capsys.readouterr().out
    bootout_idx = next((i for i, c in enumerate(calls) if c[0] == "bootout"), -1)
    wait_idx = next((i for i, c in enumerate(calls) if c[0] == "wait"), -1)
    retry_idx = next((i for i, c in enumerate(calls) if c[0] == "retry"), -1)
    assert 0 <= bootout_idx < wait_idx < retry_idx
    assert ("wait", 417) in calls
    assert ("retry", "com.hermes.agent") in calls
    assert ("Service installed and loaded!" in out) is registered
    if not registered:
        assert "may be unloaded" in out


def test_explain_service_failure_mentions_launchd_not_systemd():
    exc = subprocess.CalledProcessError(5, ["launchctl", "bootstrap"])
    lines = explain_service_failure(exc) or []
    joined = " ".join(str(x) for x in lines)
    assert "launchd" in joined
    assert "systemd" not in joined
    assert "journalctl" not in joined

    # systemd control path remains for non-launchctl errors
    exc2 = subprocess.CalledProcessError(1, ["systemctl", "start"])
    lines2 = explain_service_failure(exc2) or []
    joined2 = " ".join(str(x) for x in lines2)
    assert "systemd" in joined2
    assert "journalctl" in joined2
    assert "launchd" not in joined2