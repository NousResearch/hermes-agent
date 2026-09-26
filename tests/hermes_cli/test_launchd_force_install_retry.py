"""Forced launchd reinstall must wait out the old supervised process before bootstrap (#120629)."""

from types import SimpleNamespace

import pytest

import hermes_cli.gateway as gateway_cli


LABEL = "ai.hermes.gateway"
DOMAIN = "gui/501"


def _forced_install_fixture(tmp_path, monkeypatch):
    plist = tmp_path / "LaunchAgents" / f"{LABEL}.plist"
    plist.parent.mkdir(parents=True)
    plist.write_text("<old/>", encoding="utf-8")
    events = []

    monkeypatch.setattr(gateway_cli, "get_launchd_plist_path", lambda: plist)
    monkeypatch.setattr(gateway_cli, "get_launchd_label", lambda: LABEL)
    monkeypatch.setattr(gateway_cli, "_launchd_domain", lambda: DOMAIN)
    monkeypatch.setattr(gateway_cli, "_launchctl_label_supervising_process", lambda _label: True)
    monkeypatch.setattr(gateway_cli, "_launchctl_supervised_pid", lambda _label: 4242)
    monkeypatch.setattr(gateway_cli, "generate_launchd_plist", lambda: "<new/>")
    monkeypatch.setattr(gateway_cli, "_refuse_temp_home_service_write", lambda *_a: False)
    monkeypatch.setattr(gateway_cli, "_clear_launchd_unsupported_marker", lambda: None)
    monkeypatch.setattr(gateway_cli, "_launchd_reload_budget", lambda: 45.0)
    monkeypatch.setattr(
        gateway_cli, "_wait_for_pid_exit",
        lambda pid, timeout: events.append(("wait", pid, timeout)) or True,
    )
    monkeypatch.setattr(gateway_cli, "_append_launchd_reload_log", lambda msg: events.append(("log", msg)))
    monkeypatch.setattr(
        gateway_cli, "_launchctl_bootstrap",
        lambda *_a, **_k: pytest.fail("forced live reinstall must use the bounded retry path"),
    )

    def fake_run(cmd, *args, **kwargs):
        events.append(("run", cmd))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)
    return plist, events


def test_force_install_waits_for_incumbent_then_retries_until_supervised(tmp_path, monkeypatch, capsys):
    plist, events = _forced_install_fixture(tmp_path, monkeypatch)

    def retry(domain, plist_path, label, *, deadline):
        events.append(("retry", domain, plist_path, label, deadline))
        return True

    monkeypatch.setattr(gateway_cli, "_retry_launchctl_bootstrap_until_registered", retry)

    gateway_cli.launchd_install(force=True)

    assert ("run", ["launchctl", "bootout", f"{DOMAIN}/{LABEL}"]) in events
    wait_index = events.index(("wait", 4242, 45.0))
    retry_index = next(i for i, event in enumerate(events) if event[0] == "retry")
    assert wait_index < retry_index
    assert plist.read_text(encoding="utf-8") == "<new/>"
    assert "Service installed and loaded" in capsys.readouterr().out


def test_force_install_fails_closed_when_launchd_never_registers(tmp_path, monkeypatch, capsys):
    _plist, events = _forced_install_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        gateway_cli, "_retry_launchctl_bootstrap_until_registered",
        lambda *_a, **_k: False,
    )
    errors = []
    monkeypatch.setattr(gateway_cli, "print_error", errors.append)

    with pytest.raises(SystemExit) as exc_info:
        gateway_cli.launchd_install(force=True)

    assert exc_info.value.code == 1
    assert any("service is NOT loaded" in message for message in errors)
    assert "Service installed and loaded" not in capsys.readouterr().out
    assert any(event[0] == "wait" for event in events)
