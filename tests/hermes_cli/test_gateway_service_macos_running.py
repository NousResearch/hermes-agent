"""macOS gateway liveness: a registered-but-parked LaunchAgent is not a running gateway.

``KeepAlive``/``SuccessfulExit=false`` leaves the job registered with no process after a clean
exit (including the config-error exit after a token collision), and ``launchctl list <label>``
exits 0 for exactly that state. Callers of ``_is_service_running`` act on its answer — setup and
``hermes import`` skip the start entirely — so the check has to ask whether launchd supervises a
process, the same question the status snapshot already answers (#153309).
"""

import pytest

import hermes_cli.gateway as gateway_cli


@pytest.fixture
def macos_launchagent(monkeypatch, tmp_path):
    """A macOS host whose gateway plist exists; ``state['pid']`` is what launchd supervises."""
    plist = tmp_path / "ai.hermes.gateway.plist"
    plist.write_text("<plist/>", encoding="utf-8")
    state = {"pid": None, "registered": True, "pids": ()}
    monkeypatch.setattr(gateway_cli, "supports_systemd_services", lambda: False)
    # Branch selection only — every OS-touching call below is stubbed, so this arm is host-independent.
    monkeypatch.setattr(gateway_cli, "is_macos", lambda: True)  # os-marker: ok — pure branch selection, launchctl stubbed
    monkeypatch.setattr(gateway_cli, "get_launchd_label", lambda: "ai.hermes.gateway")
    monkeypatch.setattr(gateway_cli, "get_launchd_plist_path", lambda: plist)
    monkeypatch.setattr(gateway_cli, "find_gateway_pids", lambda: state["pids"])
    # ``launchctl list <label>`` exits 0 for a job launchd merely registered.
    monkeypatch.setattr(
        gateway_cli, "_launchd_service_registered", lambda label, timeout=5: state["registered"]
    )
    monkeypatch.setattr(gateway_cli, "_launchctl_supervised_pid", lambda label: state["pid"])
    return state


def test_registered_but_parked_launchagent_is_not_running(macos_launchagent):
    macos_launchagent["registered"] = True
    macos_launchagent["pid"] = None

    assert gateway_cli._is_service_running() is False


def test_launchagent_supervising_a_process_is_running(macos_launchagent):
    macos_launchagent["pid"] = 4242

    assert gateway_cli._is_service_running() is True


def test_parked_launchagent_does_not_hide_a_manual_gateway_process(
    macos_launchagent, monkeypatch, tmp_path
):
    """Without a plist the service question is not launchd's; a live gateway still decides."""
    monkeypatch.setattr(
        gateway_cli, "get_launchd_plist_path", lambda: tmp_path / "absent.plist"
    )
    macos_launchagent["pid"] = None
    macos_launchagent["pids"] = (77,)

    assert gateway_cli._is_service_running() is True


def test_launchctl_failure_reads_as_not_running(monkeypatch, tmp_path):
    """No launchctl on the host must answer False, not raise out of a setup/import path."""
    plist = tmp_path / "ai.hermes.gateway.plist"
    plist.write_text("<plist/>", encoding="utf-8")
    monkeypatch.setattr(gateway_cli, "supports_systemd_services", lambda: False)
    # Same host-independent branch selection; this case asserts the "launchctl unavailable" answer.
    monkeypatch.setattr(gateway_cli, "is_macos", lambda: True)  # os-marker: ok — asserts the launchctl-unavailable answer
    monkeypatch.setattr(gateway_cli, "get_launchd_label", lambda: "ai.hermes.gateway")
    monkeypatch.setattr(gateway_cli, "get_launchd_plist_path", lambda: plist)

    assert gateway_cli._is_service_running() is False
