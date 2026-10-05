"""`hermes gateway install --no-start-now` and the setup wizard's "Start the gateway now?" = No on launchd.

The plist carries RunAtLoad, so loading it (`launchctl bootstrap`) starts the gateway on the spot. A
no-start install must therefore write the plist without loading it; `hermes gateway start` loads it later.
"""
import argparse
import plistlib
import subprocess
from types import SimpleNamespace

import pytest

import hermes_cli.gateway as gateway_cli
from hermes_cli.subcommands.gateway import build_gateway_parser

LABEL = "ai.hermes.gateway"
DOMAIN = "gui/501"


def _parse(*argv):
    parser = argparse.ArgumentParser()
    build_gateway_parser(
        parser.add_subparsers(dest="command"),
        cmd_gateway=lambda a: None, cmd_proxy=lambda a: None, cmd_gateway_enroll=lambda a: None,
    )
    return parser.parse_args(["gateway", *argv])


@pytest.fixture
def launchd(tmp_path, monkeypatch):
    """launchd backend with every launchctl call recorded instead of run."""
    state = SimpleNamespace(
        plist=tmp_path / "LaunchAgents" / f"{LABEL}.plist",
        supervised_pid=None, bootstraps=[], launchctl=[], refreshes=[], starts=[],
    )
    monkeypatch.setattr(gateway_cli, "_service_backend", lambda *a, **k: "launchd")
    monkeypatch.setattr(gateway_cli, "is_managed", lambda: False)
    monkeypatch.setattr(gateway_cli, "is_termux", lambda: False)
    monkeypatch.setattr(gateway_cli, "_guard_named_profile_under_multiplexer", lambda **k: None)
    monkeypatch.setattr(gateway_cli, "get_launchd_plist_path", lambda: state.plist)
    monkeypatch.setattr(gateway_cli, "get_launchd_label", lambda: LABEL)
    monkeypatch.setattr(gateway_cli, "_launchd_domain", lambda: DOMAIN)
    monkeypatch.setattr(gateway_cli, "_refuse_temp_home_service_write", lambda *a: False)
    monkeypatch.setattr(gateway_cli, "_clear_launchd_unsupported_marker", lambda: None)
    monkeypatch.setattr(gateway_cli, "_launchctl_supervised_pid", lambda label: state.supervised_pid)
    monkeypatch.setattr(gateway_cli, "_launchctl_bootstrap", lambda *a, **k: state.bootstraps.append(a))
    monkeypatch.setattr(gateway_cli, "refresh_launchd_plist_if_needed", lambda: state.refreshes.append(1) or True)
    monkeypatch.setattr(gateway_cli, "_setup_service_action", lambda *a, **k: state.starts.append(a))

    def fake_run(cmd, *a, **k):
        state.launchctl.append(cmd)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return state


BOOTOUT = ["launchctl", "bootout", f"{DOMAIN}/{LABEL}"]


@pytest.mark.parametrize(
    "stale_plist, supervised_pid, reloads",
    [(False, None, False), (True, None, False), (True, 4242, True)],
    ids=["fresh-install", "repair-stopped-gateway", "repair-running-gateway"],
)
def test_cli_no_start_now_loads_only_a_gateway_launchd_already_runs(
    launchd, capsys, stale_plist, supervised_pid, reloads,
):
    """Nothing running: the plist is written, never loaded, and an idle registration left from before
    (a parked clean exit) is booted out, since that is what `gateway start` would kickstart. A gateway
    launchd already runs is reloaded as before, not stopped."""
    if stale_plist:
        launchd.plist.parent.mkdir(parents=True)
        launchd.plist.write_text("<plist>old</plist>", encoding="utf-8")
    launchd.supervised_pid = supervised_pid

    gateway_cli.gateway_command(_parse("install", "--no-start-now"))

    assert launchd.bootstraps == []
    if reloads:
        assert launchd.refreshes == [1]
        assert BOOTOUT not in launchd.launchctl
    else:
        assert launchd.refreshes == []
        assert BOOTOUT in launchd.launchctl
        assert plistlib.loads(launchd.plist.read_bytes())["Label"] == LABEL
        assert "not started" in capsys.readouterr().out


def test_wizard_no_to_start_now_installs_without_loading(launchd, monkeypatch):
    answers = iter([False, True])  # "Start the gateway now?" No, "on login" Yes
    monkeypatch.setattr(gateway_cli, "prompt_yes_no", lambda *a, **k: next(answers))
    monkeypatch.setattr(gateway_cli, "is_wsl", lambda: False)

    gateway_cli._wizard_install_service("launchd")

    assert launchd.plist.exists()
    assert launchd.bootstraps == []
    assert launchd.starts == []
