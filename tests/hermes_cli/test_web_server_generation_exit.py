"""Dashboard dependency retirement uses the real admission fence and owning selection."""

from types import SimpleNamespace

import pytest

from hermes_cli.backend_retirement import RetirementFence
from hermes_cli.web_server_generation_exit import start_generation_watchdog
from tests.pm.generation_fixture import make_install


def test_supervised_dashboard_flag_is_explicit_and_legacy_default_is_unchanged():
    import argparse
    from hermes_cli.subcommands.dashboard import build_dashboard_parser

    parser = argparse.ArgumentParser()
    build_dashboard_parser(
        parser.add_subparsers(dest="command"),
        cmd_dashboard=lambda args: None,
        cmd_dashboard_register=lambda args: None,
    )
    assert not parser.parse_args(["dashboard", "--no-open"]).external_supervisor
    assert parser.parse_args(
        ["dashboard", "--no-open", "--external-supervisor"]
    ).external_supervisor


@pytest.mark.parametrize("state", ["busy", "approval", "unknown", "idle"])
def test_dashboard_preserves_work_until_fenced_idle_then_retires(
    tmp_path, monkeypatch, state
):
    from hermes_cli import web_server_idle_proof

    install = make_install(tmp_path, monkeypatch)
    install.publish("new")
    server, fence = SimpleNamespace(should_exit=False), RetirementFence()
    monkeypatch.setattr(
        web_server_idle_proof,
        "idle_proof",
        lambda: {"idle": None if state == "unknown" else state not in {"approval"}},
    )
    admitted = fence.acquire() if state == "busy" else False
    thread = start_generation_watchdog(
        server,
        observer=install.observer,
        fence=fence,
        owned=True,
        poll_s=0,
        max_polls=3,
    )
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert server.should_exit is (state == "idle")
    if admitted:
        fence.release()
    if state != "idle":
        monkeypatch.setattr(web_server_idle_proof, "idle_proof", lambda: {"idle": True})
        thread = start_generation_watchdog(
            server,
            observer=install.observer,
            fence=fence,
            owned=True,
            poll_s=0,
            max_polls=3,
        )
        thread.join(timeout=10)
        assert server.should_exit
    assert not fence.acquire()  # admission closes before the exit signal


def test_manual_dashboard_and_current_generation_do_not_retire(tmp_path, monkeypatch):
    install = make_install(tmp_path, monkeypatch)
    server = SimpleNamespace(should_exit=False)
    assert (
        start_generation_watchdog(server, observer=install.observer, owned=False)
        is None
    )
    thread = start_generation_watchdog(
        server, observer=install.observer, owned=True, poll_s=0, max_polls=2
    )
    thread.join(timeout=10)
    assert not server.should_exit


@pytest.mark.parametrize("record", ["[]", '{"generation": 5}'])
def test_malformed_runtime_selection_does_not_kill_dashboard_watchdog(
    tmp_path, monkeypatch, record
):
    import threading

    install = make_install(tmp_path, monkeypatch)
    runtime = install.state / "pm-runtime"
    (runtime / "generations" / "superseded").mkdir(parents=True)
    (runtime / "selected.json").write_text(record, encoding="utf-8")
    probes = []
    probe = install.observer.probe
    install.observer.probe = lambda: probes.append(probe()) or probes[-1]
    failures = []
    monkeypatch.setattr(threading, "excepthook", failures.append)
    thread = start_generation_watchdog(
        SimpleNamespace(should_exit=False),
        observer=install.observer,
        owned=True,
        poll_s=0,
        max_polls=2,
    )
    thread.join(timeout=10)
    assert not failures
    assert probes == ["", ""]  # the watchdog keeps polling after maintenance
