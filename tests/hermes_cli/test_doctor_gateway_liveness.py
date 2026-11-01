"""``hermes doctor`` must fail when a configured gateway is not actually up (#139549).

Supervision checks only prove a unit was installed. Liveness is a live
``(pid, start_time)`` match against ``gateway.pid`` plus a TCP connect to every
configured A2A listener. A bare pid is not enough: the same number can belong
to a different process.
"""

from __future__ import annotations

import json
import os
import socket
from contextlib import contextmanager
from pathlib import Path

import pytest

from hermes_cli.doctor_gateway_liveness import _check_gateway_liveness
from hermes_cli.doctor_report import Finding


@pytest.fixture(autouse=True)
def _no_ambient_a2a_port(monkeypatch):
    monkeypatch.delenv("A2A_PORT", raising=False)


def _root() -> Path:
    from hermes_constants import get_default_hermes_root

    return get_default_hermes_root()


def _run(should_fix: bool = False) -> Finding:
    finding = Finding()
    _check_gateway_liveness.__wrapped__(should_fix, finding)
    return finding


def _write_env(home: Path, port: int) -> None:
    (home / ".env").write_text(f"A2A_PORT={port}\n", encoding="utf-8")


def _write_pid(pid: int, start_time: object | None) -> None:
    payload: dict = {"pid": pid}
    if start_time is not None:
        payload["start_time"] = start_time
    (_root() / "gateway.pid").write_text(json.dumps(payload), encoding="utf-8")


def _real_start() -> int:
    from gateway.status import get_process_start_time

    start = get_process_start_time(os.getpid())
    assert start is not None
    return start


@contextmanager
def _listening():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    try:
        yield sock.getsockname()[1]
    finally:
        sock.close()


def _closed_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def test_unconfigured_gateway_stays_quiet(capsys):
    finding = _run()

    assert finding.issues == []
    assert "Gateway Liveness" not in capsys.readouterr().out


def test_process_env_alone_does_not_configure_a2a(monkeypatch, capsys):
    # Under multiplex os.environ is the launch profile. A shell A2A_PORT must
    # not make every home look configured.
    monkeypatch.setenv("A2A_PORT", "9902")

    finding = _run()

    assert finding.issues == []
    assert "Gateway Liveness" not in capsys.readouterr().out


def test_explicit_disable_ignores_a2a_port(capsys):
    home = _root()
    (home / "config.yaml").write_text(
        "gateway:\n  platforms:\n    a2a:\n      enabled: false\n",
        encoding="utf-8",
    )
    _write_env(home, 9902)

    finding = _run()

    assert finding.issues == []
    assert "Gateway Liveness" not in capsys.readouterr().out


def test_missing_pidfile_fails_and_names_the_start_command(capsys):
    _write_env(_root(), _closed_port())

    finding = _run()
    out = capsys.readouterr().out

    assert "Gateway Liveness" in out
    assert "Gateway configured but unreachable" in out
    assert any("hermes gateway start" in issue for issue in finding.issues)
    assert "Offered fix" not in out


def test_fix_offers_gateway_start_without_claiming_it_ran(capsys):
    _write_env(_root(), _closed_port())

    finding = _run(should_fix=True)
    out = capsys.readouterr().out

    assert "Offered fix: hermes gateway start" in out
    assert finding.issues == ["hermes gateway start"]
    assert finding.fixed == 0


def test_bare_pid_fails_even_when_that_process_is_alive_and_the_port_is_open(capsys):
    with _listening() as port:
        _write_env(_root(), port)
        _write_pid(os.getpid(), None)

        finding = _run()
        out = capsys.readouterr().out

    assert finding.issues
    assert "start_time" in out
    assert "Gateway configured but unreachable" in out


def test_reused_pid_fails_when_start_time_does_not_match(capsys):
    with _listening() as port:
        _write_env(_root(), port)
        _write_pid(os.getpid(), _real_start() + 100_000)

        finding = _run()
        out = capsys.readouterr().out

    assert finding.issues
    assert "start_time does not match" in out


def test_matching_process_with_a_closed_port_fails(capsys):
    port = _closed_port()
    _write_env(_root(), port)
    _write_pid(os.getpid(), _real_start())

    finding = _run()
    out = capsys.readouterr().out

    assert finding.issues
    assert f"nothing listening on 127.0.0.1:{port}" in out
    assert "start_time does not match" not in out


def test_dead_pid_fails_even_if_the_port_is_open(capsys):
    with _listening() as port:
        _write_env(_root(), port)
        _write_pid(1 << 30, _real_start())

        finding = _run()
        out = capsys.readouterr().out

    assert finding.issues
    assert "is not running" in out


def test_matching_pid_start_time_and_open_port_passes(capsys):
    with _listening() as port:
        _write_env(_root(), port)
        _write_pid(os.getpid(), _real_start())

        finding = _run()
        out = capsys.readouterr().out

    assert finding.issues == []
    assert "Gateway Liveness" in out
    assert "pid and start_time match" in out


def test_yaml_extra_port_counts_as_configured(capsys):
    with _listening() as port:
        (_root() / "config.yaml").write_text(
            "gateway:\n  platforms:\n    a2a:\n      enabled: true\n"
            f"      extra:\n        port: {port}\n",
            encoding="utf-8",
        )
        _write_pid(os.getpid(), _real_start())

        finding = _run()

    assert finding.issues == []
    assert "pid and start_time match" in capsys.readouterr().out


def test_env_port_wins_over_a_closed_yaml_port(capsys):
    closed = _closed_port()
    with _listening() as port:
        (_root() / "config.yaml").write_text(
            "gateway:\n  platforms:\n    a2a:\n      enabled: true\n"
            f"      extra:\n        port: {closed}\n",
            encoding="utf-8",
        )
        _write_env(_root(), port)
        _write_pid(os.getpid(), _real_start())

        finding = _run()

    assert finding.issues == []


def test_enabled_without_a_port_probes_the_adapter_default(monkeypatch, capsys):
    with _listening() as port:
        monkeypatch.setattr(
            "hermes_cli.doctor_gateway_liveness._A2A_DEFAULT_PORT", port
        )
        (_root() / "config.yaml").write_text(
            "gateway:\n  platforms:\n    a2a:\n      enabled: true\n",
            encoding="utf-8",
        )
        _write_pid(os.getpid(), _real_start())

        finding = _run()

    assert finding.issues == []
    assert "pid and start_time match" in capsys.readouterr().out


def test_sibling_profile_closed_port_fails_the_host_gateway(capsys):
    other = _root() / "profiles" / "coder"
    other.mkdir(parents=True)
    closed = _closed_port()
    _write_env(other, closed)
    with _listening() as port:
        _write_env(_root(), port)
        _write_pid(os.getpid(), _real_start())

        finding = _run()
        out = capsys.readouterr().out

    assert finding.issues
    assert f"nothing listening on 127.0.0.1:{closed}" in out


def test_legacy_bare_integer_pidfile_is_not_a_live_gateway(capsys):
    with _listening() as port:
        _write_env(_root(), port)
        (_root() / "gateway.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")

        finding = _run()
        out = capsys.readouterr().out

    assert finding.issues
    assert "start_time" in out


def test_doctor_registers_the_liveness_check_beside_supervision():
    from hermes_cli.doctor import DOCTOR_CHECKS
    from hermes_cli.doctor_platform import _check_gateway_supervision

    checks = [check for _title, check in DOCTOR_CHECKS]
    assert checks.index(_check_gateway_liveness) == checks.index(_check_gateway_supervision) + 1
