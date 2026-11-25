"""Unit tests for SystemHost.gateway_verb() systemd timeouts (BM-7)."""

from __future__ import annotations

import subprocess

import pytest

from hermes_cli import gateway as gw
from hermes_cli import gateway_migrate_backends_apply as engine
from hermes_cli import gateway_migrate_backends_host as host_module
from hermes_cli.gateway_migrate_backends_host import SystemHost


def _systemd_host(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    unit = tmp_path / "units" / "hermes-gateway.service"
    unit.parent.mkdir()
    unit.write_text(
        f"[Service]\nEnvironment=HERMES_HOME={home}\nExecStart=/usr/bin/hermes gateway run\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(gw, "get_systemd_unit_path", lambda system=False: unit)
    monkeypatch.setattr(gw, "get_service_name", lambda: "hermes-gateway")
    planned_stops: list[bool] = []
    monkeypatch.setattr(gw, "_mark_planned_stop", lambda: planned_stops.append(True))

    calls: list[tuple[str, list[str], int]] = []

    def fake_systemctl(scope, args, *, timeout=host_module._TIMEOUT):
        calls.append((scope, list(args), timeout))
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(host_module, "_systemctl", fake_systemctl)

    class _Host(SystemHost):
        def gateway_kind(self):
            return "systemd"

    return _Host(home), calls, planned_stops


def test_gateway_verb_restart_timeout_covers_handoff_and_verify_budget(tmp_path, monkeypatch):
    host, calls, _ = _systemd_host(tmp_path, monkeypatch)

    host.gateway_verb("restart")

    assert len(calls) == 1
    scope, args, timeout = calls[0]
    assert scope == "user"
    assert args == ["restart", "hermes-gateway"]
    assert timeout == 540
    assert timeout >= int(getattr(engine, "RESTART_HANDOFF_SECONDS", 240.0) + engine.VERIFY_SECONDS)


def test_gateway_verb_start_timeout_covers_verify_budget(tmp_path, monkeypatch):
    host, calls, _ = _systemd_host(tmp_path, monkeypatch)

    host.gateway_verb("start")

    assert calls == [("user", ["start", "hermes-gateway"], 300)]


def test_gateway_verb_stop_marks_planned_stop_and_covers_handoff_budget(tmp_path, monkeypatch):
    host, calls, planned_stops = _systemd_host(tmp_path, monkeypatch)

    host.gateway_verb("stop")

    assert planned_stops == [True]
    assert calls == [("user", ["stop", "hermes-gateway"], 240)]


@pytest.mark.parametrize("verb", ["enable", "disable"])
def test_gateway_verb_non_lifecycle_verbs_keep_90s_timeout(tmp_path, monkeypatch, verb):
    host, calls, _ = _systemd_host(tmp_path, monkeypatch)

    host.gateway_verb(verb)

    assert calls == [("user", [verb, "hermes-gateway"], 90)]


def test_gateway_verb_timeout_derives_from_engine_budget_constants_with_90s_floor(tmp_path, monkeypatch):
    host, calls, _ = _systemd_host(tmp_path, monkeypatch)
    monkeypatch.setattr(engine, "RESTART_HANDOFF_SECONDS", 120.0, raising=False)
    monkeypatch.setattr(engine, "VERIFY_SECONDS", 180.0)

    host.gateway_verb("restart")
    host.gateway_verb("start")
    host.gateway_verb("stop")

    assert [timeout for _, _, timeout in calls] == [300, 180, 120]

    calls.clear()
    monkeypatch.setattr(engine, "RESTART_HANDOFF_SECONDS", 0.0, raising=False)
    monkeypatch.setattr(engine, "VERIFY_SECONDS", 0.0)

    host.gateway_verb("restart")
    host.gateway_verb("start")
    host.gateway_verb("stop")

    assert [timeout for _, _, timeout in calls] == [90, 90, 90]
