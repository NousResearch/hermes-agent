"""Unit tests for SystemHost.gateway_verb() systemd timeouts (BM-7)."""

from __future__ import annotations

import itertools
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import gateway as gw
from hermes_cli import gateway_migrate_backends_apply as engine
from hermes_cli import gateway_migrate_backends_host as host_module
from hermes_cli import gateway_migrate_backends_timeouts as timeouts
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
    assert timeout >= timeouts.RESTART_HANDOFF_SECONDS + timeouts.VERIFY_SECONDS


def test_gateway_verb_start_timeout_covers_verify_budget(tmp_path, monkeypatch):
    host, calls, _ = _systemd_host(tmp_path, monkeypatch)

    host.gateway_verb("start")

    [(scope, args, timeout)] = calls
    assert (scope, args) == ("user", ["start", "hermes-gateway"]) and timeout >= timeouts.VERIFY_SECONDS


def test_gateway_verb_stop_marks_planned_stop_and_covers_handoff_budget(tmp_path, monkeypatch):
    host, calls, planned_stops = _systemd_host(tmp_path, monkeypatch)

    host.gateway_verb("stop")

    assert planned_stops == [True]
    [(scope, args, timeout)] = calls
    assert (scope, args) == ("user", ["stop", "hermes-gateway"]) and timeout >= timeouts.RESTART_HANDOFF_SECONDS


@pytest.mark.parametrize("verb", ["enable", "disable"])
def test_gateway_verb_non_lifecycle_verbs_keep_90s_timeout(tmp_path, monkeypatch, verb):
    host, calls, _ = _systemd_host(tmp_path, monkeypatch)

    host.gateway_verb(verb)

    assert calls == [("user", [verb, "hermes-gateway"], 90)]


def test_gateway_verb_timeout_derives_from_shared_budget_constants_with_90s_floor(tmp_path, monkeypatch):
    host, calls, _ = _systemd_host(tmp_path, monkeypatch)
    assert engine.RESTART_HANDOFF_SECONDS == timeouts.RESTART_HANDOFF_SECONDS
    assert engine.VERIFY_SECONDS == timeouts.VERIFY_SECONDS

    monkeypatch.setattr(timeouts, "RESTART_HANDOFF_SECONDS", 120.0)
    monkeypatch.setattr(timeouts, "VERIFY_SECONDS", 180.0)

    host.gateway_verb("restart")
    host.gateway_verb("start")
    host.gateway_verb("stop")

    assert [timeout for _, _, timeout in calls] == [300, 180, 120]

    calls.clear()
    monkeypatch.setattr(timeouts, "RESTART_HANDOFF_SECONDS", 0.0)
    monkeypatch.setattr(timeouts, "VERIFY_SECONDS", 0.0)

    host.gateway_verb("restart")
    host.gateway_verb("start")
    host.gateway_verb("stop")

    assert [timeout for _, _, timeout in calls] == [90, 90, 90]


def _subprocess_env() -> dict[str, str]:
    repo_root = str(Path(__file__).resolve().parents[3])
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{existing}" if existing else repo_root
    return env


def test_plan_and_host_do_not_import_apply_engine(tmp_path):
    """Importing plan/host and reading their timeout budgets never pulls in the apply engine."""
    script = (
        "import sys\n"
        "from hermes_cli.gateway_migrate_backends_plan import BackendPlan\n"
        "from hermes_cli.gateway_migrate_backends_host import _gateway_verb_timeout\n"
        f"plan = BackendPlan(home={str(tmp_path)!r}, new_host='0.0.0.0', new_port=9119)\n"
        "plan.step_titles()\n"
        "for verb in ('restart', 'start', 'stop'):\n"
        "    _gateway_verb_timeout(verb)\n"
        "assert 'hermes_cli.gateway_migrate_backends_apply' not in sys.modules\n"
    )
    res = subprocess.run(
        [sys.executable, "-c", script],
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )
    assert res.returncode == 0, res.stderr or res.stdout


def test_plan_host_apply_import_cleanly_in_any_order():
    """Importing plan, host, and apply in every permutation succeeds without an import cycle."""
    modules = (
        "hermes_cli.gateway_migrate_backends_plan",
        "hermes_cli.gateway_migrate_backends_host",
        "hermes_cli.gateway_migrate_backends_apply",
    )
    env = _subprocess_env()
    for order in itertools.permutations(modules):
        script = (
            "import importlib\n"
            f"for name in {order!r}:\n"
            "    importlib.import_module(name)\n"
        )
        res = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
        assert res.returncode == 0, f"order={order}: {res.stderr or res.stdout}"
