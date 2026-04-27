"""Integration tests for the OpenShell sandbox provider.

Run manually with:
    HERMES_TEST_OPENSHELL=1 pytest -o addopts='' tests/integration/test_openshell_sandbox.py -q
"""

from __future__ import annotations

import json
import os
import signal
import sys
import uuid

import pytest

pytestmark = pytest.mark.integration

if os.getenv("HERMES_TEST_OPENSHELL") != "1":
    pytest.skip("HERMES_TEST_OPENSHELL=1 not set", allow_module_level=True)


@pytest.fixture(autouse=True)
def _allow_slow_openshell_startup():
    """OpenShell image builds/gateway startup can exceed the repo's 30s unit-test alarm."""

    if sys.platform != "win32":
        signal.alarm(0)
    yield
    if sys.platform != "win32":
        signal.alarm(0)


@pytest.fixture()
def openshell_task(monkeypatch, tmp_path):
    from plugins.sandbox.openshell.environment import (
        build_sandbox_name,
        bundled_source_dir,
        ensure_openshell_gateway,
        normalize_openshell_config,
        run_openshell_cli,
    )
    from tools.terminal_tool import cleanup_vm

    scope = f"it-{uuid.uuid4().hex[:10]}"
    task_id = f"openshell_{scope}"
    source = str(bundled_source_dir())
    gateway = "hermes-openshell-test"
    gateway_port = int(os.getenv("HERMES_TEST_OPENSHELL_GATEWAY_PORT", "18080"))

    monkeypatch.setenv("HERMES_SANDBOX", "openshell")
    monkeypatch.setenv("HERMES_SANDBOX_MODE", "mirror")
    monkeypatch.setenv("HERMES_SANDBOX_SCOPE", scope)
    monkeypatch.setenv("HERMES_SANDBOX_SOURCE", source)
    monkeypatch.setenv("HERMES_OPENSHELL_GATEWAY", gateway)
    monkeypatch.setenv("HERMES_OPENSHELL_GATEWAY_PORT", str(gateway_port))
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    monkeypatch.delenv("TERMINAL_ENV", raising=False)

    cfg = normalize_openshell_config(
        {
            "source": source,
            "gateway": gateway,
            "gateway_port": gateway_port,
        }
    )
    started = ensure_openshell_gateway(cfg)
    if started is not None and started.returncode != 0:
        detail = (started.stderr or started.stdout or "").strip()
        pytest.skip(f"OpenShell gateway unavailable: {detail[:500]}")

    yield task_id, scope, tmp_path

    cleanup_vm(task_id)
    run_openshell_cli(cfg, ["sandbox", "delete", build_sandbox_name(scope)], timeout=120)


def _run_terminal(command: str, task_id: str) -> dict:
    from tools.terminal_tool import terminal_tool

    return json.loads(terminal_tool(command, task_id=task_id, timeout=120))


def test_openshell_terminal_runs_in_sandbox(openshell_task):
    task_id, _scope, _tmp_path = openshell_task

    result = _run_terminal("pwd", task_id)

    assert result["exit_code"] == 0
    assert "/sandbox" in result["output"]


def test_openshell_file_tools_mirror_back_to_host(openshell_task):
    from tools.file_tools import read_file_tool, write_file_tool

    task_id, _scope, tmp_path = openshell_task

    write_result = json.loads(write_file_tool("/sandbox/openshell-file.txt", "hello from openshell\n", task_id=task_id))
    assert "error" not in write_result

    read_result = json.loads(read_file_tool("/sandbox/openshell-file.txt", task_id=task_id))
    assert "hello from openshell" in read_result.get("content", "")
    assert (tmp_path / "openshell-file.txt").read_text() == "hello from openshell\n"


def test_no_sandbox_forces_host_execution(monkeypatch, tmp_path):
    task_id = f"host_{uuid.uuid4().hex[:10]}"
    monkeypatch.setenv("HERMES_SANDBOX", "host")
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))

    result = _run_terminal("pwd", task_id)

    assert result["exit_code"] == 0
    assert str(tmp_path) in result["output"]
