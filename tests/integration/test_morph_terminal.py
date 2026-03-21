"""Integration tests for the Morph terminal backend.

Requires:
    TERMINAL_ENV=morph
    MORPH_API_KEY=...
    TERMINAL_MORPH_IMAGE_ID=<Morph base image ID>  # optional, defaults to morphvm-minimal

Run with:
    TERMINAL_ENV=morph pytest tests/integration/test_morph_terminal.py -v
"""

import json
import os
import sys
import uuid
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.integration

if not os.getenv("MORPH_API_KEY"):
    pytest.skip("MORPH_API_KEY not set", allow_module_level=True)

try:
    import morphcloud  # noqa: F401
except ImportError:
    pytest.skip("morphcloud not installed", allow_module_level=True)

import importlib.util
import signal

parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))
tools_pkg = ModuleType("tools")
tools_pkg.__path__ = [str(parent_dir / "tools")]
sys.modules["tools"] = tools_pkg

spec = importlib.util.spec_from_file_location(
    "terminal_tool", parent_dir / "tools" / "terminal_tool.py"
)
terminal_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(terminal_module)

terminal_tool = terminal_module.terminal_tool
cleanup_vm = terminal_module.cleanup_vm
RUN_TOKEN = uuid.uuid4().hex[:8]


def _integration_timeout_handler(signum, frame):
    raise TimeoutError("Test exceeded 240 second timeout")


@pytest.fixture(autouse=True)
def _force_morph(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "morph")
    monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "false")


@pytest.fixture(autouse=True)
def _enforce_test_timeout():
    old = signal.signal(signal.SIGALRM, _integration_timeout_handler)
    signal.alarm(240)
    yield
    signal.alarm(0)
    signal.signal(signal.SIGALRM, old)


def _run(command, task_id, **kwargs):
    kwargs.setdefault("timeout", 180)
    kwargs.setdefault("workdir", "/root")
    return json.loads(terminal_tool(command, task_id=task_id, **kwargs))


def _task_id(name: str) -> str:
    return f"{name}_{RUN_TOKEN}"


def test_echo_roundtrip():
    task_id = _task_id("morph_test_echo")
    try:
        result = _run("echo 'Hello from Morph!'", task_id)
        assert result["exit_code"] == 0, result
        assert "Hello from Morph!" in result["output"]
    finally:
        cleanup_vm(task_id)


def test_nonzero_exit_propagates():
    task_id = _task_id("morph_test_nonzero")
    try:
        result = _run("exit 42", task_id)
        assert result["exit_code"] == 42, result
    finally:
        cleanup_vm(task_id)


def test_persistent_workspace_reattaches_after_cleanup(monkeypatch):
    task_id = _task_id("morph_test_persist")
    monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "true")
    try:
        first = _run("echo 'survive' > /tmp/morph_persist.txt", task_id)
        assert first["exit_code"] == 0, first

        cleanup_vm(task_id)

        second = _run("cat /tmp/morph_persist.txt", task_id)
        assert second["exit_code"] == 0, second
        assert "survive" in second["output"]
    finally:
        monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "false")
        cleanup_vm(task_id)


def test_task_ids_are_isolated():
    task_a = _task_id("morph_test_iso_a")
    task_b = _task_id("morph_test_iso_b")
    try:
        first = _run("echo 'secret' > /tmp/morph_isolated.txt", task_a)
        assert first["exit_code"] == 0, first

        second = _run(
            "cat /tmp/morph_isolated.txt 2>/dev/null || echo NOT_FOUND",
            task_b,
        )
        assert second["exit_code"] == 0, second
        assert "NOT_FOUND" in second["output"]
        assert "secret" not in second["output"]
    finally:
        cleanup_vm(task_a)
        cleanup_vm(task_b)


def test_timeout_returns_timeout_result():
    task_id = _task_id("morph_test_timeout")
    try:
        result = _run("sleep 30", task_id, timeout=3)
        assert result["exit_code"] == 124, result
        assert result["error"] is None, result
    finally:
        cleanup_vm(task_id)
