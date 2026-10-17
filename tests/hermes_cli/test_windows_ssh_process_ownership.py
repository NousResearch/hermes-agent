"""Exercise Desktop SSH ownership against real Python child processes."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import psutil


_SOURCE = Path(__file__).resolve().parents[2] / "hermes_cli" / "windows_ssh_runtime.py"
_SPEC = importlib.util.spec_from_file_location("windows_ssh_runtime_ownership_test", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
_RUNTIME = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RUNTIME)
_NONCE = "0123456789abcdef"


def _process_state(*, isolated=True, module="hermes_cli.main", expected_nonce=_NONCE,
                   creation_offset=0, serve_isolated=True):
    code = f"import time; time.sleep(30)  # {module}"
    args = [sys.executable, *(["-I"] if isolated else []), "-c", code, "serve"]
    if serve_isolated:
        args.append("--isolated")
    args.extend(["--ssh-owner-nonce", _NONCE])
    child = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
    try:
        created = int(psutil.Process(child.pid).create_time() * 1_000_000_000)
        return _RUNTIME.process_state(child.pid, created + creation_offset,
                                      str(_SOURCE.parent / "hermes.exe"), expected_nonce)
    finally:
        child.terminate()
        child.wait(timeout=10)


def test_isolated_store_python_bootstrap_is_owned():
    state = _process_state()
    assert state["alive"] is True
    assert state["owned"] is True
    assert state["reason"] == "owned"


def test_plain_python_bootstrap_remains_owned():
    assert _process_state(isolated=False)["owned"] is True


def test_other_python_module_is_not_owned():
    assert _process_state(module="other.module")["owned"] is False


def test_wrong_owner_nonce_is_not_owned():
    assert _process_state(expected_nonce="fedcba9876543210")["owned"] is False


def test_wrong_creation_time_is_not_owned():
    state = _process_state(creation_offset=1)
    assert state["owned"] is False
    assert state["reason"] == "creation-time"


def test_nonisolated_serve_is_not_owned():
    assert _process_state(serve_isolated=False)["owned"] is False
