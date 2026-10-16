"""Store update contracts use the actual API's sequence and result shapes."""
from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
MODULE = "hermes_cli.windows_store_update"


class State(IntEnum):
    DOWNLOADING = 1
    COMPLETED = 3
    CANCELED = 4
    OTHER_ERROR = 5


class Operation:
    def __init__(self, result):
        self.result = result
        self.progress = None


class Context:
    def __init__(self, updates, state=State.COMPLETED):
        self.updates = updates
        self.state = state
        self.calls = []

    def get_app_and_optional_store_package_updates_async(self):
        self.calls.append("check")
        return Operation(self.updates)

    def request_download_store_package_updates_async(self, updates):
        assert updates is self.updates
        self.calls.append("download")
        return Operation(SimpleNamespace(overall_state=self.state))

    def request_download_and_install_store_package_updates_async(self, updates):
        assert updates is self.updates
        self.calls.append("install")
        return Operation(SimpleNamespace(overall_state=self.state))


@pytest.fixture
def checker():
    return importlib.import_module(MODULE)


@pytest.mark.parametrize("mode", ["check", "download", "install"])
@pytest.mark.parametrize("state", [State.COMPLETED, State.CANCELED, State.DOWNLOADING])
def test_store_operation_passes_update_sequence_and_requires_completed(checker, mode, state):
    updates = [SimpleNamespace(package=SimpleNamespace(id=SimpleNamespace(full_name="package-v2")))]
    context = Context(updates, state)
    result = checker.perform(context, mode, lambda operation: operation.result, State.COMPLETED)
    assert context.calls == (["check"] if mode == "check" else ["check", mode])
    assert result["available"] is True
    assert result["packages"] == ["package-v2"]
    if mode != "check":
        assert result["ok"] is (state == State.COMPLETED)
        assert ("error" in result) is (state != State.COMPLETED)
    empty = Context([])
    assert checker.perform(empty, mode, lambda operation: operation.result, State.COMPLETED) == {
        "available": False, "packages": [], "ok": True,
    }
    assert empty.calls == ["check"]


@pytest.mark.platforms("windows")
def test_native_projection_has_required_update_contract(tmp_path):
    code = (
        "from winrt.runtime import init_apartment,uninit_apartment,ApartmentType; "
        "from winrt.runtime.interop import initialize_with_window; "
        "from winrt.windows.services.store import StoreContext,StorePackageUpdateState; "
        "from winrt.windows.foundation import IAsyncOperation, IAsyncOperationWithProgress; "
        "import winrt.windows.foundation.collections,win32gui; "
        "init_apartment(ApartmentType.SINGLE_THREADED); "
        "c=StoreContext.get_default(); "
        "assert callable(c.get_app_and_optional_store_package_updates_async); "
        "assert callable(c.request_download_store_package_updates_async); "
        "assert callable(c.request_download_and_install_store_package_updates_async); "
        "assert callable(initialize_with_window); "
        "assert StorePackageUpdateState.COMPLETED != StorePackageUpdateState.DOWNLOADING; "
        "assert callable(IAsyncOperation.get); "
        "assert callable(IAsyncOperationWithProgress.get); "
        "op=c.get_app_and_optional_store_package_updates_async(); "
        "assert callable(op.get_results); op.cancel(); del op; "
        "assert callable(win32gui.PumpWaitingMessages); "
        "del c; uninit_apartment(); print('store projection ready')"
    )
    result = subprocess.run([sys.executable, "-I", "-c", code], cwd=tmp_path,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "store projection ready"


def test_entry_reports_json_when_the_projection_is_unavailable(tmp_path):
    # The desktop spawns `python -P -m <module>` and parses stdout as JSON; a
    # failure inside the checker must still reach it as JSON, never as nothing.
    (tmp_path / "winrt.py").write_text("raise ImportError('fixture unavailable')", encoding="utf-8")
    child = subprocess.run([sys.executable, "-P", "-m", MODULE, "--mode", "check"], capture_output=True,
                           text=True, timeout=30,
                           env={**os.environ, "PYTHONPATH": os.pathsep.join([str(REPO), str(tmp_path)])})
    assert child.returncode == 1, child.stderr
    payload = json.loads(child.stdout)
    assert payload["ok"] is False and payload["available"] is None
    assert "fixture unavailable" in payload["error"]
