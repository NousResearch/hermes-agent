"""App Installer checker output contracts and native WinRT dependency coverage.

Stubbed subprocess checks cover OS outcomes without a packaged process.
The native Windows test checks the URI and async projection types.

Root causes pinned here (audit C08):
- ``Package.current`` is a PROPERTY, not a callable.
- ``PackageManager.check_package_update_availability_async`` does not exist;
  the method lives on the Package instance.
- A not-packaged process is identifiable by the package-identity HRESULT
  (0x80073D54); ANY other failure must surface as unknown, never as
  "no update".
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


HERMES_PYTHON = sys.executable
SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "apps/desktop/scripts/check-appinstaller-update.py"
)

@pytest.mark.parametrize("state,code,available", [
    ("not-packaged", 0, False), ("AVAILABLE", 2, True), ("REQUIRED", 2, True),
    ("NO_UPDATES", 0, False), ("UNKNOWN", 1, None), ("ERROR", 1, None),
    ("no-source", 1, None), ("check-error", 1, None), ("identity-error", 1, None),
    ("missing-projection", 1, None), ("missing-current", 1, None),
])
def test_projection_outcomes(monkeypatch, capsys, state, code, available):
    import enum
    import runpy
    from types import SimpleNamespace

    main = runpy.run_path(str(SCRIPT))["main"]
    availability = enum.IntEnum("Availability", "UNKNOWN NO_UPDATES AVAILABLE REQUIRED ERROR", start=0)
    class PackageInstance:
        def get_app_installer_info(self):
            return None if state == "no-source" else SimpleNamespace(
                uri=SimpleNamespace(absolute_uri="https://registered.example/updates.appinstaller"))

        def check_update_availability_async(self):
            def get():
                if state == "check-error":
                    raise RuntimeError("rpc boom")
                return SimpleNamespace(availability=availability[state])
            return SimpleNamespace(get=get)

    class PackageMeta(type):
        @property
        def current(cls):
            if state in ("not-packaged", "identity-error"):
                error = OSError("identity failure")
                error.winerror = -2147009196 if state == "not-packaged" else -2147024891
                raise error
            return PackageInstance()

    class Package(metaclass=PackageMeta):
        pass

    def load():
        if state == "missing-projection":
            raise ImportError("missing winrt")
        return (object if state == "missing-current" else Package), availability

    monkeypatch.setitem(main.__globals__, "_load_projection", load)
    assert main() == code
    payload = json.loads(capsys.readouterr().out)
    assert payload["available"] is available
    if state == "not-packaged":
        assert payload == {"available": False, "reason": "not-packaged"}
    elif state in ("AVAILABLE", "REQUIRED", "NO_UPDATES"):
        assert payload["availability"] == state
        assert payload["source_uri"] == "https://registered.example/updates.appinstaller"
    else:
        assert payload["error"]
        if state == "no-source":
            assert payload["reason"] == "no-app-installer-source"
            assert ".appinstaller" in payload["error"]


def test_script_import_failure_json_and_exit(tmp_path):
    # A real subprocess executes __main__ and imports the absent projection.
    (tmp_path / "winrt.py").write_text("raise ImportError('fixture unavailable')", encoding="utf-8")
    child = subprocess.run([HERMES_PYTHON, str(SCRIPT)], capture_output=True,
                           text=True, timeout=30, env={**os.environ, "PYTHONPATH": str(tmp_path)})
    assert child.returncode == 1, child.stderr
    payload = json.loads(child.stdout)
    assert payload["available"] is None
    assert "fixture unavailable" in payload["error"]


@pytest.mark.platforms("windows")
def test_installed_winrt_projects_checker_uri_and_async_types(tmp_path):
    # A dev process has no package identity, so exercise the types that the
    # packaged update call projects only after Package.current succeeds.
    probe = subprocess.run(
        [HERMES_PYTHON, "-I", "-c",
         "import runpy; "
         "from winrt.windows.foundation import IAsyncOperation, Uri; "
         "uri = Uri('https://example.invalid/updates.appinstaller'); "
         "assert uri.absolute_uri == 'https://example.invalid/updates.appinstaller'; "
         "assert callable(IAsyncOperation.get); "
         f"runpy.run_path({str(SCRIPT)!r})['_load_projection'](); "
         "print('WINRT_CHECKER_TYPES_OK')"],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
    )
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.strip() == "WINRT_CHECKER_TYPES_OK"
