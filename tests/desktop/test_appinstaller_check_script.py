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

_NO_PACKAGE_IDENTITY_HRESULT = -2147009196  # 0x80073D54


def _stub_module(*, identity_error: bool, availability: str | None, check_raises: bool) -> str:
    current_body = (
        "        exc = OSError('The process has no package identity.')\n"
        f"        exc.winerror = {_NO_PACKAGE_IDENTITY_HRESULT}\n"
        "        raise exc"
        if identity_error
        else "        return _FakePackage()"
    )
    get_body = (
        '        raise RuntimeError("rpc boom")'
        if check_raises
        else "        return _AvailabilityResult(self._availability)"
    )
    check_body = (
        '        raise RuntimeError("winrt blew up")'
        if check_raises
        else f"        return _AsyncOp({availability!r})"
    )
    return "\n".join(
        [
            "import enum",
            "",
            "class PackageUpdateAvailability(enum.Enum):",
            '    UNKNOWN = "unknown"',
            '    NO_UPDATES = "noUpdates"',
            '    AVAILABLE = "available"',
            '    REQUIRED = "required"',
            '    ERROR = "error"',
            "",
            "class _AvailabilityResult:",
            "    def __init__(self, availability):",
            "        self.availability = availability",
            "",
            "class _AsyncOp:",
            "    def __init__(self, availability):",
            "        self._availability = availability",
            "",
            "    def get(self):",
            get_body,
            "",
            "class _FakePackage:",
            "    def get_app_installer_info(self):",
            "        from types import SimpleNamespace",
            "        return SimpleNamespace(uri=SimpleNamespace(absolute_uri='https://registered.example/updates.appinstaller'))",
            "    def check_update_availability_async(self):",
            check_body,
            "",
            "class _PackageMeta(type):",
            "    @property",
            "    def current(cls):",
            current_body,
            "",
            "class Package(metaclass=_PackageMeta):",
            "    pass",
            "",
        ]
    )


def _stub_winrt(tmp_path: Path, package_module: str) -> Path:
    """Render a stub winrt tree whose Package comes from ``package_module``.
    PackageManager deliberately has NO check_package_update_availability_async:
    the real projection's shape is part of the contract under test."""
    root = tmp_path / "stub"
    appmodel = root / "winrt/windows/applicationmodel"
    deployment = root / "winrt/windows/management/deployment"
    for d in (appmodel, deployment):
        d.mkdir(parents=True)
    (root / "winrt/__init__.py").write_text("")
    (root / "winrt/windows/__init__.py").write_text("")
    (root / "winrt/windows/management/__init__.py").write_text("")
    (deployment / "__init__.py").write_text("from .deployment import PackageManager\n")
    (deployment / "deployment.py").write_text(
        "class PackageManager:\n"
        "    # No check_package_update_availability_async — it does not exist in\n"
        "    # the real projection; the method lives on Package instances.\n"
        "    pass\n"
    )
    (appmodel / "__init__.py").write_text(package_module)
    return root


def _run(root: Path) -> tuple[int, dict]:
    env = {"PYTHONPATH": str(root)}
    if os.environ.get("SYSTEMROOT"):
        env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    proc = subprocess.run(
        [HERMES_PYTHON, str(SCRIPT)],
        capture_output=True, text=True, timeout=60, env=env,
    )
    return proc.returncode, json.loads(proc.stdout.strip().splitlines()[-1])


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


def test_not_packaged_is_reported_as_not_packaged(tmp_path):
    root = _stub_winrt(tmp_path, _stub_module(identity_error=True, availability=None, check_raises=False))
    code, payload = _run(root)
    assert code == 0
    assert payload == {"available": False, "reason": "not-packaged"}


def test_update_available_exit_2(tmp_path):
    root = _stub_winrt(tmp_path, _stub_module(identity_error=False, availability="available", check_raises=False))
    code, payload = _run(root)
    assert code == 2
    assert payload["available"] is True
    assert payload["availability"] == "AVAILABLE"
    assert payload["source_uri"] == "https://registered.example/updates.appinstaller"


def test_no_update_exit_0(tmp_path):
    root = _stub_winrt(tmp_path, _stub_module(identity_error=False, availability="noUpdates", check_raises=False))
    code, payload = _run(root)
    assert code == 0
    assert payload["available"] is False


def test_unknown_availability_is_not_no_update(tmp_path):
    """UNKNOWN (enum 0) is an unknown, never available:false (C06-09 review)."""
    root = _stub_winrt(tmp_path, _stub_module(identity_error=False, availability="unknown", check_raises=False))
    code, payload = _run(root)
    assert code == 1
    assert payload["available"] is None
    assert "error" in payload


def test_error_availability_is_not_no_update(tmp_path):
    """ERROR (enum 4) is an unknown with an extended error, never available:false."""
    root = _stub_winrt(tmp_path, _stub_module(identity_error=False, availability="error", check_raises=False))
    code, payload = _run(root)
    assert code == 1
    assert payload["available"] is None
    assert "error" in payload


def test_missing_appinstaller_source_explains_the_remedy(tmp_path):
    module = _stub_module(identity_error=False, availability="unknown", check_raises=False).replace(
        "return SimpleNamespace(uri=SimpleNamespace(absolute_uri='https://registered.example/updates.appinstaller'))",
        "return None",
    )
    root = _stub_winrt(tmp_path, module)
    code, payload = _run(root)
    assert code == 1
    assert payload["available"] is None
    assert payload["reason"] == "no-app-installer-source"
    assert ".appinstaller" in payload["error"]


def test_check_failure_is_unknown_not_no_update(tmp_path):
    root = _stub_winrt(tmp_path, _stub_module(identity_error=False, availability=None, check_raises=True))
    code, payload = _run(root)
    assert code == 1
    assert payload["available"] is None
    assert "error" in payload


def test_unexpected_package_identity_error_is_unknown(tmp_path):
    """A non-identity failure while reading the current package is a real
    error — it must never masquerade as not-packaged (available:false)."""
    module = _stub_module(identity_error=False, availability="available", check_raises=False).replace(
        "        return _FakePackage()",
        '        raise OSError(5, "access denied", None, -2147024891)',
    )
    root = _stub_winrt(tmp_path, module)
    code, payload = _run(root)
    assert code == 1
    assert payload["available"] is None


def test_projection_without_current_property_is_unknown(tmp_path):
    """A projection that does not expose the current-package static must
    degrade to unknown — the checker may not guess an API shape."""
    module = _stub_module(identity_error=False, availability="available", check_raises=False).replace(
        "class Package(metaclass=_PackageMeta):", "class PackageNotProjected(metaclass=_PackageMeta):"
    )
    root = _stub_winrt(tmp_path, module)
    code, payload = _run(root)
    assert code == 1
    assert payload["available"] is None
