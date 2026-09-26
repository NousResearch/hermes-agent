"""#124214: `hermes doctor` must surface a broken web surface (dashboard).

fastapi/starlette pin drift kills the dashboard at import time with zero
visible output; doctor previously reported nothing. These tests drive the
new web-surface check: import failure ⇒ issue, pin drift ⇒ warning, healthy
pair ⇒ clean.
"""

import importlib
import importlib.metadata

import pytest
import zoneinfo  # noqa: F401  (local-env workaround, see 124033)

from hermes_cli import doctor_platform

_WEB_PINS = {
    "fastapi": "0.133.1",
    "uvicorn": "0.41.0",
    "starlette": "1.3.1",
    "python-multipart": "0.0.32",
}


_REAL_IMPORT = importlib.import_module


@pytest.fixture(autouse=True)
def _no_real_web_import(monkeypatch):
    """Guard: never let the test actually import the (heavy) web surface."""
    def _fake(name):
        if name == "hermes_cli.web_server":
            raise ImportError("test: web surface import is simulated")
        return _REAL_IMPORT(name)

    monkeypatch.setattr(importlib, "import_module", _fake)


class TestWebSurfaceImportFailure:
    def test_import_failure_is_reported(self, monkeypatch):
        def boom(name):
            if name == "hermes_cli.web_server":
                raise ImportError(
                    "Router.__init__() got an unexpected keyword argument "
                    "'on_startup' (fastapi/starlette pin drift)"
                )
            return _REAL_IMPORT(name)

        monkeypatch.setattr(importlib, "import_module", boom)
        f = doctor_platform._check_web_surface(False)
        assert f.issues, "import failure must produce an issue"
        assert any("web" in i.lower() or "dashboard" in i.lower() for i in f.issues)


class TestWebPinDrift:
    def test_pin_drift_is_reported(self, monkeypatch):
        def versions(pkg):
            return {"fastapi": "0.115.0", "starlette": "1.3.1"}.get(pkg, _WEB_PINS[pkg])

        monkeypatch.setattr("importlib.metadata.version", versions)
        f = doctor_platform._check_web_surface(False)
        assert f.issues or f.manual_issues
        assert any("fastapi" in i or "0.115" in i for i in f.issues + f.manual_issues)

    def test_clean_surface_passes(self, monkeypatch):
        def ok(name):
            return object() if name == "hermes_cli.web_server" else _REAL_IMPORT(name)

        monkeypatch.setattr(importlib, "import_module", ok)
        monkeypatch.setattr(
            "importlib.metadata.version", lambda pkg: _WEB_PINS[pkg]
        )
        f = doctor_platform._check_web_surface(False)
        assert not f.issues and not f.manual_issues

    def test_missing_web_extra_is_warned_not_crash(self, monkeypatch):
        def ok(name):
            return object() if name == "hermes_cli.web_server" else _REAL_IMPORT(name)

        monkeypatch.setattr(importlib, "import_module", ok)

        def versions(pkg):
            raise importlib.metadata.PackageNotFoundError(pkg)

        monkeypatch.setattr("importlib.metadata.version", versions)
        f = doctor_platform._check_web_surface(False)
        assert not f.issues  # absent-but-optional must not be a failure