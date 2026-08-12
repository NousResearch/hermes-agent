"""Security-floor tests for the Google Workspace runtime installer."""

from __future__ import annotations

import importlib.util
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import pytest


SETUP_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills/productivity/google-workspace/scripts/setup.py"
)


@pytest.fixture()
def setup_module():
    spec = importlib.util.spec_from_file_location(
        "test_google_workspace_setup_module",
        SETUP_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stale_google_transitives_are_reported_missing(setup_module, monkeypatch):
    installed = {
        "google-api-python-client": "2.194.0",
        "google-auth": "2.55.0",
        "google-auth-oauthlib": "1.3.1",
        "google-auth-httplib2": "0.3.1",
        "httplib2": "0.31.2",
        "pyasn1": "0.6.3",
    }

    def fake_version(name):
        try:
            return installed[name]
        except KeyError:
            raise PackageNotFoundError(name) from None

    monkeypatch.setattr(setup_module, "_distribution_version", fake_version)

    assert setup_module._missing_required_packages() == [
        "google-auth==2.55.1",
        "httplib2==0.32.0",
        "pyasn1==0.6.4",
    ]


def test_installer_repairs_stale_transitives(setup_module, monkeypatch):
    states = iter(
        [
            [
                "google-auth==2.55.1",
                "httplib2==0.32.0",
                "pyasn1==0.6.4",
            ],
            [],
        ]
    )
    monkeypatch.setattr(
        setup_module,
        "_missing_required_packages",
        lambda: next(states),
    )
    calls = []
    monkeypatch.setattr(
        setup_module.subprocess,
        "check_call",
        lambda argv, **kwargs: calls.append(argv),
    )

    assert setup_module.install_deps() is True
    assert calls == [
        [
            setup_module.sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "google-auth==2.55.1",
            "httplib2==0.32.0",
            "pyasn1==0.6.4",
        ]
    ]
