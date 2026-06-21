"""Regression: ``GET /api/status`` must be side-effect-free.

The dashboard SPA polls ``/api/status`` every few seconds, and the endpoint is
in ``PUBLIC_API_PATHS`` (an unauthenticated liveness probe). It previously
resolved gateway config on every call, which ran each plugin adapter's
``check_fn`` — and adapter ``check_fn``s lazy-install their SDK via
``tools.lazy_deps.ensure`` on first call. In a sealed venv (NixOS / uv venv
with no ``pip``) that install can never persist, so it re-ran the
pip → ensurepip → ``pip install`` ladder on every poll, stalling the UI for
seconds per request.

These tests assert the status path never shells out to a package installer.
"""

from __future__ import annotations

import subprocess

import pytest
from fastapi.testclient import TestClient

from hermes_cli import web_server


def _argv_text(cmd) -> str:
    if isinstance(cmd, (list, tuple)):
        return " ".join(str(part) for part in cmd)
    return str(cmd)


def _looks_like_install(cmd) -> bool:
    text = _argv_text(cmd)
    if "ensurepip" in text:
        return True
    return "pip" in text and ("install" in text or "--version" in text)


@pytest.fixture
def install_tripwire(monkeypatch):
    """Make any pip/ensurepip subprocess a hard test failure."""
    calls: list[str] = []
    real_run = subprocess.run
    real_popen = subprocess.Popen

    def guarded_run(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("args")
        if _looks_like_install(cmd):
            calls.append(_argv_text(cmd))
            raise AssertionError(f"unexpected installer subprocess: {_argv_text(cmd)}")
        return real_run(*args, **kwargs)

    def guarded_popen(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("args")
        if _looks_like_install(cmd):
            calls.append(_argv_text(cmd))
            raise AssertionError(f"unexpected installer subprocess: {_argv_text(cmd)}")
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", guarded_run)
    monkeypatch.setattr(subprocess, "Popen", guarded_popen)
    return calls


@pytest.fixture
def loopback_client(monkeypatch):
    prev_host = getattr(web_server.app.state, "bound_host", None)
    prev_port = getattr(web_server.app.state, "bound_port", None)
    prev_required = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.bound_host = "127.0.0.1"
    web_server.app.state.bound_port = 8080
    web_server.app.state.auth_required = False
    client = TestClient(web_server.app, base_url="http://127.0.0.1:8080")
    try:
        yield client
    finally:
        web_server.app.state.bound_host = prev_host
        web_server.app.state.bound_port = prev_port
        web_server.app.state.auth_required = prev_required


def test_status_does_not_trigger_lazy_install(loopback_client, install_tripwire):
    """A status poll must not shell out to pip/ensurepip/uv install."""
    resp = loopback_client.get("/api/status")
    assert resp.status_code == 200
    assert install_tripwire == [], (
        "/api/status triggered an installer subprocess: " + ", ".join(install_tripwire)
    )


def test_load_gateway_config_does_not_trigger_lazy_install(install_tripwire):
    """The config-resolution path status depends on stays install-free."""
    from gateway.config import load_gateway_config

    load_gateway_config()
    assert install_tripwire == [], (
        "load_gateway_config triggered an installer subprocess: "
        + ", ".join(install_tripwire)
    )


def test_installs_suppressed_makes_ensure_refuse_instead_of_installing(
    install_tripwire,
):
    """``installs_suppressed()`` converts a missing feature to a read-only refusal.

    Inside the scope, ``ensure`` for an unsatisfiable feature raises
    ``FeatureUnavailable`` (which adapter ``check_fn``s already treat as "not
    available") rather than shelling out to an installer.
    """
    from tools import lazy_deps

    # Pick any real feature whose deps are not satisfied in the test venv so we
    # exercise the install branch; if everything happens to be installed there
    # is nothing to assert about suppression, so skip.
    feature = next(
        (f for f in lazy_deps.LAZY_DEPS if lazy_deps.feature_missing(f)),
        None,
    )
    if feature is None:
        pytest.skip("no lazy feature is missing in this venv")

    with lazy_deps.installs_suppressed():
        with pytest.raises(lazy_deps.FeatureUnavailable):
            lazy_deps.ensure(feature, prompt=False)

    assert install_tripwire == [], "suppressed ensure() still shelled out to installer"
