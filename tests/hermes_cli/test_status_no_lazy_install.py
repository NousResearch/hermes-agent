"""Regression: status / config-resolution paths must be side-effect-free.

The dashboard SPA polls ``/api/status`` every few seconds, and the endpoint is
in ``PUBLIC_API_PATHS`` (an unauthenticated liveness probe). It previously
resolved gateway config on every call, which ran each plugin adapter's
``check_fn`` — and adapter ``check_fn``s lazy-install their SDK via
``tools.lazy_deps.ensure`` on first call. In a sealed venv (NixOS / uv venv
with no ``pip``) that install can never persist, so it re-ran the
pip → ensurepip → ``pip install`` ladder on every poll, stalling the UI for
seconds per request.

The same install ladder is reachable from a *periodic* in-process caller, not
just the request-driven ``/api/status`` handler: every code path that resolves
gateway config funnels through ``gateway.config._apply_env_overrides()`` (cron
delivery-target resolution, job delivery, any background status recompute), and
that pass is what runs the adapter ``check_fn``s. So a background loop that
recomputes gateway config while the dashboard is idle would re-fire the ladder
just like a poll did. The single guard in ``_apply_env_overrides`` therefore
covers BOTH callers — the request path and any periodic path — because they
share that one chokepoint.

These tests assert that:
  * ``GET /api/status`` never shells out to a package installer;
  * the shared ``load_gateway_config`` chokepoint stays install-free even with
    an enabled platform (which forces ``check_fn`` into its install branch) and
    even when called *repeatedly*, as a periodic background loop would;
  * the guard is load-bearing — neuter it and the same path DOES install.
"""

from __future__ import annotations

import contextlib
import subprocess
import textwrap
from pathlib import Path

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


# ---------------------------------------------------------------------------
# Periodic / background status path.
#
# The bug report this guards against: while the dashboard is fully idle (zero
# HTTP requests), an in-process loop that recomputes gateway status still fired
# the pip/ensurepip/uv-install ladder roughly every few seconds. Every such
# recompute resolves gateway config, and config resolution runs each enabled
# platform's ``check_fn``, which lazy-installs the SDK. The fixtures below force
# that install branch (an enabled platform with an unsatisfied SDK) in an
# isolated HERMES_HOME, then drive ``load_gateway_config`` REPEATEDLY — the
# shape of a periodic loop — and assert it stays install-free.
# ---------------------------------------------------------------------------


def _missing_platform_feature():
    """A ``platform.<name>`` lazy feature whose SDK is not installed here.

    Returns ``(feature, platform_name)`` or ``None`` when every platform SDK is
    already present (then the install branch can't be exercised and the caller
    skips).
    """
    from tools import lazy_deps

    for feature in lazy_deps.LAZY_DEPS:
        if not feature.startswith("platform."):
            continue
        if lazy_deps.feature_missing(feature):
            return feature, feature.split(".", 1)[1]
    return None


@pytest.fixture
def home_with_enabled_platform(tmp_path, monkeypatch):
    """Isolated HERMES_HOME whose config enables a platform with a missing SDK.

    Enabling the platform is what pushes ``check_fn`` into its install branch
    during config resolution — otherwise a not-configured platform short-circuits
    before ``ensure`` and the test would pass vacuously.
    """
    found = _missing_platform_feature()
    if found is None:
        pytest.skip("no platform SDK is missing in this venv")
    _feature, platform = found

    home = tmp_path / "hermes_home"
    home.mkdir()
    # Minimal config: enable the platform and give it just enough config that the
    # adapter's check_fn proceeds to the SDK-import (ensure) step.
    (home / "config.yaml").write_text(
        textwrap.dedent(
            f"""\
            gateway:
              platforms:
                {platform}:
                  enabled: true
            {platform}:
              enabled: true
            """
        )
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    return platform


def test_periodic_gateway_config_resolution_is_install_free(
    home_with_enabled_platform, install_tripwire
):
    """A periodic loop recomputing gateway status must not install — ever.

    Calls ``load_gateway_config`` repeatedly (the shape of an idle background
    poller) with an enabled platform forcing the ``check_fn`` install branch.
    The shared ``_apply_env_overrides`` guard must keep every iteration
    install-free.
    """
    from gateway.config import load_gateway_config

    for _ in range(3):
        load_gateway_config()

    assert install_tripwire == [], (
        "periodic load_gateway_config() shelled out to an installer: "
        + ", ".join(install_tripwire)
    )


def test_guard_is_load_bearing_neutered_path_would_install(
    home_with_enabled_platform, monkeypatch
):
    """Negated-fix sanity check: without the guard the same path DOES install.

    Neuter ``installs_suppressed`` to a no-op so ``_apply_env_overrides`` no
    longer suppresses installs, then assert ``load_gateway_config`` now reaches
    the installer subprocess. This proves the passing test above is a genuine
    guard, not a vacuous pass (e.g. a path that never tries to install anyway).
    """
    import tools.lazy_deps as lazy_deps

    @contextlib.contextmanager
    def _noop():
        yield

    # _apply_env_overrides does ``from tools.lazy_deps import installs_suppressed``
    # at call time, so patching the module attribute is sufficient.
    monkeypatch.setattr(lazy_deps, "installs_suppressed", _noop)

    installs: list[str] = []
    real_run = subprocess.run
    real_popen = subprocess.Popen

    def record_run(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("args")
        if _looks_like_install(cmd):
            installs.append(_argv_text(cmd))
            raise FileNotFoundError("installer blocked in test")  # don't actually run
        return real_run(*args, **kwargs)

    def record_popen(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("args")
        if _looks_like_install(cmd):
            installs.append(_argv_text(cmd))
            raise FileNotFoundError("installer blocked in test")
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", record_run)
    monkeypatch.setattr(subprocess, "Popen", record_popen)

    from gateway.config import load_gateway_config

    load_gateway_config()

    assert installs, (
        "neutering installs_suppressed() did NOT cause an install — the passing "
        "test may be vacuous (this path never installs even unguarded)"
    )
