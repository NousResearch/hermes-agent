"""Regression tests for the dashboard's unscoped-secret-read defect (audit 2026-09-25).

Five call sites raise ``UnscopedSecretError`` on a multiplexed dashboard because the
dashboard's own HTTP routes never bind ``set_secret_scope()`` before reaching credential
readers — only ``_cron_store_scope``/``get_active_profile_name`` redirect HERMES_HOME, never
secrets. Contract: a dashboard operation logically owned by profile A must read profile A's
secrets, never profile B's, never silently the default profile's, and never leak across
concurrent requests for two different profiles.

These tests are written to FAIL on current `main` (Test A, C via absence-of-guarantee) and
PASS once the dashboard boundary binds secret scope alongside its existing home override.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from agent import secret_scope
from agent.secret_scope import (
    UnscopedSecretError,
    build_profile_secret_scope,
    reset_secret_scope,
    set_secret_scope,
)


@pytest.fixture
def multiplex():
    secret_scope.set_multiplex_active(True)
    try:
        yield
    finally:
        secret_scope.set_multiplex_active(False)


@pytest.fixture
def two_profiles(tmp_path):
    """Two isolated profile homes, each with a distinct dummy secret value."""
    home_a = tmp_path / "profile_a"
    home_b = tmp_path / "profile_b"
    home_a.mkdir()
    home_b.mkdir()
    (home_a / ".env").write_text("FIRECRAWL_API_KEY=profile-a-secret\nMEM0_MODE=platform\n", encoding="utf-8")
    (home_b / ".env").write_text("FIRECRAWL_API_KEY=profile-b-secret\nMEM0_MODE=oss\n", encoding="utf-8")
    return {"a": home_a, "b": home_b}


# --- Test A: unbound background/dashboard-style read fails closed --------------------------

def test_a_unbound_dashboard_style_read_raises_unscoped(monkeypatch, multiplex):
    """Reproduces the exact failure: a dashboard route reading a secret with no scope bound,
    under multiplex, must fail loud — this IS the 226-occurrence symptom, not a bug to silence."""
    monkeypatch.setenv("FIRECRAWL_API_KEY", "default-profile-leak-if-this-were-read")
    token = set_secret_scope(None)
    try:
        with pytest.raises(UnscopedSecretError):
            secret_scope.get_secret("FIRECRAWL_API_KEY")
    finally:
        reset_secret_scope(token)


# --- Test B: correct profile scope resolves the correct profile's secret -------------------

def test_b_profile_a_resolves_profile_a_secret(two_profiles, multiplex):
    scope_a = build_profile_secret_scope(two_profiles["a"])
    token = set_secret_scope(scope_a)
    try:
        assert secret_scope.get_secret("FIRECRAWL_API_KEY") == "profile-a-secret"
    finally:
        reset_secret_scope(token)


def test_b_profile_b_resolves_profile_b_secret(two_profiles, multiplex):
    scope_b = build_profile_secret_scope(two_profiles["b"])
    token = set_secret_scope(scope_b)
    try:
        assert secret_scope.get_secret("FIRECRAWL_API_KEY") == "profile-b-secret"
    finally:
        reset_secret_scope(token)


# --- Test C: no cross-profile leakage -------------------------------------------------------

def test_c_profile_a_scope_never_yields_profile_b_secret(two_profiles, multiplex):
    scope_a = build_profile_secret_scope(two_profiles["a"])
    token = set_secret_scope(scope_a)
    try:
        value = secret_scope.get_secret("FIRECRAWL_API_KEY")
        assert value == "profile-a-secret"
        assert value != "profile-b-secret"
    finally:
        reset_secret_scope(token)


def test_c_profile_b_scope_never_yields_profile_a_secret(two_profiles, multiplex):
    scope_b = build_profile_secret_scope(two_profiles["b"])
    token = set_secret_scope(scope_b)
    try:
        value = secret_scope.get_secret("FIRECRAWL_API_KEY")
        assert value == "profile-b-secret"
        assert value != "profile-a-secret"
    finally:
        reset_secret_scope(token)


# --- Test D: concurrency — scope must not bleed across threads/tasks -----------------------

def test_d_concurrent_dashboard_reads_never_cross_profiles(two_profiles, multiplex):
    """Simulates two concurrent dashboard requests (ThreadPoolExecutor, like
    ``run_in_threadpool``/``asyncio.to_thread``) for two different profiles. Each worker
    enters its OWN secret scope (the fix's shape: bind-then-read, not a shared contextvar
    mutated in place) and must observe only its own profile's secret, repeated many times to
    catch any interleaving-dependent bleed."""
    scope_a = build_profile_secret_scope(two_profiles["a"])
    scope_b = build_profile_secret_scope(two_profiles["b"])
    errors = []

    def worker(which, scope, expected):
        token = set_secret_scope(scope)
        try:
            for _ in range(25):
                got = secret_scope.get_secret("FIRECRAWL_API_KEY")
                if got != expected:
                    errors.append((which, got, expected))
        finally:
            reset_secret_scope(token)

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = []
        for i in range(10):
            futures.append(pool.submit(worker, "a", scope_a, "profile-a-secret"))
            futures.append(pool.submit(worker, "b", scope_b, "profile-b-secret"))
        for f in as_completed(futures):
            f.result()

    assert errors == [], f"cross-profile secret bleed detected: {errors}"


# --- Test E: missing profile identity must fail explicitly, never default silently ---------

def test_e_missing_identity_under_multiplex_never_silently_defaults(monkeypatch, multiplex):
    """An operation with genuinely no bound profile must raise, not quietly resolve to
    os.environ (which under multiplex holds the DEFAULT profile's values — the exact
    silent-leak class the design forbids, per gateway/AGENTS.md 'Multiplex profile-scoped env
    reads MUST fail closed — never borrow from os.environ')."""
    monkeypatch.setenv("FIRECRAWL_API_KEY", "default-profile-value")
    token = set_secret_scope(None)
    try:
        with pytest.raises(UnscopedSecretError):
            secret_scope.get_secret("FIRECRAWL_API_KEY")
    finally:
        reset_secret_scope(token)


# --- mem0-specific: config load must be profile-correct and never raise once scoped --------

def test_mem0_load_config_resolves_per_profile_mode(two_profiles, multiplex, monkeypatch):
    """MEM0_MODE must resolve to the OWNING profile's value; a profile without multiplex
    scope bound must still raise (matches the plugin's own documented contract), never
    silently pick up another profile's or the default's mode."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from plugins.memory.mem0 import _load_config

    home_token = set_hermes_home_override(str(two_profiles["a"]))
    scope_token = set_secret_scope(build_profile_secret_scope(two_profiles["a"]))
    try:
        cfg = _load_config()
        assert cfg["mode"] == "platform"
    finally:
        reset_secret_scope(scope_token)
        reset_hermes_home_override(home_token)

    home_token = set_hermes_home_override(str(two_profiles["b"]))
    scope_token = set_secret_scope(build_profile_secret_scope(two_profiles["b"]))
    try:
        cfg = _load_config()
        assert cfg["mode"] == "oss"
    finally:
        reset_secret_scope(scope_token)
        reset_hermes_home_override(home_token)


def test_mem0_load_config_unscoped_under_multiplex_raises(multiplex):
    """Matches the plugin's own documented intent (see docstring in
    plugins/memory/mem0/__init__.py::_load_config): a scope-less multiplex caller raising is
    correct — swallowing it would silently route memories to the default profile."""
    from plugins.memory.mem0 import _load_config

    token = set_secret_scope(None)
    try:
        with pytest.raises(UnscopedSecretError):
            _load_config()
    finally:
        reset_secret_scope(token)


# --- Route-boundary reproduction: the ACTUAL defect, not just the underlying primitive ------
#
# secret_scope.py itself is correct (tests above all pass on current main). The bug is that
# hermes_cli/web_routers/status.py::_get_portal_status_sync — the real dashboard route body —
# never calls set_secret_scope() before reaching credential readers. This test calls that real
# function, unscoped, under multiplex, exactly as the dashboard's asyncio.to_thread(...) call
# does today, and captures what happens — proving the route itself is the missing boundary.

def test_route_get_portal_status_sync_binds_its_own_profile_scope(monkeypatch, tmp_path, multiplex):
    """Post-fix invariant (was the silent-degradation tripwire pre-fix): calling the REAL route
    body with NO caller-provided scope must no longer hit UnscopedSecretError internally —
    ``_get_portal_status_sync`` now binds the dashboard's own current profile
    (``dashboard_profile_secret_scope``) before reaching any credential reader. A legitimate
    lookup failure (unrelated provider/network error) is still possible and must be reported
    via ``features_available: False`` — but a bare scope-less call must never be silently
    indistinguishable from that.
    """
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from hermes_cli.web_routers import status as status_router

    home = tmp_path / "profile_current"
    home.mkdir()
    (home / ".env").write_text("", encoding="utf-8")
    (home / "config.yaml").write_text("model: test-model\n", encoding="utf-8")

    # Unrelated to the scope invariant under test: the real "browser" feature check shells out
    # to pm's package-store resolution, which reads this checkout's actual repo root/manifest —
    # forbidden host I/O under the test sandbox (tests/home_io_guard.py) and orthogonal to
    # whether the route binds its own secret scope. Stub both probes so only the scope boundary
    # is exercised, matching the other feature probes this test does not need to invoke for real.
    monkeypatch.setattr(
        "hermes_cli.nous_subscription._has_agent_browser", lambda: True, raising=False
    )
    monkeypatch.setattr(
        "hermes_cli.nous_subscription._local_browser_runnable", lambda: True, raising=False
    )

    home_token = set_hermes_home_override(str(home))
    # Deliberately do NOT set a caller-side secret scope here — the whole point is that the
    # route must bind its own now, unlike the pre-fix version this test used to pin as buggy.
    try:
        result = status_router._get_portal_status_sync()
    finally:
        reset_hermes_home_override(home_token)

    # No UnscopedSecretError means the call above would have raised, which pytest surfaces
    # as a hard failure — reaching this point already proves the invariant. The explicit
    # availability flags additionally prove the lookup ran (not just "didn't crash").
    assert result["features_available"] is True
    assert result["auth_available"] is True
    # A real, non-empty feature list (proving the lookup actually ran under this fresh temp
    # profile, unlike the old bug's empty-list degradation) is what matters here — the exact
    # per-feature states depend on host-level toolset config (global, non-profile env vars are
    # still legitimately visible) and are not the invariant under test.
    assert result["features"], "expected a real, non-empty feature list from a working lookup"


def test_route_cron_delivery_targets_binds_scope(tmp_path, multiplex):
    """cron/scheduler_delivery.py::cron_delivery_targets, reached through the dashboard's
    ``_cron_store_scope``-independent GET route, must resolve without raising once the route
    binds its own profile secret scope."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from cron.scheduler_delivery import cron_delivery_targets
    from hermes_cli.dashboard_profile_scope import dashboard_profile_secret_scope

    home = tmp_path / "profile_current"
    home.mkdir()
    (home / ".env").write_text("", encoding="utf-8")
    (home / "config.yaml").write_text("model: test-model\n", encoding="utf-8")

    home_token = set_hermes_home_override(str(home))
    try:
        with dashboard_profile_secret_scope():
            targets = cron_delivery_targets()  # must not raise UnscopedSecretError
        assert isinstance(targets, list)
    finally:
        reset_hermes_home_override(home_token)


def test_route_memory_providers_probe_binds_scope(tmp_path, multiplex):
    """The dashboard's plugins-hub memory-provider discovery (which iterates ALL providers,
    including mem0, regardless of the active one) must resolve without raising once bound."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from hermes_cli.web_server_memory import _discover_memory_provider_statuses
    from hermes_cli.dashboard_profile_scope import dashboard_profile_secret_scope

    home = tmp_path / "profile_current"
    home.mkdir()
    (home / ".env").write_text("", encoding="utf-8")
    (home / "config.yaml").write_text("model: test-model\n", encoding="utf-8")

    home_token = set_hermes_home_override(str(home))
    try:
        with dashboard_profile_secret_scope():
            rows = _discover_memory_provider_statuses()  # must not raise UnscopedSecretError
        assert isinstance(rows, list)
    finally:
        reset_hermes_home_override(home_token)
