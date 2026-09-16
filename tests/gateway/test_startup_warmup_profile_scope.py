"""The boot turn-machinery warm-up runs inside the launch profile's scope under multiplex.

``get_tool_definitions`` runs every ``check_fn``; the vision probe resolves live Nous runtime
credentials through ``hermes_cli.auth_nous``, whose Portal / inference routing overrides read through
``agent.secret_scope.get_secret``. With multiplex active and no scope installed that read fails
closed, the override is absent, and a non-production Portal's refresh token is POSTed to the
production Portal — ``invalid_grant``, quarantined login, ~10 s after every boot and before any
inbound turn (hosted staging, 2026-09-16). The warm-up must run under the same binding a routed
turn gets, and the binding must reach the executor thread.
"""
from __future__ import annotations

import asyncio
import types
from pathlib import Path

import pytest

from agent import secret_scope
from gateway import run as gateway_run
from gateway.run_startup import GatewayStartupMixin

PORTAL = "https://portal.staging-nousresearch.com"


@pytest.fixture
def multiplex_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".env").write_text(f"HERMES_PORTAL_BASE_URL={PORTAL}\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_PORTAL_BASE_URL", raising=False)
    monkeypatch.delenv("NOUS_PORTAL_BASE_URL", raising=False)
    secret_scope.set_multiplex_active(True)
    try:
        yield home
    finally:
        secret_scope.set_multiplex_active(False)


def _run_warmup(monkeypatch, *, multiplex: bool) -> dict:
    """Drive ``_warm_turn_prerequisites`` with the sync warm-up replaced by a probe that records what
    the executor thread can see."""
    seen: dict = {}

    def probe() -> int:
        from hermes_cli.auth_nous import _nous_portal_env_override
        seen["scope_installed"] = secret_scope._SECRET_SCOPE.get() is not None
        try:
            seen["portal_override"] = _nous_portal_env_override()
        except secret_scope.UnscopedSecretError:
            seen["portal_override"] = "UNSCOPED"
        return 1

    monkeypatch.setattr(gateway_run, "_warm_turn_machinery_sync", probe)
    runner = types.SimpleNamespace(config=types.SimpleNamespace(multiplex_profiles=multiplex))
    asyncio.run(GatewayStartupMixin._warm_turn_prerequisites(runner))
    return seen


def test_multiplex_warmup_runs_check_fns_inside_the_launch_profile_scope(multiplex_home, monkeypatch):
    """The executor thread sees the launch profile's secret scope, so the .env routing override resolves
    exactly as it does on a routed turn — never the fail-closed 'absent' that heals to production."""
    seen = _run_warmup(monkeypatch, multiplex=True)
    assert seen["scope_installed"] is True
    assert seen["portal_override"] == PORTAL


def test_multiplex_warmup_leaves_no_scope_behind(multiplex_home, monkeypatch):
    """The binding is per activity: once the warm-up returns, the loop thread is unscoped again."""
    _run_warmup(monkeypatch, multiplex=True)
    assert secret_scope._SECRET_SCOPE.get() is None
    from hermes_constants import get_hermes_home_override
    assert get_hermes_home_override() is None


def test_single_profile_warmup_keeps_environ_semantics(tmp_path, monkeypatch):
    """Multiplex off: no scope is installed and the process env stays the override source."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PORTAL_BASE_URL", PORTAL)
    secret_scope.set_multiplex_active(False)
    seen = _run_warmup(monkeypatch, multiplex=False)
    assert seen["scope_installed"] is False
    assert seen["portal_override"] == PORTAL
