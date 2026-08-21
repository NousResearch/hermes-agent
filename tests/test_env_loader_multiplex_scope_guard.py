"""Regression tests: under ``gateway.multiplex_profiles`` a profile-scoped
``load_hermes_dotenv()`` must not overwrite process-global ``os.environ``.

One multiplexed gateway process serves every profile, so ``os.environ`` is
shared by all of their adapters. Agent turns run in-process under a per-profile
scope, and any module imported lazily mid-turn re-runs its import-time
``load_hermes_dotenv()`` — which resolves the home to the *routed* profile and
loaded that profile's ``.env`` over the shared environment with
``override=True``. Every other profile's adapter config was clobbered as a side
effect of one profile taking a turn (observed: a shared Discord channel
allowlist collapsing to one profile's list minutes after each restart, after
which the other bots silently ignored their own channels).

Profile credentials reach adapters through the isolated secret scope in
multiplex mode, so the global mutation is skipped entirely when a profile scope
is active.
"""
from __future__ import annotations

import os

import pytest

import hermes_constants
from agent import secret_scope
from hermes_cli import env_loader

SHARED_KEY = "HERMES_TEST_SHARED_ADAPTER_CONFIG"


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """Keep multiplex state, the home override and the probe key process-local."""
    monkeypatch.delenv(SHARED_KEY, raising=False)
    was_multiplex = secret_scope.is_multiplex_active()
    yield
    secret_scope.set_multiplex_active(was_multiplex)


def _make_home(tmp_path, name: str, shared_value: str):
    home = tmp_path / name
    home.mkdir()
    (home / ".env").write_text(f"{SHARED_KEY}={shared_value}\n")
    return home


def test_profile_scoped_load_does_not_clobber_shared_env(tmp_path, monkeypatch):
    """Multiplex + active profile scope → shared process env is left alone."""
    default_home = _make_home(tmp_path, "default", "all-channels")
    profile_home = _make_home(tmp_path, "profile-a", "profile-a-only")

    # Gateway startup: the default profile's .env populates the shared env.
    secret_scope.set_multiplex_active(False)
    env_loader.load_hermes_dotenv(hermes_home=str(default_home))
    assert os.environ[SHARED_KEY] == "all-channels"

    # A turn routed to profile-a lazily imports a module whose import-time
    # loader call resolves the home to profile-a.
    secret_scope.set_multiplex_active(True)
    token = hermes_constants.set_hermes_home_override(str(profile_home))
    try:
        loaded = env_loader.load_hermes_dotenv(hermes_home=str(profile_home))
    finally:
        hermes_constants.reset_hermes_home_override(token)

    assert os.environ[SHARED_KEY] == "all-channels", (
        "profile-scoped .env load overwrote shared process env under multiplex"
    )
    assert loaded == [], "no files should be reported as globally applied"


def test_multiplex_without_profile_scope_still_loads(tmp_path, monkeypatch):
    """Multiplex alone must not disable loading — startup has no scope yet."""
    home = _make_home(tmp_path, "default", "startup-value")

    secret_scope.set_multiplex_active(True)
    assert hermes_constants.get_hermes_home_override() is None

    loaded = env_loader.load_hermes_dotenv(hermes_home=str(home))

    assert os.environ[SHARED_KEY] == "startup-value"
    assert (home / ".env") in loaded


def test_single_profile_gateway_is_unaffected(tmp_path, monkeypatch):
    """Without multiplex, a scoped load keeps its historical override behaviour."""
    default_home = _make_home(tmp_path, "default", "first")
    other_home = _make_home(tmp_path, "other", "second")

    secret_scope.set_multiplex_active(False)
    env_loader.load_hermes_dotenv(hermes_home=str(default_home))
    assert os.environ[SHARED_KEY] == "first"

    token = hermes_constants.set_hermes_home_override(str(other_home))
    try:
        loaded = env_loader.load_hermes_dotenv(hermes_home=str(other_home))
    finally:
        hermes_constants.reset_hermes_home_override(token)

    assert os.environ[SHARED_KEY] == "second"
    assert (other_home / ".env") in loaded
