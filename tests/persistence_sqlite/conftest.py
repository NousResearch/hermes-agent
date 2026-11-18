"""Fixtures for the SQLite backend's tests: a backend over two profile stores under the test's HERMES_HOME, with
the contract suites' two scopes, and a registry left empty after every test."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_constants import get_hermes_home
from persistence import ProfileScope
from persistence.sqlite import SQLiteBackend
from persistence.sqlite import _registry
from tests.contracts._scopes import profile_scope


@pytest.fixture
def scope_a() -> ProfileScope:
    return profile_scope("profile-a")


@pytest.fixture
def scope_b() -> ProfileScope:
    return profile_scope("profile-b")


def store_for(scope: ProfileScope) -> Path:
    """Where a test scope's store lives: a profile home under the test's HERMES_HOME."""
    return get_hermes_home() / "profiles" / scope.name / "state.db"


def profile_home(name: str) -> Path:
    """A created profile home under the test's HERMES_HOME (a writable open never creates a missing one)."""
    home = get_hermes_home() / "profiles" / name
    home.mkdir(parents=True, exist_ok=True)
    return home


@pytest.fixture
def backend(scope_a, scope_b):
    for scope in (scope_a, scope_b):
        profile_home(scope.name)
    instance = SQLiteBackend({scope.hermes_home_key: store_for(scope) for scope in (scope_a, scope_b)})
    try:
        yield instance
    finally:
        instance.close()


@pytest.fixture(autouse=True)
def _empty_registry():
    yield
    _registry.close_all()
