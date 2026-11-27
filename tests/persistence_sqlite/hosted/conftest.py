"""Fixtures for the hosted-room backend tests: one install under the test's HERMES_HOME with a root store and two profile stores, scopes whose folders are those real homes (so ``'shared'`` resolves to the install's root ``shared-state.db``), a fixed wall clock (``clock``, which the modules using it request), and dumps of a file's hosted-room tables."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable

import pytest

from hermes_constants import get_hermes_home
from persistence import ProfileScope
from persistence.sqlite import SQLiteBackend

NOW = 1_800_000_000.0

# The tables the rooms and replica schema steps own, plus the import marker.
HOSTED_TABLES = (
    "hosted_rooms", "hosted_room_events", "hosted_room_retired_ids", "hosted_room_links", "hosted_room_remote_runs",
    "hosted_room_revoked_grants", "hosted_room_peer_reservations", "hosted_room_replicas",
    "hosted_room_replica_events", "hosted_room_legacy_imports")


class Clock:
    def __init__(self) -> None:
        self.now = NOW

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch) -> Clock:
    fixed = Clock()
    monkeypatch.setattr(time, "time", fixed)
    return fixed


@pytest.fixture
def root() -> Path:
    """The install root: the test's HERMES_HOME."""
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    return home


def profile_home(root: Path, name: str) -> Path:
    home = root / "profiles" / name
    home.mkdir(parents=True, exist_ok=True)
    return home


@pytest.fixture
def scopes(root: Path) -> dict[str, ProfileScope]:
    """The root profile and profiles ``a`` and ``b`` of the install, each scope on its real home."""
    return {
        "root": ProfileScope(profile_dir=root, name="default"),
        "a": ProfileScope(profile_dir=profile_home(root, "a"), name="a"),
        "b": ProfileScope(profile_dir=profile_home(root, "b"), name="b"),
    }


@pytest.fixture
def backend(scopes):
    instance = SQLiteBackend({scope.hermes_home_key: scope.profile_dir / "state.db" for scope in scopes.values()})
    try:
        yield instance
    finally:
        instance.close()


def tables(path: Path) -> set[str]:
    with sqlite3.connect(path) as conn:
        return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def dump(path: Path, names: Iterable[str] = HOSTED_TABLES) -> dict[str, list[tuple[Any, ...]]]:
    """Every row of each named table the file holds, in a total order."""
    present = tables(path)
    out: dict[str, list[tuple[Any, ...]]] = {}
    with sqlite3.connect(path) as conn:
        for name in names:
            if name in present:
                out[name] = sorted(conn.execute(f"SELECT * FROM {name}").fetchall(), key=repr)
    return out
