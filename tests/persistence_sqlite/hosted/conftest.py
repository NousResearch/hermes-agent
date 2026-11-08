"""Fixtures for the hosted-room backend tests: one install under the test's HERMES_HOME with a root store and two
profile stores, scopes whose folders are those real homes (so ``'shared'`` resolves to the install's root
``shared-state.db``), a fixed wall clock (``clock``, which the modules using it request), and the helpers the parity cases use to run main's function on one copy
of a seeded file and the backend verb on another."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Iterable

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


def seeded_pair(backend, scope: ProfileScope, placement: str, tmp_path: Path, seed: Callable[[Path], None]) -> tuple[Path, Path]:
    """Seed the placement's file (``seed`` gets its path), then copy it aside: returns (main's copy, the backend's file).
    Main's copy keeps the file name, which decides whether main runs the pre-isolation import."""
    files = backend.open(scope).hosted_rooms._files
    target = files.path(placement)
    if placement == "profile":
        backend.open(scope).state_meta.get("seed")  # the holder creates the profile's state.db
        backend.reopen()
        from persistence.sqlite import _registry
        _registry.close_all()
    seed(target)
    legacy = tmp_path / "m1" / target.name
    copy_file(target, legacy)
    return legacy, target


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(src) as source, sqlite3.connect(dst) as target:
        source.backup(target)


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


def same_rows(target: Path, legacy: Path) -> tuple[dict, dict]:
    """The hosted-room rows of the backend's file and of main's copy, over the hosted tables main's copy holds."""
    names = [name for name in HOSTED_TABLES if name in tables(legacy)]
    return dump(target, names), dump(legacy, names)


def outcome(call: Callable[[], Any]) -> Any:
    """The value a call returns, or the class name and message of what it raises."""
    try:
        return ("ok", call())
    except Exception as exc:
        return ("raised", type(exc).__name__, str(exc))
