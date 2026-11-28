"""The tests root must never be importable as a package directory.

``tests/`` holds a subdirectory per area, and several of those names are taken:
``acp`` is a third-party dependency, and ``cron``, ``plugins``, ``providers``
and ``tools`` are first-party packages of this repository. Putting ``tests/`` on
``sys.path`` makes those directories SHADOW the real modules for the rest of the
session -- ``from acp.schema import ToolKind`` starts raising
``ModuleNotFoundError`` against a stub with no ``schema`` submodule, in a test
that passes when run on its own.

The insert that caused it was an off-by-one (``parents[1]`` from a file in
``tests/hermes_cli/`` is ``tests/``, not the repo root) at MODULE level in a
file whose every test is skipped off Windows: collection alone was enough to
poison the run. This gate is cheap and catches the whole class -- collection
happens before any test runs, so a module-level insert anywhere is already
visible here.
"""
from __future__ import annotations

import sys
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TESTS_ROOT.parent


def test_the_tests_root_is_not_on_sys_path() -> None:
    on_path = [entry for entry in sys.path if Path(entry).resolve() == TESTS_ROOT]
    assert not on_path, (
        f"{TESTS_ROOT} is on sys.path, so every directory under it shadows a "
        f"top-level module of the same name. Insert the REPO root ({REPO_ROOT}) "
        f"if a test needs it -- pytest already prepends it."
    )


def test_the_shadowing_names_are_real_so_the_gate_above_matters() -> None:
    """A floor: the gate would be theatre if no name under tests/ collided."""
    collisions = sorted(
        entry.name
        for entry in TESTS_ROOT.iterdir()
        if entry.is_dir() and (REPO_ROOT / entry.name).is_dir()
    )
    # `acp` is not in this list -- it lives in site-packages, not the repo --
    # and it is the one that actually broke, so the floor stays a floor.
    assert collisions, f"nothing under {TESTS_ROOT} shares a name with a repo package"
