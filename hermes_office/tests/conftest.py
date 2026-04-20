"""Pytest fixtures for hermes_office tests.

Crucially: every test runs against an isolated temp directory used as
``HERMES_HOME`` so we never touch the user's real ~/.hermes state.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture()
def hermes_home(tmp_path: Path, monkeypatch) -> Path:
    home = tmp_path / "dot-hermes"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Force re-resolution by hermes_constants on every call.
    try:
        import hermes_constants

        # Some hermes_constants impls cache get_hermes_home()'s result.
        # Clear any module-level cache attributes if they exist.
        for attr in ("_HERMES_HOME", "_CACHED_HOME"):
            if hasattr(hermes_constants, attr):
                setattr(hermes_constants, attr, None)
    except Exception:
        pass
    return home


@pytest.fixture()
def office_root(hermes_home: Path) -> Path:
    root = hermes_home / "office"
    root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture()
def store(office_root: Path):
    from hermes_office.store import Store

    s = Store(office_root)
    s.boot_from_disk()
    return s


@pytest.fixture()
def department(store):
    from hermes_office.models import Department

    dept = Department(name="Lab", mission="Experiments", color="#22c55e")
    return store.create_department(dept)


@pytest.fixture()
def employee(store, department):
    from hermes_office.models import Employee

    emp = Employee(
        department_id=department.id,
        name="Alice",
        role="Researcher",
        model="gemma4-e2b-hermes",
        runtime="simulated",
    )
    return store.create_employee(emp)
