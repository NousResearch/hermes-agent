"""The assembly's admin doors branch on the configured backend's ``physical_admin`` capability."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import persistence_assembly
from persistence.sqlite import SQLiteBackend
from persistence.sqlite.admin import SQLiteProfileRepairAdmin, SQLiteStoreAdmin


def test_both_doors_open_sqlite_handles_bound_to_their_target_and_touch_no_file(tmp_path):
    scope = persistence_assembly.profile_scope(home=tmp_path)
    file = tmp_path / "copy" / "state.db"
    store_admin = persistence_assembly.open_store_admin(scope)
    file_admin = persistence_assembly.open_store_admin(file, read_only=False)
    repair = persistence_assembly.open_profile_repair(scope, read_only=True)
    assert isinstance(store_admin, SQLiteStoreAdmin) and isinstance(file_admin, SQLiteStoreAdmin)
    assert isinstance(repair, SQLiteProfileRepairAdmin)
    state_db = Path(scope.hermes_home_key) / "state.db"
    assert (store_admin.db_path, store_admin.read_only) == (state_db, True)
    assert (file_admin.db_path, file_admin.read_only) == (file, False)
    assert (repair.db_path, repair.read_only) == (state_db, True)
    assert not (tmp_path / "state.db").exists() and not file.parent.exists()


def test_the_backend_refuses_an_admin_handle_for_a_profile_it_holds_no_store_for(tmp_path):
    served = persistence_assembly.profile_scope(home=tmp_path / "served")
    other = persistence_assembly.profile_scope(home=tmp_path / "other")
    backend = SQLiteBackend({served.hermes_home_key: tmp_path / "served" / "state.db"})
    with pytest.raises(ValueError):
        backend.open_store_admin(other)
    with pytest.raises(ValueError):
        backend.open_profile_repair(other, read_only=True)


def test_a_backend_with_the_capability_opens_the_handle_for_the_target(tmp_path, monkeypatch):
    calls = []

    class Backend:
        capabilities = SimpleNamespace(physical_admin=True)

        def __init__(self, stores, read_only):
            calls.append(("assemble", dict(stores), read_only))

        def open_store_admin(self, target, *, read_only):
            return ("store-admin", target, read_only)

        def open_profile_repair(self, scope, *, read_only):
            return ("profile-repair", scope, read_only)

    monkeypatch.setattr(persistence_assembly, "_assemble", lambda provider, stores, *, read_only=False: Backend(stores, read_only))
    scope = persistence_assembly.profile_scope(home=tmp_path)
    file = tmp_path / "copy" / "state.db"
    assert persistence_assembly.open_store_admin(scope) == ("store-admin", scope, True)
    assert persistence_assembly.open_store_admin(file, read_only=False) == ("store-admin", file, False)
    assert persistence_assembly.open_profile_repair(scope, read_only=True) == ("profile-repair", scope, True)
    assert [stores for _, stores, _ in calls][1] == {persistence_assembly.profile_scope(home=file.parent).hermes_home_key: file.resolve()}


def test_a_backend_without_the_capability_answers_unsupported(tmp_path, monkeypatch):
    from persistence.core.admin import PHYSICAL_ADMIN, AdminUnsupported

    class Backend:
        capabilities = SimpleNamespace(physical_admin=False)
        closed = 0

        def __init__(self, stores, read_only):
            pass

        def close(self):
            Backend.closed += 1

    monkeypatch.setattr(persistence_assembly, "_assemble", lambda provider, stores, *, read_only=False: Backend(stores, read_only))
    scope = persistence_assembly.profile_scope(home=tmp_path)
    unsupported = AdminUnsupported(PHYSICAL_ADMIN, "sqlite")
    assert persistence_assembly.open_store_admin(scope) == unsupported
    assert persistence_assembly.open_store_admin(tmp_path / "state.db", read_only=False) == unsupported
    assert persistence_assembly.open_profile_repair(scope, read_only=True) == unsupported
    assert Backend.closed == 3
    assert not (tmp_path / "state.db").exists()
