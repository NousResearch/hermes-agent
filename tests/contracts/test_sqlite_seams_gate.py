"""The SQLite seams on a backend without its conversation family (no repository registered, no module importable): the seams load and run on the store holder alone (the epoch seam included), and a seam that needs an unregistered repository refuses with the refusal the registration gate skips."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from types import SimpleNamespace

import pytest

from persistence import CapabilityUnsupported
from tests.contracts import conftest
from tests.contracts._scopes import profile_scope


def _sqlite_present() -> bool:
    try:
        return importlib.util.find_spec("persistence.sqlite") is not None
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _sqlite_present(), reason="no SQLite backend in this tree")

_FAMILY = "persistence.sqlite.conversation"
_SEAMS = "tests.contracts._sqlite_seams"


@pytest.fixture
def seams_without_conversation(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """The seams module imported afresh while the SQLite conversation family has no registered repository and cannot be imported, and one handle it opens."""
    # The backend package's public names import every module on disk, so the family registers before it is withdrawn.
    importlib.import_module("persistence.sqlite").SQLiteBackend
    registry = importlib.import_module("persistence.sqlite._backend")
    factories = {attribute: factory for attribute, factory in registry._REPOSITORY_FACTORIES.items() if not factory.__module__.startswith(_FAMILY + ".")}
    monkeypatch.setattr(registry, "_REPOSITORY_FACTORIES", factories)
    monkeypatch.delattr(importlib.import_module("persistence.sqlite"), "conversation", raising=False)
    for name in [name for name in sys.modules if name == _FAMILY or name.startswith(_FAMILY + ".")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, _FAMILY, None)
    monkeypatch.delitem(sys.modules, _SEAMS, raising=False)
    seams = importlib.import_module(_SEAMS)
    scope = profile_scope("seams-gate")
    backend = seams.ContractSQLiteBackend({scope.hermes_home_key: tmp_path / "state.db"}, root=tmp_path)
    try:
        yield backend, scope, backend.open(scope)
    finally:
        backend.close()


def test_the_seams_run_on_the_holder_without_the_conversation_family(seams_without_conversation) -> None:
    backend, scope, store = seams_without_conversation
    assert {"sessions", "messages", "search", "usage"} <= importlib.import_module("persistence.sqlite").unfolded_capabilities()
    store.search.seed_deferred_rebuild(7, 3)
    holder = backend.session_db(scope)
    assert holder is store.holder()
    assert holder.get_meta("fts_rebuild_high_water") == "7"
    assert holder.get_meta("fts_rebuild_progress") == "3"
    with pytest.raises(KeyError):
        backend.scope_store(scope)["sessions"]["sess-absent"]


def test_the_epoch_seam_advances_the_runtime_epoch_on_the_holder(seams_without_conversation) -> None:
    backend, scope, store = seams_without_conversation
    assert [backend.begin_epoch(scope, "instance-a"), backend.begin_epoch(scope, "instance-b")] == [1, 2]
    assert tuple(store.holder()._read_one("SELECT epoch, instance_id FROM runtime_epoch WHERE singleton=1", ())) == (2, "instance-b")


def test_the_seams_on_family_modules_refuse_without_the_family(seams_without_conversation) -> None:
    _backend, _scope, store = seams_without_conversation
    for seam in (lambda: store.owner_profile, lambda: store.runtime.seed_local_receipt("s-x", {}), lambda: store.runtime.seed_retirement("s-x")):
        with pytest.raises(CapabilityUnsupported, match="capability .sessions.$"):
            seam()


def test_the_registration_gate_skips_an_unregistered_repository_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(conftest._UNFOLDED_CAPABILITIES, "sqlite", frozenset({"sessions"}))
    item = SimpleNamespace(callspec=SimpleNamespace(params={"backend_name": "sqlite"}))
    with pytest.raises(pytest.skip.Exception, match=r"^sessions branch not folded on this backend yet$"):
        conftest._skip_unfolded_refusal(item, CapabilityUnsupported("sessions"))
