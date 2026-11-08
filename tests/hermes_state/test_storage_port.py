"""The session storage ports as a plugin kind, and the local backend's pin.

``resolve_session_storage(name)`` picks a backend by explicit name (no caller is wired to it yet):
``"local"`` is the local ``state.db``; another name loads a provider through real discovery
(``$HERMES_HOME/plugins/<name>/``), and a missing one is an error, never a fallback to the local
file. The local backend serves only the local tenant and its own profile.
"""
import textwrap
from contextlib import closing

import pytest

from agent.runtime_scope import bind_profile_scope, bind_state, bind_turn
from hermes_scope import Group, Principal, Scope
from hermes_state import SessionDB
from hermes_state_errors import TurnFenceRequiredError
from hermes_state_port import (
    SessionStorageProvider, SessionStore, StoragePinRefused, TurnAdmission, resolve_session_storage,
)
from hermes_state_port_local import LocalSessionStorageProvider

_PLUGIN = '''
from hermes_state_port import SessionStorageProvider, SessionStorage


class Probe(SessionStorageProvider):
    @property
    def name(self):
        return "probe"

    def open(self, db=None):
        raise NotImplementedError

    def assert_isolating(self):
        raise RuntimeError("probe role bypasses row security")


def register(ctx):
    ctx.register_session_storage(Probe())
'''


def test_local_names_the_local_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for name in ("local", ""):
        provider = resolve_session_storage(name)
        assert isinstance(provider, LocalSessionStorageProvider) and provider.levels == frozenset({"agent"})
        provider.assert_isolating()


def test_a_provider_that_does_not_check_isolation_cannot_be_built():
    class Unchecked(SessionStorageProvider):
        @property
        def name(self):
            return "unchecked"

        def open(self, db=None):
            raise NotImplementedError

    with pytest.raises(TypeError, match="assert_isolating"):
        Unchecked()


def test_a_named_provider_loads_through_discovery_and_runs_its_isolation_check(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    plugin = tmp_path / "plugins" / "probe"
    plugin.mkdir(parents=True)
    (plugin / "__init__.py").write_text(textwrap.dedent(_PLUGIN))
    provider = resolve_session_storage("probe")
    assert isinstance(provider, SessionStorageProvider) and provider.name == "probe"
    with pytest.raises(RuntimeError, match="bypasses row security"):
        provider.assert_isolating()


def test_a_missing_named_provider_is_an_error_not_the_local_store(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with pytest.raises(LookupError, match="postgres"):
        resolve_session_storage("postgres")


@pytest.fixture
def ports(tmp_path):
    with closing(SessionDB(tmp_path / "state.db")) as db:
        storage = LocalSessionStorageProvider().open(db)
        assert isinstance(storage.sessions, SessionStore) and isinstance(storage.admission, TurnAdmission)
        yield db, storage


def test_the_local_backend_is_todays_store(ports, tmp_path):
    db, storage = ports
    with bind_profile_scope(tmp_path):
        storage.sessions.create("s1", "cli")
        storage.sessions.append("s1", [{"role": "user", "content": "hi"}])
        assert storage.sessions.set_title("s1", "first")
        epoch = storage.admission.begin_epoch(instance_id="owner")
    # The same rows today's SessionDB and ledger callers see.
    assert db.get_session("s1")["title"] == "first"
    assert [m["content"] for m in db.get_messages("s1")] == ["hi"]
    from hermes_state_runtime import _epoch
    with db._read_ctx() as conn:
        _epoch(conn, epoch)


@pytest.mark.parametrize("pin", [
    # A hosted tenant naming this very path as its agent is still not this local store's.
    lambda home: Scope("acme", str(home), Principal("u"), owner=Group("support"), role="member"),
    lambda home: Scope("local", str(home.parent / "elsewhere"), Principal("owner")),
], ids=["another_tenant", "another_profile"])
def test_the_local_backend_refuses_any_pin_but_its_own_profile(ports, tmp_path, pin):
    db, storage = ports
    db.create_session("s1", "cli")
    scope = pin(tmp_path)
    with bind_state(scope) if not scope.is_local else _bound_local(scope):
        for call in (lambda: storage.sessions.get("s1"), lambda: storage.sessions.create("s2", "cli"),
                     lambda: storage.admission.list("s1"),
                     lambda: storage.admission.begin_epoch(instance_id="x")):
            with pytest.raises(StoragePinRefused):
                call()
    assert db.get_session("s2") is None


def test_with_no_scope_bound_the_pin_is_the_ambient_profile(ports, tmp_path, monkeypatch):
    db, storage = ports
    db.create_session("s1", "cli")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert storage.sessions.get("s1")["id"] == "s1"  # a local caller with no scope, as today
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "other"))
    with pytest.raises(StoragePinRefused):
        storage.sessions.get("s1")


def _bound_local(scope):
    """A local scope of another profile, bound the way a routed activity binds it."""
    from hermes_scope import push_state
    return push_state(scope=scope)


def test_a_turn_write_needs_its_claimed_turn(ports, tmp_path):
    db, storage = ports
    with bind_profile_scope(tmp_path):
        epoch = storage.admission.begin_epoch(instance_id="owner")
        from tests.hermes_state.test_target_advance_fence import _local_session
        sid = _local_session(db, epoch, tmp_path)
        storage.admission.admit(epoch=epoch, principal_id="human", session_id=sid, request_id="r",
                                payload={"text": "hi"})
        with pytest.raises(TurnFenceRequiredError):
            storage.sessions.append_turn(sid, [{"role": "user", "content": "unfenced"}])
        row = storage.admission.claim_next(epoch=epoch, session_id=sid)
        with bind_turn(sid, row["admission_id"], storage.admission.fence(row), store=db.db_path):
            assert storage.sessions.append_turn(sid, [{"role": "user", "content": "fenced"}]) == 1
            with pytest.raises(TurnFenceRequiredError):  # another session's turn is not this one
                storage.sessions.append_turn("other", [{"role": "user", "content": "x"}])
        settled = storage.admission.settle(epoch=epoch, admission_id=row["admission_id"],
                                           generation=row["generation"], outcome="completed", result={})
    assert settled["status"] == "terminal"
    assert [m["content"] for m in db.get_messages(sid)] == ["fenced"]
