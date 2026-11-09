"""A hosted scope's turn-path session writes route to its configured storage provider.

``sessions.storage`` names a provider plugin, discovered from ``$HERMES_HOME/plugins`` and
isolation-proven before its ports open. With one configured, the gateway turn's message appends
and the session authority's admission ledger run through that provider's ports, and nothing
lands in a local ``state.db`` — not the profile's own, and no store is opened at the ambient
home either. The provider is an error when it is missing, never a fallback to the local store.
"""
from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import threading
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.runtime_scope import bind_state, bind_turn
from agent.session_persistence import SessionPersistenceMixin
from agent.session_storage import session_storage
from gateway.config import Platform
from gateway.session_authority import SessionAuthority
from gateway.session_contract import SessionActor, SessionRef
from hermes_scope import Fence, Principal, Scope
from hermes_state import SessionDB
from hermes_state_port import SessionStorage, SessionStorageProvider

PROVIDER = "recording"

# A provider plugin that records every port call to ``calls.jsonl`` beside its own file and
# serves nothing locally: the wire shape the storage plugin kind loads from user plugins.
STUB_INIT = '''\
"""A recording session storage provider: every port call lands in calls.jsonl beside this file."""
import json
from pathlib import Path
from hermes_state_port import SessionStorage, SessionStorageProvider

_CALLS = Path(__file__).resolve().parent / "calls.jsonl"


def _record(op, **fields):
    with _CALLS.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"op": op, **fields}) + "\\n")


class Sessions:
    def append_turn(self, session_id, messages, *, turn_lease_holder=None, turn_lease_ttl_seconds=300.0):
        _record("append_turn", session_id=session_id, messages=messages,
                turn_lease_holder=turn_lease_holder, turn_lease_ttl_seconds=turn_lease_ttl_seconds)
        return len(messages)

    def create(self, session_id, source, **fields):
        _record("create", session_id=session_id)
        return session_id

    def get(self, session_id):
        _record("get_session", session_id=session_id)
        return None


class Admission:
    def begin_epoch(self, *, instance_id):
        _record("begin_epoch", instance_id=instance_id)
        return 1

    def recover(self, *, epoch):
        _record("recover", epoch=epoch)
        return 0

    def admit(self, *, epoch, principal_id, session_id, request_id, payload, intent="queue"):
        _record("admit", session_id=session_id, request_id=request_id)
        return {"admission_id": "stub", "status": "queued"}

    def get(self, admission_id):
        _record("get_admission", admission_id=admission_id)
        return None

    def find(self, *, principal_id, session_id, request_id):
        _record("find_admission", session_id=session_id, request_id=request_id)
        return None

    def list(self, session_id, *, pending_only=True):
        _record("list_admissions", session_id=session_id)
        return []

    def claim_next(self, *, epoch, session_id, preconditions=None):
        _record("claim_next", epoch=epoch, session_id=session_id)
        return None

    def cancel(self, *, epoch, admission_id):
        _record("cancel", epoch=epoch, admission_id=admission_id)
        return {"status": "terminal", "outcome": "cancelled"}

    def resolve_unknown(self, *, epoch, admission_id, generation):
        _record("resolve_unknown", admission_id=admission_id)
        return {"status": "terminal", "outcome": "unknown"}


class RecordingSessionStorage(SessionStorageProvider):
    """Every port call is recorded; nothing is served locally."""

    @property
    def name(self):
        return "recording"

    def open(self, db=None):
        return SessionStorage(sessions=Sessions(), admission=Admission())

    def assert_isolating(self):
        _record("assert_isolating")
'''


@pytest.fixture
def provider_home():
    """The deployment home: the user plugin lives under it and its config selects the backend."""
    home = Path(os.environ["HERMES_HOME"])
    plugin = home / "plugins" / PROVIDER
    plugin.mkdir(parents=True)
    (plugin / "__init__.py").write_text(STUB_INIT, encoding="utf-8")
    (home / "config.yaml").write_text(f"sessions:\n  storage: {PROVIDER}\n", encoding="utf-8")
    return home


@pytest.fixture
def served(tmp_path):
    """The profile the authority serves: its config names the provider, and its state.db is the
    local store nothing must land in."""
    home = tmp_path / "served"
    home.mkdir()
    (home / "config.yaml").write_text(f"sessions:\n  storage: {PROVIDER}\n", encoding="utf-8")
    return home


def _recorded(provider_home) -> list:
    calls = provider_home / "plugins" / PROVIDER / "calls.jsonl"
    if not calls.exists():
        return []
    return [json.loads(line) for line in calls.read_text(encoding="utf-8").splitlines()]


def _local_rows(served: Path, table: str) -> int:
    with closing(sqlite3.connect(served / "state.db")) as conn:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def _hosted_scope() -> Scope:
    return Scope("tenant-a", "agent-x", Principal("member"),
                 owner=Principal("owner"), can_administer=True)


def test_the_provider_is_isolation_proven_before_its_ports_open(tmp_path, provider_home, served):
    with closing(SessionDB(served / "state.db")) as db:
        db.create_session("s1", source="tui")
        storage = session_storage(db)
        assert storage is not None
        ops = {call["op"] for call in _recorded(provider_home)}
        assert "assert_isolating" in ops
        # Its ports serve the provider, and nothing was written to the local store.
        assert storage.admission.list("s1") == []
        assert _local_rows(served, "session_admissions") == 0


def test_a_hosted_scopes_turn_rows_reach_the_provider_and_no_local_store(tmp_path, provider_home):
    """The turn-path append: a claimed turn of a hosted scope flushes through the provider's
    append_turn; no state.db is opened at the ambient home."""
    agent = SimpleNamespace(
        session_id="s1", _session_db=None, _session_db_created=False,
        _ensure_db_session=lambda: None, _persist_disabled=False,
    )
    messages = [{"role": "user", "content": "hello"}]
    with bind_state(_hosted_scope()), bind_turn("s1", "t1", Fence(1, 1)):
        assert SessionPersistenceMixin._flush_messages_to_session_db_unlocked(agent, messages) is True
    appends = [call for call in _recorded(provider_home) if call["op"] == "append_turn"]
    assert [(call["session_id"], [m["content"] for m in call["messages"]]) for call in appends] == \
        [("s1", ["hello"])]
    assert not (Path(os.environ["HERMES_HOME"]) / "state.db").exists()


@pytest.mark.asyncio
async def test_a_hosted_scopes_admission_ledger_reaches_the_provider(tmp_path, provider_home, served):
    """The authority's admission calls (its FIFO reads, the claim, a cancel, the receipt read)
    run through the provider; the profile's local store keeps no admission row or message."""
    with closing(SessionDB(served / "state.db")) as db:
        db.create_session("s1", source="tui")
        runner = SimpleNamespace(_draining=False, _persist_active_agents=None)
        authority = SessionAuthority(runner, profile_id="served", instance_id="inst", db=db, epoch=3)
        ref = SessionRef("served", "s1")
        live = SimpleNamespace(
            claim_gate=asyncio.Lock(), mutation_lock=asyncio.Lock(), rescan=False,
            source=SimpleNamespace(platform=Platform("webhook")), route="r", detached=False,
            event_stream=SimpleNamespace(lock=threading.RLock(), execution={},
                                        publish=lambda *a, **k: None),
            settling={},
        )
        authority.sessions["s1"] = live
        actor = SessionActor("member", "served", frozenset({"session:submit", "session:control"}), "t")

        with bind_state(_hosted_scope()):
            # The FIFO read a paused drain releases its waiters over.
            authority._pause(ref, "testing")
            # The claim: the turn-start write of the FIFO (it takes the session's own locks).
            claimed = await authority._claim_next(ref, live)
            assert claimed == (None, None)
            # The receipt read: the provider's answer, not the local store's.
            from hermes_state_runtime import RuntimeStoreError
            with pytest.raises(RuntimeStoreError) as refused:
                await authority.receipt(actor, ref, "a1")
            assert refused.value.reason == "not_found"
            # A cancelled queued admission commits through the provider.
            live.task = asyncio.get_running_loop().create_future()  # never done: no drain spawns
            await authority._commit_cancel(ref, "a1", SimpleNamespace(status="queued"))

        ops = [call["op"] for call in _recorded(provider_home)]
        assert "list_admissions" in ops
        assert "claim_next" in ops
        assert "get_admission" in ops
        assert "cancel" in ops
        assert _local_rows(served, "session_admissions") == 0
        assert _local_rows(served, "messages") == 0


def test_a_provider_missing_is_an_error_never_a_local_store(tmp_path, served):
    (served / "config.yaml").write_text("sessions:\n  storage: missing\n", encoding="utf-8")
    with closing(SessionDB(served / "state.db")) as db:
        db.create_session("s1", source="tui")
        with pytest.raises(LookupError, match="missing"):
            session_storage(db)
