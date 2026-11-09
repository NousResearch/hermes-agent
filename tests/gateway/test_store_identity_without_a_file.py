"""Callers that identify a store reach it by its profile scope, so a provider with no store file still works.

``store_location`` answers ``None`` for a provider whose store is not a file. These cases replace it with that answer
while the store itself stays the SQLite one, then drive the delivery ledger, the async-delegation ledger, a worktree's
retention record and the gateway's session-authority bring-up, which must reach the same stores they reach with a
file. A feature that works on the file itself refuses and says it needs one.
"""
import time

import pytest

import persistence_assembly
from tests.gateway import test_profile_reserve_reopen_warning as reopen

served = reopen.served


@pytest.fixture
def no_store_file(monkeypatch):
    monkeypatch.setattr(persistence_assembly, "store_location", lambda scope: None)


def test_the_delivery_ledger_records_and_reads_without_a_store_file(no_store_file):
    from gateway import delivery_ledger as dl
    dl.record_obligation(obligation_id="ob-1", session_key="agent:main:slack:channel:C1", platform="slack",
                         chat_id="C1", thread_id=None, content="the final answer")
    dl.mark_failed("ob-1", "boom")
    assert [entry["platform"] for entry in dl.pending_retries()] == ["slack"]


def test_the_delegation_ledger_records_and_reads_without_a_store_file(no_store_file):
    from tools import async_delegation as ad
    assert ad.restore_undelivered_completions(None) == 0
    ad._persist_dispatch({"delegation_id": "d-1", "session_key": "agent:main:cli", "dispatched_at": time.time(),
                          "goal": "look"})
    assert ad.get_durable_delegation("d-1")["state"] == "running"


def test_a_worktree_lock_names_the_profile_home_of_a_store_without_a_file(no_store_file):
    from hermes_cli.worktree_ops import _retained_store, _retention_store
    from hermes_constants import get_hermes_home
    from persistence_assembly import open_persistence, profile_scope
    scope = profile_scope(home=get_hermes_home())
    assert _retention_store() == scope.profile_dir
    assert _retained_store(str(scope.profile_dir)) is None
    with open_persistence(scope, eager=True):
        pass
    assert _retained_store(str(scope.profile_dir)) == scope.profile_dir


def test_setting_the_journal_mode_refuses_a_store_without_a_file(no_store_file, capsys):
    from types import SimpleNamespace
    from hermes_cli.sessions_cmd_journal_mode import cmd_set_journal_mode
    assert cmd_set_journal_mode(SimpleNamespace(mode="wal", db=None, force=False)) == 1
    assert "needs a file-backed session store" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_each_profile_authority_comes_up_on_its_own_store_without_a_store_file(served, no_store_file):
    from gateway.run_runtime import initialize_gateway_runtime
    runner, alpha = served
    await initialize_gateway_runtime(runner)
    authorities = list(runner.session_authorities)
    assert len({authority.persistence.scope for authority in authorities}) == 2
    assert runner.session_authorities.for_home(alpha.resolve()).persistence.scope.profile_dir == alpha.resolve()
    epochs = runner.session_store._local_authority_epochs
    assert sorted(epochs.values()) == sorted(authority.epoch for authority in authorities)
