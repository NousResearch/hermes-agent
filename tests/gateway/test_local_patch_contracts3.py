"""Contracts for the boot-intent local patch (L2 restart continuity).

These pin behaviour upstream does NOT provide: an intent registered before a planned restart is
claimed exactly once at boot and never double-runs a session the resume path is already reviving.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from gateway import boot_intents as store
from gateway.run_startup import GatewayStartupMixin


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "get_hermes_home", lambda: tmp_path)
    return tmp_path


# --- store ---------------------------------------------------------------------

def test_add_and_list(home):
    item = store.add_intent("k1", "do X", ttl_seconds=60)
    rows = store.list_intents()
    assert len(rows) == 1
    assert rows[0]["id"] == item["id"]
    assert rows[0]["session_key"] == "k1"
    assert rows[0]["prompt"] == "do X"


def test_add_requires_key_and_prompt(home):
    with pytest.raises(ValueError):
        store.add_intent("", "x")
    with pytest.raises(ValueError):
        store.add_intent("k", "   ")
    with pytest.raises(ValueError):
        store.add_intent("k", "x", ttl_seconds=0)


def test_claim_is_single_consumer(home):
    item = store.add_intent("k1", "do X")
    first = store.claim_intents()
    second = store.claim_intents()
    assert [i["id"] for i in first] == [item["id"]]
    assert second == []


def test_claim_retakes_abandoned_then_not_before_timeout(home):
    store.add_intent("k1", "do X")
    assert len(store.claim_intents(now=1000.0)) == 1
    assert store.claim_intents(now=1000.0 + store.CLAIM_TIMEOUT_SECONDS - 1) == []
    assert len(store.claim_intents(now=1000.0 + store.CLAIM_TIMEOUT_SECONDS + 1)) == 1


def test_expired_never_fires(home):
    store.add_intent("k1", "do X", ttl_seconds=10, now=1000.0)
    assert store.claim_intents(now=1000.0 + 11) == []
    rows = store.list_intents(include_consumed=True)
    assert rows[0]["result"] == "expired"


def test_complete_and_clear(home):
    item = store.add_intent("k1", "do X")
    store.claim_intents()
    assert store.complete_intent(item["id"], "done") is True
    assert store.list_intents() == []
    assert store.list_intents(include_consumed=True)[0]["result"] == "done"
    assert store.clear_intents() == 1


def test_corrupt_store_is_empty_not_fatal(home):
    (home / "boot_intents.json").write_text("{not json", encoding="utf-8")
    assert store.list_intents() == []
    assert store.claim_intents() == []
    store.add_intent("k", "x")
    assert list(home.glob("boot_intents.corrupt-*.json")), "corrupt store must be quarantined, not overwritten"


def test_has_pending_intent(home):
    assert store.has_pending_intent("k1") is False
    item = store.add_intent("k1", "do X")
    assert store.has_pending_intent("k1") is True
    store.complete_intent(item["id"])
    assert store.has_pending_intent("k1") is False


# --- boot scheduler ------------------------------------------------------------

def _runner(*, entry=True, adapter=True, running=False, authorized=True, tripped=False, stub_intent=True):
    runner = GatewayStartupMixin()
    runner._startup_should_abort = lambda: False
    runner._restart_loop_is_tripped = lambda: tripped
    runner._session_entry_for_key = lambda key: SimpleNamespace(origin=object()) if entry else None
    runner._restored_source = lambda e: SimpleNamespace(platform="feishu")
    runner._delivery_adapter_for = lambda s: object() if adapter else None
    runner._resume_owner_authorized = lambda key, source: authorized
    runner._is_session_running = lambda key: running
    runner._session_state = lambda key: SimpleNamespace(turn=SimpleNamespace(agent=None, started_ts=None))
    runner._persist_active_agents = lambda: None
    runner._track_startup_restore_task = lambda task: None
    runner._tasks = []
    runner._retain_background_task = lambda task: (runner._tasks.append(task), task)[1]
    runner.session_store = SimpleNamespace(clear_resume_pending=lambda key: True)

    async def _noop(*args, **kwargs):
        return None

    if stub_intent:
        runner._run_boot_intent_event = _noop
    return runner


def _invoke(runner, **kwargs):
    async def _go():
        count = runner._schedule_boot_intents(**kwargs)
        await asyncio.sleep(0)
        return count

    return asyncio.run(_go())


@pytest.fixture()
def scheduler_env(home, monkeypatch):
    import gateway.run_startup as rs

    monkeypatch.setattr(rs, "note_running_chat_owner", lambda *a, **k: None)
    monkeypatch.setattr(rs, "MessageEvent", lambda **kw: SimpleNamespace(**kw))
    return home


def test_scheduler_runs_one_intent(scheduler_env):
    store.add_intent("k1", "do X")
    runner = _runner()
    assert _invoke(runner) == 1


def test_scheduler_preempts_resume_pending_marker(scheduler_env):
    """An explicit intent wins over the generic resume note for the same session."""
    store.add_intent("k1", "do X")
    cleared = []
    runner = _runner()
    runner.session_store = SimpleNamespace(
        clear_resume_pending=lambda key: cleared.append(key) or True)
    assert _invoke(runner) == 1
    assert cleared == ["k1"]


def test_scheduler_releases_when_adapter_down(scheduler_env):
    store.add_intent("k1", "do X")
    runner = _runner(adapter=False)
    assert _invoke(runner) == 0
    # released, not consumed: a reconnect rescan can still claim it
    assert len(store.list_intents()) == 1
    assert _invoke(_runner()) == 1


def test_scheduler_skips_running_session(scheduler_env):
    store.add_intent("k1", "do X")
    assert _invoke(_runner(running=True)) == 0
    assert len(store.list_intents()) == 1  # released, retried next boot


def test_scheduler_skips_unauthorized(scheduler_env):
    store.add_intent("k1", "do X")
    assert _invoke(_runner(authorized=False)) == 0
    assert store.list_intents(include_consumed=True)[0]["result"] == "skipped: unauthorized"


def test_scheduler_skips_missing_session_entry(scheduler_env):
    store.add_intent("k1", "do X")
    assert _invoke(_runner(entry=False)) == 0
    assert store.list_intents(include_consumed=True)[0]["result"] == "skipped: no session entry"


def test_scheduler_breaker_blocks_everything(scheduler_env):
    store.add_intent("k1", "do X")
    assert _invoke(_runner(tripped=True)) == 0
    assert len(store.list_intents()) == 1  # still pending, next clean boot runs it


def test_scheduler_runs_at_most_one_per_boot(scheduler_env):
    store.add_intent("k1", "do X")
    store.add_intent("k2", "do Y")
    assert _invoke(_runner()) == 1
    rows = {row["session_key"]: row for row in store.list_intents()}
    assert len(rows) == 2                       # neither consumed
    assert rows["k1"]["claimed_at"] is not None  # the one that ran
    assert rows["k2"]["claimed_at"] is None      # released for the next boot


# --- dispatch semantics --------------------------------------------------------

def test_boot_intent_records_dispatch_not_completion(scheduler_env):
    """The row says dispatched, never done: the turn outlives the sentinel."""
    item = store.add_intent("k1", "do X")
    store.claim_intents()

    async def _noop_resume(*args, **kwargs):
        return None

    runner = _runner(stub_intent=False)
    runner._run_startup_resume_event = _noop_resume
    runner._is_session_running = lambda key: True   # still running: must not change the verdict

    asyncio.run(runner._run_boot_intent_event(object(), object(), "k1", item["id"]))
    assert store.list_intents(include_consumed=True)[0]["result"] == "dispatched"


def test_boot_intent_cancel_releases_the_claim(scheduler_env):
    item = store.add_intent("k1", "do X")
    store.claim_intents()

    async def _cancelled(*args, **kwargs):
        raise asyncio.CancelledError()

    runner = _runner(stub_intent=False)
    runner._run_startup_resume_event = _cancelled
    runner._is_session_running = lambda key: False

    async def _go():
        with pytest.raises(asyncio.CancelledError):
            await runner._run_boot_intent_event(object(), object(), "k1", item["id"])

    asyncio.run(_go())
    assert store.list_intents()[0]["claimed_at"] is None   # retried on the next boot


# --- /restart continuity -------------------------------------------------------

def test_same_restart_route_matches_platform_chat_and_thread():
    from gateway.slash_commands import _same_restart_route

    src = SimpleNamespace(platform="feishu", chat_id="oc_1", thread_id="omt_1")
    assert _same_restart_route(SimpleNamespace(platform="feishu", chat_id="oc_1", thread_id="omt_1"), src)
    assert not _same_restart_route(SimpleNamespace(platform="feishu", chat_id="oc_1", thread_id="omt_2"), src)
    assert not _same_restart_route(SimpleNamespace(platform="telegram", chat_id="oc_1", thread_id="omt_1"), src)
    assert not _same_restart_route(None, src)


def test_restart_command_registers_an_intent_for_its_own_session(scheduler_env):
    """A user-typed /restart needs no pre-registration: the command writes the intent itself."""
    import threading

    from gateway.slash_commands import GatewaySlashCommandsMixin

    entry = SimpleNamespace(origin=SimpleNamespace(
        platform="feishu", chat_id="oc_1", thread_id="omt_1"))
    runner = object.__new__(GatewaySlashCommandsMixin)
    runner.session_store = SimpleNamespace(
        _lock=threading.Lock(),
        _entries={"k1": entry},
        _ensure_loaded_locked=lambda: None,
    )
    event = SimpleNamespace(source=SimpleNamespace(
        platform="feishu", chat_id="oc_1", thread_id="omt_1", profile=None))

    asyncio.run(runner._register_restart_boot_intent(event))

    rows = store.list_intents()
    assert len(rows) == 1
    assert rows[0]["session_key"] == "k1"
    assert rows[0]["source"] == "restart_command"
    assert "Do not re-run" in rows[0]["prompt"]


def test_restart_command_without_a_session_registers_nothing(scheduler_env):
    import threading

    from gateway.slash_commands import GatewaySlashCommandsMixin

    runner = object.__new__(GatewaySlashCommandsMixin)
    runner.session_store = SimpleNamespace(
        _lock=threading.Lock(), _entries={}, _ensure_loaded_locked=lambda: None)
    event = SimpleNamespace(source=SimpleNamespace(
        platform="feishu", chat_id="oc_missing", thread_id=None, profile=None))

    asyncio.run(runner._register_restart_boot_intent(event))
    assert store.list_intents() == []
