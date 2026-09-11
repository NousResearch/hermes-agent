"""Pin BASE (main) semantics at the lock-fallback sites and the ``_is_openai_client_closed`` truth table.

Review-fix for the simplify-codebase PR: the refactor briefly rewrote several ``lock = getattr(agent,
"_x_lock", None); if lock is not None: with lock: <direct attr read>`` blocks as ``with getattr(...) or
nullcontext(): getattr(agent, slot, None)``, which (a) silently runs the critical section unlocked and (b)
turns a missing slot under the lock from a loud AttributeError into a silent None. These tests hold the
original shape: an initialized agent (lock present) reads its slots directly and fails loud; only
``object.__new__`` stubs (no lock) get the getattr fallback.
"""
from __future__ import annotations

import threading
from unittest.mock import Mock

import pytest

from agent.agent_runtime_helpers import _requeue_pending_steer
from agent.client_lifecycle import ClientLifecycleMixin
from agent.interrupt_control import InterruptControlMixin
from agent.session_persistence import SessionPersistenceMixin
from agent.stream_delivery import StreamDeliveryMixin
from gateway.run_agent_cache import GatewayAgentCacheMixin
from gateway.slash_commands import GatewaySlashCommandsMixin


class _RecLock:
    def __init__(self):
        self.entered = 0
        self._lock = threading.Lock()

    def __enter__(self):
        self.entered += 1
        return self._lock.__enter__()

    def __exit__(self, *exc):
        return self._lock.__exit__(*exc)


class _Bare:
    pass


# ---------------------------------------------------------------- steer requeue


def test_requeue_pending_steer_locked_path_reads_slot_directly_and_fails_loud():
    agent = _Bare()
    agent._pending_steer_lock = _RecLock()
    with pytest.raises(AttributeError):
        _requeue_pending_steer(agent, "new")  # lock present but slot missing => real bug, loud
    agent._pending_steer = "old"
    _requeue_pending_steer(agent, "new")
    assert agent._pending_steer == "old\nnew"
    assert agent._pending_steer_lock.entered >= 1


def test_requeue_pending_steer_unlocked_stub_fallback():
    agent = _Bare()  # object.__new__-style stub: no lock, no slot
    _requeue_pending_steer(agent, "new")
    assert agent._pending_steer == "new"


# ------------------------------------------------------- interrupt_control slots


@pytest.mark.parametrize(
    "method, lock_attr, slot",
    [
        ("steer", "_pending_steer_lock", "_pending_steer"),
        ("_drain_pending_steer", "_pending_steer_lock", "_pending_steer"),
        ("_has_pending_redirect", "_pending_redirect_lock", "_pending_redirect"),
        ("_drain_pending_redirect", "_pending_redirect_lock", "_pending_redirect"),
    ],
)
def test_interrupt_control_slot_reads_fail_loud_only_under_lock(method, lock_attr, slot):
    fn = getattr(InterruptControlMixin, method)
    call = (lambda a: fn(a, "new")) if method == "steer" else fn

    locked = _Bare()
    setattr(locked, lock_attr, threading.Lock())
    with pytest.raises(AttributeError):
        call(locked)

    unlocked = _Bare()
    call(unlocked)  # no lock: stub fallback, no raise

    setattr(locked, slot, "old")
    call(locked)  # slot present: normal path


def test_steer_and_drain_roundtrip_with_lock():
    agent = _Bare()
    agent._pending_steer_lock = threading.Lock()
    agent._pending_steer = None
    assert InterruptControlMixin.steer(agent, "a")
    assert InterruptControlMixin.steer(agent, "b")
    assert InterruptControlMixin._drain_pending_steer(agent) == "a\nb"
    assert agent._pending_steer is None


# ---------------------------------------------------------- session persistence


def test_flush_messages_uses_lock_when_present_and_runs_unlocked_for_stubs():
    class P(_Bare):
        def _flush_messages_to_session_db_unlocked(self, messages, conversation_history=None):
            return ("called", len(messages))

    locked = P()
    locked._session_persist_lock = _RecLock()
    assert SessionPersistenceMixin._flush_messages_to_session_db(locked, [{"role": "user"}]) == ("called", 1)
    assert locked._session_persist_lock.entered == 1

    stub = P()
    assert SessionPersistenceMixin._flush_messages_to_session_db(stub, [{"role": "user"}]) == ("called", 1)


# ---------------------------------------------------------------- stream writer


def test_stream_writer_lock_is_created_unconditionally_at_init():
    from agent.agent_init import _STREAM_STATE

    assert _STREAM_STATE["_stream_writer_lock"] is threading.Lock
    # The lazy path is only for __new__-built stubs and never replaces an existing lock.
    agent = StreamDeliveryMixin.__new__(StreamDeliveryMixin)
    agent._stream_writer_lock = existing = threading.Lock()
    StreamDeliveryMixin._ensure_stream_writer_state(agent)
    assert agent._stream_writer_lock is existing


# ------------------------------------------------------- gateway agent-cache sites


class _Runner(_Bare):
    def _peek_session_state(self, key):
        return None

    def _running_agent_ids(self):
        return set()

    def _spawn_release_thread(self, *a, **k):
        pass

    def _release_evicted_agent_soft(self, *a):
        pass


def test_cached_agent_for_reads_only_under_lock():
    runner = _Runner()
    runner._agent_cache = {"k": ("AGENT", "sig")}
    assert GatewaySlashCommandsMixin._cached_agent_for(runner, "k") is None  # no lock: no unlocked read
    runner._agent_cache_lock = _RecLock()
    assert GatewaySlashCommandsMixin._cached_agent_for(runner, "k") == "AGENT"
    assert runner._agent_cache_lock.entered == 1


def test_evict_cached_agent_reads_cache_directly_when_locked():
    runner = _Runner()
    runner._agent_cache_lock = threading.Lock()
    with pytest.raises(AttributeError):
        GatewayAgentCacheMixin._evict_cached_agent(runner, "k")  # lock present, cache missing => loud
    runner._agent_cache = {"k": ("AGENT", "sig")}
    GatewayAgentCacheMixin._evict_cached_agent(runner, "k")
    assert runner._agent_cache == {}
    stub = _Runner()
    stub._agent_cache = {"k": ("AGENT", "sig")}
    GatewayAgentCacheMixin._evict_cached_agent(stub, "k")  # lock-less test runner: evicts lock-free
    assert stub._agent_cache == {}


# ------------------------------------------------------ _is_openai_client_closed


class _Inner:
    def __init__(self, is_closed):
        self.is_closed = is_closed


def _duck(outer=None, inner=None, *, has_outer=True, has_inner=True):
    d = _Bare()
    if has_outer:
        d.is_closed = outer
    if has_inner:
        d._client = inner
    return d


@pytest.mark.parametrize(
    "client, expected",
    [
        # the reviewer's 4 combos: duck outer is_closed True/False, with/without _client
        (_duck(True, has_inner=False), True),
        (_duck(False, has_inner=False), False),
        (_duck(True, _Inner(False)), True),  # outer says closed => closed, inner not consulted
        (_duck(False, _Inner(True)), True),  # outer open, inner closed => inner is the answer
        (_duck(False, _Inner(False)), False),
        (_duck(lambda: True, _Inner(False)), True),  # openai.OpenAI.is_closed() method form
        (_duck(lambda: False, _Inner(True)), True),
        (_duck(lambda: False, _Inner(False)), False),
        (_duck(lambda: False, has_inner=False), False),
        (_duck(None, _Inner(True)), True),  # outer None => fall through to inner
        (_duck(None, None), False),
        (_duck(has_outer=False, inner=_Inner(True)), True),
        (_duck(has_outer=False, has_inner=False), False),
        (Mock(), False),
    ],
)
def test_is_openai_client_closed_truth_table(client, expected):
    assert ClientLifecycleMixin._is_openai_client_closed(client) is expected


def test_cached_agent_for_lockless_fallback_only_when_requested():
    # Manual codex /compress historically read the cache lock-free when the lock was absent.
    runner = _Runner()
    runner._agent_cache = {"k": ("AGENT", "sig")}
    runner._agent_cache_lock = None
    assert GatewaySlashCommandsMixin._cached_agent_for(runner, "k") is None
    assert GatewaySlashCommandsMixin._cached_agent_for(runner, "k", lockless_fallback=True) == "AGENT"
