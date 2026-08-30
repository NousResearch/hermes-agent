"""Regressions for #76354 review F3/F4/F5 — worker isolation, durable lease
cancellation, and session ContextVar repair.

F3: a timed-out worker running an IN-PLACE-MUTATING context engine must not
be able to touch the caller's live transcript — assertions run WHILE the
worker is still blocked inside the engine (released only afterwards).

F4: the reviewer's exact 5-step regression — block summary indefinitely →
host timeout → NEW compressor acquires the durable lock while the old
summary is STILL blocked → release old worker → prove it cannot clear
cooldown / release the new holder's lease / publish state.

F5: after a successful out-of-place rotation, the CALLER's session
ContextVar resolves to the child id (get_session_env / HERMES_SESSION_ID).
"""

from __future__ import annotations

import copy
import os
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from hermes_state import SessionDB


def _build_agent_with_db(db: SessionDB, session_id: str, **compressor_kwargs):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )

    compressor = MagicMock()
    compressor.compress.return_value = [
        {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
        {"role": "user", "content": "tail"},
    ]
    compressor.compression_count = 1
    compressor.last_prompt_tokens = 0
    compressor.last_completion_tokens = 0
    compressor._last_summary_error = None
    compressor._last_compress_aborted = False
    compressor._last_aux_model_failure_model = None
    compressor._last_aux_model_failure_error = None
    compressor._last_compression_made_progress = True
    compressor._last_summary_fallback_used = False
    agent.context_compressor = compressor
    # The compressor is a stub — the one-time compression-model feasibility
    # probe would resolve a REAL auxiliary provider (credential pools, live
    # token exchange) before the engine runs. In hermetic CI there are no
    # credentials, so the probe aborts compression before the stub engine
    # ever starts and every blocked-state assertion goes vacuous. These
    # tests exercise isolation/fencing, never aux-model feasibility.
    agent._compression_feasibility_checked = True
    return agent


def test_f3_mutating_engine_cannot_touch_live_transcript_after_timeout(
    tmp_path: Path, monkeypatch
) -> None:
    """In-place-mutating engine + host timeout → caller transcript untouched.

    Byte-identity is asserted WHILE the worker is still blocked inside the
    engine; the worker is released only after those assertions.
    """
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "F3_ISOLATION"
    db.create_session(session_id, source="cli")
    agent = _build_agent_with_db(db, session_id)
    agent._cached_system_prompt = "sys"

    # Fast host timeout for the owned wrapper.
    monkeypatch.setattr(
        "agent.conversation_compression.resolve_context_compression_timeouts",
        lambda cfg=None: (0.6, 1.2),
    )

    engine_started = threading.Event()
    release_engine = threading.Event()
    mutated_lists = []

    def _mutating_engine(msgs, **_kwargs):
        # Legacy/plugin-engine contract: mutate the input list IN PLACE.
        engine_started.set()
        msgs[:] = [{"role": "assistant", "content": "ENGINE GARBAGE"}]
        mutated_lists.append(msgs)
        assert release_engine.wait(timeout=30)
        return msgs

    agent.context_compressor.compress.side_effect = _mutating_engine

    live = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    baseline = copy.deepcopy(live)

    try:
        returned, _sp = agent._compress_context(
            live, "sys", approx_tokens=120_000
        )
        # Host timed out and returned while the engine is STILL blocked.
        assert engine_started.wait(timeout=5)
        assert not release_engine.is_set()
        assert returned is live
        # ── The core assertion, made while the worker keeps running ──────
        assert live == baseline, (
            "live transcript mutated by a detached compression worker"
        )
        # The engine did mutate a list — the SNAPSHOT, not the caller's.
        assert mutated_lists and mutated_lists[0] is not live
        # Give the blocked worker extra time to prove no delayed publication.
        time.sleep(0.2)
        assert live == baseline
    finally:
        release_engine.set()
    # After the late worker finishes, the live transcript must STILL be
    # untouched (publication only on admitted commit — which was cancelled).
    deadline = time.time() + 5
    while time.time() < deadline and db.get_compression_lock_holder(session_id):
        time.sleep(0.02)
    assert live == baseline


def test_host_timeout_releases_pool_slot_while_protected_provider_is_still_blocked(
    tmp_path: Path, monkeypatch
) -> None:
    """Fence timeout must unwind the compression owner, not occupy the pool.

    Protected auxiliary calls isolate their provider stream on a daemon thread.
    The compression owner must observe its commit-fence cancellation and unwind
    immediately; otherwise four slow streams consume all four shared compression
    workers until the auxiliary stream's much longer absolute ceiling expires.
    """
    from agent import auxiliary_client as aux
    from agent import conversation_compression as cc

    deadline = time.time() + 5
    while time.time() < deadline:
        with cc._compress_admission_lock:
            if cc._compress_admitted_count == 0:
                break
        time.sleep(0.02)
    with cc._compress_admission_lock:
        assert cc._compress_admitted_count == 0

    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "F3_PROVIDER_OWNER_RELEASE"
    db.create_session(session_id, source="cli")
    agent = _build_agent_with_db(db, session_id)
    agent._cached_system_prompt = "sys"
    monkeypatch.setattr(
        "agent.conversation_compression.resolve_context_compression_timeouts",
        lambda cfg=None: (0.05, 0.1),
    )

    provider_started = threading.Event()
    release_provider = threading.Event()

    def _blocked_provider(_kwargs):
        provider_started.set()
        assert release_provider.wait(timeout=10)
        return "late-provider-result"

    def _compress_with_protected_provider(msgs, **_kwargs):
        aux._run_protected_sync_provider_call(_blocked_provider, {})
        return msgs

    agent.context_compressor.compress.side_effect = _compress_with_protected_provider
    live = [{"role": "user", "content": f"m{i}"} for i in range(20)]

    try:
        returned, _sp = agent._compress_context(
            live, "sys", approx_tokens=120_000
        )
        assert returned is live
        assert provider_started.wait(timeout=1)
        assert not release_provider.is_set()

        deadline = time.time() + 1
        while time.time() < deadline:
            with cc._compress_admission_lock:
                if cc._compress_admitted_count == 0:
                    break
            time.sleep(0.01)
        with cc._compress_admission_lock:
            assert cc._compress_admitted_count == 0, (
                "timed-out compression owner retained its shared pool slot "
                "while the isolated provider stream was still blocked"
            )
    finally:
        release_provider.set()
        deadline = time.time() + 5
        while time.time() < deadline:
            with cc._compress_admission_lock:
                if cc._compress_admitted_count == 0:
                    break
            time.sleep(0.02)


def test_f4_five_step_stale_holder_regression(tmp_path: Path) -> None:
    """Reviewer's exact 5-step durable-lease regression (#76354 F4).

    1. Block the original summary indefinitely.
    2. Let the host time out.
    3. Prove another compressor can acquire the durable lock BEFORE the
       original summary is released.
    4. Release the old worker.
    5. Prove it cannot clear cooldown, release the new holder's lease, or
       publish stale state.
    """
    from agent.conversation_compression import (
        CompressionCommitFence,
        run_compress_context_with_progress_timeout,
    )

    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "F4_FIVE_STEP"
    db.create_session(session_id, source="telegram")
    db.append_message(session_id, "user", "original durable")

    agent = _build_agent_with_db(db, session_id)
    agent.compression_in_place = True
    agent._cached_system_prompt = "sys"

    summary_started = threading.Event()
    release_summary = threading.Event()

    def _blocked_summary(*_args, **_kwargs):
        summary_started.set()
        assert release_summary.wait(timeout=30)  # step 1: blocked
        return [
            {"role": "user", "content": "[CONTEXT COMPACTION] stale summary"},
            {"role": "user", "content": "tail"},
        ]

    agent.context_compressor.compress.side_effect = _blocked_summary
    # Track cooldown-clear attempts on the OLD worker's compressor.
    cooldown_cleared = []
    agent.context_compressor._clear_compression_failure_cooldown = (
        lambda: cooldown_cleared.append(True)
    )

    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]

    def _worker(fence):
        return agent._compress_context(
            messages, "sys", approx_tokens=120_000, commit_fence=fence
        )

    # Step 2: host-owned progress wait times out while summary is blocked.
    result_msgs, _prompt = run_compress_context_with_progress_timeout(
        worker=_worker,
        messages=messages,
        system_prompt_fallback="fallback",
        idle_timeout_seconds=0.6,
        total_ceiling_seconds=1.2,
    )
    assert summary_started.wait(timeout=5)
    assert not release_summary.is_set()  # old worker STILL blocked
    assert result_msgs is messages

    # Step 3: a NEW compressor acquires the durable lock while the old
    # summary remains blocked. The host's holder-qualified release freed
    # the old lease (refresher stopped + row deleted, holder-scoped).
    new_holder = "pid:new:contender"
    deadline = time.time() + 5
    acquired = False
    while time.time() < deadline:
        if db.try_acquire_compression_lock(session_id, new_holder, ttl_seconds=60):
            acquired = True
            break
        time.sleep(0.02)
    assert acquired, (
        "a new compressor must be able to acquire the durable lock while "
        "the timed-out worker is still blocked in its summary"
    )
    assert not release_summary.is_set()  # provably still step-3 state
    assert db.get_compression_lock_holder(session_id) == new_holder

    pre_release_rows = db.get_messages_as_conversation(session_id)

    # Step 4: release the old worker.
    release_summary.set()
    # Wait for the late worker to fully unwind (it must NOT touch the lock).
    deadline = time.time() + 5
    while time.time() < deadline:
        if db.get_compression_lock_holder(session_id) != new_holder:
            break  # would be a failure — checked below
        if cooldown_cleared:
            break
        time.sleep(0.02)
    time.sleep(0.3)  # settle: give the stale worker every chance to misbehave

    # Step 5a: it cannot clear the cooldown.
    assert not cooldown_cleared, (
        "late cancelled worker cleared the compression failure cooldown"
    )
    # Step 5b: it cannot release the NEW holder's lease (holder-qualified).
    assert db.get_compression_lock_holder(session_id) == new_holder, (
        "late worker released the replacement holder's durable lease (ABA)"
    )
    # Step 5c: it cannot publish stale state — transcript unchanged, no
    # in-place compaction landed, session id did not rotate.
    post_release_rows = db.get_messages_as_conversation(session_id)
    assert post_release_rows == pre_release_rows
    assert agent.session_id == session_id
    db.release_compression_lock(session_id, new_holder)


def test_f5_session_contextvar_rebound_after_rotation(
    tmp_path: Path, monkeypatch
) -> None:
    """Post-compression tool reads of HERMES_SESSION_ID see the CHILD id."""
    from gateway.session_context import (
        clear_session_vars,
        get_session_env,
        set_session_vars,
    )

    db = SessionDB(db_path=tmp_path / "state.db")
    parent_sid = "F5_CTXVAR_PARENT"
    db.create_session(parent_sid, source="telegram")
    agent = _build_agent_with_db(db, parent_sid)
    agent.compression_in_place = False  # rotation mode
    agent._cached_system_prompt = "sys"

    # Enable the owned pooled wrapper so rotation happens on a WORKER thread
    # (the caller's ContextVar can only be repaired by the caller).
    monkeypatch.setattr(
        "agent.conversation_compression.resolve_context_compression_timeouts",
        lambda cfg=None: (5.0, 10.0),
    )

    # Simulate the gateway's bound session context for the caller.
    tokens = set_session_vars(session_id=parent_sid, platform="telegram")
    try:
        assert get_session_env("HERMES_SESSION_ID") == parent_sid

        messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
        agent._compress_context(messages, "sys", approx_tokens=120_000)

        assert agent.session_id != parent_sid  # rotation happened
        # ── The F5 contract: caller-context reads resolve to the child ──
        assert get_session_env("HERMES_SESSION_ID") == agent.session_id, (
            "caller's session ContextVar still returns the parent id after "
            "an out-of-place compression rotation"
        )
    finally:
        clear_session_vars(tokens)
