"""Gateway-facing race hardening for background-prepared compression (task 7).

Contract under test:

* background *preparation* never takes the per-session SQLite compression
  lock — the gateway must NOT see the session as blocked, and new messages
  flow in normally while a candidate is being prepared;
* *applying* a candidate holds the same SQLite lock as synchronous
  compression for a short critical window — the existing #56391 interrupt
  demotion then queues new messages for the duration of the commit;
* eviction paths (``_release_evicted_agent_soft`` → ``release_clients``)
  shut the controller down so no worker outlives its agent instance;
* the background-review fork (shares the parent's live ``session_id``,
  ``compression_enabled=False``) can neither prepare nor apply;
* a deterministic 1,000-run interleaving stress (message arrival, worker
  completion, reset, session switch, apply) shows zero message loss, zero
  duplication and zero cross-session application.
"""

import copy
import os
import random
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.async_context_compression import (
    BackgroundCompressionConfig,
    BackgroundCompressionController,
    CandidateState,
    maybe_apply_prepared_candidate,
    maybe_prepare_background_compression,
    merge_candidate_with_live_messages,
)


def _enabled_config(**overrides) -> BackgroundCompressionConfig:
    base = {
        "enabled": True,
        "shadow_only": False,
        "prepare_threshold": 0.65,
        "apply_threshold": 0.82,
        "min_delta_tokens": 0,
        "min_frozen_messages": 2,
        "max_candidate_age_turns": 12,
        "max_workers": 1,
    }
    base.update(overrides)
    return BackgroundCompressionConfig.from_dict(base)


def _make_messages(n_turns: int = 12) -> list:
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(n_turns):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": f"message {i}"})
    return msgs


def _summary_prepare_fn(frozen_prefix: list) -> list:
    return [
        {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
        copy.deepcopy(frozen_prefix[-1]),
    ]


def _make_runner_over_db(raw_db, session_id: str):
    """GatewayRunner shell wired to a REAL SessionDB for the in-flight probe."""
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    store = MagicMock()
    store._lock = threading.Lock()
    store._loaded = True
    store._entries = {"k": SimpleNamespace(session_id=session_id)}
    store._ensure_loaded_locked = lambda: None
    runner.session_store = store
    runner._session_db = SimpleNamespace(_db=raw_db)
    return runner


def _make_real_agent(session_db, session_id):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=session_db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )
    agent._compression_feasibility_checked = True
    return agent


# ── preparation never blocks the session ───────────────────────────────────


class TestPreparationDoesNotBlockSession:
    @pytest.mark.asyncio
    async def test_in_flight_preparation_holds_no_lock_and_does_not_demote(self):
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "state.db")
            sid = "prep-no-lock"
            db.create_session(sid, "gateway", model="test/model")

            started = threading.Event()
            release = threading.Event()

            def blocked_prepare(prefix):
                started.set()
                assert release.wait(timeout=5.0)
                return _summary_prepare_fn(prefix)

            ctl = BackgroundCompressionController(_enabled_config())
            msgs = _make_messages()
            assert ctl.try_start_preparation(
                session_id=sid, messages=msgs, prefix_count=len(msgs) - 4,
                current_turn=1, source_prompt_tokens=180_000,
                prepare_fn=blocked_prepare,
            )
            assert started.wait(timeout=5.0)
            try:
                # Mid-preparation: no SQLite lock, gateway sees no compression.
                assert db.get_compression_lock_holder(sid) is None
                runner = _make_runner_over_db(db, sid)
                assert await runner._session_has_compression_in_flight("k") is False
            finally:
                release.set()
                assert ctl.wait_until_settled(timeout=5.0)
                ctl.shutdown(wait=True)
            db.close()

    def test_new_message_during_preparation_flows_and_suffix_survives(self):
        ctl = BackgroundCompressionController(_enabled_config())
        msgs = _make_messages()
        assert ctl.try_start_preparation(
            session_id="sess-1", messages=msgs, prefix_count=len(msgs) - 4,
            current_turn=1, source_prompt_tokens=180_000,
            prepare_fn=_summary_prepare_fn,
        )
        assert ctl.wait_until_settled(timeout=5.0)

        # Messages arriving while/after preparation enter the live transcript
        # normally — nothing queues, nothing blocks.
        late = [
            {"role": "user", "content": "arrived during preparation"},
            {"role": "assistant", "content": "answered during preparation"},
        ]
        msgs.extend(late)

        cand = ctl.take_valid_candidate(
            session_id="sess-1", messages=msgs, current_turn=2
        )
        assert cand is not None
        merged = merge_candidate_with_live_messages(cand, msgs)
        assert merged[-2] is late[0]
        assert merged[-1] is late[1]
        ctl.shutdown(wait=True)


# ── apply marks a short critical window via the shared SQLite lock ─────────


class TestApplyCriticalWindow:
    @pytest.mark.asyncio
    async def test_apply_holds_lock_briefly_and_gateway_demotes_to_queue(self):
        from agent.async_context_compression import (
            PreparedCompressionCandidate,
            canonical_prefix_digest,
        )
        from agent.conversation_compression import apply_prepared_candidate
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "state.db")
            sid = "apply-window"
            db.create_session(sid, "gateway", model="test/model")
            agent = _make_real_agent(db, sid)
            agent.compression_in_place = True

            messages = [
                {"role": "user" if i % 2 == 0 else "assistant",
                 "content": f"message {i}"}
                for i in range(12)
            ]
            agent._flush_messages_to_session_db(messages, [])

            prefix_count = len(messages) - 2
            candidate = PreparedCompressionCandidate(
                session_id=sid,
                generation=1,
                prefix_message_count=prefix_count,
                prefix_digest=canonical_prefix_digest(messages, prefix_count),
                prepared_messages=(
                    {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
                    dict(messages[prefix_count - 1]),
                ),
                source_prompt_tokens=180_000,
                created_at_monotonic=0.0,
                created_at_turn=1,
                used_fallback=False,
                summary_error=None,
            )

            entered = threading.Event()
            release = threading.Event()
            orig_archive = db.archive_and_compact

            def blocking_archive(session_id, compressed, *a, **k):
                entered.set()
                assert release.wait(timeout=10.0)
                return orig_archive(session_id, compressed, *a, **k)

            db.archive_and_compact = blocking_archive
            result_box = {}

            def _apply():
                result_box["result"] = apply_prepared_candidate(
                    agent, candidate, messages, "sys"
                )

            t = threading.Thread(target=_apply, daemon=True)
            t.start()
            try:
                assert entered.wait(timeout=10.0)
                # Critical window: the same lock synchronous compression
                # takes is held, so the #56391 demotion queues new messages.
                assert db.get_compression_lock_holder(sid) is not None
                runner = _make_runner_over_db(db, sid)
                assert await runner._session_has_compression_in_flight("k") is True
            finally:
                release.set()
                t.join(timeout=10.0)
            assert not t.is_alive()

            assert result_box["result"] is not None
            # Window closed: lock released, gateway unblocked.
            assert db.get_compression_lock_holder(sid) is None
            runner = _make_runner_over_db(db, sid)
            assert await runner._session_has_compression_in_flight("k") is False

            agent.close()
            db.close()


# ── eviction / reset / fork isolation ──────────────────────────────────────


class TestLifecycleIsolation:
    def test_soft_eviction_shuts_down_controller(self):
        from gateway.run import GatewayRunner
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "state.db")
            sid = "evict-soft"
            db.create_session(sid, "gateway", model="test/model")
            agent = _make_real_agent(db, sid)

            ctl = BackgroundCompressionController(_enabled_config())
            msgs = _make_messages()
            assert ctl.try_start_preparation(
                session_id=sid, messages=msgs, prefix_count=len(msgs) - 4,
                current_turn=1, source_prompt_tokens=180_000,
                prepare_fn=_summary_prepare_fn,
            )
            assert ctl.wait_until_settled(timeout=5.0)
            agent.background_compression_config = ctl.config
            agent.background_compression = ctl

            runner = GatewayRunner.__new__(GatewayRunner)
            runner._release_evicted_agent_soft(agent)

            assert agent.background_compression is None
            assert ctl.peek_candidate() is None
            agent.close()
            db.close()

    def test_background_review_fork_can_neither_prepare_nor_apply(self):
        cfg = _enabled_config()
        engine = SimpleNamespace(
            context_length=100_000,
            last_prompt_tokens=90_000,
            can_prepare_compression=lambda messages, current_tokens=None: True,
            prepare_compression=(
                lambda messages, current_tokens=None, focus_topic=None:
                    _summary_prepare_fn(messages)
            ),
        )
        # Fork shape (agent/background_review.py): shares the parent's live
        # session_id but has compression_enabled=False and its own (empty)
        # controller slot.
        parent_msgs = _make_messages()
        parent_ctl = BackgroundCompressionController(cfg)
        assert parent_ctl.try_start_preparation(
            session_id="parent-sess", messages=parent_msgs,
            prefix_count=len(parent_msgs) - 4, current_turn=1,
            source_prompt_tokens=180_000, prepare_fn=_summary_prepare_fn,
        )
        assert parent_ctl.wait_until_settled(timeout=5.0)

        fork = SimpleNamespace(
            background_compression_config=cfg,
            background_compression=None,
            context_compressor=engine,
            session_id="parent-sess",
            compression_enabled=False,
        )
        assert maybe_prepare_background_compression(
            fork, parent_msgs, current_tokens=90_000, current_turn=2
        ) is False
        assert fork.background_compression is None

        # Even if a future refactor wrongly shared the parent's controller,
        # the compression_enabled gate keeps the fork from applying.
        fork.background_compression = parent_ctl
        assert maybe_apply_prepared_candidate(
            fork, parent_msgs, "sys", current_tokens=90_000, current_turn=2
        ) is None
        assert parent_ctl.peek_candidate() is not None  # untouched
        parent_ctl.shutdown(wait=True)


# ── deterministic interleaving stress ──────────────────────────────────────


class TestDeterministicInterleavingStress:
    """1,000 seeded runs permuting message arrival, worker completion,
    reset, session switch and apply. Invariants: zero suffix loss, zero
    duplication, zero cross-session application."""

    N_RUNS = 1000

    def _run_one(self, seed: int) -> dict:
        rng = random.Random(seed)
        sid = "sess-stress"
        current_sid = sid
        ctl = BackgroundCompressionController(_enabled_config())

        msgs = _make_messages(n_turns=rng.randint(6, 14))
        base_snapshot = copy.deepcopy(msgs)
        prefix_request = len(msgs) - rng.randint(2, 4)

        release = threading.Event()

        def gated_prepare(prefix):
            assert release.wait(timeout=10.0)
            return _summary_prepare_fn(prefix)

        started = ctl.try_start_preparation(
            session_id=sid, messages=msgs, prefix_count=prefix_request,
            current_turn=1, source_prompt_tokens=180_000,
            prepare_fn=gated_prepare,
        )
        assert started is True

        ops = (
            ["worker_done"]
            + ["new_message"] * rng.randint(0, 2)
            + ["apply"] * rng.randint(1, 2)
        )
        if rng.random() < 0.30:
            ops.append("reset")
        if rng.random() < 0.20:
            ops.append("session_switch")
        rng.shuffle(ops)

        appended = []
        worker_done = False
        reset_done = False
        applied = False
        turn = 1

        for op in ops:
            turn += 1
            if op == "worker_done":
                release.set()
                assert ctl.wait_until_settled(timeout=10.0)
                worker_done = True
            elif op == "new_message":
                if applied:
                    continue
                m = {"role": "user", "content": f"late {seed}-{turn}"}
                appended.append(m)
                msgs.append(m)
            elif op == "reset":
                ctl.reset()
                reset_done = True
            elif op == "session_switch":
                current_sid = "sess-OTHER"
            elif op == "apply":
                cand = ctl.take_valid_candidate(
                    session_id=current_sid, messages=msgs, current_turn=turn
                )
                if cand is None:
                    continue
                # Zero cross-session application, ever.
                assert current_sid == sid, f"seed {seed}: cross-session apply"
                assert cand.session_id == sid
                assert not reset_done, f"seed {seed}: applied after reset"
                assert worker_done, f"seed {seed}: applied unfinished work"

                merged = merge_candidate_with_live_messages(cand, msgs)
                prepared_n = len(cand.prepared_messages)
                # Prefix is exactly the prepared set.
                assert merged[:prepared_n] == list(cand.prepared_messages)
                # Suffix: same objects, same order, exactly once (no loss,
                # no duplication).
                live_suffix = msgs[cand.prefix_message_count:]
                merged_suffix = merged[prepared_n:]
                assert len(merged_suffix) == len(live_suffix)
                assert all(a is b for a, b in zip(merged_suffix, live_suffix))
                suffix_ids = [id(m) for m in merged_suffix]
                assert len(suffix_ids) == len(set(suffix_ids))
                for late_msg in appended:
                    assert merged.count(late_msg) == 1

                ctl.mark_applied(cand)
                msgs = merged
                applied = True

        # Ensure the worker is never left dangling.
        release.set()
        ctl.shutdown(wait=True)

        # Preparation never mutated the frozen part of the live transcript.
        if not applied:
            assert msgs[: len(base_snapshot)] == base_snapshot
            for late_msg in appended:
                assert msgs.count(late_msg) == 1
        # A second apply after a successful one must be impossible.
        assert ctl.take_valid_candidate(
            session_id=sid, messages=msgs, current_turn=turn + 1
        ) is None
        return {"applied": applied, "reset": reset_done,
                "switched": current_sid != sid}

    def test_thousand_deterministic_interleavings(self):
        outcomes = {"applied": 0, "reset": 0, "switched": 0}
        for seed in range(self.N_RUNS):
            result = self._run_one(seed)
            for key in outcomes:
                if result[key]:
                    outcomes[key] += 1
        # The permutation space must actually exercise every branch.
        assert outcomes["applied"] > 100
        assert outcomes["reset"] > 100
        assert outcomes["switched"] > 50
