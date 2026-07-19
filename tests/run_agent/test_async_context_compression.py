"""Safety contract for background-prepared ("async") context compression.

These tests pin the non-negotiable invariants of the candidate lifecycle
introduced by ``agent/async_context_compression.py`` — a fresh implementation
of the idea explored in PR #23892, rebuilt against the current compression
locks and persistence (no code is reused from that PR).

Invariants under test (see the design plan):

  1.  a valid candidate preserves the live suffix byte-for-byte;
  2.  a diverging prefix digest discards the candidate;
  3.  a diverging session_id discards the candidate;
  4.  a stale generation discards the candidate;
  5.  a candidate is never applied while a tool call is open;
  6.  cancel and reset clear the candidate;
  7.  two preparations leave only the newest generation;
  8.  a worker exception never leaks into the foreground;
  9.  with the feature disabled nothing is called;
  10. ``shadow_only`` never touches persistence.

Preparation must never mutate live state: the worker receives a deep copy of
a frozen prefix and the live ``messages`` list stays untouched.
"""

import copy
import os
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
    PreparedCompressionCandidate,
    PrepareResult,
    align_prefix_boundary,
    canonical_prefix_digest,
    has_open_tool_call,
    maybe_apply_prepared_candidate,
    maybe_prepare_background_compression,
    merge_candidate_with_live_messages,
    resolve_effective_gate_tokens,
    validate_candidate,
)


# ── helpers ────────────────────────────────────────────────────────────────


def _make_messages(n_turns: int = 16) -> list:
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(n_turns):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": f"message {i}"})
    return msgs


def _tool_call_group(idx: int, *, answered: bool = True) -> list:
    call_id = f"call_{idx}"
    group = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        }
    ]
    if answered:
        group.append(
            {"role": "tool", "tool_call_id": call_id, "name": "read_file",
             "content": f"tool result {idx}"}
        )
    return group


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


def _default_prepare_fn(frozen_prefix: list) -> list:
    """Stand-in summariser: one compaction marker + the last frozen message."""
    return [
        {"role": "user", "content": "[CONTEXT COMPACTION] summary of frozen prefix"},
        copy.deepcopy(frozen_prefix[-1]),
    ]


def _prepare_ready_controller(messages, *, session_id="sess-1", config=None,
                              prepare_fn=_default_prepare_fn, current_turn=3):
    ctl = BackgroundCompressionController(config or _enabled_config())
    started = ctl.try_start_preparation(
        session_id=session_id,
        messages=messages,
        prefix_count=len(messages) - 4,
        current_turn=current_turn,
        source_prompt_tokens=180_000,
        prepare_fn=prepare_fn,
    )
    assert started is True
    assert ctl.wait_until_settled(timeout=5.0)
    return ctl


# ── config parsing ─────────────────────────────────────────────────────────


class TestBackgroundCompressionConfig:
    def test_defaults_are_off_and_shadow(self):
        cfg = BackgroundCompressionConfig.from_dict(None)
        assert cfg.enabled is False
        assert cfg.shadow_only is True
        assert cfg.prepare_threshold == pytest.approx(0.65)
        assert cfg.apply_threshold == pytest.approx(0.82)
        assert cfg.min_delta_tokens == 20_000
        assert cfg.min_frozen_messages == 12
        assert cfg.max_candidate_age_turns == 12
        assert cfg.max_workers == 1
        assert cfg.fallback_sync is True
        assert cfg.apply_only_between_turns is True

    def test_from_dict_reads_overrides(self):
        cfg = BackgroundCompressionConfig.from_dict(
            {"enabled": True, "shadow_only": False, "prepare_threshold": 0.5,
             "min_delta_tokens": 5}
        )
        assert cfg.enabled is True
        assert cfg.shadow_only is False
        assert cfg.prepare_threshold == pytest.approx(0.5)
        assert cfg.min_delta_tokens == 5


# ── digest & boundary primitives ───────────────────────────────────────────


class TestDigestAndBoundary:
    def test_digest_is_deterministic_and_prefix_scoped(self):
        msgs = _make_messages()
        d1 = canonical_prefix_digest(msgs, len(msgs) - 2)
        d2 = canonical_prefix_digest(copy.deepcopy(msgs), len(msgs) - 2)
        assert d1 == d2
        assert d1 != canonical_prefix_digest(msgs, len(msgs) - 3)

    def test_digest_ignores_internal_persistence_metadata(self):
        msgs = _make_messages()
        marked = copy.deepcopy(msgs)
        for m in marked:
            m["_db_persisted"] = True
        assert canonical_prefix_digest(msgs, len(msgs)) == canonical_prefix_digest(
            marked, len(marked)
        )

    def test_digest_changes_when_semantic_content_changes(self):
        msgs = _make_messages()
        edited = copy.deepcopy(msgs)
        edited[3]["content"] += " EDITED"
        assert canonical_prefix_digest(msgs, len(msgs)) != canonical_prefix_digest(
            edited, len(edited)
        )

    def test_boundary_never_splits_tool_call_group(self):
        msgs = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "u0"}]
        assistant_idx = len(msgs)
        msgs.extend(_tool_call_group(0))          # assistant @2, tool @3
        msgs.append({"role": "user", "content": "u1"})

        # Boundary between the assistant tool_call and its tool result must
        # retreat to before the assistant message.
        assert align_prefix_boundary(msgs, assistant_idx + 1) == assistant_idx
        # A boundary at a safe cut point is left untouched.
        assert align_prefix_boundary(msgs, assistant_idx) == assistant_idx
        assert align_prefix_boundary(msgs, len(msgs)) == len(msgs)

    def test_boundary_retreats_across_multiple_tool_results(self):
        msgs = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "u0"}]
        assistant_idx = len(msgs)
        call_ids = ["call_a", "call_b"]
        msgs.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": cid, "type": "function",
                 "function": {"name": "t", "arguments": "{}"}}
                for cid in call_ids
            ],
        })
        for cid in call_ids:
            msgs.append({"role": "tool", "tool_call_id": cid, "name": "t",
                         "content": "r"})
        # Cutting between the two tool results must retreat before the assistant.
        assert align_prefix_boundary(msgs, assistant_idx + 2) == assistant_idx

    def test_has_open_tool_call(self):
        closed = _make_messages(4) + _tool_call_group(0, answered=True)
        assert has_open_tool_call(closed) is False
        open_ = _make_messages(4) + _tool_call_group(1, answered=False)
        assert has_open_tool_call(open_) is True


# ── candidate lifecycle ────────────────────────────────────────────────────


class TestCandidateLifecycle:
    def test_valid_candidate_preserves_new_suffix_byte_for_byte(self):
        msgs = _make_messages()
        ctl = _prepare_ready_controller(msgs)
        # New messages arrive after preparation started.
        msgs.append({"role": "user", "content": "new after prepare"})
        msgs.append({"role": "assistant", "content": "answer after prepare"})

        cand = ctl.take_valid_candidate(
            session_id="sess-1", messages=msgs, current_turn=4
        )
        assert cand is not None
        assert isinstance(cand, PreparedCompressionCandidate)
        merged = merge_candidate_with_live_messages(cand, msgs)
        suffix = merged[len(cand.prepared_messages):]
        live_suffix = msgs[cand.prefix_message_count:]
        # Same content AND the very same objects — byte-for-byte survival.
        assert suffix == live_suffix
        assert all(a is b for a, b in zip(suffix, live_suffix))
        assert merged[: len(cand.prepared_messages)] == list(cand.prepared_messages)

    def test_worker_receives_deep_copy_and_never_touches_live_messages(self):
        msgs = _make_messages()
        snapshot = copy.deepcopy(msgs)
        seen = {}

        def prepare_fn(frozen_prefix):
            seen["prefix"] = frozen_prefix
            # A worker mutating its input must never reach the live list.
            frozen_prefix[0]["content"] = "MUTATED BY WORKER"
            return _default_prepare_fn(frozen_prefix)

        ctl = _prepare_ready_controller(msgs, prepare_fn=prepare_fn)
        assert ctl.peek_candidate() is not None
        assert msgs == snapshot
        assert all(
            frozen is not live for frozen, live in zip(seen["prefix"], msgs)
        )

    def test_diverging_digest_discards_candidate(self):
        msgs = _make_messages()
        ctl = _prepare_ready_controller(msgs)
        msgs[2]["content"] += " EDITED AFTER SNAPSHOT"
        assert ctl.take_valid_candidate(
            session_id="sess-1", messages=msgs, current_turn=4
        ) is None
        assert ctl.peek_candidate() is None
        assert ctl.state is CandidateState.STALE

    def test_diverging_session_id_discards_candidate(self):
        msgs = _make_messages()
        ctl = _prepare_ready_controller(msgs, session_id="sess-1")
        assert ctl.take_valid_candidate(
            session_id="sess-OTHER", messages=msgs, current_turn=4
        ) is None
        assert ctl.peek_candidate() is None
        assert ctl.state is CandidateState.STALE

    def test_stale_generation_discards_candidate(self):
        msgs = _make_messages()
        ctl = _prepare_ready_controller(msgs)
        old = ctl.peek_candidate()
        assert old is not None

        # A second preparation supersedes the first generation.
        assert ctl.try_start_preparation(
            session_id="sess-1",
            messages=msgs,
            prefix_count=len(msgs) - 4,
            current_turn=5,
            source_prompt_tokens=190_000,
            prepare_fn=_default_prepare_fn,
        )
        assert ctl.wait_until_settled(timeout=5.0)
        current = ctl.peek_candidate()
        assert current is not None
        assert current.generation > old.generation

        ok, reason = validate_candidate(
            old,
            session_id="sess-1",
            messages=msgs,
            current_generation=ctl.generation,
        )
        assert ok is False
        assert reason == "stale_generation"

    def test_expired_candidate_age_discards_candidate(self):
        msgs = _make_messages()
        ctl = _prepare_ready_controller(
            msgs, config=_enabled_config(max_candidate_age_turns=2), current_turn=3
        )
        assert ctl.take_valid_candidate(
            session_id="sess-1", messages=msgs, current_turn=3 + 2 + 1
        ) is None
        assert ctl.state is CandidateState.STALE

    def test_open_tool_call_blocks_apply_but_keeps_candidate(self):
        msgs = _make_messages()
        ctl = _prepare_ready_controller(msgs)
        msgs.extend(_tool_call_group(9, answered=False))

        assert ctl.take_valid_candidate(
            session_id="sess-1", messages=msgs, current_turn=4
        ) is None
        # The candidate is not consumed: once the tool result lands it can apply.
        assert ctl.state is CandidateState.READY
        assert ctl.peek_candidate() is not None

        msgs.append({"role": "tool", "tool_call_id": "call_9", "name": "read_file",
                     "content": "late result"})
        assert ctl.take_valid_candidate(
            session_id="sess-1", messages=msgs, current_turn=4
        ) is not None

    def test_two_preparations_keep_only_newest_generation(self):
        msgs = _make_messages()
        ctl = BackgroundCompressionController(_enabled_config())
        first_started = threading.Event()
        release_first = threading.Event()

        def slow_prepare(frozen_prefix):
            first_started.set()
            assert release_first.wait(timeout=5.0)
            return [{"role": "user", "content": "[CONTEXT COMPACTION] FIRST"}]

        assert ctl.try_start_preparation(
            session_id="sess-1", messages=msgs, prefix_count=len(msgs) - 4,
            current_turn=1, source_prompt_tokens=100_000, prepare_fn=slow_prepare,
        )
        assert first_started.wait(timeout=5.0)
        # Second preparation supersedes the in-flight first one.
        assert ctl.try_start_preparation(
            session_id="sess-1", messages=msgs, prefix_count=len(msgs) - 4,
            current_turn=2, source_prompt_tokens=110_000,
            prepare_fn=lambda p: [
                {"role": "user", "content": "[CONTEXT COMPACTION] SECOND"}
            ],
        )
        release_first.set()
        assert ctl.wait_until_settled(timeout=5.0)

        cand = ctl.peek_candidate()
        assert cand is not None
        assert cand.prepared_messages[0]["content"].endswith("SECOND")
        assert cand.generation == ctl.generation

    def test_cancel_and_reset_clear_candidate(self):
        msgs = _make_messages()

        # Cancel while preparing: the late worker result must be discarded.
        ctl = BackgroundCompressionController(_enabled_config())
        started = threading.Event()
        release = threading.Event()

        def blocked_prepare(frozen_prefix):
            started.set()
            assert release.wait(timeout=5.0)
            return _default_prepare_fn(frozen_prefix)

        assert ctl.try_start_preparation(
            session_id="sess-1", messages=msgs, prefix_count=len(msgs) - 4,
            current_turn=1, source_prompt_tokens=100_000,
            prepare_fn=blocked_prepare,
        )
        assert started.wait(timeout=5.0)
        ctl.cancel()
        release.set()
        assert ctl.wait_until_settled(timeout=5.0)
        assert ctl.peek_candidate() is None
        assert ctl.state is CandidateState.CANCELLED

        # Reset after ready: candidate cleared, state back to idle.
        ctl2 = _prepare_ready_controller(msgs)
        assert ctl2.peek_candidate() is not None
        ctl2.reset()
        assert ctl2.peek_candidate() is None
        assert ctl2.state is CandidateState.IDLE

    def test_worker_exception_never_leaks_into_foreground(self):
        msgs = _make_messages()
        ctl = BackgroundCompressionController(_enabled_config())

        def broken_prepare(frozen_prefix):
            raise RuntimeError("summariser exploded")

        assert ctl.try_start_preparation(
            session_id="sess-1", messages=msgs, prefix_count=len(msgs) - 4,
            current_turn=1, source_prompt_tokens=100_000,
            prepare_fn=broken_prepare,
        )
        # Foreground calls never raise.
        assert ctl.wait_until_settled(timeout=5.0)
        assert ctl.state is CandidateState.FAILED
        assert ctl.peek_candidate() is None
        assert "summariser exploded" in (ctl.last_error or "")
        assert ctl.take_valid_candidate(
            session_id="sess-1", messages=msgs, current_turn=2
        ) is None

    def test_prepare_result_metadata_is_recorded_on_candidate(self):
        msgs = _make_messages()

        def fallback_prepare(frozen_prefix):
            return PrepareResult(
                messages=_default_prepare_fn(frozen_prefix),
                used_fallback=True,
                summary_error="aux provider down",
            )

        ctl = _prepare_ready_controller(msgs, prepare_fn=fallback_prepare)
        cand = ctl.peek_candidate()
        assert cand is not None
        assert cand.used_fallback is True
        assert cand.summary_error == "aux provider down"

    def test_candidate_is_immutable(self):
        msgs = _make_messages()
        ctl = _prepare_ready_controller(msgs)
        cand = ctl.peek_candidate()
        assert isinstance(cand.prepared_messages, tuple)
        with pytest.raises(Exception):
            cand.session_id = "other"


# ── feature flag & shadow mode ─────────────────────────────────────────────


class TestFeatureFlagAndShadow:
    def test_disabled_feature_changes_no_calls(self):
        # Controller level: preparation refuses to start.
        cfg = BackgroundCompressionConfig.from_dict(None)  # enabled=False default
        ctl = BackgroundCompressionController(cfg)
        prepare_fn = MagicMock()
        assert ctl.try_start_preparation(
            session_id="sess-1", messages=_make_messages(),
            prefix_count=8, current_turn=1, source_prompt_tokens=200_000,
            prepare_fn=prepare_fn,
        ) is False
        assert ctl.state is CandidateState.IDLE
        prepare_fn.assert_not_called()

        # Loop-facing helper: nothing on the agent is touched.
        agent = SimpleNamespace(
            background_compression_config=cfg,
            background_compression=None,
            context_compressor=MagicMock(),
            session_id="sess-1",
            compression_enabled=True,
        )
        assert maybe_prepare_background_compression(
            agent, _make_messages(), current_tokens=250_000, current_turn=1
        ) is False
        assert agent.background_compression is None
        assert agent.context_compressor.mock_calls == []

        assert maybe_apply_prepared_candidate(
            agent, _make_messages(), "sys", current_tokens=250_000, current_turn=1
        ) is None

    def test_shadow_only_never_calls_persistence(self):
        msgs = _make_messages()
        cfg = _enabled_config(shadow_only=True)
        ctl = _prepare_ready_controller(msgs, config=cfg)

        db = MagicMock()
        agent = SimpleNamespace(
            background_compression_config=cfg,
            background_compression=ctl,
            context_compressor=MagicMock(),
            _session_db=db,
            session_id="sess-1",
            compression_enabled=True,
        )
        result = maybe_apply_prepared_candidate(
            agent, msgs, "sys", current_tokens=250_000, current_turn=4
        )
        # Shadow mode validates and measures but never applies or persists.
        assert result is None
        db.archive_and_compact.assert_not_called()
        db.replace_messages.assert_not_called()
        db.end_session.assert_not_called()
        db.create_session.assert_not_called()
        db.update_system_prompt.assert_not_called()


# ── loop integration (task 6): prepare between turns, apply in preflight ───


def _fake_engine(*, context_length=100_000, last_prompt_tokens=0,
                 can_prepare=True, prepare_fn=_default_prepare_fn):
    """Duck-typed ContextEngine exposing exactly the hooks the loop uses."""
    eng = SimpleNamespace()
    eng.context_length = context_length
    eng.last_prompt_tokens = last_prompt_tokens
    eng.can_prepare_compression = (
        lambda messages, current_tokens=None: can_prepare
    )
    eng.prepare_compression = (
        lambda messages, current_tokens=None, focus_topic=None: prepare_fn(messages)
    )
    return eng


def _fake_agent(cfg, engine, *, controller=None, session_id="sess-1"):
    return SimpleNamespace(
        background_compression_config=cfg,
        background_compression=controller,
        context_compressor=engine,
        session_id=session_id,
        compression_enabled=True,
    )


class TestPreparationTriggers:
    """maybe_prepare_background_compression() trigger conditions (task 6)."""

    def test_prepare_starts_above_threshold_without_mutating_live_messages(self):
        msgs = _make_messages()
        snapshot = copy.deepcopy(msgs)
        agent = _fake_agent(_enabled_config(), _fake_engine())

        started = maybe_prepare_background_compression(
            agent, msgs, current_tokens=80_000, current_turn=2
        )
        assert started is True
        ctl = agent.background_compression
        assert isinstance(ctl, BackgroundCompressionController)
        assert ctl.wait_until_settled(timeout=5.0)
        assert ctl.state is CandidateState.READY
        assert ctl.peek_candidate() is not None
        assert msgs == snapshot

    def test_prepare_skipped_below_prepare_threshold(self):
        agent = _fake_agent(_enabled_config(), _fake_engine())
        assert maybe_prepare_background_compression(
            agent, _make_messages(), current_tokens=30_000, current_turn=2
        ) is False
        assert agent.background_compression is None

    def test_prepare_skipped_when_engine_opts_out(self):
        agent = _fake_agent(_enabled_config(), _fake_engine(can_prepare=False))
        assert maybe_prepare_background_compression(
            agent, _make_messages(), current_tokens=80_000, current_turn=2
        ) is False
        assert agent.background_compression is None

    def test_prepare_skipped_when_tool_call_open(self):
        msgs = _make_messages() + _tool_call_group(7, answered=False)
        agent = _fake_agent(_enabled_config(), _fake_engine())
        assert maybe_prepare_background_compression(
            agent, msgs, current_tokens=80_000, current_turn=2
        ) is False

    def test_prepare_skipped_while_preparation_in_flight(self):
        msgs = _make_messages()
        started = threading.Event()
        release = threading.Event()

        def blocked(prefix):
            started.set()
            assert release.wait(timeout=5.0)
            return _default_prepare_fn(prefix)

        cfg = _enabled_config()
        agent = _fake_agent(cfg, _fake_engine(prepare_fn=blocked))
        assert maybe_prepare_background_compression(
            agent, msgs, current_tokens=80_000, current_turn=2
        ) is True
        assert started.wait(timeout=5.0)
        # In flight — a second trigger must not supersede the running one.
        assert maybe_prepare_background_compression(
            agent, msgs, current_tokens=90_000, current_turn=3
        ) is False
        release.set()
        assert agent.background_compression.wait_until_settled(timeout=5.0)

    def test_prepare_respects_min_delta_tokens(self):
        msgs = _make_messages()
        cfg = _enabled_config(min_delta_tokens=20_000)
        agent = _fake_agent(cfg, _fake_engine())
        assert maybe_prepare_background_compression(
            agent, msgs, current_tokens=80_000, current_turn=2
        ) is True
        assert agent.background_compression.wait_until_settled(timeout=5.0)
        # Barely grown context: don't re-summarise on every message.
        assert maybe_prepare_background_compression(
            agent, msgs, current_tokens=85_000, current_turn=3
        ) is False
        # Past the delta: a fresh candidate is worth preparing.
        assert maybe_prepare_background_compression(
            agent, msgs, current_tokens=101_000, current_turn=4
        ) is True
        assert agent.background_compression.wait_until_settled(timeout=5.0)


class TestEffectiveGateDerivation:
    """resolve_effective_gate_tokens(): prepare < apply < sync for every profile."""

    def _engine(self, *, context_length, threshold_tokens):
        eng = _fake_engine(context_length=context_length)
        eng.threshold_tokens = threshold_tokens
        return eng

    def test_defaults_clamp_under_upstream_sync_threshold(self):
        # Shipped defaults: sync 0.50 of a 400K window → 200K trigger. The
        # absolute 0.65/0.82 gates (260K/328K) sit past it and must clamp.
        prepare, apply, sync = resolve_effective_gate_tokens(
            _enabled_config(),
            self._engine(context_length=400_000, threshold_tokens=200_000),
        )
        assert sync == 200_000
        assert prepare == pytest.approx(160_000)  # 80% of the sync trigger
        assert apply == pytest.approx(192_000)    # 96% of the sync trigger
        assert prepare < apply < sync

    def test_small_context_floor_profile(self):
        # 100K window under the raise-only 0.75 floor → 75K trigger.
        prepare, apply, sync = resolve_effective_gate_tokens(
            _enabled_config(),
            self._engine(context_length=100_000, threshold_tokens=75_000),
        )
        assert prepare == pytest.approx(60_000)
        assert apply == pytest.approx(72_000)
        assert prepare < apply < sync

    def test_high_sync_threshold_honors_configured_values(self):
        # 0.85-style profile (Codex autoraise): 0.65 already fires first and
        # is honored verbatim; 0.82 clamps marginally to 96% of the trigger.
        prepare, apply, sync = resolve_effective_gate_tokens(
            _enabled_config(),
            self._engine(context_length=100_000, threshold_tokens=85_000),
        )
        assert prepare == pytest.approx(65_000)
        assert apply == pytest.approx(81_600)
        assert prepare < apply < sync

    def test_inverted_user_config_is_reordered(self):
        cfg = _enabled_config(prepare_threshold=0.9, apply_threshold=0.6)
        prepare, apply, _sync = resolve_effective_gate_tokens(
            cfg, self._engine(context_length=100_000, threshold_tokens=85_000)
        )
        assert prepare < apply

    def test_unknown_engine_keeps_gates_silent(self):
        prepare, apply, sync = resolve_effective_gate_tokens(_enabled_config(), None)
        assert (prepare, apply, sync) == (0.0, 0.0, 0.0)


class TestApplyGate:
    """maybe_apply_prepared_candidate() must respect apply_threshold."""

    def test_apply_below_threshold_keeps_candidate_warm(self):
        msgs = _make_messages()
        cfg = _enabled_config(shadow_only=False)
        ctl = _prepare_ready_controller(msgs, config=cfg)
        agent = _fake_agent(cfg, _fake_engine(), controller=ctl)

        result = maybe_apply_prepared_candidate(
            agent, msgs, "sys", current_tokens=70_000, current_turn=4
        )
        assert result is None
        # Below the gate is a temporal refusal — the candidate survives and
        # no apply/fallback was even attempted.
        assert ctl.peek_candidate() is not None
        assert ctl.state is CandidateState.READY
        assert "sync_fallback_count" not in ctl.stats
        assert "candidate_applied" not in ctl.stats

    def test_shadow_above_threshold_records_and_drops(self):
        msgs = _make_messages()
        cfg = _enabled_config(shadow_only=True)
        ctl = _prepare_ready_controller(msgs, config=cfg)
        agent = _fake_agent(cfg, _fake_engine(), controller=ctl)

        result = maybe_apply_prepared_candidate(
            agent, msgs, "sys", current_tokens=90_000, current_turn=4
        )
        assert result is None
        assert ctl.peek_candidate() is None
        assert ctl.stats.get("candidate_shadow_validated") == 1

    def test_sync_preemption_when_sync_fires_below_apply_gate(self):
        # ``should_compress`` can fire on non-token triggers (message-count
        # hygiene) while usage still sits under the apply gate. A ready
        # candidate must preempt that synchronous run, not stay warm.
        msgs = _make_messages()
        cfg = _enabled_config(shadow_only=True)
        ctl = _prepare_ready_controller(msgs, config=cfg)
        eng = _fake_engine(context_length=1_000_000)
        eng.threshold_tokens = 900_000            # apply gate stays at 820K
        eng.should_compress = lambda tokens: True  # hygiene-style trigger
        agent = _fake_agent(cfg, eng, controller=ctl)

        result = maybe_apply_prepared_candidate(
            agent, msgs, "sys", current_tokens=600_000, current_turn=4
        )
        assert result is None  # shadow mode: observe, never apply
        assert ctl.stats.get("candidate_shadow_validated") == 1

    def test_below_gate_and_sync_quiet_still_keeps_candidate_warm(self):
        # Same engine surface, sync NOT firing: the temporal refusal from
        # test_apply_below_threshold_keeps_candidate_warm must survive the
        # preemption backstop.
        msgs = _make_messages()
        cfg = _enabled_config(shadow_only=False)
        ctl = _prepare_ready_controller(msgs, config=cfg)
        eng = _fake_engine(context_length=1_000_000)
        eng.threshold_tokens = 900_000
        eng.should_compress = lambda tokens: tokens >= 900_000
        agent = _fake_agent(cfg, eng, controller=ctl)

        result = maybe_apply_prepared_candidate(
            agent, msgs, "sys", current_tokens=600_000, current_turn=4
        )
        assert result is None
        assert ctl.peek_candidate() is not None
        assert ctl.state is CandidateState.READY


class _FinalizeStubAgent:
    """Minimal duck-typed agent surface for finalize_turn wiring tests."""

    def __init__(self):
        self.max_iterations = 10
        self.iteration_budget = SimpleNamespace(used=1, max_total=10, remaining=9)
        self.context_compressor = SimpleNamespace(last_prompt_tokens=0)
        self.model = "stub/model"
        self.provider = "stub"
        self.base_url = "http://stub"
        self.session_id = "sess-1"
        self.quiet_mode = True
        self.platform = "cli"
        self._interrupt_requested = False
        self._interrupt_message = None
        self._tool_guardrail_halt_decision = None
        self._response_was_previewed = False
        self._skill_nudge_interval = 0
        self._iters_since_skill = 0
        for attr in (
            "session_input_tokens", "session_output_tokens",
            "session_cache_read_tokens", "session_cache_write_tokens",
            "session_reasoning_tokens", "session_prompt_tokens",
            "session_completion_tokens", "session_total_tokens",
            "session_estimated_cost_usd",
        ):
            setattr(self, attr, 0)
        self.session_cost_status = "ok"
        self.session_cost_source = "stub"
        self.prepare_calls = []

    def _maybe_prepare_background_compression(self, messages, **kwargs):
        self.prepare_calls.append(list(messages))
        return True

    # -- inert surfaces --------------------------------------------------
    def _save_trajectory(self, *a, **k): pass
    def _cleanup_task_resources(self, *a, **k): pass
    def _drop_trailing_empty_response_scaffolding(self, *a, **k): pass
    def _persist_session(self, *a, **k): pass
    def _emit_status(self, *a, **k): pass
    def _safe_print(self, *a, **k): pass
    def _handle_max_iterations(self, messages, n): return "SUMMARY"
    def _file_mutation_verifier_enabled(self): return False
    def _turn_completion_explainer_enabled(self): return False
    def _drain_pending_steer(self): return None
    def clear_interrupt(self): pass
    def _sync_external_memory_for_turn(self, **k): pass


def _run_finalize(agent, *, interrupted=False, failed=False,
                  final_response="done"):
    from agent.turn_finalizer import finalize_turn

    messages = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": final_response or "answer"},
    ]
    return finalize_turn(
        agent,
        final_response=final_response,
        api_call_count=1,
        interrupted=interrupted,
        failed=failed,
        messages=messages,
        conversation_history=None,
        effective_task_id="task-1",
        turn_id="turn-1",
        user_message="question",
        original_user_message="question",
        _should_review_memory=False,
        _turn_exit_reason="text_response(finish_reason=stop)",
    )


class TestFinalizeTurnWiring:
    """The between-turns finalizer must trigger background preparation."""

    def test_completed_turn_triggers_preparation(self):
        agent = _FinalizeStubAgent()
        result = _run_finalize(agent)
        assert result["final_response"] == "done"
        assert len(agent.prepare_calls) == 1
        assert agent.prepare_calls[0][-1]["role"] == "assistant"

    def test_interrupted_turn_does_not_trigger_preparation(self):
        agent = _FinalizeStubAgent()
        result = _run_finalize(agent, interrupted=True, final_response=None)
        assert result["interrupted"] is True
        assert agent.prepare_calls == []

    def test_preparation_hook_exception_never_breaks_the_turn(self):
        agent = _FinalizeStubAgent()

        def _boom(messages, **kwargs):
            raise RuntimeError("background exploded")

        agent._maybe_prepare_background_compression = _boom
        result = _run_finalize(agent)
        assert result["final_response"] == "done"

    def test_agent_without_hook_still_finalizes(self):
        # Backward compatibility: stubs/forks without the hook keep working.
        agent = _FinalizeStubAgent()
        agent._maybe_prepare_background_compression = None  # not callable
        result = _run_finalize(agent)
        assert result["final_response"] == "done"


class TestConfigTelemetryKillSwitch:
    """Task 8: operable without code edits — config block, telemetry, kill switch."""

    def test_default_config_ships_disabled_shadow_block(self):
        from hermes_cli.config import DEFAULT_CONFIG

        bg = DEFAULT_CONFIG["compression"]["background"]
        assert bg["enabled"] is False
        assert bg["shadow_only"] is True
        assert bg["prepare_threshold"] == pytest.approx(0.65)
        assert bg["apply_threshold"] == pytest.approx(0.82)
        assert bg["min_delta_tokens"] == 20_000
        assert bg["min_frozen_messages"] == 12
        assert bg["max_candidate_age_turns"] == 12
        assert bg["max_workers"] == 1
        assert bg["foreground_priority"] is True
        assert bg["fallback_sync"] is True
        assert bg["apply_only_between_turns"] is True
        # The shipped defaults and the dataclass defaults must never drift.
        assert BackgroundCompressionConfig.from_dict(bg) == BackgroundCompressionConfig()

    def test_gateway_cache_signature_busts_on_background_config_edit(self):
        from gateway.run import GatewayRunner

        on = GatewayRunner._extract_cache_busting_config(
            {"compression": {"background": {"enabled": True}}}
        )
        off = GatewayRunner._extract_cache_busting_config(
            {"compression": {"background": {"enabled": False}}}
        )
        assert "compression.background" in on
        assert on["compression.background"] != off["compression.background"]

    def test_kill_switch_via_config_set_restores_baseline(self):
        from hermes_cli.config import _set_nested

        config = {"compression": {"background": {"enabled": True,
                                                 "shadow_only": False}}}
        _set_nested(config, "compression.background.enabled", False)
        cfg = BackgroundCompressionConfig.from_dict(
            config["compression"]["background"]
        )
        assert cfg.enabled is False

        agent = _fake_agent(cfg, _fake_engine())
        assert maybe_prepare_background_compression(
            agent, _make_messages(), current_tokens=90_000, current_turn=1
        ) is False
        assert agent.background_compression is None
        assert maybe_apply_prepared_candidate(
            agent, _make_messages(), "sys", current_tokens=90_000, current_turn=1
        ) is None

    def test_telemetry_snapshot_counts_lifecycle_events(self):
        msgs = _make_messages()
        ctl = _prepare_ready_controller(msgs)
        snap = ctl.telemetry_snapshot()
        assert snap["candidate_started"] == 1
        assert snap["candidate_ready"] == 1
        assert snap["candidate_failed"] == 0
        assert snap["state"] == "ready"
        assert snap["candidate_prepare_ms"]["count"] == 1
        assert snap["candidate_prepare_ms"]["p50_ms"] >= 0.0

    def test_telemetry_counts_worker_failure_as_provider_error(self):
        ctl = BackgroundCompressionController(_enabled_config())

        def broken(prefix):
            raise RuntimeError("provider 500")

        assert ctl.try_start_preparation(
            session_id="sess-1", messages=_make_messages(),
            prefix_count=10, current_turn=1, source_prompt_tokens=100_000,
            prepare_fn=broken,
        )
        assert ctl.wait_until_settled(timeout=5.0)
        snap = ctl.telemetry_snapshot()
        assert snap["candidate_failed"] == 1
        assert snap["background_provider_error"] == 1

    def test_apply_records_duration_and_suffix_preserved(self):
        msgs = _make_messages()
        cfg = _enabled_config(shadow_only=False)
        ctl = _prepare_ready_controller(msgs, config=cfg)
        agent = _fake_agent(cfg, _fake_engine(), controller=ctl)

        with patch(
            "agent.conversation_compression.apply_prepared_candidate",
            return_value=(["compressed"], "new-prompt"),
        ):
            result = maybe_apply_prepared_candidate(
                agent, msgs, "sys", current_tokens=90_000, current_turn=4
            )
        assert result == (["compressed"], "new-prompt")
        snap = ctl.telemetry_snapshot()
        assert snap["candidate_apply_ms"]["count"] == 1
        # Helper froze len(msgs) - 4 messages; the 4-message suffix survived.
        assert snap["suffix_messages_preserved"] == 4

    def test_apply_failure_counts_sync_fallback(self):
        msgs = _make_messages()
        cfg = _enabled_config(shadow_only=False)
        ctl = _prepare_ready_controller(msgs, config=cfg)
        agent = _fake_agent(cfg, _fake_engine(), controller=ctl)

        with patch(
            "agent.conversation_compression.apply_prepared_candidate",
            side_effect=RuntimeError("db locked"),
        ):
            result = maybe_apply_prepared_candidate(
                agent, msgs, "sys", current_tokens=90_000, current_turn=4
            )
        assert result is None
        assert ctl.telemetry_snapshot()["sync_fallback_count"] == 1


class TestAgentLifecycleIntegration:
    """Real-agent surface: hooks exist, are safe, and honor session boundaries."""

    def _make_real_agent(self, session_db, session_id):
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

    def test_feature_disabled_by_default_and_hooks_are_noops(self):
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "state.db")
            sid = "bg-defaults"
            db.create_session(sid, "cli", model="test/model")
            agent = self._make_real_agent(db, sid)
            try:
                cfg = agent.background_compression_config
                assert isinstance(cfg, BackgroundCompressionConfig)
                assert cfg.enabled is False
                assert cfg.shadow_only is True
                assert agent.background_compression is None

                msgs = _make_messages()
                assert agent._maybe_prepare_background_compression(msgs) is False
                assert agent._maybe_apply_prepared_compression(msgs, "sys") is None
                assert agent.background_compression is None
            finally:
                agent.close()
                db.close()

    def test_reset_session_state_invalidates_candidate(self):
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "state.db")
            sid = "bg-reset"
            db.create_session(sid, "cli", model="test/model")
            agent = self._make_real_agent(db, sid)
            try:
                msgs = _make_messages()
                ctl = _prepare_ready_controller(msgs, session_id=sid)
                agent.background_compression_config = ctl.config
                agent.background_compression = ctl
                assert ctl.peek_candidate() is not None

                agent.reset_session_state()
                assert ctl.peek_candidate() is None
                assert ctl.state is CandidateState.IDLE
            finally:
                agent.close()
                db.close()

    def test_close_shuts_down_controller(self):
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "state.db")
            sid = "bg-close"
            db.create_session(sid, "cli", model="test/model")
            agent = self._make_real_agent(db, sid)
            msgs = _make_messages()
            ctl = _prepare_ready_controller(msgs, session_id=sid)
            agent.background_compression_config = ctl.config
            agent.background_compression = ctl

            agent.close()
            assert agent.background_compression is None
            assert ctl.peek_candidate() is None
            db.close()

    def test_apply_hook_swallows_controller_exceptions(self):
        from hermes_state import SessionDB

        class _BrokenController(BackgroundCompressionController):
            def take_valid_candidate(self, **kwargs):
                raise RuntimeError("controller exploded")

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "state.db")
            sid = "bg-broken"
            db.create_session(sid, "cli", model="test/model")
            agent = self._make_real_agent(db, sid)
            try:
                cfg = _enabled_config(shadow_only=False)
                agent.background_compression_config = cfg
                agent.background_compression = _BrokenController(cfg)
                # Invariant 7: background failures never reach the foreground.
                assert agent._maybe_apply_prepared_compression(
                    _make_messages(), "sys", current_tokens=250_000
                ) is None
            finally:
                agent.close()
                db.close()


# ── E2E preflight ordering with shipped defaults (PR #66619 review) ────────


class TestPreflightDefaultsEndToEnd:
    """The real ``build_turn_context`` preflight must apply a candidate
    prepared under SHIPPED defaults instead of invoking the synchronous
    ``_compress_context`` — the hermes-sweeper blocking finding on #66619.

    Everything is real except the summariser LLM call: AIAgent, SessionDB,
    ContextCompressor threshold math (0.50 config + floors), the background
    controller, the preflight chain and the atomic apply."""

    _SUMMARY = "[CONTEXT COMPACTION] e2e deterministic summary"

    def _grow_until(self, messages, estimator, target_tokens, *, tag, repeat=30):
        """Append user/assistant turns until the rough estimate crosses the
        target. Returns how many USER rows were appended so the caller can
        advance ``_user_turn_count`` exactly like live turns would."""
        i = 0
        users = 0
        filler = ("conversa longa sobre o projeto, decisões e arquivos. " * repeat).strip()
        while estimator(messages) < target_tokens:
            role = "user" if i % 2 == 0 else "assistant"
            if role == "user":
                users += 1
            messages.append({"role": role, "content": f"[{tag} {i}] {filler}"})
            i += 1
        return users

    def test_default_candidate_applies_instead_of_sync_compression(self):
        import agent.conversation_loop as loop_mod
        import agent.turn_context as turn_ctx_mod
        from agent.context_compressor import ContextCompressor
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
                from run_agent import AIAgent

                db = SessionDB(db_path=Path(tmpdir) / "state.db")
                sid = "e2e-preflight-defaults"
                db.create_session(sid, "cli", model="test/model")
                agent = AIAgent(
                    api_key="test-key",
                    base_url="https://openrouter.ai/api/v1",
                    model="test/model",
                    quiet_mode=True,
                    session_db=db,
                    session_id=sid,
                    skip_context_files=True,
                    skip_memory=True,
                )
            try:
                agent._compression_feasibility_checked = True
                engine = agent.context_compressor
                # Real engine API + real threshold math on a bounded window;
                # the shipped compression config (0.50 + raise-only floors)
                # stays untouched.
                engine.update_model("test/model", 120_000)
                sync_trigger = engine.threshold_tokens
                assert sync_trigger > 0

                # Turn the feature on exactly as an operator would; every
                # threshold keeps its shipped default (0.65 / 0.82).
                agent.background_compression_config = (
                    BackgroundCompressionConfig.from_dict(
                        {"enabled": True, "shadow_only": False}
                    )
                )
                cfg = agent.background_compression_config
                prepare_gate, apply_gate, sync_tokens = (
                    resolve_effective_gate_tokens(cfg, engine)
                )
                # The review's invariant, on the real engine with defaults:
                assert prepare_gate < apply_gate < sync_tokens == sync_trigger

                def _estimate(msgs):
                    return turn_ctx_mod.estimate_request_tokens_rough(
                        msgs,
                        system_prompt="",
                        tools=agent.tools or None,
                    )

                # Deterministic summariser: the only stubbed boundary.
                with patch.object(
                    ContextCompressor,
                    "_generate_summary",
                    return_value=self._SUMMARY,
                ):
                    # 1. History lands between the prepare and apply gates →
                    #    the real between-turn hook starts preparation. The
                    #    turn counter is hydrated first, exactly as the live
                    #    loop keeps it across the accumulated turns.
                    history = []
                    self._grow_until(
                        history, _estimate,
                        int((prepare_gate + apply_gate) / 2), tag="base",
                    )
                    agent._user_turn_count = sum(
                        1 for m in history if m.get("role") == "user"
                    )
                    prepare_tokens = _estimate(history)
                    assert prepare_gate <= prepare_tokens < apply_gate
                    started = agent._maybe_prepare_background_compression(
                        history, current_tokens=prepare_tokens
                    )
                    assert started is True
                    ctl = agent.background_compression
                    assert ctl is not None and ctl.wait_until_settled(timeout=30.0)
                    assert ctl.state is CandidateState.READY
                    candidate = ctl.peek_candidate()
                    assert candidate is not None

                    # 2. Conversation keeps growing (append-only) past the
                    #    synchronous trigger — agentic-sized turns, with the
                    #    turn counter advancing like live turns would.
                    grown_users = self._grow_until(
                        history, _estimate, int(sync_trigger * 1.05),
                        tag="growth", repeat=120,
                    )
                    agent._user_turn_count += grown_users
                    tail_before = copy.deepcopy(history[-5:])
                    agent._flush_messages_to_session_db(history, [])

                    # 3. The REAL preflight, wired exactly like the loop.
                    sync_spy = MagicMock(
                        side_effect=lambda msgs, sysm, **kw: (msgs, sysm)
                    )
                    with patch.object(agent, "_compress_context", sync_spy):
                        ctx = turn_ctx_mod.build_turn_context(
                            agent,
                            "e segue o baile",
                            None,
                            history,
                            None,
                            None,
                            None,
                            restore_or_build_system_prompt=(
                                loop_mod._restore_or_build_system_prompt
                            ),
                            install_safe_stdio=loop_mod._install_safe_stdio,
                            sanitize_surrogates=loop_mod._sanitize_surrogates,
                            summarize_user_message_for_log=(
                                loop_mod._summarize_user_message_for_log
                            ),
                            set_session_context=loop_mod.set_session_context,
                            set_current_write_origin=(
                                loop_mod.set_current_write_origin
                            ),
                            ra=loop_mod._ra,
                        )

                # The candidate applied; the synchronous path never ran.
                assert sync_spy.call_count == 0
                assert ctl.stats.get("candidate_applied") == 1
                joined = "\n".join(
                    str(m.get("content")) for m in ctx.messages
                )
                assert self._SUMMARY in joined
                # Suffix survival across the swap: the merge must keep the
                # exact live dict objects (identity), and their semantic
                # content must be untouched (persistence markers aside).
                merged_tail = ctx.messages[-6:-1]
                assert all(
                    a is b for a, b in zip(merged_tail, history[-5:])
                )
                assert [
                    (m.get("role"), m.get("content")) for m in merged_tail
                ] == [
                    (m.get("role"), m.get("content")) for m in tail_before
                ]
                # And the applied context sits back under the sync trigger.
                assert not engine.should_compress(_estimate(ctx.messages))
            finally:
                agent.close()
                db.close()
