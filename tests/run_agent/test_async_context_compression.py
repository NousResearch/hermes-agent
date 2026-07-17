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
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

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
