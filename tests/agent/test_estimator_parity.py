"""Regression tests for #84371 — compaction dead-loop on reasoning-heavy
codex_responses sessions.

Root cause: the compaction TRIGGER (``estimate_messages_tokens_rough`` /
``estimate_request_tokens_rough``) charged stale ``reasoning`` /
``reasoning_content`` on EVERY assistant message, while the tail-protection
walk (``_find_tail_cut_by_tokens``) charged them on the newest turn only
(#73624).  On a session where most tokens live in stale reasoning replay the
trigger fires above threshold while the walk protects everything —
``middle_window_tokens == 0`` / "insufficient progress" — and the same
compaction re-fires every turn, each attempt burning a full aux
summarization.

Wire truth (``_chat_messages_to_responses_input``): the codex_responses
input builder never reads the text thinking keys; reasoning continuity rides
the encrypted ``codex_reasoning_items`` sidecar, which both estimators charge
unconditionally.  So the TRIGGER overcounted reality and the fix makes the
trigger route-aware (charge stale thinking only when the route echoes it),
while echo-back chat-completions routes (DeepSeek/Kimi/MiMo thinking mode)
now charge it in the walk too — one policy per session shape, chosen by
``message_sanitization.stale_thinking_reaches_wire``.
"""

from unittest.mock import patch

from agent.context_compressor import (
    ContextCompressor,
    _estimate_msg_budget_tokens,
)
from agent.message_sanitization import stale_thinking_reaches_wire
from agent.model_metadata import (
    estimate_messages_tokens_rough,
    estimate_request_tokens_rough,
)


STALE_THINKING = "considering the next move carefully... " * 200  # ~2K tok


def _reasoning_heavy_session(n_turns: int = 40) -> list:
    """Transcript whose bulk is stale reasoning replay (the #84371 shape)."""
    msgs = [{"role": "system", "content": "You are Hermes."}]
    msgs.append({"role": "user", "content": "do the big task"})
    for i in range(n_turns):
        msgs.append(
            {
                "role": "assistant",
                "content": f"step {i}",
                "reasoning_content": STALE_THINKING,
                "tool_calls": [
                    {
                        "id": f"c{i}",
                        "type": "function",
                        "function": {"name": "t", "arguments": "{}"},
                    }
                ],
            }
        )
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": f"r{i}"})
    return msgs


class TestWireTruthPredicate:
    def test_codex_responses_never_ships_stale_thinking_text(self):
        assert stale_thinking_reaches_wire(
            "codex_responses", "deepseek", "deepseek-v4-flash", ""
        ) is False

    def test_chat_completions_echo_family_ships_it(self):
        # DeepSeek thinking mode over chat_completions echoes stored
        # reasoning_content back on every assistant turn.
        assert stale_thinking_reaches_wire(
            "", "deepseek", "deepseek-reasoner", "https://api.deepseek.com"
        ) is True

    def test_strict_chat_completions_strips_it(self):
        assert stale_thinking_reaches_wire(
            "", "mistral", "mistral-large", "https://api.mistral.ai"
        ) is False


class TestEstimatorParity:
    """Trigger-fires must imply the walk finds a compactable middle."""

    def test_trigger_fires_implies_walk_finds_middle(self):
        msgs = _reasoning_heavy_session()
        wire = stale_thinking_reaches_wire(
            "codex_responses", "deepseek", "deepseek-v4-flash", ""
        )
        trigger = estimate_messages_tokens_rough(
            msgs, charge_stale_thinking=wire
        )

        cc = ContextCompressor(
            model="deepseek-v4-flash",
            provider="deepseek",
            api_mode="codex_responses",
            quiet_mode=True,
            config_context_length=200_000,
        )
        # THE RELATION, not literals: whenever the route-aware trigger says
        # the session is over threshold, the tail walk must leave a real
        # middle region so the fired compaction can actually make progress.
        if trigger >= cc.threshold_tokens:
            start = cc._protect_head_size(msgs)
            end = cc._find_tail_cut_by_tokens(msgs, start)
            assert end > start, (
                "trigger fired but the tail walk protected everything — "
                "the #84371 dead-loop shape"
            )
            middle_tokens = estimate_messages_tokens_rough(msgs[start:end])
            assert middle_tokens > 0

    def test_route_aware_trigger_matches_walk_size_class(self):
        """On codex_responses the trigger no longer counts stale thinking
        the walk excludes: both figures land in the same size class."""
        msgs = _reasoning_heavy_session()
        legacy = estimate_messages_tokens_rough(msgs)
        route_aware = estimate_messages_tokens_rough(
            msgs, charge_stale_thinking=False
        )
        from agent.context_compressor import _last_assistant_index

        newest = _last_assistant_index(msgs)
        walk = sum(
            _estimate_msg_budget_tokens(m, charge_stale_thinking=(i == newest))
            for i, m in enumerate(msgs)
        )
        # Stale thinking dominates this transcript, so the legacy figure is
        # several times the walk's; the route-aware figure must not be.
        assert legacy > 3 * walk
        assert route_aware < 2 * walk

    def test_newest_turn_thinking_still_charged(self):
        msgs = _reasoning_heavy_session(n_turns=2)
        stripped = estimate_messages_tokens_rough(
            msgs, charge_stale_thinking=False
        )
        no_thinking = estimate_messages_tokens_rough(
            [
                {k: v for k, v in m.items()
                 if k not in ("reasoning", "reasoning_content")}
                for m in msgs
            ]
        )
        # The newest assistant turn's thinking survives the stale strip.
        assert stripped > no_thinking

    def test_walk_charges_stale_thinking_on_echo_route(self):
        """Echo-back chat_completions route: the walk now charges stale
        thinking on every turn, matching the trigger's full charge; the
        codex_responses route keeps newest-turn-only.  Assert the per-route
        charge policy directly — for each route, the walk's per-message sum
        must land in the same size class as that route's trigger estimate."""
        msgs = _reasoning_heavy_session()
        from agent.context_compressor import _last_assistant_index

        cc_echo = ContextCompressor(
            model="deepseek-reasoner",
            provider="deepseek",
            api_mode="",
            base_url="https://api.deepseek.com",
            quiet_mode=True,
            config_context_length=200_000,
        )
        cc_codex = ContextCompressor(
            model="deepseek-v4-flash",
            provider="deepseek",
            api_mode="codex_responses",
            quiet_mode=True,
            config_context_length=200_000,
        )
        assert cc_echo._stale_thinking_on_wire() is True
        assert cc_codex._stale_thinking_on_wire() is False

        newest = _last_assistant_index(msgs)

        def walk_sum(cc):
            charge_all = cc._stale_thinking_on_wire()
            return sum(
                _estimate_msg_budget_tokens(
                    m, charge_stale_thinking=(charge_all or i == newest)
                )
                for i, m in enumerate(msgs)
            )

        trigger_echo = estimate_messages_tokens_rough(
            msgs, charge_stale_thinking=True
        )
        trigger_codex = estimate_messages_tokens_rough(
            msgs, charge_stale_thinking=False
        )
        walk_echo = walk_sum(cc_echo)
        walk_codex = walk_sum(cc_codex)

        # Per route, trigger and walk agree within a small rough-estimator
        # factor — the pre-fix codex disagreement was >3x.
        assert walk_echo <= trigger_echo * 2 and trigger_echo <= walk_echo * 2
        assert walk_codex <= trigger_codex * 2 and trigger_codex <= walk_codex * 2
        # And the echo route genuinely charges the stale thinking bulk.
        assert walk_echo > 3 * walk_codex


class TestReasoningDoubleCount:
    """``reasoning`` and ``reasoning_content`` carrying the same text must be
    charged once — the wire ships at most one of them."""

    def test_trigger_counts_identical_pair_once(self):
        base = [{"role": "assistant", "content": "x"}]
        rc_only = [{"role": "assistant", "content": "x",
                    "reasoning_content": "y" * 4000}]
        both = [{"role": "assistant", "content": "x",
                 "reasoning": "y" * 4000, "reasoning_content": "y" * 4000}]
        e0 = estimate_messages_tokens_rough(base)
        e1 = estimate_messages_tokens_rough(rc_only)
        e2 = estimate_messages_tokens_rough(both)
        # Adding a duplicate `reasoning` key must not (materially) grow the
        # estimate: the increment over rc_only stays far below a second copy.
        assert (e2 - e0) < 1.5 * (e1 - e0)

    def test_walk_counts_identical_pair_once(self):
        base = {"role": "assistant", "content": "x"}
        rc_only = {"role": "assistant", "content": "x",
                   "reasoning_content": "y" * 4000}
        both = {"role": "assistant", "content": "x",
                "reasoning": "y" * 4000, "reasoning_content": "y" * 4000}
        w0 = _estimate_msg_budget_tokens(base, charge_stale_thinking=True)
        w1 = _estimate_msg_budget_tokens(rc_only, charge_stale_thinking=True)
        w2 = _estimate_msg_budget_tokens(both, charge_stale_thinking=True)
        assert (w2 - w0) < 1.5 * (w1 - w0)

    def test_reasoning_alone_still_charged(self):
        """No reasoning_content to displace it → `reasoning` is the
        promotion proxy and must stay counted."""
        base = [{"role": "assistant", "content": "x"}]
        r_only = [{"role": "assistant", "content": "x",
                   "reasoning": "y" * 4000}]
        assert (
            estimate_messages_tokens_rough(r_only)
            > estimate_messages_tokens_rough(base) + 500
        )
        w_base = _estimate_msg_budget_tokens(
            {"role": "assistant", "content": "x"}, charge_stale_thinking=True
        )
        w_r = _estimate_msg_budget_tokens(
            {"role": "assistant", "content": "x", "reasoning": "y" * 4000},
            charge_stale_thinking=True,
        )
        assert w_r > w_base + 500

    def test_pad_reasoning_content_does_not_displace(self):
        """A one-space echo pad is not real content; `reasoning` still
        carries the chargeable text."""
        msg = {"role": "assistant", "content": "x",
               "reasoning": "y" * 4000, "reasoning_content": " "}
        w = _estimate_msg_budget_tokens(msg, charge_stale_thinking=True)
        w_base = _estimate_msg_budget_tokens(
            {"role": "assistant", "content": "x", "reasoning_content": " "},
            charge_stale_thinking=True,
        )
        assert w > w_base + 500


class TestNoProgressDeadLoopBreaker:
    """A fired compaction that returns the transcript unchanged must arm the
    structural backoff so it cannot re-fire (and re-summarize) every turn."""

    def test_no_progress_arms_structural_backoff(self):
        import time

        cc = ContextCompressor(
            model="deepseek-v4-flash",
            provider="deepseek",
            api_mode="codex_responses",
            quiet_mode=True,
            config_context_length=200_000,
        )
        assert cc._structural_no_op_backoff_until <= time.monotonic()
        cc._record_structural_no_op("compaction returned unchanged")
        assert cc._structural_no_op_backoff_until > time.monotonic()
        # Over-threshold but blocked: no second aux summarization this turn.
        over = cc.threshold_tokens + 1
        assert cc.should_compress(over) is False
        reason = cc._compression_block_reason() or ""
        assert reason.startswith("structural_backoff")

    def test_commit_layer_no_progress_calls_recorder(self):
        """The conversation_compression no_progress path must invoke the
        compressor's structural no-op recorder (it used to record telemetry
        only, so auto-compress re-fired next turn)."""
        import tempfile
        from pathlib import Path
        from unittest.mock import MagicMock
        import os

        from hermes_state import SessionDB
        from run_agent import AIAgent

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "t.db")
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
                agent = AIAgent(
                    api_key="test-key",
                    base_url="https://openrouter.ai/api/v1",
                    model="test/model",
                    quiet_mode=True,
                    session_db=db,
                    session_id="s-84371",
                    skip_context_files=True,
                    skip_memory=True,
                )
            agent.compression_in_place = False
            compressor = MagicMock()
            # No-op compression: returns input unchanged.
            compressor.compress.side_effect = (
                lambda messages, **_kwargs: messages
            )
            compressor._last_compress_aborted = False
            agent.context_compressor = compressor
            messages = [{"role": "user", "content": "request"}]

            returned, _ = agent._compress_context(
                messages, "sys", approx_tokens=100
            )

            assert returned is messages
            assert compressor._record_structural_no_op.called, (
                "no_progress must arm the per-session backoff — otherwise "
                "the dead loop re-fires a full aux summarization every turn"
            )
