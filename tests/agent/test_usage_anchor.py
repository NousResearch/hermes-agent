"""Usage-anchored context accounting (agent/usage_anchor.py).

Context-size checks anchor on the provider-reported ``usage.prompt_tokens``
of the last main-loop response and estimate ONLY the messages appended
since. These tests cover:

  * anchor + delta arithmetic (exact base, small estimated delta);
  * the image-heavy divergence the anchor eliminates (flat 1500/image
    heuristic vs provider truth);
  * fallback to full estimation when no anchor exists (first request,
    usage-less providers);
  * invalidation when compaction rewrites the transcript (content fingerprint
    fails closed) while a DB-reloaded transcript with the same content still
    matches (the gateway re-reads history every turn);
  * persistence on the session row and restore in a fresh process;
  * the preflight consumer (_preflight_request_tokens) preferring the
    anchor, plus a sabotage check proving the anchored path (not the
    heuristic) produces the number.
"""

from types import SimpleNamespace

import pytest

from agent.model_metadata import estimate_messages_tokens_rough
from agent.turn_context import _preflight_request_tokens
from agent.usage_anchor import (
    anchored_context_tokens,
    capture_usage_anchor,
    restore_usage_anchor,
    set_usage_anchor,
)


def _msg(role, content):
    return {"role": role, "content": content}


def _image_msg():
    # ~40KB of fake base64 — the rough estimator charges a flat 1500
    # tokens per image part regardless of true provider accounting.
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": "look at this"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64," + "A" * 40000},
            },
        ],
    }


def _plain_history():
    return [_msg("user", "start"), _msg("assistant", "hello"), _msg("user", "do the thing"), _msg("assistant", "done")]


def _history_with_images(n_images=10):
    msgs = [_msg("user", "start")]
    for i in range(n_images):
        msgs.append(_msg("assistant", f"taking screenshot {i}"))
        msgs.append(_image_msg())
    msgs.append(_msg("assistant", "done looking"))
    return msgs


class TestAnchorArithmetic:
    def test_anchor_plus_small_delta(self):
        messages = _history_with_images(10)
        anchor = capture_usage_anchor(50_000, 250, messages)
        assert anchor is not None
        assert anchor["prompt_tokens"] == 50_000
        assert anchor["base_count"] == len(messages)

        # Main loop appends the response's own assistant reply, then a tool
        # result / user follow-up.
        messages.append(_msg("assistant", "the anchored reply itself"))
        messages.append(_msg("user", "short follow-up"))

        anchored = anchored_context_tokens(messages, anchor)
        assert anchored is not None
        # Exact base + completion; the assistant reply at base_count is
        # covered by completion_tokens, so only the follow-up is estimated.
        delta_est = estimate_messages_tokens_rough([messages[-1]])
        assert anchored == 50_000 + 250 + delta_est
        assert delta_est < 50  # the estimated window is one small message

    def test_image_heavy_divergence_eliminated(self):
        messages = _history_with_images(10)
        # Provider ground truth: say the real prompt was 12,000 tokens
        # (providers often charge far less than 1500/image, or the images
        # were downscaled). The heuristic charges 10 * 1500 + text.
        anchor = capture_usage_anchor(12_000, 100, messages)
        messages.append(_msg("assistant", "reply"))
        messages.append(_msg("user", "ok"))

        rough = estimate_messages_tokens_rough(messages)
        anchored = anchored_context_tokens(messages, anchor)
        assert rough >= 15_000  # flat 1500 x 10 images dominates
        assert anchored is not None
        assert anchored < 12_200
        # The whole-history heuristic diverges by thousands of tokens;
        # the anchored figure is provider truth + a tiny delta.
        assert rough - anchored > 2_800

    def test_no_usage_returns_none(self):
        messages = [_msg("user", "hi")]
        assert capture_usage_anchor(0, 0, messages) is None
        assert capture_usage_anchor(None, None, messages) is None
        assert capture_usage_anchor("garbage", 1, messages) is None

    def test_missing_anchor_falls_back(self):
        messages = _history_with_images(2)
        assert anchored_context_tokens(messages, None) is None


class TestAnchorInvalidation:
    def test_compaction_rewrite_fails_closed(self):
        messages = _history_with_images(4)
        anchor = capture_usage_anchor(30_000, 50, messages)
        # Compaction: transcript rebuilt as a new, shorter list.
        compacted = [
            _msg("user", "summary handoff"),
            _msg("assistant", "[compressed summary]"),
        ]
        assert anchored_context_tokens(compacted, anchor) is None

    def test_middle_splice_shifts_base_and_fails_closed(self):
        messages = _history_with_images(4)
        anchor = capture_usage_anchor(30_000, 50, messages)
        # Micro-compact style splice: middle window replaced by one marker.
        spliced = messages[:1] + [_msg("assistant", "[marker]")] + messages[5:]
        assert anchored_context_tokens(spliced, anchor) is None

    def test_reloaded_transcript_with_same_content_still_matches(self):
        """The gateway re-reads history from the DB every turn (fresh dicts, extra
        persistence keys); identity must survive that or every gateway turn falls
        back to the whole-history estimate."""
        messages = _history_with_images(4)
        anchor = capture_usage_anchor(30_000, 50, messages)
        reloaded = [dict(m, timestamp=1.0, _row_id=i) for i, m in enumerate(messages)]
        assert anchored_context_tokens(reloaded, anchor) == 30_050
        edited = [dict(m) for m in messages]
        edited[-1] = dict(edited[-1], content="different last message")
        assert anchored_context_tokens(edited, anchor) is None

    def test_persist_and_restore_across_processes(self, tmp_path):
        """A fresh agent (desktop per-turn ``serve``, ``--resume``) adopts the persisted anchor
        while the durable transcript still matches, and clears it once it does not."""
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        sid = "anchor-restore"
        db.create_session(sid, source="cli")
        messages = _plain_history()
        for m in messages:
            db.append_message(sid, m["role"], m["content"])
        durable = db.get_messages_as_conversation(sid)
        live = SimpleNamespace(session_id=sid, _session_db=db, _persist_disabled=False, _usage_anchor=None)
        set_usage_anchor(live, capture_usage_anchor(10_000, 20, durable))

        fresh = SimpleNamespace(session_id=sid, _session_db=db, _persist_disabled=False, _usage_anchor=None)
        restore_usage_anchor(fresh, db.get_messages_as_conversation(sid))
        assert fresh._usage_anchor is not None
        assert anchored_context_tokens(db.get_messages_as_conversation(sid), fresh._usage_anchor) == 10_020

        set_usage_anchor(live, None)  # compaction / reset clears the row too
        stale = SimpleNamespace(session_id=sid, _session_db=db, _persist_disabled=False, _usage_anchor=None)
        restore_usage_anchor(stale, durable)
        assert stale._usage_anchor is None
        db.close()


class TestPreflightConsumer:
    def _agent(self, anchor):
        return SimpleNamespace(
            _usage_anchor=anchor,
            tools=None,
            api_mode="",
            provider="openai",
        )

    def test_preflight_prefers_anchor(self):
        messages = _history_with_images(10)
        anchor = capture_usage_anchor(50_000, 250, messages)
        messages.append(_msg("assistant", "reply"))
        messages.append(_msg("user", "ok"))
        agent = self._agent(anchor)

        got = _preflight_request_tokens(agent, messages, "SYSTEM PROMPT " * 500)
        expected = anchored_context_tokens(messages, anchor)
        assert got == expected
        # The anchored figure ignores the (already-counted) system prompt
        # text passed in — provider usage includes the real one.
        assert 50_000 < got < 50_500

    def test_preflight_falls_back_without_anchor(self):
        messages = _history_with_images(3)
        agent = self._agent(None)
        got = _preflight_request_tokens(agent, messages, "sys")
        # Pure heuristic: flat image cost dominates.
        assert got >= 4_500

    def test_sabotage_disabling_anchor_changes_result(self):
        """Prove the anchored path produced the number: with the anchor
        removed (the sabotage), the same inputs yield the heuristic figure,
        which diverges by thousands of tokens on an image-heavy history."""
        messages = _history_with_images(10)
        anchor = capture_usage_anchor(12_000, 100, messages)
        messages.append(_msg("assistant", "reply"))
        messages.append(_msg("user", "ok"))

        anchored_result = _preflight_request_tokens(
            self._agent(anchor), messages, ""
        )
        sabotaged_result = _preflight_request_tokens(
            self._agent(None), messages, ""
        )
        assert sabotaged_result - anchored_result > 2_800


class TestCompressionTriggerUsesAnchor:
    def test_threshold_decision_flips_with_anchor(self):
        """An image-heavy history the heuristic pushes over a 15K threshold
        stays under it when the provider reports the real 12K prompt."""
        messages = _history_with_images(10)
        anchor = capture_usage_anchor(12_000, 100, messages)
        messages.append(_msg("assistant", "reply"))

        threshold = 15_000
        heuristic = estimate_messages_tokens_rough(messages)
        anchored = anchored_context_tokens(messages, anchor)
        assert heuristic >= threshold  # old behavior: spurious compression
        assert anchored is not None and anchored < threshold


class TestCodexAppServerAnchor:
    """The codex_app_server runtime bypasses the conversation loop, so its
    usage recording is the only site that can maintain agent._usage_anchor.
    Without it, hermes-mode preflight falls back to the rough mirror-transcript
    heuristic, which grows monotonically (native compaction preserves the
    mirror) and fires thread compaction on tiny real threads (#100381)."""

    def _agent(self, anchor=None):
        return SimpleNamespace(
            _usage_anchor=anchor,
            session_api_calls=0,
            session_prompt_tokens=0,
            session_completion_tokens=0,
            session_total_tokens=0,
            session_input_tokens=0,
            session_output_tokens=0,
            session_cache_read_tokens=0,
            session_cache_write_tokens=0,
            session_reasoning_tokens=0,
            context_compressor=None,
            event_callback=None,
            _session_db=None,
            model="codex-test-model",
            provider="openai",
            base_url=None,
        )

    def _turn(self, usage):
        return SimpleNamespace(token_usage_last=usage, model_context_window=None)

    def _usage(self, input_tokens=12_000, output_tokens=100):
        return {
            "inputTokens": input_tokens,
            "cachedInputTokens": 0,
            "outputTokens": output_tokens,
            "reasoningOutputTokens": 0,
            "totalTokens": input_tokens + output_tokens,
        }

    def test_turn_usage_sets_anchor(self):
        from agent.codex_runtime import _record_codex_app_server_usage

        messages = _history_with_images(10)
        agent = self._agent()

        _record_codex_app_server_usage(
            agent, self._turn(self._usage()), messages=messages
        )

        anchor = agent._usage_anchor
        assert anchor is not None
        assert anchor["prompt_tokens"] == 12_000
        assert anchor["base_count"] == len(messages)

        # The next turn's preflight estimate anchors on provider truth plus
        # only the appended delta — not the flat-1500-per-image heuristic.
        messages.append(_msg("user", "follow-up"))
        got = _preflight_request_tokens(agent, messages, "")
        assert 12_000 < got < 12_200

    def test_usage_less_turn_keeps_previous_anchor(self):
        from agent.codex_runtime import _record_codex_app_server_usage

        messages = _history_with_images(2)
        prior = capture_usage_anchor(9_000, 50, messages)
        agent = self._agent(anchor=prior)

        _record_codex_app_server_usage(agent, self._turn(None), messages=messages)

        assert agent._usage_anchor is prior


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
