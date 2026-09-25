"""Turn-base display anchor: the context meter shows durable-transcript cost.

On reasoning models a long tool loop replays the current turn's thinking +
scaffolding on every request, so the LAST request's ``prompt_tokens`` can
exceed the durable transcript by hundreds of K — all of which evaporates at
the turn boundary. Display surfaces (CLI status bar, /context breakdown)
therefore anchor on the turn's FIRST response (``_turn_base_usage_anchor``)
plus a stale-thinking-free delta estimate, instead of the raw last-request
figure. Compression trigger math is unchanged (real last-request usage).

Covers:
  * anchored_context_tokens(charge_stale_thinking=False) excludes stale
    reasoning text in the delta while keeping the newest assistant turn;
  * the CLI status snapshot prefers the turn-base anchored figure over
    compressor.last_prompt_tokens and falls back cleanly without an anchor;
  * compute_session_context_breakdown prefers the turn-base anchor over the
    last-response anchor;
  * invalidation sites clear _turn_base_usage_anchor alongside _usage_anchor.
"""

from types import SimpleNamespace

from agent.model_metadata import estimate_messages_tokens_rough
from agent.usage_anchor import anchored_context_tokens, capture_usage_anchor

def _msg(role, content, **extra):
    m = {"role": role, "content": content}
    m.update(extra)
    return m

class TestChargeStaleThinkingKwarg:
    def test_delta_excludes_stale_reasoning(self):
        messages = [_msg("user", "start"), _msg("assistant", "base reply")]
        anchor = capture_usage_anchor(10_000, 100, messages)
        assert anchor is not None

        # Simulate a tool loop appending reasoning-heavy assistant turns.
        big_thinking = "deliberation " * 5_000  # ~65K chars ≈ 16K tokens
        messages.append(_msg("assistant", "the anchored reply itself"))
        messages.append(
            _msg("assistant", "step one", reasoning_content=big_thinking)
        )
        messages.append(_msg("tool", "tool output", tool_call_id="c1"))
        messages.append(
            _msg("assistant", "step two", reasoning_content=big_thinking)
        )

        charged = anchored_context_tokens(messages, anchor)
        uncharged = anchored_context_tokens(
            messages, anchor, charge_stale_thinking=False
        )
        assert charged is not None and uncharged is not None
        # Stale thinking on the non-newest assistant message is excluded;
        # the newest assistant message keeps its reasoning charge.
        one_thinking_tokens = estimate_messages_tokens_rough(
            [_msg("assistant", "", reasoning_content=big_thinking)]
        )
        assert charged - uncharged >= one_thinking_tokens * 0.9
        assert uncharged >= 10_000 + 100  # anchor base still counted exactly

    def test_default_remains_full_charge(self):
        messages = [_msg("user", "s"), _msg("assistant", "r")]
        anchor = capture_usage_anchor(1_000, 10, messages)
        messages.append(_msg("assistant", "reply"))
        assert anchored_context_tokens(messages, anchor) == anchored_context_tokens(
            messages, anchor, charge_stale_thinking=True
        )

class TestContextBreakdownPrefersTurnBaseAnchor:
    def test_breakdown_uses_turn_base_over_last_response(self, monkeypatch):
        from agent import context_breakdown as cb

        messages = [_msg("user", "start"), _msg("assistant", "reply")]
        turn_base = capture_usage_anchor(400_000, 200, messages)
        messages.append(_msg("assistant", "anchored reply"))
        last_anchor = capture_usage_anchor(900_000, 50, messages)

        agent = SimpleNamespace(
            _usage_anchor=last_anchor,
            _turn_base_usage_anchor=turn_base,
            _memory_store=None,
            tools=[],
            model="test/model",
            context_compressor=SimpleNamespace(
                context_length=1_000_000, last_prompt_tokens=900_000
            ),
        )
        monkeypatch.setattr(
            "agent.system_prompt.build_system_prompt_parts",
            lambda a: {"stable": "sys", "context": "", "volatile": ""},
        )
        payload = cb.compute_session_context_breakdown(agent, messages)
        assert 400_000 <= payload["context_used"] < 450_000

    def test_breakdown_falls_back_to_last_response_anchor(self, monkeypatch):
        from agent import context_breakdown as cb

        messages = [_msg("user", "start"), _msg("assistant", "reply")]
        last_anchor = capture_usage_anchor(300_000, 50, messages)

        agent = SimpleNamespace(
            _usage_anchor=last_anchor,
            _turn_base_usage_anchor=None,
            _memory_store=None,
            tools=[],
            model="test/model",
            context_compressor=SimpleNamespace(
                context_length=1_000_000, last_prompt_tokens=1
            ),
        )
        monkeypatch.setattr(
            "agent.system_prompt.build_system_prompt_parts",
            lambda a: {"stable": "sys", "context": "", "volatile": ""},
        )
        payload = cb.compute_session_context_breakdown(agent, messages)
        assert payload["context_used"] >= 300_000

class TestInvalidationSitesClearTurnBaseAnchor:
    def test_clearing_the_anchor_clears_the_turn_base_too(self):
        """Compaction and the codex-native rewrite clear via set_usage_anchor(None): both the
        last-response anchor and the display turn-base anchor go; a later same-turn capture
        (turn_base=False) leaves the turn-base untouched."""
        from agent.usage_anchor import set_usage_anchor

        messages = [_msg("user", "a"), _msg("assistant", "b")]
        agent = SimpleNamespace(_usage_anchor=None, _turn_base_usage_anchor=None, _session_db=None, session_id=None)
        first = capture_usage_anchor(1_000, 10, messages)
        set_usage_anchor(agent, first, turn_base=True)
        set_usage_anchor(agent, capture_usage_anchor(2_000, 10, messages))
        assert agent._turn_base_usage_anchor is first
        set_usage_anchor(agent, None)
        assert agent._usage_anchor is None and agent._turn_base_usage_anchor is None
