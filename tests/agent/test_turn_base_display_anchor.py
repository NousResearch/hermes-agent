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

from agent.model_metadata import (
    anchored_context_tokens,
    capture_usage_anchor,
    estimate_messages_tokens_rough,
)


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


class TestCliStatusSnapshotPrefersTurnBaseAnchor:
    def _agent_with(self, last_prompt_tokens, messages, anchor):
        compressor = SimpleNamespace(
            last_prompt_tokens=last_prompt_tokens,
            context_length=1_000_000,
            compression_count=0,
        )
        return SimpleNamespace(
            context_compressor=compressor,
            _session_messages=messages,
            _turn_base_usage_anchor=anchor,
        )

    def _snapshot_context_tokens(self, agent):
        """Mirror the cli.py snapshot block's context_tokens resolution."""
        compressor = agent.context_compressor
        context_tokens = getattr(compressor, "last_prompt_tokens", 0) or 0
        if context_tokens < 0:
            context_tokens = 0
        msgs = getattr(agent, "_session_messages", None)
        anchored = anchored_context_tokens(
            msgs if isinstance(msgs, list) else [],
            getattr(agent, "_turn_base_usage_anchor", None),
            charge_stale_thinking=False,
        )
        if anchored is not None and anchored > 0:
            context_tokens = anchored
        return context_tokens

    def test_turn_base_anchor_wins_over_inflated_last_request(self):
        messages = [_msg("user", "start"), _msg("assistant", "reply")]
        anchor = capture_usage_anchor(600_000, 500, messages)
        messages.append(_msg("assistant", "anchored reply"))
        agent = self._agent_with(850_000, messages, anchor)
        # Bar shows the durable figure, not the inflated last request.
        tokens = self._snapshot_context_tokens(agent)
        assert 600_000 <= tokens < 650_000

    def test_fallback_without_anchor(self):
        agent = self._agent_with(123_456, [_msg("user", "x")], None)
        assert self._snapshot_context_tokens(agent) == 123_456

    def test_stale_anchor_falls_back(self):
        messages = [_msg("user", "start"), _msg("assistant", "reply")]
        anchor = capture_usage_anchor(50_000, 10, messages)
        agent = self._agent_with(77_000, [_msg("user", "rebuilt")], anchor)
        # Compaction rebuilt the list: structural check fails, raw fallback.
        assert self._snapshot_context_tokens(agent) == 77_000

    def test_negative_sentinel_still_clamped(self):
        agent = self._agent_with(-1, [], None)
        assert self._snapshot_context_tokens(agent) == 0


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
    def test_compression_invalidation_clears_both(self):
        import inspect
        from agent import conversation_compression

        src = inspect.getsource(conversation_compression)
        block = src.split("agent._usage_anchor = None", 1)[1][:200]
        assert "_turn_base_usage_anchor = None" in block

    def test_codex_native_invalidation_clears_both(self):
        import inspect
        from agent import codex_runtime

        src = inspect.getsource(codex_runtime)
        block = src.split("agent._usage_anchor = None", 1)[1][:200]
        assert "_turn_base_usage_anchor = None" in block

    def test_agent_init_defines_turn_base_anchor(self):
        import inspect
        from agent import agent_init

        src = inspect.getsource(agent_init)
        # Init defaults are declared in the ``_USAGE_STATE`` table.
        assert '"_turn_base_usage_anchor": None' in src
