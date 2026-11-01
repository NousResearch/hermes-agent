"""Regression: the compaction TRIGGER must not charge stale reasoning on re-export
routes whose provider accounting ignores the replayed bytes.

``stale_thinking_reaches_wire`` is the one predicate both the trigger estimator and
the tail-budget walks share (see ``tests/agent/test_estimator_parity.py`` for the
#84371 dead-loop it exists to prevent). It answers whether stale thinking is on the
wire — but on some re-export providers the field is accepted yet NOT counted in the
reported ``prompt_tokens``. A live probe against Ollama Cloud
(``deepseek-v4.1-flash``) showed a ~24k-token ``reasoning_content`` replay leaving
``prompt_tokens`` unchanged (42 -> 42): the echo is protocol-required, the accounting
ignores it.

Charging those bytes made the trigger estimate ~2.7x the provider's real prompt on a
reasoning-heavy session (1,415,136 estimated vs 528,988 real for a 533-message
session), firing compaction far too early. The estimator now excludes those provider
ids; the send-side policy (``needs_reasoning_echo`` / ``apply_reasoning_content_policy``)
is untouched, so no request changes shape.
"""

from agent.context_compressor import ContextCompressor, _estimate_msg_budget_tokens
from agent.message_sanitization import (
    matches_reasoning_echo_family,
    needs_reasoning_echo,
    stale_thinking_reaches_wire,
)
from agent.model_metadata import estimate_messages_tokens_rough

STALE_THINKING = "considering the next move carefully... " * 200  # ~2K tok


def _reasoning_heavy_session(n_turns: int = 40) -> list:
    msgs = [{"role": "system", "content": "You are Hermes."}]
    msgs.append({"role": "user", "content": "do the big task"})
    for i in range(n_turns):
        msgs.append({
            "role": "assistant", "content": f"step {i}",
            "reasoning_content": STALE_THINKING,
            "tool_calls": [{"id": f"c{i}", "type": "function",
                            "function": {"name": "t", "arguments": "{}"}}],
        })
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": f"r{i}"})
    return msgs


_OLLAMA = ("", "ollama-cloud", "deepseek-v4.1-flash", "https://ollama.com/v1")
_NATIVE = ("", "deepseek", "deepseek-reasoner", "https://api.deepseek.com")
_OPENROUTER = ("", "openrouter", "deepseek/deepseek-v3", "https://openrouter.ai")


class TestReexportExclusion:
    def test_ollama_cloud_is_excluded_from_the_estimate_charge(self):
        assert stale_thinking_reaches_wire(*_OLLAMA) is False

    def test_ollama_cloud_still_requires_the_send_side_echo(self):
        # Protocol side unchanged: the send policy must keep attaching the field.
        assert needs_reasoning_echo(*_OLLAMA[1:]) is True
        assert matches_reasoning_echo_family(
            "deepseek", "ollama-cloud", "deepseek-v4.1-flash", "https://ollama.com/v1"
        ) is True

    def test_native_and_openrouter_routes_still_charge_stale_thinking(self):
        assert stale_thinking_reaches_wire(*_NATIVE) is True
        assert stale_thinking_reaches_wire(*_OPENROUTER) is True

    def test_codex_responses_still_never_charges(self):
        assert stale_thinking_reaches_wire(
            "codex_responses", "deepseek", "deepseek-v4-flash", ""
        ) is False

    def test_exclusion_is_provider_keyed_not_host_or_model_substring(self):
        # The same model behind a DIFFERENT provider must keep the full charge.
        assert stale_thinking_reaches_wire(
            "", "deepseek", "deepseek-v4.1-flash", "https://ollama.com/v1"
        ) is True


class TestTriggerWalkLockstep:
    def test_trigger_and_walk_land_in_the_same_size_class_on_the_reexport(self):
        from agent.context_compressor import _last_assistant_index

        msgs = _reasoning_heavy_session()
        charge = stale_thinking_reaches_wire(*_OLLAMA)
        assert charge is False
        newest = _last_assistant_index(msgs)
        trigger = estimate_messages_tokens_rough(msgs, charge_stale_thinking=charge)
        walk = sum(
            _estimate_msg_budget_tokens(m, charge_stale_thinking=(charge or i == newest))
            for i, m in enumerate(msgs)
        )
        assert trigger <= walk * 2 and walk <= trigger * 2

    def test_excluding_stale_thinking_cuts_the_hot_figure_hard(self):
        msgs = _reasoning_heavy_session()
        full = estimate_messages_tokens_rough(msgs, charge_stale_thinking=True)
        route = estimate_messages_tokens_rough(
            msgs, charge_stale_thinking=stale_thinking_reaches_wire(*_OLLAMA)
        )
        assert route < full / 2

    def test_newest_turn_thinking_is_still_charged(self):
        msgs = _reasoning_heavy_session(n_turns=2)
        stripped = estimate_messages_tokens_rough(msgs, charge_stale_thinking=False)
        no_thinking = estimate_messages_tokens_rough(
            [{k: v for k, v in m.items() if k not in ("reasoning", "reasoning_content")}
             for m in msgs]
        )
        assert stripped > no_thinking


class TestCompressorRoutePredicate:
    def test_compressor_walk_route_matches_trigger_on_the_reexport(self):
        cc = ContextCompressor(
            model="deepseek-v4.1-flash", provider="ollama-cloud", api_mode="",
            base_url="https://ollama.com/v1", quiet_mode=True, config_context_length=200_000,
        )
        assert cc._stale_thinking_on_wire() is False

    def test_compressor_walk_route_still_true_for_native_deepseek(self):
        cc = ContextCompressor(
            model="deepseek-reasoner", provider="deepseek", api_mode="",
            base_url="https://api.deepseek.com", quiet_mode=True, config_context_length=200_000,
        )
        assert cc._stale_thinking_on_wire() is True
