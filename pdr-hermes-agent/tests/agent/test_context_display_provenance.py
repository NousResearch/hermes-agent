"""Estimated occupancy must not masquerade as provider usage (or vice versa)."""
from types import SimpleNamespace
from unittest.mock import patch

from agent.context_breakdown import compute_session_context_breakdown, render_context_breakdown_lines
from agent.context_compressor import ContextCompressor
from agent.usage_anchor import capture_usage_anchor


def test_context_provenance_follows_the_selected_number():
    comp = ContextCompressor(model="fixture", config_context_length=100_000, quiet_mode=True)
    agent = SimpleNamespace(context_compressor=comp, tools=[], model="fixture")
    messages = [{"role": "user", "content": "fixture question"}]
    with patch("agent.system_prompt.build_system_prompt_parts", return_value={"stable": "fixture"}):
        for source in ("local_estimate", "provider_usage", "provider_usage_plus_estimate"):
            if source == "provider_usage":
                comp.update_from_response({"prompt_tokens": 1234, "completion_tokens": 20})
                agent._usage_anchor = capture_usage_anchor(1234, 20, messages)
                messages.append({"role": "assistant", "content": "fixture answer"})
            elif source == "provider_usage_plus_estimate":
                messages.append({"role": "user", "content": "a new unpriced question"})
            data = compute_session_context_breakdown(agent, messages)
            assert data.get("context_source") == source
            estimated = source != "provider_usage"
            assert data.get("context_estimated") is estimated
            summary = next(s for s in render_context_breakdown_lines(data) if s.startswith("Context window:"))
            assert ("Context window: ~" in summary) is estimated


def test_preflight_seed_does_not_label_actual_usage_estimated():
    from tui_gateway.server import _get_usage
    comp = ContextCompressor(model="fixture", config_context_length=100_000, quiet_mode=True)
    agent = SimpleNamespace(context_compressor=comp, model="fixture")
    comp.maybe_seed_preflight_display_tokens(1234)
    assert _get_usage(agent).get("context_estimated") is True
    comp.update_from_response({"prompt_tokens": 1234, "completion_tokens": 20})
    assert _get_usage(agent).get("context_estimated") is False
    assert _get_usage(agent)["context_used"] == 1234
    comp.last_prompt_tokens = -1
    assert "context_used" not in _get_usage(agent)
    from agent.context_breakdown import context_display_source
    # A cleared live gauge must not re-label a persisted provider fallback.
    assert context_display_source(comp) == "provider_usage"
