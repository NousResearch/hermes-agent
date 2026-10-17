"""Thinking-toggle / reasoning_effort wire invariants shared by the Moonshot- and
DeepSeek-style chat_completions profiles (all route through
``agent.reasoning_effort.thinking_toggle_extras``)."""

import pytest

from agent.reasoning_effort import DEEPSEEK_V4_EFFORTS
from providers import get_provider_profile

REASONING_MATRIX = (
    None,
    {"enabled": False},
    {"enabled": True},
    *({"enabled": True, "effort": e} for e in ("low", "medium", "high", "xhigh", "max", "none")),
)


@pytest.mark.parametrize("reasoning_config", REASONING_MATRIX, ids=str)
def test_thinking_toggle_and_effort_never_both_on_moonshot_wire(reasoning_config):
    for provider, model in (("kimi-coding", "kimi-k3"), ("opencode-go", "kimi-k2.6"), ("opencode-go", "deepseek-v4-pro")):
        extra_body, top_level = get_provider_profile(provider).build_api_kwargs_extras(
            reasoning_config=reasoning_config, model=model
        )
        assert not ("thinking" in extra_body and "reasoning_effort" in top_level), (provider, model, reasoning_config)

    # DeepSeek's own API wants the toggle on every request (omitting it defaults thinking on
    # and then demands reasoning_content echoes); effort rides alongside only when supported.
    extra_body, top_level = get_provider_profile("deepseek").build_api_kwargs_extras(
        reasoning_config=reasoning_config, model="deepseek-v4-pro"
    )
    assert extra_body["thinking"]["type"] in ("enabled", "disabled")
    assert top_level.get("reasoning_effort", DEEPSEEK_V4_EFFORTS[0]) in DEEPSEEK_V4_EFFORTS
    if isinstance(reasoning_config, dict) and reasoning_config.get("enabled") is False:
        assert (extra_body, top_level) == ({"thinking": {"type": "disabled"}}, {})


def test_batch_reasoning_disabled_disables_deepseek_and_kimi_wires(monkeypatch):
    """``hermes batch --reasoning_disabled`` sends a bare ``effort=none`` config.

    Keep that batch-runner shape in the regression rather than hand-writing an
    ``enabled=False`` config: profile wires must treat both disabled forms alike.
    """
    import sys
    from types import SimpleNamespace

    # The profile suite does not require Fire; stub only its import so this
    # regression can exercise batch_runner.main's config construction.
    monkeypatch.setitem(sys.modules, "fire", SimpleNamespace())
    import batch_runner

    captured = {}

    class FakeRunner:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self, *, resume):
            assert resume is False

    monkeypatch.setattr(batch_runner, "BatchRunner", FakeRunner)
    batch_runner.main(dataset_file="prompts.jsonl", batch_size=1, run_name="disabled", reasoning_disabled=True)

    reasoning_config = captured["reasoning_config"]
    assert reasoning_config == {"effort": "none"}
    for provider, model in (("deepseek", "deepseek-v4-pro"), ("kimi-coding", "kimi-k3")):
        extra_body, top_level = get_provider_profile(provider).build_api_kwargs_extras(
            reasoning_config=reasoning_config, model=model
        )
        assert (extra_body, top_level) == ({"thinking": {"type": "disabled"}}, {})


@pytest.mark.parametrize("reasoning_config", REASONING_MATRIX, ids=str)
def test_ox_alpha_translation_on_zen(reasoning_config):
    extra_body, top_level = get_provider_profile("opencode-zen").build_api_kwargs_extras(
        reasoning_config=reasoning_config, model="x-preview-f-free"
    )
    # Ox Alpha's wire carries reasoning_effort at top level — never the thinking
    # toggle — and every emitted effort is inside the wire vocabulary.
    assert "thinking" not in extra_body
    effort = top_level.get("reasoning_effort")
    assert effort in (None, "low", "high", "max"), (reasoning_config, effort)
