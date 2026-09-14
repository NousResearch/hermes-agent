"""``moa.<preset>.sampling: {mode: random, k: N}`` bounds advisor fan-out to a random
subset of the enabled reference pool per fan-out, instead of running every enabled slot.

Unit coverage for ``agent.moa_loop._select_reference_models`` plus an E2E check through
``MoAChatCompletions`` (mirrors ``test_moa_fanout_cadence.py``'s config/fake-LLM harness)
proving only k advisor calls happen out of a larger configured pool.
"""

from types import SimpleNamespace

from agent.moa_loop import _select_reference_models


def _slots(n):
    return [{"provider": "p", "model": f"m{i}", "enabled": True} for i in range(n)]


def test_select_reference_models_mode_all_is_noop():
    pool = _slots(4)
    assert _select_reference_models(pool, {"mode": "all"}) == pool
    assert _select_reference_models(pool, None) == pool
    assert _select_reference_models(pool, {}) == pool


def test_select_reference_models_random_draws_k():
    pool = _slots(5)
    selected = _select_reference_models(pool, {"mode": "random", "k": 2})
    assert len(selected) == 2
    # Every drawn slot is a genuine member of the pool (no fabrication).
    assert all(slot in pool for slot in selected)
    # No duplicates.
    labels = [s["model"] for s in selected]
    assert len(set(labels)) == len(labels)


def test_select_reference_models_random_k_at_or_above_pool_runs_whole_pool():
    pool = _slots(3)
    assert _select_reference_models(pool, {"mode": "random", "k": 3}) == pool
    assert _select_reference_models(pool, {"mode": "random", "k": 10}) == pool


def test_select_reference_models_random_empty_pool_is_noop():
    assert _select_reference_models([], {"mode": "random", "k": 2}) == []


def _response(content="done"):
    message = SimpleNamespace(content=content, tool_calls=[])
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=None, model="fake-model")


def _sampling_config(home, k=1):
    home.mkdir()
    (home / "config.yaml").write_text(
        f"""
moa:
  default_preset: review
  presets:
    review:
      sampling:
        mode: random
        k: {k}
      reference_models:
        - provider: openai-codex
          model: gpt-5.5
        - provider: openrouter
          model: deepseek/deepseek-v4-pro
        - provider: anthropic
          model: claude-sonnet-5
      aggregator:
        provider: openrouter
        model: anthropic/claude-opus-4.8
""".strip(),
        encoding="utf-8",
    )


def test_random_sampling_bounds_advisor_calls_below_pool_size(monkeypatch, tmp_path):
    """A 3-slot pool with sampling k=1 must only ever run 1 advisor per fan-out,
    never all 3 — proving the config knob actually reaches the fan-out call."""
    home = tmp_path / ".hermes"
    _sampling_config(home, k=1)
    monkeypatch.setenv("HERMES_HOME", str(home))

    ref_runs = []

    def fake_call_llm(**kwargs):
        if kwargs["task"] == "moa_reference":
            ref_runs.append(kwargs["model"])
            return _response(f"advice #{len(ref_runs)}")
        return _response("acted")

    monkeypatch.setattr("agent.moa_loop.call_llm", fake_call_llm)

    from agent.moa_loop import MoAChatCompletions

    facade = MoAChatCompletions("review")
    facade.create(messages=[{"role": "user", "content": "task"}], tools=[])

    assert len(ref_runs) == 1
