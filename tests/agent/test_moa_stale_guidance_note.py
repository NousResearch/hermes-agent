"""Cached advisor guidance must announce that it predates the tool results below it.

With `user_turn` (and off-cadence `every_n`) fanout the advisors run once per user turn and
their guidance is replayed verbatim into every later iteration of that turn. A reference that
proposes a tool call therefore keeps proposing it after the acting model already ran it and
got an answer, which is how a clarify card gets issued twice in one turn: once answered, once
a duplicate the user has to dismiss.
"""

from types import SimpleNamespace


def _response(content="done", *, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")],
                           usage=None, model="fake-model")


def _config(home, fanout="user_turn"):
    home.mkdir()
    (home / "config.yaml").write_text(
        f"""
moa:
  default_preset: review
  presets:
    review:
      fanout: "{fanout}"
      reference_models:
        - provider: openai-codex
          model: gpt-5.5
      aggregator:
        provider: openrouter
        model: anthropic/claude-opus-4.8
""".strip(),
        encoding="utf-8",
    )


def _install_fake_llm(monkeypatch, ref_runs):
    def fake_call_llm(**kwargs):
        if kwargs["task"] == "moa_reference":
            ref_runs.append(kwargs["model"])
            return _response("call the clarify tool with three questions")
        return _response("acted")

    monkeypatch.setattr("agent.moa_loop.call_llm", fake_call_llm)


def _after_tool_call(base):
    return base + [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "function": {"name": "clarify", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": '{"responses": [{"user_response": "Development"}]}'},
    ]


def _prepared(monkeypatch, tmp_path, fanout="user_turn"):
    home = tmp_path / ".hermes"
    _config(home, fanout)
    monkeypatch.setenv("HERMES_HOME", str(home))
    ref_runs = []
    _install_fake_llm(monkeypatch, ref_runs)

    from agent.moa_loop import MoAChatCompletions

    facade = MoAChatCompletions("review")
    base = [{"role": "user", "content": "use the clarify tool now"}]
    first = facade.create(messages=base, tools=[], _moa_prepare_only=True)
    second = facade.create(messages=_after_tool_call(base), tools=[], _moa_prepare_only=True)
    return first, second, ref_runs


def test_reused_guidance_is_marked_as_predating_the_tool_results(monkeypatch, tmp_path):
    from agent.moa_loop import _STALE_GUIDANCE_NOTE

    first, second, ref_runs = _prepared(monkeypatch, tmp_path)

    assert len(ref_runs) == 1, "user_turn fanout reuses the first run's advisors"
    assert _STALE_GUIDANCE_NOTE not in first["guidance"]
    assert _STALE_GUIDANCE_NOTE in second["guidance"]
    # The advice itself is still handed over unchanged.
    assert "call the clarify tool with three questions" in second["guidance"]


def test_fresh_guidance_carries_no_note(monkeypatch, tmp_path):
    """per_iteration advisors see the tool result themselves, so nothing is stale."""
    from agent.moa_loop import _STALE_GUIDANCE_NOTE

    first, second, ref_runs = _prepared(monkeypatch, tmp_path, fanout="per_iteration")

    assert len(ref_runs) == 2
    assert _STALE_GUIDANCE_NOTE not in first["guidance"]
    assert _STALE_GUIDANCE_NOTE not in second["guidance"]

