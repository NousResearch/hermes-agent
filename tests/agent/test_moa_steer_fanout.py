"""MoA + /steer (issue #121342): a mid-turn steer row must not force a full
reference re-fan-out, and the advisory view must not end on the raw steer row.

Mechanism 1: the steer is persisted as a real role:user row, so the user_turn
fan-out cache key (prefix up to the last real user message) always MISSES.
Mechanism 2: _reference_messages only appends _ADVISORY_INSTRUCTION after an
assistant turn, so a steer-terminated view ends on OUT-OF-BAND meta-text.
"""

from types import SimpleNamespace


def _response(content="done"):
    message = SimpleNamespace(content=content, tool_calls=[])
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=None, model="fake-model")


def _review_config(home):
    home.mkdir()
    (home / "config.yaml").write_text(
        """
moa:
  default_preset: review
  presets:
    review:
      reference_models:
        - provider: openai-codex
          model: gpt-5.5
      aggregator:
        provider: openrouter
        model: anthropic/claude-opus-4.8
""".strip(),
        encoding="utf-8",
    )


def _install_fake_llm(monkeypatch, ref_runs, advisor_seen):
    def fake_call_llm(**kwargs):
        if kwargs["task"] == "moa_reference":
            ref_runs.append(kwargs["model"])
            advisor_seen.append(list(kwargs["messages"]))
            return _response(f"advice #{len(ref_runs)}")
        return _response("acted")

    monkeypatch.setattr("agent.moa_loop.call_llm", fake_call_llm)


def _steer_api_row(monkeypatch_text=None):
    """A steer row as MoA actually receives it: build_api_messages strips
    display_kind (PERSISTENCE_ONLY_MESSAGE_FIELDS), so only marker text remains."""
    from agent.prompt_builder import steer_user_row

    row = steer_user_row(monkeypatch_text or "keep going, focus on auth")
    row.pop("display_kind", None)
    return row


def test_steer_row_does_not_trigger_reference_refanout(monkeypatch, tmp_path):
    """Steer iterations reuse the turn-start advice (cache HIT): exactly 1
    advisor run across turn-start + steer-appended creates."""
    home = tmp_path / ".hermes"
    _review_config(home)
    monkeypatch.setenv("HERMES_HOME", str(home))

    ref_runs, advisor_seen = [], []
    _install_fake_llm(monkeypatch, ref_runs, advisor_seen)

    from agent.moa_loop import MoAChatCompletions

    facade = MoAChatCompletions("review")
    base = [{"role": "user", "content": "do the thing"}]
    facade.create(messages=list(base), tools=[])
    assert len(ref_runs) == 1

    facade.create(messages=list(base) + [_steer_api_row()], tools=[])
    assert len(ref_runs) == 1


def test_steer_terminated_advisory_view_ends_on_instruction(monkeypatch, tmp_path):
    """The advisory view must end on _ADVISORY_INSTRUCTION, not on the raw
    OUT-OF-BAND steer meta-text."""
    from agent.moa_loop import _ADVISORY_INSTRUCTION, _reference_messages

    view = _reference_messages(
        [
            {"role": "user", "content": "do the thing"},
            {"role": "assistant", "content": "working", "tool_calls": []},
            _steer_api_row(),
        ]
    )
    assert view[-1] == {"role": "user", "content": _ADVISORY_INSTRUCTION}
    assert "[OUT-OF-BAND USER MESSAGE" not in view[-1]["content"]


def test_steer_row_keeps_stale_guidance_note(monkeypatch, tmp_path):
    """A steer HIT reuses turn-start guidance that predates tool results, so it
    stays marked with the stale note like any other tool-iteration HIT."""
    home = tmp_path / ".hermes"
    _review_config(home)
    monkeypatch.setenv("HERMES_HOME", str(home))

    ref_runs, advisor_seen = [], []
    _install_fake_llm(monkeypatch, ref_runs, advisor_seen)

    from agent.moa_loop import _STALE_GUIDANCE_NOTE, MoAChatCompletions

    facade = MoAChatCompletions("review")
    tool_iter = [
        {"role": "user", "content": "do the thing"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "result 1"},
    ]
    first = facade.create(messages=list(tool_iter), tools=[], _moa_prepare_only=True)
    steered = facade.create(
        messages=list(tool_iter) + [_steer_api_row()], tools=[], _moa_prepare_only=True
    )
    assert len(ref_runs) == 1
    assert "advice #1" in steered["guidance"]
    assert _STALE_GUIDANCE_NOTE in steered["guidance"]
    assert steered["guidance"].replace(_STALE_GUIDANCE_NOTE, "") == first["guidance"]
