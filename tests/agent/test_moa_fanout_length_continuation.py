"""user_turn fan-out cadence must survive a length-continuation.

A response that hits the output limit is continued by appending a synthetic
``role="user"`` nudge (``agent/turn_truncation.py``). The user_turn cache key stops at
"the last REAL user message", so before the fix that nudge re-signed the prefix and the
whole advisor fan-out re-ran on the grown context — once per continuation. Observed on a
live preset: four fan-outs (~$0.73 each) for a single user turn.
"""

from types import SimpleNamespace


def _response(content="done", *, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=None, model="fake-model")


def _preset_config(home, fanout="user_turn"):
    home.mkdir()
    (home / "config.yaml").write_text(
        f"""
moa:
  default_preset: review
  presets:
    review:
      fanout: "{fanout}"
      reference_models:
        - provider: openrouter
          model: openai/gpt-6.1-sol
      aggregator:
        provider: openrouter
        model: anthropic/claude-opus-5.5
""".strip(),
        encoding="utf-8",
    )


def _install_fake_llm(monkeypatch, ref_runs):
    def fake_call_llm(**kwargs):
        if kwargs["task"] == "moa_reference":
            ref_runs.append(kwargs["model"])
            return _response(f"advice #{len(ref_runs)}")
        return _response("acted")

    monkeypatch.setattr("agent.moa_loop.call_llm", fake_call_llm)


def _facade():
    from agent.moa_loop import MoAChatCompletions

    return MoAChatCompletions("review")


def test_length_continuation_nudge_does_not_re_run_references(monkeypatch, tmp_path):
    """A truncation continuation is the same user turn: advisors must not run twice."""
    home = tmp_path / ".hermes"
    _preset_config(home)
    monkeypatch.setenv("HERMES_HOME", str(home))

    ref_runs = []
    _install_fake_llm(monkeypatch, ref_runs)

    from agent.conversation_loop import _LENGTH_CONTINUATION_OUTPUT_LIMIT

    facade = _facade()
    base = [{"role": "user", "content": "schreibe eine lange Begründung"}]
    facade.create(messages=list(base), tools=[])
    assert len(ref_runs) == 1

    continued = list(base) + [
        {"role": "assistant", "content": "Teil 1 der Begründung …"},
        {
            "role": "user",
            "content": _LENGTH_CONTINUATION_OUTPUT_LIMIT,
            "_length_continuation_nudge": True,
        },
    ]
    facade.create(messages=continued, tools=[])

    assert len(ref_runs) == 1, "the continuation re-ran the advisor fan-out"


def test_network_continuation_nudge_does_not_re_run_references(monkeypatch, tmp_path):
    """The transport-cutoff variant carries the same rule (whole bug class, not one string)."""
    home = tmp_path / ".hermes"
    _preset_config(home)
    monkeypatch.setenv("HERMES_HOME", str(home))

    ref_runs = []
    _install_fake_llm(monkeypatch, ref_runs)

    from agent.conversation_loop import (
        _LEGACY_LENGTH_CONTINUATION_NETWORK_STUB,
        _LENGTH_CONTINUATION_NETWORK_STUB,
    )

    facade = _facade()
    base = [{"role": "user", "content": "task"}]
    facade.create(messages=list(base), tools=[])
    for stub in (_LENGTH_CONTINUATION_NETWORK_STUB, _LEGACY_LENGTH_CONTINUATION_NETWORK_STUB):
        facade.create(
            messages=list(base)
            + [{"role": "assistant", "content": "cut …"}, {"role": "user", "content": stub}],
            tools=[],
        )

    assert len(ref_runs) == 1, "a transport-continuation stub re-ran the fan-out"


def test_genuine_new_user_turn_still_runs_references(monkeypatch, tmp_path):
    """Counter-check: the fix must not swallow real turn boundaries."""
    home = tmp_path / ".hermes"
    _preset_config(home)
    monkeypatch.setenv("HERMES_HOME", str(home))

    ref_runs = []
    _install_fake_llm(monkeypatch, ref_runs)

    facade = _facade()
    first = [{"role": "user", "content": "erstes Ziel"}]
    facade.create(messages=list(first), tools=[])
    assert len(ref_runs) == 1

    second = list(first) + [
        {"role": "assistant", "content": "Antwort"},
        {"role": "user", "content": "jetzt etwas anderes"},
    ]
    facade.create(messages=second, tools=[])

    assert len(ref_runs) == 2, "a genuine new user turn must refresh the advice"


def test_synthetic_row_predicate_matches_every_continuation_variant(monkeypatch, tmp_path):
    """Pin the predicate to the whole nudge class, not just the output-limit string."""
    from agent.moa_loop import _is_synthetic_user_row
    from agent.conversation_loop import (
        _LEGACY_LENGTH_CONTINUATION_NETWORK_STUB,
        _LENGTH_CONTINUATION_NETWORK_STUB,
        _LENGTH_CONTINUATION_OUTPUT_LIMIT,
    )

    for content in (
        _LENGTH_CONTINUATION_OUTPUT_LIMIT,
        _LENGTH_CONTINUATION_NETWORK_STUB,
        _LEGACY_LENGTH_CONTINUATION_NETWORK_STUB,
    ):
        assert _is_synthetic_user_row({"role": "user", "content": content}) is True, content

    assert _is_synthetic_user_row({"role": "user", "content": "echte Frage"}) is False
    assert _is_synthetic_user_row({"role": "assistant", "content": _LENGTH_CONTINUATION_OUTPUT_LIMIT}) is False
