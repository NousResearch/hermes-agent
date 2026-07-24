"""every_n fanout cadence: advisors refresh every Nth tool iteration and
off-cadence iterations reuse the cached guidance from the last on-cadence run.

Redesigned from PR #63448's intent (issue #63393 — advisor fan-out multiplies
turn latency/cost by the tool-iteration count). Unlike the submitted shape
(which dropped references entirely on off-cadence iterations), off-cadence
iterations here still feed the aggregator the LAST advisor guidance via the
same cache-reuse mechanism the user_turn fanout uses.
"""

from types import SimpleNamespace


def _response(content="done", *, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=None, model="fake-model")


def _cadence_config(home, fanout="every_n:3"):
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
            return _response(f"advice #{len(ref_runs)}")
        return _response("acted")

    monkeypatch.setattr("agent.moa_loop.call_llm", fake_call_llm)


def _iteration_messages(base, iterations):
    """Yield message lists simulating a growing tool loop: the base user turn,
    then one new (assistant tool_call, tool result) pair per iteration."""
    msgs = list(base)
    yield list(msgs)
    for i in range(1, iterations):
        msgs = msgs + [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": f"c{i}", "function": {"name": "f", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": f"c{i}", "content": f"result {i}"},
        ]
        yield list(msgs)


def test_every_n_cadence_runs_references_every_nth_iteration(monkeypatch, tmp_path):
    """With every_n:3, references run on iterations 1 and 4 of a 6-iteration
    tool loop (1 on-cadence, then every 3rd), not on all 6."""
    home = tmp_path / ".hermes"
    _cadence_config(home, "every_n:3")
    monkeypatch.setenv("HERMES_HOME", str(home))

    ref_runs = []
    _install_fake_llm(monkeypatch, ref_runs)

    from agent.moa_loop import MoAChatCompletions

    events = []
    facade = MoAChatCompletions("review", reference_callback=lambda ev, **kw: events.append(ev))
    base = [{"role": "user", "content": "do the thing"}]
    for msgs in _iteration_messages(base, 6):
        facade.create(messages=msgs, tools=[{"type": "function"}])

    # 1 reference model × iterations {1, 4} on-cadence = 2 advisor runs.
    assert len(ref_runs) == 2
    # Display blocks only surface when references actually ran.
    assert events.count("moa.reference") == 2
    assert events.count("moa.aggregating") == 2


def test_every_n_off_cadence_iterations_reuse_cached_guidance(monkeypatch, tmp_path):
    """Off-cadence iterations must still give the aggregator the last
    on-cadence advisor guidance (cache reuse), not run advisor-less."""
    home = tmp_path / ".hermes"
    _cadence_config(home, "every_n:3")
    monkeypatch.setenv("HERMES_HOME", str(home))

    ref_runs = []
    _install_fake_llm(monkeypatch, ref_runs)

    from agent.moa_loop import MoAChatCompletions

    facade = MoAChatCompletions("review")
    base = [{"role": "user", "content": "task"}]
    prepared = [
        facade.create(messages=msgs, tools=[], _moa_prepare_only=True)
        for msgs in _iteration_messages(base, 3)
    ]

    # Iteration 1 ran the references; iterations 2-3 are off-cadence.
    assert len(ref_runs) == 1
    # Every iteration's aggregator request carries reference guidance...
    assert all(p["guidance"] for p in prepared)
    # ...and the off-cadence ones reuse iteration 1's exact advice text.
    assert "advice #1" in prepared[0]["guidance"]
    assert prepared[1]["guidance"] == prepared[0]["guidance"]
    assert prepared[2]["guidance"] == prepared[0]["guidance"]


def test_every_n_off_cadence_does_not_double_charge_usage(monkeypatch, tmp_path):
    """Cache-reuse iterations must not re-report advisor usage/cost."""
    home = tmp_path / ".hermes"
    _cadence_config(home, "every_n:2")
    monkeypatch.setenv("HERMES_HOME", str(home))

    ref_runs = []
    _install_fake_llm(monkeypatch, ref_runs)

    from agent.moa_loop import MoAChatCompletions

    facade = MoAChatCompletions("review")
    base = [{"role": "user", "content": "task"}]
    it = _iteration_messages(base, 2)

    facade.create(messages=next(it), tools=[])
    usage1, _cost1 = facade.consume_reference_usage()

    facade.create(messages=next(it), tools=[])  # off-cadence: reuse
    usage2, cost2 = facade.consume_reference_usage()

    assert len(ref_runs) == 1
    # The reuse iteration reports zero advisor usage and no cost.
    assert usage2.input_tokens == 0 and usage2.output_tokens == 0
    assert cost2 is None
    assert usage1 is not usage2


def test_every_n_counter_resets_on_new_user_turn(monkeypatch, tmp_path):
    """A new user message starts a new turn: iteration 1 is on-cadence again,
    regardless of where the previous turn's counter stood."""
    home = tmp_path / ".hermes"
    _cadence_config(home, "every_n:3")
    monkeypatch.setenv("HERMES_HOME", str(home))

    ref_runs = []
    _install_fake_llm(monkeypatch, ref_runs)

    from agent.moa_loop import MoAChatCompletions

    facade = MoAChatCompletions("review")
    turn1 = [{"role": "user", "content": "turn one"}]
    msgs: list = turn1
    for msgs in _iteration_messages(turn1, 2):
        facade.create(messages=msgs, tools=[])
    assert len(ref_runs) == 1  # iteration 2 was off-cadence

    # New user turn appended after the tool loop → counter resets, advisors
    # run immediately (fresh advice for the fresh request).
    turn2 = msgs + [
        {"role": "assistant", "content": "done with turn one"},
        {"role": "user", "content": "turn two"},
    ]
    facade.create(messages=turn2, tools=[])
    assert len(ref_runs) == 2


def test_every_n_redundant_create_does_not_consume_cadence_slot(monkeypatch, tmp_path):
    """A repeat create() with IDENTICAL state (e.g. a streaming retry) must not
    advance the cadence counter — only real state changes count."""
    home = tmp_path / ".hermes"
    _cadence_config(home, "every_n:2")
    monkeypatch.setenv("HERMES_HOME", str(home))

    ref_runs = []
    _install_fake_llm(monkeypatch, ref_runs)

    from agent.moa_loop import MoAChatCompletions

    facade = MoAChatCompletions("review")
    base = [{"role": "user", "content": "task"}]
    it = _iteration_messages(base, 3)

    first = next(it)
    facade.create(messages=first, tools=[])
    facade.create(messages=first, tools=[])  # retry: same state, no slot used
    assert facade._fanout_iteration_count == 1

    facade.create(messages=next(it), tools=[])  # iteration 2: off-cadence
    facade.create(messages=next(it), tools=[])  # iteration 3: on-cadence (every 2nd)
    assert len(ref_runs) == 2


def test_per_iteration_default_unchanged_by_cadence_state(monkeypatch, tmp_path):
    """Default fanout still re-runs references on every state change."""
    home = tmp_path / ".hermes"
    _cadence_config(home, "per_iteration")
    monkeypatch.setenv("HERMES_HOME", str(home))

    ref_runs = []
    _install_fake_llm(monkeypatch, ref_runs)

    from agent.moa_loop import MoAChatCompletions

    facade = MoAChatCompletions("review")
    base = [{"role": "user", "content": "task"}]
    for msgs in _iteration_messages(base, 3):
        facade.create(messages=msgs, tools=[])

    assert len(ref_runs) == 3
