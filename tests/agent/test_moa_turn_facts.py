from __future__ import annotations

from types import SimpleNamespace

import pytest


def _response(content="done", *, model="fake-model"):
    message = SimpleNamespace(content=content, tool_calls=[])
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=None, model=model)


def _write_moa_config(home, *, refs, aggregator_model="agg-model", fanout=None):
    home.mkdir()
    ref_yaml = "\n".join(
        f"        - provider: openrouter\n          model: {model}" for model in refs
    )
    fanout_yaml = f"\n      fanout: {fanout}" if fanout else ""
    (home / "config.yaml").write_text(
        f"""
moa:
  default_preset: review
  presets:
    review:{fanout_yaml}
      reference_models:
{ref_yaml if ref_yaml else '        []'}
      aggregator:
        provider: openrouter
        model: {aggregator_model}
""".strip(),
        encoding="utf-8",
    )


def _install_fake_moa(monkeypatch, tmp_path, *, refs, fail_refs=(), aggregator_raises_for=(), stream=False):
    home = tmp_path / ".hermes"
    _write_moa_config(home, refs=refs)
    monkeypatch.setenv("HERMES_HOME", str(home))
    calls = []
    fail_refs = set(fail_refs)
    aggregator_raises_for = set(aggregator_raises_for)

    def fake_call_llm(**kwargs):
        calls.append(kwargs)
        if kwargs["task"] == "moa_reference":
            model = kwargs["model"]
            if model in fail_refs:
                raise RuntimeError(f"ref failed: {model}")
            return _response(f"advice from {model}", model=model)

        joined_messages = "\n".join(
            str(message.get("content") or "") for message in kwargs.get("messages") or []
        )
        for marker in aggregator_raises_for:
            if marker in joined_messages:
                raise RuntimeError(f"aggregator failed for {marker}")
        if stream:
            return iter(["chunk-1", "chunk-2"])
        return _response("aggregator answer", model="agg-model")

    monkeypatch.setattr("agent.moa_loop.call_llm", fake_call_llm)
    from agent.moa_loop import MoAClient

    return MoAClient("review"), calls


def _bar_for(facts):
    from agent.route_depth_bar import format_route_depth_bar
    from agent.turn_receipt import TurnReceipt, apply_turn_facts, update_turn_receipt_from_result

    receipt = TurnReceipt.start(
        session_id="s",
        turn_id="t",
        provider="openrouter",
        model="agg-model",
        platform="cli",
    )
    update_turn_receipt_from_result(
        receipt,
        completed=True,
        failed=False,
        interrupted=False,
        api_calls=1,
        exit_reason="stop",
        messages=[],
    )
    apply_turn_facts(receipt, facts)
    return format_route_depth_bar(receipt)


def test_moa_client_reports_only_successful_references_and_completed_aggregator(monkeypatch, tmp_path) -> None:
    client, _calls = _install_fake_moa(
        monkeypatch,
        tmp_path,
        refs=["ref-a", "ref-b", "ref-c"],
        fail_refs={"ref-c"},
    )

    client.chat.completions.create(messages=[{"role": "user", "content": "q"}], tools=[])

    facts = client.coordination_turn_facts()
    assert facts["moa"]["reference_models"] == ["openrouter:ref-a", "openrouter:ref-b"]
    assert facts["moa"]["reference_count"] == 2
    assert facts["moa"]["aggregator_count"] == 1
    assert "MoA 2+1" in _bar_for(facts)


def test_moa_client_reports_zero_references_when_all_refs_fail_but_aggregator_succeeds(monkeypatch, tmp_path) -> None:
    client, _calls = _install_fake_moa(
        monkeypatch,
        tmp_path,
        refs=["ref-a", "ref-b"],
        fail_refs={"ref-a", "ref-b"},
    )

    client.chat.completions.create(messages=[{"role": "user", "content": "q"}], tools=[])

    facts = client.coordination_turn_facts()
    assert facts["moa"]["reference_models"] == []
    assert facts["moa"]["reference_count"] == 0
    assert facts["moa"]["aggregator_count"] == 1
    assert "MoA 0+1" in _bar_for(facts)


def test_moa_client_clears_stale_facts_when_aggregator_raises(monkeypatch, tmp_path) -> None:
    client, _calls = _install_fake_moa(
        monkeypatch,
        tmp_path,
        refs=["ref-a", "ref-b"],
        aggregator_raises_for={"fail-turn"},
    )
    client.chat.completions.create(messages=[{"role": "user", "content": "ok-turn"}], tools=[])
    assert "MoA 2+1" in _bar_for(client.coordination_turn_facts())

    with pytest.raises(RuntimeError, match="aggregator failed"):
        client.chat.completions.create(messages=[{"role": "user", "content": "fail-turn"}], tools=[])

    assert client.coordination_turn_facts() == {}
    assert "MoA" not in _bar_for(client.coordination_turn_facts())


def test_moa_streaming_facts_publish_only_after_successful_post_consume(monkeypatch, tmp_path) -> None:
    client, _calls = _install_fake_moa(
        monkeypatch,
        tmp_path,
        refs=["ref-a", "ref-b"],
        stream=True,
    )

    stream = client.chat.completions.create(
        messages=[{"role": "user", "content": "q"}],
        tools=[],
        stream=True,
    )

    assert list(stream) == ["chunk-1", "chunk-2"]
    assert client.coordination_turn_facts() == {}

    client.consume_and_save_trace("sess", aggregator_output_fallback="streamed answer")

    facts = client.coordination_turn_facts()
    assert facts["moa"]["reference_models"] == ["openrouter:ref-a", "openrouter:ref-b"]
    assert facts["moa"]["aggregator_count"] == 1
    assert "MoA 2+1" in _bar_for(facts)


class _FailingStream:
    def __iter__(self):
        raise RuntimeError("stream exploded")


def test_moa_failing_stream_without_post_consume_leaves_facts_empty(monkeypatch, tmp_path) -> None:
    home = tmp_path / ".hermes"
    _write_moa_config(home, refs=["ref-a"])
    monkeypatch.setenv("HERMES_HOME", str(home))

    def fake_call_llm(**kwargs):
        if kwargs["task"] == "moa_reference":
            return _response("advice", model=kwargs["model"])
        return _FailingStream()

    monkeypatch.setattr("agent.moa_loop.call_llm", fake_call_llm)
    from agent.moa_loop import MoAClient

    client = MoAClient("review")
    stream = client.chat.completions.create(
        messages=[{"role": "user", "content": "q"}],
        tools=[],
        stream=True,
    )

    with pytest.raises(RuntimeError, match="stream exploded"):
        list(stream)
    assert client.coordination_turn_facts() == {}


def test_moa_cache_hit_reuses_successful_labels_without_fabricating_failed_refs_or_leaking_failed_turn(monkeypatch, tmp_path) -> None:
    client, _calls = _install_fake_moa(
        monkeypatch,
        tmp_path,
        refs=["ref-a", "ref-b"],
        fail_refs={"ref-b"},
        aggregator_raises_for={"new-fail-turn"},
    )

    messages = [{"role": "user", "content": "same-turn"}]
    client.chat.completions.create(messages=messages, tools=[])
    first = client.coordination_turn_facts()
    assert first["moa"]["reference_models"] == ["openrouter:ref-a"]
    assert "MoA 1+1" in _bar_for(first)

    client.chat.completions.create(messages=messages, tools=[])
    cached = client.coordination_turn_facts()
    assert cached["moa"]["reference_models"] == ["openrouter:ref-a"]
    assert "openrouter:ref-b" not in cached["moa"]["reference_models"]
    assert "MoA 1+1" in _bar_for(cached)

    with pytest.raises(RuntimeError, match="aggregator failed"):
        client.chat.completions.create(
            messages=[{"role": "user", "content": "new-fail-turn"}],
            tools=[],
        )
    assert client.coordination_turn_facts() == {}
