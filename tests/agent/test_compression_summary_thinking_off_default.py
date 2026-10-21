"""Compression's aux summary call defaults to thinking-off (#135602).

With no ``auxiliary.compression.reasoning_effort`` configured, the summary call used to inherit
the main model's thinking ON. A reasoning main model then spent most of the output budget on
thinking before writing the summary, every summary landed as ``finish_reason=length``, and the
PARTIAL-summary guard routed each attempt through the fallback/abort machinery — compaction failed
deterministically once it triggered and the session ended in ``context_overflow``. These tests pin
the default (thinking-off for ``compression`` only) and that every explicit override still wins.
"""

import agent.auxiliary_client as aux


def _task_extra_body(monkeypatch, task, config):
    monkeypatch.setattr(aux, "_get_auxiliary_task_config", lambda _task: config)
    return aux._get_task_extra_body(task)


def test_compression_defaults_to_thinking_off(monkeypatch):
    extra_body = _task_extra_body(monkeypatch, "compression", {})

    assert extra_body["reasoning"] == {"enabled": False}


def test_explicit_reasoning_effort_beats_the_default(monkeypatch):
    extra_body = _task_extra_body(
        monkeypatch, "compression", {"reasoning_effort": "high"}
    )

    assert extra_body["reasoning"] == {"enabled": True, "effort": "high"}


def test_explicit_false_keeps_the_documented_disabled_shape(monkeypatch):
    extra_body = _task_extra_body(
        monkeypatch, "compression", {"reasoning_effort": False}
    )

    assert extra_body["reasoning"] == {"enabled": False}


def test_explicit_extra_body_reasoning_beats_the_default(monkeypatch):
    configured = {"enabled": True, "effort": "low"}
    extra_body = _task_extra_body(
        monkeypatch, "compression", {"extra_body": {"reasoning": configured}}
    )

    assert extra_body["reasoning"] == configured


def test_other_tasks_get_no_default(monkeypatch):
    for task in ("title_generation", "session_search", "vision"):
        assert _task_extra_body(monkeypatch, task, {}) == {}


def test_unset_compression_reaches_the_wire_as_a_disable(monkeypatch):
    """End to end: an unconfigured compression route must project a thinking-off wire shape —
    here DeepSeek's native ``thinking: disabled``, the same encoding an explicit ``none`` uses."""
    monkeypatch.setattr(aux, "_get_auxiliary_task_config", lambda _task: {})

    kwargs = aux._build_call_kwargs(
        provider="deepseek",
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": "hello"}],
        extra_body=aux._get_task_extra_body("compression"),
        task="compression",
    )

    assert kwargs["extra_body"]["thinking"] == {"type": "disabled"}
    assert "reasoning" not in kwargs["extra_body"]
