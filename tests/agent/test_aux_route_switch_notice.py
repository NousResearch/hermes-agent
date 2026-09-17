"""Auxiliary route-switch notices — the side-task half of "no silent fallback".

The main conversation loop already surfaces a provider/model switch through
``AIAgent._emit_pending_fallback_notice``. Auxiliary work (compression, titles,
vision, web extraction, session search) walks its own fallback chain and used to
report a switch only to the debug log.

Contract under test:

* an auxiliary call served by a fallback route reports that switch once through
  the sink the agent loop publishes for the turn;
* with no published sink nothing is emitted and nothing raises;
* the same switch inside one scope is reported once; a fresh scope reports it
  again, so a switch in a later turn is still visible;
* the successful response is still returned (the notice is additive).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import List

import pytest

from agent import auxiliary_client
from agent.aux_notices import (
    emit_aux_notice,
    reset_notice_sink,
    set_notice_sink,
)


def _fake_response(model: str = "gemma4:31b-cloud") -> SimpleNamespace:
    """A response shaped like the ones ``_validate_llm_response`` accepts."""
    message = SimpleNamespace(content="ok", tool_calls=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=None,
        model=model,
    )


class _FakeClient:
    """Minimal stand-in for a resolved OpenAI-compatible client.

    Carries ``_hermes_fallback_destination`` so ``_fallback_destination()`` takes
    its early-return path and no real provider resolution runs.
    """

    def __init__(self, provider: str, model: str, base_url: str) -> None:
        self.base_url = base_url
        self._hermes_fallback_destination = auxiliary_client._FallbackDestination(
            provider, base_url, "chat_completions", model
        )


# ---------------------------------------------------------------------------
# agent/aux_notices.py
# ---------------------------------------------------------------------------

def test_emit_aux_notice_is_a_noop_without_a_published_sink():
    # Nothing published (an auxiliary call outside any agent turn): must not
    # raise and must not invent a channel.
    emit_aux_notice("⚠ Auxiliary 'compression': deepseek unavailable — using custom/x.")


def test_emit_aux_notice_delivers_once_per_scope():
    seen: List[str] = []
    token = set_notice_sink(seen.append)
    try:
        emit_aux_notice("switch-A")
        emit_aux_notice("switch-A")  # duplicate inside the same scope
        emit_aux_notice("switch-B")
    finally:
        reset_notice_sink(token)

    assert seen == ["switch-A", "switch-B"]


def test_a_fresh_scope_reports_the_same_switch_again():
    first: List[str] = []
    token_a = set_notice_sink(first.append)
    try:
        emit_aux_notice("switch-A")
    finally:
        reset_notice_sink(token_a)

    second: List[str] = []
    token_b = set_notice_sink(second.append)
    try:
        emit_aux_notice("switch-A")
    finally:
        reset_notice_sink(token_b)

    assert first == ["switch-A"]
    assert second == ["switch-A"], "a later turn must still see its own switch"


def test_reset_restores_the_previous_scope():
    outer: List[str] = []
    inner: List[str] = []
    token_outer = set_notice_sink(outer.append)
    try:
        token_inner = set_notice_sink(inner.append)
        emit_aux_notice("inner-switch")
        reset_notice_sink(token_inner)
        emit_aux_notice("outer-switch")
    finally:
        reset_notice_sink(token_outer)

    assert inner == ["inner-switch"]
    assert outer == ["outer-switch"]


def test_a_failing_sink_never_propagates():
    def _boom(_message: str) -> None:
        raise RuntimeError("sink exploded")

    token = set_notice_sink(_boom)
    try:
        emit_aux_notice("switch-A")  # must be swallowed
    finally:
        reset_notice_sink(token)


# ---------------------------------------------------------------------------
# _notify_aux_route_switch — message construction
# ---------------------------------------------------------------------------

def _notify(monkeypatch, task, provider, model, main_provider):
    monkeypatch.setattr(
        auxiliary_client, "_read_main_provider", lambda: main_provider
    )
    seen: List[str] = []
    token = set_notice_sink(seen.append)
    try:
        auxiliary_client._notify_aux_route_switch(
            task,
            auxiliary_client._FallbackDestination(
                provider, "https://example.invalid/v1", "chat_completions", model
            ),
            f"fallback_providers[0]({provider})",
        )
    finally:
        reset_notice_sink(token)
    return seen


def test_route_switch_message_names_the_unavailable_provider_and_the_actual_one(
    monkeypatch,
):
    seen = _notify(
        monkeypatch, "compression", "custom", "gemma4:31b-cloud", "deepseek"
    )
    assert len(seen) == 1
    assert "compression" in seen[0]
    assert "deepseek" in seen[0]
    assert "custom/gemma4:31b-cloud" in seen[0]


def test_route_switch_message_reads_as_a_reroute_when_the_provider_is_unchanged(
    monkeypatch,
):
    seen = _notify(
        monkeypatch, "compression", "deepseek", "deepseek-flash", "deepseek"
    )
    assert len(seen) == 1
    assert "re-routed" in seen[0]
    assert "deepseek/deepseek-flash" in seen[0]


# ---------------------------------------------------------------------------
# Integration — a successful auxiliary fallback call reports AND returns
# ---------------------------------------------------------------------------

def test_successful_auxiliary_fallback_reports_the_switch_and_returns_the_response(
    monkeypatch,
):
    response = _fake_response()
    monkeypatch.setattr(
        auxiliary_client, "_relay_sync_completion", lambda *a, **k: response
    )
    monkeypatch.setattr(auxiliary_client, "_build_call_kwargs", lambda *a, **k: {})
    monkeypatch.setattr(
        auxiliary_client, "_read_main_provider", lambda: "deepseek"
    )

    seen: List[str] = []
    token = set_notice_sink(seen.append)
    try:
        returned = auxiliary_client._call_fallback_candidate_sync(
            _FakeClient("custom", "gemma4:31b-cloud", "http://127.0.0.1:11434/v1"),
            "gemma4:31b-cloud",
            "fallback_providers[1](custom)",
            task="compression",
            messages=[{"role": "user", "content": "summarize"}],
            temperature=None,
            max_tokens=None,
            tools=None,
            effective_timeout=30.0,
            effective_extra_body={},
            reasoning_config=None,
        )
    finally:
        reset_notice_sink(token)

    assert returned is response, "the notice must not change the return value"
    assert len(seen) == 1, f"expected exactly one notice, got {seen!r}"
    assert "custom/gemma4:31b-cloud" in seen[0]
    assert "deepseek" in seen[0]


def test_auxiliary_fallback_without_a_published_sink_still_returns(
    monkeypatch,
):
    response = _fake_response()
    monkeypatch.setattr(
        auxiliary_client, "_relay_sync_completion", lambda *a, **k: response
    )
    monkeypatch.setattr(auxiliary_client, "_build_call_kwargs", lambda *a, **k: {})

    returned = auxiliary_client._call_fallback_candidate_sync(
        _FakeClient("custom", "gemma4:31b-cloud", "http://127.0.0.1:11434/v1"),
        "gemma4:31b-cloud",
        "fallback_providers[1](custom)",
        task="compression",
        messages=[{"role": "user", "content": "summarize"}],
        temperature=None,
        max_tokens=None,
        tools=None,
        effective_timeout=30.0,
        effective_extra_body={},
        reasoning_config=None,
    )

    assert returned is response


# ---------------------------------------------------------------------------
# End-to-end — a real turn publishes the sink auxiliary work reports through
# ---------------------------------------------------------------------------

def _model_response(*, content, finish_reason, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def test_a_real_turn_routes_an_auxiliary_switch_to_the_user_visible_warning():
    """The agent loop publishes the sink for the whole turn.

    Drives ``AIAgent.run_conversation`` with a stubbed model that calls one
    tool. The tool handler performs the same auxiliary route switch
    ``_call_fallback_candidate_sync`` reports, and the notice must come out of
    the agent's user-visible warning surface — not just the debug log.
    """
    from unittest.mock import MagicMock, patch

    from run_agent import AIAgent

    warnings: List[str] = []

    def _status_callback(kind, message):
        if kind == "warn":
            warnings.append(str(message))

    tool_defs = [
        {
            "type": "function",
            "function": {
                "name": "terminal",
                "description": "test tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    tool_call = SimpleNamespace(
        id="call-1",
        type="function",
        function=SimpleNamespace(name="terminal", arguments="{}"),
    )

    def _fake_tool(*_args, **_kwargs):
        # Auxiliary work switching provider mid-turn, as compression would.
        auxiliary_client._notify_aux_route_switch(
            "compression",
            auxiliary_client._FallbackDestination(
                "custom",
                "http://127.0.0.1:11434/v1",
                "chat_completions",
                "gemma4:31b-cloud",
            ),
            "fallback_providers[1](custom)",
        )
        return "ok"

    with (
        patch("model_tools.get_tool_definitions", return_value=tool_defs),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("model_tools.handle_function_call", side_effect=_fake_tool),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://example.invalid/v1/",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            status_callback=_status_callback,
        )

        agent._cached_system_prompt = "You are helpful."
        agent._use_prompt_caching = False
        agent.compression_enabled = False
        agent.save_trajectories = False
        agent.valid_tool_names = {"terminal"}
        agent.client = MagicMock()
        agent.client.chat.completions.create.side_effect = [
            _model_response(
                content="Working on it.",
                finish_reason="tool_calls",
                tool_calls=[tool_call],
            ),
            _model_response(content="Done.", finish_reason="stop"),
        ]

        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("do the work")

    assert result["final_response"] == "Done."
    assert warnings, (
        "an auxiliary provider switch during a real turn must reach the "
        "agent's warning surface"
    )
    joined = "\n".join(warnings)
    assert "compression" in joined
    assert "custom/gemma4:31b-cloud" in joined
