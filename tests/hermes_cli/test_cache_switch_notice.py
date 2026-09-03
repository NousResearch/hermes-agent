"""Unit + integration tests for the mid-session cache-rebuild notice.

Pins the gate and message-builder behaviour of
``hermes_cli.cache_switch_notice`` so a future refactor cannot silently
regress the cost signal.

Regression guard (PR #94753 review, P0): the real ``AIAgent`` stores
history in ``_session_messages`` / prompt in ``_cached_system_prompt``, and
``agent.switch_model()`` zeroes ``compressor.last_prompt_tokens`` and
clears the cached prompt. Tests therefore cover BOTH agent shapes and the
post-switch (zeroed) shape — a helper that only understands the CLI shape
or estimates after the switch must fail here.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hermes_cli.cache_switch_notice import (
    MIN_CONTEXT_TOKENS,
    build_cache_switch_notice,
    cache_switch_notice_enabled,
    cache_switch_notice_for_agent,
    estimate_context_tokens,
    same_model_reselect,
    snapshot_pre_switch_state,
)


class _FakeCompressor:
    def __init__(self, last_prompt_tokens: int):
        self.last_prompt_tokens = last_prompt_tokens


class _CLIShapeAgent:
    """The classic HermesCLI object shape: public attribute names."""

    def __init__(self, *, last_prompt_tokens=0, messages=None, system_prompt="", tools=None):
        self.context_compressor = (
            _FakeCompressor(last_prompt_tokens) if last_prompt_tokens is not None else None
        )
        self.conversation_history = list(messages or [])
        self.system_prompt = system_prompt
        self.tools = tools


class _AIAgentShape:
    """The real AIAgent shape: underscore-private attributes only.

    Mirrors what the reviewer probed on a live agent: no
    ``conversation_history``, no ``system_prompt``, history lives in
    ``_session_messages``.
    """

    def __init__(self, *, last_prompt_tokens=0, messages=None, system_prompt="", tools=None):
        self.context_compressor = (
            _FakeCompressor(last_prompt_tokens) if last_prompt_tokens is not None else None
        )
        self._session_messages = list(messages or [])
        self._cached_system_prompt = system_prompt
        if tools is not None:
            self._tool_definitions = tools


def _post_switch_ai_agent(tools=None):
    """An AIAgent-shaped object AFTER switch_model() ran.

    update_model() zeroed last_prompt_tokens and cleared the cached prompt —
    this is the state every wired surface is in when it builds its reply.
    """
    return _AIAgentShape(
        last_prompt_tokens=0,
        messages=[{"role": "user", "content": "hi"}],
        system_prompt="",
        tools=tools,
    )


# ---------------------------------------------------------------------------
# estimate_context_tokens — both shapes
# ---------------------------------------------------------------------------


def test_estimate_prefers_provider_reported_tokens():
    for agent in (
        _CLIShapeAgent(last_prompt_tokens=84_000),
        _AIAgentShape(last_prompt_tokens=84_000),
    ):
        assert estimate_context_tokens(agent) == 84_000


def test_estimate_clamps_compression_sentinel():
    # last_prompt_tokens parks at -1 right after a compression until the next
    # real API call reports usage — treat that as "unknown" (0).
    for agent in (
        _CLIShapeAgent(last_prompt_tokens=-1),
        _AIAgentShape(last_prompt_tokens=-1),
    ):
        assert estimate_context_tokens(agent) == 0


def test_estimate_returns_zero_for_none_agent():
    assert estimate_context_tokens(None) == 0


def test_estimate_reads_underscored_fields_on_real_agent_shape(monkeypatch):
    """P0 regression: AIAgent keeps history/prompt under private names."""
    monkeypatch.setattr(
        "agent.model_metadata.estimate_request_tokens_rough",
        lambda messages, system_prompt="", tools=None: (
            70_000 if messages and system_prompt else 0
        ),
    )
    agent = _AIAgentShape(
        last_prompt_tokens=0,
        messages=[{"role": "user", "content": "hi"}],
        system_prompt="You are Hermes.",
    )
    assert estimate_context_tokens(agent) == 70_000


def test_post_switch_agent_shape_does_not_look_like_empty_session(monkeypatch):
    """After switch_model() zeroes counters, the rough fallback must still
    count _session_messages content — not collapse to tools-only ~19k."""
    monkeypatch.setattr(
        "agent.model_metadata.estimate_request_tokens_rough",
        lambda messages, system_prompt="", tools=None: sum(
            len(str(m.get("content", ""))) // 4 for m in messages
        ),
    )
    agent = _post_switch_ai_agent()
    agent._session_messages = [
        {"role": "user", "content": "x" * 160_000},
        {"role": "assistant", "content": "y" * 40_000},
    ]
    # 50k tokens of conversation → well above threshold even with the
    # provider-reported path dead.
    assert estimate_context_tokens(agent) >= MIN_CONTEXT_TOKENS


# ---------------------------------------------------------------------------
# snapshot_pre_switch_state
# ---------------------------------------------------------------------------


def test_snapshot_returns_reported_tokens_before_switch():
    agent = _AIAgentShape(last_prompt_tokens=84_000)
    assert snapshot_pre_switch_state(agent) == 84_000


# ---------------------------------------------------------------------------
# build_cache_switch_notice
# ---------------------------------------------------------------------------


def test_build_silent_below_threshold():
    assert (
        build_cache_switch_notice(
            old_model_display="Grok 4.6",
            new_model_display="Grok 4.5",
            est_context_tokens=MIN_CONTEXT_TOKENS - 1,
        )
        is None
    )


def test_build_silent_on_reselect_flag():
    assert (
        build_cache_switch_notice(
            old_model_display="Grok 4.6",
            new_model_display="Grok 4.6",
            est_context_tokens=100_000,
            is_reselect=True,
        )
        is None
    )


def test_build_silent_on_empty_names():
    assert (
        build_cache_switch_notice(
            old_model_display="",
            new_model_display="Grok 4.5",
            est_context_tokens=100_000,
        )
        is None
    )


def test_build_emits_notice_and_revert_hint_above_threshold():
    notice = build_cache_switch_notice(
        old_model_display="Ox Alpha",
        new_model_display="Grok 4.5",
        est_context_tokens=84_000,
    )
    assert notice is not None
    lines = notice.splitlines()
    assert len(lines) == 2
    assert "Grok 4.5" in lines[0]
    assert "~84k" in lines[0]
    assert "uncached" in lines[0].lower() or "uncached" in lines[0]
    assert "Ox Alpha" in lines[1]
    assert "/model Ox Alpha" in lines[1]


def test_build_can_omit_revert_hint_for_once_switches():
    notice = build_cache_switch_notice(
        old_model_display="Ox Alpha",
        new_model_display="Grok 4.5",
        est_context_tokens=84_000,
        include_revert_hint=False,
    )
    assert notice is not None
    assert "/model" not in notice
    assert "~84k" in notice


def test_build_rounds_tokens_half_up():
    notice = build_cache_switch_notice(
        old_model_display="A", new_model_display="B", est_context_tokens=30_499
    )
    assert notice is not None and "~30k" in notice

    notice = build_cache_switch_notice(
        old_model_display="A", new_model_display="B", est_context_tokens=30_500
    )
    assert notice is not None and "~31k" in notice


# ---------------------------------------------------------------------------
# same_model_reselect — provider-aware identity (P2)
# ---------------------------------------------------------------------------


def test_reselect_same_model_same_provider():
    assert same_model_reselect(
        old_model="grok-4.6", old_provider="xai-oauth",
        new_model="grok-4.6", new_provider="xai-oauth",
    ) is True


def test_cross_provider_same_model_is_not_reselect():
    """grok-4.6 xAI-OAuth → grok-4.6 OpenRouter IS a real cache miss."""
    assert same_model_reselect(
        old_model="grok-4.6", old_provider="xai-oauth",
        new_model="grok-4.6", new_provider="openrouter",
    ) is False


def test_reselect_with_unknown_providers_falls_back_to_model_only():
    assert same_model_reselect(
        old_model="grok-4.6", old_provider="",
        new_model="grok-4.6", new_provider="",
    ) is True


def test_different_models_never_reselect():
    assert same_model_reselect(
        old_model="a", old_provider="p",
        new_model="b", new_provider="p",
    ) is False


# ---------------------------------------------------------------------------
# cache_switch_notice_enabled — toggle parsing
# ---------------------------------------------------------------------------


def test_enabled_defaults_true_on_config_error(monkeypatch):
    def _boom():
        raise RuntimeError("config unreadable")

    monkeypatch.setattr("hermes_cli.config.load_config_readonly", _boom)
    assert cache_switch_notice_enabled() is True


def test_enabled_respects_false(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"display": {"cache_switch_notice": False}},
    )
    assert cache_switch_notice_enabled() is False


def test_enabled_parses_string_false_as_off(monkeypatch):
    """bool("false") is True — YAML users may quote the value."""
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"display": {"cache_switch_notice": "false"}},
    )
    assert cache_switch_notice_enabled() is False


def test_enabled_parses_string_true_as_on(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"display": {"cache_switch_notice": "true"}},
    )
    assert cache_switch_notice_enabled() is True


# ---------------------------------------------------------------------------
# cache_switch_notice_for_agent — composition
# ---------------------------------------------------------------------------


def test_for_agent_respects_config_toggle_off(monkeypatch):
    agent = _AIAgentShape(last_prompt_tokens=100_000)
    monkeypatch.setattr(
        "hermes_cli.cache_switch_notice.cache_switch_notice_enabled",
        lambda: False,
    )
    assert (
        cache_switch_notice_for_agent(
            agent=agent,
            old_model_display="A",
            new_model_display="B",
        )
        is None
    )


def test_for_agent_emits_when_enabled_and_above_threshold():
    agent = _AIAgentShape(last_prompt_tokens=100_000)
    notice = cache_switch_notice_for_agent(
        agent=agent,
        old_model_display="A",
        new_model_display="B",
    )
    assert notice is not None
    assert "B" in notice
    assert "/model A" in notice


def test_for_agent_accepts_pre_captured_token_count_without_agent():
    """The production pattern: tokens captured BEFORE switch_model(), agent
    already swapped/evicted by the time the reply is built."""
    notice = cache_switch_notice_for_agent(
        agent=None,
        old_model_display="A",
        new_model_display="B",
        est_context_tokens=84_000,
    )
    assert notice is not None
    assert "~84k" in notice


def test_for_agent_silent_on_empty_session():
    agent = _AIAgentShape(last_prompt_tokens=0)
    notice = cache_switch_notice_for_agent(
        agent=agent,
        old_model_display="A",
        new_model_display="B",
    )
    assert notice is None


def test_for_agent_silent_on_cross_provider_same_model_is_wrong():
    """Cross-provider same-model must NOT be treated as reselect."""
    notice = cache_switch_notice_for_agent(
        agent=None,
        old_model_display="grok-4.6",
        new_model_display="grok-4.6",
        est_context_tokens=100_000,
        old_provider="xai-oauth",
        new_provider="openrouter",
    )
    assert notice is not None


# ---------------------------------------------------------------------------
# Integration: gateway apply-path shape (mirrors _finish_switch flow)
# ---------------------------------------------------------------------------


def test_gateway_flow_snapshot_before_zeroing(monkeypatch):
    """Simulate the exact _finish_switch sequence: capture pre-switch tokens,
    then run switch_model() (zeroes counters), then build the reply from the
    snapshot. The confirmation must contain ~84k."""
    agent = _AIAgentShape(
        last_prompt_tokens=84_000,
        messages=[{"role": "user", "content": "big"}],
        system_prompt="sys",
    )

    def _fake_switch_model():
        # Mirror context_compressor.update_model(): zero the live counters.
        agent.context_compressor.last_prompt_tokens = 0
        agent._cached_system_prompt = ""

    captured = snapshot_pre_switch_state(agent)
    _fake_switch_model()

    lines: list[str] = []
    notice = cache_switch_notice_for_agent(
        agent=agent,
        old_model_display="old-model",
        new_model_display="new-model",
        est_context_tokens=captured,
    )
    if notice:
        lines.extend(notice.splitlines())

    assert any("~84k" in line for line in lines)
    assert any("/model old-model" in line for line in lines)
