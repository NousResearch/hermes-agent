from __future__ import annotations

import copy
import logging
from types import SimpleNamespace

import pytest

from agent.anthropic_message_convert import convert_messages_to_anthropic
from agent.message_sanitization import stale_thinking_reaches_wire
from agent.model_metadata import estimate_tokens_rough

ANTHROPIC = ("anthropic", "https://api.anthropic.com")
NOUS = ("nous", "https://inference-api.nousresearch.com/v1/messages")
OPENROUTER = ("openrouter", "https://openrouter.ai/api/v1")
KIMI = ("anthropic", "https://api.kimi.com/coding")


def _signed_turn(question, answer, sig, *, thinking=None):
    thinking = thinking or f"thought-{sig}"
    return [
        {"role": "user", "content": question},
        {
            "role": "assistant",
            "content": answer,
            "reasoning": thinking,
            "reasoning_details": [{"type": "thinking", "thinking": thinking, "signature": sig}],
        },
    ]


def _carrier(thinking="secret chain", opaque=""):
    """Assistant row carrying the same thinking in every field the stores write."""
    signed = {"type": "thinking", "thinking": thinking, "signature": "sig_bad" + "s" * len(opaque)}
    redacted = {"type": "redacted_thinking", "data": "red_bad" + opaque}
    return [
        {"role": "user", "content": "Q"},
        {
            "role": "assistant",
            "content": "answer",
            "reasoning": thinking,
            "reasoning_content": thinking,
            "timestamp": 1234567890.0,
            "finish_reason": "tool_calls",
            "reasoning_details": [dict(signed), dict(redacted)],
            "anthropic_content_blocks": [
                dict(signed),
                {"type": "text", "text": "answer"},
                {"type": "tool_use", "id": "tool_1", "name": "search", "input": {"q": "x"}},
                dict(redacted),
            ],
            "tool_calls": [
                {"id": "tool_1", "type": "function",
                 "function": {"name": "search", "arguments": "{\"q\":\"x\"}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "tool_1", "content": "result"},
    ]


def _native_turn(agent, question, size, sig):
    """A turn stored by the real producer: Anthropic response -> transport -> assistant row."""
    from agent.chat_completion_helpers import build_assistant_message
    from agent.transports.anthropic import AnthropicTransport

    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking="x" * size, signature=sig),
            SimpleNamespace(type="text", text="A"),
        ],
        stop_reason="end_turn",
        stop_details=None,
    )
    normalized = AnthropicTransport().normalize_response(response)
    return [{"role": "user", "content": question},
            build_assistant_message(agent, normalized, normalized.finish_reason)]


def _session_db(tmp_path, *session_ids):
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    for session_id in session_ids:
        db.create_session(session_id, source="cli", model="claude-opus-4-6")
    return db


def _agent(db, session_id="s1", model="claude-opus-4-6", route=ANTHROPIC):
    from agent.agent_runtime_helpers import copy_reasoning_content_for_api
    from agent.message_sanitization import ReasoningReplayRoute
    from agent.context_compressor import ContextCompressor

    agent = SimpleNamespace(
        api_mode="anthropic_messages", provider=route[0], model=model, base_url=route[1],
        session_id=session_id, _session_db=db, _persist_disabled=False,
        _current_turn_timestamp=1.0, ephemeral_system_prompt="", prefill_messages=[],
        tools=[], _use_prompt_caching=False, _usage_anchor=None, verbose_logging=False,
        reasoning_callback=None, stream_delta_callback=None, _stream_callback=None, log_prefix="",
    )
    agent._needs_thinking_reasoning_pad = lambda: False
    agent._reasoning_replay_route = lambda: ReasoningReplayRoute(False, None)
    agent._copy_reasoning_content_for_api = (
        lambda source, target: copy_reasoning_content_for_api(agent, source, target)
    )
    agent._should_sanitize_tool_calls = lambda: False
    agent._sanitize_api_messages = lambda value: value
    agent._emit_warning = agent._vprint = lambda *args, **kwargs: None
    agent._drop_thinking_only_and_merge_users = lambda value, **kwargs: value
    agent._extract_reasoning = lambda message: getattr(message, "reasoning", None)
    agent._strip_think_blocks = lambda text: text
    agent.context_compressor = ContextCompressor(
        model, provider=route[0], base_url=route[1], api_mode=agent.api_mode,
        quiet_mode=True, config_context_length=200_000,
    )
    agent.context_compressor.bind_session_state(db, session_id)
    return agent


def _patch_assembly_loop(monkeypatch, selector=None):
    import agent.conversation_loop as loop

    monkeypatch.setattr(
        loop, "_apply_context_engine_selection",
        selector or (lambda agent, api, history, incoming, logger=None: api),
    )
    monkeypatch.setattr(loop, "_canonicalize_api_tool_calls", lambda messages: None)
    monkeypatch.setattr(
        loop, "_midturn_request_pressure_tokens", lambda agent, messages, system, approx: approx
    )
    monkeypatch.setattr(loop, "_pressure_with_real_floor", lambda compressor, value: value)


def _select_canonical_clone(agent, api, canonical, incoming, logger=None):
    return copy.deepcopy(canonical)


def _assemble_and_wire(agent, history):
    """Production request assembly, then the native converter: (assembled, wire messages)."""
    from agent.turn_request_assembly import assemble_api_request

    assembled = assemble_api_request(
        agent, messages=copy.deepcopy(history), current_turn_user_idx=len(history) - 1,
        _ext_prefetch_cache="", _plugin_user_context="", moa_config=None,
        active_system_prompt="", original_user_message=history[-1]["content"],
        pending_moa_prepared_request=None,
        request_logger=logging.getLogger("anthropic-preserved-thinking-test"),
    )
    _, wire = convert_messages_to_anthropic(
        copy.deepcopy(assembled.api_messages), base_url=agent.base_url, model=agent.model
    )
    return assembled, wire


def _thinking_blocks(wire):
    return [
        block
        for message in wire
        if message["role"] == "assistant" and isinstance(message["content"], list)
        for block in message["content"]
        if isinstance(block, dict) and block.get("type") in {"thinking", "redacted_thinking"}
    ]


def _assistant_text_blocks(wire):
    return [
        block
        for message in wire
        if message["role"] == "assistant" and isinstance(message["content"], list)
        for block in message["content"]
        if isinstance(block, dict) and block.get("type") == "text"
    ]


def _reject_signatures(agent, history):
    """Drive production signature-rejection recovery; the canonical history is never mutated."""
    from agent.error_classifier import FailoverReason
    from agent.turn_recovery import _recover_format_errors
    from agent.turn_retry_state import TurnRetryState

    before, request, retry = copy.deepcopy(history), copy.deepcopy(history), TurnRetryState()
    assert _recover_format_errors(
        agent, RuntimeError("Invalid signature in thinking block"),
        SimpleNamespace(reason=FailoverReason.thinking_signature), retry, history, request,
    )
    assert retry.thinking_sig_retry_attempted and history == before
    assert not _thinking_blocks(convert_messages_to_anthropic(request, model=agent.model)[1])


_PRESERVED = [
    ("claude-opus-4-4", False), ("claude-opus-4-5", True), ("claude-opus-4-6", True),
    ("claude-sonnet-4-5", False), ("claude-sonnet-4-6", True), ("claude-haiku-4-5", False),
    ("claude-opus-4-20250514", False), ("claude-sonnet-4-20250514", False),
    ("claude-opus-4-5-20251101", True), ("claude-opus-4-6-20250414", True),
    ("claude-sonnet-4-5-20250929", False), ("claude-opus-5", True), ("claude-sonnet-5-5", True),
    ("claude-fable-5-1", True), ("claude-mythos-5", True), ("claude-mythos-preview", True),
    # Future ids keep by default; known last-turn-only generations and non-Claude ids do not.
    ("claude-fable-5-2", True), ("claude-mythos-6", True), ("claude-newfamily-7", True),
    ("claude-haiku-5", False), ("claude-3-7-sonnet-20250219", False), ("claude-sonnet-4", False),
    ("hermes-4-405b", False),
]
# case -> (model, route, number of growing thinking blocks that must reach the wire)
_ACCOUNTING_CASES = {
    **{f"route:{m}": (m, ANTHROPIC, int(p)) for m, p in _PRESERVED},
    "route:nous": ("claude-opus-4-6", NOUS, 1),
    "route:openrouter": ("claude-opus-4-6", OPENROUTER, 0),
    "producer": ("claude-opus-4-6", ANTHROPIC, 2),
    "context_selection_clone": ("claude-opus-4-6", ANTHROPIC, 1),
    "usage_anchor": ("claude-opus-4-6", ANTHROPIC, 1),
    "carrier_dedupe": ("claude-opus-4-6", ANTHROPIC, 1),
    "opaque_bytes": ("claude-opus-4-6", ANTHROPIC, 0),
    "rejected": ("claude-opus-4-6", ANTHROPIC, 0),
    "rejected_unpersisted": ("claude-opus-4-6", ANTHROPIC, 0),
    "reasoning_only": ("claude-opus-4-6", ANTHROPIC, 0),
    "invalid_ordered_then_details": ("claude-opus-4-6", ANTHROPIC, 1),
    # Promoted final reply: canonical content stays empty, the api_content sidecar ships.
    "assistant_api_content": ("claude-opus-4-6", ANTHROPIC, 1),
}


@pytest.mark.parametrize("case", list(_ACCOUNTING_CASES))
def test_estimates_charge_exactly_the_thinking_the_wire_replays(tmp_path, monkeypatch, case):
    """Invariant: growing thinking text moves the wire, the preflight estimate, the compressor
    tail walk, and assembled request pressure by the same amount — the full block for every
    signed historical turn the route replays, nothing for latest-only routes, rejected,
    storage-only, or opaque (signature/redacted) bytes. A usage anchor still wins pressure."""
    import agent.turn_request_assembly as assembly
    from agent.turn_context import _preflight_request_tokens

    model, route, replayed_blocks = _ACCOUNTING_CASES[case]
    agent = _agent(_session_db(tmp_path, "s1"), model=model, route=route)
    agent._persist_disabled = case == "rejected_unpersisted"
    _patch_assembly_loop(
        monkeypatch, _select_canonical_clone if case == "context_selection_clone" else None
    )
    if case == "usage_anchor":
        monkeypatch.setattr(assembly, "anchored_context_tokens", lambda messages, anchor: 1234)
        agent._usage_anchor = object()

    def history(size):
        if case == "producer":
            body = _native_turn(agent, "Q1", size, "sig_1") + _native_turn(agent, "Q2", size, "sig_2")
        elif case == "carrier_dedupe":
            body = _carrier(thinking="x" * size)
        elif case == "opaque_bytes":
            body = _carrier(thinking="x" * 8000, opaque="r" * size)
        else:
            body = _signed_turn("Q1", "A1", "sig_1", thinking="x" * size)
            if case == "reasoning_only":  # rows written by another provider before a switch
                body[1] = {"role": "assistant", "content": "A1", "reasoning": "x" * size}
            elif case == "invalid_ordered_then_details":
                # Dataless redacted_thinking sanitizes away; the converter falls back to details.
                body[1]["anthropic_content_blocks"] = [{"type": "redacted_thinking"}]
            elif case == "assistant_api_content":
                body = _signed_turn("Q1", "", "sig_1", thinking="t" * 4000)
                body[1]["api_content"] = "x" * size
            body += _signed_turn("Q2", "A2", "sig_2")
        return body + [{"role": "user", "content": "continue"}]

    if case.startswith("rejected"):
        _reject_signatures(agent, history(1))

    def measure(size):
        canonical = history(size)
        preflight = _preflight_request_tokens(agent, copy.deepcopy(canonical), "")
        assembled, wire = _assemble_and_wire(agent, canonical)  # assembly sets the anchor flag
        replayed_text = (
            b.get("thinking", "") if b.get("type") == "thinking" else b.get("text", "")
            for b in _thinking_blocks(wire) + _assistant_text_blocks(wire)
        )
        return {
            "wire": sum(estimate_tokens_rough(text) for text in replayed_text),
            "preflight": preflight,
            "tail_walk": agent.context_compressor._walk_tail_budget(
                canonical, 0, 10**9, 0, cut_at_break=False
            )[1],
            "assembled": assembled.approx_tokens,
            "pressure": assembled.request_pressure_tokens,
        }

    small, large = measure(1), measure(8000)
    delta = {key: large[key] - small[key] for key in small}
    expected = replayed_blocks * (estimate_tokens_rough("x" * 8000) - estimate_tokens_rough("x"))
    if case == "usage_anchor":
        assert small["pressure"] == large["pressure"] == 1234
        assert agent._request_pressure_anchored is True
        delta["pressure"] = delta["assembled"]
    assert delta == dict.fromkeys(delta, expected)
    if case.startswith("route:"):
        assert stale_thinking_reaches_wire(agent.api_mode, route[0], model, route[1]) is bool(
            replayed_blocks
        )
        # A formerly-latest turn stays byte-stable when history keeps thinking, else loses it.
        prefix = history(1)[:-1]
        short = convert_messages_to_anthropic(prefix, base_url=route[1], model=model)[1]
        longer = convert_messages_to_anthropic(
            prefix + _signed_turn("Q3", "A3", "sig_3"), base_url=route[1], model=model
        )[1]
        assert longer[3] == short[3] if replayed_blocks else not _thinking_blocks([longer[3]])


@pytest.mark.parametrize(
    "boundary",
    ["same_agent", "rebuild", "session_switch", "compression_child", "context_selection",
     "carrier", "unfingerprintable", "kimi_route", "future_family"],
)
def test_rejected_thinking_never_returns_and_nothing_else_is_suppressed(
    tmp_path, monkeypatch, boundary
):
    """Invariant: once Anthropic rejects a signature, every later request in that conversation
    replays exactly the never-rejected signed thinking, whatever lifetime boundary it crosses;
    the rejection drops the stale usage anchor. Routes Anthropic does not sign for keep their
    own replay contract."""
    from agent.conversation_compression import _carry_session_state_to_child

    db = _session_db(tmp_path, "s1", "s2")
    agent = (
        _agent(db, route=KIMI, model="kimi-k2.5") if boundary == "kimi_route"
        else _agent(db, model="claude-fable-5-2") if boundary == "future_family"
        else _agent(db)
    )
    agent._usage_anchor = {"prompt_tokens": 999}
    _patch_assembly_loop(
        monkeypatch, _select_canonical_clone if boundary == "context_selection" else None
    )
    rejected = _signed_turn("Q1", "A1", "sig_rejected_1") + _signed_turn("Q2", "A2", "sig_rejected_2")
    if boundary == "unfingerprintable":
        rejected = [{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A1"}]
    elif boundary == "carrier":
        rejected = _carrier()
    _reject_signatures(agent, rejected)
    if boundary != "kimi_route":
        assert agent._usage_anchor is None
    reader = agent

    if boundary in {"rebuild", "context_selection", "carrier"}:
        reader = _agent(db)  # fresh process on the same session
    elif boundary == "session_switch":
        # s2 was rejected by an earlier process; the live agent then resumes s2.
        rejected = _signed_turn("Q1", "A1", "sig_rejected_s2")
        _reject_signatures(_agent(db, "s2"), rejected)
        agent.session_id = "s2"
    later = (
        rejected
        + _signed_turn("Q3", "A3", "sig_kept")
        + _signed_turn("Q4", "A4", "sig_new")
        + [{"role": "user", "content": "continue"}]
    )
    if boundary == "compression_child":
        # Rotation publishes the child with the session's initial model_config, then carries
        # per-session state; the child's retained tail still holds the rejected rows.
        db.publish_compression_child(
            parent_session_id="s1", child_session_id="s1-child", source="cli",
            model=agent.model, model_config={}, messages=later[2:], require_compression_lease=False,
        )
        agent.session_id = "s1-child"
        _carry_session_state_to_child(agent, "s1", None)
        reader = _agent(db, "s1-child")  # resume the child in a fresh process
        later = later[2:]

    _, wire = _assemble_and_wire(reader, later)
    replayed = ["sig_kept", "sig_new"]
    if boundary == "kimi_route":  # one-request repair only; Kimi replays its history as-is
        replayed = ["sig_rejected_1", "sig_rejected_2"] + replayed
    assert [block.get("signature") for block in _thinking_blocks(wire)] == replayed
    # The rejected carrier's visible answer and tool call still replay; only thinking goes.
    assert ("tool_use" in repr(wire)) is (boundary == "carrier")
    assert "secret chain" not in repr(wire)
