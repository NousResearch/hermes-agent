"""Model-scoped opt-outs of reasoning replay (``model.reasoning_replay_off`` /
``model.reasoning_replay_filter``).

Stored reasoning is replayed to every chat-completions route by default, and a session whose
reasoning degenerates into a repetition loop is re-seeded by that replay on every turn (cron
36579fbca6c2, 2026-11-02: with the stored reasoning replayed the incident state looped 7/7,
without it 0/9 — the replay, not the context size, is the dominant trigger). The opt-outs key on
the MODEL, not the route, so a gateway serving a loop-prone model next to must-echo models keeps
every sibling's contract:

* ``model.reasoning_replay_off`` — the replayed text never rides the wire again; on a must-echo
  schema the field still rides blanked (dropping it is an HTTP 400 on tool-call turns).
* ``model.reasoning_replay_filter`` — the field and the echo contract stay intact, but text the
  repo's own repetition guard judges degenerate is replaced by the pad; legitimate reasoning
  passes byte-identical.

Mirrors tests/agent/test_reasoning_replay_routes.py (wire shape) and, at the wire level, the
incident mechanism behind tests/agent/test_stream_runaway_repetition.py (what got re-sent).
"""

from __future__ import annotations

from copy import deepcopy

import pytest

from agent.message_sanitization import (
    ReasoningReplayRoute, apply_reasoning_content_policy, model_replay_filter, model_replay_opt_out,
    reasoning_replay_route, reapply_reasoning_echo, stale_thinking_reaches_wire,
)
from agent.transports.chat_completions import ChatCompletionsTransport
from run_agent import AIAgent

_OPENCODE = ("custom", "https://opencode.ai/zen/go/v1", "beam-free")
_MIMO = ("xiaomi", "https://api.xiaomimimo.com/v1", "mimo-v2.5")

_TURN = {"role": "assistant", "content": "", "reasoning": "thought", "reasoning_content": "thought",
         "reasoning_details": [{"type": "reasoning.text", "text": "thought"}],
         "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "t", "arguments": "{}"}}]}
_BARE = {"role": "assistant", "content": "", "tool_calls": [{"id": "c2", "type": "function",
                                                            "function": {"name": "t", "arguments": "{}"}}]}

# The incident's degenerate reasoning shape (verbatim deliberation lines re-emitted; the live
# sample in the root task's reasoning_default_1.txt looped exactly two lines 261x).
_LOOP_SEED = ("De a user preferálja a magyar nyelvet a chat/jelentésben. De a skill F) szekció angol.\n"
              "A final response magyarul kell lennie. De a beam-free válasz angolul van. Fordítom magyarra.\n") * 40
# Legitimate reasoning of comparable length: every line distinct, no repeated 60-char window.
_LEGIT = "".join(
    f"Step {i}: the scheduler window closes at {i * 7 + 3} o'clock, so the wrap-up draft must "
    f"mention result row {i * 13} before the summary is translated for the user's report.\n"
    for i in range(12)
)


@pytest.fixture()
def model_config(monkeypatch):
    """Patch ``hermes_cli.config.load_config_readonly`` — the opt-outs read it lazily."""
    state = {"model": {}}
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: state)
    return state


def _wire(provider, base_url, model, model_config_state, rejected=()):
    """Route the source turns through the active policy, then through the transport strip."""
    model_config_state.setdefault("model", {})
    route = reasoning_replay_route("chat_completions", provider, model, base_url, rejected=rejected)
    out = []
    for source in (_TURN, _BARE):
        api_msg = deepcopy(source)
        apply_reasoning_content_policy(source, api_msg, route.pad, route.carriers, route.text_mode)
        out.append(api_msg)
    return route, ChatCompletionsTransport().convert_messages(
        out, model=model, base_url=base_url, provider=provider)


def test_opt_out_drops_replay_on_a_plain_route(model_config):
    """``reasoning_replay_off`` + a non-must-echo model: no reasoning key rides the wire."""
    model_config["model"]["reasoning_replay_off"] = ["beam-free"]
    route, (turn, bare) = _wire(*_OPENCODE, model_config)
    assert route == ReasoningReplayRoute(False, frozenset(), "blank")
    assert not {k for k in ("reasoning_content", "reasoning", "reasoning_details") if k in turn}
    assert not {k for k in ("reasoning_content", "reasoning", "reasoning_details") if k in bare}
    assert _TURN["reasoning_content"] == "thought" and _TURN["reasoning_details"]  # history untouched


def test_opt_out_keeps_the_must_echo_field_but_blanks_the_text(model_config):
    """``reasoning_replay_off`` + a must-echo model: the field still rides (HTTP 400 otherwise),
    as the pad — the degenerate text never does."""
    model_config["model"]["reasoning_replay_off"] = ["mimo"]
    route, (turn, bare) = _wire(*_MIMO, model_config)
    assert route == ReasoningReplayRoute(True, frozenset({"reasoning_content"}), "blank")
    assert turn["reasoning_content"] == " "
    assert "reasoning" not in turn and "reasoning_details" not in turn
    assert bare["reasoning_content"] == " "  # the tool-call echo contract stays satisfiable


def test_filter_keeps_legitimate_reasoning_byte_identical(model_config):
    """``reasoning_replay_filter`` + a must-echo model: clean reasoning replays verbatim, the
    details carrier is dropped (it would bypass the filter with the same text)."""
    model_config["model"]["reasoning_replay_filter"] = ["mimo"]
    source = deepcopy(_TURN)
    source["reasoning_content"] = source["reasoning"] = _LEGIT
    route = reasoning_replay_route("chat_completions", _MIMO[0], _MIMO[2], _MIMO[1])
    assert route == ReasoningReplayRoute(True, frozenset({"reasoning_content"}), "filtered")
    api_msg = deepcopy(source)
    apply_reasoning_content_policy(source, api_msg, route.pad, route.carriers, route.text_mode)
    assert api_msg["reasoning_content"] == _LEGIT
    assert "reasoning_details" not in api_msg


def test_filter_blanks_degenerate_reasoning(model_config):
    """``reasoning_replay_filter`` + a must-echo model: the loop seed becomes the pad, on the
    route's own judge (``agent/repetition_guard.is_runaway_repetition``)."""
    model_config["model"]["reasoning_replay_filter"] = ["mimo"]
    assert model_replay_filter("mimo-v2.5")
    source = {"role": "assistant", "content": "", "reasoning_content": _LOOP_SEED,
              "tool_calls": _TURN["tool_calls"]}
    route = reasoning_replay_route("chat_completions", _MIMO[0], _MIMO[2], _MIMO[1])
    api_msg = deepcopy(source)
    apply_reasoning_content_policy(source, api_msg, route.pad, route.carriers, route.text_mode)
    assert api_msg["reasoning_content"] == " "


def test_opt_out_wins_over_filter(model_config):
    model_config["model"]["reasoning_replay_off"] = ["beam-free"]
    model_config["model"]["reasoning_replay_filter"] = ["beam-free"]
    route = reasoning_replay_route("chat_completions", _OPENCODE[0], _OPENCODE[2], _OPENCODE[1])
    assert route.text_mode == "blank"
    _, (turn, _bare) = _wire(*_OPENCODE, model_config)
    assert "reasoning_content" not in turn


def test_no_flags_leave_every_route_unchanged(model_config):
    """Regression: without the opt-outs the echo tier still echoes and the default routes still
    replay (the mimo/Xiaomi 400 contract, the cache/billing families — all untouched)."""
    for provider, base_url, model, expected in (
        (*_MIMO, "thought"),
        ("deepseek", "https://api.deepseek.com/v1", "deepseek-v4", "thought"),
        (*_OPENCODE, "thought"),
        ("zai", "https://api.z.ai/api/paas/v4", "glm-5.3", "thought"),
        ("mistral", "https://api.mistral.ai/v1", "deepseek-r1", None),
    ):
        route, (turn, _bare) = _wire(provider, base_url, model, model_config)
        if expected is None:
            assert not {k for k in ("reasoning_content", "reasoning", "reasoning_details") if k in turn}
        else:
            assert turn.get("reasoning_content") == expected
        assert route.text_mode == "full"
    assert model_replay_opt_out("beam-free") is False and model_replay_filter("mimo-v2.5") is False


@pytest.mark.parametrize("provider,base_url,model,expected", [
    (*_OPENCODE, False),   # plain route, opted out: the wire is clean
    ("custom", "https://api.xiaomimimo.com/v1", "mimo-v2.5", True),  # no flags: replay (truthful charge)
])
def test_stale_thinking_predicate_matches_the_wire(model_config, provider, base_url, model, expected):
    """The compaction estimator and the tail-budget walks share this predicate; a blanked route
    must stop being charged as stale thinking, or every long opted-out session over-compacts."""
    if model == "beam-free":
        model_config["model"]["reasoning_replay_off"] = ["beam-free"]
    assert stale_thinking_reaches_wire("chat_completions", provider, model, base_url) is expected


def test_stale_thinking_mimo_opt_out(model_config):
    model_config["model"]["reasoning_replay_off"] = ["mimo"]
    assert stale_thinking_reaches_wire("chat_completions", *_MIMO) is False


def test_native_wires_keep_their_contract(model_config):
    """Outside chat_completions the native adapters own replay; the opt-outs are a no-op there."""
    model_config["model"]["reasoning_replay_off"] = ["mimo"]
    assert reasoning_replay_route("anthropic_messages", *_MIMO) == ReasoningReplayRoute(True, None)


def test_agent_route_picks_up_a_config_change(model_config):
    """The agent-side cache resolves the opt-outs into its key, so a config edit reaches the wire
    without a restart (and never disagrees with the estimator predicate)."""
    agent = object.__new__(AIAgent)
    agent.api_mode = "chat_completions"
    agent.provider, agent.base_url, agent.model = _OPENCODE
    agent._base_url_lower = _OPENCODE[1].lower()
    agent._reasoning_echo_flag = False
    agent.session_id = None

    assert agent._reasoning_replay_route().text_mode == "full"
    model_config["model"]["reasoning_replay_off"] = ["beam-free"]
    route = agent._reasoning_replay_route()
    assert route == ReasoningReplayRoute(False, frozenset(), "blank")
    # Idempotent across the reapply path (a fallback/rebuild reconciles to the same wire):
    # the turn with stored reasoning is stripped once, a repeat changes nothing.
    msgs = [deepcopy(_TURN), deepcopy(_BARE)]
    assert reapply_reasoning_echo(msgs, route.pad, route.carriers, route.text_mode) == 1
    assert reapply_reasoning_echo(msgs, route.pad, route.carriers, route.text_mode) == 0
    assert "reasoning_content" not in msgs[0]
