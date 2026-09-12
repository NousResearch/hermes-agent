"""Adaptive reasoning effort (`effort: "auto"`).

`auto` is accepted wherever an effort level is configured, resolved to a concrete
ladder level per user turn from deterministic request-shape signals (no LLM
classifier), pinned for the turn's tool loop (changing reasoning config mid-loop
costs a cold prefix write on config-sensitive providers), and NEVER emitted on
the wire — transports only ever see a concrete level.
"""

import json

import pytest

from agent.reasoning_effort import (
    AUTO_EFFORT,
    resolve_auto_effort,
)
from hermes_constants import parse_reasoning_effort, resolve_reasoning_config


# ── Config acceptance ───────────────────────────────────────────────────────

def test_parse_reasoning_effort_accepts_auto():
    assert parse_reasoning_effort("auto") == {"enabled": True, "effort": "auto"}
    assert parse_reasoning_effort("Auto") == {"enabled": True, "effort": "auto"}


def test_auto_flows_through_resolve_reasoning_config():
    cfg = {"agent": {"reasoning_effort": "auto"}}
    assert resolve_reasoning_config(cfg, "glm-5.3") == {"enabled": True, "effort": "auto"}


def test_auto_not_added_to_valid_reasoning_efforts():
    # The public ladder stays wire-safe: only the seam may resolve auto, and it
    # must never leak into the tuple transports clamp against.
    from hermes_constants import VALID_REASONING_EFFORTS
    assert AUTO_EFFORT not in VALID_REASONING_EFFORTS
    assert AUTO_EFFORT == "auto"


# ── The resolver: deterministic signal → level mapping ─────────────────────

def test_short_simple_prompt_maps_low():
    level = resolve_auto_effort(user_chars=40, est_ctx_tokens=800, tool_results=0, turn_depth=1)
    assert level == "low"


def test_long_complex_prompt_maps_high():
    level = resolve_auto_effort(user_chars=4000, est_ctx_tokens=6000, tool_results=0, turn_depth=1)
    assert level == "high"


def test_mid_complexity_maps_medium():
    level = resolve_auto_effort(user_chars=600, est_ctx_tokens=2500, tool_results=0, turn_depth=1)
    assert level == "medium"


def test_deep_tool_loop_escalates():
    # A turn that has grown many tool results is doing agentic work: at least medium.
    level = resolve_auto_effort(user_chars=100, est_ctx_tokens=20000, tool_results=12, turn_depth=1)
    assert level in {"medium", "high"}


def test_resolver_is_pure_and_total():
    # Any non-negative signal combination yields a valid wire level, never auto.
    from hermes_constants import VALID_REASONING_EFFORTS
    for user_chars in (0, 10, 300, 2000, 9000):
        for est_ctx_tokens in (0, 1500, 30000, 200000):
            for tool_results in (0, 2, 9, 40):
                for turn_depth in (1, 3, 8):
                    level = resolve_auto_effort(
                        user_chars=user_chars, est_ctx_tokens=est_ctx_tokens,
                        tool_results=tool_results, turn_depth=turn_depth,
                    )
                    assert level in VALID_REASONING_EFFORTS, (user_chars, est_ctx_tokens, tool_results, turn_depth)


# ── The seam: resolve once per user turn, pin through the tool loop ─────────

class _StubAgent:
    model = "gpt-5"
    api_mode = "chat_completions"
    base_url = "https://api.example.com"
    provider = "custom"
    tools = []
    max_tokens = 4096
    _base_url_hostname = "api.example.com"
    _base_url_lower = "https://api.example.com"
    session_id = "test-session"

    def __init__(self, reasoning_config):
        self.reasoning_config = reasoning_config
        self.captured_configs = []
        self.captured_effort = None

    def __getattr__(self, name):
        # Catch-all for builder-probed attrs: anything not explicitly defined
        # reads as None (falsy, harmless). Called methods are all defined
        # explicitly above; a None slip would raise loudly, not silently pass.
        if name.startswith("__"):
            raise AttributeError(name)
        return None

    def _prepare_messages_for_non_vision_model(self, msgs):
        return msgs

    def _resolved_api_call_timeout(self):
        return 30

    def _max_tokens_param(self, *a, **k):
        return {}

    def _supports_reasoning_extra_body(self):
        return False

    def _is_qwen_portal(self):
        return False

    def _is_openrouter_url(self):
        return False

    def _github_models_reasoning_extra_body(self):
        return None

    def _qwen_meta(self):
        return None

    def _openrouter_preferences(self):
        return None

    def _lmstudio_reasoning_options_cached(self):
        return None

    def _qwen_prepare_chat_messages(self, msgs):
        return msgs

    def _qwen_prepare_chat_messages_inplace(self, msgs):
        return msgs

    def _prompt_cache_scope(self):
        return "test-scope"

    def _ephemeral_reasoning_off_flag(self):
        return False

    def _ollama_num_ctx(self):
        return None

    def _get_transport(self):
        return _RecordingTransport(self)


class _RecordingTransport:
    """Records the reasoning_config the seam handed the transport."""
    def __init__(self, agent):
        self._agent = agent

    def build_kwargs(self, model, messages, tools, reasoning_config=None, **kw):
        self._agent.captured_configs.append(reasoning_config)
        if isinstance(reasoning_config, dict):
            self._agent.captured_effort = reasoning_config.get("effort")
        return {"model": model, "messages": messages}


def _msg(role, content):
    return {"role": role, "content": content}


def test_seam_resolves_auto_to_concrete_level():
    from agent.chat_completion_helpers import build_api_kwargs

    agent = _StubAgent({"enabled": True, "effort": "auto"})
    messages = [_msg("user", "Explain " + "quantum tunneling " * 40)]
    kwargs = build_api_kwargs(agent, messages)
    wire_effort = kwargs.get("reasoning_effort") or (kwargs.get("extra_body") or {}).get("reasoning_effort")
    # Never auto on the wire; a concrete level or provider-default omission.
    assert wire_effort != "auto"
    if isinstance(wire_effort, str):
        from hermes_constants import VALID_REASONING_EFFORTS
        assert wire_effort in VALID_REASONING_EFFORTS


def test_seam_pins_level_across_tool_loop():
    """Second request in the SAME user turn must reuse the first resolution."""
    from agent.chat_completion_helpers import _auto_effort_for_request, _reset_auto_effort_pin

    agent = _StubAgent({"enabled": True, "effort": "auto"})
    # Trivial opening ask -> resolves low.
    user_turn = [_msg("user", "list the files")]
    first = _auto_effort_for_request(agent, user_turn)
    assert first == "low"
    # Tool round: same turn, messages grew into heavy tool work — WITHOUT the
    # pin this would re-resolve to high; the pin must hold it at low.
    grown = user_turn + [_msg("assistant", json.dumps([{"callee": "t", "arguments": {}}])),
                         _msg("tool", "x" * 5000)] * 7
    second = _auto_effort_for_request(agent, grown)
    assert second == first == "low"
    # New user turn re-resolves.
    new_turn = grown + [_msg("user", "short")]
    _reset_auto_effort_pin(agent)
    third = _auto_effort_for_request(agent, new_turn)
    assert isinstance(third, str)


def test_static_effort_bypasses_resolver():
    from agent.chat_completion_helpers import _auto_effort_for_request

    agent = _StubAgent({"enabled": True, "effort": "high"})
    assert _auto_effort_for_request(agent, [_msg("user", "hi")]) is None
