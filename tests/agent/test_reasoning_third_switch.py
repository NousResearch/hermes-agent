"""Audit for a THIRD reasoning self-switch in the turn path.

Two hidden self-switchers are already healed (see ``test_reasoning_self_heal.py``):
a thinking-only length truncation armed a one-shot reasoning-off override, and a
400 on an ENABLED reasoning config set a session-sticky drop. Both are gone; the
contract now is: ``agent.reasoning_config`` keeps the configured level on every
main-turn request.

This file is the audit for a THIRD one. The code-side answer it pins down is that
there is none left: the only remaining session-sticky reasoning flags are
``_reasoning_disable_rejected`` / ``_reasoning_floor_required``, and they can only
*raise* reasoning (drop a disable), never silence it. Every assignment to
``agent.reasoning_config`` in ``agent/`` re-resolves from config (whose global
``agent.reasoning_effort: max`` yields ``{"enabled": True, "effort": "max"}``),
so no turn can leave it disabled.

What the live gateway actually showed (2026-09-25, session 20260925_224939,
model ``glm-5.3-flash:cloud`` via local Ollama) is provider-side, not code-side:
after a long tool loop the assistant rows carry empty reasoning, and replaying
that exact context against the live route reproduces it *with the configured
``reasoning_effort: max`` still on the wire* — streaming and non-streaming, both
giving ``completion_tokens: 65`` with zero reasoning. Adding ``think: true``,
``reasoning: {"enabled": true}`` or ``options.think`` does not restore it either:
GLM on Ollama does adaptive thinking and skips the thinking channel on dense
tool-loop continuations. The agent cannot force it back with a wire field, and
no request-side code path is turning it off. The tests below guard that.
"""

from __future__ import annotations

import re
from pathlib import Path

from agent.chat_completion_helpers import _reasoning_config_for_wire
from agent.transports.chat_completions import ChatCompletionsTransport
from providers import get_provider_profile

_AGENT_DIR = Path(__file__).resolve().parents[2] / "agent"
_MODEL = "glm-5.3-flash:cloud"
_BASE_URL = "http://localhost:11434/v1"
_CONFIGURED = {"enabled": True, "effort": "max"}


class _Agent:
    """The attributes the reasoning wire path reads on a live main-turn agent."""

    def __init__(self, reasoning_config=None, *, provider="custom:local-ollama"):
        self.reasoning_config = dict(reasoning_config) if reasoning_config else None
        self.provider = provider
        self.model = _MODEL
        self.api_mode = "chat_completions"
        self.base_url = _BASE_URL
        self.tools = []
        self.request_overrides = None
        self.service_tier = None
        self._fast_until = 0.0
        # Session-sticky flags that DO exist after the two healed switches.
        self._ephemeral_reasoning_omit = False
        self._reasoning_disable_rejected = False
        self._reasoning_floor_required = False
        self._ollama_num_ctx: int | None = None
        self._wire_reasoning_config = None

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def _wire_reasoning_effort(reasoning_config) -> object:
    """What the custom profile actually puts on the wire for this config."""
    kwargs = ChatCompletionsTransport().build_kwargs(
        _MODEL, [{"role": "user", "content": "hi"}], tools=None,
        provider_profile=get_provider_profile("custom:local-ollama"),
        reasoning_config=reasoning_config, base_url=_BASE_URL,
    )
    return kwargs.get("reasoning_effort")


def _tool_loop_messages():
    """The shape of a mid-turn tool-loop request: an assistant tool call + its result."""
    return [
        {"role": "user", "content": "Investigate the reasoning drop."},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "terminal", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
    ]


def test_every_request_of_a_tool_loop_keeps_the_configured_effort():
    """Consecutive tool-loop requests all carry ``reasoning_effort=max``.

    This is the exact shape that empties reasoning in the live gateway. If a
    third self-switch existed, one of these later calls would go out without the
    configured level (or as ``none``); it must not.
    """
    agent = _Agent(_CONFIGURED)
    for call in range(1, 6):
        cfg = _reasoning_config_for_wire(agent)
        assert cfg == _CONFIGURED, f"call {call} dropped the configured reasoning config: {cfg!r}"
        assert _wire_reasoning_effort(cfg) == "max", f"call {call} wire effort != max"
        # The live config itself is never rewritten by building a request.
        assert agent.reasoning_config == _CONFIGURED


def test_sticky_error_flags_never_disable_an_enabled_config():
    """The only surviving session-sticky reasoning flags cannot mute thinking.

    ``_reasoning_disable_rejected`` / ``_reasoning_floor_required`` exist to drop a
    *disable* on routes that refuse one (i.e. they raise reasoning). With an enabled
    config they must be inert — for the whole session, not just one call.
    """
    agent = _Agent(_CONFIGURED)
    agent._reasoning_disable_rejected = True
    agent._reasoning_floor_required = True
    agent._disable_streaming = True
    for _ in range(3):
        cfg = _reasoning_config_for_wire(agent)
        assert cfg == _CONFIGURED, cfg
        assert _wire_reasoning_effort(cfg) == "max"


def test_the_one_shot_omit_never_turns_into_a_disable():
    """The healed one-shot omit (reasoning-effort 400) skips one call, then restores.

    It returns ``None`` (route default / field omitted) — which on this Ollama route
    still thinks — never a disable, and never sticks.
    """
    agent = _Agent(_CONFIGURED)
    agent._ephemeral_reasoning_omit = True
    assert _reasoning_config_for_wire(agent) is None, "the retry omits the fields"
    assert not agent._ephemeral_reasoning_omit, "must be consumed exactly once"
    assert _wire_reasoning_effort(_reasoning_config_for_wire(agent)) == "max"


_OBSOLETE_ATTRS = ("_ephemeral_reasoning_off", "_reasoning_effort_rejected")
_DISABLED_ASSIGNMENT = re.compile(
    r"(?:agent|self)\.reasoning_config\s*=\s*\{[^}]*"
    r"(?:\"enabled\"\s*:\s*False|\"effort\"\s*:\s*\"none\")"
)
_REASONING_CONFIG_ASSIGNMENT = re.compile(r"(?:agent|self)\.reasoning_config\s*=\s*(.+)$", re.MULTILINE)
_ALLOWED_ASSIGNMENT_RHS = ("resolve_reasoning_config(", "dict(saved_reasoning)")


def test_no_session_sticky_reasoning_switch_in_the_source():
    """Static guard: no ``agent/`` module re-introduces a reasoning self-switch.

    - the two obsolete attributes never come back,
    - no assignment writes a disabled dict straight onto the agent,
    - every assignment re-resolves from config (the only legitimate writers are the
      model-switch / fallback / config chokepoints, not the turn path).
    """
    for path in sorted(_AGENT_DIR.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for attr in _OBSOLETE_ATTRS:
            assert attr not in text, f"{path.relative_to(_AGENT_DIR.parent)}: obsolete {attr}"
        match = _DISABLED_ASSIGNMENT.search(text)
        assert match is None, f"{path.relative_to(_AGENT_DIR.parent)} writes a disabled reasoning_config: {match.group(0)!r}"
        for rhs in _REASONING_CONFIG_ASSIGNMENT.findall(text):
            stripped = rhs.strip().rstrip(".")
            assert any(allowed in stripped for allowed in _ALLOWED_ASSIGNMENT_RHS), (
                f"{path.relative_to(_AGENT_DIR.parent)} assigns reasoning_config from "
                f"an unaudited expression: {stripped!r}"
            )
