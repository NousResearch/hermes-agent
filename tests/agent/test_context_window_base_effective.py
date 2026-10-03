"""Base vs effective context window across a session boundary (regression for #9181).

The overflow-recovery path reduces the window for a session-scoped reason (an Anthropic
long-context-tier 429 caps the model at 200K "extra usage" for now) and separately installs a
provider-CONFIRMED real limit. Both went through the same mutable ``context_length``, so the
session-scoped reduction was indistinguishable from the model's declared window and survived
``/new`` / ``/reset``: a conversation started fresh after a tier 429 kept the reduced window and
compressed at the reduced trigger for the rest of the process.

These tests pin the RELATION the recovery path needs — a temporary reduction resolves to a
different effective number while the declared base is preserved and comes back at a session
boundary, and a provider-confirmed limit still becomes the base.
"""

from unittest.mock import MagicMock

from agent.context_compressor import ContextCompressor
from agent.context_engine import ContextEngine
from agent.turn_recovery import _cap_long_context_tier

DECLARED_BASE = 1_000_000
TIER_CAP = 200_000


def _agent_with_compressor() -> tuple[MagicMock, ContextCompressor]:
    """Agent wired only as far as the long-context-tier cap reads it."""
    compressor = ContextCompressor(
        model="claude-opus-4-6",
        provider="anthropic",
        base_url="https://api.anthropic.com",
        config_context_length=DECLARED_BASE,
        quiet_mode=True,
    )
    agent = MagicMock()
    agent.model = "claude-opus-4-6"
    agent.provider = "anthropic"
    agent.base_url = "https://api.anthropic.com"
    agent.api_key = "sk-test"
    agent.api_mode = "anthropic"
    agent.context_compressor = compressor
    return agent, compressor


def test_temporary_tier_reduction_does_not_survive_session_reset():
    """A tier 429 lowers the EFFECTIVE window and leaves the declared base alone, so
    /new and /reset (on_session_reset) restore the window the model actually declares."""
    agent, compressor = _agent_with_compressor()
    assert compressor.context_length == DECLARED_BASE

    _cap_long_context_tier(agent)

    # The recovery path needs the reduced window to do its job this session...
    assert compressor.context_length == TIER_CAP

    # ...but a fresh conversation must not inherit a session-scoped reduction.
    compressor.on_session_reset()
    assert compressor.context_length == DECLARED_BASE


def test_provider_confirmed_limit_still_becomes_the_base():
    """A limit the provider actually reported is real, not temporary: it stays the base after
    a session reset, so the guard above cannot be satisfied by ignoring confirmed limits."""
    agent, compressor = _agent_with_compressor()

    compressor.update_model(
        model=agent.model, context_length=TIER_CAP, base_url=agent.base_url,
        api_key=agent.api_key, provider=agent.provider, api_mode=agent.api_mode,
    )

    compressor.on_session_reset()
    assert compressor.context_length == TIER_CAP


class _FakePluginEngine(ContextEngine):
    """A plugin engine implementing only the required abstract methods, so it inherits the
    BASE ``reduce_context_window_temporarily`` / ``on_session_reset`` — the shape a plugin
    author gets unless they override the session boundary themselves."""

    def __init__(self, context_length: int) -> None:
        self.context_length = context_length
        self.threshold_percent = 0.75
        self.model = "plugin-model"
        self.base_url = ""
        self.api_key = ""
        self.provider = ""
        self.api_mode = ""

    @property
    def name(self) -> str:
        return "fake-plugin"

    def update_from_response(self, usage):  # unused by this regression
        pass

    def should_compress(self, prompt_tokens=None):
        return False

    def compress(self, messages, current_tokens=None, focus_topic=None, force=False, memory_context=""):
        return messages


def test_plugin_engine_inheriting_base_methods_does_not_leak_the_cap():
    """_cap_long_context_tier() runs for EVERY engine, plugin included. An engine inheriting
    the base methods must still honour the "for this session only" contract: the tier cap
    resolves while the gate is up, and /new / /reset puts the declared window back."""
    engine = _FakePluginEngine(context_length=DECLARED_BASE)
    agent = MagicMock()
    agent.context_compressor = engine

    _cap_long_context_tier(agent)
    assert engine.context_length == TIER_CAP

    engine.on_session_reset()
    assert engine.context_length == DECLARED_BASE


def test_plugin_engine_second_cap_does_not_clobber_the_declared_base():
    """Two tier 429s in one session must still restore the ORIGINAL declared window, not the
    already-reduced one: the first recorded value wins."""
    engine = _FakePluginEngine(context_length=DECLARED_BASE)

    engine.reduce_context_window_temporarily(500_000)
    engine.reduce_context_window_temporarily(TIER_CAP)
    assert engine.context_length == TIER_CAP

    engine.on_session_reset()
    assert engine.context_length == DECLARED_BASE