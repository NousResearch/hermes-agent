"""Contract tests for the ContextEngine lifecycle hooks (PR-1).

Covers:
- new optional hooks default to no-ops (None) and never raise
- ``capabilities()`` defaults declare nothing (legacy call pattern preserved)
- the built-in ``ContextCompressor`` inherits all defaults unchanged (T1)
- a minimal legacy engine (only the 4 abstract members) still works (T11)
- ``engine_hook`` fail-open wrapper semantics
"""

import logging

import pytest

from agent.context_engine import (
    ContextEngine,
    ContextEngineCapabilities,
    RequestContext,
    TurnInfo,
    engine_hook,
)


class MinimalLegacyEngine(ContextEngine):
    """Engine implementing ONLY the 4 abstract members (pre-lifecycle style)."""

    @property
    def name(self):
        return "minimal-legacy"

    def update_from_response(self, usage):
        self.last_prompt_tokens = usage.get("prompt_tokens", 0)

    def should_compress(self, prompt_tokens=None):
        return False

    def compress(self, messages, current_tokens=None, focus_topic=None):
        return messages


@pytest.fixture
def engine():
    return MinimalLegacyEngine()


# ── Default hook behavior ────────────────────────────────────────────────────


class TestLifecycleHookDefaults:
    def test_on_turn_complete_default_none(self, engine):
        turn = TurnInfo(session_id="s1")
        assert engine.on_turn_complete([{"role": "user", "content": "hi"}], turn) is None

    def test_prepare_request_messages_default_none(self, engine):
        ctx = RequestContext(budget_tokens=1000, model="m")
        msgs = [{"role": "user", "content": "hi"}]
        assert engine.prepare_request_messages(msgs, ctx) is None

    def test_on_pre_compress_default_none(self, engine):
        assert engine.on_pre_compress([{"role": "user", "content": "hi"}]) is None

    def test_carry_over_default_none(self, engine):
        assert engine.carry_over_new_session_context("old", "new") is None

    def test_default_hooks_do_not_mutate_messages(self, engine):
        msgs = [{"role": "user", "content": "hi"}]
        snapshot = [dict(m) for m in msgs]
        engine.on_turn_complete(msgs, TurnInfo(session_id="s"))
        engine.prepare_request_messages(msgs, RequestContext())
        engine.on_pre_compress(msgs)
        assert msgs == snapshot


# ── Capabilities ─────────────────────────────────────────────────────────────


class TestCapabilities:
    def test_default_declares_nothing(self, engine):
        caps = engine.capabilities()
        assert isinstance(caps, ContextEngineCapabilities)
        assert caps.observation is False
        assert caps.request_assembly is False
        assert caps.lossless_snapshot is False
        assert caps.tools is False

    def test_default_owns_compaction(self, engine):
        # Slot semantics: a selected engine owns compaction by default.
        assert engine.capabilities().owns_compaction is True

    def test_full_engine_can_declare(self):
        class FullEngine(MinimalLegacyEngine):
            def capabilities(self):
                return ContextEngineCapabilities(
                    observation=True,
                    request_assembly=True,
                    lossless_snapshot=True,
                    tools=True,
                )

        caps = FullEngine().capabilities()
        assert caps.observation and caps.request_assembly
        assert caps.lossless_snapshot and caps.tools


# ── Built-in compressor: zero behavior change (T1 contract level) ───────────


class TestBuiltinCompressorUnchanged:
    def _compressor(self):
        from agent.context_compressor import ContextCompressor

        return ContextCompressor(model="gpt-4o", api_key="", base_url="")

    def test_compressor_declares_nothing(self):
        caps = self._compressor().capabilities()
        assert caps.observation is False
        assert caps.request_assembly is False
        assert caps.lossless_snapshot is False

    def test_compressor_hooks_are_inherited_noops(self):
        comp = self._compressor()
        assert comp.on_turn_complete([], TurnInfo(session_id="s")) is None
        assert comp.prepare_request_messages([], RequestContext()) is None
        assert comp.on_pre_compress([]) is None

    def test_compressor_does_not_override_new_hooks(self):
        from agent.context_compressor import ContextCompressor

        for hook in (
            "on_turn_complete",
            "prepare_request_messages",
            "on_pre_compress",
            "capabilities",
        ):
            assert hook not in ContextCompressor.__dict__, (
                f"ContextCompressor must not override {hook} (zero-change red line)"
            )


# ── engine_hook fail-open wrapper ────────────────────────────────────────────


class TestEngineHookFailOpen:
    def test_returns_hook_value(self, engine):
        assert engine_hook(engine, "should_compress", 100) is False

    def test_exception_returns_default(self):
        class Broken(MinimalLegacyEngine):
            def on_turn_complete(self, messages, turn):
                raise RuntimeError("boom")

        result = engine_hook(
            Broken(), "on_turn_complete", [], TurnInfo(session_id="s"), default=None
        )
        assert result is None

    def test_exception_logs_warning(self, caplog):
        class Broken(MinimalLegacyEngine):
            def prepare_request_messages(self, messages, ctx):
                raise ValueError("nope")

        logger = logging.getLogger("test.engine_hook")
        with caplog.at_level(logging.WARNING, logger="test.engine_hook"):
            engine_hook(
                Broken(), "prepare_request_messages", [], RequestContext(),
                default=None, logger=logger,
            )
        assert any("fail-open" in r.message for r in caplog.records)

    def test_custom_default(self, engine):
        class Broken(MinimalLegacyEngine):
            def should_compress(self, prompt_tokens=None):
                raise RuntimeError("boom")

        assert engine_hook(Broken(), "should_compress", 1, default=False) is False


# ── Dataclass shapes ─────────────────────────────────────────────────────────


class TestDataclasses:
    def test_turn_info_defaults(self):
        t = TurnInfo(session_id="s")
        assert t.turn_index is None
        assert t.usage is None
        assert t.compressed_during_turn is False
        assert t.interrupted is False
        assert t.completed is True

    def test_request_context_defaults(self):
        c = RequestContext()
        assert c.incoming_message is None
        assert c.budget_tokens == 0
        assert c.model is None
        assert c.prefetch_context is None
