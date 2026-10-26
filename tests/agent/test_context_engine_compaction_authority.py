"""Context-engine compaction authority under ``compression.checkpoint_required``.

``compress()`` is the one compaction verb the host dispatches behind ``on_pre_compress()``;
an engine that compacts from ``on_turn_complete()`` or its own scheduler never crosses
that seam. These tests pin how the host tells the two apart
(``engine_compacts_outside_compress``) and where it acts on the verdict.
"""

from types import SimpleNamespace


def _engine_stubs():
    """Built against the CURRENTLY imported ContextEngine: sibling suites purge ``agent.*``
    from sys.modules, and stubs bound to a stale ABC would fail isinstance for the wrong reason."""
    from agent.context_engine import ContextEngine

    class _CompressOnlyEngine(ContextEngine):
        """The real third-party shape: should_compress() + compress() only."""

        @property
        def name(self) -> str:
            return "compress-only"

        def update_from_response(self, usage):
            return None

        def should_compress(self, prompt_tokens=None):
            return False

        def compress(self, messages, current_tokens=None, focus_topic=None, force=False, memory_context=""):
            return messages

    class _TurnCompleteEngine(_CompressOnlyEngine):
        """Also takes the post-turn hook, which no checkpoint precedes."""

        def on_turn_complete(self, messages, usage=None, **kwargs):
            return None

    return _CompressOnlyEngine, _TurnCompleteEngine


def test_engine_declaration_defaults_to_undeclared():
    """Tri-state, inheriting ``None``: False would make every existing engine an implicit
    waiver, True would refuse engines the host can prove safe."""
    from agent.context_engine import ContextEngine

    compress_only, _turn_complete = _engine_stubs()
    assert ContextEngine.compacts_outside_compress is None
    assert compress_only().compacts_outside_compress is None


def test_undeclared_engine_overriding_on_turn_complete_is_refused():
    from agent.context_engine import engine_compacts_outside_compress

    _compress_only, turn_complete = _engine_stubs()
    unsafe, reason = engine_compacts_outside_compress(turn_complete())
    assert unsafe is True
    assert "on_turn_complete" in reason


def test_compress_only_engine_resolves_safe():
    """Also with the hooks that do not count: ``prune_tool_results_only`` (gated at its call
    site), ``select_context`` (request-only), ``on_session_end`` (return value discarded)."""
    from agent.context_engine import engine_compacts_outside_compress

    compress_only, _turn_complete = _engine_stubs()
    assert engine_compacts_outside_compress(compress_only())[0] is False

    class _GatedHookEngine(compress_only):
        def prune_tool_results_only(self, messages, current_tokens=None):
            return messages, 0

        def select_context(self, request_messages, **kwargs):
            return request_messages

        def on_session_end(self, session_id, messages):
            return None

    assert engine_compacts_outside_compress(_GatedHookEngine())[0] is False
    # No engine installed is not an engine that compacts.
    assert engine_compacts_outside_compress(None)[0] is False


def test_non_context_engine_object_is_refused():
    """The directory loader has no isinstance gate, so a foreign object is refused rather
    than hook-inferred."""
    from agent.context_engine import ContextEngine, engine_compacts_outside_compress

    unsafe, reason = engine_compacts_outside_compress(SimpleNamespace())
    assert unsafe is True
    assert "ContextEngine" in reason

    # A duck borrowing the ABC's default hooks matches every __func__ identity, so hook
    # inference alone would clear it; the isinstance rule must not.
    class _BorrowedDefaultsDuck:
        name = "duck"
        on_turn_complete = ContextEngine.on_turn_complete

        def should_compress(self, prompt_tokens=None):
            return False

        def compress(self, messages, **kwargs):
            return messages

    assert engine_compacts_outside_compress(_BorrowedDefaultsDuck())[0] is True


def test_explicit_declaration_overrides_inference():
    """Both directions: a pure observer keeps ``on_turn_complete``; an engine with its own
    scheduler (invisible to inference) declares True."""
    from agent.context_engine import engine_compacts_outside_compress

    compress_only, turn_complete = _engine_stubs()

    class _DeclaredObserver(turn_complete):
        compacts_outside_compress = False

    class _DeclaredScheduler(compress_only):
        compacts_outside_compress = True

    assert engine_compacts_outside_compress(_DeclaredObserver())[0] is False
    assert engine_compacts_outside_compress(_DeclaredScheduler())[0] is True

    # Truthy-but-not-True is not a declaration (MagicMock auto-attributes).
    class _AutoAttributeEngine(turn_complete):
        compacts_outside_compress = 1

    assert engine_compacts_outside_compress(_AutoAttributeEngine())[0] is True


# --- Init refuse at the engine slot (agent_init._build_context_engine) ---------------


def _init_agent_with_engine(engine, *, checkpoint_required: bool):
    """A real AIAgent init with ``engine`` in the slot (harness of test_plugin_context_engine_init)."""
    from unittest.mock import patch

    cfg = {"context": {"engine": "stub"}, "compression": {"checkpoint_required": checkpoint_required}, "agent": {}}
    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("plugins.context_engine.load_context_engine", return_value=engine),
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        from run_agent import AIAgent

        return AIAgent(
            api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1",
            quiet_mode=True, skip_context_files=True, skip_memory=True,
        )


def test_refuse_covers_uninterceptable_engines_and_permits_the_rest():
    """A waiver, a gate that is off and an empty slot stay permitted."""
    import pytest

    from agent.conversation_compression_checkpoint import _refuse_checkpoint_required_on_plugin_context_engine as refuse

    compress_only, turn_complete = _engine_stubs()
    # The resolver's reason survives into the operator-facing text.
    with pytest.raises(RuntimeError, match="BLOCKED_MISSING_PREREQUISITE.*on_turn_complete"):
        refuse(True, turn_complete())

    class _DeclaredObserver(turn_complete):
        compacts_outside_compress = False

    refuse(True, _DeclaredObserver())
    refuse(True, compress_only())
    refuse(True, None)
    refuse(False, turn_complete())


def test_init_refuses_an_uninterceptable_engine_before_the_slot_is_assigned():
    """Only a real init observes the ordering: the refuse reads the local config value, the
    agent attribute is assigned after the engine slot. Negative probe (by hand, reverted):
    reading ``agent.compression_checkpoint_required`` there fails only this test."""
    import pytest

    _compress_only, turn_complete = _engine_stubs()
    with pytest.raises(RuntimeError, match="BLOCKED_MISSING_PREREQUISITE"):
        _init_agent_with_engine(turn_complete(), checkpoint_required=True)
    # Gate off: the same engine initializes (control).
    engine = turn_complete()
    assert _init_agent_with_engine(engine, checkpoint_required=False).context_compressor is engine


def test_compress_only_engine_still_initializes_with_gate_armed():
    """The compress()-only third-party shape (lcm) keeps initializing under the gate."""
    compress_only, _turn_complete = _engine_stubs()
    engine = compress_only()
    agent = _init_agent_with_engine(engine, checkpoint_required=True)

    assert agent.context_compressor is engine
    assert agent.compression_checkpoint_required is True
    # Fully configured, not merely unrefused (#9071 ordering).
    assert engine.context_length == 204_800


def test_refuse_message_survives_a_hostile_name_property():
    """The diagnosis names the class: ``engine.name`` is an abstract property whose raising
    getter would replace it."""
    import pytest

    from agent.conversation_compression_checkpoint import _refuse_checkpoint_required_on_plugin_context_engine as refuse

    _compress_only, turn_complete = _engine_stubs()

    class _HostileNameEngine(turn_complete):
        @property
        def name(self) -> str:
            raise RuntimeError("name getter exploded")

    with pytest.raises(RuntimeError, match="BLOCKED_MISSING_PREREQUISITE") as excinfo:
        refuse(True, _HostileNameEngine())
    assert "_HostileNameEngine" in str(excinfo.value)
    assert "name getter exploded" not in str(excinfo.value)
