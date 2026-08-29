"""Host-side contract tests for the opt-in pre-compress checkpoint API (v2).

The contract has three parts:
- providers opt in by advertising ``pre_compress_checkpoint_api_version = 2``
  (v1 is the implicit historical best-effort contract with raw messages);
- ``MemoryManager`` exposes capability probing and a ``require_checkpoint``
  mode whose failure must propagate instead of being swallowed;
- the compression host normalizes messages to direct user/assistant evidence
  before handing them to v2+ providers.
"""

import pytest

from agent.conversation_compression import (
    CompressionCheckpointUnavailable,
    _checkpoint_blocked,
    _direct_messages_for_pre_compress_memory,
)
from agent.context_compressor import COMPRESSED_SUMMARY_METADATA_KEY
from agent.memory_manager import MemoryManager
from agent.memory_provider import (
    PRE_COMPRESS_CHECKPOINT_API_VERSION,
    MemoryProvider,
)


class _BaseStubProvider(MemoryProvider):
    def __init__(self, name="stub"):
        self._name = name
        self.pre_compress_calls = []

    @property
    def name(self):
        return self._name

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        return None

    def get_tool_schemas(self):
        return []

    def on_pre_compress(self, messages):
        self.pre_compress_calls.append(messages)
        return f"{self._name} context"


class _CheckpointProvider(_BaseStubProvider):
    pre_compress_checkpoint_api_version = PRE_COMPRESS_CHECKPOINT_API_VERSION


class _FailingCheckpointProvider(_CheckpointProvider):
    def on_pre_compress(self, messages):
        raise RuntimeError("durable store unreachable")


class _FailingLegacyProvider(_BaseStubProvider):
    def on_pre_compress(self, messages):
        raise RuntimeError("legacy best-effort failure")


def test_provider_base_class_defaults_to_implicit_historical_api_version_one():
    assert MemoryProvider.pre_compress_checkpoint_api_version == 1
    assert PRE_COMPRESS_CHECKPOINT_API_VERSION == 2


def test_v1_providers_receive_raw_messages_and_v2_receive_evidence():
    """The historical (v1) contract is untouched: raw message list.

    Only providers that opted into checkpoint API v2 receive the
    host-normalized evidence handoff.
    """
    manager = MemoryManager()
    legacy = _BaseStubProvider("legacy")
    manager.add_provider(legacy)
    raw = [
        {"role": "user", "content": "evidence"},
        {"role": "tool", "content": "tool output", "tool_call_id": "t1"},
    ]
    evidence = [{"role": "user", "content": "evidence"}]

    manager.on_pre_compress(raw, evidence_messages=evidence)
    assert legacy.pre_compress_calls == [raw]

    durable_manager = MemoryManager()
    durable = _CheckpointProvider("durable")
    durable_manager.add_provider(durable)
    durable_manager.on_pre_compress(raw, evidence_messages=evidence)
    assert durable.pre_compress_calls == [evidence]


def test_direct_messages_filter_keeps_only_direct_source_evidence():
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "durable user decision"},
        {"role": "assistant", "content": "direct assistant answer"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "t1"}]},
        {"role": "tool", "content": "tool output", "tool_call_id": "t1"},
        {
            "role": "assistant",
            "content": "previous compaction summary",
            COMPRESSED_SUMMARY_METADATA_KEY: True,
        },
        "not-a-dict",
    ]

    direct = _direct_messages_for_pre_compress_memory(messages)

    assert [m["content"] for m in direct] == [
        "durable user decision",
        "direct assistant answer",
    ]


def test_direct_messages_filter_keeps_prose_of_tool_call_messages():
    """Assistant prose next to tool_calls is evidence; the payload is not."""
    messages = [
        {"role": "user", "content": "please scan the network"},
        {
            "role": "assistant",
            "content": "Scanning now — the last sweep found 26 hosts.",
            "tool_calls": [{"id": "t1", "function": {"name": "terminal"}}],
        },
        {"role": "assistant", "content": "   ", "tool_calls": [{"id": "t2"}]},
    ]

    direct = _direct_messages_for_pre_compress_memory(messages)

    assert [m["content"] for m in direct] == [
        "please scan the network",
        "Scanning now — the last sweep found 26 hosts.",
    ]
    assert all("tool_calls" not in m for m in direct)
    # The original message list is not mutated.
    assert messages[1]["tool_calls"]


def test_manager_advertises_checkpoint_capability_only_with_capable_provider():
    # The host allows one external provider per manager, so capability is
    # probed on two separate managers.
    legacy_manager = MemoryManager()
    legacy_manager.add_provider(_BaseStubProvider("legacy"))
    assert legacy_manager.supports_pre_compress_checkpoint(
        PRE_COMPRESS_CHECKPOINT_API_VERSION
    ) is False

    durable_manager = MemoryManager()
    durable_manager.add_provider(_CheckpointProvider("durable"))
    assert durable_manager.supports_pre_compress_checkpoint(
        PRE_COMPRESS_CHECKPOINT_API_VERSION
    ) is True


def test_manager_require_checkpoint_raises_without_capable_provider():
    manager = MemoryManager()
    manager.add_provider(_BaseStubProvider("legacy"))

    with pytest.raises(RuntimeError, match="pre-compress checkpoint"):
        manager.on_pre_compress(
            [{"role": "user", "content": "evidence"}],
            require_checkpoint=True,
            checkpoint_api_version=PRE_COMPRESS_CHECKPOINT_API_VERSION,
        )


def test_manager_require_checkpoint_propagates_checkpoint_provider_failure():
    manager = MemoryManager()
    manager.add_provider(_FailingCheckpointProvider("durable"))

    with pytest.raises(RuntimeError, match="durable store unreachable"):
        manager.on_pre_compress(
            [{"role": "user", "content": "evidence"}],
            require_checkpoint=True,
            checkpoint_api_version=PRE_COMPRESS_CHECKPOINT_API_VERSION,
        )


def test_manager_require_checkpoint_succeeds_and_returns_provider_context():
    manager = MemoryManager()
    durable = _CheckpointProvider("durable")
    manager.add_provider(durable)

    combined = manager.on_pre_compress(
        [{"role": "user", "content": "evidence"}],
        require_checkpoint=True,
        checkpoint_api_version=PRE_COMPRESS_CHECKPOINT_API_VERSION,
    )

    assert "durable context" in combined
    assert durable.pre_compress_calls


def test_manager_best_effort_mode_keeps_historical_swallow_semantics():
    manager = MemoryManager()
    manager.add_provider(_FailingLegacyProvider("legacy"))

    combined = manager.on_pre_compress([{"role": "user", "content": "evidence"}])

    assert combined == ""


def test_checkpoint_blocked_error_is_prefixed_and_typed():
    error = _checkpoint_blocked("no active provider")
    assert isinstance(error, CompressionCheckpointUnavailable)
    assert str(error).startswith("BLOCKED_MISSING_PREREQUISITE:")
    assert "no active provider" in str(error)


def test_compressed_summary_marker_survives_restart_via_resume_history(tmp_path):
    """The persistent marker reaches the resumed model history — and only it.

    ``get_messages_as_conversation`` keeps its existing marker-free contract;
    the resume path carries ``_compressed_summary`` so checkpoint providers
    keep excluding derivative summaries after a process restart.
    """
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "state.db")
    db.create_session("s1", source="cli")
    db.append_message("s1", "user", "durable user evidence")
    db.append_message(
        "s1", "assistant", "derivative summary", _compressed_summary=True
    )

    reopened = SessionDB(tmp_path / "state.db")
    model_history, _display = reopened.get_resume_conversations("s1")
    by_content = {m.get("content"): m for m in model_history}
    assert by_content["derivative summary"].get("_compressed_summary") is True
    assert "_compressed_summary" not in by_content["durable user evidence"]

    plain = reopened.get_messages_as_conversation("s1")
    assert all("_compressed_summary" not in m for m in plain)


def test_compressed_summary_column_is_added_to_legacy_databases(tmp_path):
    """Pre-upgrade databases gain the marker column via declarative reconcile.

    ``_init_schema()`` diffs live columns against SCHEMA_SQL on every
    writable open and ADDs whatever is missing, so a database created
    before this feature must accept marker writes after a plain reopen.
    """
    import sqlite3

    from hermes_state import SessionDB

    db_path = tmp_path / "state.db"
    SessionDB(db_path)

    # Simulate a pre-upgrade database: the marker column does not exist.
    conn = sqlite3.connect(db_path)
    conn.execute("ALTER TABLE messages DROP COLUMN _compressed_summary")
    conn.commit()
    legacy_cols = {
        row[1] for row in conn.execute("PRAGMA table_info(messages)")
    }
    conn.close()
    assert "_compressed_summary" not in legacy_cols

    upgraded = SessionDB(db_path)
    upgraded.create_session("legacy", source="cli")
    upgraded.append_message(
        "legacy", "assistant", "derivative summary", _compressed_summary=True
    )

    model_history, _display = upgraded.get_resume_conversations("legacy")
    assert model_history[-1].get("_compressed_summary") is True


def test_native_responses_compaction_is_suppressed_when_checkpoint_required():
    """checkpoint_required must keep ``context_management`` off the wire.

    Server-side native compaction is a lossy boundary the provider owns; no
    pre-compress checkpoint can run before it, so the gate suppresses the
    payload while ordinary checkpoint-aware Hermes compression stays
    available.
    """
    from types import SimpleNamespace

    from agent.native_compaction import native_compaction_context_management

    def agent(checkpoint_required):
        return SimpleNamespace(
            model="gpt-5.6",
            base_url="https://api.openai.com/v1",
            codex_responses_native_compaction=True,
            compression_enabled=True,
            compression_checkpoint_required=checkpoint_required,
            codex_responses_compact_threshold=0.8,
            context_compressor=None,
        )

    assert native_compaction_context_management(
        agent(False), is_codex_backend=True
    )
    assert (
        native_compaction_context_management(agent(True), is_codex_backend=True)
        is None
    )


def test_codex_app_server_turn_fails_closed_before_codex_can_compact():
    """checkpoint_required + app-server must never reach ``run_turn()``.

    The codex agent compacts its own thread; once ``run_turn()`` executes, a
    codex-owned compaction may already have happened with no checkpoint. The
    turn entrypoint must raise first — the session is never even created.
    """
    from types import SimpleNamespace

    from agent.codex_runtime import run_codex_app_server_turn

    class _ExplodingSession:
        def run_turn(self, *args, **kwargs):  # pragma: no cover - must not run
            raise AssertionError("run_turn() must not be reached")

    agent = SimpleNamespace(
        api_mode="codex_app_server",
        compression_checkpoint_required=True,
        _codex_session=_ExplodingSession(),
    )

    with pytest.raises(CompressionCheckpointUnavailable, match="codex_app_server"):
        run_codex_app_server_turn(
            agent,
            user_message="hello",
            original_user_message="hello",
            messages=[],
            effective_task_id="t1",
        )


def test_agent_init_refuses_checkpoint_required_on_codex_app_server():
    """The incompatible configuration must fail closed at init time.

    In the default "native" auto-compaction mode Hermes never initiates the
    compaction, so the compress_context() guard alone cannot cover native
    turns — init_agent has to refuse before a turn exists.
    """
    from agent.agent_init import (
        _refuse_checkpoint_required_on_codex_app_server,
    )

    with pytest.raises(RuntimeError, match="BLOCKED_MISSING_PREREQUISITE"):
        _refuse_checkpoint_required_on_codex_app_server(True, "codex_app_server")

    # Every other combination stays permitted.
    _refuse_checkpoint_required_on_codex_app_server(True, "chat_completions")
    _refuse_checkpoint_required_on_codex_app_server(True, "codex_responses")
    _refuse_checkpoint_required_on_codex_app_server(False, "codex_app_server")
    _refuse_checkpoint_required_on_codex_app_server(False, None)


def test_turn_finalizer_never_micro_compacts_while_checkpoint_gate_armed(
    monkeypatch,
):
    """Micro-compaction is a lossy rewrite authority with no checkpoint hook.

    Even if a live agent's compressor has ``_micro_compact_enabled`` flipped
    on (agent init forces it off under the gate, but it is plain mutable
    state), the post-turn finalizer must refuse to call ``_micro_compact()``
    while ``compression_checkpoint_required`` is armed — otherwise assistant
    evidence is absorbed into a rolling summary that the checkpoint filter
    later excludes, and the evidence never reaches the durable provider.
    """
    from tests.agent.test_turn_finalizer_final_response_persistence import (
        FakeAgent,
    )
    from agent.turn_finalizer import finalize_turn

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])

    class _RecordingCompressor:
        _micro_compact_enabled = True

        def __init__(self):
            self.calls = 0

        def _micro_compact(self, messages):
            self.calls += 1
            return list(messages)

    def _run(checkpoint_required: bool):
        agent = FakeAgent()
        compressor = _RecordingCompressor()
        agent.context_compressor = compressor
        agent.compression_checkpoint_required = checkpoint_required
        finalize_turn(
            agent,
            final_response="Done.",
            api_call_count=1,
            interrupted=False,
            failed=False,
            messages=[
                {"role": "user", "content": "do it"},
                {"role": "assistant", "content": "Done."},
            ],
            conversation_history=[],
            effective_task_id="task",
            turn_id="turn",
            user_message="do it",
            original_user_message="do it",
            _should_review_memory=False,
            _turn_exit_reason="completed",
        )
        return compressor.calls

    # Gate armed: micro-compaction never runs.
    assert _run(checkpoint_required=True) == 0

    # Gate off: micro-compaction remains reachable — sabotage control proving
    # this harness genuinely exercises the call site (the finalizer swallows
    # compressor exceptions, so a call counter is the observable signal).
    assert _run(checkpoint_required=False) == 1


def test_agent_init_suppresses_micro_compaction_under_checkpoint_gate():
    """checkpoint_required forces micro-compaction off at init.

    Both keys can be enabled together in config; the gate must win so every
    lossy rewrite passes through the checkpoint-aware batch compressor.
    """
    import inspect

    from agent import agent_init

    source = inspect.getsource(agent_init)
    # The suppression must happen before the compressor attribute assignment.
    suppress_idx = source.find(
        "if compression_checkpoint_required and compression_micro_compact:"
    )
    assign_idx = source.find("_cc._micro_compact_enabled = compression_micro_compact")
    assert suppress_idx != -1, (
        "init_agent must suppress micro-compaction when checkpoint_required"
    )
    assert assign_idx != -1
    assert suppress_idx < assign_idx
