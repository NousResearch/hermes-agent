"""DAG context engine native loading and non-destructive compression tests."""

from unittest.mock import MagicMock, patch


class _ProjectionOnlyCompressor:
    name = "dag"
    last_prompt_tokens = 0
    last_completion_tokens = 0
    compression_count = 0

    def __init__(self, projected):
        self.projected = projected

    def compress(self, messages, current_tokens=None, focus_topic=None):
        from agent.context_engine import ContextCompressionResult

        return ContextCompressionResult(
            messages=[dict(m) for m in self.projected],
            projection_only=True,
            preserves_session=True,
        )


def test_context_engine_default_remains_builtin_compressor():
    cfg = {"agent": {}, "compression": {"enabled": False}}
    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    assert agent.context_compressor.name == "compressor"


def test_context_engine_dag_loads_native_without_plugin(tmp_path):
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    cfg = {"context": {"engine": "dag"}, "agent": {}, "compression": {"enabled": False}}
    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("plugins.context_engine.load_context_engine") as plugin_loader,
        patch("agent.model_metadata.get_model_context_length", return_value=131_072),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            session_id="sess",
            session_db=db,
        )

    assert agent.context_compressor.name == "dag"
    plugin_loader.assert_not_called()


def test_dag_projection_only_persist_skips_projection_rows_after_compression(tmp_path):
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("sess", "cli")
    db.append_message("sess", "user", "old user")
    db.append_message("sess", "assistant", "old assistant")

    cfg = {"agent": {}, "compression": {"enabled": True}}
    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("agent.model_metadata.get_model_context_length", return_value=131_072),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            session_id="sess",
            session_db=db,
        )

    raw_messages = [
        {"role": "user", "content": "old user"},
        {"role": "assistant", "content": "old assistant"},
        {"role": "user", "content": "current user"},
    ]
    agent._persist_user_message_idx = 2
    agent.context_compressor = _ProjectionOnlyCompressor(
        [
            {"role": "user", "content": "DAG SUMMARY: old user / old assistant"},
            {"role": "assistant", "content": "old assistant"},
            {"role": "user", "content": "current user"},
        ]
    )

    projected, _ = agent._compress_context(raw_messages, "", approx_tokens=100)
    projected.append({"role": "assistant", "content": "current answer"})

    # Simulates the PR6 blocker path: projection shrank the live messages list,
    # caller dropped conversation_history, and final persist would previously
    # append projection/fresh-tail rows from index 0 into the raw transcript.
    agent._persist_session(projected, None)

    contents = [m["content"] for m in db.get_messages("sess")]
    assert contents == [
        "old user",
        "old assistant",
        "current user",
        "current answer",
    ]


def test_real_dag_compress_reconcile_current_user_not_persisted_twice(tmp_path):
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("sess", "cli")
    old_user_id = db.append_message("sess", "user", "old user")
    old_assistant_id = db.append_message("sess", "assistant", "old assistant")

    cfg = {"context": {"engine": "dag"}, "agent": {}, "compression": {"enabled": True}}
    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("agent.model_metadata.get_model_context_length", return_value=131_072),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            session_id="sess",
            session_db=db,
        )

    # Real DAG projection: old transcript is represented by a summary, while
    # compress() reconciles the caller transcript (including current user) into
    # the raw DB before returning the projection view.
    store = agent.context_compressor.store
    summary = store.create_summary_node(
        session_id="sess",
        node_id="summary-old",
        summary_text="summary of old user and old assistant",
        token_estimate=8,
        metadata={"source_span": {"start_message_id": old_user_id, "end_message_id": old_assistant_id}},
    )
    store.write_active_projection(
        session_id="sess",
        engine_version=agent.context_compressor.ENGINE_VERSION,
        projection=[
            {
                "kind": "summary",
                "summary_id": summary.id,
                "source_span": {"start_message_id": old_user_id, "end_message_id": old_assistant_id},
                "token_estimate": 8,
            }
        ],
        latest_raw_message_id=old_assistant_id,
    )

    raw_messages = [
        {"role": "user", "content": "old user"},
        {"role": "assistant", "content": "old assistant"},
        {"role": "user", "content": "current user"},
    ]
    agent._persist_user_message_idx = 2

    projected, _ = agent._compress_context(raw_messages, "", approx_tokens=100)
    assert any("summary of old user" in m.get("content", "") for m in projected)
    assert projected[-1]["content"] == "current user"

    projected.append({"role": "assistant", "content": "current answer"})
    agent._persist_session(projected, None)

    contents = [m["content"] for m in db.get_messages("sess")]
    assert contents == [
        "old user",
        "old assistant",
        "current user",
        "current answer",
    ]
    assert contents.count("current user") == 1
    assert all("REFERENCE-ONLY CONTEXT SUMMARY" not in content for content in contents)
    assert all("summary of old user" not in content for content in contents)


def test_real_dag_checkpoint_failure_after_raw_append_does_not_duplicate_current_user(tmp_path):
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("sess", "cli")
    old_user_id = db.append_message("sess", "user", "old user")
    old_assistant_id = db.append_message("sess", "assistant", "old assistant")

    cfg = {"context": {"engine": "dag"}, "agent": {}, "compression": {"enabled": True}}
    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("agent.model_metadata.get_model_context_length", return_value=131_072),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            session_id="sess",
            session_db=db,
        )

    store = agent.context_compressor.store
    summary = store.create_summary_node(
        session_id="sess",
        node_id="summary-old",
        summary_text="summary of old user and old assistant",
        token_estimate=8,
        metadata={"source_span": {"start_message_id": old_user_id, "end_message_id": old_assistant_id}},
    )
    store.write_active_projection(
        session_id="sess",
        engine_version=agent.context_compressor.ENGINE_VERSION,
        projection=[
            {
                "kind": "summary",
                "summary_id": summary.id,
                "source_span": {"start_message_id": old_user_id, "end_message_id": old_assistant_id},
                "token_estimate": 8,
            }
        ],
        latest_raw_message_id=old_assistant_id,
    )

    raw_messages = [
        {"role": "user", "content": "old user"},
        {"role": "assistant", "content": "old assistant"},
        {"role": "user", "content": "current user"},
    ]
    agent._persist_user_message_idx = 2

    with patch.object(store, "write_checkpoint", side_effect=RuntimeError("checkpoint unavailable")):
        projected, _ = agent._compress_context(raw_messages, "", approx_tokens=100)

    assert any("summary of old user" in m.get("content", "") for m in projected)
    assert projected[-1]["content"] == "current user"
    assert store.read_checkpoint("sess") is None

    projected.append({"role": "assistant", "content": "current answer"})
    agent._persist_session(projected, None)

    contents = [m["content"] for m in db.get_messages("sess")]
    assert contents == [
        "old user",
        "old assistant",
        "current user",
        "current answer",
    ]
    assert contents.count("current user") == 1
    assert contents.count("current answer") == 1
    assert all("REFERENCE-ONLY CONTEXT SUMMARY" not in content for content in contents)


def test_real_dag_reconcile_failure_fallback_keeps_current_user_persistable(tmp_path):
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("sess", "cli")
    db.append_message("sess", "user", "old user")
    db.append_message("sess", "assistant", "old assistant")

    cfg = {"context": {"engine": "dag"}, "agent": {}, "compression": {"enabled": True}}
    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("agent.model_metadata.get_model_context_length", return_value=131_072),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            session_id="sess",
            session_db=db,
        )

    raw_messages = [
        {"role": "user", "content": "old user"},
        {"role": "assistant", "content": "old assistant"},
        {"role": "user", "content": "current user"},
    ]
    agent._persist_user_message_idx = 2

    with patch.object(agent.context_compressor, "reconcile_transcript", side_effect=RuntimeError("transient db write failure")):
        projected, _ = agent._compress_context(raw_messages, "", approx_tokens=100)

    # The transient failure is gone before final persistence. The fallback API
    # projection must not mark the current caller row as non-persistable, or the
    # raw user message would be lost when conversation_history is unavailable.
    assert projected == raw_messages
    projected.append({"role": "assistant", "content": "current answer"})
    agent._persist_session(projected, None)

    contents = [m["content"] for m in db.get_messages("sess")]
    assert contents == [
        "old user",
        "old assistant",
        "current user",
        "current answer",
    ]
    assert contents.count("current user") == 1
    assert contents.count("old user") == 1
    assert contents.count("old assistant") == 1


def test_dag_compress_context_preserves_session_and_skips_destructive_side_effects(tmp_path):
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("sess", "cli")
    db.append_message("sess", "user", "hello")
    cfg = {"context": {"engine": "dag"}, "agent": {}, "compression": {"enabled": True}}
    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("agent.model_metadata.get_model_context_length", return_value=131_072),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            session_id="sess",
            session_db=db,
        )

    db.end_session = MagicMock()
    agent.commit_memory_session = MagicMock()
    old_session_id = agent.session_id

    compressed, _ = agent._compress_context(
        [{"role": "user", "content": "hello"}],
        "",
        approx_tokens=100,
    )

    assert compressed[-1]["content"] == "hello"
    assert agent.session_id == old_session_id
    db.end_session.assert_not_called()
    agent.commit_memory_session.assert_not_called()
