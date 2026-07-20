"""Current-main regressions for bounded compression handoff."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest

from agent import conversation_compression as compression
from agent.conversation_compression import (
    compress_context,
    conversation_history_after_compression,
    create_handoff_packet,
    persist_rejoined_partial_compression,
)
from cli import HermesCLI
from hermes_cli.commands import COMMAND_REGISTRY
from hermes_cli.config import DEFAULT_CONFIG
from hermes_cli.partial_compress import (
    rejoin_compressed_head_and_tail,
    split_history_for_partial_compress,
)
from hermes_cli.web_server import CONFIG_SCHEMA
from hermes_state import SessionDB
from run_agent import AIAgent


class FakeCompressor:
    def __init__(
        self,
        returned: list[dict[str, Any]],
        *,
        compression_count: int = 1,
    ) -> None:
        self.returned = returned
        self.compression_count = compression_count
        self._last_compress_aborted = False
        self._last_summary_error = None
        self._last_summary_fallback_used = False
        self._last_compression_made_progress = True
        self._last_aux_model_failure_model = None
        self._last_aux_model_failure_error = None
        self._verify_compaction_cleared_threshold = False
        self.last_prompt_tokens = 1234
        self.last_completion_tokens = 55
        self.last_compression_rough_tokens = 0
        self.awaiting_real_usage_after_compression = False
        self.reset_called = False
        self.session_start_calls: list[dict[str, Any]] = []

    def compress(self, *_args, **_kwargs):
        self.compression_count += 1
        return [message.copy() for message in self.returned]

    def on_session_reset(self) -> None:
        self.reset_called = True
        self.compression_count = 0
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_compression_rough_tokens = 0
        self.awaiting_real_usage_after_compression = False
        self._last_summary_error = None
        self._last_summary_fallback_used = False
        self._last_compression_made_progress = False
        self._verify_compaction_cleared_threshold = False

    def on_session_start(self, session_id: str, **kwargs) -> None:
        self.session_start_calls.append({"session_id": session_id, **kwargs})

    def record_completed_compaction(self, *, used_fallback: bool = False) -> None:
        self._verify_compaction_cleared_threshold = True


class FakeTodos:
    def read(self):
        return [
            {
                "id": "red-tests",
                "content": "Add failing tests",
                "status": "in_progress",
            }
        ]

    def format_for_injection(self):
        return (
            "[Your active task list was preserved across context compression]\n"
            "- [>] red-tests. Add failing tests"
        )


def _messages() -> list[dict[str, str]]:
    return [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
        {"role": "assistant", "content": "four"},
    ]


def _compressed() -> list[dict[str, str]]:
    return [
        {"role": "user", "content": "compressed summary"},
        {"role": "assistant", "content": "preserved recent tail"},
    ]


def _create_parent(db: SessionDB, session_id: str = "parent") -> None:
    db.create_session(
        session_id,
        source="telegram",
        model="test-model",
        model_config={"max_iterations": 90, "_auto_handoff_count": 0},
        system_prompt="parent system",
        session_key="telegram:chat:thread",
        chat_id="chat",
        chat_type="private",
        thread_id="thread",
    )
    db.replace_messages(session_id, _messages())


def _artifact_paths(tmp_path: Path) -> list[Path]:
    return sorted((tmp_path / "handoffs").glob("*.md"))


def make_agent(
    tmp_path: Path,
    compressor: FakeCompressor,
    *,
    db: SessionDB | None,
    session_id: str | None = "parent",
    in_place: bool = True,
    mode: str = "fresh_session",
    enabled: bool = True,
    after: int = 2,
    maximum: int = 1,
    consumed: int = 0,
) -> Any:
    agent = cast(Any, object.__new__(AIAgent))
    agent.context_compressor = compressor
    agent.session_id = session_id
    agent.model = "test-model"
    agent.provider = "openai-codex"
    agent.platform = "telegram"
    agent.logs_dir = tmp_path / "sessions"
    agent._todo_store = FakeTodos()
    agent._memory_manager = None
    agent._session_db = db
    agent._cached_system_prompt = "cached-parent-prompt"
    agent._compression_feasibility_checked = True
    agent._last_flushed_db_idx = 17
    agent._flushed_db_message_ids = {1, 2}
    agent._flushed_db_message_session_id = session_id
    agent._session_db_created = db is not None
    agent._session_init_model_config = {
        "max_iterations": 90,
        "_auto_handoff_count": consumed,
    }
    agent._parent_session_id = None
    agent._gateway_session_key = "telegram:chat:thread"
    agent._gateway_turn_context_notes = "parent note"
    agent._current_task_id = "default"
    agent._current_turn_id = f"{session_id}:default:parentturn"
    agent._current_api_request_id = "parent-request"
    agent._inflight_turn_id = "parent-inflight"
    agent._inflight_turn_session_id = session_id
    agent.tools = []
    agent.log_prefix = ""
    agent.status_callback = None
    agent.event_callback = None
    agent.compression_in_place = in_place
    agent._persist_disabled = False
    agent._auto_handoff_on_compression_enabled = enabled
    agent._auto_handoff_after_compressions = after
    agent._auto_handoff_max_auto_handoffs = maximum
    agent._auto_handoff_count = consumed
    agent._auto_handoff_mode = mode
    agent._auto_handoff_artifact_dir = "handoffs"
    agent._memory_write_origin = "assistant_tool"
    agent._memory_write_context = "foreground"
    agent._last_compaction_in_place = False
    agent._compression_warning = None
    agent._last_aux_fallback_warning_key = ("old-model", "old-error")
    agent.session_prompt_tokens = 101
    agent.session_completion_tokens = 102
    agent.session_total_tokens = 203
    agent.session_api_calls = 4
    agent.session_input_tokens = 101
    agent.session_output_tokens = 102
    agent.session_cache_read_tokens = 3
    agent.session_cache_write_tokens = 4
    agent.session_reasoning_tokens = 5
    agent._user_turn_count = 2
    agent._turns_since_memory = 2
    agent._iters_since_skill = 2
    agent.session_estimated_cost_usd = 1.0
    agent.session_cost_status = "estimated"
    agent.session_cost_source = "local"
    agent._session_messages = ["parent"]
    agent.emitted_statuses: list[str] = []
    agent.emitted_warnings: list[str] = []
    agent.commit_memory_session = lambda *_args, **_kwargs: None
    agent._invalidate_system_prompt = lambda *_args, **_kwargs: None
    agent._build_system_prompt = lambda system_message: f"built::{system_message}"
    agent._vprint = lambda *_args, **_kwargs: None
    agent._emit_status = agent.emitted_statuses.append
    agent._emit_warning = agent.emitted_warnings.append

    def flush(messages):
        if db is not None and session_id:
            db.replace_messages(agent.session_id, messages, active_only=True)

    agent._flush_messages_to_session_db = flush
    return agent


def test_config_defaults_schema_and_command_are_visible():
    auto = DEFAULT_CONFIG["agent"]["auto_handoff_on_compression"]
    assert auto == {
        "enabled": False,
        "after_compressions": 2,
        "max_auto_handoffs": 1,
        "mode": "prompt_user",
        "handoff_artifact_dir": ".hermes/handoffs",
    }
    for key in auto:
        assert f"agent.auto_handoff_on_compression.{key}" in CONFIG_SCHEMA
    command = next(cmd for cmd in COMMAND_REGISTRY if cmd.name == "handoff-packet")
    assert command.cli_only is True
    assert "handoff_packet" in command.aliases


def test_fresh_handoff_overrides_in_place_compaction_atomically(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        _create_parent(db)
        compressor = FakeCompressor(_compressed())
        agent = make_agent(tmp_path, compressor, db=db, in_place=True)

        returned, prompt = compress_context(
            agent,
            _messages(),
            "CUSTOM SYSTEM MESSAGE",
            approx_tokens=250_000,
        )

        child_id = agent.session_id
        assert child_id and child_id != "parent"
        assert prompt == "built::CUSTOM SYSTEM MESSAGE"
        assert returned and len(returned) == 1
        assert returned[0]["role"] == "user"
        assert "# Hermes handoff packet" in returned[0]["content"]
        assert db.get_session("parent")["end_reason"] == "compression"
        assert db.get_session(child_id)["parent_session_id"] == "parent"
        assert db.get_auto_handoff_count(child_id) == 1
        assert agent._auto_handoff_count == 1
        assert compressor.reset_called is True
        assert agent._last_compaction_in_place is False
        assert conversation_history_after_compression(agent, returned) is None
        assert agent._current_turn_id.startswith(f"{child_id}:")
        assert agent._current_api_request_id == ""
        assert agent.session_total_tokens == 0
        assert agent.session_api_calls == 0
        assert agent._last_aux_fallback_warning_key is None
        assert agent._cached_system_prompt == prompt
        assert len(_artifact_paths(tmp_path)) == 1
        assert db.get_compression_lock_holder("parent") is None
    finally:
        db.close()


def test_legacy_rotation_preserves_lineage_messages_and_gateway_coordinates(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        _create_parent(db)
        agent = make_agent(
            tmp_path,
            FakeCompressor(_compressed(), compression_count=0),
            db=db,
            in_place=False,
            after=99,
        )

        returned, _ = compress_context(
            agent, _messages(), "legacy system", approx_tokens=200_000
        )

        child_id = agent.session_id
        assert child_id and child_id != "parent"
        assert db.get_session("parent")["end_reason"] == "compression"
        assert db.get_session(child_id)["parent_session_id"] == "parent"
        assert db.get_compression_lineage(child_id) == ["parent", child_id]
        assert [row["content"] for row in db.get_messages(child_id)] == [
            message["content"] for message in returned
        ]
        assert conversation_history_after_compression(agent, returned) is None
        assert agent._last_flushed_db_idx == len(returned)
        assert agent._flushed_db_message_session_id == child_id
        assert agent._last_compaction_in_place is False
    finally:
        db.close()


@pytest.mark.parametrize(("enabled", "after"), [(False, 2), (True, 3)])
def test_disabled_and_below_threshold_keep_current_in_place_semantics(
    tmp_path, monkeypatch, enabled, after
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        _create_parent(db)
        agent = make_agent(
            tmp_path,
            FakeCompressor(_compressed()),
            db=db,
            in_place=True,
            enabled=enabled,
            after=after,
        )

        returned, _ = compress_context(
            agent, _messages(), "system", approx_tokens=200_000
        )

        assert agent.session_id == "parent"
        assert db.get_session("parent")["ended_at"] is None
        assert [row["content"] for row in db.get_messages("parent")] == [
            message["content"] for message in returned
        ]
        assert db.has_archived_messages("parent") is True
        assert agent._auto_handoff_count == 0
        assert _artifact_paths(tmp_path) == []
        assert agent._last_compaction_in_place is True
        assert db.get_compression_lock_holder("parent") is None
    finally:
        db.close()


def test_child_creation_failure_is_fail_closed_and_falls_back_in_place(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        _create_parent(db)
        agent = make_agent(tmp_path, FakeCompressor(_compressed()), db=db)
        with patch.object(
            db,
            "rotate_session_for_compression",
            side_effect=RuntimeError("forced rotation failure"),
        ):
            returned, _ = compress_context(
                agent, _messages(), "system", approx_tokens=200_000
            )

        assert agent.session_id == "parent"
        assert db.get_session("parent")["ended_at"] is None
        assert db._conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE parent_session_id = 'parent'"
        ).fetchone()[0] == 0
        assert [row["content"] for row in db.get_messages("parent")] == [
            message["content"] for message in returned
        ]
        assert agent._last_compaction_in_place is True
        assert agent._auto_handoff_count == 0
        assert _artifact_paths(tmp_path) == []
    finally:
        db.close()


def test_multi_generation_quota_allows_exactly_n_fresh_handoffs(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        _create_parent(db)
        compressor = FakeCompressor(_compressed())
        agent = make_agent(
            tmp_path,
            compressor,
            db=db,
            in_place=True,
            maximum=2,
        )

        first, _ = compress_context(
            agent, _messages(), "system", approx_tokens=200_000
        )
        first_child = agent.session_id
        assert first_child != "parent"

        compressor.compression_count = 1
        agent._compression_feasibility_checked = True
        second, _ = compress_context(
            agent, first + _messages(), "system", approx_tokens=200_000
        )
        second_child = agent.session_id
        assert second_child != first_child
        assert db.get_auto_handoff_count(second_child) == 2

        compressor.compression_count = 1
        agent._compression_feasibility_checked = True
        third, _ = compress_context(
            agent, second + _messages(), "system", approx_tokens=200_000
        )

        assert agent.session_id == second_child
        assert db.get_compression_lineage(second_child) == [
            "parent",
            first_child,
            second_child,
        ]
        assert db.get_auto_handoff_count(second_child) == 2
        assert third[0]["content"] == "compressed summary"
        assert agent._last_compaction_in_place is True
        assert len(_artifact_paths(tmp_path)) == 2
        assert any("Consider /new" in message for message in agent.emitted_statuses)
    finally:
        db.close()


def test_prompt_user_packet_does_not_rotate_in_place_session(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        _create_parent(db)
        agent = make_agent(
            tmp_path,
            FakeCompressor(_compressed()),
            db=db,
            in_place=True,
            mode="prompt_user",
        )

        returned, _ = compress_context(
            agent, _messages(), "system", approx_tokens=200_000
        )

        assert agent.session_id == "parent"
        assert returned[0]["content"] == "compressed summary"
        assert db.get_session("parent")["ended_at"] is None
        assert db.get_auto_handoff_count("parent") == 1
        assert len(_artifact_paths(tmp_path)) == 1
    finally:
        db.close()


def test_failed_prompt_user_publication_does_not_consume_quota(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        _create_parent(db)
        agent = make_agent(
            tmp_path,
            FakeCompressor(_compressed()),
            db=db,
            mode="prompt_user",
        )
        with patch.object(
            compression,
            "_publish_handoff_artifact",
            side_effect=OSError("forced artifact failure"),
        ):
            compress_context(agent, _messages(), "system", approx_tokens=200_000)

        assert db.get_auto_handoff_count("parent") == 0
        assert agent._auto_handoff_count == 0
        assert _artifact_paths(tmp_path) == []
    finally:
        db.close()


def test_manual_partial_compression_cannot_trigger_handoff_or_drop_recent_tail(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        source = _messages() + [
            {"role": "user", "content": "RECENT USER"},
            {"role": "assistant", "content": "RECENT ASSISTANT"},
        ]
        _create_parent(db)
        agent = make_agent(tmp_path, FakeCompressor(_compressed()), db=db)
        head, tail = split_history_for_partial_compress(source, keep_last=1)

        compressed_head, _ = compress_context(
            agent,
            head,
            "system",
            approx_tokens=200_000,
            focus_topic="manual partial",
            force=True,
        )
        joined = rejoin_compressed_head_and_tail(compressed_head, tail)
        assert persist_rejoined_partial_compression(agent, joined) is True

        assert agent.session_id == "parent"
        assert agent._auto_handoff_count == 0
        assert _artifact_paths(tmp_path) == []
        assert "RECENT USER" in str(joined[-2].get("content", ""))
        assert joined[-1]["content"] == "RECENT ASSISTANT"
        persisted_tail = [
            row["content"] for row in db.get_messages("parent")
        ][-2:]
        assert "RECENT USER" in persisted_tail[0]
        assert persisted_tail[1] == "RECENT ASSISTANT"
    finally:
        db.close()


def test_fresh_handoff_with_null_session_db_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    agent = make_agent(
        tmp_path,
        FakeCompressor(_compressed()),
        db=None,
        session_id=None,
        in_place=True,
    )

    returned, prompt = compress_context(
        agent, _messages(), "NULL DB SYSTEM", approx_tokens=200_000
    )

    assert agent.session_id is None
    assert returned[0]["content"] == "compressed summary"
    assert prompt == "built::NULL DB SYSTEM"
    assert agent._auto_handoff_count == 0
    assert _artifact_paths(tmp_path) == []


def test_restart_resumes_durable_auto_handoff_count(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session(
            "resumed-child",
            source="cli",
            model_config={"_auto_handoff_count": 2},
        )
        cfg = {
            "agent": {
                "auto_handoff_on_compression": {
                    "enabled": True,
                    "after_compressions": 2,
                    "max_auto_handoffs": 3,
                    "mode": "fresh_session",
                    "handoff_artifact_dir": "handoffs",
                }
            }
        }
        with (
            patch("run_agent.get_tool_definitions", return_value=[]),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
            patch("hermes_cli.config.load_config", return_value=cfg),
        ):
            agent = AIAgent(
                api_key="dummy",
                base_url="https://example.test/v1",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                session_id="resumed-child",
                session_db=db,
            )
        assert agent._auto_handoff_count == 2
        assert agent._session_init_model_config["_auto_handoff_count"] == 2
    finally:
        db.close()


def test_prompt_user_legacy_packet_uses_final_child_coordinates(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        _create_parent(db)
        agent = make_agent(
            tmp_path,
            FakeCompressor(_compressed()),
            db=db,
            in_place=False,
            mode="prompt_user",
        )

        compress_context(agent, _messages(), "system", approx_tokens=200_000)

        child_id = agent.session_id
        assert child_id != "parent"
        assert db.get_auto_handoff_count(child_id) == 1
        artifacts = _artifact_paths(tmp_path)
        assert len(artifacts) == 1
        packet = artifacts[0].read_text()
        assert f"- Active session: `{child_id}`" in packet
        assert "- Parent session: `parent`" in packet
    finally:
        db.close()


def test_prompt_user_rotation_failure_packet_uses_preserved_parent_coordinates(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        _create_parent(db)
        agent = make_agent(
            tmp_path,
            FakeCompressor(_compressed()),
            db=db,
            in_place=False,
            mode="prompt_user",
        )
        with patch.object(
            db,
            "rotate_session_for_compression",
            side_effect=RuntimeError("forced rotation failure"),
        ):
            compress_context(agent, _messages(), "system", approx_tokens=200_000)

        assert agent.session_id == "parent"
        artifacts = _artifact_paths(tmp_path)
        assert len(artifacts) == 1
        packet = artifacts[0].read_text()
        assert "- Active session: `parent`" in packet
        assert "- Parent session: `none`" in packet
        assert db.get_auto_handoff_count("parent") == 1
    finally:
        db.close()


def test_fresh_packet_publication_failure_keeps_parent_and_quota(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        _create_parent(db)
        agent = make_agent(tmp_path, FakeCompressor(_compressed()), db=db)
        with patch.object(
            compression,
            "_publish_handoff_artifact",
            side_effect=OSError("forced artifact failure"),
        ):
            compress_context(agent, _messages(), "system", approx_tokens=200_000)

        assert agent.session_id == "parent"
        assert db.get_session("parent")["ended_at"] is None
        assert db.get_auto_handoff_count("parent") == 0
        assert agent._last_compaction_in_place is True
        assert _artifact_paths(tmp_path) == []
    finally:
        db.close()


def test_post_commit_exception_adopts_authoritative_child_instead_of_fallback(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        _create_parent(db)
        agent = make_agent(tmp_path, FakeCompressor(_compressed()), db=db)
        real_rotate = db.rotate_session_for_compression

        def committed_then_raised(**kwargs):
            real_rotate(**kwargs)
            raise RuntimeError("response lost after commit")

        with patch.object(
            db, "rotate_session_for_compression", side_effect=committed_then_raised
        ):
            returned, _ = compress_context(
                agent, _messages(), "system", approx_tokens=200_000
            )

        child_id = agent.session_id
        assert child_id != "parent"
        assert db.get_session("parent")["end_reason"] == "compression"
        assert db.get_session(child_id)["parent_session_id"] == "parent"
        assert db.get_auto_handoff_count(child_id) == 1
        assert returned[0]["role"] == "user"
        assert agent._last_compaction_in_place is False
    finally:
        db.close()


def test_artifact_collision_retries_without_clobbering_existing_file(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    agent = make_agent(
        tmp_path,
        FakeCompressor(_compressed()),
        db=None,
        in_place=True,
    )
    real_link = compression.os.link
    link_calls = 0

    def collide_once(*args, **kwargs):
        nonlocal link_calls
        link_calls += 1
        if link_calls == 1:
            raise FileExistsError("forced collision")
        return real_link(*args, **kwargs)

    with patch.object(compression.os, "link", side_effect=collide_once):
        _, path = create_handoff_packet(agent, _messages())

    assert link_calls == 2
    assert path is not None and path.exists()
    assert len(_artifact_paths(tmp_path)) == 1


def test_fresh_handoff_preserves_title(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        _create_parent(db)
        db.set_session_title("parent", "Compression project")
        agent = make_agent(tmp_path, FakeCompressor(_compressed()), db=db)

        compress_context(agent, _messages(), "system", approx_tokens=200_000)

        child_id = agent.session_id
        assert child_id != "parent"
        assert db.get_session_title(child_id) == "Compression project #2"
    finally:
        db.close()


def test_prompt_user_publication_happens_before_quota_consumption(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        _create_parent(db)
        agent = make_agent(
            tmp_path,
            FakeCompressor(_compressed()),
            db=db,
            mode="prompt_user",
        )
        real_consume = db.consume_auto_handoff

        def consume_after_publish(*args, **kwargs):
            assert len(_artifact_paths(tmp_path)) == 1
            return real_consume(*args, **kwargs)

        with patch.object(db, "consume_auto_handoff", side_effect=consume_after_publish):
            compress_context(agent, _messages(), "system", approx_tokens=200_000)

        assert db.get_auto_handoff_count("parent") == 1
        assert len(_artifact_paths(tmp_path)) == 1
    finally:
        db.close()


def test_manual_packet_and_cli_command_never_rotate_session(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        _create_parent(db)
        agent = make_agent(tmp_path, FakeCompressor(_compressed()), db=db)
        packet, path = create_handoff_packet(
            agent,
            [{"role": "user", "content": "Current task"}],
            approx_tokens=123,
        )
        assert path is not None and path.exists()
        assert "manual packet; no new session started" in packet.lower()
        assert agent.session_id == "parent"
        assert db.get_session("parent")["ended_at"] is None
        assert db.get_auto_handoff_count("parent") == 0

        cli = cast(Any, HermesCLI.__new__(HermesCLI))
        cli.agent = agent
        cli.conversation_history = [{"role": "user", "content": "Need handoff"}]
        cli._busy_command = lambda _message: nullcontext()
        outputs: list[str] = []
        with patch("cli._cprint", lambda message="": outputs.append(str(message))):
            cli._handle_handoff_packet_command(
                "/handoff-packet before risky edit"
            )
        assert agent.session_id == "parent"
        assert db.get_session("parent")["ended_at"] is None
        assert any("Handoff packet written" in line for line in outputs)
    finally:
        db.close()


def test_artifact_directory_cannot_escape_or_follow_symlink(tmp_path, monkeypatch):
    home = tmp_path / "home"
    outside = tmp_path / "outside"
    home.mkdir()
    outside.mkdir()
    (home / "escape").symlink_to(outside, target_is_directory=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    agent = make_agent(
        tmp_path,
        FakeCompressor(_compressed()),
        db=None,
        in_place=True,
    )

    agent._auto_handoff_artifact_dir = "../outside"
    packet, path = create_handoff_packet(agent, _messages())
    assert "# Hermes handoff packet" in packet
    assert path is None

    agent._auto_handoff_artifact_dir = "escape"
    _, symlink_path = create_handoff_packet(agent, _messages())
    assert symlink_path is None
    assert list(outside.iterdir()) == []
