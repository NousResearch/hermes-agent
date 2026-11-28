"""compress_context facade resets guardrail no-progress memory only when a rewrite survives."""
import copy
import json
import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from agent.tool_guardrails import ToolCallGuardrailConfig, ToolCallGuardrailController

from tests.fixtures.storage_read import conversation
from tests.fixtures.storage_seed import seed_messages, seed_session
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle

ARGS = {"name": "some-skill"}
RESULT = json.dumps({"success": True, "name": "some-skill", "content": "x" * 600})


def _agent_at_block_threshold(**kwargs):
    guard = ToolCallGuardrailController(ToolCallGuardrailConfig(hard_stop_enabled=True))
    for _ in range(5):
        guard.before_call("skill_view", ARGS)
        guard.after_call("skill_view", ARGS, RESULT, failed=False)
    assert not guard.before_call("skill_view", ARGS).allows_execution
    with patch("agent.process_bootstrap.OpenAI"):
        from run_agent import AIAgent

        agent = AIAgent(api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1",
                        quiet_mode=True, skip_context_files=True, skip_memory=True, **kwargs)
    agent._tool_guardrails = guard
    compressor = MagicMock()
    compressor.compression_count = 1
    compressor.last_prompt_tokens = 0
    compressor.last_completion_tokens = 0
    compressor._last_summary_error = None
    compressor._last_compress_aborted = False
    agent.context_compressor = compressor
    return agent


def _compress(agent, messages, rewrite):
    agent.context_compressor.compress.side_effect = rewrite
    return agent._compress_context(messages, "sys", approx_tokens=10000, force=True)


def test_aborted_compaction_keeps_no_progress_memory():
    agent = _agent_at_block_threshold()
    messages = [{"role": "user", "content": "a" * 400}, {"role": "assistant", "content": "b"}]
    _compress(agent, messages, lambda current, **_kwargs: current)
    assert not agent._tool_guardrails.before_call("skill_view", ARGS).allows_execution


def test_committed_compaction_resets_no_progress_memory():
    agent = _agent_at_block_threshold()
    messages = [{"role": "user", "content": "a" * 400}, {"role": "assistant", "content": "b"}]
    _compress(agent, messages, lambda current, **_kwargs: [current[0]])
    assert agent._tool_guardrails.before_call("skill_view", ARGS).allows_execution


def test_in_place_compaction_resets_no_progress_memory():
    agent = _agent_at_block_threshold()
    messages = [{"role": "user", "content": "a" * 400}, {"role": "assistant", "content": "b"}]
    def rewrite(current, **_kwargs):
        del current[1]
        return current

    _compress(agent, messages, rewrite)
    assert agent._tool_guardrails.before_call("skill_view", ARGS).allows_execution


def test_agent_without_guardrails_is_ignored():
    messages = [{"role": "user", "content": "a" * 400}]
    agent = _agent_at_block_threshold()
    agent._tool_guardrails = None
    _compress(agent, messages, lambda current, **_kwargs: [{"role": "user", "content": "s"}])


@pytest.mark.parametrize("in_place", [True, False], ids=["in-place", "rotated"])
@pytest.mark.parametrize("commit_fails", [True, False], ids=["rollback", "commit"])
def test_persistent_compaction_resets_guard_only_when_rewrite_survives(
    tmp_path, monkeypatch, in_place, commit_fails,
):
    db = storage_handle(tmp_path)
    sid = "guard-window"
    messages = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"message {i} " + "x" * 400}
        for i in range(8)
    ]
    seed_session(db, sid, "cli")
    seed_messages(db, sid, *messages)
    for message in messages:
        message["_db_persisted"] = True
    original = copy.deepcopy(messages)
    durable_before = conversation(db, sid)
    agent = _agent_at_block_threshold(persistence=db, session_id=sid)
    agent.compression_in_place = in_place
    attempts = []

    if commit_fails:
        def fail_commit(*args, **kwargs):
            attempts.append(True)
            raise sqlite3.OperationalError("injected compaction persistence failure")

        repository, method = (db.messages, "compact") if in_place else (db.lineage, "publish_compression_child")
        monkeypatch.setattr(repository, method, fail_commit)

    rewritten = [{"role": "user", "content": "[CONTEXT COMPACTION] short summary"}]
    compressed, _ = _compress(agent, messages, lambda _current, **_kwargs: rewritten)
    if commit_fails:
        assert attempts == [True]
        assert agent.session_id == sid
        assert compressed is messages and compressed == original
        assert conversation(db, sid) == durable_before
        assert not agent._tool_guardrails.before_call("skill_view", ARGS).allows_execution
    else:
        assert (agent.session_id == sid) is in_place
        assert compressed != original
        assert conversation(db, agent.session_id)[0]["content"] == rewritten[0]["content"]
        assert agent._tool_guardrails.before_call("skill_view", ARGS).allows_execution
