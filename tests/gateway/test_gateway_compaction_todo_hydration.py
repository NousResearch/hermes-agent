"""Gateway compaction retains structured tasks and opaque snapshot-only fallbacks."""

import json
import os
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import run_agent
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key
from tools.todo_tool import TODO_INJECTION_HEADER, TodoStore
from hermes_state import SessionDB

_TASK = "Review checklist:\n- [ ] tests. Run tests"
_TODOS = [{"id": "t1", "content": _TASK, "status": "in_progress"}]


def _history() -> list[dict]:
    rows = [
        {"role": "user", "content": "plan the release " + "x" * 400},
        {"role": "assistant", "content": "Earlier work was summarised. " + "y" * 400},
    ]
    for i in range(8):
        rows.append({"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i} " + "z" * 400})
    rows.append({"role": "user", "content": "keep going"})
    return rows


def _compressed_tail() -> list[dict]:
    # The summariser folded the todo tool call and result into prose: no todo rows remain.
    return [
        {"role": "user", "content": "[CONTEXT COMPACTION] earlier work summarised"},
        {"role": "assistant", "content": "acknowledged"},
        {"role": "user", "content": "keep going"},
    ]


def _real_agent_factory(built: list, tail=None):
    """Build a real AIAgent from the gateway builder's kwargs; only the summariser is faked."""
    real_cls = run_agent.AIAgent

    def _factory(**kwargs):
        kwargs.setdefault("base_url", "https://openrouter.ai/api/v1")
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            agent = real_cls(skip_context_files=True, **kwargs)
        compressor = MagicMock()
        compressor.compression_count = 1
        compressor.context_length = 272_000  # a MagicMock window would read as 1 token
        compressor.last_prompt_tokens = 0
        compressor.last_completion_tokens = 0
        compressor._last_summary_error = None
        compressor._last_compress_aborted = False
        compressor._last_summary_auth_failure = False
        compressor._last_aux_model_failure_model = None
        compressor._last_aux_model_failure_error = None
        compressor.has_content_to_compress.return_value = True
        compressor.compress.return_value = tail if tail is not None else _compressed_tail()
        agent.context_compressor = compressor
        built.append(agent)
        return agent

    return _factory


def _assert_todo_carried(compressed: list) -> None:
    text = "\n".join(str(m.get("content") or "") for m in compressed if isinstance(m, dict))
    assert TODO_INJECTION_HEADER in text
    assert _TASK in text


def _source() -> SessionSource:
    return SessionSource(platform=Platform.TELEGRAM, user_id="u1", chat_id="c1", user_name="tester", chat_type="dm")


def _todo_pair(call_id: str, todos: list, revision: int) -> list[dict]:
    return [
        {"role": "assistant", "content": None, "tool_calls": [{"id": call_id, "type": "function", "function": {
            "name": "todo", "arguments": json.dumps({"todos": todos})}}]},
        {"role": "tool", "tool_call_id": call_id, "content": json.dumps({"todos": todos, "revision": revision})},
    ]


_REV1 = [{"id": "t1", "content": "Step one", "status": "in_progress"}]
_SNAPSHOT_REV3 = {"role": "user", "content": "keep going\n\n" + TODO_INJECTION_HEADER + "\n- [>] t3. Step three (in_progress)"}


async def _manual_compress(history: list, tail=None, db=None, built=None) -> list:
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")})
    entry = SessionEntry(session_key=build_session_key(_source()), session_id="sess-todo",
                         created_at=datetime.now(), updated_at=datetime.now(),
                         platform=Platform.TELEGRAM, chat_type="dm")
    runner.session_store = MagicMock()
    runner.session_store.get_model_override.return_value = None
    runner.session_store.get_reasoning_override.return_value = None
    runner.session_store.get_or_create_session.return_value = entry
    runner.session_store.load_transcript.return_value = history
    runner._session_db = _async_db(db)
    runner._persist_manual_compression = AsyncMock()
    built = [] if built is None else built

    with (
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "test-key"}),
        patch("gateway.run._resolve_gateway_model", return_value="test/model"),
        patch("run_agent.AIAgent", side_effect=_real_agent_factory(built, tail)),
        patch("agent.model_metadata.estimate_request_tokens_rough", side_effect=lambda m, **_k: 10 * len(m)),
    ):
        await runner._handle_compress_command(MessageEvent(text="/compress", source=_source(), message_id="m1"))

    assert built, "the gateway builder did not construct the compaction agent"
    runner._persist_manual_compression.assert_awaited_once()
    return runner._persist_manual_compression.await_args.args[3]


@pytest.mark.asyncio
async def _hygiene_compress(history, db=None, built=None):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._session_db = _async_db(db)
    built = [] if built is None else built
    results: list = []
    factory = _real_agent_factory(built)

    def _capturing_factory(**kwargs):
        agent = factory(**kwargs)
        real_compress = agent._compress_context

        def _compress(*args, **kw):
            out = real_compress(*args, **kw)
            results.append(out[0])
            return out

        agent._compress_context = _compress
        return agent

    async def _wait(attempt, _hs, _entry):
        return await attempt.future

    runner._hmwa_hygiene_wait_for_summary = _wait
    runner._hmwa_hygiene_apply_result = AsyncMock()
    runner._track_deferred_agent_worker = lambda *_a, **_k: None
    runner._cleanup_agent_resources_off_loop = AsyncMock()
    runner._evict_cached_agent = lambda _key: None
    entry = SimpleNamespace(session_id="sess-todo")
    attempt = SimpleNamespace(agent=None, future=None, commit_fence=None, wait_started=None, history=None,
                              cleanup_deferred=False)
    hs = SimpleNamespace(total_ceiling_seconds=30)
    plan = SimpleNamespace(approx_tokens=200_000)
    msgs = [m for m in history if m.get("role") in {"user", "assistant", "tool"}]

    with patch("run_agent.AIAgent", side_effect=_capturing_factory):
        await runner._hmwa_hygiene_detached_attempt(
            attempt, hs, plan, history, msgs, "test/model", {"api_key": "test-key"},
            _source(), entry, "key", "quick", 1,
        )

    assert built and results, "the hygiene attempt did not compact through the built agent"
    return results[0]


def _async_db(db):
    return None if db is None else SimpleNamespace(_db=db, get_session=AsyncMock(side_effect=db.get_session))


@pytest.fixture
def todo_db(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("todo-parent", source="telegram")
    pair = _todo_pair("todo-call", _TODOS, 3)
    db.append_message("todo-parent", "assistant", "", tool_calls=pair[0]["tool_calls"])
    db.append_message("todo-parent", "tool", pair[1]["content"], tool_call_id="todo-call", tool_name="todo")
    db.archive_and_compact("todo-parent", [{"role": "user", "content": "Earlier work summarised"}])
    db.end_session("todo-parent", "compression")
    db.create_session("sess-todo", source="telegram", parent_session_id="todo-parent")
    yield db
    db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("compact", [_hygiene_compress, _manual_compress], ids=["hygiene", "legacy"])
async def test_detached_compaction_restores_summarised_tasks_exactly(todo_db, compact):
    built = []
    compressed = await compact(_history(), db=todo_db, built=built)
    _assert_todo_carried(compressed)
    assert built[0]._todo_store.snapshot() == {"todos": _TODOS, "revision": 3}
    assert TodoStore.format_for_injection(built[0]._todo_store) in compressed[-1]["content"]


@pytest.mark.asyncio
async def test_older_result_cannot_override_newer_snapshot_even_when_summarised():
    snapshot = TODO_INJECTION_HEADER + "\n- [>] t1. " + _TASK + " (in_progress)"
    history = [*_history()[:-1], *_todo_pair("old", _REV1, 1),
               {"role": "user", "content": "keep going\n\n" + snapshot}]
    built = []
    compressed = await _manual_compress(history, built=built)
    assert built[0]._todo_store.read() == []  # rendered text is never parsed into items
    assert compressed[-1]["content"] == "keep going\n\n" + snapshot
    assert "Step one" not in compressed[-1]["content"]


@pytest.mark.asyncio
async def test_transcript_structured_result_after_snapshot_is_restored():
    history = [*_history(), dict(_SNAPSHOT_REV3), *_todo_pair("new", _TODOS, 4),
               {"role": "user", "content": "continue"}]
    built = []
    _assert_todo_carried(await _manual_compress(history, built=built))
    assert built[0]._todo_store.snapshot() == {"todos": _TODOS, "revision": 4}


async def _canonical_compress(db, tmp_path, monkeypatch, payload=None, compressed_head=None):
    from contextlib import nullcontext
    from gateway.session_mutation_compress import prepare_compress
    from agent.context_compressor import ContextCompressor

    policy = SimpleNamespace(config=lambda _: {}, safe_mode=False)
    monkeypatch.setattr("gateway.session_policy.restore_policy", lambda _: policy)
    monkeypatch.setattr("gateway.session_policy.policy_scope", lambda *a, **k: nullcontext())
    monkeypatch.setattr(ContextCompressor, "__init__", lambda *a, **k: None)
    tail = _compressed_tail() if compressed_head is None else compressed_head
    tail[0]["_compressed_summary"] = True
    monkeypatch.setattr(ContextCompressor, "compress", lambda *a, **k: json.loads(json.dumps(tail)))
    runner = SimpleNamespace(_resolve_session_agent_runtime=lambda **_: ("test/model", {}))
    authority = SimpleNamespace(db=db, runner=runner, profile_id=str(tmp_path))
    prepared = {"snapshot": {"target": "sess-todo", "receipt": {"policy": {}}}}
    return await prepare_compress(authority, SimpleNamespace(source=_source(), route="logical-session"),
                                  payload or {}, prepared), tail


@pytest.mark.asyncio
@pytest.mark.parametrize("partial", [False, True], ids=["full", "partial"])
async def test_canonical_compress_restores_summarised_tasks(todo_db, tmp_path, monkeypatch, partial):
    for msg in _history():
        todo_db.append_message("sess-todo", msg["role"], msg["content"])
    result, _ = await _canonical_compress(todo_db, tmp_path, monkeypatch,
                                         {"partial": True, "keep_last": 2} if partial else {})
    _assert_todo_carried(result["messages"])
    expected = TodoStore()
    expected.restore(_TODOS, revision=3)
    assert any(msg["content"].endswith(expected.format_for_injection()) for msg in result["messages"])


@pytest.mark.asyncio
@pytest.mark.parametrize("plugin", [False, True], ids=["builtin", "plugin"])
@pytest.mark.parametrize("head_role", ["user", "assistant"], ids=["human-head", "no-human-head"])
async def test_compress_here_keeps_tail_verbatim_with_archived_todos(
    todo_db, tmp_path, monkeypatch, plugin, head_role,
):
    from hermes_cli.partial_compress import split_history_for_partial_compress

    for msg in _history():
        todo_db.append_message("sess-todo", msg["role"], msg["content"])
    before = todo_db.get_messages_as_conversation("sess-todo")
    _, retained = split_history_for_partial_compress(before, 2)
    tail_bytes = json.dumps(retained, ensure_ascii=False).encode("utf-8")
    head = [{"role": "assistant", "content": "Summary", "_compressed_summary": True}]
    if head_role == "user":
        head.append({"role": "user", "content": "Earlier work summarised"})
    if plugin:
        engine = SimpleNamespace(compress=lambda *a, **k: json.loads(json.dumps(head)))
        monkeypatch.setattr("gateway.session_mutation_compress._plugin_engine", lambda *a: engine)
    result, _ = await _canonical_compress(
        todo_db, tmp_path, monkeypatch, {"focus": "here 2"}, compressed_head=head,
    )
    messages = result["messages"]
    assert json.dumps(messages[-len(retained):], ensure_ascii=False).encode("utf-8") == tail_bytes
    _assert_todo_carried(messages[:-len(retained)])
    expected = TodoStore()
    expected.restore(_TODOS, revision=3)
    assert messages[-len(retained) - 1] == {"role": "assistant", "content": ""}
    snapshot_row = messages[-len(retained) - 2]
    if head_role == "user":
        assert snapshot_row["content"] == "Earlier work summarised\n\n" + expected.format_for_injection()
    else:
        assert snapshot_row == {"role": "user", "content": expected.format_for_injection(),
                                "_todo_snapshot_synthetic": True}
    assert todo_db.get_messages_as_conversation("sess-todo") == before


@pytest.mark.asyncio
async def test_compaction_without_any_todo_state_leaves_transcript_unchanged(todo_db, tmp_path, monkeypatch):
    # Keep the target name fixed for the canonical path while supplying a genuinely empty database.
    empty = SessionDB(tmp_path / "empty.db")
    empty.create_session("sess-todo", source="cli")
    for msg in _history():
        empty.append_message("sess-todo", msg["role"], msg["content"])
    try:
        with monkeypatch.context() as scoped:
            result, tail = await _canonical_compress(empty, tmp_path, scoped)
        assert json.dumps(result["messages"]) == json.dumps(tail)
        for compact in (_manual_compress, _hygiene_compress):
            assert await compact(_history()) == _compressed_tail()
    finally:
        empty.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("partial", [False, True], ids=["full", "partial"])
async def test_plugin_compress_restores_summarised_tasks(todo_db, tmp_path, monkeypatch, partial):
    for msg in _history():
        todo_db.append_message("sess-todo", msg["role"], msg["content"])
    engine = SimpleNamespace(compress=lambda *a, **k: _compressed_tail())
    monkeypatch.setattr("gateway.session_mutation_compress._plugin_engine", lambda *a: engine)
    result, _ = await _canonical_compress(todo_db, tmp_path, monkeypatch,
                                         {"partial": True, "keep_last": 2} if partial else {})
    _assert_todo_carried(result["messages"])
    assert result["engine_owner"].context_compressor is engine
    if partial:
        assert result["messages"][-2]["content"] == _history()[-2]["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("plugin", [False, True], ids=["builtin", "plugin"])
@pytest.mark.parametrize("partial", [False, True], ids=["full", "partial"])
async def test_canonical_compress_refuses_growth_after_todo_fold(todo_db, tmp_path, monkeypatch, plugin, partial):
    from agent.model_metadata import estimate_messages_tokens_rough
    from gateway.compaction_todo_hydration import fold_todos_for_compaction

    todos = [{"id": f"t{i}", "content": "archived task " * 250, "status": "pending"} for i in range(8)]
    pair = _todo_pair("large", todos, 4)
    todo_db.append_message("sess-todo", "assistant", "", tool_calls=pair[0]["tool_calls"])
    todo_db.append_message("sess-todo", "tool", pair[1]["content"], tool_call_id="large", tool_name="todo")
    todo_db.archive_and_compact("sess-todo", [])
    for msg in _history():
        todo_db.append_message("sess-todo", msg["role"], msg["content"])
    before = todo_db.get_messages_as_conversation("sess-todo")
    candidate = _compressed_tail()
    assert estimate_messages_tokens_rough(candidate) < estimate_messages_tokens_rough(before)
    folded = fold_todos_for_compaction(before, candidate, todo_db, "sess-todo")
    assert estimate_messages_tokens_rough(folded) > estimate_messages_tokens_rough(before)
    ended = []
    engine = SimpleNamespace(compress=lambda *a, **k: _compressed_tail(),
                             on_session_end=lambda *a: ended.append(a))
    if plugin:
        monkeypatch.setattr("gateway.session_mutation_compress._plugin_engine", lambda *a: engine)
    result, _ = await _canonical_compress(todo_db, tmp_path, monkeypatch,
                                         {"partial": True, "keep_last": 2} if partial else {})
    assert result.get("status") == "noop"
    assert "messages" not in result
    assert todo_db.get_messages_as_conversation("sess-todo") == before
    if plugin:
        assert engine._last_compress_refused_would_grow is True
        assert len(ended) == 1
        assert ended[0] == ("sess-todo", before)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["noop", "compress_error", "fold_error"])
async def test_plugin_todo_fold_preserves_noop_and_failure_cleanup(todo_db, tmp_path, monkeypatch, outcome):
    from gateway import compaction_todo_hydration as hydration

    for msg in _history():
        todo_db.append_message("sess-todo", msg["role"], msg["content"])
    before = todo_db.get_messages_as_conversation("sess-todo")
    events = []

    def compress(messages, **kwargs):
        events.append("compress")
        if outcome == "compress_error":
            raise RuntimeError("compress failed")
        return messages if outcome == "noop" else _compressed_tail()

    def fold(*args):
        events.append("fold")
        raise RuntimeError("fold failed")

    engine = SimpleNamespace(compress=compress, on_session_end=lambda *a: events.append("end"))
    monkeypatch.setattr("gateway.session_mutation_compress._plugin_engine", lambda *a: engine)
    monkeypatch.setattr(hydration, "fold_todos_for_compaction", fold)
    if outcome == "noop":
        result, _ = await _canonical_compress(todo_db, tmp_path, monkeypatch)
        assert result["status"] == "noop"
        assert events == ["compress", "end"]
    else:
        with pytest.raises(RuntimeError, match="fold failed" if outcome == "fold_error" else "compress failed"):
            await _canonical_compress(todo_db, tmp_path, monkeypatch)
        assert events == (["compress", "fold", "end"] if outcome == "fold_error" else ["compress", "end"])
    assert todo_db.get_messages_as_conversation("sess-todo") == before
