"""Gateway compaction paths must carry the session's todo list across the boundary.

Both gateway compaction paths (session hygiene and manual ``/compress``) compact with a freshly built
``AIAgent`` whose ``TodoStore`` starts empty. A live turn hydrates that store from the transcript before
it compacts; these detached agents did not, so ``_fold_todo_snapshot`` had nothing to re-inject and the
list vanished once the todo tool rows were summarised away.
"""

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
from tools.todo_tool import TODO_INJECTION_HEADER

_TASK = "Ship the parser fix before the release cut"


def _history() -> list[dict]:
    todo_args = json.dumps({"todos": [{"id": "t1", "content": _TASK, "status": "in_progress"}]})
    todo_result = json.dumps({"todos": [{"id": "t1", "content": _TASK, "status": "in_progress"}], "revision": 1})
    rows = [
        {"role": "user", "content": "plan the release " + "x" * 400},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call-todo", "type": "function", "function": {"name": "todo", "arguments": todo_args}}]},
        {"role": "tool", "tool_call_id": "call-todo", "content": todo_result},
        {"role": "assistant", "content": "Tracking it. " + "y" * 400},
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


async def _manual_compress(history: list, tail=None) -> list:
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")})
    entry = SessionEntry(session_key=build_session_key(_source()), session_id="sess-todo",
                         created_at=datetime.now(), updated_at=datetime.now(),
                         platform=Platform.TELEGRAM, chat_type="dm")
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = entry
    runner.session_store.load_transcript.return_value = history
    runner._session_db = None
    runner._persist_manual_compression = AsyncMock()
    built: list = []

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
async def test_manual_compress_carries_todo_list_from_history():
    _assert_todo_carried(await _manual_compress(_history()))


@pytest.mark.asyncio
async def test_manual_compress_keeps_newer_folded_snapshot_over_older_head_todo():
    # Compaction 1 kept a small rev-1 todo pair in the protected head and folded the newer rev-3 list in as a
    # snapshot. Replaying rev 1 would make the store authoritative and strip the rev-3 snapshot.
    history = [{"role": "user", "content": "plan it"}, *_todo_pair("c1", _REV1, 1),
               {"role": "user", "content": "[CONTEXT COMPACTION] t1 done; t3 added"},
               {"role": "assistant", "content": "ok"}, dict(_SNAPSHOT_REV3), {"role": "assistant", "content": "working"},
               {"role": "user", "content": "more"}]
    text = "\n".join(str(m.get("content") or "") for m in await _manual_compress(history, tail=history[3:]))
    assert "Step three" in text
    assert "Step one" not in text


@pytest.mark.asyncio
async def test_manual_compress_keeps_folded_snapshot_the_summariser_drops():
    # The rev-3 snapshot is the newest todo state and no todo call follows it. When the compressor summarises
    # the snapshot row away, the boundary must still re-fold the active task.
    history = [{"role": "user", "content": "plan it"}, *_todo_pair("c1", _REV1, 1),
               {"role": "user", "content": "[CONTEXT COMPACTION] t1 done; t3 added"},
               {"role": "assistant", "content": "ok"}, dict(_SNAPSHOT_REV3), {"role": "assistant", "content": "working"},
               {"role": "user", "content": "more"}]
    tail = [*history[:3], {"role": "user", "content": "[CONTEXT COMPACTION] t3 in progress"},
            {"role": "assistant", "content": "ok"}, {"role": "user", "content": "more"}]
    compressed = await _manual_compress(history, tail=tail)
    text = "\n".join(str(m.get("content") or "") for m in compressed)
    # The protected head still holds the rev-1 tool row; the re-folded list must not revive it.
    folded = text.rsplit(TODO_INJECTION_HEADER, 1)[-1]
    assert "Step three" in folded
    assert "Step one" not in folded
    assert _SNAPSHOT_REV3["content"] not in [m.get("content") for m in compressed]


def test_hydration_restores_nested_snapshot_items():
    from gateway.compaction_todo_hydration import hydrate_todos_for_compaction

    snapshot = {"role": "user", "content": [{"type": "text", "text": "go"}, {"type": "text", "text": (
        TODO_INJECTION_HEADER + "\n- [x] p1. Parent. task (completed)\n  - [ ] k1. Child task (pending)\n"
        "- [>] t2. Other (in_progress)\n\n[Skills pruned during compression — reload before acting on these tasks]"
        "\n- some-skill")}]}
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        agent = run_agent.AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model",
                                  quiet_mode=True, skip_context_files=True, skip_memory=True)
    store = hydrate_todos_for_compaction(agent, [*_todo_pair("c1", _REV1, 1), snapshot])._todo_store
    assert store.read() == [{"id": "p1", "content": "Parent. task", "status": "completed"},
                            {"id": "k1", "content": "Child task", "status": "pending", "parent": "p1"},
                            {"id": "t2", "content": "Other", "status": "in_progress"}]


def test_hydration_keeps_blank_lines_inside_snapshot_items():
    # TodoStore accepts multi-line content with blank lines; the folded snapshot must round-trip it whole
    # (status and later items included), stopping only at the reload notice the fold appends.
    from gateway.compaction_todo_hydration import hydrate_todos_for_compaction
    from tools.todo_tool import TodoStore

    todos = [{"id": "t1", "content": "Prepare release\n\nVerify signatures", "status": "in_progress"},
             {"id": "t2", "content": "Publish release", "status": "pending"}]
    source = TodoStore()
    source.write(todos)
    folded = source.format_for_injection()
    snapshot = {"role": "user", "content": "go\n\n" + folded + "\n\n[Skills pruned during compression — reload "
                "before acting on these tasks]\nThe task list above crossed the boundary."}
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        agent = run_agent.AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model",
                                  quiet_mode=True, skip_context_files=True, skip_memory=True)
    store = hydrate_todos_for_compaction(agent, [*_todo_pair("c1", _REV1, 1), snapshot])._todo_store
    assert store.read() == todos
    assert store.format_for_injection() == folded


def test_hydration_still_replays_todo_results_after_the_snapshot():
    from gateway.compaction_todo_hydration import hydrate_todos_for_compaction

    rev4 = [{"id": "t4", "content": "Step four", "status": "in_progress"}]
    history = [{"role": "user", "content": "plan it"}, *_todo_pair("c1", _REV1, 1), dict(_SNAPSHOT_REV3),
               *_todo_pair("c4", rev4, 4), {"role": "user", "content": "more"}]
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        agent = run_agent.AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model",
                                  quiet_mode=True, skip_context_files=True, skip_memory=True)
    assert hydrate_todos_for_compaction(agent, history)._todo_store.snapshot() == {"todos": rev4, "revision": 4}


@pytest.mark.asyncio
async def test_hygiene_compaction_carries_todo_list_from_history():
    from gateway.run import GatewayRunner

    history = _history()
    runner = object.__new__(GatewayRunner)
    runner._session_db = None
    built: list = []
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
    _assert_todo_carried(results[0])


def _bare_agent():
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        return run_agent.AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model",
                                 quiet_mode=True, skip_context_files=True, skip_memory=True)


@pytest.mark.parametrize("todos", [
    [{"id": "t1", "content": "Release\n\n- [ ] 1. Verify signatures", "status": "in_progress"}],
    [{"id": "p1", "content": "Cut the release\n- [x] 2. Tag it (completed)\n\n- [>] 3. Sign it", "status": "in_progress"},
     {"id": "k1", "content": "Build\n  - [ ] k2. Notarize (pending)", "parent": "p1", "status": "pending"},
     {"id": "g1", "content": "Upload\n\n\nthe artifacts", "parent": "k1", "status": "pending"},
     {"id": "k3", "content": "Check (in_progress)\n- [~] 9. Skip", "parent": "p1", "status": "cancelled"},
     {"id": "g2", "content": "Mirror", "parent": "k3", "status": "pending"},
     {"id": "t2", "content": "Publish (pending)\nnotes", "status": "pending"}],
], ids=["checkbox-line", "nested-multiline"])
def test_hydration_round_trips_the_rendered_snapshot(todos):
    from gateway.compaction_todo_hydration import hydrate_todos_for_compaction
    from tools.todo_tool import TodoStore

    source = TodoStore()
    source.write(todos)
    assert source.read() == todos
    snapshot = {"role": "user", "content": "go\n\n" + source.format_for_injection()}
    store = hydrate_todos_for_compaction(_bare_agent(), [*_todo_pair("c1", _REV1, 1), snapshot])._todo_store
    assert store.read() == todos


@pytest.mark.asyncio
async def test_manual_compress_leaves_an_unparseable_snapshot_as_it_is():
    # The status does not match the marker, so this is not the renderer's output: restoring items from it
    # would be a guess. Hydration restores nothing and the boundary keeps the snapshot row unchanged.
    from gateway.compaction_todo_hydration import hydrate_todos_for_compaction

    snapshot = {"role": "user", "content": "keep going\n\n" + TODO_INJECTION_HEADER + "\n- [>] t3. Step three (pending)"}
    history = [{"role": "user", "content": "plan it"}, *_todo_pair("c1", _REV1, 1),
               {"role": "user", "content": "[CONTEXT COMPACTION] t1 done; t3 added"},
               {"role": "assistant", "content": "ok"}, snapshot, {"role": "assistant", "content": "working"},
               {"role": "user", "content": "more"}]
    assert not hydrate_todos_for_compaction(_bare_agent(), history)._todo_store.has_items()
    compressed = await _manual_compress(history, tail=history[3:])
    assert snapshot["content"] in [m.get("content") for m in compressed]
