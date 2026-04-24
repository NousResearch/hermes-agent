"""Tests for S4/S5/S6 middlewares:
  - MemoryExtractionMiddleware (se-024)
  - TodoListMiddleware
  - SummarizationMiddleware (se-025)

All three run on top of the S3 middleware chain framework.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from unittest import mock

import pytest

from agent_bus.middleware import (
    MiddlewareChain,
    MiddlewareContext,
    clear_registry,
)


@pytest.fixture(autouse=True)
def reset_registry(monkeypatch, tmp_path):
    # Keep the global memory-auto.json out of tests (inject off unless test
    # explicitly re-enables it).
    monkeypatch.setenv("HERMES_AUTO_MEMORY_PATH", str(tmp_path / "empty-auto.json"))
    monkeypatch.setenv("HERMES_AUTO_MEMORY_INJECT", "off")
    clear_registry()
    yield
    clear_registry()


# ====================================================================
#  S4 — Memory extraction
# ====================================================================
class TestMemoryExtraction:
    def _reg(self):
        from agent_bus.middlewares import register_defaults

        register_defaults()

    def test_enqueue_on_after_model(self, tmp_path):
        self._reg()
        with mock.patch.dict(os.environ, {
            "HERMES_AUTO_MEMORY_PATH": str(tmp_path / "memory-auto.json"),
            "HERMES_AUTO_MEMORY_DEBOUNCE": "1",
        }):
            chain = MiddlewareChain.build()
            ctx = MiddlewareContext(
                thread_id="t-test",
                messages=[
                    {"role": "user", "content": "我每天早上習慣喝黑咖啡"},
                    {"role": "assistant", "content": "OK, noted"},
                ],
            )
            ctx = chain.run("after_model", ctx)
            assert any(d["action"] == "enqueued" for d in ctx.decisions)

    def test_flush_writes_heuristic_facts(self, tmp_path):
        self._reg()
        store_path = tmp_path / "memory-auto.json"
        with mock.patch.dict(os.environ, {
            "HERMES_AUTO_MEMORY_PATH": str(store_path),
            "HERMES_AUTO_MEMORY_DEBOUNCE": "60",  # won't auto-fire
            "HERMES_AUTO_MEMORY_LLM": "off",
        }):
            chain = MiddlewareChain.build()
            ctx = MiddlewareContext(
                thread_id="t-test",
                messages=[
                    {"role": "user", "content": "我每天早上習慣喝黑咖啡"},
                    {"role": "assistant", "content": "OK, noted"},
                ],
            )
            chain.run("after_model", ctx)
            # Force flush via on_session_end
            chain.run("on_session_end", ctx)

            assert store_path.exists()
            import json as _j
            data = _j.loads(store_path.read_text())
            assert len(data["facts"]) >= 1
            assert "咖啡" in data["facts"][0]["content"]

    def test_dedup_of_duplicate_content(self, tmp_path):
        self._reg()
        store_path = tmp_path / "memory-auto.json"
        with mock.patch.dict(os.environ, {
            "HERMES_AUTO_MEMORY_PATH": str(store_path),
            "HERMES_AUTO_MEMORY_DEBOUNCE": "60",
        }):
            chain = MiddlewareChain.build()
            # Run twice with same content
            for _ in range(2):
                ctx = MiddlewareContext(
                    thread_id="t-test",
                    messages=[{"role": "user", "content": "我每天早上習慣喝黑咖啡"}],
                )
                chain.run("after_model", ctx)
                chain.run("on_session_end", ctx)
            import json as _j
            data = _j.loads(store_path.read_text())
            # Dedup means still only 1 fact
            assert len(data["facts"]) == 1

    def test_no_op_when_no_relevant_messages(self, tmp_path):
        self._reg()
        with mock.patch.dict(os.environ, {
            "HERMES_AUTO_MEMORY_PATH": str(tmp_path / "m.json"),
        }):
            chain = MiddlewareChain.build()
            ctx = MiddlewareContext(thread_id="t-test", messages=[
                {"role": "tool", "content": "irrelevant"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
            ])
            ctx = chain.run("after_model", ctx)
            # assistant has tool_calls → excluded from relevant filter
            # no user messages → should not enqueue
            assert not any(d["middleware"] == "memory-extraction" and d["action"] == "enqueued"
                           for d in ctx.decisions)


# ====================================================================
#  S5 — TodoList
# ====================================================================
class TestTodoList:
    def _reg(self):
        from agent_bus.middlewares import register_defaults
        register_defaults()

    def test_write_todos_call_populates_metadata(self):
        self._reg()
        chain = MiddlewareChain.build()
        ctx = MiddlewareContext(messages=[
            {"role": "user", "content": "plan please"},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "c1",
                "name": "write_todos",
                "args": {"todos": [
                    {"content": "Fetch docs", "status": "in_progress", "activeForm": "Fetching docs"},
                    {"content": "Write tests", "status": "pending", "activeForm": "Writing tests"},
                ]}
            }]},
        ])
        ctx = chain.run("after_model", ctx)
        todos = ctx.metadata.get("todos")
        assert todos is not None
        assert len(todos) == 2
        assert todos[0]["status"] == "in_progress"
        assert todos[1]["status"] == "pending"

    def test_invalid_status_normalized_to_pending(self):
        self._reg()
        chain = MiddlewareChain.build()
        ctx = MiddlewareContext(messages=[{
            "role": "assistant", "content": "", "tool_calls": [{
                "id": "c1", "name": "write_todos",
                "args": {"todos": [{"content": "X", "status": "whatever"}]},
            }],
        }])
        ctx = chain.run("after_model", ctx)
        assert ctx.metadata["todos"][0]["status"] == "pending"

    def test_missing_content_dropped(self):
        self._reg()
        chain = MiddlewareChain.build()
        ctx = MiddlewareContext(messages=[{
            "role": "assistant", "content": "", "tool_calls": [{
                "id": "c1", "name": "write_todos",
                "args": {"todos": [
                    {"status": "pending"},  # no content → drop
                    {"content": "Good"},
                ]},
            }],
        }])
        ctx = chain.run("after_model", ctx)
        assert len(ctx.metadata["todos"]) == 1
        assert ctx.metadata["todos"][0]["content"] == "Good"

    def test_json_string_args_parsed(self):
        self._reg()
        chain = MiddlewareChain.build()
        ctx = MiddlewareContext(messages=[{
            "role": "assistant", "content": "", "tool_calls": [{
                "id": "c1", "name": "write_todos",
                "args": '{"todos":[{"content":"StringArg","status":"completed"}]}',
            }],
        }])
        ctx = chain.run("after_model", ctx)
        assert ctx.metadata["todos"][0]["status"] == "completed"

    def test_incomplete_reported_on_session_end(self):
        self._reg()
        chain = MiddlewareChain.build()
        ctx = MiddlewareContext(messages=[{
            "role": "assistant", "content": "", "tool_calls": [{
                "id": "c1", "name": "write_todos",
                "args": {"todos": [
                    {"content": "A", "status": "completed"},
                    {"content": "B", "status": "in_progress"},
                ]},
            }],
        }])
        chain.run("after_model", ctx)
        chain.run("on_session_end", ctx)
        assert any(
            d["middleware"] == "todo-list" and d["action"] == "incomplete-todos"
            for d in ctx.decisions
        )

    def test_unrelated_tool_call_ignored(self):
        self._reg()
        chain = MiddlewareChain.build()
        ctx = MiddlewareContext(messages=[{
            "role": "assistant", "content": "", "tool_calls": [{
                "id": "c1", "name": "wiki_search", "args": {"q": "x"}
            }],
        }])
        ctx = chain.run("after_model", ctx)
        assert "todos" not in ctx.metadata


# ====================================================================
#  S6 — Summarization
# ====================================================================
class TestSummarization:
    def _reg(self):
        from agent_bus.middlewares import register_defaults
        register_defaults()

    def test_no_compression_below_threshold(self):
        self._reg()
        with mock.patch.dict(os.environ, {"HERMES_SUMM_TRIGGER_TOKENS": "100000"}):
            chain = MiddlewareChain.build()
            msgs = [{"role": "user", "content": "hi"}] * 5
            ctx = MiddlewareContext(messages=list(msgs))
            ctx = chain.run("before_model", ctx)
            assert ctx.messages == msgs
            assert not any(
                d["middleware"] == "summarization" and d["action"] == "compressed"
                for d in ctx.decisions
            )

    def test_compression_triggered(self):
        self._reg()
        with mock.patch.dict(os.environ, {
            "HERMES_SUMM_TRIGGER_TOKENS": "100",
            "HERMES_SUMM_KEEP_LAST": "3",
        }):
            chain = MiddlewareChain.build()
            # Build messages totaling > 100 tokens estimate
            big = "x" * 2000  # ~500 tokens each
            msgs = [{"role": "user", "content": big} for _ in range(10)]
            ctx = MiddlewareContext(messages=msgs)
            ctx = chain.run("before_model", ctx)
            # Head compressed; 3 tail kept; 1 summary added
            assert len(ctx.messages) == 4  # 1 summary + 3 tail
            assert ctx.messages[0].get("_summary") is True
            assert ctx.messages[0]["_compressed_count"] == 7
            assert any(
                d["middleware"] == "summarization" and d["action"] == "compressed"
                for d in ctx.decisions
            )

    def test_leading_system_message_preserved(self):
        self._reg()
        with mock.patch.dict(os.environ, {
            "HERMES_SUMM_TRIGGER_TOKENS": "100",
            "HERMES_SUMM_KEEP_LAST": "2",
        }):
            chain = MiddlewareChain.build()
            big = "x" * 2000
            msgs = [
                {"role": "system", "content": "system prompt"},
                *[{"role": "user", "content": big} for _ in range(6)],
            ]
            ctx = MiddlewareContext(messages=msgs)
            ctx = chain.run("before_model", ctx)
            # Expect: [system_orig, summary, last-2-user]
            # head (msgs[:-2]) = [system, user*4]; leading_system = [system];
            # compressible = user*4 → _compressed_count == 4
            assert ctx.messages[0]["content"] == "system prompt"
            assert ctx.messages[1].get("_summary") is True
            assert ctx.messages[1]["_compressed_count"] == 4
            assert len([m for m in ctx.messages if m["role"] == "user"]) == 2

    def test_not_enough_msgs_to_compress(self):
        self._reg()
        with mock.patch.dict(os.environ, {
            "HERMES_SUMM_TRIGGER_TOKENS": "100",
            "HERMES_SUMM_KEEP_LAST": "10",  # keep 10, but only have 5
        }):
            chain = MiddlewareChain.build()
            big = "x" * 2000
            msgs = [{"role": "user", "content": big} for _ in range(5)]
            ctx = MiddlewareContext(messages=list(msgs))
            ctx = chain.run("before_model", ctx)
            assert ctx.messages == msgs
