"""Tests for S7 sandbox substrate, S10 tracing, S9 dispatch wire-in,
and S4 LLM extract fallback.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from unittest import mock

import pytest

from agent_bus.middleware import MiddlewareChain, MiddlewareContext, clear_registry


@pytest.fixture(autouse=True)
def reset_registry():
    clear_registry()
    yield
    clear_registry()


# ================================================================
#  S7 — Sandbox substrate
# ================================================================
class TestVirtualPath:
    def test_translate_workspace(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_THREADS_ROOT", str(tmp_path / "threads"))
        from agent_bus.sandbox.virtual_path import translate_virtual_path
        p = translate_virtual_path("/mnt/user-data/workspace/foo.md", "tid-1")
        assert str(p).endswith("threads/tid-1/user-data/workspace/foo.md")

    def test_translate_skills(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_SKILLS_ROOT", str(tmp_path / "skills"))
        from agent_bus.sandbox.virtual_path import translate_virtual_path
        p = translate_virtual_path("/mnt/skills/public/foo/SKILL.md", "any")
        assert str(p).endswith("skills/public/foo/SKILL.md")

    def test_escape_attempt_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_THREADS_ROOT", str(tmp_path / "threads"))
        from agent_bus.sandbox.virtual_path import translate_virtual_path
        # Ensure the containing directory exists so resolve() doesn't create symlinks
        (tmp_path / "threads" / "tid" / "user-data").mkdir(parents=True, exist_ok=True)
        with pytest.raises(ValueError):
            translate_virtual_path("/mnt/user-data/../../../etc/passwd", "tid")

    def test_unknown_virtual_root_raises(self):
        from agent_bus.sandbox.virtual_path import translate_virtual_path
        with pytest.raises(ValueError):
            translate_virtual_path("/etc/passwd", "tid")

    def test_replace_in_text(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_THREADS_ROOT", str(tmp_path / "threads"))
        from agent_bus.sandbox.virtual_path import replace_virtual_paths_in_text
        out = replace_virtual_paths_in_text(
            "cat /mnt/user-data/workspace/a.txt | head",
            "t1",
        )
        assert "/mnt/user-data" not in out
        assert "threads/t1/user-data" in out


class TestLocalSandbox:
    def test_acquire_creates_dirs(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_THREADS_ROOT", str(tmp_path / "threads"))
        from agent_bus.sandbox.local import LocalSandboxProvider
        p = LocalSandboxProvider()
        sb = p.acquire("tid-1")
        assert sb.workspace_dir.exists()
        assert sb.uploads_dir.exists()
        assert sb.outputs_dir.exists()
        assert sb.id == "local:tid-1"

    def test_write_and_read(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_THREADS_ROOT", str(tmp_path / "threads"))
        from agent_bus.sandbox.local import LocalSandboxProvider
        p = LocalSandboxProvider()
        sb = p.acquire("tid-1")
        sb.write_file("/mnt/user-data/workspace/hello.txt", "world")
        assert sb.read_file("/mnt/user-data/workspace/hello.txt") == "world"

    def test_list_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_THREADS_ROOT", str(tmp_path / "threads"))
        from agent_bus.sandbox.local import LocalSandboxProvider
        p = LocalSandboxProvider()
        sb = p.acquire("tid-1")
        sb.write_file("/mnt/user-data/workspace/a.txt", "x")
        sb.write_file("/mnt/user-data/workspace/b.txt", "y")
        names = sb.list_dir("/mnt/user-data/workspace")
        assert set(names) == {"a.txt", "b.txt"}

    def test_acquire_cached(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_THREADS_ROOT", str(tmp_path / "threads"))
        from agent_bus.sandbox.local import LocalSandboxProvider
        p = LocalSandboxProvider()
        sb1 = p.acquire("tid-cached")
        sb2 = p.acquire("tid-cached")
        assert sb1 is sb2


class TestThreadDataMiddleware:
    def test_before_model_acquires_sandbox(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_THREADS_ROOT", str(tmp_path / "threads"))
        # Force a fresh provider so tests are independent
        from agent_bus.sandbox import local as _local
        _local._default_provider = None

        from agent_bus.middlewares import register_defaults
        register_defaults()
        chain = MiddlewareChain.build()
        ctx = MiddlewareContext(thread_id="tid-mw")
        ctx = chain.run("before_model", ctx)
        assert ctx.metadata.get("sandbox") is not None
        assert ctx.metadata.get("sandbox_id") == "local:tid-mw"
        # Decision recorded
        assert any(
            d["middleware"] == "thread-data" and d["action"] == "acquired"
            for d in ctx.decisions
        )

    def test_no_thread_id_no_op(self):
        from agent_bus.middlewares import register_defaults
        register_defaults()
        chain = MiddlewareChain.build()
        ctx = MiddlewareContext()  # no thread_id
        ctx = chain.run("before_model", ctx)
        assert ctx.metadata.get("sandbox") is None


# ================================================================
#  S10 — Tracing
# ================================================================
class TestTracing:
    def test_spans_recorded_without_providers(self):
        # Reset provider init state so no cached clients leak
        from agent_bus.middlewares import tracing as _t
        _t._reset_providers_for_test()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LANGSMITH_TRACING", None)
            os.environ.pop("LANGFUSE_TRACING", None)
            from agent_bus.middlewares import register_defaults
            register_defaults()
            chain = MiddlewareChain.build()
            ctx = MiddlewareContext(thread_id="t-trace")
            chain.run("before_model", ctx)
            chain.run("after_model", ctx)
            spans = ctx.metadata.get("trace_spans")
            assert spans is not None
            assert len(spans) >= 2
            assert all("duration_ms" in s for s in spans)

    def test_on_session_end_summary(self):
        from agent_bus.middlewares import tracing as _t
        _t._reset_providers_for_test()
        from agent_bus.middlewares import register_defaults
        register_defaults()
        chain = MiddlewareChain.build()
        ctx = MiddlewareContext(thread_id="t-trace-end")
        chain.run("before_model", ctx)
        chain.run("on_session_end", ctx)
        # Summary decision recorded
        assert any(
            d["middleware"] == "tracing" and d["action"] == "summary"
            for d in ctx.decisions
        )


# ================================================================
#  S9 — Dispatch guardrail wire-in
# ================================================================
class TestDispatchGuardrail:
    @pytest.fixture
    def core_with_tmpdb(self, tmp_path, monkeypatch):
        db_path = tmp_path / f"agent_bus_{uuid.uuid4().hex[:8]}.db"
        monkeypatch.setenv("AGENT_BUS_DB_PATH", str(db_path))
        from agent_bus import storage as _storage
        _storage._DB_CONN = None
        from agent_bus import core as _core
        monkeypatch.setattr(_core, "_slack_post_assignment", lambda *a, **k: (None, None))
        monkeypatch.setattr(_core, "_slack_reply", lambda *a, **k: True)
        monkeypatch.setattr(_core, "_notify_agent", lambda *a, **k: None)
        monkeypatch.setattr(_core, "_notify_openclaw", lambda *a, **k: None)
        monkeypatch.setattr(_core, "_notify_hermes_via_slack", lambda *a, **k: None)
        monkeypatch.setattr(_core, "_notify_user_of_outcome", lambda *a, **k: False)
        clear_registry()
        _core._middlewares_registered = False
        yield _core
        _storage._DB_CONN = None
        clear_registry()

    def test_denylist_blocks_dispatch(self, core_with_tmpdb, monkeypatch):
        monkeypatch.setenv("HERMES_GUARDRAIL_MODE", "denylist")
        monkeypatch.setenv("HERMES_TOOL_DENYLIST", "STOP,rm")
        core = core_with_tmpdb
        with pytest.raises(ValueError) as ei:
            core.assign_task(
                from_agent="hermes",
                to_agent="openclaw",
                goal="STOP: terminate everything",
                skip_prior_learnings=True,
            )
        assert "GUARDRAIL_DENY" in str(ei.value)

    def test_guardrail_off_allows_anything(self, core_with_tmpdb, monkeypatch):
        monkeypatch.setenv("HERMES_GUARDRAIL_MODE", "off")
        core = core_with_tmpdb
        t = core.assign_task(
            from_agent="hermes",
            to_agent="openclaw",
            goal="STOP: terminate",
            skip_prior_learnings=True,
        )
        assert t["status"] == "pending"

    def test_middleware_master_off_allows_dispatch(self, core_with_tmpdb, monkeypatch):
        monkeypatch.setenv("HERMES_MIDDLEWARE_CHAIN", "off")
        monkeypatch.setenv("HERMES_TOOL_DENYLIST", "STOP")
        core = core_with_tmpdb
        t = core.assign_task(
            from_agent="hermes",
            to_agent="openclaw",
            goal="STOP: terminate",
            skip_prior_learnings=True,
        )
        assert t["status"] == "pending"

    def test_normal_goal_allowed(self, core_with_tmpdb, monkeypatch):
        monkeypatch.setenv("HERMES_GUARDRAIL_MODE", "denylist")
        monkeypatch.setenv("HERMES_TOOL_DENYLIST", "rm,DROP")
        core = core_with_tmpdb
        t = core.assign_task(
            from_agent="hermes",
            to_agent="openclaw",
            goal="research X and report findings",
            skip_prior_learnings=True,
        )
        assert t["status"] == "pending"


# ================================================================
#  S4 — LLM extractor fallback behavior
# ================================================================
class TestLLMExtractorFallback:
    def test_llm_no_sdk_returns_empty(self):
        from agent_bus.middlewares.memory_extraction import MemoryExtractionMiddleware
        mw = MemoryExtractionMiddleware()
        # If anthropic is not installed, _llm_extract returns []
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            facts = mw._llm_extract("tid", [{"role": "user", "content": "I prefer black coffee"}])
            assert facts == []

    def test_llm_no_api_key_returns_empty(self):
        # anthropic may be installed, but no key → skip
        from agent_bus.middlewares.memory_extraction import MemoryExtractionMiddleware
        mw = MemoryExtractionMiddleware()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            assert mw._llm_extract("tid", [{"role": "user", "content": "hi"}]) == []
