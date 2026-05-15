"""Tests for gateway.agent_context — per-turn HERMES_HOME override.

These tests cover the contextvar plumbing and verify that
``hermes_constants.get_hermes_home()`` consults it before falling back
to the env var.  Critical for the multi-agent gateway where a single
process serves multiple profiles in parallel.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import threading
from pathlib import Path

import pytest

from gateway.agent_context import (
    agent_home_scope,
    current_agent_home,
    reset_agent_home,
)
from hermes_constants import get_hermes_home


class TestAgentContextVar:
    """ContextVar primitive — set / read / reset."""

    def test_default_is_none(self):
        """Without any scope, current_agent_home returns None."""
        reset_agent_home()
        assert current_agent_home() is None

    def test_scope_sets_and_restores(self, tmp_path: Path):
        """agent_home_scope sets inside, restores on exit."""
        reset_agent_home()
        assert current_agent_home() is None
        with agent_home_scope(tmp_path) as h:
            assert h == tmp_path
            assert current_agent_home() == tmp_path
        assert current_agent_home() is None

    def test_scope_nesting(self, tmp_path: Path):
        """Nested scopes save/restore the previous value (orchestrator → sub-agent)."""
        reset_agent_home()
        outer = tmp_path / "outer"
        inner = tmp_path / "inner"
        with agent_home_scope(outer):
            assert current_agent_home() == outer
            with agent_home_scope(inner):
                assert current_agent_home() == inner
            assert current_agent_home() == outer
        assert current_agent_home() is None

    def test_scope_restores_on_exception(self, tmp_path: Path):
        """The previous value is restored even when the block raises."""
        reset_agent_home()
        with pytest.raises(RuntimeError):
            with agent_home_scope(tmp_path):
                assert current_agent_home() == tmp_path
                raise RuntimeError("boom")
        assert current_agent_home() is None

    def test_string_coerced_to_path(self, tmp_path: Path):
        """Passing a str is accepted and round-tripped as a Path."""
        reset_agent_home()
        with agent_home_scope(tmp_path) as h:
            assert isinstance(h, Path)
            assert current_agent_home() == tmp_path


class TestGetHermesHomeIntegration:
    """get_hermes_home() consults the contextvar before the env var."""

    def test_contextvar_overrides_env(self, tmp_path: Path, monkeypatch):
        """ContextVar wins over HERMES_HOME env var."""
        reset_agent_home()
        env_path = tmp_path / "env"
        ctx_path = tmp_path / "ctx"
        env_path.mkdir()
        ctx_path.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(env_path))
        assert get_hermes_home() == env_path
        with agent_home_scope(ctx_path):
            assert get_hermes_home() == ctx_path
        assert get_hermes_home() == env_path

    def test_no_override_falls_back_to_env(self, tmp_path: Path, monkeypatch):
        """When the contextvar is unset, env var is honoured (legacy path)."""
        reset_agent_home()
        env_path = tmp_path / "env"
        env_path.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(env_path))
        assert get_hermes_home() == env_path

    def test_no_override_no_env_falls_back_to_home(self, tmp_path: Path, monkeypatch):
        """With neither contextvar nor env, falls back to ~/.hermes."""
        reset_agent_home()
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        assert get_hermes_home() == tmp_path / ".hermes"


class TestExecutorPropagation:
    """ContextVar must propagate through copy_context() (which the gateway uses)."""

    def test_copy_context_carries_value(self, tmp_path: Path):
        """copy_context() snapshots the current value; running a callable in
        that context sees the captured agent home — even when the executor
        thread itself doesn't have it set."""
        reset_agent_home()
        target = tmp_path / "captured"

        def _worker() -> Path | None:
            return current_agent_home()

        with agent_home_scope(target):
            ctx = contextvars.copy_context()
        # Outside the scope, the contextvar is back to default in this thread
        assert current_agent_home() is None
        # But ctx.run sees the captured value
        result = ctx.run(_worker)
        assert result == target

    def test_to_thread_inherits_via_copy_context(self, tmp_path: Path):
        """asyncio.to_thread inherits the contextvar from the awaiting coroutine."""
        reset_agent_home()
        target = tmp_path / "async"

        async def _main() -> Path | None:
            with agent_home_scope(target):
                return await asyncio.to_thread(current_agent_home)

        result = asyncio.run(_main())
        assert result == target

    def test_bare_thread_does_not_inherit(self, tmp_path: Path):
        """Raw threading.Thread does NOT inherit contextvars.

        This documents the behaviour callers must work around (use
        copy_context() or set the contextvar inside the thread).
        """
        reset_agent_home()
        target = tmp_path / "raw-thread"
        captured: list[Path | None] = []

        def _worker():
            captured.append(current_agent_home())

        with agent_home_scope(target):
            t = threading.Thread(target=_worker)
            t.start()
            t.join()
        # Raw thread did not see the parent's contextvar
        assert captured == [None]

    def test_executor_with_copy_context_inherits(self, tmp_path: Path):
        """ThreadPoolExecutor.submit + copy_context() propagates correctly."""
        reset_agent_home()
        target = tmp_path / "pool"

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            with agent_home_scope(target):
                ctx = contextvars.copy_context()
                future = pool.submit(ctx.run, current_agent_home)
            assert future.result() == target
