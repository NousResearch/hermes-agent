"""The runtime cwd scope is ContextVar-backed and restores on every exit."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from agent.runtime_cwd import resolve_agent_cwd, scoped_session_cwd


def test_scoped_session_cwd_restores_after_exception(tmp_path):
    outer = tmp_path / "outer"
    inner = tmp_path / "inner"
    outer.mkdir()
    inner.mkdir()

    with scoped_session_cwd(str(outer)):
        assert resolve_agent_cwd() == outer.resolve()
        with pytest.raises(RuntimeError, match="boom"):
            with scoped_session_cwd(str(inner)):
                assert resolve_agent_cwd() == inner.resolve()
                raise RuntimeError("boom")
        assert resolve_agent_cwd() == outer.resolve()


def test_scoped_session_cwd_isolated_between_threads(tmp_path):
    roots = [tmp_path / "one", tmp_path / "two"]
    for root in roots:
        root.mkdir()
    barrier = threading.Barrier(2)

    def resolve(root: Path):
        with scoped_session_cwd(str(root)):
            barrier.wait(timeout=3)
            return resolve_agent_cwd()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(resolve, roots))

    assert results == [root.resolve() for root in roots]
