"""Tests for agent/context_file_io.py — bounded reads and LRU cache."""

import asyncio
from pathlib import Path

import pytest

from agent.context_file_io import clear_context_file_cache, read_many_sync, read_text_async, read_text_sync


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_context_file_cache()
    yield
    clear_context_file_cache()


def test_read_text_sync_roundtrip(tmp_path: Path):
    p = tmp_path / "ctx.md"
    p.write_text("alpha", encoding="utf-8")
    assert read_text_sync(p) == "alpha"
    assert read_text_sync(p) == "alpha"


def test_read_text_sync_missing(tmp_path: Path):
    assert read_text_sync(tmp_path / "nope.md") == ""


def test_cache_invalidates_on_mtime_change(tmp_path: Path):
    p = tmp_path / "mut.md"
    p.write_text("v1", encoding="utf-8")
    assert read_text_sync(p) == "v1"
    p.write_text("v2", encoding="utf-8")
    assert read_text_sync(p) == "v2"


def test_read_many_sync_order(tmp_path: Path):
    a = tmp_path / "a.mdc"
    b = tmp_path / "b.mdc"
    a.write_text("A", encoding="utf-8")
    b.write_text("B", encoding="utf-8")
    bodies = read_many_sync([a, b])
    assert bodies == ["A", "B"]


@pytest.mark.asyncio
async def test_read_text_async_roundtrip(tmp_path: Path):
    p = tmp_path / "async.md"
    p.write_text("beta", encoding="utf-8")
    assert await read_text_async(p) == "beta"


def test_read_text_sync_from_running_loop_offloads(tmp_path: Path):
    p = tmp_path / "loop.md"
    p.write_text("gamma", encoding="utf-8")

    async def _runner():
        return read_text_sync(p)

    assert asyncio.run(_runner()) == "gamma"
