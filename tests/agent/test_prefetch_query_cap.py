"""Tests for the memory-prefetch query-length cap (HERMES_PREFETCH_QUERY_MAX_CHARS).

A large "user message" (e.g. a host that packs its whole system prompt + tool
catalogue into the prompt text, arriving as a 50-100KB blob) is bounded before it
is handed to the memory backend's prefetch_all(), so embedding/hybrid search isn't
stalled by boilerplate. The env var tunes the cap; 0 disables it.

These tests call the REAL production helper ``agent.turn_context._bound_prefetch_query``.
"""
import importlib

import pytest

from agent.turn_context import _bound_prefetch_query


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("HERMES_PREFETCH_QUERY_MAX_CHARS", raising=False)
    yield


def test_long_query_truncated_to_default_cap():
    big = "x" * 60000
    out = _bound_prefetch_query(big)
    assert len(out) == 1500
    assert out == big[:1500]


def test_short_query_unchanged():
    short = "what is the weather where this machine is"
    assert _bound_prefetch_query(short) == short


def test_env_override_changes_cap(monkeypatch):
    monkeypatch.setenv("HERMES_PREFETCH_QUERY_MAX_CHARS", "10")
    assert _bound_prefetch_query("y" * 500) == "y" * 10


def test_zero_disables_cap(monkeypatch):
    monkeypatch.setenv("HERMES_PREFETCH_QUERY_MAX_CHARS", "0")
    big = "z" * 9000
    assert _bound_prefetch_query(big) == big  # unbounded when disabled


def test_malformed_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("HERMES_PREFETCH_QUERY_MAX_CHARS", "not-an-int")
    big = "w" * 4000
    out = _bound_prefetch_query(big)
    assert len(out) == 1500  # bad value -> default cap, not a crash


def test_empty_query_unchanged():
    assert _bound_prefetch_query("") == ""
