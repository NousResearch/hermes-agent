"""Tests for the tool-side Telegram inline-query dispatch router.

The framework ships dispatch only — registry matching, executor lookup, the
response deadline, and user-space executor discovery. Concrete executors live
in inline_executors/ and are not exercised here.
"""
from __future__ import annotations

import asyncio

import pytest
import yaml

from gateway.platforms.telegram_inline_router import (
    InlineExecutor,
    InlineToolRegistry,
    RESPONSE_DEADLINE,
    TelegramInlineRouter,
)


def _write_registry(tmp_path, tools) -> str:
    p = tmp_path / "inline_tools.yaml"
    p.write_text(yaml.safe_dump({"version": 1, "tools": tools}))
    return str(p)


class _StubExecutor(InlineExecutor):
    def __init__(self, results, delay: float = 0.0):
        self._results = results
        self._delay = delay

    async def execute(self, user_id: int, query: str):
        if self._delay:
            await asyncio.sleep(self._delay)
        return self._results


# ---------------------------------------------------------------------------
# Registry matching
# ---------------------------------------------------------------------------

def test_registry_missing_file_matches_nothing(tmp_path):
    reg = InlineToolRegistry(str(tmp_path / "absent.yaml"))
    assert reg.match("anything") is None


def test_registry_url_match(tmp_path):
    tools = [{"id": "t", "executor": "e", "enabled": True,
              "match": [{"type": "url", "pattern": r"https?://x\.com/"}]}]
    reg = InlineToolRegistry(_write_registry(tmp_path, tools))
    assert reg.match("see https://x.com/a")["executor"] == "e"
    assert reg.match("no url here") is None


def test_registry_prefix_beats_catch_all_by_priority(tmp_path):
    tools = [
        {"id": "catch", "executor": "c", "enabled": True,
         "match": [{"type": "search", "priority": 10}]},
        {"id": "bang", "executor": "b", "enabled": True,
         "match": [{"type": "prefix", "pattern": "!", "priority": 0}]},
    ]
    reg = InlineToolRegistry(_write_registry(tmp_path, tools))
    assert reg.match("!go")["executor"] == "b"     # priority 0 wins
    assert reg.match("plain")["executor"] == "c"   # falls to catch-all


def test_registry_disabled_tool_skipped(tmp_path):
    tools = [{"id": "t", "executor": "e", "enabled": False,
              "match": [{"type": "search"}]}]
    reg = InlineToolRegistry(_write_registry(tmp_path, tools))
    assert reg.match("x") is None


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dispatch_no_match_returns_empty(tmp_path):
    r = TelegramInlineRouter(bot=None, registry_path=str(tmp_path / "none.yaml"))
    assert await r.dispatch(1, "q") == []


@pytest.mark.asyncio
async def test_dispatch_unregistered_executor_returns_empty(tmp_path):
    tools = [{"id": "t", "executor": "missing", "enabled": True,
              "match": [{"type": "search"}]}]
    r = TelegramInlineRouter(bot=None, registry_path=_write_registry(tmp_path, tools))
    assert await r.dispatch(1, "q") == []


@pytest.mark.asyncio
async def test_dispatch_runs_registered_executor(tmp_path):
    tools = [{"id": "t", "executor": "e", "enabled": True,
              "match": [{"type": "search"}]}]
    r = TelegramInlineRouter(bot=None, registry_path=_write_registry(tmp_path, tools))
    r.register_executor("e", lambda cfg, bot: _StubExecutor(["R1"]))
    assert await r.dispatch(7, "q") == ["R1"]


@pytest.mark.asyncio
async def test_dispatch_enforces_response_deadline(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "gateway.platforms.telegram_inline_router.RESPONSE_DEADLINE", 0.05
    )
    tools = [{"id": "t", "executor": "e", "enabled": True,
              "match": [{"type": "search"}]}]
    r = TelegramInlineRouter(bot=None, registry_path=_write_registry(tmp_path, tools))
    r.register_executor("e", lambda cfg, bot: _StubExecutor(["slow"], delay=1.0))
    assert await r.dispatch(1, "q") == []  # cancelled at the deadline


# ---------------------------------------------------------------------------
# Executor discovery
# ---------------------------------------------------------------------------

def test_discover_executors_registers_user_modules(tmp_path, monkeypatch):
    execdir = tmp_path / "inline_executors"
    execdir.mkdir()
    (execdir / "my_exec.py").write_text(
        "def register(router):\n"
        "    router.register_executor('mye', lambda cfg, bot: None)\n"
    )
    # Underscore-prefixed files are skipped.
    (execdir / "_helper.py").write_text("raise RuntimeError('should not load')\n")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    r = TelegramInlineRouter(bot=None, registry_path=str(tmp_path / "none.yaml"))
    r.discover_executors()

    assert "mye" in r._executors
