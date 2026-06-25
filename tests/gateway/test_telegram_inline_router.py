"""Tests for the tool-side Telegram inline-query dispatch router.

Covers the dispatch surface this PR ships — registry matching, routing
decisions, and user-space executor discovery. The execution phase (running a
concrete executor to produce results) is not part of this PR and is not
exercised here.
"""
from __future__ import annotations

import pytest
import yaml

from gateway.platforms.telegram_inline_router import (
    InlineToolRegistry,
    TelegramInlineRouter,
)


def _write_registry(tmp_path, tools) -> str:
    p = tmp_path / "inline_tools.yaml"
    p.write_text(yaml.safe_dump({"version": 1, "tools": tools}))
    return str(p)


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
