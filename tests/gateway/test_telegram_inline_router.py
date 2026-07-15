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


def test_registry_prefix_shorthand_routes_without_explicit_match_entry(tmp_path):
    """`prefix` is documented as shorthand for a type=prefix matcher — it
    must work with no explicit `match` list at all, not just alongside one."""
    tools = [{"id": "bang", "executor": "b", "enabled": True, "prefix": "!"}]
    reg = InlineToolRegistry(_write_registry(tmp_path, tools))
    assert reg.match("!go")["executor"] == "b"
    assert reg.match("go") is None


def test_registry_prefix_shorthand_combines_with_explicit_match(tmp_path):
    """`prefix` folds in alongside explicit `match` entries rather than
    replacing them — either one can route to the tool."""
    tools = [{"id": "t", "executor": "e", "enabled": True, "prefix": "!",
              "match": [{"type": "url", "pattern": r"https?://x\.com/"}]}]
    reg = InlineToolRegistry(_write_registry(tmp_path, tools))
    assert reg.match("!go")["executor"] == "e"
    assert reg.match("see https://x.com/a")["executor"] == "e"
    assert reg.match("neither") is None


def test_registry_prefix_is_escaped_not_treated_as_regex(tmp_path):
    """A literal prefix like "." must not act as a regex wildcard."""
    tools = [{"id": "t", "executor": "e", "enabled": True, "prefix": "."}]
    reg = InlineToolRegistry(_write_registry(tmp_path, tools))
    assert reg.match(".go")["executor"] == "e"
    assert reg.match("Xgo") is None


def test_registry_handles_restricts_to_named_bots(tmp_path):
    """A tool declaring `handles` only matches for those bot usernames."""
    tools = [{"id": "t", "executor": "e", "enabled": True,
              "handles": ["other_bot"], "match": [{"type": "search"}]}]
    reg = InlineToolRegistry(_write_registry(tmp_path, tools))
    assert reg.match("q", bot_username="other_bot") is not None
    assert reg.match("q", bot_username="hermes_bot") is None
    assert reg.match("q") is None  # no bot_username given at all


def test_registry_handles_match_is_case_insensitive_and_ignores_at(tmp_path):
    tools = [{"id": "t", "executor": "e", "enabled": True,
              "handles": ["@Other_Bot"], "match": [{"type": "search"}]}]
    reg = InlineToolRegistry(_write_registry(tmp_path, tools))
    assert reg.match("q", bot_username="other_bot") is not None
    assert reg.match("q", bot_username="@OTHER_BOT") is not None


def test_registry_no_handles_matches_any_bot(tmp_path):
    """Omitting `handles` entirely means the tool matches regardless of
    which bot's router is asking."""
    tools = [{"id": "t", "executor": "e", "enabled": True,
              "match": [{"type": "search"}]}]
    reg = InlineToolRegistry(_write_registry(tmp_path, tools))
    assert reg.match("q", bot_username="any_bot_at_all") is not None
    assert reg.match("q") is not None


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
async def test_dispatch_threads_bot_username_to_registry_handles_filter(tmp_path):
    """The router must pass its own bot_username into the registry's match()
    so a tool's `handles` allowlist is actually enforced during dispatch,
    not just when calling InlineToolRegistry.match() directly."""
    tools = [{"id": "t", "executor": "e", "enabled": True,
              "handles": ["hermes_bot"], "match": [{"type": "search"}]}]
    r = TelegramInlineRouter(bot=None, registry_path=_write_registry(tmp_path, tools))
    r.register_executor("e", lambda cfg, bot: _StubExecutor())

    r.bot_username = "some_other_bot"
    assert await r.dispatch(1, "q") == []

    r.bot_username = "hermes_bot"
    assert await r.dispatch(1, "q") == ["ok"]


class _StubExecutor:
    async def execute(self, user_id, query):
        return ["ok"]


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
