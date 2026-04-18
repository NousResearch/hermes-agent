"""Unit tests for ``gateway.subagent_topic_router``.

Phase A3 coverage:

* Pure helpers — ``_clean_for_topic``, ``_summarize``, ``_derive_topic_name``,
  ``_format_event``.
* State I/O — ``_load_state``/``_save_state`` atomic write, malformed-file
  fallback, chmod, parent-dir auto-create.
* ``SubagentTopicRouter`` routing — non-Telegram noop, blocklist short-circuit,
  lazy-create success + failure classification, state updates, exception
  swallowing.
* Backpressure — ``asyncio.sleep`` gets called once the rate window is full.
* Singleton accessor — ``get_subagent_topic_router()`` returns the same
  instance across calls.

All tests stay fully offline: Telegram ``_run_topic_op`` / ``_load_telegram_token``
get monkey-patched, adapter ``.send`` is an ``AsyncMock``.  Each test is
independent and completes in well under a second.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import stat
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.subagent_topic_router import (
    SubagentTopicRouter,
    _clean_for_topic,
    _derive_topic_name,
    _format_event,
    _load_state,
    _save_state,
    _summarize,
    get_subagent_topic_router,
)
import gateway.subagent_topic_router as router_mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_state_path(tmp_path: Path) -> Path:
    """A state-file path in a nested dir — forces mkdir code path."""
    return tmp_path / "state" / "subagent_topics.json"


@pytest.fixture
def router(tmp_state_path: Path) -> SubagentTopicRouter:
    return SubagentTopicRouter(state_path=tmp_state_path)


@pytest.fixture
def telegram_source() -> SimpleNamespace:
    return SimpleNamespace(
        platform=Platform.TELEGRAM,
        chat_id="-100123456",
        thread_id="77",
        user_id="42",
    )


@pytest.fixture
def non_telegram_source() -> SimpleNamespace:
    return SimpleNamespace(
        platform=Platform.DISCORD if hasattr(Platform, "DISCORD") else "discord",
        chat_id="-100",
        thread_id=None,
        user_id="1",
    )


@pytest.fixture
def adapter() -> MagicMock:
    mock = MagicMock(name="adapter")
    mock.send = AsyncMock(
        return_value=SimpleNamespace(success=True, message_id="m1")
    )
    return mock


@pytest.fixture
def patched_topic_tool(monkeypatch: pytest.MonkeyPatch):
    """Monkeypatch ``tools.telegram_topic_tool`` symbols.

    The router imports these lazily inside ``lazy_create_topic``:

        from tools.telegram_topic_tool import (
            _classify_error,
            _load_telegram_token,
            _run_topic_op,
        )

    We patch the names on the real module so the import keeps working
    and the router picks up our stand-ins.
    """
    import tools.telegram_topic_tool as tgt

    load_token = MagicMock(return_value=("fake-token", None))
    # The default run_op returns success; individual tests replace it.
    run_op = AsyncMock(
        return_value={
            "success": True,
            "thread_id": "9999",
            "name": "SA test",
        }
    )
    classify = tgt._classify_error  # real impl is fine

    monkeypatch.setattr(tgt, "_load_telegram_token", load_token)
    monkeypatch.setattr(tgt, "_run_topic_op", run_op)
    monkeypatch.setattr(tgt, "_classify_error", classify)
    return SimpleNamespace(load_token=load_token, run_op=run_op, classify=classify)


def _run(coro):
    """Run an ``async`` coroutine on a fresh loop (for sync tests)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# A. Pure helpers — no I/O, no mocks
# ---------------------------------------------------------------------------


class TestCleanForTopic:
    def test_strips_control_chars(self):
        assert _clean_for_topic("hi\x00\x01\x1fworld") == "hi world"

    def test_strips_markdown(self):
        # Markdown chars are deleted (not replaced with space), so adjacent
        # words collapse with a single separating space from the original.
        out = _clean_for_topic("*bold* `code` #h _i_ [l]")
        assert "*" not in out and "`" not in out and "#" not in out
        assert "_" not in out and "[" not in out and "]" not in out
        assert "bold" in out and "code" in out

    def test_strips_surrounding_ws(self):
        assert _clean_for_topic("   hello   ") == "hello"

    def test_empty_input(self):
        assert _clean_for_topic("") == ""


class TestSummarize:
    def test_first_line_only(self):
        out = _summarize("line one\nline two")
        assert out == "line one"

    def test_first_sentence_cjk(self):
        out = _summarize("查詢天氣。第二句不要")
        assert out == "查詢天氣"

    def test_truncates_at_max(self):
        text = "a" * 50
        out = _summarize(text, max_chars=10)
        assert out.endswith("…")
        assert len(out) <= 11  # 10 chars + ellipsis

    def test_empty_returns_empty(self):
        assert _summarize("") == ""


class TestDeriveTopicName:
    def test_short_goal(self):
        name = _derive_topic_name("查詢天氣", "abcd1234efgh")
        assert name == "SA 查詢天氣"

    def test_long_goal_truncates(self):
        name = _derive_topic_name("a" * 50, "sid12345")
        assert name.startswith("SA ")
        assert name.endswith("…")
        # "SA " + 20-char summary = "SA " + 19 chars + "…"
        assert len(name) <= _TOPIC_NAME_MAX_REF

    def test_empty_goal_fallback(self):
        name = _derive_topic_name("", "abcd1234efghijk")
        assert re.match(r"^SA abcd1234 · \d{2}:\d{2}$", name), name

    def test_none_goal_fallback(self):
        name = _derive_topic_name(None, "xyzw9999aaaa")
        assert re.match(r"^SA xyzw9999 · \d{2}:\d{2}$", name), name

    def test_short_goal_falls_back_after_cleaning(self):
        # "ab" only has 2 chars after cleaning → fallback
        name = _derive_topic_name("ab", "shortsid12345")
        assert re.match(r"^SA shortsid · \d{2}:\d{2}$", name), name

    def test_whitespace_only_goal_fallback(self):
        name = _derive_topic_name("   \n\t  ", "hello123abc")
        assert re.match(r"^SA hello123 · \d{2}:\d{2}$", name), name

    def test_markdown_stripped_in_name(self):
        name = _derive_topic_name("*hello* `world`", "sid12345")
        # "*hello* `world`" → cleaned "hello  world" (spaces preserved)
        assert "*" not in name and "`" not in name
        assert name.startswith("SA ")

    def test_control_chars_stripped(self):
        name = _derive_topic_name("fo\x00o\x01ba\x02r", "sid12345")
        assert "\x00" not in name
        assert "\x01" not in name
        assert "\x02" not in name
        assert name.startswith("SA ")

    def test_respects_64_char_cap(self):
        # Extremely long goal on a single line
        name = _derive_topic_name("x" * 500, "sid12345")
        assert len(name) <= 64

    def test_splits_on_newline(self):
        name = _derive_topic_name("first line\nsecond line never reached", "sid1234")
        assert "second" not in name
        assert "first line" in name

    def test_splits_on_sentence_terminator(self):
        name = _derive_topic_name("Do the thing. Then the other.", "sid1234")
        assert "Then" not in name


# Hard cap referenced for clarity; keep in sync with router._TOPIC_NAME_MAX
_TOPIC_NAME_MAX_REF = 64


class TestBuildTopicLink:
    def test_private_supergroup_builds_link(self):
        from gateway.subagent_topic_router import _build_topic_link
        assert (
            _build_topic_link("-1003837358001", "2254")
            == "https://t.me/c/3837358001/2254"
        )

    def test_thread_id_as_int_string(self):
        from gateway.subagent_topic_router import _build_topic_link
        assert _build_topic_link("-1001234567890", "42") == "https://t.me/c/1234567890/42"

    def test_non_private_chat_returns_none(self):
        from gateway.subagent_topic_router import _build_topic_link
        # DM / public chats don't have the -100 prefix.
        assert _build_topic_link("123456", "42") is None
        assert _build_topic_link("-123", "42") is None

    def test_malformed_chat_id_returns_none(self):
        from gateway.subagent_topic_router import _build_topic_link
        assert _build_topic_link("", "42") is None
        assert _build_topic_link("-100abc", "42") is None

    def test_malformed_thread_id_returns_none(self):
        from gateway.subagent_topic_router import _build_topic_link
        assert _build_topic_link("-1001234567890", "") is None
        assert _build_topic_link("-1001234567890", "abc") is None
        assert _build_topic_link("-1001234567890", "0") is None
        assert _build_topic_link("-1001234567890", "-5") is None


class TestFormatEvent:
    def test_start_with_preview(self):
        assert _format_event("subagent.start", None, "begin task") == "🔀 Started: begin task"

    def test_start_with_tool_name_only(self):
        assert _format_event("subagent.start", "search", None) == "🔀 Started: search"

    def test_start_empty(self):
        # Still emits a non-empty prefix.
        assert _format_event("subagent.start", None, None).startswith("🔀 Started:")

    def test_thinking_formats(self):
        out = _format_event("subagent.thinking", None, "hmm, let me think")
        assert out.startswith("💭 ")
        assert "let me think" in out

    def test_thinking_truncates_at_200(self):
        long_preview = "x" * 500
        out = _format_event("subagent.thinking", None, long_preview)
        # "💭 " prefix + up-to-200 chars.
        # Remove leading emoji + space and count.
        body = out[len("💭 "):]
        assert len(body) == 200

    def test_tool_uses_emoji_and_name(self, monkeypatch: pytest.MonkeyPatch):
        # The router imports lazily; patch on the actual module.
        import agent.display as display
        monkeypatch.setattr(display, "get_tool_emoji", lambda name, default="⚙️": "🔧")
        out = _format_event("subagent.tool", "terminal", 'ls -la')
        assert out.startswith("🔧 terminal")
        assert '"ls -la"' in out

    def test_tool_without_preview(self, monkeypatch: pytest.MonkeyPatch):
        import agent.display as display
        monkeypatch.setattr(display, "get_tool_emoji", lambda name, default="⚙️": "🔧")
        out = _format_event("subagent.tool", "terminal", None)
        assert out == "🔧 terminal"

    def test_tool_emoji_fallback_on_import_error(self, monkeypatch: pytest.MonkeyPatch):
        # Blow up the display module lookup to force the except branch.
        import agent.display as display

        def raising_get(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(display, "get_tool_emoji", raising_get)
        out = _format_event("subagent.tool", "terminal", "echo hi")
        assert out.startswith("⚙️ terminal")

    def test_tool_preview_truncates_at_60(self, monkeypatch: pytest.MonkeyPatch):
        import agent.display as display
        monkeypatch.setattr(display, "get_tool_emoji", lambda name, default="⚙️": "⚙️")
        out = _format_event("subagent.tool", "terminal", "y" * 500)
        # The quoted snippet body is capped at 60 chars between quotes.
        m = re.search(r'"(.*)"$', out)
        assert m is not None
        assert len(m.group(1)) == 60

    def test_progress(self):
        assert _format_event("subagent.progress", None, "step 2/5") == "⏳ step 2/5"

    def test_complete_formats(self):
        assert _format_event("subagent.complete", None, "all good").startswith("✅ Done:")

    def test_complete_empty_preview(self):
        out = _format_event("subagent.complete", None, None)
        assert out.startswith("✅ Done:")

    def test_complete_truncates_at_300(self):
        out = _format_event("subagent.complete", None, "z" * 500)
        body = out[len("✅ Done: "):]
        assert len(body) == 300

    def test_unknown_event_falls_through(self):
        out = _format_event("subagent.weird", "tool", "stuff")
        assert "subagent.weird" in out
        assert out.startswith("· ")


# ---------------------------------------------------------------------------
# B. State I/O
# ---------------------------------------------------------------------------


class TestLoadState:
    def test_missing_file_returns_default(self, tmp_state_path: Path):
        data = _load_state(tmp_state_path)
        assert data == {"version": 1, "topics": {}}

    def test_malformed_json_returns_default(self, tmp_state_path: Path):
        tmp_state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_state_path.write_text("not json {{{", encoding="utf-8")
        data = _load_state(tmp_state_path)
        assert data == {"version": 1, "topics": {}}

    def test_not_a_dict_returns_default(self, tmp_state_path: Path):
        tmp_state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_state_path.write_text("[]", encoding="utf-8")
        data = _load_state(tmp_state_path)
        assert data == {"version": 1, "topics": {}}

    def test_string_payload_returns_default(self, tmp_state_path: Path):
        tmp_state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_state_path.write_text('"just a string"', encoding="utf-8")
        data = _load_state(tmp_state_path)
        assert data == {"version": 1, "topics": {}}

    def test_topics_not_dict_repaired(self, tmp_state_path: Path):
        tmp_state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_state_path.write_text(
            json.dumps({"version": 1, "topics": ["not", "a", "dict"]}),
            encoding="utf-8",
        )
        data = _load_state(tmp_state_path)
        assert data["topics"] == {}

    def test_version_defaulted_when_missing(self, tmp_state_path: Path):
        tmp_state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_state_path.write_text(json.dumps({"topics": {}}), encoding="utf-8")
        data = _load_state(tmp_state_path)
        assert data["version"] == 1


class TestSaveState:
    def test_roundtrip(self, tmp_state_path: Path):
        payload = {
            "version": 1,
            "topics": {
                "abc": {
                    "chat_id": "-100",
                    "thread_id": "7",
                    "topic_name": "SA hello",
                    "created_ts": 1.0,
                    "last_message_ts": 2.0,
                    "parent_chat_id": "-100",
                    "parent_thread_id": None,
                }
            },
        }
        _save_state(tmp_state_path, payload)
        loaded = _load_state(tmp_state_path)
        assert loaded == payload

    def test_creates_parent_dir(self, tmp_path: Path):
        target = tmp_path / "deep" / "nested" / "path" / "state.json"
        assert not target.parent.exists()
        _save_state(target, {"version": 1, "topics": {}})
        assert target.exists()

    def test_no_tmp_leftovers(self, tmp_state_path: Path):
        _save_state(tmp_state_path, {"version": 1, "topics": {"a": {}}})
        # The directory should only contain the final file.
        siblings = list(tmp_state_path.parent.iterdir())
        tmp_files = [p for p in siblings if p.suffix == ".tmp" or ".tmp" in p.name]
        assert tmp_files == []
        assert siblings == [tmp_state_path]

    @pytest.mark.skipif(
        sys.platform.startswith("win"),
        reason="POSIX-only perm test",
    )
    def test_chmod_0o600(self, tmp_state_path: Path):
        _save_state(tmp_state_path, {"version": 1, "topics": {}})
        mode = stat.S_IMODE(os.stat(tmp_state_path).st_mode)
        # On some filesystems (e.g., VFAT) this may relax; allow 0o600 or stricter
        assert mode == 0o600, f"expected 0o600, got {oct(mode)}"

    def test_overwrites_existing(self, tmp_state_path: Path):
        _save_state(tmp_state_path, {"version": 1, "topics": {"a": {"k": 1}}})
        _save_state(tmp_state_path, {"version": 1, "topics": {"b": {"k": 2}}})
        loaded = _load_state(tmp_state_path)
        assert "a" not in loaded["topics"]
        assert loaded["topics"]["b"] == {"k": 2}


# ---------------------------------------------------------------------------
# C. Router routing logic
# ---------------------------------------------------------------------------


class TestRouterAsyncRoute:
    @pytest.mark.asyncio
    async def test_noop_on_non_telegram(
        self,
        router: SubagentTopicRouter,
        non_telegram_source: SimpleNamespace,
        adapter: MagicMock,
        patched_topic_tool,
    ):
        await router._async_route(
            session_id="sid1",
            source=non_telegram_source,
            event_type="subagent.start",
            tool_name=None,
            preview="hi",
            goal="goal goal",
            adapter=adapter,
        )
        adapter.send.assert_not_called()
        patched_topic_tool.run_op.assert_not_called()

    @pytest.mark.asyncio
    async def test_noop_on_blocklisted_chat(
        self,
        router: SubagentTopicRouter,
        telegram_source: SimpleNamespace,
        adapter: MagicMock,
        patched_topic_tool,
    ):
        router._forum_blocklist.add(str(telegram_source.chat_id))
        await router._async_route(
            session_id="sid2",
            source=telegram_source,
            event_type="subagent.start",
            tool_name=None,
            preview="hi",
            goal="goal goal",
            adapter=adapter,
        )
        adapter.send.assert_not_called()
        patched_topic_tool.run_op.assert_not_called()

    @pytest.mark.asyncio
    async def test_lazy_creates_topic_on_first_event(
        self,
        router: SubagentTopicRouter,
        telegram_source: SimpleNamespace,
        adapter: MagicMock,
        patched_topic_tool,
    ):
        await router._async_route(
            session_id="sid_new",
            source=telegram_source,
            event_type="subagent.start",
            tool_name=None,
            preview="kick off",
            goal="Do a reasonable thing",
            adapter=adapter,
        )

        # _run_topic_op called exactly once for the create.
        patched_topic_tool.run_op.assert_awaited_once()
        args, kwargs = patched_topic_tool.run_op.call_args
        assert args[1] == "create"
        assert kwargs["chat_id"] == str(telegram_source.chat_id)

        # adapter.send is called 3 times on the create path:
        #   1. pointer message in parent chat
        #   2. opening message inside the new topic
        #   3. the triggering event itself forwarded to the new topic
        # The router re-loads state from disk after lazy_create_topic so the
        # first event is not silently dropped.
        assert adapter.send.await_count == 3

        # State file persists mapping.
        state = _load_state(router.state_path)
        assert "sid_new" in state["topics"]
        entry = state["topics"]["sid_new"]
        assert entry["chat_id"] == str(telegram_source.chat_id)
        assert entry["thread_id"] == "9999"
        assert entry["topic_name"].startswith("SA ")
        assert entry["parent_thread_id"] == str(telegram_source.thread_id)
        assert entry["last_message_ts"] >= entry["created_ts"]

    @pytest.mark.asyncio
    async def test_second_event_after_create_sends_actual_event(
        self,
        router: SubagentTopicRouter,
        telegram_source: SimpleNamespace,
        adapter: MagicMock,
        patched_topic_tool,
    ):
        """After lazy-create, the *next* event must be forwarded normally."""
        # Event 1 — triggers create (pointer + opening only).
        await router._async_route(
            session_id="sid_seq",
            source=telegram_source,
            event_type="subagent.start",
            tool_name=None,
            preview="go",
            goal="Reasonable goal text",
            adapter=adapter,
        )
        count_after_create = adapter.send.await_count

        # Event 2 — topic already exists, so the actual event is forwarded.
        await router._async_route(
            session_id="sid_seq",
            source=telegram_source,
            event_type="subagent.progress",
            tool_name=None,
            preview="tick",
            goal=None,
            adapter=adapter,
        )
        assert adapter.send.await_count == count_after_create + 1
        # Topic op not re-run.
        patched_topic_tool.run_op.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_reuses_existing_topic(
        self,
        router: SubagentTopicRouter,
        telegram_source: SimpleNamespace,
        adapter: MagicMock,
        patched_topic_tool,
    ):
        # Pre-populate state.
        now = time.time()
        state = {
            "version": 1,
            "topics": {
                "sid_existing": {
                    "chat_id": str(telegram_source.chat_id),
                    "thread_id": "5555",
                    "topic_name": "SA existing",
                    "created_ts": now - 100,
                    "last_message_ts": now - 100,
                    "parent_chat_id": str(telegram_source.chat_id),
                    "parent_thread_id": str(telegram_source.thread_id),
                }
            },
        }
        _save_state(router.state_path, state)

        await router._async_route(
            session_id="sid_existing",
            source=telegram_source,
            event_type="subagent.progress",
            tool_name=None,
            preview="tick",
            goal=None,
            adapter=adapter,
        )

        patched_topic_tool.run_op.assert_not_called()
        # Only the event itself — no pointer/opening.
        assert adapter.send.await_count == 1
        _, kwargs = adapter.send.call_args
        assert kwargs["chat_id"] == str(telegram_source.chat_id)
        assert kwargs["metadata"]["thread_id"] == 5555
        assert kwargs["content"].startswith("⏳")

    @pytest.mark.asyncio
    async def test_updates_last_message_ts(
        self,
        router: SubagentTopicRouter,
        telegram_source: SimpleNamespace,
        adapter: MagicMock,
        patched_topic_tool,
    ):
        old_ts = time.time() - 10_000
        _save_state(
            router.state_path,
            {
                "version": 1,
                "topics": {
                    "sid_ts": {
                        "chat_id": str(telegram_source.chat_id),
                        "thread_id": "77",
                        "topic_name": "SA ts",
                        "created_ts": old_ts,
                        "last_message_ts": old_ts,
                        "parent_chat_id": str(telegram_source.chat_id),
                        "parent_thread_id": None,
                    }
                },
            },
        )
        await router._async_route(
            session_id="sid_ts",
            source=telegram_source,
            event_type="subagent.progress",
            tool_name=None,
            preview="moving",
            goal=None,
            adapter=adapter,
        )
        loaded = _load_state(router.state_path)
        new_ts = loaded["topics"]["sid_ts"]["last_message_ts"]
        assert new_ts > old_ts

    @pytest.mark.asyncio
    async def test_lazy_create_failure_blocklists(
        self,
        router: SubagentTopicRouter,
        telegram_source: SimpleNamespace,
        adapter: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ):
        import tools.telegram_topic_tool as tgt

        monkeypatch.setattr(
            tgt, "_load_telegram_token", MagicMock(return_value=("tok", None))
        )
        monkeypatch.setattr(
            tgt,
            "_run_topic_op",
            AsyncMock(side_effect=RuntimeError("network unavailable")),
        )

        await router._async_route(
            session_id="sid_fail",
            source=telegram_source,
            event_type="subagent.start",
            tool_name=None,
            preview="x",
            goal="Goal goal goal",
            adapter=adapter,
        )
        assert str(telegram_source.chat_id) in router._forum_blocklist
        state = _load_state(router.state_path)
        assert "sid_fail" not in state["topics"]
        adapter.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_rights_classified_and_blocklisted(
        self,
        router: SubagentTopicRouter,
        telegram_source: SimpleNamespace,
        adapter: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ):
        import tools.telegram_topic_tool as tgt

        monkeypatch.setattr(
            tgt, "_load_telegram_token", MagicMock(return_value=("tok", None))
        )
        monkeypatch.setattr(
            tgt,
            "_run_topic_op",
            AsyncMock(side_effect=RuntimeError(
                "Bad Request: not enough rights to manage topics"
            )),
        )

        await router._async_route(
            session_id="sid_nr",
            source=telegram_source,
            event_type="subagent.start",
            tool_name=None,
            preview="x",
            goal="Decent length goal",
            adapter=adapter,
        )
        assert str(telegram_source.chat_id) in router._forum_blocklist

    @pytest.mark.asyncio
    async def test_not_a_forum_classified_and_blocklisted(
        self,
        router: SubagentTopicRouter,
        telegram_source: SimpleNamespace,
        adapter: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ):
        import tools.telegram_topic_tool as tgt

        monkeypatch.setattr(
            tgt, "_load_telegram_token", MagicMock(return_value=("tok", None))
        )
        monkeypatch.setattr(
            tgt,
            "_run_topic_op",
            AsyncMock(side_effect=RuntimeError(
                "Bad Request: chat is not a forum"
            )),
        )

        await router._async_route(
            session_id="sid_nf",
            source=telegram_source,
            event_type="subagent.start",
            tool_name=None,
            preview="x",
            goal="Decent length goal",
            adapter=adapter,
        )
        assert str(telegram_source.chat_id) in router._forum_blocklist

    @pytest.mark.asyncio
    async def test_no_token_blocklists(
        self,
        router: SubagentTopicRouter,
        telegram_source: SimpleNamespace,
        adapter: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ):
        import tools.telegram_topic_tool as tgt

        monkeypatch.setattr(
            tgt,
            "_load_telegram_token",
            MagicMock(return_value=(None, "no token configured")),
        )
        monkeypatch.setattr(tgt, "_run_topic_op", AsyncMock())

        await router._async_route(
            session_id="sid_notok",
            source=telegram_source,
            event_type="subagent.start",
            tool_name=None,
            preview="x",
            goal="some reasonable goal",
            adapter=adapter,
        )
        assert str(telegram_source.chat_id) in router._forum_blocklist
        adapter.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_thread_id_in_result_blocklists(
        self,
        router: SubagentTopicRouter,
        telegram_source: SimpleNamespace,
        adapter: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ):
        import tools.telegram_topic_tool as tgt

        monkeypatch.setattr(
            tgt, "_load_telegram_token", MagicMock(return_value=("tok", None))
        )
        monkeypatch.setattr(
            tgt,
            "_run_topic_op",
            AsyncMock(return_value={"success": True, "name": "SA x"}),
        )

        await router._async_route(
            session_id="sid_mthr",
            source=telegram_source,
            event_type="subagent.start",
            tool_name=None,
            preview="x",
            goal="good goal here",
            adapter=adapter,
        )
        assert str(telegram_source.chat_id) in router._forum_blocklist
        state = _load_state(router.state_path)
        assert "sid_mthr" not in state["topics"]

    @pytest.mark.asyncio
    async def test_non_success_result_blocklists(
        self,
        router: SubagentTopicRouter,
        telegram_source: SimpleNamespace,
        adapter: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ):
        import tools.telegram_topic_tool as tgt

        monkeypatch.setattr(
            tgt, "_load_telegram_token", MagicMock(return_value=("tok", None))
        )
        monkeypatch.setattr(
            tgt,
            "_run_topic_op",
            AsyncMock(return_value={"success": False, "error": "nope"}),
        )

        await router._async_route(
            session_id="sid_nosuccess",
            source=telegram_source,
            event_type="subagent.start",
            tool_name=None,
            preview="x",
            goal="good goal here",
            adapter=adapter,
        )
        assert str(telegram_source.chat_id) in router._forum_blocklist

    @pytest.mark.asyncio
    async def test_send_failure_logged_not_raised(
        self,
        router: SubagentTopicRouter,
        telegram_source: SimpleNamespace,
        patched_topic_tool,
    ):
        # Pre-seed state so we skip creation.
        old_ts = 123.0
        _save_state(
            router.state_path,
            {
                "version": 1,
                "topics": {
                    "sid_sendfail": {
                        "chat_id": str(telegram_source.chat_id),
                        "thread_id": "321",
                        "topic_name": "SA x",
                        "created_ts": old_ts,
                        "last_message_ts": old_ts,
                        "parent_chat_id": str(telegram_source.chat_id),
                        "parent_thread_id": None,
                    }
                },
            },
        )
        failing_adapter = MagicMock()
        failing_adapter.send = AsyncMock(side_effect=RuntimeError("send died"))

        # Must not raise.
        await router._async_route(
            session_id="sid_sendfail",
            source=telegram_source,
            event_type="subagent.progress",
            tool_name=None,
            preview="hi",
            goal=None,
            adapter=failing_adapter,
        )
        # last_message_ts unchanged on failure.
        state = _load_state(router.state_path)
        assert state["topics"]["sid_sendfail"]["last_message_ts"] == old_ts

    @pytest.mark.asyncio
    async def test_exception_swallowed_when_load_state_raises(
        self,
        router: SubagentTopicRouter,
        telegram_source: SimpleNamespace,
        adapter: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ):
        # Make _load_state blow up.
        monkeypatch.setattr(
            router_mod,
            "_load_state",
            MagicMock(side_effect=RuntimeError("disk on fire")),
        )
        # Should not raise.
        await router._async_route(
            session_id="sid_boom",
            source=telegram_source,
            event_type="subagent.start",
            tool_name=None,
            preview="hi",
            goal="Some valid goal",
            adapter=adapter,
        )

    @pytest.mark.asyncio
    async def test_topic_name_stored_matches_derive(
        self,
        router: SubagentTopicRouter,
        telegram_source: SimpleNamespace,
        adapter: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ):
        import tools.telegram_topic_tool as tgt

        monkeypatch.setattr(
            tgt, "_load_telegram_token", MagicMock(return_value=("tok", None))
        )
        # Make _run_topic_op echo whatever name it was called with so we can
        # confirm state["topic_name"] matches _derive_topic_name(goal, sid).
        async def fake_run_op(token, op, *, chat_id, thread_id, name):
            return {"success": True, "thread_id": "4242", "name": name}

        monkeypatch.setattr(tgt, "_run_topic_op", fake_run_op)

        goal = "Investigate flaky tests"
        await router._async_route(
            session_id="sid_name",
            source=telegram_source,
            event_type="subagent.start",
            tool_name=None,
            preview="starting",
            goal=goal,
            adapter=adapter,
        )
        state = _load_state(router.state_path)
        expected = _derive_topic_name(goal, "sid_name")
        assert state["topics"]["sid_name"]["topic_name"] == expected

    @pytest.mark.asyncio
    async def test_unknown_event_type_still_sends(
        self,
        router: SubagentTopicRouter,
        telegram_source: SimpleNamespace,
        adapter: MagicMock,
        patched_topic_tool,
    ):
        _save_state(
            router.state_path,
            {
                "version": 1,
                "topics": {
                    "sid_unk": {
                        "chat_id": str(telegram_source.chat_id),
                        "thread_id": "10",
                        "topic_name": "SA x",
                        "created_ts": 1.0,
                        "last_message_ts": 1.0,
                        "parent_chat_id": str(telegram_source.chat_id),
                        "parent_thread_id": None,
                    }
                },
            },
        )
        await router._async_route(
            session_id="sid_unk",
            source=telegram_source,
            event_type="subagent.weird_event",
            tool_name="x",
            preview="stuff",
            goal=None,
            adapter=adapter,
        )
        adapter.send.assert_awaited_once()
        _, kwargs = adapter.send.call_args
        assert "subagent.weird_event" in kwargs["content"]


# ---------------------------------------------------------------------------
# D. Sync route() entry point
# ---------------------------------------------------------------------------


class TestRouterSyncEntry:
    def test_route_sync_schedules_on_loop(
        self,
        router: SubagentTopicRouter,
        telegram_source: SimpleNamespace,
        adapter: MagicMock,
        patched_topic_tool,
    ):
        """``route()`` is callable from a sync thread and actually schedules.

        We run the asyncio loop on a background thread, submit the work via
        ``router.route()`` (sync), and wait for the scheduled future so the
        test is deterministic (no ``sleep``-based flakiness).
        """
        import threading

        loop = asyncio.new_event_loop()
        ready = threading.Event()

        def _run_loop():
            asyncio.set_event_loop(loop)
            ready.set()
            loop.run_forever()

        t = threading.Thread(target=_run_loop, daemon=True)
        t.start()
        try:
            assert ready.wait(timeout=2.0)
            router.route(
                session_id="sid_sync",
                source=telegram_source,
                event_type="subagent.start",
                tool_name=None,
                preview="sync hi",
                goal="Sync goal is fine",
                adapter=adapter,
                loop=loop,
            )
            # Wait for the scheduled coroutine to complete by polling
            # until adapter.send has been awaited at least once.
            deadline = time.time() + 3.0
            while time.time() < deadline and adapter.send.await_count == 0:
                time.sleep(0.02)
        finally:
            loop.call_soon_threadsafe(loop.stop)
            t.join(timeout=2.0)
            loop.close()

        adapter.send.assert_awaited()

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_route_sync_swallows_exceptions(
        self,
        router: SubagentTopicRouter,
        telegram_source: SimpleNamespace,
        adapter: MagicMock,
    ):
        """An invalid loop must not raise out of ``route()``."""
        # Passing a closed loop forces asyncio.run_coroutine_threadsafe to
        # raise internally; route() must swallow it.  (The created coroutine
        # never runs; the un-awaited warning is expected and filtered.)
        dead_loop = asyncio.new_event_loop()
        dead_loop.close()

        # Must not raise.
        router.route(
            session_id="sid_bad",
            source=telegram_source,
            event_type="subagent.start",
            tool_name=None,
            preview="x",
            goal="Some reasonable goal",
            adapter=adapter,
            loop=dead_loop,
        )


# ---------------------------------------------------------------------------
# E. Backpressure
# ---------------------------------------------------------------------------


class TestBackpressure:
    @pytest.mark.asyncio
    async def test_sleep_when_rate_window_full(
        self,
        router: SubagentTopicRouter,
        monkeypatch: pytest.MonkeyPatch,
    ):
        sleep_mock = AsyncMock()
        monkeypatch.setattr(router_mod.asyncio, "sleep", sleep_mock)

        # Fire 5 back-to-back — should trigger at least one sleep since span < 3s.
        for _ in range(5):
            await router._apply_backpressure("sid_rate")

        assert sleep_mock.await_count >= 1
        # Verify the delay is our configured backoff.
        sleep_mock.assert_any_await(router_mod._RATE_BACKOFF_SECONDS)

    @pytest.mark.asyncio
    async def test_no_sleep_when_events_slow(
        self,
        router: SubagentTopicRouter,
        monkeypatch: pytest.MonkeyPatch,
    ):
        sleep_mock = AsyncMock()
        monkeypatch.setattr(router_mod.asyncio, "sleep", sleep_mock)

        # Mock time so the window spans > threshold.
        current = [1000.0]

        def fake_time():
            current[0] += 10.0
            return current[0]

        monkeypatch.setattr(router_mod.time, "time", fake_time)

        for _ in range(router_mod._RATE_WINDOW):
            await router._apply_backpressure("sid_slow")

        sleep_mock.assert_not_called()


# ---------------------------------------------------------------------------
# F. Singleton accessor
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_returns_same_instance(self, monkeypatch: pytest.MonkeyPatch):
        # Reset global so the test doesn't depend on order.
        monkeypatch.setattr(router_mod, "_SINGLETON", None)
        a = get_subagent_topic_router()
        b = get_subagent_topic_router()
        assert a is b
        assert isinstance(a, SubagentTopicRouter)

    def test_singleton_survives_reset_of_fixture(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(router_mod, "_SINGLETON", None)
        first = get_subagent_topic_router()
        # Even multiple calls in a row return the same instance.
        for _ in range(5):
            assert get_subagent_topic_router() is first


# ---------------------------------------------------------------------------
# G. Router constructor
# ---------------------------------------------------------------------------


class TestRouterInit:
    def test_default_state_path_under_hermes_home(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        fake_home = tmp_path / "fake_hermes_home"
        fake_home.mkdir()
        monkeypatch.setattr(router_mod, "get_hermes_home", lambda: fake_home)
        r = SubagentTopicRouter()
        assert r.state_path == fake_home / "state" / "subagent_topics.json"
        # Parent dir should have been created.
        assert r.state_path.parent.exists()

    def test_custom_state_path_used_as_is(self, tmp_path: Path):
        custom = tmp_path / "custom" / "foo.json"
        r = SubagentTopicRouter(state_path=custom)
        assert r.state_path == custom
        assert r.state_path.parent.exists()

    def test_session_lock_caching(self, router: SubagentTopicRouter):
        l1 = router._get_session_lock("x")
        l2 = router._get_session_lock("x")
        l3 = router._get_session_lock("y")
        assert l1 is l2
        assert l1 is not l3
