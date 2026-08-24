"""Tests for the Telegram inline command picker (@botname <query>).

Covers the PTB-free logic module ``plugins/platforms/telegram/inline_picker``
(catalog collection, ranking, pagination) and the adapter's
``_handle_inline_query`` (auth gate, personal caching, empty-answer on
deny). The picker exists because the BotCommand menu is capped (60-slot
Hermes default / 100 API max, ~4KB payload) while inline mode is uncapped —
every command and skill must be reachable through it.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from plugins.platforms.telegram.inline_picker import (
    PAGE_SIZE,
    build_inline_results,
    collect_inline_catalog,
    filter_catalog,
)


# ---------------------------------------------------------------------------
# Logic module — no PTB required
# ---------------------------------------------------------------------------


class TestFilterCatalog:
    CATALOG = [
        {"name": "help", "description": "Show available commands"},
        {"name": "plan", "description": "Write a markdown implementation plan"},
        {"name": "plugin_tools", "description": "List plugin tools"},
        {"name": "songsee", "description": "Audio spectrograms and planning aids"},
    ]

    def test_empty_term_returns_everything_in_order(self):
        assert filter_catalog(self.CATALOG, "") == self.CATALOG
        assert filter_catalog(self.CATALOG, "   ") == self.CATALOG

    def test_prefix_beats_substring_beats_description(self):
        ranked = filter_catalog(self.CATALOG, "plan")
        names = [i["name"] for i in ranked]
        # prefix match first, then description matches ("...plan" in
        # plugin_tools description? no — songsee mentions planning).
        assert names[0] == "plan"
        assert "songsee" in names  # description hit ranks after name hits

    def test_hyphen_underscore_equivalent(self):
        catalog = [{"name": "gif_search", "description": "Find GIFs"}]
        assert filter_catalog(catalog, "gif-se")[0]["name"] == "gif_search"

    def test_leading_slash_stripped(self):
        assert filter_catalog(self.CATALOG, "/plan")[0]["name"] == "plan"

    def test_no_match_returns_empty(self):
        assert filter_catalog(self.CATALOG, "zzzznope") == []


class TestBuildInlineResults:
    def _fake_catalog(self, n):
        return [
            {"name": f"cmd-{i:03d}", "description": f"desc {i}"} for i in range(n)
        ]

    def test_first_page_and_next_offset(self):
        with patch(
            "plugins.platforms.telegram.inline_picker.collect_inline_catalog",
            return_value=self._fake_catalog(PAGE_SIZE + 10),
        ):
            results, next_offset = build_inline_results("", offset="")
        assert len(results) == PAGE_SIZE
        assert next_offset == str(PAGE_SIZE)

    def test_last_page_signals_stop_with_empty_offset(self):
        with patch(
            "plugins.platforms.telegram.inline_picker.collect_inline_catalog",
            return_value=self._fake_catalog(PAGE_SIZE + 10),
        ):
            results, next_offset = build_inline_results("", offset=str(PAGE_SIZE))
        assert len(results) == 10
        assert next_offset == ""

    def test_args_after_first_token_carry_into_message_text(self):
        with patch(
            "plugins.platforms.telegram.inline_picker.collect_inline_catalog",
            return_value=[{"name": "plan", "description": "Plan mode"}],
        ):
            results, _ = build_inline_results("plan migrate auth to OIDC")
        assert results[0]["message_text"] == "/plan migrate auth to OIDC"
        assert results[0]["title"] == "/plan"

    def test_bare_query_sends_bare_command(self):
        with patch(
            "plugins.platforms.telegram.inline_picker.collect_inline_catalog",
            return_value=[{"name": "plan", "description": "Plan mode"}],
        ):
            results, _ = build_inline_results("plan")
        assert results[0]["message_text"] == "/plan"

    def test_garbage_offset_treated_as_zero(self):
        with patch(
            "plugins.platforms.telegram.inline_picker.collect_inline_catalog",
            return_value=self._fake_catalog(3),
        ):
            results, _ = build_inline_results("", offset="not-a-number")
        assert len(results) == 3

    def test_result_ids_unique_across_pages(self):
        catalog = self._fake_catalog(PAGE_SIZE * 2)
        with patch(
            "plugins.platforms.telegram.inline_picker.collect_inline_catalog",
            return_value=catalog,
        ):
            page1, off = build_inline_results("", offset="")
            page2, _ = build_inline_results("", offset=off)
        ids = {r["id"] for r in page1} | {r["id"] for r in page2}
        assert len(ids) == PAGE_SIZE * 2


class TestCollectInlineCatalog:
    def test_catalog_is_uncapped_and_includes_all_skills(self, tmp_path, monkeypatch):
        """The whole point: unlike the 60-slot menu, EVERY skill appears."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        skills = tmp_path / "skills"
        names = [f"filler-{i:02d}" for i in range(70)] + ["zzz-last-skill"]
        for n in names:
            d = skills / n
            d.mkdir(parents=True)
            (d / "SKILL.md").write_text(
                f"---\nname: {n}\ndescription: test skill {n}\n---\n# {n}\n"
            )
        # SKILLS_DIR is resolved at import time — in a full-suite run it
        # points at an earlier test's HERMES_HOME, so pin it (both the
        # scanner in agent.skill_commands and the prefix allowlist in
        # _collect_gateway_skill_entries import it from tools.skills_tool).
        from tools import skills_tool

        monkeypatch.setattr(skills_tool, "SKILLS_DIR", skills)
        catalog = collect_inline_catalog()
        got = {i["name"] for i in catalog}
        # Late-alphabet skill that the capped menu would trim is present.
        assert "zzz_last_skill" in got or "zzz-last-skill" in got
        # All 71 skills present (names are telegram-sanitized).
        assert sum(1 for n in got if n.startswith("filler_")) == 70
        # Core commands present too.
        assert "help" in got and "plan" in got

    def test_no_duplicate_names(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "skills").mkdir()
        catalog = collect_inline_catalog()
        names = [i["name"] for i in catalog]
        assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# Adapter handler — telegram mock from tests/gateway/conftest.py
# ---------------------------------------------------------------------------


def _make_adapter(authorized=True):
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="***", extra={})
    adapter._is_callback_user_authorized = MagicMock(return_value=authorized)
    return adapter


def _inline_update(query="", offset="", user_id=42):
    inline_query = SimpleNamespace(
        query=query,
        offset=offset,
        from_user=SimpleNamespace(id=user_id, username="tester"),
        answer=AsyncMock(),
    )
    return SimpleNamespace(inline_query=inline_query)


@pytest.mark.asyncio
async def test_inline_query_authorized_answers_with_articles():
    adapter = _make_adapter(authorized=True)
    update = _inline_update(query="plan")
    with patch(
        "plugins.platforms.telegram.inline_picker.collect_inline_catalog",
        return_value=[{"name": "plan", "description": "Plan mode"}],
    ):
        await adapter._handle_inline_query(update, None)
    update.inline_query.answer.assert_awaited_once()
    args, kwargs = update.inline_query.answer.call_args
    assert len(args[0]) == 1
    assert kwargs["is_personal"] is True
    assert kwargs["next_offset"] == ""


@pytest.mark.asyncio
async def test_inline_query_unauthorized_gets_empty_results():
    adapter = _make_adapter(authorized=False)
    update = _inline_update(query="plan")
    await adapter._handle_inline_query(update, None)
    update.inline_query.answer.assert_awaited_once()
    args, kwargs = update.inline_query.answer.call_args
    assert args[0] == []
    assert kwargs["is_personal"] is True


@pytest.mark.asyncio
async def test_inline_query_missing_user_denied():
    adapter = _make_adapter(authorized=True)
    update = _inline_update(query="plan")
    update.inline_query.from_user = None
    await adapter._handle_inline_query(update, None)
    args, _kwargs = update.inline_query.answer.call_args
    assert args[0] == []


@pytest.mark.asyncio
async def test_inline_query_pagination_offset_passthrough():
    adapter = _make_adapter(authorized=True)
    catalog = [
        {"name": f"cmd-{i:03d}", "description": ""} for i in range(PAGE_SIZE + 5)
    ]
    update = _inline_update(query="", offset=str(PAGE_SIZE))
    with patch(
        "plugins.platforms.telegram.inline_picker.collect_inline_catalog",
        return_value=catalog,
    ):
        await adapter._handle_inline_query(update, None)
    args, kwargs = update.inline_query.answer.call_args
    assert len(args[0]) == 5
    assert kwargs["next_offset"] == ""


@pytest.mark.asyncio
async def test_inline_query_answer_failure_is_swallowed():
    adapter = _make_adapter(authorized=True)
    update = _inline_update(query="plan")
    update.inline_query.answer = AsyncMock(side_effect=RuntimeError("boom"))
    # Must not raise — inline answering is best-effort.
    await adapter._handle_inline_query(update, None)


@pytest.mark.asyncio
async def test_inline_query_none_update_is_noop():
    adapter = _make_adapter(authorized=True)
    await adapter._handle_inline_query(SimpleNamespace(inline_query=None), None)
