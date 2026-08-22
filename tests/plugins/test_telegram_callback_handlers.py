"""Tests for Telegram custom script-backed callback handlers.

Covers the ``platforms.telegram.extra.callback_handlers`` extension: config
parsing, prefix routing (including the CCC ``ccc:*`` / ``ai:*`` producer
pattern), authorization, script execution, acknowledgement labels, and
keyboard stripping. Regression for the shimmer-forever bug: every callback
press must be answered exactly once, success or failure.
"""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Ensure the repo root is importable
# ---------------------------------------------------------------------------
_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


# ---------------------------------------------------------------------------
# Minimal Telegram mock so TelegramAdapter can be imported
# ---------------------------------------------------------------------------
def _ensure_telegram_mock():
    """Wire up the minimal mocks required to import TelegramAdapter."""
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return

    mod = MagicMock()
    mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    mod.constants.ParseMode.MARKDOWN = "Markdown"
    mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    mod.constants.ParseMode.HTML = "HTML"
    mod.constants.ChatType.PRIVATE = "private"
    mod.constants.ChatType.GROUP = "group"
    mod.constants.ChatType.SUPERGROUP = "supergroup"
    mod.constants.ChatType.CHANNEL = "channel"
    mod.error.NetworkError = type("NetworkError", (OSError,), {})
    mod.error.TimedOut = type("TimedOut", (OSError,), {})
    mod.error.BadRequest = type("BadRequest", (Exception,), {})

    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, mod)
    sys.modules.setdefault("telegram.error", mod.error)


_ensure_telegram_mock()

from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402
from gateway.config import PlatformConfig  # noqa: E402


def _make_adapter(extra=None):
    config = PlatformConfig(enabled=True, token="test-token", extra=extra or {})
    adapter = TelegramAdapter(config)
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    return adapter


def _make_query(data: str, *, text_html: str = "Decision #42", user_id: str = "111"):
    query = MagicMock()
    query.data = data
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()
    query.from_user = SimpleNamespace(id=user_id, first_name="Michael")
    query.message = MagicMock()
    query.message.chat_id = 555
    query.message.chat = SimpleNamespace(type="private")
    query.message.message_thread_id = None
    query.message.text_html = text_html
    query.message.text = text_html
    return query


def _handlers_extra(script: str, prefixes=None, **kwargs):
    entry = {"prefixes": prefixes or ["ccc:", "ai:"], "script": script}
    entry.update(kwargs)
    return {"callback_handlers": [entry]}


def _write_script(tmp_path, body: str) -> str:
    script = tmp_path / "handler.sh"
    script.write_text(f"#!/bin/bash\n{body}\n")
    script.chmod(0o755)
    return str(script)


async def _dispatch(adapter, query):
    update = MagicMock()
    update.callback_query = query
    await adapter._handle_callback_query(update, None)


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------
class TestCallbackHandlersParsing:
    def test_valid_config_is_parsed(self, tmp_path):
        script = _write_script(tmp_path, "exit 0")
        adapter = _make_adapter(_handlers_extra(script, timeout=15, keep_keyboard=True))
        assert len(adapter._callback_handlers) == 1
        handler = adapter._callback_handlers[0]
        assert handler["prefixes"] == ("ccc:", "ai:")
        assert handler["timeout"] == 15.0
        assert handler["keep_keyboard"] is True

    def test_non_list_config_is_ignored(self):
        adapter = _make_adapter({"callback_handlers": "not-a-list"})
        assert adapter._callback_handlers == []

    def test_entry_without_script_is_dropped(self):
        adapter = _make_adapter({"callback_handlers": [{"prefixes": ["x:"]}]})
        assert adapter._callback_handlers == []

    def test_entry_without_prefixes_is_dropped(self, tmp_path):
        script = _write_script(tmp_path, "exit 0")
        adapter = _make_adapter({"callback_handlers": [{"script": script}]})
        assert adapter._callback_handlers == []

    def test_reserved_prefixes_are_rejected(self, tmp_path):
        script = _write_script(tmp_path, "exit 0")
        adapter = _make_adapter(
            _handlers_extra(script, prefixes=["mp:", "gt:", "ea:", "cl:", "update_prompt:"])
        )
        assert adapter._callback_handlers == []

    def test_reserved_prefix_dropped_but_valid_kept(self, tmp_path):
        script = _write_script(tmp_path, "exit 0")
        adapter = _make_adapter(_handlers_extra(script, prefixes=["gt:", "ccc:"]))
        assert adapter._callback_handlers[0]["prefixes"] == ("ccc:",)

    def test_script_path_is_expanded(self):
        adapter = _make_adapter(
            {"callback_handlers": [{"prefixes": ["x:"], "script": "~/scripts/h.sh"}]}
        )
        assert not adapter._callback_handlers[0]["script"].startswith("~")

    def test_resolve_matches_prefix(self, tmp_path):
        script = _write_script(tmp_path, "exit 0")
        adapter = _make_adapter(_handlers_extra(script))
        assert adapter._resolve_callback_handler("ccc:decision:approve:42") is not None
        assert adapter._resolve_callback_handler("ai:done:m1:0") is not None
        assert adapter._resolve_callback_handler("unrelated:x") is None

    def test_no_handlers_by_default(self):
        adapter = _make_adapter()
        assert adapter._callback_handlers == []
        assert adapter._resolve_callback_handler("ccc:decision:approve:42") is None


# ---------------------------------------------------------------------------
# Dispatch behaviour
# ---------------------------------------------------------------------------
class TestCallbackHandlerDispatch:
    @pytest.fixture(autouse=True)
    def _authorized(self, monkeypatch):
        monkeypatch.setattr(
            TelegramAdapter, "_is_callback_user_authorized", lambda self, *a, **k: True
        )

    @pytest.mark.asyncio
    async def test_success_answers_and_strips_keyboard(self, tmp_path):
        script = _write_script(tmp_path, 'echo "✅ Approved"')
        adapter = _make_adapter(_handlers_extra(script))
        query = _make_query("ccc:decision:approve:42")

        await _dispatch(adapter, query)

        query.answer.assert_awaited_once_with(text="✅ Approved")
        query.edit_message_text.assert_awaited_once()
        kwargs = query.edit_message_text.await_args.kwargs
        assert kwargs["reply_markup"] is None
        assert "✅ Approved" in kwargs["text"]
        assert "Michael" in kwargs["text"]

    @pytest.mark.asyncio
    async def test_success_without_stdout_uses_default_label(self, tmp_path):
        script = _write_script(tmp_path, "exit 0")
        adapter = _make_adapter(_handlers_extra(script))
        query = _make_query("ai:done:m7:1")

        await _dispatch(adapter, query)

        query.answer.assert_awaited_once_with(text="✓ Done")

    @pytest.mark.asyncio
    async def test_keep_keyboard_skips_edit(self, tmp_path):
        script = _write_script(tmp_path, "exit 0")
        adapter = _make_adapter(_handlers_extra(script, keep_keyboard=True))
        query = _make_query("ccc:decision:view:42")

        await _dispatch(adapter, query)

        query.answer.assert_awaited_once()
        query.edit_message_text.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_failure_surfaces_stderr_and_keeps_keyboard(self, tmp_path):
        script = _write_script(tmp_path, 'echo "decision already approved" >&2; exit 1')
        adapter = _make_adapter(_handlers_extra(script))
        query = _make_query("ccc:decision:approve:42")

        await _dispatch(adapter, query)

        query.answer.assert_awaited_once()
        label = query.answer.await_args.kwargs["text"]
        assert label.startswith("❌")
        assert "already approved" in label
        query.edit_message_text.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_missing_script_answers_with_error(self, tmp_path):
        adapter = _make_adapter(_handlers_extra(str(tmp_path / "nope.sh")))
        query = _make_query("ccc:decision:approve:42")

        await _dispatch(adapter, query)

        query.answer.assert_awaited_once_with(text="❌ Callback handler script missing")
        query.edit_message_text.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_timeout_answers_with_error(self, tmp_path):
        script = _write_script(tmp_path, "sleep 5")
        adapter = _make_adapter(_handlers_extra(script, timeout=1))
        query = _make_query("ccc:decision:approve:42")

        await _dispatch(adapter, query)

        query.answer.assert_awaited_once_with(text="❌ Handler timed out")

    @pytest.mark.asyncio
    async def test_script_receives_raw_callback_data(self, tmp_path):
        out = tmp_path / "received.txt"
        script = _write_script(tmp_path, f'echo -n "$1" > "{out}"')
        adapter = _make_adapter(_handlers_extra(script))
        query = _make_query("ccc:followup:snooze1d:99")

        await _dispatch(adapter, query)

        assert out.read_text() == "ccc:followup:snooze1d:99"

    @pytest.mark.asyncio
    async def test_unmatched_data_falls_through_untouched(self, tmp_path):
        script = _write_script(tmp_path, "exit 0")
        adapter = _make_adapter(_handlers_extra(script))
        query = _make_query("something:else:entirely")

        await _dispatch(adapter, query)

        # Not ours, not update_prompt → adapter must not answer or edit.
        query.answer.assert_not_awaited()
        query.edit_message_text.assert_not_awaited()


class TestCallbackHandlerAuthorization:
    @pytest.mark.asyncio
    async def test_unauthorized_user_is_blocked(self, tmp_path, monkeypatch):
        out = tmp_path / "ran.txt"
        script = _write_script(tmp_path, f'touch "{out}"')
        adapter = _make_adapter(_handlers_extra(script))
        monkeypatch.setattr(
            TelegramAdapter,
            "_is_callback_user_authorized",
            lambda self, *a, **k: False,
        )
        query = _make_query("ccc:decision:approve:42", user_id="999")

        await _dispatch(adapter, query)

        query.answer.assert_awaited_once_with(text="⛔ Not authorized.")
        assert not out.exists(), "script must not run for unauthorized users"
        query.edit_message_text.assert_not_awaited()
