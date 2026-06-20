"""Tests for the generic script-button dispatcher on the Telegram adapter.

Covers _load_button_handlers registration, the _handle_callback_query dispatch
(authorization gate, verb allow-list, precedence vs built-in prefixes, spinner
on malformed data) and _run_script_button subprocess error handling.
"""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


def _ensure_telegram_mock():
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

from gateway.platforms.telegram import TelegramAdapter
from gateway.config import PlatformConfig


_HANDLERS = [
    {
        "prefix": "na",
        "script": "~/.hermes/scripts/news_action_button_handler.py",
        "actions": {"st": "Story", "x": "𝕏"},
    },
    {
        "prefix": "wx",
        "script": "/opt/scripts/weather.py",
        "actions": {"fc": "Forecast"},
    },
]


def _make_adapter(handlers=None):
    extra = {}
    if handlers is not None:
        extra["button_handlers"] = handlers
    config = PlatformConfig(enabled=True, token="test-token", extra=extra)
    adapter = TelegramAdapter(config)
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    # Authorization defers to the runner via _message_handler; with no handler
    # set, _is_callback_user_authorized falls back to TELEGRAM_ALLOWED_USERS.
    adapter._message_handler = None
    return adapter


def _make_query(data, *, user_id="123", chat_id=100, chat_type="private",
                thread_id=None, reply_to_id=None):
    chat = SimpleNamespace(type=chat_type)
    reply_to = (
        SimpleNamespace(message_id=reply_to_id) if reply_to_id is not None else None
    )
    message = SimpleNamespace(
        chat_id=chat_id,
        chat=chat,
        message_thread_id=thread_id,
        reply_to_message=reply_to,
        message_id=999,
    )
    query = SimpleNamespace(
        data=data,
        from_user=SimpleNamespace(id=user_id, first_name="Tester"),
        message=message,
        answer=AsyncMock(),
        edit_message_text=AsyncMock(),
    )
    return query


def _make_update(query):
    return SimpleNamespace(callback_query=query)


# --------------------------------------------------------------------------
# _load_button_handlers
# --------------------------------------------------------------------------

class TestLoadButtonHandlers:

    def test_two_handlers_registered_by_prefix(self):
        adapter = _make_adapter(_HANDLERS)
        assert set(adapter._button_handlers) == {"na", "wx"}
        assert adapter._button_handlers["na"]["script"].endswith(
            "news_action_button_handler.py"
        )

    def test_entry_missing_prefix_is_skipped(self, caplog):
        handlers = [
            {"script": "/x.py", "actions": {"a": "A"}},  # no prefix
            {"prefix": "ok", "script": "/y.py", "actions": {"b": "B"}},
        ]
        with caplog.at_level("WARNING"):
            adapter = _make_adapter(handlers)
        assert set(adapter._button_handlers) == {"ok"}
        assert any(
            "missing" in r.message and "prefix" in r.message
            for r in caplog.records
        )

    def test_model_picker_shadowed_prefix_is_skipped(self, caplog):
        """A configured prefix shadowed by the bare model-picker tokens
        (mb/mx) is dropped with a warning, since it could never fire."""
        handlers = [
            {"prefix": "mboard", "script": "/x.py", "actions": {"a": "A"}},
            {"prefix": "mx", "script": "/y.py", "actions": {"b": "B"}},
            {"prefix": "ok", "script": "/z.py", "actions": {"c": "C"}},
        ]
        with caplog.at_level("WARNING"):
            adapter = _make_adapter(handlers)
        assert set(adapter._button_handlers) == {"ok"}
        assert any("shadowed" in r.message for r in caplog.records)

    def test_absent_config_no_handlers(self):
        adapter = _make_adapter(None)
        assert adapter._button_handlers == {}

    def test_empty_config_no_handlers(self):
        adapter = _make_adapter([])
        assert adapter._button_handlers == {}


# --------------------------------------------------------------------------
# dispatch via _handle_callback_query
# --------------------------------------------------------------------------

def _patch_subprocess(monkeypatch, returncode=0, stdout=b"ok", stderr=b""):
    """Patch asyncio.create_subprocess_exec; return (mock, captured_argv)."""
    captured = {}
    proc = MagicMock()
    proc.returncode = returncode
    proc.communicate = AsyncMock(return_value=(stdout, stderr))

    async def fake_exec(*argv, **kwargs):
        captured["argv"] = list(argv)
        return proc

    mock = AsyncMock(side_effect=fake_exec)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", mock)
    return mock, captured


class TestDispatch:

    @pytest.mark.asyncio
    async def test_authorized_known_verb_spawns_subprocess(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "123")
        adapter = _make_adapter(_HANDLERS)
        adapter._send_message_with_thread_fallback = AsyncMock()
        mock, captured = _patch_subprocess(monkeypatch)
        query = _make_query("na:st:42", user_id="123")

        await adapter._handle_callback_query(_make_update(query), None)
        # Drain the GC-safe task created by the dispatcher.
        await asyncio.gather(*list(adapter._background_tasks))

        assert mock.await_count == 1
        argv = captured["argv"]
        assert argv[0] == sys.executable
        assert argv[1].endswith("news_action_button_handler.py")
        assert "--action" in argv and argv[argv.index("--action") + 1] == "st"
        assert "--action-id" in argv and argv[argv.index("--action-id") + 1] == "42"
        assert "--chat-id" in argv and argv[argv.index("--chat-id") + 1] == "100"
        query.answer.assert_awaited()
        # Spinner-clear answer carries no error text on success.
        assert query.answer.await_args.kwargs.get("text") is None
        # A successful run sends no warning message.
        adapter._send_message_with_thread_fallback.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_reply_to_and_thread_appended_when_present(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "123")
        adapter = _make_adapter(_HANDLERS)
        _, captured = _patch_subprocess(monkeypatch)
        query = _make_query(
            "na:st:42", user_id="123", chat_type="supergroup",
            thread_id=7, reply_to_id=55,
        )

        await adapter._handle_callback_query(_make_update(query), None)
        await asyncio.gather(*list(adapter._background_tasks))

        argv = captured["argv"]
        assert argv[argv.index("--reply-to-message-id") + 1] == "55"
        assert argv[argv.index("--thread-id") + 1] == "7"

    @pytest.mark.asyncio
    async def test_arg_with_embedded_colons_preserved(self, monkeypatch):
        """split(":", 2) keeps colons in the arg as a single argv element."""
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "123")
        adapter = _make_adapter(_HANDLERS)
        _, captured = _patch_subprocess(monkeypatch)
        query = _make_query("na:st:a:b", user_id="123")

        await adapter._handle_callback_query(_make_update(query), None)
        await asyncio.gather(*list(adapter._background_tasks))

        argv = captured["argv"]
        assert argv[argv.index("--action") + 1] == "st"
        assert argv[argv.index("--action-id") + 1] == "a:b"

    @pytest.mark.asyncio
    async def test_reply_to_and_thread_omitted_when_absent(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "123")
        adapter = _make_adapter(_HANDLERS)
        _, captured = _patch_subprocess(monkeypatch)
        query = _make_query("na:st:42", user_id="123")  # no thread, no reply_to

        await adapter._handle_callback_query(_make_update(query), None)
        await asyncio.gather(*list(adapter._background_tasks))

        argv = captured["argv"]
        assert "--reply-to-message-id" not in argv
        assert "--thread-id" not in argv

    @pytest.mark.asyncio
    async def test_unauthorized_user_does_not_spawn(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "999")
        adapter = _make_adapter(_HANDLERS)
        mock, _ = _patch_subprocess(monkeypatch)
        query = _make_query("na:st:42", user_id="123")  # not allowed

        await adapter._handle_callback_query(_make_update(query), None)

        assert mock.await_count == 0
        assert adapter._background_tasks == set()
        query.answer.assert_awaited_once()
        text = query.answer.await_args.kwargs.get("text", "")
        assert "not authorized" in text.lower()

    @pytest.mark.asyncio
    async def test_unknown_verb_does_not_spawn(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "123")
        adapter = _make_adapter(_HANDLERS)
        mock, _ = _patch_subprocess(monkeypatch)
        query = _make_query("na:bogus:42", user_id="123")  # verb not in actions

        await adapter._handle_callback_query(_make_update(query), None)

        assert mock.await_count == 0
        assert adapter._background_tasks == set()
        query.answer.assert_awaited_once()
        assert query.answer.await_args.kwargs.get("text") == "Unknown action."

    @pytest.mark.asyncio
    async def test_malformed_data_answers_without_spawn(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "123")
        adapter = _make_adapter(_HANDLERS)
        mock, _ = _patch_subprocess(monkeypatch)
        query = _make_query("na:onlytwo", user_id="123")  # split -> 2 parts

        await adapter._handle_callback_query(_make_update(query), None)

        assert mock.await_count == 0
        assert adapter._background_tasks == set()
        query.answer.assert_awaited_once()
        # Malformed data clears the spinner with a bare answer (no error text).
        assert query.answer.await_args.kwargs.get("text") is None

    @pytest.mark.asyncio
    async def test_builtin_prefix_gt_takes_precedence(self, monkeypatch):
        """A configured prefix that startswith-collides with gt: must route
        to the built-in gmail-triage handler, not the generic script path."""
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "123")
        handlers = [
            {"prefix": "gt", "script": "/evil.py", "actions": {"send": "Send"}},
        ]
        adapter = _make_adapter(handlers)
        mock, _ = _patch_subprocess(monkeypatch)
        triage = AsyncMock()
        adapter._handle_gmail_triage_callback = triage
        query = _make_query("gt:send:42", user_id="123")

        await adapter._handle_callback_query(_make_update(query), None)

        assert mock.await_count == 0
        assert adapter._background_tasks == set()
        triage.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "prefix, data",
        [
            ("ea", "ea:once:42"),
            ("sc", "sc:y:42"),
        ],
    )
    async def test_builtin_prefix_precedence_no_generic_spawn(
        self, monkeypatch, prefix, data
    ):
        """Configured prefixes that startswith-collide with other built-ins
        (ea:/sc:) are handled by those built-ins, never the generic loop.

        The generic dispatcher is the only path that spawns a subprocess /
        registers a background task, so the absence of both proves the
        built-in branch ran instead."""
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "123")
        handlers = [
            {"prefix": prefix, "script": "/evil.py", "actions": {"once": "X", "y": "Y"}},
        ]
        adapter = _make_adapter(handlers)
        mock, _ = _patch_subprocess(monkeypatch)
        query = _make_query(data, user_id="123")

        await adapter._handle_callback_query(_make_update(query), None)

        assert mock.await_count == 0
        assert adapter._background_tasks == set()


# --------------------------------------------------------------------------
# _run_script_button
# --------------------------------------------------------------------------

class TestRunScriptButton:

    @pytest.mark.asyncio
    async def test_nonzero_returncode_sends_warning(self, monkeypatch):
        adapter = _make_adapter(_HANDLERS)
        adapter._send_message_with_thread_fallback = AsyncMock()
        _patch_subprocess(monkeypatch, returncode=2, stdout=b"", stderr=b"boom")

        await adapter._run_script_button(
            _HANDLERS[0], "st", "42", "100",
            reply_to_message_id="55", thread_id="7",
        )

        adapter._send_message_with_thread_fallback.assert_awaited_once()
        kwargs = adapter._send_message_with_thread_fallback.await_args.kwargs
        text = kwargs["text"]
        assert "⚠️" in text
        assert "boom" in text
        # The mapped action label is rendered, not the raw verb.
        assert "Story" in text
        # Routing kwargs are int-converted and plumbed through correctly.
        assert kwargs["chat_id"] == 100
        assert kwargs["reply_to_message_id"] == 55
        assert kwargs["message_thread_id"] == 7
        assert kwargs["parse_mode"] is None

    @pytest.mark.asyncio
    async def test_exception_sends_crash_warning_no_raise(self, monkeypatch):
        adapter = _make_adapter(_HANDLERS)
        adapter._send_message_with_thread_fallback = AsyncMock()

        async def boom(*a, **k):
            raise RuntimeError("kaboom")

        monkeypatch.setattr(asyncio, "create_subprocess_exec", boom)

        # Must not raise.
        await adapter._run_script_button(_HANDLERS[0], "st", "42", "100")

        adapter._send_message_with_thread_fallback.assert_awaited_once()
        text = adapter._send_message_with_thread_fallback.await_args.kwargs["text"]
        assert "⚠️" in text
        assert "kaboom" in text

    @pytest.mark.asyncio
    async def test_missing_script_returns_early(self, monkeypatch):
        adapter = _make_adapter(_HANDLERS)
        mock, _ = _patch_subprocess(monkeypatch)

        await adapter._run_script_button(
            {"actions": {"st": "Story"}}, "st", "42", "100",
        )

        assert mock.await_count == 0
