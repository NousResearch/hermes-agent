"""Integration tests for the gateway /sessions ↔ Telegram picker wiring.

The adapter-level state and callback dispatch live in
``test_telegram_sessions_picker.py``. This file covers the gateway side:

* ``_handle_sessions_command`` renders an interactive picker when the
  active adapter exposes ``send_sessions_picker``.
* Otherwise it falls back to the numbered text listing (the existing
  behavior).
* The picker's ``on_session_selected`` callback re-runs the IDOR guard
  via ``_resume_target_allowed`` before delegating to the shared
  ``_resume_session_by_id`` helper — a co-member in a shared group
  cannot use the button to bind to another user's persisted session.
* ``_resume_session_by_id`` performs the full session-switch cleanup
  (release running agent, switch session entry, clear conversation
  scope, evict cached agent) — the same funnel as the text ``/resume``
  path, so the picker cannot bypass any of the bug-class regressions
  that funnel fixes (e.g. #10702, #58403, #6672).
"""

from __future__ import annotations

import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest


_NO_TELEGRAM = "telegram" not in sys.modules or not hasattr(
    sys.modules["telegram"], "__file__"
)
if _NO_TELEGRAM:
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


from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent, SendResult
from gateway.session import SessionSource, build_session_key


def _event(text: str = "/sessions") -> MessageEvent:
    return MessageEvent(
        text=text,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="12345",
            chat_type="group",
            user_id="777",
            thread_id="42",
        ),
        message_id="msg-1",
    )


def _make_runner(tmp_path):
    """Build a bare-bones GatewayRunner with the methods used by
    ``_handle_sessions_command`` stubbed out — drives the real session
    listing path and only the picker/integration code is exercised end-to-end.
    """
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = fake_dotenv

    config = GatewayConfig()
    from gateway import run as gateway_run

    runner = gateway_run.GatewayRunner(config)
    runner.adapters = {}
    runner._running_agents = {}
    runner._is_user_authorized = lambda _source: True
    runner._cache_session_source = lambda _key, _source: None
    runner._reply_anchor_for_event = lambda _event: None
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()

    # Session store facade used by _handle_sessions_command.
    runner._session_db = MagicMock()
    # query_session_listing calls getattr(self._session_db, "_db",
    # self._session_db) — a MagicMock returns a MagicMock for `_db`, so the
    # real call lands on `_session_db._db.list_sessions_rich`, not the
    # `_session_db` we set up. Wire both.
    runner._session_db.list_sessions_rich = MagicMock(
        return_value=[
            {"id": "s1", "title": "Research", "preview": "notes"},
            {"id": "s2", "title": "Coding", "preview": "fix bug"},
        ]
    )
    runner._session_db._db = runner._session_db
    # _session_db.get_session_title is awaited in _resume_session_by_id.
    runner._session_db.get_session_title = AsyncMock(return_value=None)

    # Async session store facade used by the gate + the helper. The real
    # facade is a property that returns a fresh AsyncSessionStore unless
    # ``_async_session_store._store is self.session_store`` — so we must
    # pin the mock's ``_store`` to the same MagicMock we wire into
    # ``session_store`` or the property getter will rebuild a real
    # AsyncSessionStore over our mock on first access.
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session = MagicMock(
        return_value=MagicMock(session_id="current-session")
    )

    async_facade = MagicMock()
    async_facade._store = runner.session_store  # pin to bypass the property rebuild
    async_facade.get_or_create_session = AsyncMock(
        return_value=MagicMock(session_id="current-session")
    )
    async_facade.switch_session = AsyncMock(
        return_value=MagicMock(session_id="s1")
    )
    async_facade.load_transcript = AsyncMock(return_value=[])
    runner._async_session_store = async_facade

    # Boundary funnels.
    runner._release_running_agent_state = MagicMock()
    runner._clear_conversation_scope = MagicMock()
    runner._evict_cached_agent = MagicMock()

    # Cross-origin / scoping primitives.
    runner._resume_caller_is_admin = lambda _source: False
    runner._resume_row_visible = AsyncMock(return_value=True)
    runner._resume_target_allowed = AsyncMock(return_value=True)
    runner._thread_metadata_for_source = MagicMock(return_value={"thread_id": "42"})
    runner._normalize_source_for_session_key = lambda source: source
    runner._session_key_for_source = lambda source: build_session_key(source)

    return runner


# ── 1. Picker is invoked when the adapter implements it ────────────────────


@pytest.mark.asyncio
async def test_sessions_command_uses_picker_when_adapter_supports_it(tmp_path):
    runner = _make_runner(tmp_path)
    adapter = MagicMock()
    adapter.send_sessions_picker = AsyncMock(
        return_value=SendResult(success=True, message_id="100")
    )
    runner.adapters[Platform.TELEGRAM] = adapter

    result = await runner._handle_sessions_command(_event("/sessions"))

    # Picker rendered, no text reply
    assert result is None
    adapter.send_sessions_picker.assert_awaited_once()
    kwargs = adapter.send_sessions_picker.await_args.kwargs
    assert kwargs["chat_id"] == "12345"
    sessions = kwargs["sessions"]
    assert {s["id"] for s in sessions} == {"s1", "s2"}
    assert kwargs["current_session_id"] == "current-session"
    # Thread metadata threaded through so the picker lands in the same topic
    assert kwargs["metadata"] == {"thread_id": "42"}
    # The runner callback is a closure that captures session_key + source
    assert callable(kwargs["on_session_selected"])


# ── 2. Falls back to text when no adapter has the picker ──────────────────


@pytest.mark.asyncio
async def test_sessions_command_falls_back_to_text_when_adapter_lacks_picker(tmp_path):
    runner = _make_runner(tmp_path)
    # Plain stub adapter — no send_sessions_picker attribute at all
    runner.adapters[Platform.TELEGRAM] = MagicMock(spec=[])

    result = await runner._handle_sessions_command(_event("/sessions"))

    # Text listing returned, no picker sent
    assert isinstance(result, str)
    assert "Research" in result
    assert "Coding" in result
    assert "/resume" in result  # the existing text footer hint


# ── 3. IDOR guard runs in the picker callback ──────────────────────────────


@pytest.mark.asyncio
async def test_picker_callback_runs_idor_guard_before_resuming(tmp_path):
    """The picker's on_session_selected closure must re-run
    _resume_target_allowed with the same SessionSource that opened the
    picker — a co-member tapping someone else's session list cannot
    bypass the gate."""
    runner = _make_runner(tmp_path)
    adapter = MagicMock()
    picker_invocation = {}
    adapter.send_sessions_picker = AsyncMock(
        side_effect=lambda **_kw: (
            picker_invocation.update(_kw) or SendResult(success=True, message_id="100")
        )
    )
    runner.adapters[Platform.TELEGRAM] = adapter

    await runner._handle_sessions_command(_event("/sessions"))
    on_selected = picker_invocation["on_session_selected"]

    # IDOR guard denies the resume
    runner._resume_target_allowed = AsyncMock(return_value=False)
    result = await on_selected("s1")

    assert result
    assert "blocked" in result.lower() or "not owner" in result.lower()
    # Critically: switch_session was NEVER called — the helper is gated
    runner.async_session_store.switch_session.assert_not_awaited()
    runner._clear_conversation_scope.assert_not_called()
    runner._evict_cached_agent.assert_not_called()


# ── 4. Picker callback runs the full session-switch funnel ─────────────────


@pytest.mark.asyncio
async def test_picker_callback_runs_resume_session_by_id(tmp_path):
    """With the IDOR guard passed, the picker callback delegates to
    _resume_session_by_id — the same funnel as the text /resume path,
    so it gets the conversation-scope clear, cached-agent eviction, and
    running-agent release for free."""
    runner = _make_runner(tmp_path)
    adapter = MagicMock()
    picker_invocation = {}
    adapter.send_sessions_picker = AsyncMock(
        side_effect=lambda **_kw: (
            picker_invocation.update(_kw) or SendResult(success=True, message_id="100")
        )
    )
    runner.adapters[Platform.TELEGRAM] = adapter

    await runner._handle_sessions_command(_event("/sessions"))
    on_selected = picker_invocation["on_session_selected"]

    # IDOR guard passes
    runner._resume_target_allowed = AsyncMock(return_value=True)
    # No prior → not already-on
    runner.async_session_store.get_or_create_session.return_value = MagicMock(
        session_id="current-session"
    )

    result = await on_selected("s1")

    # IDOR guard ran with the captured source
    runner._resume_target_allowed.assert_awaited()
    _, target_id = runner._resume_target_allowed.await_args.args
    assert target_id == "s1"
    # Funnel calls all happened
    runner._release_running_agent_state.assert_called_once()
    runner.async_session_store.switch_session.assert_awaited_once()
    runner._clear_conversation_scope.assert_called_once()
    runner._evict_cached_agent.assert_called_once()
    # Result is a string (rendered via edit_message_text by the adapter)
    assert isinstance(result, str)


# ── 5. Same chat, same chat_id → session list is filtered by IDOR gate ─────


@pytest.mark.asyncio
async def test_sessions_command_scopes_list_to_origin_visible_rows(tmp_path):
    """The picker must receive only the rows the runner already filtered
    through _resume_row_visible + _resume_target_allowed — a co-member's
    own picker list must not include sessions from another user."""
    runner = _make_runner(tmp_path)
    adapter = MagicMock()
    adapter.send_sessions_picker = AsyncMock(
        return_value=SendResult(success=True, message_id="100")
    )
    runner.adapters[Platform.TELEGRAM] = adapter

    # Filter hook drops s2 — "this user can't see this one"
    async def _row_visible(source, row, allow_all):
        return row["id"] == "s1"

    runner._resume_row_visible = _row_visible

    await runner._handle_sessions_command(_event("/sessions"))

    sessions = adapter.send_sessions_picker.await_args.kwargs["sessions"]
    assert {s["id"] for s in sessions} == {"s1"}, (
        "Picker must inherit the runner's origin-scoped listing — "
        "otherwise the IDOR guard at the callback cannot stop what the "
        "listing already exposed."
    )


# ── 6. Picker gets the FULL origin-scoped list, not the 10-cap ──────────────


@pytest.mark.asyncio
async def test_picker_passes_full_list_beyond_text_list_cap(tmp_path):
    """The text picker caps at 10 rows (the legacy UX). The picker branch
    must pass the full origin-scoped list — the picker paginates internally
    via Prev/Next at 8 per page, so capping at 10 would silently hide
    older sessions behind a single page that ends with no Next button."""
    runner = _make_runner(tmp_path)
    # 15 sessions — past the text fallback cap (10) and past one picker
    # page (8). The picker is supposed to get all 15 so the user can flip
    # to the second page.
    runner._session_db.list_sessions_rich = MagicMock(
        return_value=[
            {"id": f"s{i:02d}", "title": f"S{i}", "preview": ""}
            for i in range(15)
        ]
    )

    adapter = MagicMock()
    adapter.send_sessions_picker = AsyncMock(
        return_value=SendResult(success=True, message_id="100")
    )
    runner.adapters[Platform.TELEGRAM] = adapter

    await runner._handle_sessions_command(_event("/sessions"))

    sessions = adapter.send_sessions_picker.await_args.kwargs["sessions"]
    assert len(sessions) == 15, (
        f"Picker got {len(sessions)}, expected 15. The `rows[:10]` cap "
        "must NOT apply to the picker branch — the picker paginates, "
        "so the runner should hand off the full origin-scoped list."
    )
