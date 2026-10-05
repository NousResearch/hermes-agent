"""A plugin command handler must see the chat's current HERMES_SESSION_ID inside its scope.

``_hm_dispatch_quick_and_plugin_commands`` binds HERMES_SESSION_* via ``_session_env_scope`` but
``_set_session_env`` never passed ``session_id``, and the dispatch built its context without a
session entry — so ``set_session_vars`` bound ``HERMES_SESSION_ID=""`` and a handler keying state
by session id had nothing to match the ``old_session_id`` the next ``/new`` reports in
``on_session_reset`` (#123245).

The routing id is peeked read-only (never minted — a slash command must not create a session or
touch the activity clock) through the same heal the turn's ``get_or_create_session`` applies:
``_prune_stale_sessions_locked`` only runs at startup, so ``_entries`` can still hold an id ended
mid-run (#54878) or left behind by a compression rotation, and a raw peek would hand the plugin
the stale id the next turn heals away. A chat whose first message is a command has no live route
at all, and an ended one heals to nothing — both fall back to the chat's own session key, the
same non-empty fallback tui_gateway uses, so per-chat plugin state never shares one "" bucket.
The scope still clears the var on exit.
"""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.session_context import get_session_env

_UNSET = object()


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


class _FakeStore:
    """Just the SessionStore surface the plugin-command peek reads (no DB behind it)."""

    def __init__(self, entries=None, tips=None, ended_ids=()):
        self._lock = threading.Lock()
        self._entries = entries if entries is not None else {}
        self._tips = tips or {}
        self._ended = set(ended_ids)

    def _generate_session_key(self, _source) -> str:
        return "sk-test"

    def _ensure_loaded_locked(self) -> None:
        pass

    def _compression_tip_for_session_id(self, session_id):
        # The real one returns db.get_compression_tip(sid) or sid: unknown ids map to themselves.
        return self._tips.get(session_id, session_id)

    def _is_session_ended_in_db(self, session_id) -> bool:
        return session_id in self._ended


def _make_runner(store=_UNSET):
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    runner.adapters = {Platform.TELEGRAM: MagicMock()}
    runner._draining = False
    runner._hm_quick_commands = lambda: {}
    if store is not _UNSET:
        runner.session_store = store
    return runner


async def _dispatch_and_capture(monkeypatch, runner):
    """Run a /planmode dispatch whose handler records HERMES_SESSION_ID; return what it saw."""
    from hermes_cli import plugins as _plugins_mod

    seen = []

    async def _handler(args: str) -> str:
        seen.append(get_session_env("HERMES_SESSION_ID"))
        return "ok"

    monkeypatch.setattr(
        _plugins_mod,
        "get_plugin_command_handler",
        lambda name: _handler if name == "planmode" else None,
    )

    event = MessageEvent(text="/planmode on", source=_make_source(), message_id="m1")
    handled, result, command = await runner._hm_dispatch_quick_and_plugin_commands(
        event, event.source, "planmode"
    )
    assert (handled, result, command) == (True, "ok", "planmode")
    return seen


@pytest.mark.asyncio
async def test_plugin_command_binds_the_routing_sessions_id(monkeypatch):
    """With a live routing entry the handler sees its session id, and only inside the scope."""
    store = _FakeStore(entries={"sk-test": SimpleNamespace(session_id="sess-live-42")})
    seen = await _dispatch_and_capture(monkeypatch, _make_runner(store=store))
    assert seen == ["sess-live-42"]
    # The scope is exited after dispatch: the var is explicitly cleared, not left behind.
    assert get_session_env("HERMES_SESSION_ID") == ""


@pytest.mark.asyncio
async def test_plugin_command_without_entry_binds_the_session_key_not_minted(monkeypatch):
    """A brand-new chat's first command has no routing entry: the chat's own key stands in as a
    non-empty, per-chat id instead of every such chat sharing one "" bucket — and no session is
    created on the command's behalf."""
    store = _FakeStore(entries={})
    seen = await _dispatch_and_capture(monkeypatch, _make_runner(store=store))
    assert seen == ["sk-test"]
    assert store._entries == {}


@pytest.mark.asyncio
async def test_plugin_command_follows_a_compression_rotation(monkeypatch):
    """An entry left pointing at a compressed parent binds the lineage tip — the healed id the
    turn's get_or_create_session would route to — without rewriting the entry itself."""
    entry = SimpleNamespace(session_id="sess-compressed-parent")
    store = _FakeStore(entries={"sk-test": entry}, tips={"sess-compressed-parent": "sess-tip-9"})
    seen = await _dispatch_and_capture(monkeypatch, _make_runner(store=store))
    assert seen == ["sess-tip-9"]
    assert entry.session_id == "sess-compressed-parent"  # peek is read-only; the turn heals


@pytest.mark.asyncio
async def test_plugin_command_skips_an_id_ended_mid_run(monkeypatch):
    """An entry whose session ended in state.db while the gateway stayed alive (#54878) heals to
    nothing here (the next turn drops and recovers it), so the handler gets the session key, not
    a dead id that /new will never report."""
    entry = SimpleNamespace(session_id="sess-ended-1")
    store = _FakeStore(entries={"sk-test": entry}, ended_ids=("sess-ended-1",))
    seen = await _dispatch_and_capture(monkeypatch, _make_runner(store=store))
    assert seen == ["sk-test"]
    assert entry.session_id == "sess-ended-1"  # read-only: the entry stays for the turn to drop


@pytest.mark.asyncio
async def test_plugin_command_without_session_store_binds_the_session_key(monkeypatch):
    """A runner with no session store at all still dispatches and binds the derived key — never
    a foreign (cron agent's os.environ) value (#108698 parity)."""
    runner = _make_runner()
    seen = await _dispatch_and_capture(monkeypatch, runner)
    assert seen == [runner._session_key_for_source(_make_source())]
    assert seen[0]  # non-empty: per-chat fallback, not a shared "" bucket
