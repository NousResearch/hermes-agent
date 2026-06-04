"""Tests for the Matrix reaction-menu choreography (present_menu capability).

Mirrors the structure of test_matrix_exec_approval.py /
test_matrix_approval_reaction_fail_closed.py. The reaction-menu path is built
PARALLEL to the exec-approval path; these tests also assert the approval path is
untouched (the do-not-refactor guard) by exercising both on the same adapter.
"""

import asyncio
import sys
import types
from collections import deque
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


# ---------------------------------------------------------------------------
# Stub mautrix so gateway.platforms.matrix imports without the SDK.
# ---------------------------------------------------------------------------

def _stub_mautrix():
    stub = types.ModuleType("mautrix")
    for sub in ("mautrix.types", "mautrix.client", "mautrix.client.api",
                "mautrix.errors", "mautrix.crypto", "mautrix.util",
                "mautrix.util.config"):
        sys.modules.setdefault(sub, types.ModuleType(sub))
    sys.modules.setdefault("mautrix", stub)
    m = sys.modules["mautrix.types"]
    for attr in (
        "ContentURI", "EventID", "EventType", "PaginationDirection",
        "PresenceState", "RoomCreatePreset", "RoomID", "SyncToken",
        "TrustState", "UserID",
    ):
        if not hasattr(m, attr):
            setattr(m, attr, str)


_stub_mautrix()

from gateway.config import Platform  # noqa: E402
from gateway.platforms.matrix import (  # noqa: E402
    MatrixAdapter,
    _MatrixApprovalPrompt,
    _MatrixMenuPrompt,
)
from gateway.session import SessionSource  # noqa: E402
from tools import reaction_menu_gateway as rmg  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_registry(tmp_path):
    rmg.reset_state()
    rmg.set_db_path(tmp_path / "menus.db")
    yield
    rmg.reset_state()
    rmg.set_db_path(None)


def _make_adapter(allowed=("@dan:x",)):
    """Bare MatrixAdapter with only the state the menu path touches."""
    adapter = object.__new__(MatrixAdapter)
    adapter._client = SimpleNamespace()
    adapter._user_id = "@bot:x"
    adapter._allowed_user_ids = set(allowed)
    adapter._approval_reaction_map = {"✅": "once", "❎": "deny"}
    adapter._approval_prompts_by_event = {}
    adapter._approval_prompt_by_session = {}
    adapter._menu_prompts_by_event = {}
    adapter._menu_reload_by_event = {}
    adapter._processed_events = deque(maxlen=512)
    adapter._processed_events_set = set()
    adapter._reaction_redaction_delay_seconds = 0.0
    adapter._reaction_redaction_tasks = set()
    # Async surface
    adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="$menu1"))
    adapter._send_reaction = AsyncMock(return_value="$seed")
    adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True, message_id="$edit"))
    # _message_handler is the non-None sentinel _inject_menu_choice guards on;
    # the synthetic turn is dispatched through handle_message (the full inbound
    # path that also delivers the response), so that's what the tests assert on.
    adapter._message_handler = AsyncMock()
    adapter.handle_message = AsyncMock()
    # Record reaction redactions instead of scheduling real tasks.
    adapter._redactions = []
    adapter._schedule_reaction_redaction = lambda room, evt, reason="": adapter._redactions.append((evt, reason))
    return adapter


def _source():
    return SessionSource(
        platform=Platform.MATRIX, chat_id="!room:x", chat_type="dm",
        user_id="@dan:x", user_name="dan",
    )


def _options():
    return [
        {"emoji": "📖", "label": "Read next", "payload": "read next passage", "terminal": False},
        {"emoji": "🛑", "label": "Stop", "payload": "stop reading", "terminal": True},
    ]


def _reaction_event(sender, reacts_to, key):
    return SimpleNamespace(
        sender=sender,
        event_id=f"$r-{sender}-{reacts_to}-{key}",
        room_id="!room:x",
        content={"m.relates_to": {"event_id": reacts_to, "key": key}},
    )


def _present(adapter, menu_id="m1"):
    """Drive send_reaction_menu and return the menu's message_id."""
    result = asyncio.run(adapter.send_reaction_menu(
        chat_id="!room:x", menu_id=menu_id, prompt="Continue?",
        options=_options(), session_key="s1", source=_source(), context_id="story",
    ))
    return result


# ---------------------------------------------------------------------------
# 1. present seeds emoji
# ---------------------------------------------------------------------------

def test_reaction_menu_present_seeds_emoji():
    adapter = _make_adapter()
    result = _present(adapter)
    assert result.success
    menu = adapter._menu_prompts_by_event["$menu1"]
    assert menu.menu_id == "m1"
    # One seed per option, in order.
    seeded = [c.args[2] for c in adapter._send_reaction.await_args_list]
    assert seeded == ["📖", "🛑"]
    assert set(menu.bot_reaction_events) == {"📖", "🛑"}
    # Registered in the consumer-agnostic store + persisted.
    assert rmg.get_by_message("$menu1").context_id == "story"


# ---------------------------------------------------------------------------
# 2. tap injects synthetic [menu-choice] turn, nothing blocked
# ---------------------------------------------------------------------------

def test_reaction_menu_tap_injects_synthetic_turn():
    adapter = _make_adapter()
    _present(adapter)
    asyncio.run(adapter._on_reaction(_reaction_event("@dan:x", "$menu1", "📖")))

    adapter.handle_message.assert_awaited_once()
    # Must go through handle_message (which delivers the response), NOT the bare
    # _message_handler (which runs the turn but never sends the reply). Guards
    # the "tap rewrote the message but nothing else happened" regression.
    adapter._message_handler.assert_not_awaited()
    injected = adapter.handle_message.await_args.args[0]
    assert injected.text.startswith(rmg.MENU_CHOICE_MARKER)
    assert "read next passage" in injected.text
    assert injected.internal is True
    assert injected.source.user_id == "@dan:x"


# ---------------------------------------------------------------------------
# 3. collapse redacts bot seeds, keeps user reaction, seeds ♻️ (non-terminal)
# ---------------------------------------------------------------------------

def test_reaction_menu_collapse_redacts_bot_seeds_keeps_user_react():
    adapter = _make_adapter()
    _present(adapter)
    asyncio.run(adapter._on_reaction(_reaction_event("@dan:x", "$menu1", "📖")))

    # Both bot seeds scheduled for redaction (the user's own reaction is never
    # touched — we only redact ids we seeded).
    redacted = {evt for evt, _ in adapter._redactions}
    assert "$seed" in redacted
    # Non-terminal choice → reload reaction seeded on the collapsed message.
    assert "$menu1" in adapter._menu_reload_by_event
    # Collapse edit happened.
    adapter.edit_message.assert_awaited()


# ---------------------------------------------------------------------------
# 4. terminal option → no reload reaction
# ---------------------------------------------------------------------------

def test_reaction_menu_terminal_option_no_reload():
    adapter = _make_adapter()
    _present(adapter)
    asyncio.run(adapter._on_reaction(_reaction_event("@dan:x", "$menu1", "🛑")))

    assert "$menu1" not in adapter._menu_reload_by_event
    adapter.handle_message.assert_awaited_once()
    assert "stop reading" in adapter.handle_message.await_args.args[0].text


# ---------------------------------------------------------------------------
# 5. reload ♻️ spawns a fresh menu, no model turn
# ---------------------------------------------------------------------------

def test_reaction_menu_reload_spawns_fresh_menu():
    adapter = _make_adapter()
    _present(adapter)
    # First, a non-terminal choice to seed the ♻️ reload handle.
    asyncio.run(adapter._on_reaction(_reaction_event("@dan:x", "$menu1", "📖")))
    adapter.handle_message.reset_mock()
    adapter.send.reset_mock()
    adapter.send.return_value = SimpleNamespace(success=True, message_id="$menu2")

    # Tap ♻️ on the collapsed message.
    asyncio.run(adapter._on_reaction(_reaction_event("@dan:x", "$menu1", rmg.RELOAD_EMOJI)))

    # A fresh menu message was sent; NO synthetic turn injected.
    adapter.send.assert_awaited()
    adapter.handle_message.assert_not_awaited()
    assert "$menu2" in adapter._menu_prompts_by_event
    # The one-shot reload handle is consumed.
    assert "$menu1" not in adapter._menu_reload_by_event


# ---------------------------------------------------------------------------
# 6. fail-closed: unauthorized tap leaves the menu live + unresolved
# ---------------------------------------------------------------------------

def test_reaction_menu_fail_closed_unauthorized(monkeypatch):
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
    adapter = _make_adapter(allowed=("@dan:x",))
    _present(adapter)
    asyncio.run(adapter._on_reaction(_reaction_event("@mallory:x", "$menu1", "📖")))

    adapter.handle_message.assert_not_awaited()
    assert not adapter._menu_prompts_by_event["$menu1"].resolved
    assert rmg.get_by_message("$menu1") is not None


# ---------------------------------------------------------------------------
# 7. dedup: a double-tap injects exactly one turn
# ---------------------------------------------------------------------------

def test_reaction_menu_dedup_double_tap():
    adapter = _make_adapter()
    _present(adapter)
    ev1 = _reaction_event("@dan:x", "$menu1", "📖")
    ev2 = SimpleNamespace(sender="@dan:x", event_id="$r-second", room_id="!room:x",
                          content={"m.relates_to": {"event_id": "$menu1", "key": "📖"}})
    asyncio.run(adapter._on_reaction(ev1))
    asyncio.run(adapter._on_reaction(ev2))
    assert adapter.handle_message.await_count == 1


# ---------------------------------------------------------------------------
# 8. persistence: a menu survives a simulated gateway restart
# ---------------------------------------------------------------------------

def test_reaction_menu_persists_across_restart():
    adapter = _make_adapter()
    _present(adapter)

    # Simulate restart: new adapter process, in-memory state gone.
    rmg.reset_state()
    adapter2 = _make_adapter()
    adapter2._restore_persisted_menus()

    assert "$menu1" in adapter2._menu_prompts_by_event
    restored = adapter2._menu_prompts_by_event["$menu1"]
    assert restored.source.user_id == "@dan:x"
    # And a tap still resolves + injects after restore.
    asyncio.run(adapter2._on_reaction(_reaction_event("@dan:x", "$menu1", "📖")))
    adapter2.handle_message.assert_awaited_once()


# ---------------------------------------------------------------------------
# 9. many live menus resolve independently
# ---------------------------------------------------------------------------

def test_reaction_menu_many_live_menus_independent():
    adapter = _make_adapter()
    adapter.send.return_value = SimpleNamespace(success=True, message_id="$mA")
    asyncio.run(adapter.send_reaction_menu(
        chat_id="!room:x", menu_id="mA", prompt="A?", options=_options(),
        session_key="s1", source=_source()))
    adapter.send.return_value = SimpleNamespace(success=True, message_id="$mB")
    asyncio.run(adapter.send_reaction_menu(
        chat_id="!room:x", menu_id="mB", prompt="B?", options=_options(),
        session_key="s1", source=_source()))

    # Resolve menu A; menu B stays live and resolvable.
    asyncio.run(adapter._on_reaction(_reaction_event("@dan:x", "$mA", "📖")))
    assert "$mA" not in adapter._menu_prompts_by_event
    assert "$mB" in adapter._menu_prompts_by_event
    asyncio.run(adapter._on_reaction(_reaction_event("@dan:x", "$mB", "📖")))
    assert adapter.handle_message.await_count == 2


# ---------------------------------------------------------------------------
# Do-not-refactor guard: the approval path still resolves on the same adapter
# ---------------------------------------------------------------------------

def test_approval_path_untouched_by_menu_branch():
    import tools.approval as _approval

    adapter = _make_adapter()
    adapter._redact_bot_approval_reactions = AsyncMock()
    adapter._approval_prompts_by_event["$appr"] = _MatrixApprovalPrompt(
        session_key="s1", chat_id="!room:x", message_id="$appr",
    )
    adapter._approval_prompt_by_session["s1"] = "$appr"

    calls = {}
    _orig = _approval.resolve_gateway_approval
    _approval.resolve_gateway_approval = lambda sk, choice: calls.setdefault("c", (sk, choice)) or 1
    try:
        asyncio.run(adapter._on_reaction(_reaction_event("@dan:x", "$appr", "✅")))
    finally:
        _approval.resolve_gateway_approval = _orig

    assert calls["c"] == ("s1", "once")
    assert "$appr" not in adapter._approval_prompts_by_event
    # The menu handler must NOT have fired for an approval reaction.
    adapter.handle_message.assert_not_awaited()
