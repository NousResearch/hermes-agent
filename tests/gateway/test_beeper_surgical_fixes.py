"""Focused tests for the two Beeper-compat bugs fixed in the surgical adapter.

Bug 2 (reactions): Beeper sends "❌️" (U+274C U+FE0F); main's exact-match lookup
    misses it against the choice map's "❌" (U+274C) → "not valid for this prompt".
    Fix: normalize U+FE0E/U+FE0F on both sides before matching.

Bug 1 (threading): a nested reply (m.in_reply_to → a bot reply that is itself in
    a thread) carries no direct m.thread relation, so main creates a NEW synthetic
    thread rooted at this event → orphan thread invisible in Element.
    Fix: walk the reply chain to the true top-level thread root.
"""
import asyncio
import types
from unittest.mock import MagicMock, AsyncMock

from plugins.platforms.matrix.adapter import (
    MatrixAdapter,
    _normalize_reaction_key,
)


# ── Bug 2: variation-selector normalization ─────────────────────────────────

def test_normalize_strips_fe0f():
    # Beeper's "❌️" = U+274C U+FE0F
    assert _normalize_reaction_key("❌\ufe0f") == "❌"


def test_normalize_strips_fe0e():
    assert _normalize_reaction_key("❌\ufe0e") == "❌"


def test_normalize_idempotent_on_plain():
    assert _normalize_reaction_key("✅") == "✅"
    assert _normalize_reaction_key("🌀") == "🌀"


def test_normalize_does_not_fold_zwj_or_skin_tone():
    # ZWJ sequences and skin tones change emoji identity — must NOT be folded.
    assert _normalize_reaction_key("🧑\ufe0f") == "🧑"
    assert _normalize_reaction_key("👍\U0001F3FD") == "👍\U0001F3FD"  # skin tone preserved


def _make_adapter():
    from plugins.platforms.matrix.adapter import MatrixAdapter
    from gateway.config import PlatformConfig
    config = PlatformConfig(
        enabled=True,
        token="syt_test_token",
        extra={
            "homeserver": "https://example.com",
            "user_id": "@hermes:example.com",
        },
    )
    adapter = MatrixAdapter(config)
    adapter._client = MagicMock()
    return adapter


def _make_prompt(chat_id, choices):
    p = MagicMock()
    p.resolved = False
    p.expires_at = None
    p.chat_id = chat_id
    p.choices = choices
    p.prompt_id = "p1"
    return p


def _wire_claim(adapter, room_id, reacts_to, key, sender, prompt):
    """Set up _claim_reaction_prompt to run the real matching path with a stubbed
    reactor-validation (returns True) and a recording invalid-feedback sender."""
    invalid = []

    async def fake_validate(room_id, target_event_id, sender, prompt, prompt_label):
        return True  # authorized reactor — let matching proceed

    async def fake_invalid(room_id, target_event_id, text):
        invalid.append(text)

    adapter._validate_matrix_prompt_reactor = fake_validate
    adapter._send_invalid_reaction_feedback = fake_invalid

    async def on_expired(room_id, reacts_to, prompt):
        pass

    registry = {reacts_to: prompt}
    label = "approval"
    invalid_text = "That reaction is not valid for this approval prompt."
    result = asyncio.run(adapter._claim_reaction_prompt(
        registry, room_id, reacts_to, key, sender, label, invalid_text, on_expired))
    return result, invalid


def test_claim_reaction_beeper_fe0f_key_maps_to_choice():
    """Beeper sends ❌️ (U+274C U+FE0F); it must match the '❌'→deny choice."""
    adapter = _make_adapter()
    room = "!room:example.com"
    prompt = _make_prompt(room, {"✅": "once", "🌀": "session", "♾️": "always", "❌": "deny"})
    (handled, got_prompt, selection), invalid = _wire_claim(
        adapter, room, "$reacted:example.com", "❌\ufe0f", "@beeper:beeper.com", prompt)
    assert handled is True
    assert got_prompt is prompt
    assert selection == "deny", f"expected 'deny', got {selection!r} — Bug 2 not fixed"
    assert invalid == [], "should NOT send invalid feedback — key should match"


def test_claim_reaction_element_plain_key_still_works():
    """Element sends plain ❌ (U+274C); must still map to deny (no regression)."""
    adapter = _make_adapter()
    room = "!r:e.com"
    prompt = _make_prompt(room, {"✅": "once", "❌": "deny"})
    (handled, _, selection), invalid = _wire_claim(
        adapter, room, "$x:e.com", "❌", "@u:e.com", prompt)
    assert selection == "deny"
    assert invalid == []


def test_claim_reaction_unknown_key_still_rejected():
    """A genuinely unknown reaction (not in the map) must still be rejected."""
    adapter = _make_adapter()
    room = "!r:e.com"
    prompt = _make_prompt(room, {"✅": "once", "❌": "deny"})
    (handled, _, selection), invalid = _wire_claim(
        adapter, room, "$x:e.com", "🎉", "@u:e.com", prompt)
    assert handled is True
    assert selection is None, "unknown key must not produce a selection"
    assert len(invalid) == 1, "invalid feedback must be sent for unknown key"


# ── Bug 1: reply-chain walk to true thread root ─────────────────────────────

def _wire_walk(adapter, events):
    async def fake_get_event(room_id, event_id):
        ev = events.get(event_id)
        if ev is None:
            raise Exception("not found")
        m = types.SimpleNamespace()
        m.room_id = room_id
        m.event_id = event_id
        m.type = "m.room.message"
        m.sender = ev["sender"]
        m.content = ev["content"]
        return m

    adapter._client.get_event = fake_get_event
    crypto = MagicMock()
    crypto.decrypt_megolm_event = AsyncMock(side_effect=lambda event: event)
    adapter._client.crypto = crypto


def test_resolve_reply_target_stops_at_thread_root():
    """
    Chain: $8Dm (in_reply_to $hCin) → $hCin (m.thread → $vP3I) → $vP3I (top-level).
    Walking from $8Dm must resolve to $vP3I (the real thread root), NOT $hCin or $8Dm.
    """
    adapter = _make_adapter()
    events = {
        "$8Dm": {"sender": "@beeper:beeper.com", "content": {
            "m.relates_to": {"m.in_reply_to": {"event_id": "$hCin"}}}},
        "$hCin": {"sender": "@hermes:example.com", "content": {
            "m.relates_to": {"rel_type": "m.thread", "event_id": "$vP3I"}}},
        "$vP3I": {"sender": "@user:example.com", "content": {}},
    }
    _wire_walk(adapter, events)
    root, _ = asyncio.run(adapter._resolve_reply_target("!room:example.com", "$8Dm"))
    assert root == "$vP3I", f"expected $vP3I, got {root}"


def test_resolve_reply_target_unresolved_when_no_thread_evidence():
    """
    A plain reply chain with NO m.thread anywhere must return UNRESOLVED (None),
    so main's synthetic-thread policy still applies (ordinary replies stay non-threads).
    Astra's refinement: never fabricate a thread root from a bare reply chain.
    """
    adapter = _make_adapter()
    events = {
        "$a": {"sender": "@u:e.com", "content": {
            "m.relates_to": {"m.in_reply_to": {"event_id": "$b"}}}},
        "$b": {"sender": "@u:e.com", "content": {
            "m.relates_to": {"m.in_reply_to": {"event_id": "$c"}}}},
        "$c": {"sender": "@u:e.com", "content": {}},
    }
    _wire_walk(adapter, events)
    root, _ = asyncio.run(adapter._resolve_reply_target("!room:example.com", "$a"))
    assert root is None, f"expected None (unresolved), got {root}"


# ── Cross-bot slash-command gating (multi-bot room) ─────────────────────────
# Live bug (room !VRZsEmfKaKFzKSmNgl): a thread created by @mentioning @hermes
# received a /stop that BOTH hermes and healthbot answered, then the follow-up
# question reached both too. Root cause: main's require_mention gate never
# drops a slash command, so a foreign bot (never in the thread) processes the
# /stop, marks the thread in its own tracker, and then answers follow-ups.
# Fix: a slash command inside a thread this bot did not start is dropped
# unless the bot is @mentioned.


class _FakeThreads:
    """Minimal stand-in for ThreadParticipationTracker: membership only."""

    def __init__(self, roots):
        self._roots = set(roots)

    def __contains__(self, thread_id):
        return thread_id in self._roots

    async def mark_async(self, thread_id):
        self._roots.add(thread_id)


def _make_gate_adapter(roots=(), user_id="@hermes:example.com"):
    """Adapter wired for _resolve_message_context with require_mention=True."""
    adapter = _make_adapter()
    adapter._user_id = user_id
    adapter._require_mention = True
    adapter._free_rooms = set()
    adapter._allowed_rooms = set()
    adapter._threads = _FakeThreads(roots)
    adapter._is_dm_room = AsyncMock(return_value=False)
    adapter._resolve_room_identity = AsyncMock(
        return_value=MagicMock(display_name="Room"))
    adapter._get_display_name = AsyncMock(return_value="Alice")
    adapter._background_read_receipt = MagicMock()
    return adapter


def test_command_in_foreign_thread_is_dropped():
    """healthbot's case: /stop in hermes's thread, not mentioned → dropped."""
    adapter = _make_gate_adapter(roots=())  # this bot did NOT start the thread
    ctx = asyncio.run(adapter._resolve_message_context(
        room_id="!room:example.com",
        sender="@admin:example.com",
        event_id="$stop1",
        body="/stop",
        source_content={"body": "/stop"},
        relates_to={"rel_type": "m.thread", "event_id": "$root"},
    ))
    assert ctx is None, "a foreign bot must not process a /stop in someone else's thread"


def test_command_in_own_thread_is_processed():
    """hermes's case: /stop in its own thread (in_bot_thread) → processed."""
    adapter = _make_gate_adapter(roots=("$root",))  # this bot DID start the thread
    ctx = asyncio.run(adapter._resolve_message_context(
        room_id="!room:example.com",
        sender="@admin:example.com",
        event_id="$stop2",
        body="/stop",
        source_content={"body": "/stop"},
        relates_to={"rel_type": "m.thread", "event_id": "$root"},
    ))
    assert ctx is not None, "the thread's own bot must still process /stop"


def test_mentioned_command_in_foreign_thread_is_processed():
    """@healthbot /stop in a foreign thread: explicit mention wins → processed.
    The adapter IS healthbot; it was not in the thread but is explicitly mentioned."""
    adapter = _make_gate_adapter(roots=(), user_id="@healthbot:example.com")
    ctx = asyncio.run(adapter._resolve_message_context(
        room_id="!room:example.com",
        sender="@admin:example.com",
        event_id="$stop3",
        body="@healthbot /stop",
        source_content={"body": "@healthbot /stop",
                       "m.mentions": {"user_ids": ["@healthbot:example.com"]}},
        relates_to={"rel_type": "m.thread", "event_id": "$root"},
    ))
    assert ctx is not None, "an explicit @mention must still reach the bot"


def test_toplevel_command_still_broadcasts():
    """Top-level /stop (no thread) is unchanged from main: both bots process it."""
    adapter = _make_gate_adapter(roots=())
    ctx = asyncio.run(adapter._resolve_message_context(
        room_id="!room:example.com",
        sender="@admin:example.com",
        event_id="$stop4",
        body="/stop",
        source_content={"body": "/stop"},
        relates_to={},  # no thread relation
    ))
    assert ctx is not None, "top-level /stop must keep main's broadcast behavior"


def test_plain_unmentioned_in_foreign_thread_still_dropped():
    """Regression guard: a plain (non-command) unmentioned message in a foreign
    thread was already dropped by main; the fix must not change that."""
    adapter = _make_gate_adapter(roots=())
    ctx = asyncio.run(adapter._resolve_message_context(
        room_id="!room:example.com",
        sender="@admin:example.com",
        event_id="$q1",
        body="why did health reply to this thread",
        source_content={"body": "why did health reply to this thread"},
        relates_to={"rel_type": "m.thread", "event_id": "$root"},
    ))
    assert ctx is None, "plain unmentioned message in a foreign thread stays dropped"
