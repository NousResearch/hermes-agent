"""Tests for the platform-neutral busy-session button module.

Runner integration is exercised separately (mocked adapter); these
tests cover the standalone wire format, primitive set, and reaction
map so the contract stays stable across platforms.
"""

import pytest

from gateway.busy_session_buttons import (
    BUSY_SESSION_PRIMITIVES,
    BUTTON_LABELS,
    BusySessionButton,
    CALLBACK_PREFIX,
    PRIMITIVE_INTERRUPT,
    PRIMITIVE_STEER,
    PRIMITIVE_STOP,
    REACTION_BY_PRIMITIVE,
    REACTION_INTERRUPT,
    REACTION_STEER,
    REACTION_STOP,
    build_buttons,
    parse_callback_data,
    reaction_for,
    status_text,
)


# --- build_buttons ----------------------------------------------------------


def test_build_buttons_returns_three_in_canonical_order():
    buttons = build_buttons("tg/123/456")
    assert len(buttons) == 3
    assert [b.primitive for b in buttons] == [
        PRIMITIVE_STEER,
        PRIMITIVE_INTERRUPT,
        PRIMITIVE_STOP,
    ]


def test_build_buttons_labels_match_primitive_table():
    buttons = build_buttons("session-x")
    for b in buttons:
        assert b.label == BUTTON_LABELS[b.primitive]


def test_build_buttons_callback_data_uses_bs_prefix():
    buttons = build_buttons("session-x")
    for b in buttons:
        assert b.callback_data.startswith(f"{CALLBACK_PREFIX}:")
        assert b.callback_data.endswith(":session-x")


def test_build_buttons_returned_objects_are_immutable():
    buttons = build_buttons("session-x")
    with pytest.raises((AttributeError, Exception)):
        buttons[0].primitive = "tampered"  # type: ignore[misc]


# --- parse_callback_data ----------------------------------------------------


def test_parse_callback_data_round_trips_through_build_buttons():
    buttons = build_buttons("tg/789/threadX")
    for b in buttons:
        parsed = parse_callback_data(b.callback_data)
        assert parsed == (b.primitive, "tg/789/threadX")


def test_parse_callback_data_returns_none_for_other_prefixes():
    # Other Telegram callback prefixes used elsewhere (ea:, sc:, mp:, etc.)
    # must not be mistaken for busy-session callbacks.
    for foreign in ["ea:approve:42", "sc:once:7", "mp:select:gpt-5", "random"]:
        assert parse_callback_data(foreign) is None


def test_parse_callback_data_returns_none_for_unknown_primitive():
    assert parse_callback_data("bs:steamroll:session-x") is None
    assert parse_callback_data("bs::session-x") is None


def test_parse_callback_data_returns_none_for_empty_session_key():
    assert parse_callback_data("bs:steer:") is None


def test_parse_callback_data_returns_none_for_empty_or_none():
    assert parse_callback_data(None) is None
    assert parse_callback_data("") is None


def test_parse_callback_data_handles_session_keys_with_colons():
    # Session keys can contain colons (e.g. tg:123:thread:42). The split
    # is bounded so the third segment captures the entire remaining key.
    parsed = parse_callback_data("bs:interrupt:tg:123:thread:42")
    assert parsed == (PRIMITIVE_INTERRUPT, "tg:123:thread:42")


# --- reaction_for / REACTION_BY_PRIMITIVE -----------------------------------


def test_reaction_for_each_primitive_distinct():
    reactions = {reaction_for(p) for p in BUSY_SESSION_PRIMITIVES}
    assert len(reactions) == 3
    assert reaction_for(PRIMITIVE_STEER) == REACTION_STEER
    assert reaction_for(PRIMITIVE_INTERRUPT) == REACTION_INTERRUPT
    assert reaction_for(PRIMITIVE_STOP) == REACTION_STOP


def test_reaction_for_unknown_primitive_returns_none():
    assert reaction_for("badverb") is None
    assert reaction_for("") is None


def test_reaction_table_keys_match_primitives():
    assert set(REACTION_BY_PRIMITIVE.keys()) == set(BUSY_SESSION_PRIMITIVES)


# --- status_text ------------------------------------------------------------


def test_status_text_includes_primitive_emoji():
    assert REACTION_STEER in status_text(PRIMITIVE_STEER)
    assert REACTION_INTERRUPT in status_text(PRIMITIVE_INTERRUPT)
    assert REACTION_STOP in status_text(PRIMITIVE_STOP)


def test_status_text_unknown_primitive_returns_empty():
    assert status_text("badverb") == ""


# --- Constants invariants ---------------------------------------------------


def test_primitives_match_button_labels_keys():
    assert set(BUTTON_LABELS.keys()) == set(BUSY_SESSION_PRIMITIVES)


def test_button_labels_short_enough_for_telegram():
    # Telegram clips inline-keyboard labels around 14 chars in practice.
    for primitive, label in BUTTON_LABELS.items():
        assert len(label) <= 14, f"{primitive} label {label!r} too long"


def test_callback_data_under_telegram_64_byte_cap():
    # Telegram's callback_data limit is 64 bytes. A long session key plus
    # primitive plus prefix must fit comfortably.
    long_session = "tg/9999999999/" + ("x" * 30)
    buttons = build_buttons(long_session)
    for b in buttons:
        assert len(b.callback_data.encode("utf-8")) <= 64, b.callback_data


def test_long_session_key_uses_hashed_handle_under_64_bytes():
    """Real group/forum session keys overflow Telegram's 64-byte cap.

    ``build_buttons_with_handles`` switches to a stable short hash and
    surfaces the handle map so the platform can resolve it on tap.
    """
    from gateway.busy_session_buttons import (
        CALLBACK_MAX_BYTES,
        HANDLE_SIGIL,
        build_buttons_with_handles,
        parse_callback_data,
    )

    very_long = "agent:main:telegram:supergroup:-1001234567890:thread:42:user:9876543210"
    spec = build_buttons_with_handles(very_long)

    # Every callback_data fits Telegram's hard cap.
    for b in spec.buttons:
        assert len(b.callback_data.encode("utf-8")) <= CALLBACK_MAX_BYTES, b.callback_data
    # Handle map is populated so the runner can resolve back.
    assert spec.handle_map, "Long key should use hashed handle"
    handle = next(iter(spec.handle_map.keys()))
    assert handle.startswith(HANDLE_SIGIL)
    assert spec.handle_map[handle] == very_long
    # parse_callback_data with the resolver returns the original key.
    parsed = parse_callback_data(spec.buttons[0].callback_data, handle_resolver=spec.handle_map)
    assert parsed == (spec.buttons[0].primitive, very_long)


def test_short_session_key_does_not_use_handle():
    from gateway.busy_session_buttons import build_buttons_with_handles

    short = "agent:main:telegram:dm:42:42"
    spec = build_buttons_with_handles(short)
    assert spec.handle_map == {}
    for b in spec.buttons:
        assert short in b.callback_data


def test_parse_callback_data_unresolvable_handle_returns_none():
    from gateway.busy_session_buttons import parse_callback_data
    # Hash-prefixed payload but no resolver supplied.
    assert parse_callback_data("bs:steer:#deadbeefcafe") is None
    # Hash-prefixed payload with empty resolver.
    assert parse_callback_data("bs:steer:#deadbeefcafe", handle_resolver={}) is None
