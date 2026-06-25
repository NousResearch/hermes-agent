"""Unit tests for the inline action-menu registry (``tools.action_gateway``).

Pure-Python: no Telegram client required. Covers registration, label de-duplication
(the duplicate-button bug that motivated #52252), keyboard-row shaping, the 64-byte
``callback_data`` budget, foreign-callback rejection, and resolve()/skip semantics.

The adapter render + callback-dispatch path (``send_action_menu`` / the ``ca:`` branch
in ``_handle_callback_query``) is exercised under CI with the Telegram mocks used by
``test_telegram_approval_buttons.py``; the outbound-delivery wiring seam is tracked in
#52252.
"""
import pytest

from tools import action_gateway as ag


@pytest.fixture(autouse=True)
def _clean_registry():
    ag._sets.clear()
    ag._dispatch_cbs.clear()
    yield
    ag._sets.clear()
    ag._dispatch_cbs.clear()


def test_register_and_resolve():
    s = ag.register([
        {"id": 1, "label": "Prep 2pm meeting"},
        {"id": 2, "label": "Open the lost-permissions thread"},
    ])
    assert len(s.actions) == 2
    chosen = ag.resolve(s.set_id, "1")
    assert chosen is not None and chosen.label == "Prep 2pm meeting"


def test_distinct_actions_never_render_identical_buttons():
    # Same verb, different object — the case that previously surfaced as two
    # identical "investigate" buttons (#52252).
    s = ag.register([
        {"verb": "investigate", "object": "oh-my-pi v16.1.16 impact"},
        {"verb": "investigate", "object": "context-mode change"},
    ])
    labels = [a.label for a in s.actions]
    assert labels[0] != labels[1]
    # And if labels genuinely collide, they are disambiguated:
    s2 = ag.register([{"label": "Open thread"}, {"label": "Open thread"}])
    assert s2.actions[0].label != s2.actions[1].label


def test_keyboard_rows_shape_and_skip():
    s = ag.register([{"id": 1, "label": "a"}, {"id": 2, "label": "b"}])
    rows = ag.build_keyboard_rows(s.set_id)
    assert [r[0][0] for r in rows] == ["a", "b", "Skip"]


def test_callback_data_within_64_byte_cap():
    s = ag.register([{"id": 1, "label": "x" * 300}])  # huge label
    rows = ag.build_keyboard_rows(s.set_id)
    for row in rows:
        for label, cb in row:
            assert len(cb.encode("utf-8")) <= ag.CALLBACK_MAX_BYTES
            assert len(label) <= ag.LABEL_MAX_CHARS + 1  # truncation applied


def test_parse_callback():
    s = ag.register([{"id": 1, "label": "a"}])
    assert ag.parse_callback(f"ca:{s.set_id}:1") == (s.set_id, "1")
    assert ag.parse_callback("ca:abc:skip") == ("abc", "skip")
    # Foreign prefixes (approval / clarify) are not ours:
    assert ag.parse_callback("ea:once:5") is None
    assert ag.parse_callback("cl:abc:0") is None


def test_skip_and_single_resolution():
    s = ag.register([{"id": 1, "label": "a"}, {"id": 2, "label": "b"}])
    assert ag.resolve(s.set_id, "skip") is None
    # A skipped/resolved set cannot fire again (stale button safety):
    assert ag.resolve(s.set_id, "1") is None


def test_resolve_unknown_token():
    s = ag.register([{"id": 1, "label": "a"}])
    assert ag.resolve(s.set_id, "nope") is None
    assert ag.resolve("no-such-set", "1") is None
