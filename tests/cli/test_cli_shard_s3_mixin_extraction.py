"""Regression tests for the cli.py shard-s3 extraction (wave 1, witness w1a).

Covers the pure methods moved into ``CLIModalConfirmMixin`` and
``CLIModelSwitchMixin`` (slash-confirm choice normalization, model-picker
viewport arithmetic, slash-confirm panel fragment rendering) plus the mixin
seam itself (identity + MRO).  All methods are driven through the ``_bound``
stand-in pattern used across ``tests/cli`` — no full ``HermesCLI``
construction required.
"""

from __future__ import annotations

import textwrap
from types import SimpleNamespace

import pytest

from hermes_cli.cli_modal_confirm_mixin import CLIModalConfirmMixin
from hermes_cli.cli_model_switch_mixin import CLIModelSwitchMixin


def _bound(fn, instance):
    """Bind an unbound method to a stand-in instance."""
    return fn.__get__(instance, type(instance))


# ---------------------------------------------------------------------------
# Seam: identity + MRO
# ---------------------------------------------------------------------------


def test_seam_modal_confirm_members_resolve_from_hermescli():
    """HermesCLI.<member> IS CLIModalConfirmMixin.<member> for moved members."""
    import cli

    for name in (
        "_run_curses_picker",
        "_prompt_text_input",
        "_prompt_text_input_modal",
        "_submit_slash_confirm_response",
        "_normalize_slash_confirm_choice",
        "_get_slash_confirm_display_fragments",
    ):
        assert getattr(cli.HermesCLI, name) is getattr(CLIModalConfirmMixin, name), name


def test_seam_model_switch_members_resolve_from_hermescli():
    """HermesCLI.<member> IS CLIModelSwitchMixin.<member> for moved members."""
    import cli

    for name in (
        "_open_model_picker",
        "_confirm_expensive_model_switch",
        "_close_model_picker",
        "_snapshot_model_runtime",
        "_restore_model_runtime_snapshot",
        "_compute_model_picker_viewport",
        "_apply_model_switch_result",
        "_handle_model_picker_selection",
        "_handle_model_switch",
    ):
        assert getattr(cli.HermesCLI, name) is getattr(CLIModelSwitchMixin, name), name


def test_seam_both_mixins_in_mro_after_existing_mixins():
    """Both new mixins are in HermesCLI.__mro__ after the pre-existing ones."""
    import cli

    mro = cli.HermesCLI.__mro__
    assert CLIModalConfirmMixin in mro
    assert CLIModelSwitchMixin in mro
    assert mro.index(CLIModalConfirmMixin) > mro.index(cli.CLIAgentSetupMixin)
    assert mro.index(CLIModalConfirmMixin) > mro.index(cli.CLICommandsMixin)
    assert mro.index(CLIModalConfirmMixin) > mro.index(cli.CLIBillingMixin)
    assert mro.index(CLIModelSwitchMixin) > mro.index(cli.CLIBillingMixin)


# ---------------------------------------------------------------------------
# _normalize_slash_confirm_choice (pure)
# ---------------------------------------------------------------------------

_CHOICES = [
    ("once", "Switch anyway", "Use this model for the current session."),
    ("always", "Remember", "Always allow this action."),
    ("cancel", "Cancel", "Keep the current state."),
]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        ("", None),
        ("   ", None),
        ("1", "once"),
        ("once", "once"),
        ("approve", "once"),
        ("yes", "once"),
        ("y", "once"),
        ("ok", "once"),
        ("2", "always"),
        ("always", "always"),
        ("remember", "always"),
        ("3", "cancel"),
        ("cancel", "cancel"),
        ("nevermind", "cancel"),
        ("no", "cancel"),
        ("n", "cancel"),
        ("ONCE", "once"),
        ("  Yes ", "once"),
        ("bogus", None),
        ("4", None),
    ],
)
def test_normalize_slash_confirm_choice(raw, expected):
    fn = CLIModalConfirmMixin._normalize_slash_confirm_choice
    assert _bound(fn, SimpleNamespace())(raw, _CHOICES) == expected


def test_normalize_slash_confirm_choice_matches_value_not_label():
    """A choice whose value is not a standard alias resolves by exact value."""
    choices = [("once", "Once"), ("custom-flag", "Custom label")]
    fn = CLIModalConfirmMixin._normalize_slash_confirm_choice
    assert _bound(fn, SimpleNamespace())("custom-flag", choices) == "custom-flag"
    assert _bound(fn, SimpleNamespace())("Custom label", choices) is None


# ---------------------------------------------------------------------------
# _compute_model_picker_viewport (pure staticmethod)
# ---------------------------------------------------------------------------


def test_compute_model_picker_viewport_fits_without_scroll():
    # _compute_model_picker_viewport is a @staticmethod — call it directly.
    fn = CLIModelSwitchMixin._compute_model_picker_viewport
    assert fn(selected=2, scroll_offset=0, n=3, term_rows=30) == (0, 3)


def test_compute_model_picker_viewport_scrolls_forward():
    fn = CLIModelSwitchMixin._compute_model_picker_viewport
    # term_rows=20 -> max_visible = 20-6-6 = 8; n=12 > 8 -> scrolls
    assert fn(selected=10, scroll_offset=0, n=12, term_rows=20) == (3, 8)


def test_compute_model_picker_viewport_scrolls_back():
    fn = CLIModelSwitchMixin._compute_model_picker_viewport
    assert fn(selected=0, scroll_offset=6, n=12, term_rows=20) == (0, 8)


def test_compute_model_picker_viewport_clamps_offset():
    fn = CLIModelSwitchMixin._compute_model_picker_viewport
    # n=4 fits in max_visible -> (0, 4) regardless of stale offset
    assert fn(selected=2, scroll_offset=9, n=4, term_rows=20) == (0, 4)


def test_compute_model_picker_viewport_min_visible_floor():
    fn = CLIModelSwitchMixin._compute_model_picker_viewport
    # term_rows=10 -> max_visible = max(3, -2) = 3
    assert fn(selected=0, scroll_offset=0, n=5, term_rows=10) == (0, 3)


# ---------------------------------------------------------------------------
# _get_slash_confirm_display_fragments (pure-ish renderer, no app required)
# ---------------------------------------------------------------------------


def _make_fragment_self(state):
    return SimpleNamespace(
        _slash_confirm_state=state,
        _slash_confirm_deadline=0,
        _app=None,
    )


def test_get_slash_confirm_display_fragments_no_state_returns_empty():
    fn = CLIModalConfirmMixin._get_slash_confirm_display_fragments
    assert _bound(fn, SimpleNamespace(_slash_confirm_state=None))() == []


def test_get_slash_confirm_display_fragments_renders_title_and_choices():
    fn = CLIModalConfirmMixin._get_slash_confirm_display_fragments
    self_ = _make_fragment_self(
        {
            "title": "Confirm",
            "detail": "Do the thing?",
            "choices": [
                ("once", "Once", "Run it now"),
                ("cancel", "Cancel", "Abort"),
            ],
            "selected": 0,
        }
    )
    fragments = _bound(fn, self_)()
    text = "".join(style_text for _style, style_text in fragments)
    assert "Confirm" in text
    assert "Do the thing?" in text
    assert "[1] Once" in text
    assert "[2] Cancel" in text
    assert "❯" in text  # marker on the selected row
    # top border opens the box, bottom border closes it
    assert "╭" in text and "╰" in text


def test_get_slash_confirm_display_fragments_truncates_long_detail():
    fn = CLIModalConfirmMixin._get_slash_confirm_display_fragments
    self_ = _make_fragment_self(
        {
            "title": "T",
            "detail": "word " * 400,
            "choices": [("once", "Once", "Run")],
            "selected": 0,
        }
    )
    fragments = _bound(fn, self_)()
    text = "".join(style_text for _style, style_text in fragments)
    assert "detail truncated" in text


def test_get_slash_confirm_display_fragments_marks_selected_row():
    fn = CLIModalConfirmMixin._get_slash_confirm_display_fragments
    self_ = _make_fragment_self(
        {
            "title": "T",
            "detail": "",
            "choices": [
                ("once", "Once", "Run it now"),
                ("always", "Always", "Remember"),
                ("cancel", "Cancel", "Abort"),
            ],
            "selected": 1,
        }
    )
    fragments = _bound(fn, self_)()
    text = "".join(style_text for _style, style_text in fragments)
    # exactly one ❯ marker, on the "always" row
    assert text.count("❯") == 1
    always_idx = text.find("[2] Always")
    assert always_idx != -1
    assert "❯ [2] Always" in text


# ---------------------------------------------------------------------------
# _snapshot_model_runtime (pure-ish capture)
# ---------------------------------------------------------------------------


def test_snapshot_model_runtime_captures_fields_and_agent_runtime():
    fn = CLIModelSwitchMixin._snapshot_model_runtime
    agent = SimpleNamespace(_primary_runtime={"kind": "primary"})
    self_ = SimpleNamespace(
        agent=agent,
        model="m1",
        provider="p1",
        requested_provider="rp1",
        _explicit_api_key="k",
        _explicit_base_url="u",
        api_key="ak",
        base_url="bu",
        api_mode="am",
    )
    snap = _bound(fn, self_)()
    assert snap["model"] == "m1"
    assert snap["provider"] == "p1"
    assert snap["api_mode"] == "am"
    assert snap["agent_primary_runtime"] == {"kind": "primary"}


def test_snapshot_model_runtime_no_agent():
    fn = CLIModelSwitchMixin._snapshot_model_runtime
    self_ = SimpleNamespace(
        agent=None,
        model="m1",
        provider="p1",
        requested_provider="rp1",
        api_key=None,
        base_url=None,
        api_mode=None,
    )
    snap = _bound(fn, self_)()
    assert snap["agent_primary_runtime"] is None
