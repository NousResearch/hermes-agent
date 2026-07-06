"""Tests for the ``display.show_full_tool_call`` config flag (#58408).

Verifies the new opt-in toggle which forces tool call previews to be rendered
without ellipsis truncation. Default behavior is preserved (truncated when
``tool_preview_length`` is set, unlimited when it is 0).

Covers three surface areas:
    1. ``agent.display._effective_preview_max_len`` — direct helper.
    2. ``agent.display.build_tool_preview`` — preview builder.
    3. ``agent.display.get_cute_tool_message`` — CLI quiet-mode line.
"""

import pytest

from agent.display import (
    _effective_preview_max_len,
    build_tool_preview,
    get_cute_tool_message,
    get_show_full_tool_call,
    get_tool_preview_max_len,
    set_show_full_tool_call,
    set_tool_preview_max_len,
)


# ---------------------------------------------------------------------------
# Test fixtures — reset global state around every test so suites stay isolated.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_display_state():
    """Snapshot then restore both globals so tests don't leak into each other."""
    prev_max = get_tool_preview_max_len()
    prev_full = get_show_full_tool_call()
    set_tool_preview_max_len(0)
    set_show_full_tool_call(False)
    yield
    set_tool_preview_max_len(prev_max)
    set_show_full_tool_call(prev_full)


# Long payload that comfortably overflows any sensible preview cap.
# Use image_generate (falls through to the generic primary_arg truncation path
# in build_tool_preview) so the literal text is rendered, not heuristically
# summarized like terminal commands or basename-extracted like file paths.
_LONG_PROMPT = "draw me a picture of " + ("rainbow " * 30) + "unicorns"


# ---------------------------------------------------------------------------
# 1. Default behavior — backward compatibility.
# ---------------------------------------------------------------------------


class TestShowFullToolCallDefault:
    """``show_full_tool_call`` defaults to False so the existing behavior is
    preserved — i.e. ``tool_preview_length`` still drives the limit."""

    def test_default_flag_is_false(self):
        """A freshly-imported module must report the flag as disabled."""
        assert get_show_full_tool_call() is False

    def test_default_helper_returns_configured_limit(self):
        """With the flag off, the effective limit equals the configured limit."""
        set_tool_preview_max_len(40)
        set_show_full_tool_call(False)
        assert _effective_preview_max_len() == 40

    def test_default_zero_limit_means_unlimited(self):
        """``tool_preview_length: 0`` keeps its unlimited semantic when the
        flag is off — keeping the long-standing no-config default."""
        set_tool_preview_max_len(0)
        set_show_full_tool_call(False)
        assert _effective_preview_max_len() == 0


class TestShowFullToolCallFlagSet:
    """When the flag is True, ellipsis truncation is bypassed regardless of
    ``tool_preview_length``."""

    def test_flag_set_force_unlimited_with_positive_limit(self):
        """A positive ``tool_preview_length`` is overridden when the flag is on."""
        set_tool_preview_max_len(40)
        set_show_full_tool_call(True)
        assert _effective_preview_max_len() == 0

    def test_flag_set_force_unlimited_with_zero_limit(self):
        """Already-unlimited configs stay unlimited when the flag is on."""
        set_tool_preview_max_len(0)
        set_show_full_tool_call(True)
        assert _effective_preview_max_len() == 0

    def test_flag_off_after_flag_on(self):
        """Toggling the flag back off restores the configured limit."""
        set_tool_preview_max_len(25)
        set_show_full_tool_call(True)
        assert _effective_preview_max_len() == 0
        set_show_full_tool_call(False)
        assert _effective_preview_max_len() == 25

    def test_setter_coerces_to_bool(self):
        """The setter must normalize truthy/falsy inputs to a real bool."""
        set_show_full_tool_call("yes")  # truthy string
        assert get_show_full_tool_call() is True
        set_show_full_tool_call("")
        assert get_show_full_tool_call() is False
        set_show_full_tool_call(1)
        assert get_show_full_tool_call() is True
        set_show_full_tool_call(0)
        assert get_show_full_tool_call() is False


# ---------------------------------------------------------------------------
# 2. build_tool_preview honors the flag.
# ---------------------------------------------------------------------------


class TestBuildToolPreviewHonorsFlag:
    """``build_tool_preview`` must consult the effective preview length."""

    def test_flag_off_truncates_long_prompt(self):
        set_tool_preview_max_len(20)
        set_show_full_tool_call(False)
        preview = build_tool_preview("image_generate", {"prompt": _LONG_PROMPT})
        assert preview is not None
        assert preview.endswith("...")
        # Anything longer than 20 chars must have been shortened with an ellipsis.
        assert len(preview) <= 20

    def test_flag_on_renders_full_prompt(self):
        set_tool_preview_max_len(20)
        set_show_full_tool_call(True)
        preview = build_tool_preview("image_generate", {"prompt": _LONG_PROMPT})
        assert preview is not None
        # When un-truncated, the rendered preview must contain the full primary
        # argument verbatim — no ellipsis, no slicing.
        assert preview == _LONG_PROMPT
        assert "..." not in preview

    def test_flag_off_short_prompt_unchanged(self):
        """Short args must not be padded with ellipsis even when the flag is
        off — the existing logic only adds '...' when the input overflows."""
        set_tool_preview_max_len(80)
        set_show_full_tool_call(False)
        preview = build_tool_preview("image_generate", {"prompt": "a cat"})
        assert preview == "a cat"

    def test_flag_on_explicit_max_len_still_honored(self):
        """If the caller passes ``max_len`` explicitly, that argument still
        wins — the flag only overrides the *implicit* default."""
        set_tool_preview_max_len(0)
        set_show_full_tool_call(True)
        preview = build_tool_preview(
            "image_generate", {"prompt": _LONG_PROMPT}, max_len=15
        )
        # The explicit max_len wins because the caller asked for truncation.
        assert preview is not None
        assert preview.endswith("...")
        assert len(preview) <= 15


# ---------------------------------------------------------------------------
# 3. get_cute_tool_message honors the flag.
# ---------------------------------------------------------------------------


class TestCuteToolMessageHonorsFlag:
    """The CLI quiet-mode helper must respect the flag for its inline trunc."""

    def test_flag_off_truncates_long_query(self):
        set_tool_preview_max_len(20)
        set_show_full_tool_call(False)
        long_query = "what is the meaning of life " + "z" * 60
        line = get_cute_tool_message(
            "web_search", {"query": long_query}, duration=1.5
        )
        assert "..." in line
        assert long_query not in line

    def test_flag_on_renders_full_query(self):
        set_tool_preview_max_len(20)
        set_show_full_tool_call(True)
        long_query = "what is the meaning of life " + "z" * 60
        line = get_cute_tool_message(
            "web_search", {"query": long_query}, duration=1.5
        )
        assert long_query in line
        # Only the inner truncation should disappear; the line may still
        # contain the surrounding formatting like "┊ 🔍 search" but no "..."
        # tail-ellipsis is introduced by the preview truncator.
        assert not line.rstrip().endswith("...")

    def test_flag_on_unlimited_prompt_full_text(self):
        # image_generate falls through to the generic primary-arg truncation
        # path — the test verifies the helper's inner _trunc respects the flag
        # so the rendered line carries the verbatim prompt.
        set_tool_preview_max_len(15)
        set_show_full_tool_call(True)
        long_prompt = "describe " + ("very " * 60) + "verbose image"
        line = get_cute_tool_message(
            "image_generate", {"prompt": long_prompt}, duration=2.0
        )
        # The full prompt text must appear unmodified in the formatted line.
        assert long_prompt in line
        # No ellipsis inserted by the preview truncator.
        assert "..." not in line


# ---------------------------------------------------------------------------
# 4. Runtime toggle — no exceptions, behavior flips cleanly between calls.
# ---------------------------------------------------------------------------


class TestRuntimeToggle:
    """Toggling at runtime must not leave dangling state between calls."""

    def test_repeated_toggle(self):
        set_tool_preview_max_len(30)
        for _ in range(4):
            set_show_full_tool_call(True)
            assert _effective_preview_max_len() == 0
            set_show_full_tool_call(False)
            assert _effective_preview_max_len() == 30

    def test_independent_of_tool_preview_length(self):
        """Changing ``tool_preview_length`` after enabling the flag keeps the
        flag's unlimited semantic intact until the flag is toggled off."""
        set_tool_preview_max_len(99)
        set_show_full_tool_call(True)
        assert _effective_preview_max_len() == 0
        set_tool_preview_max_len(7)
        assert _effective_preview_max_len() == 0
        set_show_full_tool_call(False)
        assert _effective_preview_max_len() == 7
