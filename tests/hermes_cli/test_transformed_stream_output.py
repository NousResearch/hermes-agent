"""Regression coverage for CLI delivery after transform_llm_output streaming."""

from cli import _post_stream_transform_output


def test_streamed_transform_prints_only_appended_suffix():
    output = _post_stream_transform_output(
        "original answer\n\n[plugin appended this]",
        {
            "response_transformed": True,
            "pre_transform_response": "original answer",
        },
    )

    assert output == "\n\n[plugin appended this]"


def test_streamed_transform_prints_full_replacement_instead_of_dropping_it():
    output = _post_stream_transform_output(
        "XYZ",
        {
            "response_transformed": True,
            "pre_transform_response": "abc",
        },
    )

    assert output.endswith("\nXYZ")
    assert "abc" not in output


def test_untransformed_stream_has_no_post_stream_output():
    assert _post_stream_transform_output("original answer", {}) == ""


def test_inline_transform_flag_suppresses_reprint_after_midtext_substitution():
    """Issue #6: Privacy-Shield's TOKEN_* restore rewrites mid-text, so the
    rewrite is NOT a prefix extension; with the plugin's inline flag the
    already-streamed copy stands, no second full print."""
    assert _post_stream_transform_output(
        "answer Max Mustermann answer",
        {
            "response_transformed": True,
            "pre_transform_response": "answer TOKEN_XYZ answer",
            "transform_is_inline": True,
        },
    ) == ""


def test_inline_transform_flag_still_prints_prefix_suffix():
    # Defensive: a prefix relationship always wins over the flag so an
    # appended tail is never lost even when a plugin sets the flag wrongly.
    output = _post_stream_transform_output(
        "original answer\n\n[plugin appended this]",
        {
            "response_transformed": True,
            "pre_transform_response": "original answer",
            "transform_is_inline": True,
        },
    )
    assert output == "\n\n[plugin appended this]"
