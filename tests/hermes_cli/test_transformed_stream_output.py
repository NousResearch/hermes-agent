"""Regression coverage for CLI delivery after transform_llm_output streaming."""

from cli import _post_stream_transform_output
from hermes_cli.cli_chat_turn_mixin import _unstreamed_final_suffix


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


def test_unstreamed_final_suffix_repairs_exact_prefix_gap():
    assert _unstreamed_final_suffix(
        "1 2 3 4 5", "1 2 3", {"completed": True},
    ) == " 4 5"


def test_unstreamed_final_suffix_does_not_guess_after_divergence_or_transform():
    assert _unstreamed_final_suffix(
        "canonical answer", "different live text", {"completed": True},
    ) == ""
    assert _unstreamed_final_suffix(
        "canonical answer + plugin", "canonical answer",
        {"completed": True, "response_transformed": True},
    ) == ""
