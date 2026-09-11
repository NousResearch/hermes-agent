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

    assert output == "\n[Response transformed after streaming]\nXYZ"


def test_untransformed_stream_has_no_post_stream_output():
    assert _post_stream_transform_output("original answer", {}) == ""
