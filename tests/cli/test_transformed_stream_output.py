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


def test_in_place_edit_prints_only_tail_from_first_divergence():
    output = _post_stream_transform_output(
        "你好，世界。",
        {
            "response_transformed": True,
            "pre_transform_response": "你好,世界。",
        },
    )

    assert output == "，世界。"


def test_in_place_edit_far_into_text_prints_only_remaining_tail():
    streamed = "The agent replied quickly, and the answer was complete."
    transformed = "The agent replied quickly — and the answer was complete."
    output = _post_stream_transform_output(
        transformed,
        {"response_transformed": True, "pre_transform_response": streamed},
    )

    assert output == transformed[len("The agent replied quickly"):]


def test_divergence_at_first_character_still_prints_full_replacement():
    output = _post_stream_transform_output(
        "XYZ",
        {
            "response_transformed": True,
            "pre_transform_response": "abc",
        },
    )

    assert output == "\n[Response transformed after streaming]\nXYZ"


def test_unchanged_transform_prints_nothing_instead_of_repeating():
    output = _post_stream_transform_output(
        "original answer",
        {
            "response_transformed": True,
            "pre_transform_response": "original answer",
        },
    )

    assert output == ""


def test_truncation_falls_back_to_full_reprint_instead_of_silent_drop():
    output = _post_stream_transform_output(
        "Hello world",
        {
            "response_transformed": True,
            "pre_transform_response": "Hello world foo",
        },
    )

    assert output == "\n[Response transformed after streaming]\nHello world"


def test_shortening_in_place_edit_falls_back_to_full_reprint():
    output = _post_stream_transform_output(
        "The answer is 42.",
        {
            "response_transformed": True,
            "pre_transform_response": "The answer is 42. Trust me.",
        },
    )

    assert output == (
        "\n[Response transformed after streaming]\nThe answer is 42."
    )


def test_expanding_edit_after_shared_prefix_prints_divergent_tail():
    streamed = "See docs at example.com/page"
    transformed = "See docs at https://example.com/page for details."
    output = _post_stream_transform_output(
        transformed,
        {"response_transformed": True, "pre_transform_response": streamed},
    )

    assert output == transformed[len("See docs at "):]
