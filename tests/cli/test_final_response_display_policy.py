from __future__ import annotations

from hermes_cli.config import DEFAULT_CONFIG
from hermes_cli.final_response_display_policy import (
    assistant_body_for_tts,
    decide_final_response_display,
)


def test_default_config_keeps_assistant_answer_body_final_only() -> None:
    assert DEFAULT_CONFIG["display"]["assistant_body_streaming"] is False


def test_transformed_final_renders_even_if_raw_stream_was_received() -> None:
    decision = decide_final_response_display(
        response="路径：native｜原因：runtime_default\nBody",
        response_previewed=False,
        response_transformed=True,
        stream_text_received=True,
        stream_text_visible=False,
        is_error_response=False,
        use_streaming_tts=False,
        streaming_tts_box_opened=False,
    )

    assert decision.render_panel is True
    assert decision.close_streaming_tts_box is False
    assert decision.reason == "transformed_final_response"


def test_visible_untransformed_stream_can_skip_legacy_final_panel() -> None:
    decision = decide_final_response_display(
        response="Body",
        response_previewed=False,
        response_transformed=False,
        stream_text_received=True,
        stream_text_visible=True,
        is_error_response=False,
        use_streaming_tts=False,
        streaming_tts_box_opened=False,
    )

    assert decision.render_panel is False
    assert decision.reason == "already_visible_stream"


def test_received_but_not_visible_stream_renders_canonical_final() -> None:
    decision = decide_final_response_display(
        response="Body",
        response_previewed=False,
        response_transformed=False,
        stream_text_received=True,
        stream_text_visible=False,
        is_error_response=False,
        use_streaming_tts=False,
        streaming_tts_box_opened=False,
    )

    assert decision.render_panel is True
    assert decision.reason == "not_visible"


def test_preview_flag_does_not_hide_final_when_draft_body_was_not_visible() -> None:
    decision = decide_final_response_display(
        response="路径：native｜原因：runtime_default\nBody",
        response_previewed=True,
        response_transformed=True,
        stream_text_received=True,
        stream_text_visible=False,
        is_error_response=False,
        use_streaming_tts=False,
        streaming_tts_box_opened=False,
    )

    assert decision.render_panel is True
    assert decision.reason == "transformed_final_response"


def test_transformed_final_overrides_visible_preview_suppression() -> None:
    decision = decide_final_response_display(
        response="路径：native｜原因：runtime_default\nBody",
        response_previewed=True,
        response_transformed=True,
        stream_text_received=True,
        stream_text_visible=True,
        is_error_response=False,
        use_streaming_tts=False,
        streaming_tts_box_opened=False,
    )

    assert decision.render_panel is True
    assert decision.reason == "transformed_final_response"


def test_transformed_final_renders_and_closes_streaming_tts_box() -> None:
    decision = decide_final_response_display(
        response="路径：native｜原因：runtime_default\nBody",
        response_previewed=False,
        response_transformed=True,
        stream_text_received=True,
        stream_text_visible=False,
        is_error_response=False,
        use_streaming_tts=True,
        streaming_tts_box_opened=True,
    )

    assert decision.render_panel is True
    assert decision.close_streaming_tts_box is True
    assert decision.reason == "transformed_final_response"


def test_tts_receives_body_without_runtime_route_bar() -> None:
    response = "路径：native｜原因：runtime_default｜工具 none\nBody to speak"

    assert assistant_body_for_tts(response) == "Body to speak"
