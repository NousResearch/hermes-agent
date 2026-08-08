"""Telegram/Discord model picker labels must keep upstream route prefixes."""

from __future__ import annotations

from hermes_cli.model_switch import format_model_picker_button_label


def test_keeps_first_and_last_path_segments():
    assert format_model_picker_button_label("opencode-go/glm-5.2") == "opencode-go/glm-5.2"
    assert format_model_picker_button_label("cursor/glm-5.2") == "cursor/glm-5.2"
    assert (
        format_model_picker_button_label("vendor/group/nested/glm-5.2")
        == "vendor/glm-5.2"
    )


def test_plain_model_ids_unchanged():
    assert format_model_picker_button_label("gpt-5.6-luna") == "gpt-5.6-luna"


def test_truncates_long_labels():
    long_id = "very-long-upstream-name/" + ("x" * 40)
    label = format_model_picker_button_label(long_id, max_len=20)
    assert len(label) == 20
    assert label.endswith("...")
    assert label.startswith("very-long")
