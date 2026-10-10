"""Shared projection has metadata authority and never mutates model-facing content."""

from copy import deepcopy
from dataclasses import FrozenInstanceError

import pytest

from agent.context_compressor import COMPRESSED_SUMMARY_METADATA_KEY, SUMMARY_PREFIX, _SUMMARY_END_MARKER
from agent.message_display import conversation_preview_text, project_message_for_display, render_message_content
from agent.prompt_builder import steer_user_row


@pytest.mark.parametrize("text", ["[System: explain this marker]", "[OUT-OF-BAND USER MESSAGE", "ordinary words"])
def test_authored_and_untyped_marker_text_is_preserved(text):
    for message in ({"role": "user", "content": text}, steer_user_row(text)):
        before = deepcopy(message)
        display = project_message_for_display(message)
        assert display.visible and conversation_preview_text(display) == text
        assert message == before
        with pytest.raises(FrozenInstanceError):
            display.visible = False


@pytest.mark.parametrize("metadata", [None, "hidden", [], {"model_only": "false"}])
def test_invalid_metadata_cannot_hide_ordinary_text(metadata):
    display = project_message_for_display({"role": "user", "content": "hello", "display_metadata": metadata})
    assert display.visible and conversation_preview_text(display) == "hello"


def test_visibility_and_preview_policy_are_separate():
    for kind in ("process_complete", "async_delegation_complete"):
        raw = {"role": "user", "content": "model-facing envelope and results", "display_kind": kind,
               "display_metadata": {"display_text": "The background task completed."}}
        display = project_message_for_display(raw)
        assert display.visible and display.content == raw["content"]
        assert display.message["display_metadata"]["display_text"] == "The background task completed."
        assert display.message["content"] == raw["content"]
        assert conversation_preview_text(display) == ""
    unknown = project_message_for_display({"role": "user", "content": "authored text", "display_kind": "new_kind"})
    assert conversation_preview_text(unknown) == "authored text"
    for metadata in ({"model_only": True}, {}):
        display = project_message_for_display({"role": "user", "content": "internal", "display_kind": "hidden",
                                               "display_metadata": metadata})
        assert not display.visible


def test_typed_compaction_keeps_live_ask_and_strips_model_only_sidecars():
    raw = {"role": "user", "content": f"{SUMMARY_PREFIX} old work\n{_SUMMARY_END_MARKER}\nPlease continue",
           COMPRESSED_SUMMARY_METADATA_KEY: True, "display_kind": "hidden", "_row_id": 9,
           "reasoning": "model-only reasoning", "message_uid": "same-request"}
    before = deepcopy(raw)
    display = project_message_for_display(raw)
    assert display.visible and conversation_preview_text(display) == "Please continue"
    assert display.message["_row_id"] == 9 and display.message["message_uid"] == "same-request"
    assert "reasoning" not in display.message
    assert raw == before
    raw["display_metadata"] = {"model_only": True}
    assert not project_message_for_display(raw).visible


@pytest.mark.parametrize("content,expected", [
    ([{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}], "[image]"),
    ([{"type": "input_audio", "input_audio": {"data": "AA=="}}], "[audio]"),
    ({"type": "unknown_block", "payload": "secret"}, "[unknown_block]"),
    ({"arbitrary": "shape"}, "[structured content]"),
])
def test_structured_rendering_is_bounded_and_never_dumps_payload(content, expected):
    assert render_message_content(content, image_urls=False).strip() == expected


def test_an_untyped_complete_compaction_envelope_is_not_visibility_authority():
    content = f"{SUMMARY_PREFIX}\nold work\n{_SUMMARY_END_MARKER}"
    display = project_message_for_display({"role": "user", "content": content})
    assert display.visible and display.content == content


def test_projection_runs_without_filesystem_reads(monkeypatch):
    from pathlib import Path

    def forbidden(*args, **kwargs):
        raise AssertionError("display projection must not inspect the filesystem")

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "read_text", forbidden)
        scoped.setattr(Path, "read_bytes", forbidden)
        scoped.setattr(Path, "stat", forbidden)
        assert project_message_for_display(steer_user_row("hello")).content == "hello"
        assert project_message_for_display({"role": "user", "content": "ordinary text"}).content == "ordinary text"
