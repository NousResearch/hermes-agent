from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.turn_media import collect_turn_media_text, select_turn_messages


def _voiced(text):
    media, _ = BasePlatformAdapter.extract_media(text)
    return [p for p, is_voice in media if is_voice]


def test_collects_every_voice_clip_emitted_across_a_turn():
    # A quiz turn reveals the previous answer (clip 1) then asks the next
    # question (clip 2). The two MEDIA tags live in separate assistant
    # segments split by a tool call, so scanning only the final segment
    # drops the reveal clip. collect_turn_media_text must surface BOTH.
    turn_messages = [
        {"role": "assistant", "content": "David goes with bronze censers. The answer is the mirrors of the women."},
        {"role": "tool", "content": "tts ok reveal"},
        {"role": "assistant", "content": "[[audio_as_voice]] MEDIA:/cache/reveal.ogg Yentl, your next question."},
        {"role": "tool", "content": "tts ok question"},
        {"role": "assistant", "content": "[[audio_as_voice]] MEDIA:/cache/question.ogg"},
    ]
    final_response = "[[audio_as_voice]] MEDIA:/cache/question.ogg"
    voiced = _voiced(collect_turn_media_text(turn_messages, final_response))
    assert "/cache/reveal.ogg" in voiced
    assert "/cache/question.ogg" in voiced


def test_does_not_duplicate_the_final_clip():
    turn_messages = [
        {"role": "assistant", "content": "[[audio_as_voice]] MEDIA:/cache/a.ogg next"},
        {"role": "assistant", "content": "[[audio_as_voice]] MEDIA:/cache/b.ogg"},
    ]
    voiced = _voiced(collect_turn_media_text(turn_messages, "[[audio_as_voice]] MEDIA:/cache/b.ogg"))
    assert voiced.count("/cache/a.ogg") == 1
    assert voiced.count("/cache/b.ogg") == 1


def test_falls_back_to_final_response_when_no_assistant_segments():
    assert collect_turn_media_text([], "final text") == "final text"
    assert collect_turn_media_text(None, "final text") == "final text"


def test_selects_the_turn_local_slice_on_a_normal_continuing_turn():
    messages = [
        {"role": "assistant", "content": "older turn"},
        {"role": "user", "content": "next question please"},
        {"role": "assistant", "content": "current turn"},
    ]
    assert select_turn_messages(messages, 2, ["some", "history"]) == [
        {"role": "assistant", "content": "current turn"}
    ]


def test_selects_everything_on_a_genuine_first_turn():
    messages = [{"role": "assistant", "content": "first ever reply"}]
    assert select_turn_messages(messages, 0, []) == messages
    assert select_turn_messages(messages, 0, None) == messages


def test_refuses_the_slice_when_compaction_rebaselined_the_offset():
    # Split / in-place compaction reports history_offset=0 while the message
    # list is the compacted transcript, so offset 0 next to a non-empty
    # incoming history must NOT be treated as a start-of-turn boundary.
    messages = [
        {"role": "assistant", "content": "[[audio_as_voice]] MEDIA:/cache/old.ogg"},
        {"role": "assistant", "content": "[[audio_as_voice]] MEDIA:/cache/new.ogg"},
    ]
    assert select_turn_messages(messages, 0, [{"role": "user", "content": "earlier"}]) == []


def test_refuses_malformed_offsets():
    messages = [{"role": "assistant", "content": "a"}]
    assert select_turn_messages(messages, -1, []) == []
    assert select_turn_messages(messages, 99, []) == []
    assert select_turn_messages(messages, "2", []) == []
    assert select_turn_messages(messages, None, []) == []
    assert select_turn_messages("not a list", 0, []) == []


def test_compacted_turn_does_not_replay_an_old_clip():
    # End to end over both helpers: with an ambiguous boundary the caller gets
    # the final response only, so the retained older clip is never delivered.
    messages = [
        {"role": "assistant", "content": "[[audio_as_voice]] MEDIA:/cache/old.ogg"},
        {"role": "assistant", "content": "[[audio_as_voice]] MEDIA:/cache/new.ogg"},
    ]
    final_response = "[[audio_as_voice]] MEDIA:/cache/new.ogg"
    turn_messages = select_turn_messages(messages, 0, [{"role": "user", "content": "earlier"}])
    voiced = _voiced(collect_turn_media_text(turn_messages, final_response))
    assert "/cache/old.ogg" not in voiced
    assert voiced == ["/cache/new.ogg"]


# The integration cases below run the same composition the gateway call site
# uses (select_turn_messages -> collect_turn_media_text ->
# _deliver_media_from_response) through the REAL delivery method, so they pin
# the wiring rather than the helpers alone.


def _event():
    source = SessionSource(
        platform=Platform.SLACK,
        chat_id="C123CHAN",
        chat_type="group",
        thread_id=None,
    )
    return MessageEvent(
        text="hi",
        message_type=MessageType.TEXT,
        source=source,
        message_id="171.000001",
    )


def _fake_runner():
    return SimpleNamespace(
        _thread_metadata_for_source=lambda source, anchor=None: {},
        _reply_anchor_for_event=lambda event: None,
    )


def _adapter():
    return SimpleNamespace(
        name="test",
        extract_media=BasePlatformAdapter.extract_media,
        extract_images=BasePlatformAdapter.extract_images,
        extract_local_files=BasePlatformAdapter.extract_local_files,
        send_voice=AsyncMock(return_value=SendResult(success=True, message_id="voice")),
        send_document=AsyncMock(return_value=SendResult(success=True, message_id="doc")),
        send_image_file=AsyncMock(return_value=SendResult(success=True, message_id="image")),
        send_video=AsyncMock(return_value=SendResult(success=True, message_id="video")),
        send_multiple_images=AsyncMock(return_value=SendResult(success=True, message_id="imgs")),
    )


def _clips(tmp_path, monkeypatch, *names):
    root = tmp_path / "media-cache"
    root.mkdir(parents=True, exist_ok=True)
    made = []
    for name in names:
        clip = root / name
        clip.write_bytes(b"ogg")
        made.append(clip.resolve())
    monkeypatch.setattr(
        "gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS",
        (root,),
    )
    return made


def _delivered_voice_paths(adapter):
    return [call.kwargs["audio_path"] for call in adapter.send_voice.await_args_list]


@pytest.mark.asyncio
async def test_every_turn_clip_reaches_the_real_delivery_method(tmp_path, monkeypatch):
    """The bug this PR fixes: a reveal clip and a next-question clip emitted in
    separate assistant segments must BOTH be uploaded, not just the last."""
    reveal, question = _clips(tmp_path, monkeypatch, "reveal.ogg", "question.ogg")
    adapter = _adapter()
    agent_messages = [
        {"role": "user", "content": "next question"},
        {"role": "assistant", "content": f"[[audio_as_voice]] MEDIA:{reveal}"},
        {"role": "tool", "content": "tts ok"},
        {"role": "assistant", "content": f"[[audio_as_voice]] MEDIA:{question}"},
    ]
    final_response = f"[[audio_as_voice]] MEDIA:{question}"

    turn_messages = select_turn_messages(agent_messages, 0, [])
    await GatewayRunner._deliver_media_from_response(
        _fake_runner(),
        collect_turn_media_text(turn_messages, final_response),
        _event(),
        adapter,
    )

    delivered = _delivered_voice_paths(adapter)
    assert str(reveal) in delivered
    assert str(question) in delivered


@pytest.mark.asyncio
async def test_ambiguous_boundary_delivers_only_the_final_clip(tmp_path, monkeypatch):
    """With a compaction-rebaselined offset the retained older clip must not be
    re-uploaded, so delivery falls back to the final response alone."""
    old, new = _clips(tmp_path, monkeypatch, "old.ogg", "new.ogg")
    adapter = _adapter()
    agent_messages = [
        {"role": "assistant", "content": f"[[audio_as_voice]] MEDIA:{old}"},
        {"role": "assistant", "content": f"[[audio_as_voice]] MEDIA:{new}"},
    ]
    final_response = f"[[audio_as_voice]] MEDIA:{new}"

    turn_messages = select_turn_messages(
        agent_messages, 0, [{"role": "user", "content": "earlier"}]
    )
    await GatewayRunner._deliver_media_from_response(
        _fake_runner(),
        collect_turn_media_text(turn_messages, final_response),
        _event(),
        adapter,
    )

    delivered = _delivered_voice_paths(adapter)
    assert delivered == [str(new)]
