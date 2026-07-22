from types import MethodType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.turn_media import collect_turn_media_text
from gateway.config import Platform
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.session import SessionSource


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


def test_rebased_boundary_without_history_snapshot_falls_back_to_final_response():
    messages = [
        {"role": "assistant", "content": "MEDIA:/cache/old.ogg"},
        {"role": "assistant", "content": "MEDIA:/cache/current.ogg"},
    ]

    assert collect_turn_media_text(
        messages,
        "MEDIA:/cache/current.ogg",
        history_offset=0,
    ) == "MEDIA:/cache/current.ogg"


def _allowed_media_path(tmp_path, monkeypatch, name):
    root = tmp_path / "media-cache"
    media_file = root / name
    media_file.parent.mkdir(parents=True, exist_ok=True)
    media_file.write_bytes(b"media")
    monkeypatch.setattr(
        "gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS",
        (root,),
    )
    return media_file.resolve()


def _event():
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-1",
        chat_type="dm",
    )
    return MessageEvent(
        text="next question",
        message_type=MessageType.TEXT,
        source=source,
        message_id="msg-1",
    )


@pytest.mark.asyncio
async def test_already_sent_rebased_offset_does_not_redeliver_old_media(
    tmp_path, monkeypatch,
):
    from gateway.run import (
        GatewayRunner,
        _collect_history_media_paths,
        _deliver_already_sent_turn_media,
    )

    old_clip = _allowed_media_path(tmp_path, monkeypatch, "old.ogg")
    reveal_clip = _allowed_media_path(tmp_path, monkeypatch, "reveal.ogg")
    question_clip = _allowed_media_path(tmp_path, monkeypatch, "question.ogg")
    adapter = SimpleNamespace(
        name="test",
        extract_media=BasePlatformAdapter.extract_media,
        extract_images=BasePlatformAdapter.extract_images,
        extract_local_files=BasePlatformAdapter.extract_local_files,
        send_voice=AsyncMock(
            return_value=SendResult(success=True, message_id="voice")
        ),
        send_document=AsyncMock(
            return_value=SendResult(success=True, message_id="document")
        ),
        send_multiple_images=AsyncMock(
            return_value=SendResult(success=True, message_id="images")
        ),
        send_video=AsyncMock(
            return_value=SendResult(success=True, message_id="video")
        ),
    )
    runner = SimpleNamespace(
        _adapter_for_source=lambda source: adapter,
        _thread_metadata_for_source=lambda source, anchor=None: None,
        _reply_anchor_for_event=lambda event: None,
    )
    runner._deliver_media_from_response = MethodType(
        GatewayRunner._deliver_media_from_response,
        runner,
    )
    history = [
        {
            "role": "assistant",
            "content": f"[[audio_as_voice]] MEDIA:{old_clip}",
        },
    ]
    messages = history + [
        {"role": "user", "content": "Continue the quiz"},
        {
            "role": "assistant",
            "content": f"[[audio_as_voice]] MEDIA:{reveal_clip}",
        },
        {"role": "tool", "content": "tts ok reveal"},
        {
            "role": "assistant",
            "content": f"[[audio_as_voice]] MEDIA:{question_clip}",
        },
    ]
    agent_result = {
        "already_sent": True,
        "failed": False,
        "history_offset": 0,
        "history_media_paths": sorted(_collect_history_media_paths(history)),
    }

    delivered = await _deliver_already_sent_turn_media(
        runner,
        agent_result=agent_result,
        agent_messages=messages,
        response=f"[[audio_as_voice]] MEDIA:{question_clip}",
        event=_event(),
    )

    assert delivered is True
    delivered_paths = [
        call.kwargs["audio_path"] for call in adapter.send_voice.await_args_list
    ]
    assert delivered_paths == [str(reveal_clip), str(question_clip)]
    assert str(old_clip) not in delivered_paths
    adapter.send_document.assert_not_awaited()
