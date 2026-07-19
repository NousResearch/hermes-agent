from gateway.turn_media import collect_turn_media_text
from gateway.platforms.base import BasePlatformAdapter


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
