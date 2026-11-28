"""Preserve a voice delegation's transcript during compaction (#160155)."""

from agent.context_compressor import ContextCompressor
from agent.turn_context import compose_user_api_content
from gateway.session_surface import admit_surface, surface_note


def _voice_turn(*, legacy_gateway=False):
    transcript = (
        "User: Keep the fare under two thousand dollars.\n"
        "Voice assistant: I can check the afternoon flights.\n"
        "User: Yes, please."
    )
    committed = admit_surface({"surface": "voice-live", "voice_context": transcript})["surface_v1"]
    content = "Yes, please."
    notes = surface_note(committed)
    if legacy_gateway:
        live_prompt = f"[An earlier note]\n\n{notes}\n\n{content}"
        api_content = compose_user_api_content(live_prompt, "", "[A later plugin note]")
    else:
        api_content = compose_user_api_content(content, "", notes)
    return {"role": "user", "content": content, "api_content": api_content}


def test_summary_input_includes_the_spoken_exchange():
    compressor = ContextCompressor(model="test/model", config_context_length=100_000, quiet_mode=True)

    turns = [_voice_turn()]
    summary_input = compressor._serialize_for_summary(turns)
    lean_summary = compressor._augment_summary_lean("Planned the trip.", turns)

    assert "User: Keep the fare under two thousand dollars." in summary_input
    assert "Voice assistant: I can check the afternoon flights." in summary_input
    assert "Your reply will be spoken aloud" not in summary_input
    assert "## Recent Voice Transcript" in lean_summary
    assert "User: Keep the fare under two thousand dollars." in lean_summary


def test_fallback_preserves_the_spoken_exchange_as_context():
    compressor = ContextCompressor(model="test/model", config_context_length=100_000, quiet_mode=True)
    # The delegation can be older than the turns included in the fallback's recent-turn list.
    later_turns = [{"role": "assistant", "content": f"Checked flight {i}."} for i in range(9)]

    summary = compressor._build_static_fallback_summary([_voice_turn(legacy_gateway=True), *later_turns])

    assert summary.count("## Recent Voice Transcript") == 1
    voice_context = summary.partition("## Recent Voice Transcript\n")[2].split("\n\n## ", 1)[0]
    real_user_quotes = summary.partition("## User Messages (verbatim, newest first)\n")[2].split("\n\n## ", 1)[0]
    assert "User: Keep the fare under two thousand dollars." in voice_context
    assert "Voice assistant: I can check the afternoon flights." in voice_context
    assert "An earlier note" not in voice_context
    assert "A later plugin note" not in voice_context
    assert "> Yes, please." in real_user_quotes
    assert "Voice assistant:" not in real_user_quotes
