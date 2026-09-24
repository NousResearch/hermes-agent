"""Google Meet captions behavior contracts."""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import json
import shutil
import subprocess
from pathlib import Path
import pytest


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    yield hermes_home


def test_caption_observer_real_chromium_round_trip():
    from plugins.google_meet.meet_bot import _CAPTION_OBSERVER_JS

    playwright = pytest.importorskip("playwright.sync_api")

    with playwright.sync_playwright() as pw:
        executable = Path(pw.chromium.executable_path)
        if not executable.is_file():
            pytest.skip("Playwright Chromium is not installed")
        browser = pw.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.set_content(
                """
                <div role="region" aria-label="Captions">
                  <div jsname="dsyhDe">
                    <span class="NWpY1d">Alex Rivera</span>
                    <span jsname="tgaKEf">Initial caption</span>
                  </div>
                </div>
                """
            )
            page.evaluate(_CAPTION_OBSERVER_JS)
            page.locator('[jsname="tgaKEf"]').evaluate(
                "element => { element.textContent = 'Updated caption text'; }"
            )
            page.wait_for_timeout(50)
            entries = page.evaluate("window.__hermesMeetDrain()")
        finally:
            browser.close()

    assert any(
        entry.get("speaker") == "Alex Rivera"
        and entry.get("text") == "Updated caption text"
        for entry in entries
    )


def test_bot_state_dedupes_captions_and_flushes_status(tmp_path):
    from plugins.google_meet.meet_bot import _BotState

    out = tmp_path / "session"
    state = _BotState(out_dir=out, meeting_id="abc-defg-hij",
                      url="https://meet.google.com/abc-defg-hij")

    state.record_caption("Alice", "Hey everyone")
    state.record_caption("Alice", "Hey everyone")  # dup — ignored
    state.record_caption("Bob", "Let's start")

    transcript = (out / "transcript.txt").read_text()
    assert "Alice: Hey everyone" in transcript
    assert "Bob: Let's start" in transcript
    # dedup — Alice line appears exactly once
    assert transcript.count("Alice: Hey everyone") == 1

    status = json.loads((out / "status.json").read_text())
    assert status["meetingId"] == "abc-defg-hij"
    assert status["transcriptLines"] == 2
    assert status["transcriptPath"].endswith("transcript.txt")


def test_bot_state_replaces_transcript_atomically(tmp_path, monkeypatch):
    from plugins.google_meet import meet_bot
    from plugins.google_meet.meet_bot import _BotState

    out = tmp_path / "session"
    state = _BotState(
        out_dir=out,
        meeting_id="abc-defg-hij",
        url="https://meet.google.com/abc-defg-hij",
    )
    state.record_caption("Alex Rivera", "First caption", caption_id="row-a")

    transcript_path = out / "transcript.txt"
    original_transcript = transcript_path.read_text()
    observed_before_replace = []
    real_replace = meet_bot.os.replace

    def observe_replace(src, dst):
        if Path(dst) == transcript_path:
            observed_before_replace.append(transcript_path.read_text())
        real_replace(src, dst)

    monkeypatch.setattr(meet_bot.os, "replace", observe_replace)
    state.record_caption("Jordan Lee", "Second caption", caption_id="row-b")

    assert observed_before_replace == [original_transcript]
    assert "First caption" in transcript_path.read_text()
    assert "Second caption" in transcript_path.read_text()


def test_bot_state_status_writes_are_thread_safe(tmp_path):
    from plugins.google_meet.meet_bot import _BotState

    out = tmp_path / "session"
    state = _BotState(
        out_dir=out,
        meeting_id="abc-defg-hij",
        url="https://meet.google.com/abc-defg-hij",
    )

    def write_status(worker: int) -> None:
        for offset in range(100):
            state.set(audio_bytes_out=(worker * 100) + offset)

    with ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(write_status, range(12)))

    status = json.loads((out / "status.json").read_text())
    assert status["meetingId"] == "abc-defg-hij"
    assert isinstance(status["audioBytesOut"], int)


def test_bot_state_rewrites_growing_same_speaker_caption_row(tmp_path):
    from plugins.google_meet.meet_bot import _BotState

    out = tmp_path / "session"
    state = _BotState(out_dir=out, meeting_id="abc-defg-hij",
                      url="https://meet.google.com/abc-defg-hij")

    state.record_caption("Alex Rivera", "Trips.")
    state.record_caption("Alex Rivera", "Trips. Yellow.")
    state.record_caption("Alex Rivera", "Trips. Yellow. There we go.")

    transcript = (out / "transcript.txt").read_text().splitlines()
    assert [line.split("] ", 1)[1] for line in transcript] == [
        "Alex Rivera: Trips. Yellow. There we go.",
    ]

    status = json.loads((out / "status.json").read_text())
    assert status["transcriptLines"] == 1


def test_bot_state_rewrites_interleaved_growing_caption_row(tmp_path):
    from plugins.google_meet.meet_bot import _BotState

    out = tmp_path / "session"
    state = _BotState(out_dir=out, meeting_id="abc-defg-hij",
                      url="https://meet.google.com/abc-defg-hij")

    state.record_caption("Alex Rivera", "We should start with requirements.")
    state.record_caption("Jordan Lee", "The background audio is duplicated.")
    state.record_caption(
        "Alex Rivera",
        "We should start with requirements and then verify the transcript.",
    )

    transcript = (out / "transcript.txt").read_text().splitlines()
    assert [line.split("] ", 1)[1] for line in transcript] == [
        "Alex Rivera: We should start with requirements and then verify the transcript.",
        "Jordan Lee: The background audio is duplicated.",
    ]

    status = json.loads((out / "status.json").read_text())
    assert status["transcriptLines"] == 2


def test_bot_state_rewrites_similar_same_speaker_caption_edit(tmp_path):
    from plugins.google_meet.meet_bot import _BotState

    out = tmp_path / "session"
    state = _BotState(out_dir=out, meeting_id="abc-defg-hij",
                      url="https://meet.google.com/abc-defg-hij")

    state.record_caption("Alex Rivera", "Trips. Yellow. Yellow. oh, let me meet")
    state.record_caption("Alex Rivera", "Trips. Yellow. Yellow. Oh, let me myself.")

    transcript = (out / "transcript.txt").read_text().splitlines()
    assert [line.split("] ", 1)[1] for line in transcript] == [
        "Alex Rivera: Trips. Yellow. Yellow. Oh, let me myself.",
    ]

    status = json.loads((out / "status.json").read_text())
    assert status["transcriptLines"] == 1


def test_bot_state_splits_growing_same_speaker_caption_before_it_gets_too_long(tmp_path):
    from plugins.google_meet.meet_bot import _BotState

    out = tmp_path / "session"
    state = _BotState(out_dir=out, meeting_id="abc-defg-hij",
                      url="https://meet.google.com/abc-defg-hij")

    first_segment = " ".join(["alpha"] * 70)
    second_segment = " ".join(["beta"] * 45)
    third_segment = " ".join(["gamma"] * 10)

    state.record_caption("Alex Rivera", first_segment)
    state.record_caption("Alex Rivera", f"{first_segment} {second_segment}")
    state.record_caption("Alex Rivera", f"{first_segment} {second_segment} {third_segment}")

    transcript = (out / "transcript.txt").read_text().splitlines()
    stripped = [line.split("] ", 1)[1] for line in transcript]
    final_caption = f"{first_segment} {second_segment} {third_segment}"
    expected_chunks = state._split_caption_text(final_caption)
    assert stripped == [f"Alex Rivera: {chunk}" for chunk in expected_chunks]
    assert all(len(line.split(": ", 1)[1]) <= 500 for line in stripped)

    status = json.loads((out / "status.json").read_text())
    assert status["transcriptLines"] == 2


def test_bot_state_revises_split_caption_row_when_dom_row_id_changes(tmp_path):
    from plugins.google_meet.meet_bot import MAX_TRANSCRIPT_TEXT_LEN, _BotState

    out = tmp_path / "session"
    state = _BotState(out_dir=out, meeting_id="abc-defg-hij",
                      url="https://meet.google.com/abc-defg-hij")

    prefix = " ".join(["alpha"] * 95)
    first_tail = " ".join(["beta"] * 35)
    final_tail = " ".join(["gamma"] * 20)
    first = f"{prefix} {first_tail}"
    final = f"{first} {final_tail}"

    state.record_caption("Alex Rivera", first, caption_id="row-live-1")
    state.record_caption("Alex Rivera", final, caption_id="row-live-2")

    transcript = (out / "transcript.txt").read_text().splitlines()
    stripped = [line.split("] ", 1)[1] for line in transcript]
    expected_chunks = state._split_caption_text(final)
    assert stripped == [f"Alex Rivera: {chunk}" for chunk in expected_chunks]
    assert len(stripped) == len(expected_chunks)
    assert all(len(line.split(": ", 1)[1]) <= MAX_TRANSCRIPT_TEXT_LEN for line in stripped)


def test_bot_state_revises_short_caption_growth_when_dom_row_id_changes(tmp_path):
    from plugins.google_meet.meet_bot import _BotState

    out = tmp_path / "session"
    state = _BotState(out_dir=out, meeting_id="abc-defg-hij",
                      url="https://meet.google.com/abc-defg-hij")

    state.record_caption("Alex Rivera", "will", caption_id="row-a")
    state.record_caption("Alex Rivera", "will still be", caption_id="row-b")

    transcript = (out / "transcript.txt").read_text().splitlines()
    assert [line.split("] ", 1)[1] for line in transcript] == [
        "Alex Rivera: will still be",
    ]


def test_bot_state_revises_caption_growth_with_punctuation_drift_and_new_row_id(tmp_path):
    from plugins.google_meet.meet_bot import _BotState

    out = tmp_path / "session"
    state = _BotState(out_dir=out, meeting_id="abc-defg-hij",
                      url="https://meet.google.com/abc-defg-hij")

    state.record_caption(
        "Alex Rivera",
        "do not fearful withdrawal perhaps it's",
        caption_id="row-a",
    )
    state.record_caption(
        "Alex Rivera",
        "do not fearful withdrawal, perhaps it's quite possible.",
        caption_id="row-b",
    )

    transcript = (out / "transcript.txt").read_text().splitlines()
    assert [line.split("] ", 1)[1] for line in transcript] == [
        "Alex Rivera: do not fearful withdrawal, perhaps it's quite possible.",
    ]


def test_bot_state_revises_partial_word_growth_with_new_row_id(tmp_path):
    from plugins.google_meet.meet_bot import _BotState

    out = tmp_path / "session"
    state = _BotState(out_dir=out, meeting_id="abc-defg-hij",
                      url="https://meet.google.com/abc-defg-hij")

    state.record_caption("Alex Rivera", "Oh, we have to rec?", caption_id="row-a")
    state.record_caption("Alex Rivera", "Oh we have to recalculate?", caption_id="row-b")
    state.record_caption(
        "Alex Rivera",
        "Oh, we have to recalculate the whole plan.",
        caption_id="row-c",
    )

    transcript = (out / "transcript.txt").read_text().splitlines()
    assert [line.split("] ", 1)[1] for line in transcript] == [
        "Alex Rivera: Oh, we have to recalculate the whole plan.",
    ]


def test_bot_state_revises_long_split_caption_group_with_word_drift(tmp_path):
    from plugins.google_meet.meet_bot import MAX_TRANSCRIPT_TEXT_LEN, _BotState

    out = tmp_path / "session"
    state = _BotState(out_dir=out, meeting_id="abc-defg-hij",
                      url="https://meet.google.com/abc-defg-hij")

    prefix = " ".join(["alpha"] * 60)
    original_tail = " ".join(["emergency"] * 35)
    revised_tail = " ".join(["merge"] * 35)
    final_tail = " ".join(["access"] * 15)
    original = f"{prefix} {original_tail}"
    revised = f"{prefix} {revised_tail} {final_tail}"

    state.record_caption("Alex Rivera", original, caption_id="row-a")
    state.record_caption("Alex Rivera", revised, caption_id="row-b")

    transcript = (out / "transcript.txt").read_text().splitlines()
    stripped = [line.split("] ", 1)[1] for line in transcript]
    expected_chunks = state._split_caption_text(revised)
    assert stripped == [f"Alex Rivera: {chunk}" for chunk in expected_chunks]
    assert all(len(line.split(": ", 1)[1]) <= MAX_TRANSCRIPT_TEXT_LEN for line in stripped)


def test_bot_state_revises_caption_growth_with_middle_word_drift(tmp_path):
    from plugins.google_meet.meet_bot import _BotState

    out = tmp_path / "session"
    state = _BotState(out_dir=out, meeting_id="abc-defg-hij",
                      url="https://meet.google.com/abc-defg-hij")

    state.record_caption(
        "Alex Rivera",
        "alpha beta gamma delta old phrase shared tail one two.",
        caption_id="row-a",
    )
    state.record_caption(
        "Alex Rivera",
        "alpha beta gamma delta new words shared tail one two three four.",
        caption_id="row-b",
    )

    transcript = (out / "transcript.txt").read_text().splitlines()
    assert [line.split("] ", 1)[1] for line in transcript] == [
        "Alex Rivera: alpha beta gamma delta new words shared tail one two three four.",
    ]


def test_bot_state_revises_same_length_tail_word_correction(tmp_path):
    from plugins.google_meet.meet_bot import _BotState

    out = tmp_path / "session"
    state = _BotState(out_dir=out, meeting_id="abc-defg-hij",
                      url="https://meet.google.com/abc-defg-hij")

    state.record_caption("Alex Rivera", "alpha beta gamma delta old", caption_id="row-a")
    state.record_caption("Alex Rivera", "alpha beta gamma delta new", caption_id="row-b")
    state.record_caption(
        "Alex Rivera",
        "alpha beta gamma delta new words shared tail.",
        caption_id="row-c",
    )

    transcript = (out / "transcript.txt").read_text().splitlines()
    assert [line.split("] ", 1)[1] for line in transcript] == [
        "Alex Rivera: alpha beta gamma delta new words shared tail.",
    ]


def test_bot_state_keeps_separate_same_speaker_restarts(tmp_path):
    from plugins.google_meet.meet_bot import _BotState

    out = tmp_path / "session"
    state = _BotState(out_dir=out, meeting_id="abc-defg-hij",
                      url="https://meet.google.com/abc-defg-hij")

    state.record_caption(
        "Alex Rivera",
        "Okay we should start with the project timeline.",
        caption_id="row-a",
    )
    state.record_caption(
        "Alex Rivera",
        "Okay we should start with the budget instead.",
        caption_id="row-b",
    )

    transcript = (out / "transcript.txt").read_text().splitlines()
    assert [line.split("] ", 1)[1] for line in transcript] == [
        "Alex Rivera: Okay we should start with the project timeline.",
        "Alex Rivera: Okay we should start with the budget instead.",
    ]


def test_bot_state_keeps_separate_same_speaker_restarts_after_revision_window(
    tmp_path,
    monkeypatch,
):
    from plugins.google_meet import meet_bot
    from plugins.google_meet.meet_bot import _BotState

    now = [100.0]
    monkeypatch.setattr(meet_bot.time, "monotonic", lambda: now[0])
    out = tmp_path / "session"
    state = _BotState(
        out_dir=out,
        meeting_id="abc-defg-hij",
        url="https://meet.google.com/abc-defg-hij",
    )

    state.record_caption(
        "Alex Rivera",
        "Okay we should start with the project timeline today.",
        caption_id="row-a",
    )
    now[0] += 3.0
    state.record_caption(
        "Alex Rivera",
        "Okay we should start with the project timeline tomorrow.",
        caption_id="row-b",
    )

    transcript = (out / "transcript.txt").read_text().splitlines()
    assert [line.split("] ", 1)[1] for line in transcript] == [
        "Alex Rivera: Okay we should start with the project timeline today.",
        "Alex Rivera: Okay we should start with the project timeline tomorrow.",
    ]


def test_bot_state_revises_matching_caption_row_without_collapsing_same_speaker_rows(tmp_path):
    from plugins.google_meet.meet_bot import _BotState

    out = tmp_path / "session"
    state = _BotState(out_dir=out, meeting_id="abc-defg-hij",
                      url="https://meet.google.com/abc-defg-hij")

    state.record_caption("Alex Rivera", "Shared prefix first thought.", caption_id="row-a")
    state.record_caption("Alex Rivera", "Shared prefix second thought.", caption_id="row-b")
    state.record_caption(
        "Alex Rivera",
        "Shared prefix first thought with more detail.",
        caption_id="row-a",
    )

    transcript = (out / "transcript.txt").read_text().splitlines()
    assert [line.split("] ", 1)[1] for line in transcript] == [
        "Alex Rivera: Shared prefix first thought with more detail.",
        "Alex Rivera: Shared prefix second thought.",
    ]


def test_bot_state_splits_single_long_caption_segment(tmp_path):
    from plugins.google_meet.meet_bot import _BotState

    out = tmp_path / "session"
    state = _BotState(out_dir=out, meeting_id="abc-defg-hij",
                      url="https://meet.google.com/abc-defg-hij")

    text = " ".join(["alpha"] * 130)
    state.record_caption("Alex Rivera", text)

    transcript = (out / "transcript.txt").read_text().splitlines()
    stripped = [line.split("] ", 1)[1] for line in transcript]
    assert len(stripped) > 1
    assert " ".join(line.split(": ", 1)[1] for line in stripped) == text
    assert all(len(line.split(": ", 1)[1]) <= 500 for line in stripped)

    status = json.loads((out / "status.json").read_text())
    assert status["transcriptLines"] == len(stripped)


def test_bot_state_dedupes_overlapping_split_caption_segments(tmp_path):
    from plugins.google_meet.meet_bot import _BotState

    out = tmp_path / "session"
    state = _BotState(out_dir=out, meeting_id="abc-defg-hij",
                      url="https://meet.google.com/abc-defg-hij")

    prefix = " ".join(["alpha"] * 95)
    state.record_caption("Alex Rivera", f"{prefix} first ending.")
    state.record_caption("Alex Rivera", f"{prefix} second ending.")

    transcript = (out / "transcript.txt").read_text().splitlines()
    texts = [line.split(": ", 1)[1] for line in transcript]
    assert len(texts) == len(set(texts))
    assert all(len(text) <= 500 for text in texts)


def test_bot_state_flushes_local_media_state(tmp_path):
    from plugins.google_meet.meet_bot import _BotState

    out = tmp_path / "session"
    state = _BotState(out_dir=out, meeting_id="abc-defg-hij",
                      url="https://meet.google.com/abc-defg-hij")
    state.set(local_microphone_on=False, local_camera_on=False)

    status = json.loads((out / "status.json").read_text())
    assert status["localMicrophoneOn"] is False
    assert status["localCameraOn"] is False


def test_bot_state_ignores_blank_text(tmp_path):
    from plugins.google_meet.meet_bot import _BotState

    state = _BotState(out_dir=tmp_path / "s", meeting_id="x-y-z",
                      url="https://meet.google.com/x-y-z")
    state.record_caption("Alice", "")
    state.record_caption("Alice", "   ")

    status = json.loads((tmp_path / "s" / "status.json").read_text())
    assert status["transcriptLines"] == 0
    transcript_path = tmp_path / "s" / "transcript.txt"
    assert not transcript_path.exists() or "Unknown:" not in transcript_path.read_text()


@pytest.mark.parametrize("caption_ids", [(None, None, None), ("a", "b", "c")])
def test_bot_state_preserves_unresolved_speaker_caption_rows_without_unknown_label(
    tmp_path, caption_ids
):
    from plugins.google_meet.meet_bot import _BotState

    out = tmp_path / "s"
    state = _BotState(
        out_dir=out, meeting_id="x-y-z", url="https://meet.google.com/x-y-z"
    )

    state.record_caption("", "text but no speaker", caption_id=caption_ids[0])
    state.record_caption(
        "Unknown", "text but no resolved speaker", caption_id=caption_ids[1]
    )
    state.record_caption(
        "unknown", "text but still unresolved", caption_id=caption_ids[2]
    )
    state.record_caption("Alice", "resolved text")

    transcript = (out / "transcript.txt").read_text()
    assert "Unknown:" not in transcript
    assert "Unresolved speaker: text but no speaker" in transcript
    assert "Unresolved speaker: text but no resolved speaker" in transcript
    assert "Unresolved speaker: text but still unresolved" in transcript
    assert "Alice: resolved text" in transcript

    status = json.loads((out / "status.json").read_text())
    assert status["transcriptLines"] == 4
    assert status["unresolvedCaptionLines"] == 3
    assert status["unresolvedCaptionDrops"] == 0


def test_bot_state_drops_unresolved_caption_rows_that_are_only_ui_chrome(tmp_path):
    from plugins.google_meet.meet_bot import _BotState

    out = tmp_path / "s"
    state = _BotState(out_dir=out, meeting_id="x-y-z",
                      url="https://meet.google.com/x-y-z")

    state.record_caption("", "Open caption settings")
    state.record_caption("Unknown", "Audio settings")

    status = json.loads((out / "status.json").read_text())
    assert status["transcriptLines"] == 0
    assert status["unresolvedCaptionDrops"] == 2
    transcript_path = out / "transcript.txt"
    assert not transcript_path.exists()


def test_bot_state_drops_resolved_caption_settings_chrome_row(tmp_path):
    from plugins.google_meet.meet_bot import _BotState

    out = tmp_path / "s"
    state = _BotState(out_dir=out, meeting_id="x-y-z",
                      url="https://meet.google.com/x-y-z")

    state.record_caption(
        "Alex Rivera",
        "language English format_size Font size circle Font colour settings Open caption settings",
        speaker_source="captionRow",
    )
    state.record_caption("Alex Rivera", "actual caption text", speaker_source="captionRow")

    transcript = (out / "transcript.txt").read_text()
    assert "Open caption settings" not in transcript
    assert "actual caption text" in transcript

    status = json.loads((out / "status.json").read_text())
    assert status["transcriptLines"] == 1
    assert status["captionUiNoiseDrops"] == 1


def test_bot_state_revises_unresolved_caption_rows_when_caption_id_is_stable(tmp_path):
    from plugins.google_meet.meet_bot import _BotState

    out = tmp_path / "s"
    state = _BotState(
        out_dir=out, meeting_id="x-y-z", url="https://meet.google.com/x-y-z"
    )

    state.record_caption("", "we should star", caption_id="row-unresolved")
    state.record_caption("", "we should start now", caption_id="row-unresolved")

    transcript = (out / "transcript.txt").read_text().splitlines()
    assert len(transcript) == 1
    assert transcript[0].endswith("Unresolved speaker: we should start now")
    status = json.loads((out / "status.json").read_text())
    assert status["transcriptLines"] == 1
    assert status["unresolvedCaptionLines"] == 2


def test_bot_state_writes_caption_debug_for_unknown_speaker_when_debug_enabled(tmp_path, monkeypatch):
    from plugins.google_meet.meet_bot import _BotState

    monkeypatch.setenv("HERMES_MEET_DEBUG_STATUS", "1")
    state = _BotState(out_dir=tmp_path / "s", meeting_id="x-y-z",
                      url="https://meet.google.com/x-y-z")
    speaker_debug = {
        "candidates": [
            {"selector": "[aria-label]", "raw": "Switch account", "clean": ""},
            {"selector": "[aria-label*='speaking']", "raw": "", "clean": ""},
        ]
    }

    state.record_caption(
        "",
        "text but no speaker",
        speaker_source="unresolved",
        speaker_debug=speaker_debug,
    )

    status = json.loads((tmp_path / "s" / "status.json").read_text())
    assert status["lastSpeakerSource"] == "unresolved"
    assert status["lastSpeakerCandidates"] == speaker_debug["candidates"]
    assert status["captionDebugPath"].endswith("caption_debug.jsonl")
    assert status["transcriptLines"] == 1
    assert status["unresolvedCaptionLines"] == 1
    assert status["unresolvedCaptionDrops"] == 0

    debug_lines = (tmp_path / "s" / "caption_debug.jsonl").read_text().splitlines()
    assert len(debug_lines) == 1
    debug = json.loads(debug_lines[0])
    assert debug["speakerSource"] == "unresolved"
    assert debug["speakerDebug"] == speaker_debug


def test_bot_state_minimizes_ui_debug_fields_by_default(tmp_path):
    from plugins.google_meet.meet_bot import _BotState

    state = _BotState(out_dir=tmp_path / "s", meeting_id="x-y-z",
                      url="https://meet.google.com/x-y-z")
    speaker_debug = {
        "candidates": [
            {"selector": "[aria-label]", "raw": "Alex Rivera alice@example.com", "clean": ""},
        ]
    }
    state.record_caption(
        "",
        "text but no speaker",
        speaker_source="unresolved",
        speaker_debug=speaker_debug,
    )
    state.heartbeat(
        phase="waiting_lobby",
        stalled_reason=None,
        last_ui_text="Alex Rivera alice@example.com is waiting",
        last_url="https://meet.google.com/x-y-z",
    )

    status = json.loads((tmp_path / "s" / "status.json").read_text())
    assert status["lastUiText"] is None
    assert status["lastSpeakerCandidates"] == []
    assert status["captionDebugPath"] is None
    assert not (tmp_path / "s" / "caption_debug.jsonl").exists()


def _run_caption_observer_js(
    *,
    body_text: str,
    caption_text: str,
    speaking_label: str,
    caption_label_rows: list[tuple[str, str]] | None = None,
    caption_label_updates: list[list[tuple[str, str]]] | None = None,
):
    node = shutil.which("node")
    if not node:
        pytest.skip("node is required to execute caption observer JavaScript")

    from plugins.google_meet.meet_bot import _CAPTION_OBSERVER_JS

    script = f"""
const intervals = [];
const observers = [];
global.setInterval = (fn) => {{ intervals.push(fn); return intervals.length; }};
global.MutationObserver = class {{
  constructor(fn) {{ this.fn = fn; observers.push(this); }}
  observe() {{}}
}};

const bodyText = {json.dumps(body_text)};
const captionText = {json.dumps(caption_text)};
const speakingLabel = {json.dumps(speaking_label)};
const captionLabelRows = {json.dumps(caption_label_rows or [])};
const captionLabelUpdates = {json.dumps(caption_label_updates or [])};

function makeNode(attrs, innerText = '') {{
  const node = {{
    innerText,
    parentElement: null,
    children: [],
    getAttribute: (name) => attrs[name] || '',
    querySelectorAll: () => [],
    querySelector: () => null,
    closest: () => null,
  }};
  return node;
}}

function makeCaptionLabelRow(speaker, text) {{
  const row = makeNode({{}}, `${{speaker}}\\n${{text}}`);
  const labelDiv = makeNode({{}}, speaker);
  const labelSpan = makeNode({{}}, speaker);
  const textDiv = makeNode({{}}, text);
  const setText = (nextText) => {{
    textDiv.innerText = nextText;
    row.innerText = `${{speaker}}\\n${{nextText}}`;
  }};
  labelSpan.parentElement = labelDiv;
  labelDiv.parentElement = row;
  textDiv.parentElement = row;
  labelDiv.children = [labelSpan];
  row.children = [labelDiv, textDiv];
  labelSpan.closest = () => row;
  labelDiv.closest = () => row;
  row.querySelectorAll = (selector) => {{
    if (selector.includes('span.NWpY1d') || selector.includes('.NWpY1d')) {{
      return [labelSpan];
    }}
    return [];
  }};
  row.querySelector = (selector) => row.querySelectorAll(selector)[0] || null;
  return {{
    row,
    labelSpan,
    labelDiv,
    textDiv,
    setText,
  }};
}}

const labelRows = captionLabelRows.map(([speaker, text]) => makeCaptionLabelRow(speaker, text));
const captionRoot = captionText || labelRows.length
  ? {{
      innerText: captionText || labelRows.map(({{ row }}) => row.innerText).join('\\n'),
      querySelectorAll: (selector) => {{
        if (selector.includes('div[jsname="dsyhDe"]') || selector.includes('div.CNusmb') || selector.includes('div.TBMuR')) {{
          return [];
        }}
        if (selector.includes('span.NWpY1d') || selector.includes('.NWpY1d')) {{
          return labelRows.map(({{ labelSpan }}) => labelSpan);
        }}
        return [];
      }},
      querySelector: (selector) => {{
        const matches = captionRoot.querySelectorAll(selector);
        return matches[0] || null;
      }},
    }}
  : null;
if (captionRoot) {{
  for (const item of labelRows) item.row.parentElement = captionRoot;
}}

global.window = {{}};
global.document = {{
  body: {{ innerText: bodyText }},
  querySelector: (selector) => {{
    if (
      captionRoot &&
      (selector.includes('[role="region"]') ||
       selector.includes('jsname="YSxPC"') ||
       selector.includes('jsname="tgaKEf"'))
    ) {{
      return captionRoot;
    }}
    return null;
  }},
  querySelectorAll: (selector) => {{
    if (selector.includes('speaking') && speakingLabel) {{
      return [makeNode({{ 'aria-label': speakingLabel }})];
    }}
    if (selector === '[aria-label]' && speakingLabel) {{
      return [makeNode({{ 'aria-label': speakingLabel }})];
    }}
    if (selector.includes('span.NWpY1d') || selector.includes('.NWpY1d')) {{
      return labelRows.map(({{ labelSpan }}) => labelSpan);
    }}
    return [];
  }},
}};

{_CAPTION_OBSERVER_JS}

for (const fn of intervals) fn();
const drained = [];
drained.push(...window.__hermesMeetDrain());
for (const update of captionLabelUpdates) {{
  update.forEach(([speaker, text], index) => {{
    if (labelRows[index]) labelRows[index].setText(text);
  }});
  for (const observer of observers) observer.fn();
  drained.push(...window.__hermesMeetDrain());
}}
process.stdout.write(JSON.stringify(drained));
"""
    proc = subprocess.run(
        [node],
        input=script,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_caption_observer_region_fallback_uses_inferred_speaker_without_throwing():
    entries = _run_caption_observer_js(
        body_text="Alex Rivera is speaking",
        caption_text="like that, but whatever.",
        speaking_label="Alex Rivera is speaking",
    )
    assert [(entry["speaker"], entry["text"]) for entry in entries] == [
        ("Alex Rivera", "like that, but whatever.")
    ]


def test_caption_observer_body_fallback_splits_live_caption_shape():
    entries = _run_caption_observer_js(
        body_text=(
            "Pin Alex Rivera to your main screen\n"
            "More options for Alex Rivera\n"
            "Open caption settings Alex Rivera like that, but whatever. "
            "keyboard_arrow_up Audio settings mic_off Turn on microphone"
        ),
        caption_text="",
        speaking_label="",
    )

    assert entries == [
        {
            "ts": entries[0]["ts"],
            "speaker": "Alex Rivera",
            "speakerSource": "captionRow",
            "speakerDebug": {"candidates": []},
            "captionId": entries[0]["captionId"],
            "text": "like that, but whatever.",
        }
    ]


def test_caption_observer_region_fallback_splits_multiple_live_speakers():
    entries = _run_caption_observer_js(
        body_text=(
            "Pin Alex Rivera to your main screen\n"
            "More options for Alex Rivera\n"
            "Pin Jordan Lee to your main screen\n"
            "More options for Jordan Lee\n"
        ),
        caption_text=(
            "Alex Rivera Hello, is this thing working? "
            "Jordan Lee Hello, is this thing working? "
            "Alex Rivera How's it going, everyone?"
        ),
        speaking_label="",
    )

    simplified = [(entry["speaker"], entry["text"]) for entry in entries]
    assert simplified == [
        ("Alex Rivera", "Hello, is this thing working?"),
        ("Jordan Lee", "Hello, is this thing working?"),
        ("Alex Rivera", "How's it going, everyone?"),
    ]
    assert all(entry["speakerSource"] == "captionRow" for entry in entries)


def test_caption_observer_skips_document_fallback_when_caption_rows_exist():
    entries = _run_caption_observer_js(
        body_text=(
            "Pin Alex Rivera to your main screen\n"
            "More options for Alex Rivera\n"
            "Pin Jordan Lee to your main screen\n"
            "More options for Jordan Lee\n"
            "Open caption settings "
            "Alex Rivera Old accumulated caption history. "
            "Jordan Lee More old accumulated caption history. "
            "Alex Rivera Another old accumulated caption fragment. "
            "keyboard_arrow_up Audio settings mic_off Turn on microphone"
        ),
        caption_text="",
        speaking_label="",
        caption_label_rows=[
            ("Alex Rivera", "Fresh visible caption."),
            ("Jordan Lee", "Another fresh visible caption."),
        ],
    )

    simplified = [(entry["speaker"], entry["text"]) for entry in entries]
    assert simplified == [
        ("Alex Rivera", "Fresh visible caption."),
        ("Jordan Lee", "Another fresh visible caption."),
    ]


def test_caption_observer_scans_live_visible_speaker_labels_without_old_row_class():
    entries = _run_caption_observer_js(
        body_text="",
        caption_text="",
        speaking_label="",
        caption_label_rows=[
            ("Alex Rivera", "Testing the first caption."),
            ("Jordan Lee", "Testing the second caption."),
        ],
    )

    simplified = [(entry["speaker"], entry["text"]) for entry in entries]
    assert simplified == [
        ("Alex Rivera", "Testing the first caption."),
        ("Jordan Lee", "Testing the second caption."),
    ]
    assert all(entry["speakerSource"] == "captionRow" for entry in entries)


def test_caption_observer_emits_full_text_for_growing_visible_caption_row():
    entries = _run_caption_observer_js(
        body_text="",
        caption_text="",
        speaking_label="",
        caption_label_rows=[
            ("Alex Rivera", "Testing the first caption."),
        ],
        caption_label_updates=[
            [("Alex Rivera", "Testing the first caption. New words from the same row.")],
        ],
    )

    simplified = [(entry["speaker"], entry["text"]) for entry in entries]
    assert simplified == [
        ("Alex Rivera", "Testing the first caption."),
        ("Alex Rivera", "Testing the first caption. New words from the same row."),
    ]
    assert all(entry["speakerSource"] == "captionRow" for entry in entries)


def test_caption_observer_emits_stable_caption_ids_for_visible_rows():
    entries = _run_caption_observer_js(
        body_text="",
        caption_text="",
        speaking_label="",
        caption_label_rows=[
            ("Alex Rivera", "Shared prefix first thought."),
            ("Alex Rivera", "Shared prefix second thought."),
        ],
        caption_label_updates=[
            [
                ("Alex Rivera", "Shared prefix first thought with more detail."),
                ("Alex Rivera", "Shared prefix second thought."),
            ],
        ],
    )

    simplified = [(entry["speaker"], entry["text"], entry.get("captionId")) for entry in entries]
    assert simplified[0][0:2] == ("Alex Rivera", "Shared prefix first thought.")
    assert simplified[1][0:2] == ("Alex Rivera", "Shared prefix second thought.")
    assert simplified[2][0:2] == ("Alex Rivera", "Shared prefix first thought with more detail.")
    assert simplified[0][2]
    assert simplified[1][2]
    assert simplified[0][2] != simplified[1][2]
    assert simplified[2][2] == simplified[0][2]
    assert all(entry["speakerSource"] == "captionRow" for entry in entries)


def test_caption_observer_caption_ids_preserve_same_speaker_rows_in_bot_state(tmp_path):
    from plugins.google_meet.meet_bot import _BotState

    entries = _run_caption_observer_js(
        body_text="",
        caption_text="",
        speaking_label="",
        caption_label_rows=[
            ("Alex Rivera", "Shared prefix first thought."),
            ("Alex Rivera", "Shared prefix second thought."),
        ],
        caption_label_updates=[
            [
                ("Alex Rivera", "Shared prefix first thought with more detail."),
                ("Alex Rivera", "Shared prefix second thought."),
            ],
        ],
    )
    state = _BotState(out_dir=tmp_path / "session", meeting_id="abc-defg-hij",
                      url="https://meet.google.com/abc-defg-hij")

    for entry in entries:
        state.record_caption(
            entry.get("speaker", ""),
            entry.get("text", ""),
            speaker_source=entry.get("speakerSource"),
            speaker_debug=entry.get("speakerDebug"),
            caption_id=entry.get("captionId"),
        )

    transcript = (tmp_path / "session" / "transcript.txt").read_text().splitlines()
    assert [line.split("] ", 1)[1] for line in transcript] == [
        "Alex Rivera: Shared prefix first thought with more detail.",
        "Alex Rivera: Shared prefix second thought.",
    ]


def test_caption_observer_fallback_segments_preserve_same_speaker_rows_in_bot_state(tmp_path):
    from plugins.google_meet.meet_bot import _BotState

    entries = _run_caption_observer_js(
        body_text=(
            "Pin Alex Rivera to your main screen\n"
            "More options for Alex Rivera\n"
        ),
        caption_text=(
            "Alex Rivera Shared prefix first thought. "
            "Alex Rivera Shared prefix second thought."
        ),
        speaking_label="",
    )
    state = _BotState(out_dir=tmp_path / "session", meeting_id="abc-defg-hij",
                      url="https://meet.google.com/abc-defg-hij")

    for entry in entries:
        state.record_caption(
            entry.get("speaker", ""),
            entry.get("text", ""),
            speaker_source=entry.get("speakerSource"),
            speaker_debug=entry.get("speakerDebug"),
            caption_id=entry.get("captionId"),
        )

    transcript = (tmp_path / "session" / "transcript.txt").read_text().splitlines()
    assert [line.split("] ", 1)[1] for line in transcript] == [
        "Alex Rivera: Shared prefix first thought.",
        "Alex Rivera: Shared prefix second thought.",
    ]
    assert entries[0].get("captionId") != entries[1].get("captionId")


def test_caption_observer_emits_initial_large_visible_caption_for_python_split():
    old_history = " ".join(f"old{i}" for i in range(220))
    entries = _run_caption_observer_js(
        body_text="",
        caption_text="",
        speaking_label="",
        caption_label_rows=[
            ("Alex Rivera", old_history),
        ],
        caption_label_updates=[
            [("Alex Rivera", f"{old_history} Fresh words after reset.")],
        ],
    )

    simplified = [(entry["speaker"], entry["text"]) for entry in entries]
    assert simplified == [
        ("Alex Rivera", old_history),
        ("Alex Rivera", f"{old_history} Fresh words after reset."),
    ]
    assert all(entry["speakerSource"] == "captionRow" for entry in entries)


def test_bot_state_exposes_v2_telemetry_fields(tmp_path):
    from plugins.google_meet.meet_bot import _BotState

    state = _BotState(out_dir=tmp_path / "s", meeting_id="x-y-z",
                      url="https://meet.google.com/x-y-z")
    # Defaults for the new fields.
    status = json.loads((tmp_path / "s" / "status.json").read_text())
    for key in (
        "realtime", "realtimeReady", "realtimeDevice",
        "realtimeAudioPumpStatus", "realtimeAudioPumpTool",
        "realtimeAudioPumpPid", "realtimeAudioPumpReturnCode",
        "realtimeAudioPumpError",
        "audioBytesOut", "lastAudioOutAt", "lastBargeInAt",
        "joinAttemptedAt", "leaveReason",
        "phase", "lastHeartbeatAt", "lastProgressAt",
        "stalledReason", "lastUiText", "lastUrl",
    ):
        assert key in status, f"missing v2 telemetry key: {key}"
    assert status["realtime"] is False
    assert status["realtimeReady"] is False
    assert status["realtimeAudioPumpStatus"] == "disabled"
    assert status["audioBytesOut"] == 0
    assert status["phase"] == "starting"

    # Setting them flushes them.
    state.set(realtime=True, realtime_ready=True, audio_bytes_out=1024,
              leave_reason="lobby_timeout")
    status = json.loads((tmp_path / "s" / "status.json").read_text())
    assert status["realtime"] is True
    assert status["realtimeReady"] is True
    assert status["audioBytesOut"] == 1024
    assert status["leaveReason"] == "lobby_timeout"


def test_bot_state_heartbeat_flushes_phase_and_diagnostics_when_debug_enabled(tmp_path, monkeypatch):
    from plugins.google_meet.meet_bot import _BotState

    monkeypatch.setenv("HERMES_MEET_DEBUG_STATUS", "1")
    state = _BotState(out_dir=tmp_path / "s", meeting_id="x-y-z",
                      url="https://meet.google.com/x-y-z")
    before = json.loads((tmp_path / "s" / "status.json").read_text())["lastHeartbeatAt"]

    state.heartbeat(
        phase="stalled",
        stalled_reason="no admission progress",
        last_ui_text="Waiting for someone to let you in",
        last_url="https://meet.google.com/x-y-z",
    )

    status = json.loads((tmp_path / "s" / "status.json").read_text())
    assert status["phase"] == "stalled"
    assert status["stalledReason"] == "no admission progress"
    assert status["lastUiText"] == "Waiting for someone to let you in"
    assert status["lastUrl"] == "https://meet.google.com/x-y-z"
    assert status["lastHeartbeatAt"] >= before
