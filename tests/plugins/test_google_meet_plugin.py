"""Tests for the google_meet plugin.

Covers the safety-gated pieces that don't require Playwright:

  * URL regex — only ``https://meet.google.com/`` URLs pass
  * Meeting-id extraction from Meet URLs
  * Status / transcript writes round-trip through the file-backed state
  * Tool handlers return well-formed JSON under all branches
  * Process manager refuses unsafe URLs and clears stale state cleanly
  * Meet cleanup hooks are defensive and only finalization stops live bots

Does NOT spawn a real Chromium — we mock ``subprocess.Popen`` where needed.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    yield hermes_home


# ---------------------------------------------------------------------------
# URL safety gate
# ---------------------------------------------------------------------------

def test_is_safe_meet_url_accepts_standard_meet_codes():
    from plugins.google_meet.meet_bot import _is_safe_meet_url

    assert _is_safe_meet_url("https://meet.google.com/abc-defg-hij")
    assert _is_safe_meet_url("https://meet.google.com/abc-defg-hij?pli=1")
    assert _is_safe_meet_url("https://meet.google.com/new")
    assert _is_safe_meet_url("https://meet.google.com/lookup/ABC123")


def test_is_safe_meet_url_rejects_non_meet_urls():
    from plugins.google_meet.meet_bot import _is_safe_meet_url

    # wrong host
    assert not _is_safe_meet_url("https://evil.example.com/abc-defg-hij")
    # wrong scheme
    assert not _is_safe_meet_url("http://meet.google.com/abc-defg-hij")
    # malformed code
    assert not _is_safe_meet_url("https://meet.google.com/not-a-meet-code")
    # subdomain hijack attempts
    assert not _is_safe_meet_url("https://meet.google.com.evil.com/abc-defg-hij")
    assert not _is_safe_meet_url("https://notmeet.google.com/abc-defg-hij")
    # empty / wrong type
    assert not _is_safe_meet_url("")
    assert not _is_safe_meet_url(None)  # type: ignore[arg-type]
    assert not _is_safe_meet_url(123)  # type: ignore[arg-type]


def test_meeting_id_extraction():
    from plugins.google_meet.meet_bot import _meeting_id_from_url

    assert _meeting_id_from_url("https://meet.google.com/abc-defg-hij") == "abc-defg-hij"
    assert _meeting_id_from_url("https://meet.google.com/abc-defg-hij?pli=1") == "abc-defg-hij"
    # fallback for codes we can't parse (e.g. /new before redirect)
    fallback = _meeting_id_from_url("https://meet.google.com/new")
    assert fallback.startswith("meet-")


# ---------------------------------------------------------------------------
# _BotState — transcript + status file round-trip
# ---------------------------------------------------------------------------

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


def test_bot_state_preserves_unresolved_speaker_caption_rows_without_unknown_label(tmp_path):
    from plugins.google_meet.meet_bot import _BotState

    out = tmp_path / "s"
    state = _BotState(out_dir=out, meeting_id="x-y-z",
                      url="https://meet.google.com/x-y-z")

    state.record_caption("", "text but no speaker")
    state.record_caption("Unknown", "text but no resolved speaker")
    state.record_caption("unknown", "text but still unresolved")
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
    state = _BotState(out_dir=out, meeting_id="x-y-z",
                      url="https://meet.google.com/x-y-z")

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


def test_caption_observer_uses_active_speaker_fallback():
    from plugins.google_meet.meet_bot import _CAPTION_OBSERVER_JS

    assert "function inferActiveSpeaker" in _CAPTION_OBSERVER_JS
    assert "__hermesMeetLastSpeaker" in _CAPTION_OBSERVER_JS
    assert '[aria-label*="speaking" i]' in _CAPTION_OBSERVER_JS
    assert "inferActiveSpeaker()" in _CAPTION_OBSERVER_JS


def test_caption_observer_filters_account_chrome_from_speaker_fallback():
    from plugins.google_meet.meet_bot import _CAPTION_OBSERVER_JS

    assert "switch account" in _CAPTION_OBSERVER_JS.lower()
    assert "getting ready" in _CAPTION_OBSERVER_JS.lower()
    assert "you'll be able to join in just a moment" in _CAPTION_OBSERVER_JS.lower()
    assert "speakerDebug" in _CAPTION_OBSERVER_JS


def test_caption_observer_has_visible_body_caption_fallback():
    from plugins.google_meet.meet_bot import _CAPTION_OBSERVER_JS

    assert "function inferParticipantNames" in _CAPTION_OBSERVER_JS
    assert "function scanDocumentFallback" in _CAPTION_OBSERVER_JS
    assert "Open caption settings" in _CAPTION_OBSERVER_JS
    assert r"Pin\s+(.+?)\s+to your main screen" in _CAPTION_OBSERVER_JS
    assert "scanDocumentFallback()" in _CAPTION_OBSERVER_JS


def test_caption_observer_strips_meet_controls_from_body_fallback():
    from plugins.google_meet.meet_bot import _CAPTION_OBSERVER_JS

    assert "function trimCaptionChrome" in _CAPTION_OBSERVER_JS
    assert "keyboard_arrow_up Audio settings" in _CAPTION_OBSERVER_JS
    assert "Turn on microphone" in _CAPTION_OBSERVER_JS
    assert "Meeting tools" in _CAPTION_OBSERVER_JS
    assert "trimCaptionChrome(text)" in _CAPTION_OBSERVER_JS


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

    assert entries == [
        {
            "ts": entries[0]["ts"],
            "speaker": "Alex Rivera",
            "speakerSource": '[aria-label*="speaking" i]',
            "speakerDebug": {
                "candidates": [
                    {
                        "selector": '[aria-label*="speaking" i]',
                        "attr": "aria-label",
                        "raw": "Alex Rivera is speaking",
                        "clean": "Alex Rivera",
                        "diagnosticOnly": False,
                    },
                    {
                        "selector": '[aria-label*="speaking" i]',
                        "attr": "innerText",
                        "raw": "",
                        "clean": "",
                        "diagnosticOnly": False,
                    },
                    {
                        "selector": '[aria-label*="is speaking" i]',
                        "attr": "aria-label",
                        "raw": "Alex Rivera is speaking",
                        "clean": "Alex Rivera",
                        "diagnosticOnly": False,
                    },
                    {
                        "selector": '[aria-label*="is speaking" i]',
                        "attr": "innerText",
                        "raw": "",
                        "clean": "",
                        "diagnosticOnly": False,
                    },
                    {
                        "selector": "[aria-label]",
                        "attr": "aria-label",
                        "raw": "Alex Rivera is speaking",
                        "clean": "Alex Rivera",
                        "diagnosticOnly": True,
                    },
                ]
            },
            "text": "like that, but whatever.",
        }
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


def test_parse_duration():
    from plugins.google_meet.meet_bot import _parse_duration

    assert _parse_duration("30m") == 30 * 60
    assert _parse_duration("2h") == 2 * 3600
    assert _parse_duration("90s") == 90
    assert _parse_duration("90") == 90
    assert _parse_duration("") is None
    assert _parse_duration("bogus") is None


# ---------------------------------------------------------------------------
# process_manager — refuses unsafe URLs, manages active pointer
# ---------------------------------------------------------------------------

def test_start_refuses_unsafe_url():
    from plugins.google_meet import process_manager as pm

    res = pm.start("https://evil.example.com/abc-defg-hij")
    assert res["ok"] is False
    assert "refusing" in res["error"]


def test_status_reports_no_active_meeting():
    from plugins.google_meet import process_manager as pm

    assert pm.status() == {"ok": False, "reason": "no active meeting"}
    assert pm.transcript() == {"ok": False, "reason": "no active meeting"}
    assert pm.stop() == {"ok": False, "reason": "no active meeting"}


def test_start_spawns_subprocess_and_writes_active_pointer(tmp_path):
    """Verify start() wires env vars correctly and records the pid."""
    from plugins.google_meet import process_manager as pm

    class _FakeProc:
        def __init__(self, pid):
            self.pid = pid

    captured_env = {}
    captured_argv = []

    def _fake_popen(argv, **kwargs):
        captured_argv.extend(argv)
        captured_env.update(kwargs.get("env") or {})
        return _FakeProc(99999)

    with patch.object(pm.subprocess, "Popen", side_effect=_fake_popen):
        # Also prevent pid liveness probe from stomping on our real pids
        with patch.object(pm, "_pid_alive", return_value=False):
            res = pm.start(
                "https://meet.google.com/abc-defg-hij",
                guest_name="Test Bot",
                duration="15m",
            )

    assert res["ok"] is True
    assert res["meeting_id"] == "abc-defg-hij"
    assert res["pid"] == 99999
    assert captured_env["HERMES_MEET_URL"] == "https://meet.google.com/abc-defg-hij"
    assert captured_env["HERMES_MEET_GUEST_NAME"] == "Test Bot"
    assert captured_env["HERMES_MEET_DURATION"] == "15m"
    # python -m plugins.google_meet.meet_bot
    assert any("plugins.google_meet.meet_bot" in a for a in captured_argv)

    # .active.json points at the bot
    active = pm._read_active()
    assert active is not None
    assert active["pid"] == 99999
    assert active["meeting_id"] == "abc-defg-hij"
    assert active["duration"] == "15m"


def test_start_headed_uses_xvfb_when_display_is_missing(monkeypatch):
    """A service-mode headed launch must be wrapped with xvfb-run."""
    from plugins.google_meet import process_manager as pm

    class _FakeProc:
        pid = 99998

    captured_env = {}
    captured_argv = []

    def _fake_popen(argv, **kwargs):
        captured_argv.extend(argv)
        captured_env.update(kwargs.get("env") or {})
        return _FakeProc()

    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setenv("HERMES_MEET_XVFB", "auto")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")

    with patch.object(pm.subprocess, "Popen", side_effect=_fake_popen), \
         patch.object(pm, "_pid_alive", return_value=False):
        res = pm.start("https://meet.google.com/abc-defg-hij", headed=True)

    assert res["ok"] is True
    assert Path(captured_argv[0]).name == "xvfb-run"
    assert captured_argv[1] == "-a"
    assert any("plugins.google_meet.meet_bot" in arg for arg in captured_argv)
    assert captured_env["HERMES_MEET_HEADED"] == "1"
    assert res["headed"] is True
    assert res["xvfb"] is True

    active = pm._read_active()
    assert active is not None
    assert active["headed"] is True
    assert active["xvfb"] is True


def test_start_headed_rejects_without_display_or_xvfb(monkeypatch):
    """Without DISPLAY or xvfb-run, fail before spawning Chromium."""
    from plugins.google_meet import process_manager as pm

    popen_called = False

    def _fake_popen(_argv, **_kwargs):
        nonlocal popen_called
        popen_called = True
        class _FakeProc:
            pid = 99997
        return _FakeProc()

    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setenv("HERMES_MEET_XVFB", "auto")
    monkeypatch.setenv("PATH", "/definitely-no-xvfb-here")

    with patch.object(pm.subprocess, "Popen", side_effect=_fake_popen), \
         patch.object(pm, "_pid_alive", return_value=False):
        res = pm.start("https://meet.google.com/abc-defg-hij", headed=True)

    assert res["ok"] is False
    assert "headed" in res["error"].lower()
    assert "xvfb-run" in res["error"]
    assert popen_called is False
    assert pm._read_active() is None


def test_status_clears_stale_active_pointer_when_bot_exited(tmp_path):
    """A dead bot with a final status file is no longer an active meeting."""
    from plugins.google_meet import process_manager as pm

    out_dir = tmp_path / "abc-defg-hij"
    out_dir.mkdir()
    (out_dir / "status.json").write_text(json.dumps({
        "meetingId": "abc-defg-hij",
        "exited": True,
        "leaveReason": "meet_landing",
        "error": "meet returned to landing before captions",
    }))
    pm._write_active({
        "pid": 11111,
        "meeting_id": "abc-defg-hij",
        "out_dir": str(out_dir),
        "url": "https://meet.google.com/abc-defg-hij",
        "started_at": 0,
    })

    with patch.object(pm, "_pid_alive", return_value=False):
        res = pm.status()

    assert res["ok"] is False
    assert res["reason"] == "no active meeting"
    assert res["lastStatus"]["leaveReason"] == "meet_landing"
    assert pm._read_active() is None


def test_transcript_does_not_read_last_meeting_by_default_after_status_clears_dead_active_pointer(tmp_path):
    from plugins.google_meet import process_manager as pm

    out_dir = tmp_path / "abc-defg-hij"
    out_dir.mkdir()
    (out_dir / "status.json").write_text(json.dumps({
        "meetingId": "abc-defg-hij",
        "exited": True,
        "leaveReason": "duration_expired",
    }))
    (out_dir / "transcript.txt").write_text(
        "[10:00:00] Alex Rivera: one\n"
        "[10:00:01] Morgan Lee: two\n",
        encoding="utf-8",
    )
    pm._write_active({
        "pid": 11111,
        "meeting_id": "abc-defg-hij",
        "out_dir": str(out_dir),
        "url": "https://meet.google.com/abc-defg-hij",
        "started_at": 0,
    })

    with patch.object(pm, "_pid_alive", return_value=False):
        status = pm.status()
    transcript = pm.transcript()

    assert status["ok"] is False
    assert transcript["ok"] is False
    assert "no active meeting" in transcript["reason"]


def test_transcript_clears_dead_active_pointer_without_prior_status_call(tmp_path):
    from plugins.google_meet import process_manager as pm

    out_dir = tmp_path / "abc-defg-hij"
    out_dir.mkdir()
    (out_dir / "status.json").write_text(json.dumps({
        "meetingId": "abc-defg-hij",
        "exited": True,
        "leaveReason": "duration_expired",
    }))
    (out_dir / "transcript.txt").write_text(
        "[10:00:00] Alex Rivera: one\n",
        encoding="utf-8",
    )
    pm._write_active({
        "pid": 11111,
        "meeting_id": "abc-defg-hij",
        "out_dir": str(out_dir),
        "url": "https://meet.google.com/abc-defg-hij",
        "started_at": 0,
        "session_id": "session-a",
    })

    with patch.object(pm, "_pid_alive", return_value=False):
        default_read = pm.transcript()
        finished_read = pm.transcript(include_finished=True, session_id="session-a")

    assert default_read["ok"] is False
    assert "no active meeting" in default_read["reason"]
    assert finished_read["ok"] is True
    assert finished_read["active"] is False
    assert finished_read["fromLast"] is True
    assert finished_read["stale"] is True
    assert pm._read_active() is None


def test_transcript_can_explicitly_read_finished_meeting_after_status_clears_dead_active_pointer(tmp_path):
    from plugins.google_meet import process_manager as pm

    out_dir = tmp_path / "abc-defg-hij"
    out_dir.mkdir()
    (out_dir / "status.json").write_text(json.dumps({
        "meetingId": "abc-defg-hij",
        "exited": True,
        "leaveReason": "duration_expired",
    }))
    (out_dir / "transcript.txt").write_text(
        "[10:00:00] Alex Rivera: one\n"
        "[10:00:01] Morgan Lee: two\n",
        encoding="utf-8",
    )
    pm._write_active({
        "pid": 11111,
        "meeting_id": "abc-defg-hij",
        "out_dir": str(out_dir),
        "url": "https://meet.google.com/abc-defg-hij",
        "started_at": 0,
        "session_id": "session-a",
    })

    with patch.object(pm, "_pid_alive", return_value=False):
        status = pm.status()
    transcript = pm.transcript(include_finished=True, session_id="session-a")

    assert status["ok"] is False
    assert transcript["ok"] is True
    assert transcript["active"] is False
    assert transcript["fromLast"] is True
    assert transcript["stale"] is True
    assert transcript["sessionId"] == "session-a"
    assert transcript["leaveReason"] == "duration_expired"
    assert transcript["total"] == 2
    assert transcript["lines"][-1].endswith("Morgan Lee: two")


def test_transcript_finished_meeting_requires_matching_session_id(tmp_path):
    from plugins.google_meet import process_manager as pm

    out_dir = tmp_path / "abc-defg-hij"
    out_dir.mkdir()
    (out_dir / "status.json").write_text(json.dumps({
        "meetingId": "abc-defg-hij",
        "exited": True,
        "leaveReason": "duration_expired",
    }))
    (out_dir / "transcript.txt").write_text(
        "[10:00:00] Alex Rivera: one\n",
        encoding="utf-8",
    )
    pm._write_active({
        "pid": 11111,
        "meeting_id": "abc-defg-hij",
        "out_dir": str(out_dir),
        "url": "https://meet.google.com/abc-defg-hij",
        "started_at": 0,
        "session_id": "session-a",
    })

    with patch.object(pm, "_pid_alive", return_value=False):
        pm.status()

    no_session = pm.transcript(include_finished=True)
    other_session = pm.transcript(include_finished=True, session_id="session-b")

    assert no_session["ok"] is False
    assert "session id" in no_session["reason"]
    assert other_session["ok"] is False
    assert "no finished meeting" in other_session["reason"]


def test_transcript_reads_last_n_lines(tmp_path):
    from plugins.google_meet import process_manager as pm

    meeting_dir = Path(os.environ["HERMES_HOME"]) / "workspace" / "meetings" / "abc-defg-hij"
    meeting_dir.mkdir(parents=True)
    (meeting_dir / "transcript.txt").write_text(
        "[10:00:00] Alice: one\n"
        "[10:00:01] Bob: two\n"
        "[10:00:02] Alice: three\n"
    )
    pm._write_active({
        "pid": 12345, "meeting_id": "abc-defg-hij",
        "out_dir": str(meeting_dir),
        "url": "https://meet.google.com/abc-defg-hij",
        "started_at": 0,
    })

    with patch.object(pm, "_pid_alive", return_value=True):
        res = pm.transcript(last=2)
    assert res["ok"] is True
    assert res["total"] == 3
    assert len(res["lines"]) == 2
    assert res["lines"][-1].endswith("Alice: three")


def test_stop_signals_process_and_clears_pointer(tmp_path):
    from plugins.google_meet import process_manager as pm

    out_dir = tmp_path / "x-y-z"
    out_dir.mkdir()
    (out_dir / "status.json").write_text(json.dumps({
        "meetingId": "x-y-z",
        "exited": False,
        "leaveReason": None,
    }))
    pm._write_active({
        "pid": 11111, "meeting_id": "x-y-z",
        "out_dir": str(out_dir),
        "url": "https://meet.google.com/x-y-z",
        "started_at": 0,
    })

    alive_seq = iter([True, True, False])  # alive at first, gone after SIGTERM
    def _alive(pid):
        try:
            return next(alive_seq)
        except StopIteration:
            return False

    sent = []
    def _kill(pid, sig):
        sent.append((pid, sig))

    with patch.object(pm, "_pid_alive", side_effect=_alive), \
         patch.object(pm.os, "kill", side_effect=_kill), \
         patch.object(pm.time, "sleep", lambda _s: None):
        res = pm.stop()

    assert res["ok"] is True
    assert (11111, signal.SIGTERM) in sent
    status = json.loads((out_dir / "status.json").read_text())
    assert status["exited"] is True
    assert status["leaveReason"] == "requested"
    # .active.json cleared
    assert pm._read_active() is None


# ---------------------------------------------------------------------------
# Tool handlers — JSON shape + safety gates
# ---------------------------------------------------------------------------

def test_meet_join_handler_missing_url_returns_error():
    from plugins.google_meet.tools import handle_meet_join

    out = json.loads(handle_meet_join({}))
    assert out["success"] is False
    assert "url is required" in out["error"]


def test_meet_join_handler_respects_safety_gate():
    from plugins.google_meet.tools import handle_meet_join

    with patch("plugins.google_meet.tools.check_meet_requirements", return_value=True):
        out = json.loads(handle_meet_join({"url": "https://evil.example.com/foo"}))
    assert out["success"] is False
    assert "refusing" in out["error"]


def test_meet_join_handler_returns_error_when_playwright_missing():
    from plugins.google_meet.tools import handle_meet_join

    with patch("plugins.google_meet.tools.check_meet_requirements", return_value=False):
        out = json.loads(handle_meet_join({"url": "https://meet.google.com/abc-defg-hij"}))
    assert out["success"] is False
    assert "prerequisites missing" in out["error"]


def test_meet_say_requires_text():
    from plugins.google_meet.tools import handle_meet_say

    out = json.loads(handle_meet_say({}))
    assert out["success"] is False
    assert "text is required" in out["error"]


def test_meet_say_no_active_meeting():
    from plugins.google_meet.tools import handle_meet_say

    out = json.loads(handle_meet_say({"text": "hello everyone"}))
    assert out["success"] is False
    # Falls through to pm.enqueue_say which reports no active meeting.
    assert "no active meeting" in out.get("reason", "")


def test_meet_status_and_transcript_no_active():
    from plugins.google_meet.tools import handle_meet_status, handle_meet_transcript

    assert json.loads(handle_meet_status({}))["success"] is False
    assert json.loads(handle_meet_transcript({}))["success"] is False


def test_meet_transcript_passes_session_id_to_process_manager():
    from plugins.google_meet.tools import handle_meet_transcript

    with patch(
        "plugins.google_meet.tools.pm.transcript",
        return_value={"ok": True, "lines": [], "total": 0},
    ) as transcript_mock:
        out = json.loads(handle_meet_transcript({"include_finished": True}, session_id="session-a"))

    assert out["success"] is True
    assert transcript_mock.call_args.kwargs["include_finished"] is True
    assert transcript_mock.call_args.kwargs["session_id"] == "session-a"


def test_meet_leave_no_active():
    from plugins.google_meet.tools import handle_meet_leave

    out = json.loads(handle_meet_leave({}))
    assert out["success"] is False


# ---------------------------------------------------------------------------
# Session cleanup hooks — per-turn end is a no-op; finalization stops owned bots
# ---------------------------------------------------------------------------

def test_on_session_end_noop_when_nothing_active():
    from plugins.google_meet import _on_session_end
    # Should not raise and should not call stop().
    with patch("plugins.google_meet.pm.stop") as stop_mock:
        _on_session_end()
    stop_mock.assert_not_called()


def test_on_session_end_does_not_stop_live_bot():
    from plugins.google_meet import _on_session_end
    from plugins.google_meet import pm

    with patch.object(pm, "status", return_value={"ok": True, "alive": True, "duration": None}), \
         patch.object(pm, "stop") as stop_mock:
        _on_session_end()
    stop_mock.assert_not_called()


def test_on_session_finalize_stops_matching_session_bot():
    from plugins.google_meet import _on_session_finalize
    from plugins.google_meet import pm

    with patch.object(
        pm,
        "status",
        return_value={
            "ok": True,
            "alive": True,
            "sessionId": "session-a",
            "persistAfterSession": False,
        },
    ), patch.object(pm, "stop") as stop_mock:
        _on_session_finalize(session_id="session-a")
    stop_mock.assert_called_once_with(reason="session ended")


def test_on_session_end_does_not_stop_matching_session_bot():
    from plugins.google_meet import _on_session_end
    from plugins.google_meet import pm

    with patch.object(
        pm,
        "status",
        return_value={
            "ok": True,
            "alive": True,
            "sessionId": "session-a",
            "persistAfterSession": False,
        },
    ), patch.object(pm, "stop") as stop_mock:
        _on_session_end(session_id="session-a")
    stop_mock.assert_not_called()


def test_on_session_finalize_does_not_stop_other_session_bot():
    from plugins.google_meet import _on_session_finalize
    from plugins.google_meet import pm

    with patch.object(
        pm,
        "status",
        return_value={
            "ok": True,
            "alive": True,
            "sessionId": "session-a",
            "persistAfterSession": False,
        },
    ), patch.object(pm, "stop") as stop_mock:
        _on_session_finalize(session_id="session-b")
    stop_mock.assert_not_called()


def test_on_session_finalize_stops_matching_remote_node_bot():
    from plugins.google_meet import _on_session_finalize
    from plugins.google_meet.node.registry import NodeRegistry

    reg = NodeRegistry()
    reg.add("my-mac", "ws://1.2.3.4:18789", "tok")

    with patch("plugins.google_meet.pm.status", return_value={"ok": False}), \
         patch(
             "plugins.google_meet.node.client.NodeClient.status",
             return_value={
                 "ok": True,
                 "alive": True,
                 "sessionId": "session-a",
                 "persistAfterSession": False,
             },
         ), patch("plugins.google_meet.node.client.NodeClient.stop", return_value={"ok": True}) as stop_mock:
        _on_session_finalize(session_id="session-a")

    stop_mock.assert_called_once_with(reason="session ended")


def test_on_session_finalize_does_not_stop_other_session_remote_node_bot():
    from plugins.google_meet import _on_session_finalize
    from plugins.google_meet.node.registry import NodeRegistry

    reg = NodeRegistry()
    reg.add("my-mac", "ws://1.2.3.4:18789", "tok")

    with patch("plugins.google_meet.pm.status", return_value={"ok": False}), \
         patch(
             "plugins.google_meet.node.client.NodeClient.status",
             return_value={
                 "ok": True,
                 "alive": True,
                 "sessionId": "session-a",
                 "persistAfterSession": False,
             },
         ), patch("plugins.google_meet.node.client.NodeClient.stop", return_value={"ok": True}) as stop_mock:
        _on_session_finalize(session_id="session-b")

    stop_mock.assert_not_called()


def test_on_session_finalize_stops_duration_limited_bot_owned_by_ending_session():
    from plugins.google_meet import _on_session_finalize
    from plugins.google_meet import pm

    with patch.object(
        pm,
        "status",
        return_value={
            "ok": True,
            "alive": True,
            "duration": "20m",
            "sessionId": "session-a",
        },
    ), \
         patch.object(pm, "stop") as stop_mock:
        _on_session_finalize(session_id="session-a")
    stop_mock.assert_called_once_with(reason="session ended")


def test_on_session_finalize_does_not_stop_bot_when_session_id_is_missing():
    from plugins.google_meet import _on_session_finalize
    from plugins.google_meet import pm

    with patch.object(
        pm,
        "status",
        return_value={
            "ok": True,
            "alive": True,
            "duration": "20m",
            "sessionId": "session-a",
        },
    ), \
         patch.object(pm, "stop") as stop_mock:
        _on_session_finalize()
    stop_mock.assert_not_called()


def test_on_session_finalize_keeps_explicitly_detached_bot_running():
    from plugins.google_meet import _on_session_finalize
    from plugins.google_meet import pm

    with patch.object(
        pm,
        "status",
        return_value={
            "ok": True,
            "alive": True,
            "duration": "20m",
            "persistAfterSession": True,
        },
    ), patch.object(pm, "stop") as stop_mock:
        _on_session_finalize()
    stop_mock.assert_not_called()


# ---------------------------------------------------------------------------
# Plugin register() — platform gating + tool registration
# ---------------------------------------------------------------------------

def test_register_refuses_on_windows():
    import plugins.google_meet as plugin

    calls = {"tools": [], "cli": [], "hooks": []}

    class _Ctx:
        def register_tool(self, **kw): calls["tools"].append(kw["name"])
        def register_cli_command(self, **kw): calls["cli"].append(kw["name"])
        def register_hook(self, name, fn): calls["hooks"].append(name)

    with patch.object(plugin.platform, "system", return_value="Windows"):
        plugin.register(_Ctx())

    assert calls == {"tools": [], "cli": [], "hooks": []}


def test_register_wires_tools_cli_and_hook_on_linux():
    import plugins.google_meet as plugin

    calls = {"tools": [], "cli": [], "hooks": []}

    class _Ctx:
        def register_tool(self, **kw): calls["tools"].append(kw["name"])
        def register_cli_command(self, **kw): calls["cli"].append(kw["name"])
        def register_hook(self, name, fn): calls["hooks"].append(name)

    with patch.object(plugin.platform, "system", return_value="Linux"):
        plugin.register(_Ctx())

    assert set(calls["tools"]) == {
        "meet_join", "meet_status", "meet_transcript", "meet_leave", "meet_say",
    }
    assert calls["cli"] == ["meet"]
    assert calls["hooks"] == ["on_session_finalize"]


# ---------------------------------------------------------------------------
# v2: process_manager.enqueue_say + realtime-mode passthrough
# ---------------------------------------------------------------------------

def test_enqueue_say_requires_text():
    from plugins.google_meet import process_manager as pm
    assert pm.enqueue_say("")["ok"] is False
    assert pm.enqueue_say("   ")["ok"] is False


def test_enqueue_say_no_active_meeting():
    from plugins.google_meet import process_manager as pm
    res = pm.enqueue_say("hi team")
    assert res["ok"] is False
    assert "no active meeting" in res["reason"]


def test_enqueue_say_rejects_transcribe_mode(tmp_path):
    from plugins.google_meet import process_manager as pm

    out_dir = Path(os.environ["HERMES_HOME"]) / "workspace" / "meetings" / "abc-defg-hij"
    out_dir.mkdir(parents=True)
    pm._write_active({
        "pid": 0, "meeting_id": "abc-defg-hij",
        "out_dir": str(out_dir), "url": "https://meet.google.com/abc-defg-hij",
        "started_at": 0, "mode": "transcribe",
    })
    res = pm.enqueue_say("hi team")
    assert res["ok"] is False
    assert "transcribe mode" in res["reason"]


def test_enqueue_say_rejects_dead_realtime_bot(tmp_path):
    from plugins.google_meet import process_manager as pm

    out_dir = Path(os.environ["HERMES_HOME"]) / "workspace" / "meetings" / "abc-defg-hij"
    out_dir.mkdir(parents=True)
    (out_dir / "status.json").write_text(json.dumps({
        "inCall": True,
        "realtime": True,
        "realtimeReady": True,
        "realtimeAudioPumpStatus": "ready",
        "localMicrophoneOn": True,
    }))
    pm._write_active({
        "pid": 12345, "meeting_id": "abc-defg-hij",
        "out_dir": str(out_dir), "url": "https://meet.google.com/abc-defg-hij",
        "started_at": 0, "mode": "realtime",
    })

    with patch.object(pm, "_pid_alive", return_value=False):
        res = pm.enqueue_say("hello everyone")

    assert res["ok"] is False
    assert "no active meeting" in res["reason"]
    assert not (out_dir / "say_queue.jsonl").exists()


def test_enqueue_say_rejects_realtime_before_bot_is_ready(tmp_path):
    from plugins.google_meet import process_manager as pm

    out_dir = Path(os.environ["HERMES_HOME"]) / "workspace" / "meetings" / "abc-defg-hij"
    out_dir.mkdir(parents=True)
    (out_dir / "status.json").write_text(json.dumps({
        "inCall": True,
        "realtime": True,
        "realtimeReady": False,
    }))
    pm._write_active({
        "pid": 12345, "meeting_id": "abc-defg-hij",
        "out_dir": str(out_dir), "url": "https://meet.google.com/abc-defg-hij",
        "started_at": 0, "mode": "realtime",
    })

    with patch.object(pm, "_pid_alive", return_value=True):
        res = pm.enqueue_say("hello everyone")

    assert res["ok"] is False
    assert "realtime is not ready" in res["reason"]
    assert not (out_dir / "say_queue.jsonl").exists()


def test_enqueue_say_rejects_realtime_when_audio_pump_is_not_ready(tmp_path):
    from plugins.google_meet import process_manager as pm

    out_dir = Path(os.environ["HERMES_HOME"]) / "workspace" / "meetings" / "abc-defg-hij"
    out_dir.mkdir(parents=True)
    (out_dir / "status.json").write_text(json.dumps({
        "inCall": True,
        "realtime": True,
        "realtimeReady": True,
        "realtimeAudioPumpStatus": "exited",
        "realtimeAudioPumpReturnCode": 1,
        "error": None,
        "exited": False,
    }))
    pm._write_active({
        "pid": 12345, "meeting_id": "abc-defg-hij",
        "out_dir": str(out_dir), "url": "https://meet.google.com/abc-defg-hij",
        "started_at": 0, "mode": "realtime",
    })

    with patch.object(pm, "_pid_alive", return_value=True):
        res = pm.enqueue_say("hello everyone")

    assert res["ok"] is False
    assert "audio pump is not ready" in res["reason"]
    assert not (out_dir / "say_queue.jsonl").exists()


def test_enqueue_say_rejects_realtime_when_meet_microphone_is_off(tmp_path):
    from plugins.google_meet import process_manager as pm

    out_dir = Path(os.environ["HERMES_HOME"]) / "workspace" / "meetings" / "abc-defg-hij"
    out_dir.mkdir(parents=True)
    (out_dir / "status.json").write_text(json.dumps({
        "inCall": True,
        "realtime": True,
        "realtimeReady": True,
        "realtimeAudioPumpStatus": "ready",
        "localMicrophoneOn": False,
        "error": None,
        "exited": False,
    }))
    pm._write_active({
        "pid": 12345, "meeting_id": "abc-defg-hij",
        "out_dir": str(out_dir), "url": "https://meet.google.com/abc-defg-hij",
        "started_at": 0, "mode": "realtime",
    })

    with patch.object(pm, "_pid_alive", return_value=True):
        res = pm.enqueue_say("hello everyone")

    assert res["ok"] is False
    assert "microphone is not enabled" in res["reason"]
    assert not (out_dir / "say_queue.jsonl").exists()


def test_enqueue_say_writes_jsonl_in_realtime_mode():
    from plugins.google_meet import process_manager as pm

    out_dir = Path(os.environ["HERMES_HOME"]) / "workspace" / "meetings" / "abc-defg-hij"
    out_dir.mkdir(parents=True)
    (out_dir / "status.json").write_text(json.dumps({
        "inCall": True,
        "realtime": True,
        "realtimeReady": True,
        "realtimeAudioPumpStatus": "ready",
        "localMicrophoneOn": True,
        "error": None,
        "exited": False,
    }))
    pm._write_active({
        "pid": 12345, "meeting_id": "abc-defg-hij",
        "out_dir": str(out_dir), "url": "https://meet.google.com/abc-defg-hij",
        "started_at": 0, "mode": "realtime",
    })
    with patch.object(pm, "_pid_alive", return_value=True):
        res = pm.enqueue_say("hello everyone")
    assert res["ok"] is True
    assert "enqueued_id" in res

    queue = out_dir / "say_queue.jsonl"
    assert queue.is_file()
    lines = [json.loads(ln) for ln in queue.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1
    assert lines[0]["text"] == "hello everyone"


def test_realtime_queue_preserves_append_during_consumer_rewrite(tmp_path):
    from plugins.google_meet import process_manager as pm
    from plugins.google_meet.realtime.openai_client import RealtimeSpeaker

    out_dir = Path(os.environ["HERMES_HOME"]) / "workspace" / "meetings" / "abc-defg-hij"
    out_dir.mkdir(parents=True)
    queue_path = out_dir / "say_queue.jsonl"
    queue_path.write_text(json.dumps({"id": "first", "text": "first"}) + "\n")
    (out_dir / "status.json").write_text(json.dumps({
        "inCall": True,
        "realtime": True,
        "realtimeReady": True,
        "realtimeAudioPumpStatus": "ready",
        "localMicrophoneOn": True,
    }))
    pm._write_active({
        "pid": 12345, "meeting_id": "abc-defg-hij",
        "out_dir": str(out_dir), "url": "https://meet.google.com/abc-defg-hij",
        "started_at": 0, "mode": "realtime",
    })

    stop = {"value": False}
    rewrite_started = threading.Event()
    original_write_text = Path.write_text

    class _Session:
        def speak(self, _text):
            stop["value"] = True
            return {"ok": True, "bytes_written": 0, "duration_ms": 0.0}

    def _slow_queue_write(path, text, *args, **kwargs):
        if path == queue_path and text == "":
            rewrite_started.set()
            time.sleep(0.2)
        return original_write_text(path, text, *args, **kwargs)

    speaker = RealtimeSpeaker(session=_Session(), queue_path=queue_path)
    with patch.object(Path, "write_text", _slow_queue_write), \
         patch.object(pm, "_pid_alive", return_value=True):
        consumer = threading.Thread(
            target=speaker.run_until_stopped,
            args=(lambda: stop["value"],),
            kwargs={"poll_interval": 0.01},
        )
        consumer.start()
        assert rewrite_started.wait(timeout=2)
        res = pm.enqueue_say("second")
        consumer.join(timeout=2)

    assert res["ok"] is True
    assert not consumer.is_alive()
    remaining = [
        json.loads(line)
        for line in queue_path.read_text().splitlines()
        if line.strip()
    ]
    assert remaining == [{"id": res["enqueued_id"], "text": "second"}]


def test_start_passes_mode_into_active_record():
    from plugins.google_meet import process_manager as pm

    class _FakeProc:
        def __init__(self, pid): self.pid = pid

    with patch.object(pm.subprocess, "Popen", return_value=_FakeProc(12345)), \
         patch.object(pm, "_pid_alive", return_value=False):
        res = pm.start(
            "https://meet.google.com/abc-defg-hij",
            mode="realtime",
        )
    assert res["ok"] is True
    assert res["mode"] == "realtime"
    assert pm._read_active()["mode"] == "realtime"


def test_start_realtime_env_vars_threaded_through():
    from plugins.google_meet import process_manager as pm

    class _FakeProc:
        def __init__(self, pid): self.pid = pid

    captured_env = {}
    def _fake_popen(argv, **kwargs):
        captured_env.update(kwargs.get("env") or {})
        return _FakeProc(11111)

    with patch.object(pm.subprocess, "Popen", side_effect=_fake_popen), \
         patch.object(pm, "_pid_alive", return_value=False):
        pm.start(
            "https://meet.google.com/abc-defg-hij",
            mode="realtime",
            realtime_model="gpt-realtime",
            realtime_voice="alloy",
            realtime_instructions="Be brief.",
            realtime_api_key="sk-test",
        )
    assert captured_env["HERMES_MEET_MODE"] == "realtime"
    assert captured_env["HERMES_MEET_REALTIME_MODEL"] == "gpt-realtime"
    assert captured_env["HERMES_MEET_REALTIME_VOICE"] == "alloy"
    assert captured_env["HERMES_MEET_REALTIME_INSTRUCTIONS"] == "Be brief."
    assert captured_env["HERMES_MEET_REALTIME_KEY"] == "sk-test"


def test_meet_join_accepts_realtime_mode():
    from plugins.google_meet.tools import handle_meet_join

    with patch("plugins.google_meet.tools.check_meet_requirements", return_value=True), \
         patch("plugins.google_meet.tools.pm.start", return_value={"ok": True, "meeting_id": "x-y-z"}) as start_mock:
        out = json.loads(handle_meet_join({
            "url": "https://meet.google.com/abc-defg-hij",
            "mode": "realtime",
        }))
    assert out["success"] is True
    assert start_mock.call_args.kwargs["mode"] == "realtime"


def test_meet_join_does_not_reuse_saved_auth_state_by_default():
    from plugins.google_meet.tools import handle_meet_join

    auth_path = Path(os.environ["HERMES_HOME"]) / "workspace" / "meetings" / "auth.json"
    auth_path.parent.mkdir(parents=True, exist_ok=True)
    auth_path.write_text(json.dumps({"cookies": [], "origins": []}))

    with patch("plugins.google_meet.tools.check_meet_requirements", return_value=True), \
         patch("plugins.google_meet.tools.pm.start", return_value={"ok": True, "meeting_id": "x-y-z"}) as start_mock:
        out = json.loads(handle_meet_join({
            "url": "https://meet.google.com/abc-defg-hij",
        }))

    assert out["success"] is True
    assert start_mock.call_args.kwargs["auth_state"] is None


def test_meet_join_reuses_saved_auth_state_only_when_explicitly_requested():
    from plugins.google_meet.tools import handle_meet_join

    auth_path = Path(os.environ["HERMES_HOME"]) / "workspace" / "meetings" / "auth.json"
    auth_path.parent.mkdir(parents=True, exist_ok=True)
    auth_path.write_text(json.dumps({"cookies": [], "origins": []}))

    with patch("plugins.google_meet.tools.check_meet_requirements", return_value=True), \
         patch("plugins.google_meet.tools.pm.start", return_value={"ok": True, "meeting_id": "x-y-z"}) as start_mock:
        out = json.loads(handle_meet_join({
            "url": "https://meet.google.com/abc-defg-hij",
            "use_auth_state": True,
        }))

    assert out["success"] is True
    assert start_mock.call_args.kwargs["auth_state"] == str(auth_path)


def test_meet_join_does_not_set_default_duration():
    from plugins.google_meet.tools import handle_meet_join

    with patch("plugins.google_meet.tools.check_meet_requirements", return_value=True), \
         patch("plugins.google_meet.tools.pm.start", return_value={"ok": True, "meeting_id": "x-y-z"}) as start_mock:
        out = json.loads(handle_meet_join({
            "url": "https://meet.google.com/abc-defg-hij",
        }))

    assert out["success"] is True
    assert start_mock.call_args.kwargs["duration"] is None


def test_meet_join_passes_persist_after_session_only_when_requested():
    from plugins.google_meet.tools import handle_meet_join

    with patch("plugins.google_meet.tools.check_meet_requirements", return_value=True), \
         patch("plugins.google_meet.tools.pm.start", return_value={"ok": True, "meeting_id": "x-y-z"}) as start_mock:
        out = json.loads(handle_meet_join({
            "url": "https://meet.google.com/abc-defg-hij",
            "persist_after_session": True,
        }))

    assert out["success"] is True
    assert start_mock.call_args.kwargs["persist_after_session"] is True


def test_meet_join_passes_session_id_from_tool_context():
    from plugins.google_meet.tools import handle_meet_join

    with patch("plugins.google_meet.tools.check_meet_requirements", return_value=True), \
         patch("plugins.google_meet.tools.pm.start", return_value={"ok": True, "meeting_id": "x-y-z"}) as start_mock:
        out = json.loads(handle_meet_join(
            {"url": "https://meet.google.com/abc-defg-hij"},
            session_id="session-a",
        ))

    assert out["success"] is True
    assert start_mock.call_args.kwargs["session_id"] == "session-a"


def test_meet_join_honors_explicit_duration():
    from plugins.google_meet.tools import handle_meet_join

    with patch("plugins.google_meet.tools.check_meet_requirements", return_value=True), \
         patch("plugins.google_meet.tools.pm.start", return_value={"ok": True, "meeting_id": "x-y-z"}) as start_mock:
        out = json.loads(handle_meet_join({
            "url": "https://meet.google.com/abc-defg-hij",
            "duration": "15m",
        }))

    assert out["success"] is True
    assert start_mock.call_args.kwargs["duration"] == "15m"


def test_meet_join_rejects_bad_mode():
    from plugins.google_meet.tools import handle_meet_join

    out = json.loads(handle_meet_join({
        "url": "https://meet.google.com/abc-defg-hij",
        "mode": "bogus",
    }))
    assert out["success"] is False
    assert "mode must be" in out["error"]


# ---------------------------------------------------------------------------
# v3: NodeClient routing from tool handlers
# ---------------------------------------------------------------------------

def test_meet_join_unknown_node_returns_clear_error():
    from plugins.google_meet.tools import handle_meet_join

    out = json.loads(handle_meet_join({
        "url": "https://meet.google.com/abc-defg-hij",
        "node": "my-mac",
    }))
    assert out["success"] is False
    assert "no registered meet node" in out["error"]


def test_meet_join_routes_to_registered_node():
    from plugins.google_meet.tools import handle_meet_join
    from plugins.google_meet.node.registry import NodeRegistry

    reg = NodeRegistry()
    reg.add("my-mac", "ws://1.2.3.4:18789", "tok")

    with patch("plugins.google_meet.node.client.NodeClient.start_bot",
               return_value={"ok": True, "meeting_id": "a-b-c"}) as call_mock:
        out = json.loads(handle_meet_join({
            "url": "https://meet.google.com/abc-defg-hij",
            "node": "my-mac",
            "mode": "realtime",
        }))
    assert out["success"] is True
    assert out["node"] == "my-mac"
    assert call_mock.call_args.kwargs["mode"] == "realtime"


def test_meet_join_rejects_auth_state_for_remote_node():
    from plugins.google_meet.tools import handle_meet_join
    from plugins.google_meet.node.registry import NodeRegistry

    reg = NodeRegistry()
    reg.add("my-mac", "ws://1.2.3.4:18789", "tok")

    out = json.loads(handle_meet_join({
        "url": "https://meet.google.com/abc-defg-hij",
        "node": "my-mac",
        "use_auth_state": True,
    }))

    assert out["success"] is False
    assert "use_auth_state is local-only" in out["error"]


def test_meet_say_routes_to_node():
    from plugins.google_meet.tools import handle_meet_say
    from plugins.google_meet.node.registry import NodeRegistry

    reg = NodeRegistry()
    reg.add("my-mac", "ws://1.2.3.4:18789", "tok")

    with patch("plugins.google_meet.node.client.NodeClient.say",
               return_value={"ok": True, "enqueued_id": "abc"}) as call_mock:
        out = json.loads(handle_meet_say({"text": "hello", "node": "my-mac"}))
    assert out["success"] is True
    assert out["node"] == "my-mac"
    call_mock.assert_called_once_with("hello")


def test_node_server_say_rejects_without_active_meeting(tmp_path):
    from plugins.google_meet.node import protocol as proto
    from plugins.google_meet.node.server import NodeServer

    server = NodeServer(token_path=tmp_path / "node_token.json")
    server._token = "tok"

    response = asyncio.run(server._handle_request(
        proto.make_request("say", "tok", {"text": "hello"})
    ))

    assert response["type"] == "response"
    assert response["payload"]["ok"] is False
    assert "no active meeting" in response["payload"]["reason"]


def test_node_server_say_rejects_transcribe_mode(tmp_path):
    from plugins.google_meet import process_manager as pm
    from plugins.google_meet.node import protocol as proto
    from plugins.google_meet.node.server import NodeServer

    out_dir = Path(os.environ["HERMES_HOME"]) / "workspace" / "meetings" / "abc-defg-hij"
    out_dir.mkdir(parents=True)
    pm._write_active({
        "pid": 0,
        "meeting_id": "abc-defg-hij",
        "out_dir": str(out_dir),
        "url": "https://meet.google.com/abc-defg-hij",
        "started_at": 0,
        "mode": "transcribe",
    })
    server = NodeServer(token_path=tmp_path / "node_token.json")
    server._token = "tok"

    response = asyncio.run(server._handle_request(
        proto.make_request("say", "tok", {"text": "hello"})
    ))

    assert response["type"] == "response"
    assert response["payload"]["ok"] is False
    assert "transcribe mode" in response["payload"]["reason"]


def test_node_server_say_uses_realtime_queue(tmp_path):
    from plugins.google_meet import process_manager as pm
    from plugins.google_meet.node import protocol as proto
    from plugins.google_meet.node.server import NodeServer

    out_dir = Path(os.environ["HERMES_HOME"]) / "workspace" / "meetings" / "abc-defg-hij"
    out_dir.mkdir(parents=True)
    (out_dir / "status.json").write_text(json.dumps({
        "inCall": True,
        "realtime": True,
        "realtimeReady": True,
        "realtimeAudioPumpStatus": "ready",
        "localMicrophoneOn": True,
    }))
    pm._write_active({
        "pid": 12345,
        "meeting_id": "abc-defg-hij",
        "out_dir": str(out_dir),
        "url": "https://meet.google.com/abc-defg-hij",
        "started_at": 0,
        "mode": "realtime",
    })
    server = NodeServer(token_path=tmp_path / "node_token.json")
    server._token = "tok"

    with patch.object(pm, "_pid_alive", return_value=True):
        response = asyncio.run(server._handle_request(
            proto.make_request("say", "tok", {"text": "hello"})
        ))

    assert response["type"] == "response"
    assert response["payload"]["ok"] is True
    assert response["payload"]["enqueued_id"]
    queued = [
        json.loads(line)
        for line in (out_dir / "say_queue.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert queued == [{"id": response["payload"]["enqueued_id"], "text": "hello"}]


def test_meet_join_auto_node_selects_sole_registered():
    from plugins.google_meet.tools import handle_meet_join
    from plugins.google_meet.node.registry import NodeRegistry

    reg = NodeRegistry()
    reg.add("only-one", "ws://1.2.3.4:18789", "tok")

    with patch("plugins.google_meet.node.client.NodeClient.start_bot",
               return_value={"ok": True}) as call_mock:
        out = json.loads(handle_meet_join({
            "url": "https://meet.google.com/abc-defg-hij",
            "node": "auto",
        }))
    assert out["success"] is True
    assert out["node"] == "only-one"
    assert call_mock.called


def test_meet_join_auto_node_ambiguous_returns_error():
    from plugins.google_meet.tools import handle_meet_join
    from plugins.google_meet.node.registry import NodeRegistry

    reg = NodeRegistry()
    reg.add("a", "ws://1.2.3.4:18789", "tok")
    reg.add("b", "ws://5.6.7.8:18789", "tok")

    out = json.loads(handle_meet_join({
        "url": "https://meet.google.com/abc-defg-hij",
        "node": "auto",
    }))
    assert out["success"] is False
    assert "no registered meet node" in out["error"]


def test_cli_register_includes_node_subcommand():
    """`hermes meet` argparse tree includes the node subtree."""
    import argparse
    from plugins.google_meet.cli import register_cli

    parser = argparse.ArgumentParser(prog="hermes meet")
    register_cli(parser)

    # Parse a known-good node invocation to prove the subtree is wired.
    ns = parser.parse_args(["node", "list"])
    assert ns.meet_command == "node"
    assert ns.node_cmd == "list"


def test_cli_join_accepts_mode_and_node_flags():
    import argparse
    from plugins.google_meet.cli import register_cli

    parser = argparse.ArgumentParser(prog="hermes meet")
    register_cli(parser)

    ns = parser.parse_args([
        "join", "https://meet.google.com/abc-defg-hij",
        "--mode", "realtime", "--node", "my-mac",
    ])
    assert ns.mode == "realtime"
    assert ns.node == "my-mac"


def test_cli_join_accepts_auth_and_persist_flags():
    import argparse
    from plugins.google_meet.cli import register_cli

    parser = argparse.ArgumentParser(prog="hermes meet")
    register_cli(parser)

    ns = parser.parse_args([
        "join", "https://meet.google.com/abc-defg-hij",
        "--use-auth-state", "--persist-after-session",
    ])
    assert ns.use_auth_state is True
    assert ns.persist_after_session is True


def test_cli_join_does_not_reuse_saved_auth_state_by_default(tmp_path):
    from plugins.google_meet.cli import _auth_state_path, _cmd_join

    auth_path = _auth_state_path()
    auth_path.parent.mkdir(parents=True, exist_ok=True)
    auth_path.write_text("{}", encoding="utf-8")

    with patch("plugins.google_meet.cli.pm.start", return_value={"ok": True}) as start_mock:
        rc = _cmd_join(
            "https://meet.google.com/abc-defg-hij",
            guest_name="Hermes Agent",
            duration=None,
            headed=False,
            mode="transcribe",
            node=None,
            use_auth_state=False,
            persist_after_session=False,
        )

    assert rc == 0
    assert start_mock.call_args.kwargs["auth_state"] is None
    assert start_mock.call_args.kwargs["persist_after_session"] is False


def test_cli_join_reuses_saved_auth_state_only_when_explicitly_requested(tmp_path):
    from plugins.google_meet.cli import _auth_state_path, _cmd_join

    auth_path = _auth_state_path()
    auth_path.parent.mkdir(parents=True, exist_ok=True)
    auth_path.write_text("{}", encoding="utf-8")

    with patch("plugins.google_meet.cli.pm.start", return_value={"ok": True}) as start_mock:
        rc = _cmd_join(
            "https://meet.google.com/abc-defg-hij",
            guest_name="Hermes Agent",
            duration=None,
            headed=False,
            mode="transcribe",
            node=None,
            use_auth_state=True,
            persist_after_session=True,
        )

    assert rc == 0
    assert start_mock.call_args.kwargs["auth_state"] == str(auth_path)
    assert start_mock.call_args.kwargs["persist_after_session"] is True


def test_cli_join_rejects_auth_state_for_remote_node(capsys):
    from plugins.google_meet.cli import _cmd_join
    from plugins.google_meet.node.registry import NodeRegistry

    reg = NodeRegistry()
    reg.add("my-mac", "ws://1.2.3.4:18789", "tok")

    rc = _cmd_join(
        "https://meet.google.com/abc-defg-hij",
        guest_name="Hermes Agent",
        duration=None,
        headed=False,
        mode="transcribe",
        node="my-mac",
        use_auth_state=True,
        persist_after_session=False,
    )

    assert rc == 1
    assert "use_auth_state is local-only" in capsys.readouterr().out


def test_cli_say_subcommand_exists():
    import argparse
    from plugins.google_meet.cli import register_cli

    parser = argparse.ArgumentParser(prog="hermes meet")
    register_cli(parser)

    ns = parser.parse_args(["say", "hello team", "--node", "my-mac"])
    assert ns.text == "hello team"
    assert ns.node == "my-mac"


# ---------------------------------------------------------------------------
# v2.1: new _BotState fields + status dict shape
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Admission detection + barge-in helper
# ---------------------------------------------------------------------------

def test_looks_like_human_speaker():
    from plugins.google_meet.meet_bot import _looks_like_human_speaker

    # Blank, "unknown", "you", and the bot's own name → not human (no barge-in)
    for s in ("", "   ", "Unknown", "unknown", "You", "you", "Hermes Agent", "hermes agent"):
        assert not _looks_like_human_speaker(s, "Hermes Agent"), f"{s!r} should NOT be human"
    # Real names → human (barge-in)
    for s in ("Alice", "Bob Lee", "@teknium"):
        assert _looks_like_human_speaker(s, "Hermes Agent"), f"{s!r} SHOULD be human"


def test_detect_admission_returns_false_on_error():
    from plugins.google_meet.meet_bot import _detect_admission

    class _FakePage:
        def evaluate(self, _js): raise RuntimeError("boom")

    assert _detect_admission(_FakePage()) is False


def test_detect_admission_true_when_probe_returns_true():
    from plugins.google_meet.meet_bot import _detect_admission

    class _FakePage:
        def evaluate(self, _js): return True

    assert _detect_admission(_FakePage()) is True


def test_detect_admission_true_from_rich_probe_dict():
    from plugins.google_meet.meet_bot import _detect_admission

    class _FakePage:
        def evaluate(self, _js):
            return {"inCall": True, "waitingLobby": False, "denied": False}

    assert _detect_admission(_FakePage()) is True


def test_detect_admission_false_when_probe_is_still_prejoin():
    from plugins.google_meet.meet_bot import _detect_admission

    class _FakePage:
        def evaluate(self, _js):
            return {
                "inCall": True,
                "waitingLobby": False,
                "denied": False,
                "preJoin": True,
                "text": "Getting ready... You'll be able to join in just a moment",
            }

    assert _detect_admission(_FakePage()) is False


def test_classify_meet_ui_blocks_getting_ready_page():
    from plugins.google_meet.meet_bot import _classify_meet_ui

    result = _classify_meet_ui(
        "Getting ready... You'll be able to join in just a moment",
        in_call_control=True,
        url="https://meet.google.com/abc-defg-hij",
    )

    assert result["inCall"] is False
    assert result["preJoin"] is True


def test_classify_meet_ui_blocks_ready_to_join_page_with_media_prompt():
    from plugins.google_meet.meet_bot import _classify_meet_ui

    result = _classify_meet_ui(
        (
            "mic Show more info videocam Show more info Ready to join? "
            "Alex Lee and Morgan Patel are in this call Join now "
            "Do you want people to see and hear you in the meeting? "
            "Continue without microphone and camera"
        ),
        in_call_control=True,
        url="https://meet.google.com/abc-defg-hij",
    )

    assert result["inCall"] is False
    assert result["preJoin"] is True


def test_classify_meet_ui_treats_host_wait_phrase_as_lobby():
    from plugins.google_meet.meet_bot import _classify_meet_ui

    result = _classify_meet_ui(
        (
            "Please wait until a meeting host brings you into the call "
            "Turn on microphone Turn on camera Leave call"
        ),
        leave=True,
        in_call_control=True,
        url="https://meet.google.com/abc-defg-hij",
    )

    assert result["inCall"] is False
    assert result["waitingLobby"] is True


def test_classify_meet_ui_blocks_landing_page_after_meet_error():
    from plugins.google_meet.meet_bot import _classify_meet_ui

    result = _classify_meet_ui(
        "Meet Secure video conferencing for everyone New meeting Join",
        in_call_text=True,
        url="https://meet.google.com/landing",
    )

    assert result["inCall"] is False
    assert result["landing"] is True


def test_classify_meet_ui_blocks_workspace_meet_product_redirect():
    from plugins.google_meet.meet_bot import _classify_meet_ui

    result = _classify_meet_ui(
        "Google Meet video meetings and calls for everyone",
        url="https://workspace.google.com/products/meet/",
    )

    assert result["inCall"] is False
    assert result["landing"] is True


def test_classify_meet_ui_blocks_could_not_start_video_call_error():
    from plugins.google_meet.meet_bot import _classify_meet_ui

    result = _classify_meet_ui(
        "Couldn't start the video call because of an error",
        in_call_control=True,
        url="https://meet.google.com/landing",
    )

    assert result["inCall"] is False
    assert result["callError"] is True


def test_detect_denied_returns_false_on_error():
    from plugins.google_meet.meet_bot import _detect_denied

    class _FakePage:
        def evaluate(self, _js): raise RuntimeError("boom")

    assert _detect_denied(_FakePage()) is False


def test_detect_denied_true_from_rich_probe_dict():
    from plugins.google_meet.meet_bot import _detect_denied

    class _FakePage:
        def evaluate(self, _js):
            return {"inCall": False, "waitingLobby": False, "denied": True}

    assert _detect_denied(_FakePage()) is True


def test_classify_meet_ui_marks_return_home_denial_terminal():
    from plugins.google_meet.meet_bot import _classify_meet_ui

    result = _classify_meet_ui(
        (
            "You can't join this video call Return to home screen "
            "No one can join a meeting unless invited or admitted by the host "
            "Returning to home screen 45 seconds left"
        ),
        url="https://meet.google.com/abc-defg-hij",
    )

    assert result["denied"] is True
    assert result["terminalDenied"] is True


def test_classify_meet_ui_marks_policy_denial_terminal_without_join_click():
    from plugins.google_meet.meet_bot import _classify_meet_ui

    result = _classify_meet_ui(
        "No one can join a meeting unless invited or admitted by the host",
        url="https://meet.google.com/abc-defg-hij",
    )

    assert result["denied"] is True
    assert result["terminalDenied"] is True


def test_classify_meet_ui_preserves_in_call_signal_with_transient_call_error():
    from plugins.google_meet.meet_bot import _classify_meet_ui

    result = _classify_meet_ui(
        "Couldn't start the video call because of an error Meeting details",
        leave=True,
        in_call_control=True,
        url="https://meet.google.com/abc-defg-hij",
    )

    assert result["callError"] is True
    assert result["inCall"] is True


def test_compute_meet_phase_marks_join_attempt_as_stalled(tmp_path):
    from plugins.google_meet.meet_bot import _BotState, _compute_meet_phase

    state = _BotState(
        out_dir=tmp_path / "meet",
        meeting_id="abc-defg-hij",
        url="https://meet.google.com/abc-defg-hij",
    )
    state.set(join_attempted_at=100.0)

    phase, reason = _compute_meet_phase(state, now=205.0, stall_after=90.0)

    assert phase == "stalled"
    assert "no admission" in reason


def test_compute_meet_phase_reports_capturing_after_transcript(tmp_path):
    from plugins.google_meet.meet_bot import _BotState, _compute_meet_phase

    state = _BotState(
        out_dir=tmp_path / "meet",
        meeting_id="abc-defg-hij",
        url="https://meet.google.com/abc-defg-hij",
    )
    state.set(in_call=True, joined_at=100.0)
    state.record_caption("Alice", "hello")

    phase, reason = _compute_meet_phase(state, now=110.0, stall_after=90.0)

    assert phase == "capturing"
    assert reason is None


def test_should_probe_admission_until_first_caption(tmp_path):
    from plugins.google_meet.meet_bot import _BotState, _should_probe_admission

    state = _BotState(
        out_dir=tmp_path / "meet",
        meeting_id="abc-defg-hij",
        url="https://meet.google.com/abc-defg-hij",
    )
    state.set(in_call=True, joined_at=100.0)

    assert _should_probe_admission(state, now=105.0, last_admission_check=100.0) is True

    state.record_caption("Alice", "hello")

    assert _should_probe_admission(state, now=110.0, last_admission_check=105.0) is False


def test_apply_admission_probe_revokes_prejoin_false_positive(tmp_path):
    from plugins.google_meet.meet_bot import _BotState, _apply_admission_probe

    state = _BotState(
        out_dir=tmp_path / "meet",
        meeting_id="abc-defg-hij",
        url="https://meet.google.com/abc-defg-hij",
    )
    state.set(in_call=True, joined_at=100.0)

    admitted, terminal = _apply_admission_probe(
        state,
        {
            "inCall": False,
            "preJoin": True,
            "waitingLobby": False,
            "denied": False,
            "text": "Getting ready... You'll be able to join in just a moment",
        },
        now=105.0,
        lobby_deadline=400.0,
    )

    assert admitted is False
    assert terminal is False
    assert state.in_call is False
    assert state.joined_at is None
    assert state.phase == "joining"


def test_apply_admission_probe_ignores_denied_before_join_attempt(tmp_path):
    from plugins.google_meet.meet_bot import _BotState, _apply_admission_probe

    state = _BotState(
        out_dir=tmp_path / "meet",
        meeting_id="abc-defg-hij",
        url="https://meet.google.com/abc-defg-hij",
    )

    admitted, terminal = _apply_admission_probe(
        state,
        {
            "inCall": False,
            "waitingLobby": False,
            "denied": True,
            "preJoin": False,
            "text": "No one can join a meeting unless invited or admitted by the host",
        },
        now=105.0,
        lobby_deadline=400.0,
    )

    assert admitted is False
    assert terminal is False
    assert state.error is None
    assert state.leave_reason is None
    assert state.phase == "starting"


def test_apply_admission_probe_exits_on_terminal_denial_before_join_attempt(tmp_path):
    from plugins.google_meet.meet_bot import _BotState, _apply_admission_probe

    state = _BotState(
        out_dir=tmp_path / "meet",
        meeting_id="abc-defg-hij",
        url="https://meet.google.com/abc-defg-hij",
    )

    admitted, terminal = _apply_admission_probe(
        state,
        {
            "inCall": False,
            "waitingLobby": False,
            "denied": True,
            "terminalDenied": True,
            "preJoin": False,
            "text": "You can't join this video call Returning to home screen",
        },
        now=105.0,
        lobby_deadline=400.0,
    )

    assert admitted is False
    assert terminal is True
    assert state.error == "host denied admission"
    assert state.leave_reason == "denied"
    assert state.phase == "exited"


def test_apply_admission_probe_exits_when_meet_returns_to_landing(tmp_path):
    from plugins.google_meet.meet_bot import _BotState, _apply_admission_probe

    state = _BotState(
        out_dir=tmp_path / "meet",
        meeting_id="abc-defg-hij",
        url="https://meet.google.com/abc-defg-hij",
    )
    state.set(join_attempted_at=100.0, in_call=True, joined_at=105.0)

    admitted, terminal = _apply_admission_probe(
        state,
        {
            "inCall": False,
            "landing": True,
            "waitingLobby": False,
            "denied": False,
            "text": "Meet Secure video conferencing for everyone New meeting Join",
            "url": "https://meet.google.com/landing",
        },
        now=110.0,
        lobby_deadline=400.0,
    )

    assert admitted is False
    assert terminal is True
    assert state.in_call is False
    assert state.joined_at is None
    assert state.leave_reason == "meet_landing"
    assert "landing" in state.error
    assert state.phase == "exited"


def test_apply_admission_probe_exits_when_meet_redirects_to_landing_before_join(tmp_path):
    from plugins.google_meet.meet_bot import _BotState, _apply_admission_probe

    state = _BotState(
        out_dir=tmp_path / "meet",
        meeting_id="abc-defg-hij",
        url="https://meet.google.com/abc-defg-hij",
    )

    admitted, terminal = _apply_admission_probe(
        state,
        {
            "inCall": False,
            "landing": True,
            "waitingLobby": False,
            "denied": False,
            "text": "Google Meet video meetings and calls for everyone",
            "url": "https://workspace.google.com/products/meet/",
        },
        now=110.0,
        lobby_deadline=400.0,
    )

    assert admitted is False
    assert terminal is True
    assert state.in_call is False
    assert state.joined_at is None
    assert state.leave_reason == "meet_landing"
    assert "landing" in state.error
    assert state.phase == "exited"


def test_apply_admission_probe_tolerates_single_transient_call_error(tmp_path):
    """A one-off Meet "couldn't start the video call" flash must not kill a live session.

    Google Meet routinely shows this banner transiently while the call is in
    fact healthy (3 named participants, mic/cam off were observed in the live
    failure). A single observation must not be treated as terminal.
    """
    from plugins.google_meet.meet_bot import _BotState, _apply_admission_probe

    state = _BotState(
        out_dir=tmp_path / "meet",
        meeting_id="abc-defg-hij",
        url="https://meet.google.com/abc-defg-hij",
    )
    state.set(join_attempted_at=100.0, in_call=True, joined_at=105.0)

    admitted, terminal = _apply_admission_probe(
        state,
        {
            "inCall": False,
            "callError": True,
            "waitingLobby": False,
            "denied": False,
            "text": "Couldn't start the video call because of an error",
            "url": "https://meet.google.com/abc-defg-hij",
        },
        now=110.0,
        lobby_deadline=400.0,
        call_error_strike_limit=3,
    )

    assert terminal is False
    assert state.in_call is True
    assert state.joined_at == 105.0
    assert state.leave_reason is None
    assert state.phase != "exited"
    assert state.call_error_strikes == 1


def test_apply_admission_probe_keeps_call_alive_when_error_overlay_still_has_in_call_controls(tmp_path):
    from plugins.google_meet.meet_bot import _BotState, _apply_admission_probe

    state = _BotState(
        out_dir=tmp_path / "meet",
        meeting_id="abc-defg-hij",
        url="https://meet.google.com/abc-defg-hij",
    )
    state.set(join_attempted_at=100.0, in_call=True, joined_at=105.0)

    terminal = False
    for now in (110.0, 113.0, 116.0, 119.0):
        admitted, terminal = _apply_admission_probe(
            state,
            {
                "inCall": True,
                "callError": True,
                "waitingLobby": False,
                "denied": False,
                "preJoin": False,
                "text": "Couldn't start the video call because of an error Meeting details",
                "url": "https://meet.google.com/abc-defg-hij",
            },
            now=now,
            lobby_deadline=400.0,
            call_error_strike_limit=3,
        )
        assert admitted is True

    assert terminal is False
    assert state.in_call is True
    assert state.joined_at == 105.0
    assert state.leave_reason is None
    assert state.phase != "exited"
    assert state.call_error_strikes == 0


def test_apply_admission_probe_resets_call_error_strikes_when_cleared(tmp_path):
    """Strikes reset once Meet stops reporting the error, so a later flash starts fresh."""
    from plugins.google_meet.meet_bot import _BotState, _apply_admission_probe

    state = _BotState(
        out_dir=tmp_path / "meet",
        meeting_id="abc-defg-hij",
        url="https://meet.google.com/abc-defg-hij",
    )
    state.set(join_attempted_at=100.0, in_call=True, joined_at=105.0)

    error_probe = {
        "inCall": False,
        "callError": True,
        "text": "Couldn't start the video call because of an error",
        "url": "",
    }
    for now in (110.0, 113.0):
        _apply_admission_probe(
            state, error_probe, now=now, lobby_deadline=400.0, call_error_strike_limit=3,
        )
    assert state.call_error_strikes == 2

    # A clean in-call probe clears the count — the error was transient.
    _apply_admission_probe(
        state,
        {"inCall": True, "callError": False, "text": "Meeting details", "url": ""},
        now=116.0,
        lobby_deadline=400.0,
        call_error_strike_limit=3,
    )
    assert state.call_error_strikes == 0
    assert state.in_call is True


def test_apply_admission_probe_exits_on_persistent_call_start_error(tmp_path):
    """Before ever being admitted, a call error persisting past the strike limit
    is terminal — the call genuinely won't start."""
    from plugins.google_meet.meet_bot import _BotState, _apply_admission_probe

    state = _BotState(
        out_dir=tmp_path / "meet",
        meeting_id="abc-defg-hij",
        url="https://meet.google.com/abc-defg-hij",
    )
    state.set(join_attempted_at=100.0)  # never admitted

    probe = {
        "inCall": False,
        "callError": True,
        "waitingLobby": False,
        "denied": False,
        "text": "Couldn't start the video call because of an error",
        "url": "https://meet.google.com/landing",
    }

    terminal = False
    for now in (110.0, 113.0, 116.0):
        _, terminal = _apply_admission_probe(
            state, probe, now=now, lobby_deadline=400.0, call_error_strike_limit=3,
        )

    assert state.ever_admitted is False
    assert terminal is True
    assert state.in_call is False
    assert state.joined_at is None
    assert state.leave_reason == "meet_error"
    assert "error" in state.error
    assert state.phase == "exited"


def test_apply_admission_probe_ignores_call_error_after_caption_evidence(tmp_path):
    """Once captions have arrived, a transient call-error overlay is non-terminal.

    Admission alone is not enough because Meet may expose roster text while
    returning to an error page. Caption evidence proves the bot reached the
    target functionality, so persistent media-layer overlays should not end the
    transcript session.
    """
    from plugins.google_meet.meet_bot import _BotState, _apply_admission_probe

    state = _BotState(
        out_dir=tmp_path / "meet",
        meeting_id="abc-defg-hij",
        url="https://meet.google.com/abc-defg-hij",
    )
    state.set(join_attempted_at=100.0)

    admitted, _ = _apply_admission_probe(
        state,
        {"inCall": True, "callError": False, "text": "Meeting details", "url": ""},
        now=105.0,
        lobby_deadline=400.0,
        call_error_strike_limit=3,
    )
    assert admitted is True
    assert state.ever_admitted is True
    state.set(last_caption_at=106.0, transcript_lines=1)

    err = {
        "inCall": False,
        "callError": True,
        "text": "Couldn't start the video call because of an error",
        "url": "",
    }
    terminal = False
    for now in (108.0, 111.0, 114.0, 117.0, 120.0):
        _, terminal = _apply_admission_probe(
            state, err, now=now, lobby_deadline=400.0, call_error_strike_limit=3,
        )

    assert terminal is False
    assert state.leave_reason != "meet_error"
    assert state.call_error_strikes == 0


def test_apply_admission_probe_exits_on_persistent_call_error_after_false_admission(tmp_path):
    """A false-positive admission must not mask a persistent Meet error page."""
    from plugins.google_meet.meet_bot import _BotState, _apply_admission_probe

    state = _BotState(
        out_dir=tmp_path / "meet",
        meeting_id="abc-defg-hij",
        url="https://meet.google.com/abc-defg-hij",
    )
    state.set(join_attempted_at=100.0)

    admitted, terminal = _apply_admission_probe(
        state,
        {"inCall": True, "callError": False, "text": "Meeting details", "url": ""},
        now=105.0,
        lobby_deadline=400.0,
        call_error_strike_limit=3,
    )
    assert admitted is True
    assert terminal is False
    assert state.ever_admitted is True
    assert state.last_caption_at is None

    err = {
        "inCall": False,
        "callError": True,
        "text": (
            "Couldn't start the video call because of an error "
            "Returning to home screen in 60 seconds."
        ),
        "url": "https://meet.google.com/abc-defg-hij?pli=1",
    }
    for now in (108.0, 111.0, 114.0):
        _, terminal = _apply_admission_probe(
            state, err, now=now, lobby_deadline=400.0, call_error_strike_limit=3,
        )

    assert terminal is True
    assert state.in_call is False
    assert state.joined_at is None
    assert state.leave_reason == "meet_error"
    assert "error" in state.error
    assert state.phase == "exited"


def test_should_retry_captions_after_join_attempt_even_before_admission_flag(tmp_path):
    from plugins.google_meet.meet_bot import _BotState, _should_retry_caption_enable

    state = _BotState(
        out_dir=tmp_path / "meet",
        meeting_id="abc-defg-hij",
        url="https://meet.google.com/abc-defg-hij",
    )
    state.set(join_attempted_at=100.0)

    assert _should_retry_caption_enable(
        state,
        now=105.0,
        last_caption_enable_check=100.0,
    ) is True


def test_should_not_retry_captions_before_join_attempt_or_admission(tmp_path):
    from plugins.google_meet.meet_bot import _BotState, _should_retry_caption_enable

    state = _BotState(
        out_dir=tmp_path / "meet",
        meeting_id="abc-defg-hij",
        url="https://meet.google.com/abc-defg-hij",
    )

    assert _should_retry_caption_enable(
        state,
        now=105.0,
        last_caption_enable_check=100.0,
    ) is False


# ---------------------------------------------------------------------------
# Realtime session counters + cancel_response (barge-in)
# ---------------------------------------------------------------------------

def test_realtime_session_cancel_response_when_disconnected():
    from plugins.google_meet.realtime.openai_client import RealtimeSession

    sess = RealtimeSession(api_key="sk-test", audio_sink_path=None)
    # No _ws yet — cancel should no-op and return False.
    assert sess.cancel_response() is False


def test_realtime_session_cancel_response_sends_cancel_frame():
    from plugins.google_meet.realtime.openai_client import RealtimeSession

    sess = RealtimeSession(api_key="sk-test", audio_sink_path=None)
    sent = []

    class _FakeWs:
        def send(self, msg): sent.append(msg)

    sess._ws = _FakeWs()
    assert sess.cancel_response() is True
    assert len(sent) == 1
    import json as _j
    envelope = _j.loads(sent[0])
    assert envelope == {"type": "response.cancel"}


def test_realtime_session_counters_initialized():
    from plugins.google_meet.realtime.openai_client import RealtimeSession

    sess = RealtimeSession(api_key="sk-test", audio_sink_path=None)
    assert sess.audio_bytes_out == 0
    assert sess.last_audio_out_at is None


def test_start_realtime_speaker_linux_streams_pcm_to_paplay_stdin(tmp_path):
    from plugins.google_meet.meet_bot import (
        SAY_PCM_FILENAME,
        _BotState,
        _start_realtime_speaker,
    )

    popen_calls = []

    class _FakeSession:
        def __init__(self, **_kwargs):
            pass

        def connect(self):
            return None

    class _FakeSpeaker:
        def __init__(self, **_kwargs):
            pass

        def run_until_stopped(self, _stop_fn):
            return None

    class _FakeProc:
        def __init__(self, argv, **kwargs):
            self.argv = list(argv)
            self.kwargs = kwargs
            self.stdin = io.BytesIO()
            popen_calls.append(self)

        def poll(self):
            return None

    state = _BotState(
        out_dir=tmp_path,
        meeting_id="abc-defg-hij",
        url="https://meet.google.com/abc-defg-hij",
    )
    rt = {}
    stop_flag = {"stop": True}

    with patch("plugins.google_meet.realtime.openai_client.RealtimeSession", _FakeSession), \
         patch("plugins.google_meet.realtime.openai_client.RealtimeSpeaker", _FakeSpeaker), \
         patch("subprocess.Popen", _FakeProc):
        _start_realtime_speaker(
            rt=rt,
            out_dir=tmp_path,
            bridge_info={"platform": "linux", "write_target": "hermes_meet_sink"},
            api_key="sk-test",
            model="gpt-test",
            voice="alloy",
            instructions="Test",
            stop_flag=stop_flag,
            state=state,
        )

    assert popen_calls
    proc = popen_calls[0]
    assert str(tmp_path / SAY_PCM_FILENAME) not in proc.argv
    assert proc.kwargs["stdin"] == subprocess.PIPE
    assert "pcm_pump_thread" in rt


def test_start_realtime_speaker_darwin_streams_pcm_to_ffmpeg_stdin(tmp_path):
    from plugins.google_meet.meet_bot import (
        SAY_PCM_FILENAME,
        _BotState,
        _start_realtime_speaker,
    )

    popen_calls = []

    class _FakeSession:
        def __init__(self, **_kwargs):
            pass

        def connect(self):
            return None

    class _FakeSpeaker:
        def __init__(self, **_kwargs):
            pass

        def run_until_stopped(self, _stop_fn):
            return None

    class _FakeProc:
        def __init__(self, argv, **kwargs):
            self.argv = list(argv)
            self.kwargs = kwargs
            self.stdin = io.BytesIO()
            popen_calls.append(self)

        def poll(self):
            return None

    state = _BotState(
        out_dir=tmp_path,
        meeting_id="abc-defg-hij",
        url="https://meet.google.com/abc-defg-hij",
    )
    rt = {}
    stop_flag = {"stop": True}

    with patch("plugins.google_meet.realtime.openai_client.RealtimeSession", _FakeSession), \
         patch("plugins.google_meet.realtime.openai_client.RealtimeSpeaker", _FakeSpeaker), \
         patch("shutil.which", return_value="/usr/local/bin/ffmpeg"), \
         patch("plugins.google_meet.meet_bot._mac_audio_device_index", return_value="7"), \
         patch("subprocess.Popen", _FakeProc):
        _start_realtime_speaker(
            rt=rt,
            out_dir=tmp_path,
            bridge_info={"platform": "darwin", "write_target": "BlackHole 2ch"},
            api_key="sk-test",
            model="gpt-test",
            voice="alloy",
            instructions="Test",
            stop_flag=stop_flag,
            state=state,
        )

    assert popen_calls
    proc = popen_calls[0]
    assert str(tmp_path / SAY_PCM_FILENAME) not in proc.argv
    assert "pipe:0" in proc.argv
    assert proc.kwargs["stdin"] == subprocess.PIPE
    assert "pcm_pump_thread" in rt


def test_start_realtime_speaker_darwin_fails_when_audio_device_is_missing(tmp_path):
    from plugins.google_meet.meet_bot import _BotState, _start_realtime_speaker

    class _FakeSession:
        def __init__(self, **_kwargs):
            pass

        def connect(self):
            return None

    class _FakeSpeaker:
        def __init__(self, **_kwargs):
            pass

        def run_until_stopped(self, _stop_fn):
            return None

    state = _BotState(
        out_dir=tmp_path,
        meeting_id="abc-defg-hij",
        url="https://meet.google.com/abc-defg-hij",
    )
    rt = {}

    with patch("plugins.google_meet.realtime.openai_client.RealtimeSession", _FakeSession), \
         patch("plugins.google_meet.realtime.openai_client.RealtimeSpeaker", _FakeSpeaker), \
         patch("shutil.which", return_value="/usr/local/bin/ffmpeg"), \
         patch("plugins.google_meet.meet_bot._mac_audio_device_index", return_value=None), \
         patch("subprocess.Popen") as popen:
        _start_realtime_speaker(
            rt=rt,
            out_dir=tmp_path,
            bridge_info={"platform": "darwin", "write_target": "BlackHole 2ch"},
            api_key="sk-test",
            model="gpt-test",
            voice="alloy",
            instructions="Test",
            stop_flag={"stop": True},
            state=state,
        )

    popen.assert_not_called()
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["realtimeReady"] is False
    assert status["realtimeAudioPumpStatus"] == "missing_tool"
    assert "audio device not found" in status["realtimeAudioPumpError"]


def test_run_bot_tears_down_realtime_resources_on_navigation_failure(tmp_path, monkeypatch):
    import sys
    import types

    from plugins.google_meet import audio_bridge
    from plugins.google_meet.meet_bot import run_bot

    closed = {"context": False, "browser": False, "bridge": False}

    class _FakeBridge:
        def setup(self):
            return {
                "platform": "linux",
                "device_name": "hermes_meet_src",
                "write_target": "hermes_meet_sink",
            }

        def teardown(self):
            closed["bridge"] = True

    class _FakePage:
        def goto(self, *_args, **_kwargs):
            raise RuntimeError("navigation failed")

    class _FakeContext:
        def new_page(self):
            return _FakePage()

        def close(self):
            closed["context"] = True

    class _FakeBrowser:
        def new_context(self, **_kwargs):
            return _FakeContext()

        def close(self):
            closed["browser"] = True

    class _FakePlaywright:
        chromium = type("_Chromium", (), {
            "launch": staticmethod(lambda **_kwargs: _FakeBrowser())
        })()

    class _FakeSyncPlaywright:
        def __enter__(self):
            return _FakePlaywright()

        def __exit__(self, *_args):
            return False

    monkeypatch.setitem(
        sys.modules,
        "playwright.sync_api",
        types.SimpleNamespace(sync_playwright=lambda: _FakeSyncPlaywright()),
    )
    monkeypatch.setattr(audio_bridge, "AudioBridge", _FakeBridge)
    monkeypatch.setenv("HERMES_MEET_URL", "https://meet.google.com/abc-defg-hij")
    monkeypatch.setenv("HERMES_MEET_OUT_DIR", str(tmp_path))
    monkeypatch.setenv("HERMES_MEET_MODE", "realtime")
    monkeypatch.setenv("HERMES_MEET_REALTIME_KEY", "sk-test")

    assert run_bot() == 4
    assert closed == {"context": True, "browser": True, "bridge": True}


def test_run_bot_tears_down_partial_realtime_bridge_on_setup_failure(tmp_path, monkeypatch):
    from plugins.google_meet import audio_bridge
    from plugins.google_meet.meet_bot import run_bot

    closed = {"bridge": False}

    class _FakeBridge:
        def setup(self):
            raise RuntimeError("virtual device missing")

        def teardown(self):
            closed["bridge"] = True

    monkeypatch.setattr(audio_bridge, "AudioBridge", _FakeBridge)
    monkeypatch.setenv("HERMES_MEET_URL", "https://meet.google.com/abc-defg-hij")
    monkeypatch.setenv("HERMES_MEET_OUT_DIR", str(tmp_path))
    monkeypatch.setenv("HERMES_MEET_MODE", "realtime")
    monkeypatch.setenv("HERMES_MEET_REALTIME_KEY", "sk-test")

    assert run_bot() == 6
    assert closed["bridge"] is True


# ---------------------------------------------------------------------------
# hermes meet install CLI
# ---------------------------------------------------------------------------

def test_cli_install_subcommand_is_registered():
    import argparse
    from plugins.google_meet.cli import register_cli

    parser = argparse.ArgumentParser(prog="hermes meet")
    register_cli(parser)

    ns = parser.parse_args(["install"])
    assert ns.meet_command == "install"
    assert ns.realtime is False
    assert ns.yes is False


def test_cli_install_flags_parse():
    import argparse
    from plugins.google_meet.cli import register_cli

    parser = argparse.ArgumentParser(prog="hermes meet")
    register_cli(parser)

    ns = parser.parse_args(["install", "--realtime", "--yes"])
    assert ns.realtime is True
    assert ns.yes is True


def test_cmd_install_refuses_windows(capsys):
    from plugins.google_meet.cli import _cmd_install

    with patch("plugins.google_meet.cli.platform" if False else "platform.system",
               return_value="Windows"):
        rc = _cmd_install(realtime=False, assume_yes=True)
    assert rc == 1
    out = capsys.readouterr().out
    assert "Windows" in out


def test_cmd_install_runs_pip_and_playwright(capsys):
    """End-to-end wiring: pip + playwright install invoked, returncodes handled."""
    from plugins.google_meet.cli import _cmd_install

    calls = []
    class _FakeRes:
        def __init__(self, rc=0): self.returncode = rc

    def _fake_run(argv, **kwargs):
        calls.append(list(argv))
        return _FakeRes(0)

    with patch("platform.system", return_value="Linux"), \
         patch("subprocess.run", side_effect=_fake_run), \
         patch("shutil.which", return_value="/usr/bin/paplay"):
        rc = _cmd_install(realtime=False, assume_yes=True)
    assert rc == 0
    # First invocation: dependency install via the uv→pip ladder
    # (shutil.which is mocked truthy, so the uv tier is taken: `<uv> pip install ...`)
    pip_cmds = [
        c for c in calls
        if "install" in c and "playwright" in c and "websockets" in c
    ]
    assert pip_cmds, f"no dependency install run: {calls}"
    # Second: playwright install chromium
    pw_cmds = [c for c in calls if len(c) > 2 and c[1:4] == ["-m", "playwright", "install"]]
    assert pw_cmds, f"no playwright install run: {calls}"
    assert "chromium" in pw_cmds[0]


def test_cmd_install_realtime_skips_when_deps_present(capsys):
    """When paplay + pactl are already on PATH, no sudo call happens."""
    from plugins.google_meet.cli import _cmd_install

    calls = []
    class _FakeRes:
        def __init__(self, rc=0): self.returncode = rc

    def _fake_run(argv, **kwargs):
        calls.append(list(argv))
        return _FakeRes(0)

    with patch("platform.system", return_value="Linux"), \
         patch("subprocess.run", side_effect=_fake_run), \
         patch("shutil.which", return_value="/usr/bin/paplay"):
        rc = _cmd_install(realtime=True, assume_yes=True)
    assert rc == 0
    # No sudo apt-get call — paplay was already on PATH.
    sudo_calls = [c for c in calls if c and c[0] == "sudo"]
    assert sudo_calls == [], f"unexpected sudo invocation: {sudo_calls}"
    out = capsys.readouterr().out
    assert "already installed" in out


def test_meet_proxy_env_pins_media_routing_args(monkeypatch):
    from plugins.google_meet.meet_bot import _apply_meet_proxy_args

    args = []
    monkeypatch.setenv("HERMES_MEET_PROXY_SERVER", "http://proxy.example:8080")
    monkeypatch.delenv("HERMES_MEET_PROXY_BYPASS", raising=False)

    _apply_meet_proxy_args(args)

    assert args == [
        "--proxy-server=http://proxy.example:8080",
        "--proxy-bypass-list=74.125.250.0/24,74.125.247.128,142.250.82.0/24",
        "--force-webrtc-ip-handling-policy=disable_non_proxied_udp",
    ]


def test_meet_proxy_bypass_can_be_disabled_without_disabling_webrtc_policy(monkeypatch):
    from plugins.google_meet.meet_bot import _apply_meet_proxy_args

    args = []
    monkeypatch.setenv("HERMES_MEET_PROXY_SERVER", "http://proxy.example:8080")
    monkeypatch.setenv("HERMES_MEET_PROXY_BYPASS", "")

    _apply_meet_proxy_args(args)

    assert args == [
        "--proxy-server=http://proxy.example:8080",
        "--force-webrtc-ip-handling-policy=disable_non_proxied_udp",
    ]


def test_transcribe_browser_config_uses_fake_device_and_fake_ui_flags_only(monkeypatch):
    from plugins.google_meet.meet_bot import _build_browser_launch_config

    monkeypatch.delenv("HERMES_MEET_PROXY_SERVER", raising=False)

    chrome_args, permissions = _build_browser_launch_config(realtime_enabled=False)

    assert "--use-fake-device-for-media-stream" in chrome_args
    assert "--use-fake-ui-for-media-stream" in chrome_args
    assert not any(arg.startswith("--use-file-for-fake-video-capture=") for arg in chrome_args)
    assert not any(arg.startswith("--use-file-for-fake-audio-capture=") for arg in chrome_args)
    assert chrome_args.count("--use-fake-ui-for-media-stream") == 1
    assert permissions == ["microphone", "camera"]


def test_realtime_browser_config_uses_real_audio_input(monkeypatch):
    from plugins.google_meet.meet_bot import _build_browser_launch_config

    monkeypatch.delenv("HERMES_MEET_PROXY_SERVER", raising=False)

    chrome_args, permissions = _build_browser_launch_config(realtime_enabled=True)

    assert "--use-fake-ui-for-media-stream" in chrome_args
    assert "--use-fake-device-for-media-stream" not in chrome_args
    assert permissions == ["microphone", "camera"]


def test_enable_captions_clicks_visible_turn_on_captions_button_before_keyboard():
    from plugins.google_meet.meet_bot import _enable_captions

    clicked = []
    pressed = []

    class _Button:
        @property
        def first(self):
            return self

        def count(self):
            return 1

        def is_visible(self):
            return True

        def click(self, **_kwargs):
            clicked.append("captions")

    class _Keyboard:
        def press(self, key):
            pressed.append(key)

    class _Page:
        keyboard = _Keyboard()

        def get_by_role(self, role, *, name, **_kwargs):
            assert role == "button"
            assert name.search("Turn on captions (c)")
            return _Button()

    assert _enable_captions(_Page()) is True
    assert clicked == ["captions"]
    assert pressed == []


def test_enable_captions_reveals_hover_tray_before_keyboard_shortcut():
    from plugins.google_meet.meet_bot import _enable_captions

    events = []

    class _Button:
        def __init__(self, visible):
            self._visible = visible

        @property
        def first(self):
            return self

        def count(self):
            return 1

        def is_visible(self):
            return self._visible()

        def click(self, **_kwargs):
            events.append("click:captions")

    class _Keyboard:
        def __init__(self, page):
            self.page = page

        def press(self, key):
            events.append(f"press:{key}")
            if key == "ArrowDown":
                self.page.tray_open = True

    class _Page:
        def __init__(self):
            self.tray_open = False
            self.keyboard = _Keyboard(self)

        def wait_for_timeout(self, ms):
            events.append(f"wait:{ms}")

        def get_by_role(self, role, *, name, **_kwargs):
            assert role == "button"
            if name.search("Turn on captions"):
                return _Button(lambda: self.tray_open)
            return _Button(lambda: False)

        def locator(self, _selector):
            return _Button(lambda: False)

    assert _enable_captions(_Page()) is True
    assert events == ["press:ArrowDown", "wait:250", "click:captions"]


def test_enable_captions_uses_playwright_keyboard_press():
    from plugins.google_meet.meet_bot import _enable_captions

    pressed = []
    evaluated = []

    class _Keyboard:
        def press(self, key):
            pressed.append(key)

    class _Page:
        keyboard = _Keyboard()

        def evaluate(self, script):
            evaluated.append(script)

    assert _enable_captions(_Page()) is True
    assert pressed == ["ArrowDown", "c"]
    assert evaluated == []


def test_enable_captions_treats_turn_off_captions_as_already_enabled():
    from plugins.google_meet.meet_bot import _enable_captions

    pressed = []

    class _Button:
        def __init__(self, visible):
            self.visible = visible

        @property
        def first(self):
            return self

        def count(self):
            return 1 if self.visible else 0

        def is_visible(self):
            return self.visible

        def click(self, **_kwargs):
            raise AssertionError("already-enabled captions must not be clicked")

    class _Keyboard:
        def press(self, key):
            pressed.append(key)

    class _Page:
        keyboard = _Keyboard()

        def get_by_role(self, _role, *, name, **_kwargs):
            if name.search("Turn off captions"):
                return _Button(True)
            return _Button(False)

        def locator(self, selector):
            return _Button("Turn off captions" in selector)

    assert _enable_captions(_Page()) is True
    assert pressed == []


def test_enable_captions_can_skip_shortcut_when_controls_are_missing():
    from plugins.google_meet.meet_bot import _enable_captions

    pressed = []
    evaluated = []

    class _Keyboard:
        def press(self, key):
            pressed.append(key)

    class _Page:
        keyboard = _Keyboard()

        def evaluate(self, script):
            evaluated.append(script)

    assert _enable_captions(_Page(), allow_shortcut=False) is False
    assert pressed == []
    assert evaluated == []


def test_retry_caption_enable_allows_shortcut_after_admission(tmp_path):
    from plugins.google_meet.meet_bot import _BotState, _retry_caption_enable

    pressed = []

    class _Keyboard:
        def __init__(self, page):
            self.page = page

        def press(self, key):
            pressed.append(key)
            if key == "c":
                self.page.enabled = True

    class _Page:
        def __init__(self):
            self.enabled = False
            self.keyboard = _Keyboard(self)

        def get_by_role(self, _role, *, name, **_kwargs):
            return _Button(self.enabled and name.search("Turn off captions"))

        def locator(self, selector):
            return _Button(self.enabled and "Turn off captions" in selector)

    class _Button:
        def __init__(self, visible):
            self.visible = bool(visible)

        @property
        def first(self):
            return self

        def count(self):
            return 1 if self.visible else 0

        def is_visible(self):
            return self.visible

    state = _BotState(
        out_dir=tmp_path / "meet",
        meeting_id="abc-defg-hij",
        url="https://meet.google.com/abc-defg-hij",
    )
    state.set(in_call=True, joined_at=100.0)

    assert _retry_caption_enable(_Page(), state) is True
    assert pressed == ["ArrowDown", "c"]

    status = json.loads((tmp_path / "meet" / "status.json").read_text())
    assert status["captioning"] is True
    assert status["captionsEnabledAttempted"] is True


def test_retry_caption_enable_does_not_report_captioning_without_verification(tmp_path):
    from plugins.google_meet.meet_bot import _BotState, _retry_caption_enable

    pressed = []

    class _Keyboard:
        def press(self, key):
            pressed.append(key)

    class _MissingButton:
        @property
        def first(self):
            return self

        def count(self):
            return 0

        def is_visible(self):
            return False

    class _Page:
        keyboard = _Keyboard()

        def get_by_role(self, *_args, **_kwargs):
            return _MissingButton()

        def locator(self, *_args, **_kwargs):
            return _MissingButton()

    state = _BotState(
        out_dir=tmp_path / "meet",
        meeting_id="abc-defg-hij",
        url="https://meet.google.com/abc-defg-hij",
    )
    state.set(in_call=True, joined_at=100.0)

    assert _retry_caption_enable(_Page(), state) is False
    assert pressed == ["ArrowDown", "c"]

    status = json.loads((tmp_path / "meet" / "status.json").read_text())
    assert status["captioning"] is False
    assert status["captionsEnabledAttempted"] is True


def test_retry_caption_enable_does_not_report_captioning_without_success(tmp_path):
    from plugins.google_meet.meet_bot import _BotState, _retry_caption_enable

    pressed = []

    class _Keyboard:
        def press(self, key):
            pressed.append(key)

    class _Page:
        keyboard = _Keyboard()

    state = _BotState(
        out_dir=tmp_path / "meet",
        meeting_id="abc-defg-hij",
        url="https://meet.google.com/abc-defg-hij",
    )
    state.set(join_attempted_at=100.0)

    assert _retry_caption_enable(_Page(), state) is False
    assert pressed == []

    status = json.loads((tmp_path / "meet" / "status.json").read_text())
    assert status["captioning"] is False
    assert status["captionsEnabledAttempted"] is False


def test_retry_caption_enable_tries_js_shortcut_after_unverified_keyboard(tmp_path):
    from plugins.google_meet.meet_bot import _BotState, _retry_caption_enable

    pressed = []
    evaluated = []

    class _Keyboard:
        def press(self, key):
            pressed.append(key)

    class _Page:
        def __init__(self):
            self.enabled = False
            self.keyboard = _Keyboard()

        def get_by_role(self, _role, *, name, **_kwargs):
            return _Button(self.enabled and name.search("Turn off captions"))

        def locator(self, selector):
            return _Button(self.enabled and "Turn off captions" in selector)

        def wait_for_timeout(self, _ms):
            pass

        def evaluate(self, script):
            evaluated.append(script)
            self.enabled = True

    class _Button:
        def __init__(self, visible):
            self.visible = bool(visible)

        @property
        def first(self):
            return self

        def count(self):
            return 1 if self.visible else 0

        def is_visible(self):
            return self.visible

    state = _BotState(
        out_dir=tmp_path / "meet",
        meeting_id="abc-defg-hij",
        url="https://meet.google.com/abc-defg-hij",
    )
    state.set(in_call=True, joined_at=100.0)

    assert _retry_caption_enable(_Page(), state) is True
    assert pressed == ["ArrowDown", "c"]
    assert evaluated

    status = json.loads((tmp_path / "meet" / "status.json").read_text())
    assert status["captioning"] is True
    assert status["captionsEnabledAttempted"] is True


def test_click_join_returns_false_when_join_button_not_ready(tmp_path):
    from plugins.google_meet.meet_bot import _BotState, _click_join

    class _MissingButton:
        @property
        def first(self):
            return self

        def count(self):
            return 0

        def is_visible(self):
            return False

    class _Page:
        def get_by_role(self, *_args, **_kwargs):
            return _MissingButton()

    state = _BotState(
        out_dir=tmp_path / "meet",
        meeting_id="abc-defg-hij",
        url="https://meet.google.com/abc-defg-hij",
    )

    assert _click_join(_Page(), state) is False
    assert state.lobby_waiting is False


def test_click_join_returns_true_and_marks_lobby_for_ask_to_join(tmp_path):
    from plugins.google_meet.meet_bot import _BotState, _click_join

    clicked = []

    class _Button:
        def __init__(self, label):
            self.label = label

        @property
        def first(self):
            return self

        def count(self):
            return 1 if self.label == "Ask to join" else 0

        def is_visible(self):
            return self.label == "Ask to join"

        def click(self, **_kwargs):
            clicked.append(self.label)

    class _Page:
        def get_by_role(self, _role, *, name, **_kwargs):
            return _Button(name)

    state = _BotState(
        out_dir=tmp_path / "meet",
        meeting_id="abc-defg-hij",
        url="https://meet.google.com/abc-defg-hij",
    )

    assert _click_join(_Page(), state) is True
    assert clicked == ["Ask to join"]
    assert state.lobby_waiting is True


def test_click_join_falls_back_to_visible_text_for_ask_to_join(tmp_path):
    from plugins.google_meet.meet_bot import _BotState, _click_join

    clicked = []

    class _EmptyLocator:
        @property
        def first(self):
            return self

        def count(self):
            return 0

        def is_visible(self):
            return False

    class _TextLocator:
        @property
        def first(self):
            return self

        def count(self):
            return 1

        def is_visible(self):
            return True

        def click(self, **_kwargs):
            clicked.append("Ask to join")

    class _Page:
        def get_by_role(self, *_args, **_kwargs):
            return _EmptyLocator()

        def locator(self, selector):
            if "Ask to join" in selector:
                return _TextLocator()
            return _EmptyLocator()

    state = _BotState(
        out_dir=tmp_path / "meet",
        meeting_id="abc-defg-hij",
        url="https://meet.google.com/abc-defg-hij",
    )

    assert _click_join(_Page(), state) is True
    assert clicked == ["Ask to join"]
    assert state.lobby_waiting is True


def test_try_guest_name_falls_back_to_placeholder_name_input():
    from plugins.google_meet.meet_bot import _try_guest_name

    filled = []

    class _EmptyLocator:
        @property
        def first(self):
            return self

        def count(self):
            return 0

        def is_visible(self):
            return False

    class _InputLocator:
        @property
        def first(self):
            return self

        def count(self):
            return 1

        def is_visible(self):
            return True

        def fill(self, value, **_kwargs):
            filled.append(value)

    class _Page:
        def locator(self, selector):
            if selector == 'input[placeholder*="name" i]':
                return _InputLocator()
            return _EmptyLocator()

    _try_guest_name(_Page(), "Catchline Assistant")

    assert filled == ["Catchline Assistant"]


def test_click_join_falls_back_to_exact_text_control_for_join_now(tmp_path):
    from plugins.google_meet.meet_bot import _BotState, _click_join

    clicked = []

    class _EmptyLocator:
        @property
        def first(self):
            return self

        def count(self):
            return 0

        def is_visible(self):
            return False

    class _TextLocator:
        @property
        def first(self):
            return self

        def count(self):
            return 1

        def is_visible(self):
            return True

        def click(self, **_kwargs):
            clicked.append("Join now")

    class _Page:
        def get_by_role(self, *_args, **_kwargs):
            return _EmptyLocator()

        def locator(self, _selector):
            return _EmptyLocator()

        def get_by_text(self, text, **_kwargs):
            if text == "Join now":
                return _TextLocator()
            return _EmptyLocator()

    state = _BotState(
        out_dir=tmp_path / "meet",
        meeting_id="abc-defg-hij",
        url="https://meet.google.com/abc-defg-hij",
    )

    assert _click_join(_Page(), state) is True
    assert clicked == ["Join now"]
    assert state.lobby_waiting is False


def test_click_join_handles_continue_without_media_gate_before_join_now(tmp_path):
    from plugins.google_meet.meet_bot import _BotState, _click_join

    clicked = []

    class _Button:
        def __init__(self, label):
            self.label = label

        @property
        def first(self):
            return self

        def count(self):
            return 1 if self.label in {"Continue without microphone and camera", "Join now"} else 0

        def is_visible(self):
            return self.label in {"Continue without microphone and camera", "Join now"}

        def click(self, **_kwargs):
            clicked.append(self.label)

    class _Page:
        def get_by_role(self, _role, *, name, **_kwargs):
            return _Button(name)

        def wait_for_timeout(self, _timeout):
            clicked.append("wait")

    state = _BotState(
        out_dir=tmp_path / "meet",
        meeting_id="abc-defg-hij",
        url="https://meet.google.com/abc-defg-hij",
    )

    assert _click_join(_Page(), state) is True
    assert clicked == ["Continue without microphone and camera", "wait", "Join now"]
    assert state.lobby_waiting is False


def test_click_join_clicks_first_visible_duplicate_join_now(tmp_path):
    from plugins.google_meet.meet_bot import _BotState, _click_join

    clicked = []

    class _Button:
        def __init__(self, label, visible):
            self.label = label
            self.visible = visible

        def is_visible(self):
            return self.visible

        def click(self, **_kwargs):
            clicked.append(self.label)

    class _Locator:
        def __init__(self, label):
            self.label = label
            self.buttons = (
                [_Button(f"{label}:hidden", False), _Button(f"{label}:visible", True)]
                if label == "Join now"
                else []
            )

        @property
        def first(self):
            return self.buttons[0] if self.buttons else _Button(self.label, False)

        def count(self):
            return len(self.buttons)

        def nth(self, index):
            return self.buttons[index]

    class _Page:
        def get_by_role(self, _role, *, name, **_kwargs):
            return _Locator(name)

    state = _BotState(
        out_dir=tmp_path / "meet",
        meeting_id="abc-defg-hij",
        url="https://meet.google.com/abc-defg-hij",
    )

    assert _click_join(_Page(), state) is True
    assert clicked == ["Join now:visible"]
    assert state.lobby_waiting is False


def test_disable_local_media_clicks_visible_turn_off_controls():
    from plugins.google_meet.meet_bot import _disable_local_media

    clicked = []

    class _Button:
        def __init__(self, label):
            self.label = label

        @property
        def first(self):
            return self

        def count(self):
            return 1 if self.label in {"Turn off microphone", "Turn off camera"} else 0

        def is_visible(self):
            return self.count() == 1

        def click(self, **_kwargs):
            clicked.append(self.label)

    class _Page:
        def get_by_role(self, _role, *, name, **_kwargs):
            if name.search("Turn off microphone"):
                return _Button("Turn off microphone")
            if name.search("Turn off camera"):
                return _Button("Turn off camera")
            return _Button("missing")

    assert _disable_local_media(_Page()) == 2
    assert clicked == ["Turn off microphone", "Turn off camera"]


def test_disable_local_media_can_preserve_microphone_for_realtime():
    from plugins.google_meet.meet_bot import _disable_local_media

    clicked = []

    class _Button:
        def __init__(self, label):
            self.label = label

        @property
        def first(self):
            return self

        def count(self):
            return 1 if self.label in {"Turn off microphone", "Turn off camera"} else 0

        def is_visible(self):
            return self.count() == 1

        def click(self, **_kwargs):
            clicked.append(self.label)

    class _Page:
        def get_by_role(self, _role, *, name, **_kwargs):
            if name.search("Turn off microphone"):
                return _Button("Turn off microphone")
            if name.search("Turn off camera"):
                return _Button("Turn off camera")
            return _Button("missing")

    assert _disable_local_media(_Page(), disable_microphone=False) == 1
    assert clicked == ["Turn off camera"]


def test_disable_local_media_clicks_first_visible_duplicate_control():
    from plugins.google_meet.meet_bot import _disable_local_media

    clicked = []

    class _Button:
        def __init__(self, label, visible):
            self.label = label
            self.visible = visible

        @property
        def first(self):
            return self

        def count(self):
            return 1

        def is_visible(self):
            return self.visible

        def click(self, **_kwargs):
            clicked.append(self.label)

    class _Locator:
        def __init__(self, label):
            self.buttons = [
                _Button(f"{label}:hidden", False),
                _Button(f"{label}:visible", True),
            ]

        @property
        def first(self):
            return self.buttons[0]

        def count(self):
            return len(self.buttons)

        def nth(self, index):
            return self.buttons[index]

    class _Page:
        def get_by_role(self, _role, *, name, **_kwargs):
            if name.search("Turn off microphone"):
                return _Locator("microphone")
            if name.search("Turn off camera"):
                return _Locator("camera")
            return _Locator("missing")

    assert _disable_local_media(_Page()) == 2
    assert clicked == ["microphone:visible", "camera:visible"]


def test_probe_local_media_state_reports_visible_control_state():
    from plugins.google_meet.meet_bot import _probe_local_media_state

    class _Button:
        def __init__(self, visible):
            self.visible = visible

        @property
        def first(self):
            return self

        def count(self):
            return 1 if self.visible else 0

        def is_visible(self):
            return self.visible

    class _Page:
        def get_by_role(self, _role, *, name, **_kwargs):
            pattern = name.pattern.lower()
            if "turn on" in pattern and "microphone" in pattern:
                return _Button(True)
            if "turn off" in pattern and "camera" in pattern:
                return _Button(True)
            return _Button(False)

    assert _probe_local_media_state(_Page()) == {
        "local_microphone_on": False,
        "local_camera_on": True,
    }


def test_probe_local_media_state_uses_live_meet_status_text():
    from plugins.google_meet.meet_bot import _probe_local_media_state

    class _MissingButton:
        @property
        def first(self):
            return self

        def count(self):
            return 0

        def is_visible(self):
            return False

    class _Page:
        def get_by_role(self, *_args, **_kwargs):
            return _MissingButton()

        def evaluate(self, _script):
            return {
                "localMicrophoneOn": True,
                "localCameraOn": False,
            }

    assert _probe_local_media_state(_Page()) == {
        "local_microphone_on": True,
        "local_camera_on": False,
    }


def test_ensure_local_media_before_join_fails_closed_for_unknown_transcribe_media(tmp_path, monkeypatch):
    from plugins.google_meet import meet_bot
    from plugins.google_meet.meet_bot import _BotState, _ensure_local_media_before_join

    state = _BotState(
        out_dir=tmp_path / "meet",
        meeting_id="abc-defg-hij",
        url="https://meet.google.com/abc-defg-hij",
    )
    calls = []

    monkeypatch.setattr(
        meet_bot,
        "_disable_local_media",
        lambda page, *, disable_microphone=True, disable_camera=True: calls.append(
            (disable_microphone, disable_camera)
        ) or 0,
    )
    monkeypatch.setattr(
        meet_bot,
        "_probe_local_media_state",
        lambda page: {"local_microphone_on": None, "local_camera_on": False},
    )

    assert _ensure_local_media_before_join(object(), state, realtime_enabled=False, attempts=2) is False
    assert calls == [(True, True), (True, True)]

    status = json.loads((tmp_path / "meet" / "status.json").read_text())
    assert status["phase"] == "exited"
    assert status["exited"] is True
    assert status["leaveReason"] == "unsafe_media_state"
    assert "local media state unsafe" in status["error"]


def test_ensure_local_media_before_join_allows_realtime_mic_but_requires_camera_off(tmp_path, monkeypatch):
    from plugins.google_meet import meet_bot
    from plugins.google_meet.meet_bot import _BotState, _ensure_local_media_before_join

    state = _BotState(
        out_dir=tmp_path / "meet",
        meeting_id="abc-defg-hij",
        url="https://meet.google.com/abc-defg-hij",
    )

    monkeypatch.setattr(meet_bot, "_disable_local_media", lambda *args, **kwargs: 0)
    monkeypatch.setattr(
        meet_bot,
        "_probe_local_media_state",
        lambda page: {"local_microphone_on": True, "local_camera_on": False},
    )

    assert _ensure_local_media_before_join(
        object(),
        state,
        realtime_enabled=True,
        realtime_route_ready=True,
        attempts=1,
    ) is True

    status = json.loads((tmp_path / "meet" / "status.json").read_text())
    assert status["localMicrophoneOn"] is True
    assert status["localCameraOn"] is False
    assert status["error"] is None


def test_ensure_local_media_before_join_realtime_requires_route_ready_before_mic_on(tmp_path, monkeypatch):
    from plugins.google_meet import meet_bot
    from plugins.google_meet.meet_bot import _BotState, _ensure_local_media_before_join

    state = _BotState(
        out_dir=tmp_path / "meet",
        meeting_id="abc-defg-hij",
        url="https://meet.google.com/abc-defg-hij",
    )

    monkeypatch.setattr(meet_bot, "_disable_local_media", lambda *args, **kwargs: 0)
    monkeypatch.setattr(meet_bot, "_enable_local_microphone", lambda *args, **kwargs: 0)
    monkeypatch.setattr(
        meet_bot,
        "_probe_local_media_state",
        lambda page: {"local_microphone_on": True, "local_camera_on": False},
    )

    assert _ensure_local_media_before_join(
        object(),
        state,
        realtime_enabled=True,
        realtime_route_ready=False,
        attempts=1,
    ) is False

    status = json.loads((tmp_path / "meet" / "status.json").read_text())
    assert status["phase"] == "exited"
    assert status["leaveReason"] == "unsafe_media_state"


def test_disable_local_media_uses_keyboard_shortcuts_when_controls_are_hidden():
    from plugins.google_meet.meet_bot import _disable_local_media

    pressed = []

    class _Keyboard:
        def press(self, key):
            pressed.append(key)

    class _MissingButton:
        @property
        def first(self):
            return self

        def count(self):
            return 0

        def is_visible(self):
            return False

    class _Page:
        keyboard = _Keyboard()

        def get_by_role(self, *_args, **_kwargs):
            return _MissingButton()

        def evaluate(self, script):
            if "localMicrophoneOn" in script:
                return {
                    "localMicrophoneOn": True,
                    "localCameraOn": True,
                }
            return False

        def wait_for_timeout(self, _ms):
            pass

    assert _disable_local_media(_Page()) == 2
    assert pressed == ["Control+D", "Control+E"]


def test_disable_local_media_ignores_already_off_controls():
    from plugins.google_meet.meet_bot import _disable_local_media

    clicked = []

    class _MissingButton:
        @property
        def first(self):
            return self

        def count(self):
            return 0

        def is_visible(self):
            return False

        def click(self, **_kwargs):
            clicked.append("unexpected")

    class _Page:
        def get_by_role(self, _role, *, name, **_kwargs):
            assert not name.search("Turn on microphone")
            assert not name.search("Turn on camera")
            return _MissingButton()

    assert _disable_local_media(_Page()) == 0
    assert clicked == []
