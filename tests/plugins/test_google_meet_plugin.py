"""Tests for the google_meet plugin.

Covers the safety-gated pieces that don't require Playwright:

  * URL regex — only ``https://meet.google.com/`` URLs pass
  * Meeting-id extraction from Meet URLs
  * Status / transcript writes round-trip through the file-backed state
  * Tool handlers return well-formed JSON under all branches
  * Process manager refuses unsafe URLs and clears stale state cleanly
  * ``_on_session_end`` hook is defensive (no-ops when no bot active)

Does NOT spawn a real Chromium — we mock ``subprocess.Popen`` where needed.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
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
    assert stripped == [
        f"Alex Rivera: {first_segment}",
        f"Alex Rivera: {second_segment} {third_segment}",
    ]
    assert all(len(line.split(": ", 1)[1]) <= 500 for line in stripped)

    status = json.loads((out / "status.json").read_text())
    assert status["transcriptLines"] == 2


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
    state.record_caption("", "text but no speaker")

    status = json.loads((tmp_path / "s" / "status.json").read_text())
    assert status["transcriptLines"] == 1
    # blank-speaker falls back to "Unknown"
    assert "Unknown: text but no speaker" in (tmp_path / "s" / "transcript.txt").read_text()


def test_bot_state_writes_caption_debug_for_unknown_speaker(tmp_path):
    from plugins.google_meet.meet_bot import _BotState

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

    debug_lines = (tmp_path / "s" / "caption_debug.jsonl").read_text().splitlines()
    assert len(debug_lines) == 1
    debug = json.loads(debug_lines[0])
    assert debug["speakerSource"] == "unresolved"
    assert debug["speakerDebug"] == speaker_debug


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


def test_caption_observer_emits_only_new_suffix_for_growing_visible_caption_row():
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
        ("Alex Rivera", "New words from the same row."),
    ]
    assert all(entry["speakerSource"] == "captionRow" for entry in entries)


def test_caption_observer_suppresses_initial_large_visible_caption_history():
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
        ("Alex Rivera", "Fresh words after reset."),
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
        "pid": 0, "meeting_id": "abc-defg-hij",
        "out_dir": str(meeting_dir),
        "url": "https://meet.google.com/abc-defg-hij",
        "started_at": 0,
    })

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


def test_meet_leave_no_active():
    from plugins.google_meet.tools import handle_meet_leave

    out = json.loads(handle_meet_leave({}))
    assert out["success"] is False


# ---------------------------------------------------------------------------
# _on_session_end — defensive cleanup
# ---------------------------------------------------------------------------

def test_on_session_end_noop_when_nothing_active():
    from plugins.google_meet import _on_session_end
    # Should not raise and should not call stop().
    with patch("plugins.google_meet.pm.stop") as stop_mock:
        _on_session_end()
    stop_mock.assert_not_called()


def test_on_session_end_stops_live_bot():
    from plugins.google_meet import _on_session_end
    from plugins.google_meet import pm

    with patch.object(pm, "status", return_value={"ok": True, "alive": True, "duration": None}), \
         patch.object(pm, "stop") as stop_mock:
        _on_session_end()
    stop_mock.assert_called_once_with(reason="session ended")


def test_on_session_end_keeps_duration_limited_bot_running():
    from plugins.google_meet import _on_session_end
    from plugins.google_meet import pm

    with patch.object(pm, "status", return_value={"ok": True, "alive": True, "duration": "20m"}), \
         patch.object(pm, "stop") as stop_mock:
        _on_session_end()
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
    assert calls["hooks"] == ["on_session_end"]


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


def test_enqueue_say_writes_jsonl_in_realtime_mode():
    from plugins.google_meet import process_manager as pm

    out_dir = Path(os.environ["HERMES_HOME"]) / "workspace" / "meetings" / "abc-defg-hij"
    out_dir.mkdir(parents=True)
    pm._write_active({
        "pid": 0, "meeting_id": "abc-defg-hij",
        "out_dir": str(out_dir), "url": "https://meet.google.com/abc-defg-hij",
        "started_at": 0, "mode": "realtime",
    })
    res = pm.enqueue_say("hello everyone")
    assert res["ok"] is True
    assert "enqueued_id" in res

    queue = out_dir / "say_queue.jsonl"
    assert queue.is_file()
    lines = [json.loads(ln) for ln in queue.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1
    assert lines[0]["text"] == "hello everyone"


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


def test_meet_join_passes_default_auth_state_when_saved():
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
    assert start_mock.call_args.kwargs["auth_state"] == str(auth_path)


def test_meet_join_defaults_duration_so_notetaking_survives_session_end():
    """meet_join without an explicit duration must still get a bounded duration.

    Otherwise session-end cleanup can reap the durationless bot immediately
    after the tool response.
    """
    from plugins.google_meet.tools import handle_meet_join

    with patch("plugins.google_meet.tools.check_meet_requirements", return_value=True), \
         patch("plugins.google_meet.tools.pm.start", return_value={"ok": True, "meeting_id": "x-y-z"}) as start_mock:
        out = json.loads(handle_meet_join({
            "url": "https://meet.google.com/abc-defg-hij",
        }))

    assert out["success"] is True
    assert start_mock.call_args.kwargs["duration"] == "120m"


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
        "audioBytesOut", "lastAudioOutAt", "lastBargeInAt",
        "joinAttemptedAt", "leaveReason",
        "phase", "lastHeartbeatAt", "lastProgressAt",
        "stalledReason", "lastUiText", "lastUrl",
    ):
        assert key in status, f"missing v2 telemetry key: {key}"
    assert status["realtime"] is False
    assert status["realtimeReady"] is False
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


def test_bot_state_heartbeat_flushes_phase_and_diagnostics(tmp_path):
    from plugins.google_meet.meet_bot import _BotState

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
    # First invocation: pip install
    pip_cmds = [c for c in calls if len(c) > 2 and c[1:4] == ["-m", "pip", "install"]]
    assert pip_cmds, f"no pip install run: {calls}"
    assert "playwright" in pip_cmds[0]
    assert "websockets" in pip_cmds[0]
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


def test_meet_proxy_env_adds_chromium_proxy_arg(monkeypatch):
    from plugins.google_meet.meet_bot import _apply_meet_proxy_args

    args = []
    monkeypatch.setenv("HERMES_MEET_PROXY_SERVER", "http://proxy.example:8080")

    _apply_meet_proxy_args(args)

    assert args == ["--proxy-server=http://proxy.example:8080"]


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


def test_realtime_browser_config_requests_microphone_with_fake_media(monkeypatch):
    from plugins.google_meet.meet_bot import _build_browser_launch_config

    monkeypatch.delenv("HERMES_MEET_PROXY_SERVER", raising=False)

    chrome_args, permissions = _build_browser_launch_config(realtime_enabled=True)

    assert "--use-fake-ui-for-media-stream" in chrome_args
    assert "--use-fake-device-for-media-stream" in chrome_args
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
    assert pressed == ["c"]
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
    assert pressed == ["c"]

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
    assert pressed == ["c"]

    status = json.loads((tmp_path / "meet" / "status.json").read_text())
    assert status["captioning"] is False
    assert status["captionsEnabledAttempted"] is False


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
    assert pressed == ["c"]
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
