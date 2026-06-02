"""Regression tests for dashboard Meet-mode voice behavior."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = REPO_ROOT / "web"


def test_meet_wake_detection_requires_hey_rolly():
    if not shutil.which("node"):
        pytest.skip("node is not installed")
    if not (WEB_DIR / "node_modules" / "typescript").exists():
        pytest.skip("web TypeScript dependency is not installed")

    script = r"""
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const webDir = process.argv[1];
const ts = require(path.join(webDir, 'node_modules', 'typescript'));
const source = fs.readFileSync(path.join(webDir, 'src/lib/voiceMeet.ts'), 'utf8');
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 } }).outputText;
const module = { exports: {} };
vm.runInNewContext(compiled, { module, exports: module.exports, RegExp, String });
const { isRollyWakePhrase } = module.exports;
const cases = [
  ['Hey Rolly, summarize this', true],
  ['hey rowley can you hear me', true],
  ['Rolly, can you hear me?', false],
  ['Rollie should we ship?', false],
  ['Hey, can you hear me?', false],
  ['we should ask rolly later', false]
];
const failures = cases.filter(([text, expected]) => isRollyWakePhrase(text) !== expected);
if (failures.length) {
  console.error(JSON.stringify(failures));
  process.exit(1);
}
"""
    result = subprocess.run(["node", "-e", script, str(WEB_DIR)], text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr


def test_start_meeting_invite_starts_meet_call_path():
    source = (WEB_DIR / "src/pages/VoiceCallPage.tsx").read_text(encoding="utf-8")

    assert "const startMeetingInvite = useCallback" in source
    assert "await startCall(\"meet\", true)" in source
    assert "preserveCallId = false" in source
    assert "if (!preserveCallId && !new URLSearchParams(window.location.search).get(\"call_id\"))" in source
    assert "onClick={startMeetingInvite}" in source
    assert "onClick={() => void startCall()}" in source


def test_start_meeting_invite_mints_fresh_call_id_before_invite_and_blocks_double_tap():
    source = (WEB_DIR / "src/pages/VoiceCallPage.tsx").read_text(encoding="utf-8")
    start = source.index("const startMeetingInvite = useCallback")
    end = source.index("const toggleMute", start)
    body = source[start:end]

    assert "const [invitePending, setInvitePending] = useState(false)" in source
    assert "const busy = invitePending ||" in source
    assert body.index("setInvitePending(true)") < body.index("api.createVoiceMeetInvite")
    assert body.index("callIdRef.current = `voice-${Date.now()}-${Math.random().toString(16).slice(2)}`") < body.index("api.createVoiceMeetInvite")
    assert "setCallIdDisplay(callIdRef.current)" in body
    assert "setInvitePending(false)" in body


def test_meet_transcripts_use_active_call_mode_not_stale_react_state():
    source = (WEB_DIR / "src/pages/VoiceCallPage.tsx").read_text(encoding="utf-8")

    assert "const activeCallModeRef = useRef<\"solo\" | \"meet\">(\"solo\")" in source
    assert "activeCallModeRef.current = callMode" in source
    assert "const currentMode = activeCallModeRef.current" in source
    assert "currentMode === \"meet\"" in source
    assert "mode === \"meet\"\n              ? { mode" not in source


def test_start_call_owns_mic_permission_and_live_mic_switching():
    source = (WEB_DIR / "src/pages/VoiceCallPage.tsx").read_text(encoding="utf-8")

    assert "Enable mic list" not in source
    assert "onClick={enableMicList}" not in source
    assert "navigator.mediaDevices.getUserMedia({ audio })" in source
    assert "if (live) void switchMicrophone(next)" in source
    assert "disabled={busy}" in source
    assert "replaceTrack(nextTrack)" in source


def test_voice_page_filters_realtime_noise_until_verbose_and_uses_compact_rows():
    source = (WEB_DIR / "src/pages/VoiceCallPage.tsx").read_text(encoding="utf-8")

    assert "const [verboseEvents, setVerboseEvents] = useState(false)" in source
    assert "isRealtimeSpeechEvent" in source
    assert "verboseEvents || !isRealtimeSpeechEvent(entry)" in source
    assert "Verbose: {verboseEvents ? \"on\" : \"off\"}" in source
    assert "bg-background-base/50 px-2 py-1" in source
    assert "bg-background-base/50 p-3" not in source


@pytest.fixture()
def voice_client(monkeypatch, _isolate_hermes_home):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    import hermes_cli.web_server as web_server

    client = TestClient(web_server.app)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    return client, web_server


def test_voice_call_endpoint_passes_meet_mode(voice_client, monkeypatch):
    client, web_server = voice_client
    seen = {}

    def fake_create_call(sdp, user=None, mode="solo"):
        seen.update({"sdp": sdp, "user": user, "mode": mode})
        return "answer-sdp"

    monkeypatch.setattr(web_server, "_create_openai_realtime_call", fake_create_call)

    resp = client.post(
        "/api/voice/call?mode=meet",
        data="offer-sdp",
        headers={"Content-Type": "application/sdp", "X-Rolly-User": "deniz"},
    )

    assert resp.status_code == 200
    assert resp.text == "answer-sdp"
    assert seen == {"sdp": "offer-sdp", "user": "deniz", "mode": "meet"}
    assert resp.headers["content-type"].startswith("application/sdp")
