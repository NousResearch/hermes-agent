"""CLI, TUI and desktop speech paths fire ``transform_spoken_text`` (issue #152238).

``cli_voice`` covers the classic CLI's batch voice reply and the TUI's whole-file path;
``read_aloud`` covers the desktop speak-stream socket (per sentence) and its ``POST`` fallback.
The plugin dispatch is faked at ``hermes_cli.plugins``; the TTS tool / streamer are fakes.
"""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock
from urllib.parse import urlencode

import pytest

from hermes_cli.cli_voice_mixin import CLIVoiceMixin


def _fake_spoken_hook(monkeypatch, reply):
    calls = []

    def _invoke(hook_name, **kwargs):
        if hook_name != "transform_spoken_text":
            return []
        calls.append(kwargs)
        return [reply(kwargs)]

    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: name == "transform_spoken_text")
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _invoke)
    return calls


def _record_tts(monkeypatch):
    spoken = []

    def _tts(text, output_path=None, **_kwargs):
        spoken.append(text)
        return json.dumps({"success": False, "error": "not synthesized in tests"})

    monkeypatch.setattr("tools.tts_tool.text_to_speech_tool", _tts)
    monkeypatch.setattr("tools.voice_mode.play_audio_file", lambda _path: None)
    return spoken


# --------------------------------------------------------------------------- cli_voice


def _cli(session_id="sess-cli"):
    return SimpleNamespace(
        _voice_tts=True, _voice_tts_done=threading.Event(), _voice_last_tts_text="", session_id=session_id)


def test_cli_voice_reply_speaks_the_transformed_text(monkeypatch):
    spoken = _record_tts(monkeypatch)
    calls = _fake_spoken_hook(monkeypatch, lambda kw: f"[sighs] {kw['text']}")
    cli = _cli()

    CLIVoiceMixin._voice_speak_response(cli, "Fine, it passed.")

    assert spoken == ["[sighs] Fine, it passed."]
    assert cli._voice_last_tts_text == "[sighs] Fine, it passed."
    assert [(c["surface"], c["platform"], c["session_id"]) for c in calls] == [("cli_voice", "cli", "sess-cli")]


def test_cli_voice_empty_replacement_speaks_nothing(monkeypatch):
    spoken = _record_tts(monkeypatch)
    _fake_spoken_hook(monkeypatch, lambda kw: "")

    CLIVoiceMixin._voice_speak_response(_cli(), "Fine, it passed.")

    assert spoken == []


def test_cli_voice_survives_a_broken_hook_dispatch(monkeypatch):
    spoken = _record_tts(monkeypatch)
    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: True)
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", MagicMock(side_effect=RuntimeError("boom")))

    CLIVoiceMixin._voice_speak_response(_cli(), "Fine, it passed.")

    assert spoken == ["Fine, it passed."]


def test_tui_whole_file_reply_carries_the_tui_session(monkeypatch):
    from hermes_cli import voice

    spoken = _record_tts(monkeypatch)
    calls = _fake_spoken_hook(monkeypatch, lambda kw: f"[sighs] {kw['text']}")

    voice._speak_whole_file("Fine, it passed.", "sess-tui")

    assert spoken == ["[sighs] Fine, it passed."]
    assert [(c["surface"], c["platform"], c["session_id"]) for c in calls] == [("cli_voice", "tui", "sess-tui")]


# --------------------------------------------------------------------------- read_aloud


@pytest.fixture
def web_client(_isolate_hermes_home):
    from starlette.testclient import TestClient

    from hermes_cli import web_server

    previous = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.auth_required = False
    client = TestClient(web_server.app)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    try:
        yield client
    finally:
        client.close()
        if previous is None:
            delattr(web_server.app.state, "auth_required")
        else:
            web_server.app.state.auth_required = previous


class _Streamer:
    sample_rate, channels = 24000, 1

    def __init__(self):
        self.requests = []

    def stream(self, text):
        self.requests.append(text)
        yield b"\x01\x02"


def _speak_stream(client, monkeypatch, text):
    from hermes_cli import web_server

    streamer = _Streamer()
    monkeypatch.setattr("tools.tts_streaming.resolve_streaming_provider", lambda cfg: streamer)
    monkeypatch.setattr("tools.tts_tool._load_tts_config", lambda: {"streaming": {"min_len": 1}})
    monkeypatch.setattr("tools.tts_tool._get_provider", lambda cfg: "fake")
    monkeypatch.setattr("tools.tts_tool._resolve_max_text_length", lambda provider, cfg: 4000)
    url = f"/api/audio/speak-stream?{urlencode({'token': web_server._SESSION_TOKEN})}"
    with client.websocket_connect(url) as conn:
        conn.send_text(json.dumps({"text": text, "done": True}))
        while True:
            message = conn.receive()
            if message.get("text") and json.loads(message["text"]).get("type") in {"end", "fallback"}:
                break
    return streamer.requests


def test_read_aloud_stream_transforms_each_sentence(web_client, monkeypatch):
    calls = _fake_spoken_hook(monkeypatch, lambda kw: f"[{kw['index']}] {kw['text']}")

    requests = _speak_stream(web_client, monkeypatch, "First one here. Second one here. ")

    assert requests == ["[0] First one here.", "[1] Second one here."]
    assert {(c["surface"], c["platform"], c["session_id"]) for c in calls} == {("read_aloud", "desktop", "")}


def test_read_aloud_stream_skips_a_sentence_replaced_with_empty(web_client, monkeypatch):
    _fake_spoken_hook(monkeypatch, lambda kw: "" if kw["index"] == 0 else None)

    assert _speak_stream(web_client, monkeypatch, "First one here. Second one here. ") == ["Second one here."]


def test_read_aloud_fallback_speaks_the_transformed_reply(web_client, monkeypatch):
    spoken = _record_tts(monkeypatch)
    calls = _fake_spoken_hook(monkeypatch, lambda kw: f"[sighs] {kw['text']}")

    web_client.post("/api/audio/speak", json={"text": "Fine, it passed."})

    assert spoken == ["[sighs] Fine, it passed."]
    assert [(c["surface"], c["platform"]) for c in calls] == [("read_aloud", "desktop")]


def test_read_aloud_fallback_without_a_plugin_sends_the_raw_text(web_client, monkeypatch):
    spoken = _record_tts(monkeypatch)
    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: False)

    web_client.post("/api/audio/speak", json={"text": "**Fine**, it passed."})

    assert spoken == ["**Fine**, it passed."]
