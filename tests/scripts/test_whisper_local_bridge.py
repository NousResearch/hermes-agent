"""Tests for the local Whisper bridge and API fallback orchestration."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def bridge():
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "whatsapp-bridge"
        / "whisper-local-bridge.py"
    )
    spec = importlib.util.spec_from_file_location("whisper_local_bridge", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_try_local_reads_whisper_output(tmp_path, monkeypatch, bridge):
    whisper_cli = tmp_path / "whisper-cli"
    model = tmp_path / "model.bin"
    input_path = tmp_path / "voice.wav"
    output_path = tmp_path / "transcript.txt"
    for path in (whisper_cli, model, input_path):
        path.write_bytes(b"data")
    monkeypatch.setattr(bridge, "WHISPER_CLI", str(whisper_cli))
    monkeypatch.setattr(bridge, "WHISPER_MODEL", str(model))

    def fake_run(command, **kwargs):
        output_path.write_text("local transcript\n", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(bridge.subprocess, "run", fake_run)
    ok, text, error = bridge._try_local(str(input_path), str(output_path))

    assert ok is True
    assert text == "local transcript"
    assert error == ""


def test_try_local_converts_non_wav_input(tmp_path, monkeypatch, bridge):
    whisper_cli = tmp_path / "whisper-cli"
    model = tmp_path / "model.bin"
    input_path = tmp_path / "voice.m4a"
    output_path = tmp_path / "transcript.txt"
    for path in (whisper_cli, model, input_path):
        path.write_bytes(b"data")
    monkeypatch.setattr(bridge, "WHISPER_CLI", str(whisper_cli))
    monkeypatch.setattr(bridge, "WHISPER_MODEL", str(model))
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        if command[0] == str(whisper_cli):
            output_path.write_text("converted transcript", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(bridge.subprocess, "run", fake_run)
    ok, text, error = bridge._try_local(str(input_path), str(output_path))

    assert ok is True
    assert text == "converted transcript"
    assert error == ""
    assert commands[0][0] == "ffmpeg"
    assert commands[1][0] == str(whisper_cli)


def test_main_falls_back_to_openai_and_writes_output(tmp_path, monkeypatch, bridge):
    input_path = tmp_path / "voice.wav"
    output_path = tmp_path / "transcript.txt"
    input_path.write_bytes(b"audio")
    records = []
    monkeypatch.setattr(bridge, "_try_local", lambda *args: (False, "", "local failed"))
    monkeypatch.setattr(
        bridge, "_try_openai", lambda *args: (True, "api transcript", "")
    )
    monkeypatch.setattr(bridge, "_log", records.append)

    result = bridge.main(["bridge", str(input_path), str(output_path)])

    assert result == 0
    assert output_path.read_text(encoding="utf-8") == "api transcript"
    assert [record["model"] for record in records] == ["local", "openai"]
    assert records[1]["fallback_reason"] == "local failed"


def test_main_returns_one_when_both_paths_fail(tmp_path, monkeypatch, bridge, capsys):
    input_path = tmp_path / "voice.wav"
    output_path = tmp_path / "transcript.txt"
    input_path.write_bytes(b"audio")
    monkeypatch.setattr(bridge, "_try_local", lambda *args: (False, "", "local failed"))
    monkeypatch.setattr(bridge, "_try_openai", lambda *args: (False, "", "api failed"))
    monkeypatch.setattr(bridge, "_log", lambda record: None)

    result = bridge.main(["bridge", str(input_path), str(output_path)])

    assert result == 1
    assert not output_path.exists()
    assert "local+openai both failed" in capsys.readouterr().err


def test_try_openai_rejects_empty_transcript(tmp_path, monkeypatch, bridge):
    input_path = tmp_path / "voice.wav"
    input_path.write_bytes(b"audio")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class FakeTranscriptions:
        @staticmethod
        def create(**kwargs):
            return "   "

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.audio = SimpleNamespace(transcriptions=FakeTranscriptions())

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)
    ok, text, error = bridge._try_openai(str(input_path))

    assert ok is False
    assert text == ""
    assert error == "OpenAI returned empty transcript"
