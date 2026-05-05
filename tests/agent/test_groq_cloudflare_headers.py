"""Regression guard: Groq Cloudflare 403/1010 mitigation headers."""
from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_groq_default_headers_use_hermes_user_agent():
    from agent.auxiliary_client import groq_default_headers

    headers = groq_default_headers()
    assert headers == {"User-Agent": headers["User-Agent"]}
    assert headers["User-Agent"].startswith("HermesAgent/")
    assert "openai-python" not in headers["User-Agent"].lower()


def test_primary_client_wires_groq_headers():
    from run_agent import AIAgent

    with patch("run_agent.OpenAI") as mock_openai:
        mock_openai.return_value = MagicMock()
        AIAgent(
            api_key="gsk-test",
            base_url="https://api.groq.com/openai/v1",
            provider="custom",
            model="llama-3.3-70b-versatile",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        headers = mock_openai.call_args.kwargs.get("default_headers") or {}
        assert headers.get("User-Agent", "").startswith("HermesAgent/")


def test_apply_client_headers_wires_groq_headers():
    from run_agent import AIAgent

    with patch("run_agent.OpenAI") as mock_openai:
        mock_openai.return_value = MagicMock()
        agent = AIAgent(
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
            provider="custom",
            model="gpt-4o-mini",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent._apply_client_headers_for_base_url("https://api.groq.com/openai/v1")
        headers = agent._client_kwargs.get("default_headers") or {}
        assert headers.get("User-Agent", "").startswith("HermesAgent/")


def test_transcription_groq_client_wires_headers(monkeypatch, tmp_path):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.setattr("tools.transcription_tools._HAS_OPENAI", True)
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"RIFFxxxxWAVEfmt ")

    with patch("openai.OpenAI") as mock_openai:
        client = MagicMock()
        client.audio.transcriptions.create.return_value = "hello"
        mock_openai.return_value = client

        from tools.transcription_tools import _transcribe_groq

        result = _transcribe_groq(str(audio_path), "whisper-large-v3-turbo")

    assert result["success"] is True
    headers = mock_openai.call_args.kwargs.get("default_headers") or {}
    assert headers.get("User-Agent", "").startswith("HermesAgent/")
