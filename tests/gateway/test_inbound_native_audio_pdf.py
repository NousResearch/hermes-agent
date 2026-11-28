"""Contract tests for gateway inbound native audio & PDF routing and TurnRunner materialization."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from gateway.run_inbound import GatewayInboundMixin
from gateway.run_turn_runner import TurnRunner
from gateway.session_state import PersistentState

_TINY_PDF = b"%PDF-1.4\n1 0 obj << /Type /Catalog >> endobj\ntrailer << /Root 1 0 R >>\n%%EOF\n"


def test_gateway_inbound_native_audio_pdf_auto_and_spoof_rejection(tmp_path: Path):
    pdf_path = tmp_path / "invoice.pdf"
    pdf_path.write_bytes(_TINY_PDF)
    wav_path = tmp_path / "clip.wav"
    wav_path.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt ")
    spoof_audio = tmp_path / "fake.mp3"
    spoof_audio.write_text("<html>not audio</html>", encoding="utf-8")

    class _FakeRunner(GatewayInboundMixin):
        def __init__(self, provider: str, model: str):
            self._provider = provider
            self._model = model
            self._st = SimpleNamespace(persistent=PersistentState())

        def _resolve_session_agent_runtime(self, *, source, session_key):
            return self._model, {"provider": self._provider, "base_url": "", "requested_provider": self._provider}

        def _session_state(self, session_key):
            return self._st

        def _peek_session_state(self, session_key):
            return self._st

    event = SimpleNamespace(
        media_urls=[str(wav_path), str(pdf_path), str(spoof_audio)],
        media_types=["audio/x-wav", "application/pdf", "audio/mpeg"],
        media_filenames=["clip.wav", "invoice.pdf", "fake.mp3"],
    )
    # Default empty config {} activates native audio & PDF on solstice/gemini-4-neon
    with patch("hermes_cli.config.load_config", return_value={}):
        runner = _FakeRunner("solstice", "gemini-4-neon")
        rem_audio, native_pdfs = runner._route_inbound_native_audio_and_pdfs(
            event, None, "s1", [str(wav_path), str(spoof_audio)]
        )
        # Valid WAV is queued with normalized audio/wav; HTML-spoofed fake.mp3 stays on fallback
        assert rem_audio == [str(spoof_audio)]
        assert native_pdfs == {str(pdf_path)}
        queued = runner._consume_pending_native_media_attachments("s1")
        assert [(q["kind"], q["mime_type"]) for q in queued] == [("audio", "audio/wav"), ("document", "application/pdf")]
        assert runner._consume_pending_native_media_attachments("s1") == []

        # Non-native model leaves both audio_file_paths and PDFs on fallback path
        ds = _FakeRunner("deepseek", "deepseek-chat")
        rem_audio_ds, native_pdfs_ds = ds._route_inbound_native_audio_and_pdfs(
            event, None, "s2", [str(wav_path)]
        )
        assert rem_audio_ds == [str(wav_path)]
        assert native_pdfs_ds == set()

        # Anthropic user-turn PDFs and Gemini 2.5 user-turn audio+PDFs route natively
        ant = _FakeRunner("anthropic", "claude-sonnet-4-5")
        rem_audio_ant, native_pdfs_ant = ant._route_inbound_native_audio_and_pdfs(
            event, None, "s3", [str(wav_path)]
        )
        assert rem_audio_ant == [str(wav_path)]
        assert native_pdfs_ant == {str(pdf_path)}

        g25 = _FakeRunner("gemini", "gemini-2.5-pro")
        rem_audio_25, native_pdfs_25 = g25._route_inbound_native_audio_and_pdfs(
            event, None, "s4", [str(wav_path)]
        )
        assert rem_audio_25 == []
        assert native_pdfs_25 == {str(pdf_path)}


def test_turn_runner_native_media_materialization(tmp_path: Path):
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(_TINY_PDF)
    wav_path = tmp_path / "voice.wav"
    wav_path.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt ")

    fake_runner = SimpleNamespace(
        _consume_pending_native_image_paths=lambda sid: [],
        _consume_pending_native_media_attachments=lambda sid: [
            {"kind": "audio", "path": str(wav_path), "mime_type": "audio/x-wav", "filename": "voice.wav"},
            {"kind": "document", "path": str(pdf_path), "mime_type": "application/pdf", "filename": "doc.pdf"},
        ],
    )
    tr = object.__new__(TurnRunner)
    tr._runner = fake_runner
    tr._ctx = SimpleNamespace(session_key="s1", message="Check attachments")
    out = tr._native_image_run_message()
    assert isinstance(out, list) and len(out) == 3
    assert out[1]["type"] == "audio" and out[1]["mime_type"] == "audio/wav"
    assert out[2]["type"] == "document" and out[2]["filename"] == "doc.pdf"

    # Missing file at materialization time emits a fallback path note instead of silently vanishing
    missing_runner = SimpleNamespace(
        _consume_pending_native_image_paths=lambda sid: [],
        _consume_pending_native_media_attachments=lambda sid: [
            {"kind": "document", "path": str(tmp_path / "gone.pdf"), "mime_type": "application/pdf", "filename": "gone.pdf"},
        ],
    )
    tr._runner = missing_runner
    tr._ctx = SimpleNamespace(session_key="s1", message="Summarize")
    fb = tr._native_image_run_message()
    assert isinstance(fb, str) and "gone.pdf" in fb and "Summarize" in fb


def test_gateway_native_media_preserves_fallback_notice_api_mode_mime_and_dedupe(tmp_path: Path):
    from gateway.run_inbound_native_media import prepare_inbound_native_audio_and_pdfs

    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt ")
    blob_pdf = tmp_path / "cached_blob_no_ext"
    blob_pdf.write_bytes(_TINY_PDF)
    dup_pdf = tmp_path / "dup.pdf"
    dup_pdf.write_bytes(_TINY_PDF)

    calls: list[bool] = []

    class _NoticeRunner(GatewayInboundMixin):
        def __init__(self, provider: str, model: str, api_mode: str = "chat_completions"):
            self._provider = provider
            self._model = model
            self._api_mode = api_mode
            self._pre_agent_fallback_notice = "KEEP_ME"

        def _resolve_session_agent_runtime(self, *, source=None, session_key=None, record_notice: bool = True):
            calls.append(record_notice)
            if record_notice:
                self._pre_agent_fallback_notice = None
            return self._model, {
                "provider": self._provider,
                "base_url": "",
                "requested_provider": self._provider,
                "api_mode": self._api_mode,
            }

    # 1. Text-only message short-circuits without touching runtime resolution or _pre_agent_fallback_notice
    r_text = _NoticeRunner("solstice", "gemini-4-neon")
    empty_event = SimpleNamespace(media_urls=[], media_types=[], media_filenames=[])
    assert prepare_inbound_native_audio_and_pdfs(r_text, empty_event, None, "s-empty", []) == ([], set(), [])
    assert calls == []
    assert r_text._pre_agent_fallback_notice == "KEEP_ME"

    with patch("hermes_cli.config.load_config", return_value={"agent": {"audio_input_mode": "native", "pdf_input_mode": "native"}}):
        # 2. Candidate audio on codex_responses api_mode passes record_notice=False AND respects api_mode gate
        r_codex = _NoticeRunner("google", "gemini-2.5-flash", api_mode="codex_responses")
        evt_wav = SimpleNamespace(media_urls=[str(wav)], media_types=["audio/wav"], media_filenames=["a.wav"])
        rem, pdfs, queued = prepare_inbound_native_audio_and_pdfs(r_codex, evt_wav, None, "s-wav", [str(wav)])
        assert calls == [False]
        assert r_codex._pre_agent_fallback_notice == "KEEP_ME"
        assert rem == [str(wav)] and pdfs == set() and queued == []

        # 3. Parameterized PDF MIME ("application/pdf; charset=binary") on extensionless blob + duplicate URL dedupe
        r_sol = _NoticeRunner("solstice", "gemini-4-neon")
        evt_pdfs = SimpleNamespace(
            media_urls=[str(blob_pdf), str(dup_pdf), str(dup_pdf)],
            media_types=["application/pdf; charset=binary", "application/pdf", "application/pdf"],
            media_filenames=["report.pdf", "dup.pdf", "dup.pdf"],
        )
        rem2, pdfs2, queued2 = prepare_inbound_native_audio_and_pdfs(r_sol, evt_pdfs, None, "s-pdf", [])
        assert rem2 == []
        assert pdfs2 == {str(blob_pdf), str(dup_pdf)}
        assert len(queued2) == 2
        assert [q["path"] for q in queued2] == [str(blob_pdf), str(dup_pdf)]
