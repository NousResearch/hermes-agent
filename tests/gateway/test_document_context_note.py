"""Tests for the document context note prepended to user turns with attachments.

A user who attaches a PDF / DOCX in chat used to see the agent treat it as
"unreadable" because the context note told the model to "Ask the user what
they'd like you to do with it" — steering it away from extracting the text it
is perfectly capable of reading. These tests pin the contract:

- text documents: note confirms the (adapter-)inlined content + records path.
- binary documents (PDF/DOCX/…): note tells the agent to extract the text
  itself and never tells it to punt back to the user.
"""

import importlib

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource

gateway_run = importlib.import_module("gateway.run")
_build_document_context_note = gateway_run._build_document_context_note


class TestTextDocumentNote:

    def test_non_inlined_text_note_tells_agent_to_read_cached_path(self):
        note = _build_document_context_note(
            "notes.txt",
            "/cache/doc_notes.txt",
            "text/plain",
            content_inlined=False,
        )
        assert "included below" not in note
        assert "/cache/doc_notes.txt" in note
        assert "read" in note.lower()

    @pytest.mark.asyncio
    async def test_event_contract_marks_non_inlined_text_and_preserves_path(self):
        runner = object.__new__(GatewayRunner)
        runner.config = GatewayConfig(
            platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake")}
        )
        runner.adapters = {}
        runner._pending_native_image_paths_by_session = {}
        runner._session_model_overrides = {}
        runner._session_reasoning_overrides = {}
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="text-document",
            chat_type="dm",
            user_id="42",
            user_name="Tester",
        )
        event = MessageEvent(
            text="summarize this",
            message_type=MessageType.DOCUMENT,
            source=source,
            media_urls=["/cache/notes.txt"],
            media_types=["text/plain"],
            media_text_inlined=[False],
        )

        prepared = await runner._prepare_inbound_message_text(
            event=event,
            source=source,
            history=[],
        )

        assert prepared is not None
        assert "/cache/notes.txt" in prepared
        assert "included below" not in prepared
        assert "read the cached file" in prepared.lower()


class TestAttachmentTextLabel:
    """The note's text/binary label follows the cached bytes, never the MIME class: platforms and
    mimetypes file source code and data (.rs, .sql, .json, ...) under application/*."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("name, mime, data, inlined, is_text", [
        # Telegram shape: octet-stream from the client, content inlined by the adapter.
        ("main.rs", "application/octet-stream", b'fn main() { println!("hi"); }\n', True, True),
        ("data.json", "application/json", '{"k": "ünïcode"}\n'.encode(), True, True),
        ("schema.sql", "application/sql", b"CREATE TABLE t (id int);\n", False, True),
        ("Makefile", "application/octet-stream", b"all:\n\techo hi\n", False, True),
        ("blob.rs", "application/octet-stream", b"\x00\x01\x02 not text", False, False),
        # A PDF can open with an ASCII-only prefix; it stays a binary document.
        ("report.pdf", "application/pdf", b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n", False, False),
        # ...also without a .pdf name (Discord/Slack keep the upload's name), with no inline flag.
        ("report", "application/pdf", b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n", None, False),
        ("scan", "application/octet-stream", b"%PDF-1.7\n1 0 obj\n<<>>\nendobj\n", None, False),
    ], ids=["rs-inlined", "json-inlined", "sql", "makefile", "nul-bytes", "ascii-pdf",
            "extensionless-pdf", "extensionless-pdf-octet-stream"])
    async def test_label_follows_cached_bytes(self, name, mime, data, inlined, is_text):
        from gateway.platforms.base import cache_media_bytes

        cached = cache_media_bytes(data, filename=name, mime_type=mime)
        runner = object.__new__(GatewayRunner)
        runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake")})
        runner.adapters = {}
        runner._pending_native_image_paths_by_session = {}
        runner._session_model_overrides = {}
        runner._session_reasoning_overrides = {}
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="c", chat_type="dm", user_id="42")
        event = MessageEvent(
            text="review this", message_type=MessageType.DOCUMENT, source=source,
            media_urls=[cached.path], media_types=[cached.media_type], media_text_inlined=[inlined],
        )

        prepared = await runner._prepare_inbound_message_text(event=event, source=source, history=[])

        note = prepared.split("\n\n", 1)[0]
        assert cached.path in note
        assert ("text document" in note) is is_text
        assert ("binary format" in note) is not is_text
        if inlined:
            assert "included below" in note


class TestBinaryDocumentNote:
    @pytest.mark.parametrize(
        "mtype",
        [
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/octet-stream",
        ],
    )
    def test_binary_note_guides_extraction(self, mtype):
        note = _build_document_context_note("contract.pdf", "/cache/doc_contract.pdf", mtype)
        # Records the path so the agent can open it.
        assert "/cache/doc_contract.pdf" in note
        # Tells the agent to read it by extracting the text...
        assert "extract" in note.lower()
        # ...and does NOT steer it into punting back to the user (the bug).
        assert "ask the user" not in note.lower()

