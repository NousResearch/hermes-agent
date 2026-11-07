"""The inbound document note must not name tooling the run may not have.

``_build_document_context_note`` named the ``terminal`` tool and the ``ocr-and-documents`` skill
unconditionally. The note is built while the inbound message is prepared, before the turn's agent
(and the toolsets it is pinned to) is resolved, so a deployment with ``agent.disabled_toolsets``
including ``terminal`` is still told to reach for it; the bundled skill library has no
``ocr-and-documents`` skill at all (its pages live under the ``pdf`` skill), so the second half of
the hint is unloadable everywhere. A ``message/rfc822`` attachment is additionally readable text,
yet it took the binary "extract the text" branch (#150537).
"""

import importlib

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource

gateway_run = importlib.import_module("gateway.run")
_build_document_context_note = gateway_run._build_document_context_note

BINARY_TYPES = [
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/octet-stream",
    "application/zip",
]


class TestBinaryNoteNamesNoUnavailableTool:
    @pytest.mark.parametrize("mtype", BINARY_TYPES)
    def test_note_does_not_name_the_terminal_tool(self, mtype):
        note = _build_document_context_note("contract.pdf", "/cache/doc_contract.pdf", mtype)
        assert "terminal" not in note.lower()

    @pytest.mark.parametrize("mtype", BINARY_TYPES)
    def test_note_does_not_name_a_skill(self, mtype):
        note = _build_document_context_note("contract.pdf", "/cache/doc_contract.pdf", mtype)
        assert "skill" not in note.lower()
        assert "ocr-and-documents" not in note

    def test_note_still_steers_the_agent_to_extract_the_text(self):
        note = _build_document_context_note("contract.pdf", "/cache/doc_contract.pdf", "application/pdf")
        assert "/cache/doc_contract.pdf" in note
        assert "extract" in note.lower()
        # The original bug this note exists for: never invite the model to punt back to the user.
        assert "ask the user" not in note.lower()


class TestEmlIsReadableText:
    def test_non_inlined_eml_note_points_at_the_cached_file(self):
        note = _build_document_context_note(
            "mail.eml", "/cache/doc_mail.eml", "message/rfc822", content_inlined=False,
        )
        assert "/cache/doc_mail.eml" in note
        assert "binary" not in note.lower()
        # Nothing has to be extracted from an .eml — it is text on disk.
        assert "extract" not in note.lower()
        assert "read the cached file" in note.lower()

    def test_inlined_eml_note_says_the_content_is_included(self):
        note = _build_document_context_note(
            "mail.eml", "/cache/doc_mail.eml", "message/rfc822", content_inlined=True,
        )
        assert "included below" in note.lower()
        assert "binary" not in note.lower()


class TestNoteThroughTheInboundPipeline:
    @pytest.fixture()
    def runner(self):
        runner = object.__new__(GatewayRunner)
        runner.config = GatewayConfig(
            platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="fake")}
        )
        runner.adapters = {}
        runner._session_model_overrides = {}
        return runner

    def _source(self, chat_id):
        return SessionSource(
            platform=Platform.DISCORD,
            chat_id=chat_id,
            chat_type="dm",
            user_id="42",
            user_name="Tester",
        )

    @pytest.mark.asyncio
    async def test_eml_attachment_gets_the_text_note(self, runner):
        source = self._source("eml-document")
        event = MessageEvent(
            text="summarize this thread",
            message_type=MessageType.DOCUMENT,
            source=source,
            media_urls=["/cache/mail.eml"],
            media_types=["message/rfc822"],
            media_text_inlined=[False],
        )

        prepared = await runner._prepare_inbound_message_text(
            event=event, source=source, history=[],
        )

        assert prepared is not None
        assert "/cache/mail.eml" in prepared
        assert "read the cached file" in prepared.lower()
        assert "binary" not in prepared.lower()

    @pytest.mark.asyncio
    async def test_pdf_attachment_note_names_no_tool_the_run_lacks(self, runner):
        source = self._source("pdf-document")
        event = MessageEvent(
            text="what does this say?",
            message_type=MessageType.DOCUMENT,
            source=source,
            media_urls=["/cache/contract.pdf"],
            media_types=["application/pdf"],
            media_text_inlined=[False],
        )

        prepared = await runner._prepare_inbound_message_text(
            event=event, source=source, history=[],
        )

        assert prepared is not None
        assert "/cache/contract.pdf" in prepared
        assert "terminal" not in prepared.lower()
        assert "skill" not in prepared.lower()
