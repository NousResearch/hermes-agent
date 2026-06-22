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

gateway_run = importlib.import_module("gateway.run")
_build_document_context_note = gateway_run._build_document_context_note
_sanitize_gateway_final_response = gateway_run._sanitize_gateway_final_response


class TestTextDocumentNote:
    @pytest.mark.parametrize("mtype", ["text/plain", "text/markdown", "text/csv"])
    def test_text_note_mentions_included_content_and_path(self, mtype):
        note = _build_document_context_note("notes.txt", "/cache/doc_notes.txt", mtype)
        assert "text document" in note
        assert "notes.txt" in note
        assert "/cache/doc_notes.txt" in note
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
        assert "paste" in note.lower()

    def test_binary_note_distinct_from_text_note(self):
        text_note = _build_document_context_note("a.txt", "/c/a.txt", "text/plain")
        pdf_note = _build_document_context_note("a.pdf", "/c/a.pdf", "application/pdf")
        assert text_note != pdf_note
        # The text path claims content is inlined; the binary path must not.
        assert "included below" in text_note
        assert "included below" not in pdf_note


class TestTelegramDocumentPathRedaction:
    def test_telegram_final_response_redacts_cached_document_paths(self):
        response = (
            "פתחתי את /home/gidon/.hermes/cache/documents/doc_a90e9ffc40fb_jobs.json "
            "והקובץ תקין."
        )

        sanitized = _sanitize_gateway_final_response("telegram", response)

        assert "/home/gidon/.hermes/cache/documents/" not in sanitized
        assert "doc_a90e9ffc40fb_jobs.json" not in sanitized
        assert "[cached document]" in sanitized

    def test_non_telegram_final_response_keeps_existing_behavior(self):
        response = "See /home/gidon/.hermes/cache/documents/doc_a90e9ffc40fb_jobs.json"

        assert _sanitize_gateway_final_response("discord", response) == response
