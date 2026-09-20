"""Shared image pre-analysis contract for document-safe OCR handoff."""

from agent.vision_preanalysis import build_preanalysis_note, IMAGE_PREANALYSIS_PROMPT


def test_prompt_requires_verbatim_document_text_and_uncertainty():
    assert "transcribe identifying text verbatim" in IMAGE_PREANALYSIS_PROMPT
    assert "names, titles, labels, identifiers, medication names, doses" in IMAGE_PREANALYSIS_PROMPT
    assert "write [unclear] instead of guessing" in IMAGE_PREANALYSIS_PROMPT
    assert "Do not silently correct" in IMAGE_PREANALYSIS_PROMPT


def test_handoff_marks_extracted_text_as_non_paraphrasable_source():
    note = build_preanalysis_note(
        description="四川大学华西厦门医院\n阿奇霉素分散片",
        image_path="/tmp/prescription.jpg",
        role_label="user",
    )

    assert "Treat the vision extraction below as source text" in note
    assert "quote exact visible wording" in note
    assert "do not silently correct or paraphrase" in note
    assert "四川大学华西厦门医院" in note
    assert "/tmp/prescription.jpg" in note