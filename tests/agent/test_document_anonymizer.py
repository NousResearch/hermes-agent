import base64
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_document_text_replaces_people_money_and_identifiers_deterministically():
    from agent.document_anonymizer import anonymize_document_text

    source = (
        "Договор с Ивановым Иваном Ивановичем. Иванов Иван Иванович получает "
        "1 250 000 рублей. Телефон +7 999 123-45-67, e-mail ivanov@example.ru. "
        "Паспорт 45 01 123456. Повторная сумма: 1 250 000 руб."
    )
    result = anonymize_document_text(source)

    assert "Иванов" not in result
    assert "ivanov@example.ru" not in result
    assert "+7 999 123-45-67" not in result
    assert "45 01 123456" not in result
    assert "1 250 000" not in result
    assert "person1" in result
    assert result.count("SUM1") == 2


def test_plain_message_is_never_anonymized_by_document_block_filter():
    from agent.document_anonymizer import anonymize_document_blocks

    text = "Позвони Иванову Ивану Ивановичу и проверь 10 000 рублей"
    assert anonymize_document_blocks(text) == text


def test_latin_full_name_birth_date_and_address_are_removed():
    from agent.document_anonymizer import anonymize_document_text

    text = (
        "John Michael Smith; дата рождения: 01.02.1980; "
        "адрес регистрации: г. Москва, ул. Тверская, д. 1\n"
    )
    result = anonymize_document_text(text)
    assert "John Michael Smith" not in result
    assert "01.02.1980" not in result
    assert "Тверская" not in result


def test_only_attached_context_block_is_anonymized():
    from agent.document_anonymizer import anonymize_document_blocks

    text = (
        "Обычный вопрос Иванову Ивану Ивановичу на 10 000 рублей\n\n"
        "--- Attached Context ---\n\n"
        "Договор Петрова Петра Петровича на 25 000 рублей"
    )
    result = anonymize_document_blocks(text)
    assert "Иванову Ивану Ивановичу" in result
    assert "10 000 рублей" in result
    assert "Петрова Петра Петровича" not in result
    assert "25 000" not in result


def test_openwebui_source_tags_are_anonymized_without_touching_user_query(monkeypatch):
    from agent.document_anonymizer import anonymize_openwebui_source_blocks

    monkeypatch.setenv("HERMES_DOCUMENT_ANONYMIZATION", "1")
    text = (
        "Вопрос Иванову Ивану Ивановичу на 10 000 рублей\n"
        '<source id="1" name="contract.pdf">Петров Петр Петрович, 25 000 рублей</source>'
    )
    result = anonymize_openwebui_source_blocks(text)
    assert "Иванову Ивану Ивановичу" in result
    assert "10 000 рублей" in result
    assert "Петров Петр Петрович" not in result
    assert "25 000" not in result
    assert '<source anonymized="true">' in result


def test_text_file_reference_is_anonymized_but_python_source_is_not(tmp_path, monkeypatch):
    from agent.context_references import ContextReference, _expand_file_reference

    monkeypatch.setenv("HERMES_DOCUMENT_ANONYMIZATION", "1")
    payload = "Сотрудник Петров Петр Петрович, выплата 99 000 руб."
    doc = tmp_path / "contract.txt"
    code = tmp_path / "contract.py"
    doc.write_text(payload, encoding="utf-8")
    code.write_text(payload, encoding="utf-8")

    ref_doc = ContextReference(raw=f"@file:{doc.name}", kind="file", target=doc.name, start=0, end=1)
    ref_code = ContextReference(raw=f"@file:{code.name}", kind="file", target=code.name, start=0, end=1)
    _, doc_block = _expand_file_reference(ref_doc, tmp_path)
    _, code_block = _expand_file_reference(ref_code, tmp_path)

    assert "Петров Петр Петрович" not in doc_block
    assert "99 000" not in doc_block
    assert "Петров Петр Петрович" in code_block
    assert "99 000" in code_block


def test_gateway_document_event_is_sanitized_and_original_path_removed(tmp_path, monkeypatch):
    from agent.document_anonymizer import sanitize_document_event

    monkeypatch.setenv("HERMES_DOCUMENT_ANONYMIZATION", "1")
    doc = tmp_path / "contract.txt"
    payload = "Сидоров Сидор Сидорович — 50 000 руб."
    caption = "Сообщение Иванову Ивану Ивановичу на 88 000 рублей"
    doc.write_text(payload, encoding="utf-8")
    event = SimpleNamespace(
        media_urls=[str(doc)],
        media_types=["text/plain"],
        message_type="document",
    )

    text = sanitize_document_event(
        f"[Content of contract.txt]:\n{payload}\n\n{caption}", event
    )
    assert caption in text
    assert "Сидоров Сидор Сидорович" not in text
    assert "50 000" not in text
    assert "person1" in text
    assert "SUM1" in text
    assert event.media_urls == []
    assert event.media_types == []


def test_gateway_anonymization_fails_closed(tmp_path, monkeypatch):
    from agent.document_anonymizer import sanitize_document_event

    monkeypatch.setenv("HERMES_DOCUMENT_ANONYMIZATION", "1")
    bad = tmp_path / "scan.pdf"
    bad.write_bytes(b"not a pdf")
    event = SimpleNamespace(
        media_urls=[str(bad)], media_types=["application/pdf"], message_type="document"
    )

    text = sanitize_document_event("Изучи", event)
    assert "не передан модели" in text
    assert str(bad) not in text
    assert event.media_urls == []


def test_context_preprocessor_hides_original_document_reference(tmp_path, monkeypatch):
    from agent.context_references import preprocess_context_references

    monkeypatch.setenv("HERMES_DOCUMENT_ANONYMIZATION", "1")
    doc = tmp_path / "Иванов Иван Иванович contract.txt"
    doc.write_text("Иванов Иван Иванович — 12 000 рублей", encoding="utf-8")
    authored = f'Проверь @file:"{doc.name}" пожалуйста'
    result = preprocess_context_references(authored, cwd=tmp_path, context_length=8192)

    assert doc.name not in result.message
    assert "[анонимизированный документ]" in result.message
    assert "Иванов Иван Иванович" not in result.message
    assert "12 000" not in result.message
    assert result.original_message == authored


def test_openai_inline_file_part_becomes_anonymized_text(monkeypatch):
    from gateway.platforms.api_server import _normalize_multimodal_content

    monkeypatch.setenv("HERMES_DOCUMENT_ANONYMIZATION", "1")
    raw = "Иванов Иван Иванович получает 123 000 рублей".encode()
    result = _normalize_multimodal_content([
        {"type": "text", "text": "Сообщение Петрову Петру Петровичу на 7 000 рублей"},
        {
            "type": "file",
            "file": {
                "filename": "contract.txt",
                "file_data": "data:text/plain;base64," + base64.b64encode(raw).decode(),
            },
        },
    ])

    assert "Сообщение Петрову Петру Петровичу на 7 000 рублей" in result
    assert "Иванов Иван Иванович" not in result
    assert "123 000" not in result
    assert "person1" in result and "SUM1" in result


def test_openai_inline_file_part_is_rejected_when_feature_disabled(monkeypatch):
    from gateway.platforms.api_server import _normalize_multimodal_content

    monkeypatch.setenv("HERMES_DOCUMENT_ANONYMIZATION", "0")
    with pytest.raises(ValueError, match="anonymize_documents=true"):
        _normalize_multimodal_content([
            {"type": "input_file", "filename": "contract.txt", "file_data": "WA=="}
        ])
