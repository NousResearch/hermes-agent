import hashlib
import json
import pathlib
import re
import sqlite3

from .models import ClarificationRequest, FieldMapping

LOCAL_FEW_SHOT_DB = pathlib.Path("/home/tobi/.hermes/profiles/taxpipeline/storage/few_shots.db")
PII_PATTERNS = [
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]){11,30}\b")),
    ("BIC", re.compile(r"\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b")),
    ("EMAIL", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)),
    ("PHONE", re.compile(r"(?:\+49|0)[\s()/.-]*\d(?:[\s()/.-]*\d){6,}")),
    ("TAX_ID", re.compile(r"\b(?:Steuer-?ID|IdNr\.?|Identifikationsnummer)?\s*\d{2}[\s-]?\d{3}[\s-]?\d{3}[\s-]?\d{3}\b", re.IGNORECASE)),
    ("TAX_NUMBER", re.compile(r"\b\d{2,3}[\s/]\d{3}[\s/]\d{4,5}\b")),
    ("ADDRESS", re.compile(r"\b[\wÄÖÜäöüß.-]+(?:straße|strasse|str\.|weg|allee|platz)\s+\d+[a-zA-Z]?(?:,\s*)?\d{5}\s+[\wÄÖÜäöüß.-]+\b", re.IGNORECASE)),
]


def anonymize_pii(text: str) -> str:
    anonymized = text
    for label, pattern in PII_PATTERNS:
        anonymized = pattern.sub(f"<{label}>", anonymized)
    return anonymized


def contains_pii(text: str) -> bool:
    return any(pattern.search(text) for _, pattern in PII_PATTERNS)


def _few_shot_id(user_id: str, request: ClarificationRequest, corrected_mapping: FieldMapping) -> str:
    payload = "|".join(
        [
            user_id,
            request.issue_type.value,
            request.field_id,
            anonymize_pii(request.question),
            corrected_mapping.evidence.document_hash,
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json_value(value: object) -> str:
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    except TypeError:
        return str(value)


def save_user_correction_as_few_shot(
    user_id: str,
    request: ClarificationRequest,
    corrected_mapping: FieldMapping,
    *,
    user_consented: bool = False,
) -> None:
    """Store a resolved dashboard correction as a local profile few-shot.

    The legacy skill requires explicit user consent for storing corrections.
    Without consent this function deliberately does nothing.
    """
    if not user_consented:
        return

    sanitized_question = anonymize_pii(request.question)
    sanitized_value = anonymize_pii(_json_value(corrected_mapping.value))
    sanitized_evidence_text = anonymize_pii(corrected_mapping.evidence.extracted_text)
    if any(contains_pii(value) for value in (sanitized_question, sanitized_value, sanitized_evidence_text)):
        return

    LOCAL_FEW_SHOT_DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(LOCAL_FEW_SHOT_DB) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS tax_few_shots (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                issue_type TEXT NOT NULL,
                field_id TEXT NOT NULL,
                question TEXT NOT NULL,
                resolved_value TEXT NOT NULL,
                evidence_hash TEXT NOT NULL,
                evidence_text TEXT NOT NULL DEFAULT '',
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        columns = {
            row[1]
            for row in cursor.execute("PRAGMA table_info(tax_few_shots)").fetchall()
        }
        if "evidence_text" not in columns:
            cursor.execute("ALTER TABLE tax_few_shots ADD COLUMN evidence_text TEXT NOT NULL DEFAULT ''")
        cursor.execute(
            """
            INSERT OR REPLACE INTO tax_few_shots
            (id, user_id, issue_type, field_id, question, resolved_value, evidence_hash, evidence_text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _few_shot_id(user_id, request, corrected_mapping),
                user_id,
                request.issue_type.value,
                request.field_id,
                sanitized_question,
                sanitized_value,
                corrected_mapping.evidence.document_hash,
                sanitized_evidence_text,
            ),
        )
