from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PrivacyAnalysis:
    contains_sensitive_data: bool
    contains_pii: bool
    contains_special_category_data: bool
    contains_financial_identifiers: bool
    contains_secrets: bool
    detected_categories: tuple[str, ...]
    match_count: int
    raw_matches: tuple[str, ...]


@dataclass(frozen=True)
class RedactionResult:
    value: Any
    applied_count: int
    categories: tuple[str, ...]


_MASKS: dict[str, str] = {
    "iban": "[IBAN_REDACTED]",
    "bic": "[BIC_REDACTED]",
    "email": "[EMAIL_REDACTED]",
    "siren": "[SIREN_REDACTED]",
    "siret": "[SIRET_REDACTED]",
    "nir": "[NIR_REDACTED]",
    "tax_identifier": "[TAX_ID_REDACTED]",
    "credit_card": "[CARD_REDACTED]",
    "secret": "[SECRET_REDACTED]",
}

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("secret", re.compile(r"\b(?:sk_live|sk_test|rk_live|rk_test)-[A-Za-z0-9_-]{16,}\b")),
    ("secret", re.compile(r"-----BEGIN(?: [A-Z]+)? PRIVATE KEY-----")),
    ("secret", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("secret", re.compile(r"\bBearer\s+[A-Za-z0-9._-]{16,}\b", re.IGNORECASE)),
    ("iban", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")),
    ("bic", re.compile(r"\b[A-Z]{6}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b")),
    ("email", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)),
    ("nir", re.compile(r"\b[12]\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{6}\d{2}\b")),
    ("siret", re.compile(r"\b\d{14}\b")),
    ("siren", re.compile(r"\b\d{9}\b")),
    ("tax_identifier", re.compile(r"\bFR[A-Z0-9]{2}\d{9}\b", re.IGNORECASE)),
    ("credit_card", re.compile(r"\b(?:\d[ -]*?){13,19}\b")),
)


def _iter_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        out: list[str] = []
        for key, item in value.items():
            out.extend(_iter_strings(key))
            out.extend(_iter_strings(item))
        return out
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            out.extend(_iter_strings(item))
        return out
    return []


def _mask_match(category: str) -> str:
    return _MASKS.get(category, "[REDACTED]")


def _normalize_excerpt(value: str) -> str:
    compact = " ".join(value.split())
    if len(compact) <= 12:
        return compact
    return compact[:4] + "…" + compact[-4:]


def analyze_privacy_payload(value: Any) -> PrivacyAnalysis:
    categories: set[str] = set()
    raw_matches: list[str] = []
    match_count = 0

    for text in _iter_strings(value):
        for category, pattern in _PATTERNS:
            for match in pattern.finditer(text):
                categories.add(category)
                match_count += 1
                raw_matches.append(f"{category}:{_normalize_excerpt(match.group(0))}")

    contains_pii = bool(categories.intersection({"email", "siren", "siret", "nir", "tax_identifier"}))
    contains_financial_identifiers = bool(categories.intersection({"iban", "bic", "credit_card"}))
    contains_secrets = "secret" in categories
    contains_special_category_data = False

    return PrivacyAnalysis(
        contains_sensitive_data=bool(categories),
        contains_pii=contains_pii,
        contains_special_category_data=contains_special_category_data,
        contains_financial_identifiers=contains_financial_identifiers,
        contains_secrets=contains_secrets,
        detected_categories=tuple(sorted(categories)),
        match_count=match_count,
        raw_matches=tuple(raw_matches[:16]),
    )


def redact_string(value: str) -> tuple[str, int, set[str]]:
    redacted = value
    applied = 0
    categories: set[str] = set()

    for category, pattern in _PATTERNS:
        redacted, count = pattern.subn(_mask_match(category), redacted)
        if count:
            applied += count
            categories.add(category)

    return redacted, applied, categories


def redact_payload(value: Any) -> RedactionResult:
    if isinstance(value, str):
        redacted, applied, categories = redact_string(value)
        return RedactionResult(redacted, applied, tuple(sorted(categories)))
    if isinstance(value, list):
        applied = 0
        categories: set[str] = set()
        redacted_items: list[Any] = []
        for item in value:
            result = redact_payload(item)
            redacted_items.append(result.value)
            applied += result.applied_count
            categories.update(result.categories)
        return RedactionResult(redacted_items, applied, tuple(sorted(categories)))
    if isinstance(value, dict):
        applied = 0
        categories: set[str] = set()
        redacted_map: dict[str, Any] = {}
        for key, item in value.items():
            key_result = redact_payload(key)
            value_result = redact_payload(item)
            redacted_map[str(key_result.value)] = value_result.value
            applied += key_result.applied_count + value_result.applied_count
            categories.update(key_result.categories)
            categories.update(value_result.categories)
        return RedactionResult(redacted_map, applied, tuple(sorted(categories)))
    return RedactionResult(value, 0, ())


def assert_no_raw_sensitive_data(value: Any) -> list[str]:
    return list(analyze_privacy_payload(value).raw_matches)
