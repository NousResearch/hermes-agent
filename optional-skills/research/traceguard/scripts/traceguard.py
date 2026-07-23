#!/usr/bin/env python3
"""Deterministic evidence-gating for RLM-style parent synthesis.

TraceGuard validates a structured parent answer against evidence handles
accepted from child calls. It is deliberately not an LLM judge: it only checks
that every structured claim names a supported ``fact_id`` and the matching
``evidence_chunk_id``/``chunk_id`` from the manifest.

Run as a terminal helper — pass a JSON payload with ``evidence_manifest``
and ``parent_synthesis`` keys via ``--input FILE`` (or stdin) and read the
JSON verdict from stdout:

    python3 traceguard.py --input payload.json

Exit codes: 0 = all claims accepted, 1 = one or more claims rejected,
2 = invalid payload.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


CLAIM_FACT_KEYS = ("fact_id", "supports_fact_id")
CLAIM_FACT_LIST_KEYS = ("fact_ids", "supports_fact_ids", "supported_fact_ids")
CHUNK_KEYS = ("chunk_id", "evidence_chunk_id", "source_chunk_id")
CANONICAL_EVIDENCE_MANIFEST_FIELDS = (
    "fact_id",
    "chunk_id",
    "text",
    "child_call_id",
)


@dataclass(frozen=True, slots=True)
class TraceGuardEvidence:
    """A child evidence handle accepted by the parent scaffold."""

    fact_id: str
    chunk_id: str
    text: str
    child_call_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "chunk_id": self.chunk_id,
            "text": self.text,
            "child_call_id": self.child_call_id,
        }


@dataclass(frozen=True, slots=True)
class TraceGuardClaim:
    """A structured fact claim extracted from a parent synthesis object."""

    fact_id: str | None
    chunk_id: str | None
    surface: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "chunk_id": self.chunk_id,
            "surface": self.surface,
            "text": self.text,
        }


@dataclass(frozen=True, slots=True)
class TraceGuardRejection:
    """A structured reason why a parent claim was rejected."""

    reason: str
    claim: TraceGuardClaim
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "claim": self.claim.to_dict(),
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class TraceGuardResult:
    """Validation result for one parent synthesis object."""

    accepted: bool
    accepted_claims: tuple[TraceGuardClaim, ...]
    rejected_claims: tuple[TraceGuardRejection, ...]
    allowed_fact_ids: tuple[str, ...]
    allowed_chunk_ids: tuple[str, ...]

    @property
    def unsupported_claim_rate(self) -> float:
        total = len(self.accepted_claims) + len(self.rejected_claims)
        if total == 0:
            return 0.0
        return round(len(self.rejected_claims) / total, 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "unsupported_claim_rate": self.unsupported_claim_rate,
            "accepted_claims": [claim.to_dict() for claim in self.accepted_claims],
            "rejected_claims": [
                rejection.to_dict() for rejection in self.rejected_claims
            ],
            "allowed_fact_ids": list(self.allowed_fact_ids),
            "allowed_chunk_ids": list(self.allowed_chunk_ids),
        }


def normalize_allowed_evidence_manifest(
    evidence_manifest: Iterable[TraceGuardEvidence | Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Return canonical evidence handles for deterministic trace artifacts."""
    entries = [item.to_dict() for item in coerce_evidence_manifest(evidence_manifest)]
    return tuple(
        sorted(
            entries,
            key=lambda entry: (
                entry["fact_id"],
                entry["chunk_id"],
                entry["child_call_id"] or "",
                entry["text"],
            ),
        )
    )


def coerce_evidence_manifest(
    evidence_manifest: Iterable[TraceGuardEvidence | Mapping[str, Any]],
) -> tuple[TraceGuardEvidence, ...]:
    """Coerce manifest mappings into ``TraceGuardEvidence`` objects.

    Mappings with no fact id or no chunk handle are ignored because they cannot
    support a claim. This matches RLM Forge's bounded-manifest behavior while
    accepting Hermes tool-call payloads that use ``evidence_chunk_id``.
    """
    evidence: list[TraceGuardEvidence] = []
    for item in evidence_manifest:
        if isinstance(item, TraceGuardEvidence):
            if item.fact_id and item.chunk_id:
                evidence.append(item)
            continue
        if not isinstance(item, Mapping):
            continue
        fact_id = _first_string(item, CLAIM_FACT_KEYS)
        chunk_id = _first_string(item, CHUNK_KEYS)
        if not fact_id or not chunk_id:
            continue
        evidence.append(
            TraceGuardEvidence(
                fact_id=fact_id,
                chunk_id=chunk_id,
                text=_claim_text(item),
                child_call_id=_optional_string(item.get("child_call_id")),
            )
        )
    return tuple(evidence)


def build_manifest_from_fixture(
    fixture: Mapping[str, Any],
) -> tuple[TraceGuardEvidence, ...]:
    """Create accepted evidence handles from retained facts in a fixture."""
    retained = fixture.get("expected_retained_facts", ())
    if not isinstance(retained, Iterable):
        return ()

    evidence: list[TraceGuardEvidence] = []
    for index, fact in enumerate(retained, start=1):
        if not isinstance(fact, Mapping):
            continue
        fact_id = _first_string(fact, CLAIM_FACT_KEYS)
        chunk_id = _first_string(fact, CHUNK_KEYS)
        text = _claim_text(fact)
        if fact_id and chunk_id and text:
            evidence.append(
                TraceGuardEvidence(
                    fact_id=fact_id,
                    chunk_id=chunk_id,
                    text=text,
                    child_call_id=f"child_{index:04d}",
                )
            )
    return tuple(evidence)


def validate_parent_synthesis(
    *,
    evidence_manifest: Iterable[TraceGuardEvidence | Mapping[str, Any]],
    parent_synthesis: Mapping[str, Any],
) -> TraceGuardResult:
    """Validate parent synthesis claims against accepted child evidence."""
    manifest = coerce_evidence_manifest(evidence_manifest)
    allowed_by_fact = {item.fact_id: item for item in manifest}
    allowed_fact_ids = tuple(allowed_by_fact)
    allowed_chunk_ids = tuple(dict.fromkeys(item.chunk_id for item in manifest))

    accepted: list[TraceGuardClaim] = []
    rejected: list[TraceGuardRejection] = []
    for claim in extract_parent_claims(parent_synthesis):
        if claim.fact_id is None:
            rejected.append(
                TraceGuardRejection(
                    reason="chunk_handle_without_fact",
                    claim=claim,
                    detail=(
                        "The parent cited a chunk handle but did not identify a "
                        "supported fact."
                    ),
                )
            )
            continue

        evidence = allowed_by_fact.get(claim.fact_id)
        if evidence is None:
            rejected.append(
                TraceGuardRejection(
                    reason="unsupported_fact_id",
                    claim=claim,
                    detail=(
                        f"{claim.fact_id} is not present in the accepted child "
                        "evidence manifest."
                    ),
                )
            )
            continue

        if claim.chunk_id is None:
            rejected.append(
                TraceGuardRejection(
                    reason="missing_evidence_handle",
                    claim=claim,
                    detail=f"{claim.fact_id} lacks a chunk/evidence handle.",
                )
            )
            continue

        if claim.chunk_id != evidence.chunk_id:
            rejected.append(
                TraceGuardRejection(
                    reason="evidence_handle_mismatch",
                    claim=claim,
                    detail=(
                        f"{claim.fact_id} must cite {evidence.chunk_id}, "
                        f"not {claim.chunk_id}."
                    ),
                )
            )
            continue

        accepted.append(claim)

    return TraceGuardResult(
        accepted=not rejected,
        accepted_claims=tuple(accepted),
        rejected_claims=tuple(rejected),
        allowed_fact_ids=allowed_fact_ids,
        allowed_chunk_ids=allowed_chunk_ids,
    )


def extract_parent_claims(
    parent_synthesis: Mapping[str, Any],
) -> tuple[TraceGuardClaim, ...]:
    """Extract structured, claim-bearing entries from a parent synthesis."""
    claims: list[TraceGuardClaim] = []
    result = parent_synthesis.get("result")
    if isinstance(result, Mapping):
        for key in (
            "retained_facts",
            "observed_facts",
            "facts",
            "retained_evidence",
            "observed_evidence",
        ):
            claims.extend(_claims_from_surface(result.get(key), f"result.{key}"))
        if isinstance(result.get("fact_id"), str):
            claims.extend(_claims_from_surface(result, "result"))

    claims.extend(
        _claims_from_surface(
            parent_synthesis.get("evidence_references"),
            "evidence_references",
        )
    )
    return tuple(claims)


def _claims_from_surface(value: Any, surface: str) -> list[TraceGuardClaim]:
    if isinstance(value, Mapping):
        return _claims_from_mapping(value, surface)
    if isinstance(value, list):
        claims: list[TraceGuardClaim] = []
        for item in value:
            if isinstance(item, Mapping):
                claims.extend(_claims_from_mapping(item, surface))
        return claims
    return []


def _claims_from_mapping(value: Mapping[str, Any], surface: str) -> list[TraceGuardClaim]:
    fact_ids = _supported_fact_ids(value)
    chunk_id = _first_string(value, CHUNK_KEYS)
    text = _claim_text(value)
    if not fact_ids and chunk_id:
        return [
            TraceGuardClaim(
                fact_id=None,
                chunk_id=chunk_id,
                surface=surface,
                text=text,
            )
        ]
    return [
        TraceGuardClaim(
            fact_id=fact_id,
            chunk_id=chunk_id,
            surface=surface,
            text=text,
        )
        for fact_id in fact_ids
    ]


def _supported_fact_ids(value: Mapping[str, Any]) -> tuple[str, ...]:
    fact_ids: list[str] = []
    for key in CLAIM_FACT_KEYS:
        item = value.get(key)
        if isinstance(item, str) and item:
            fact_ids.append(item)
    for key in CLAIM_FACT_LIST_KEYS:
        item = value.get(key)
        if isinstance(item, list):
            fact_ids.extend(entry for entry in item if isinstance(entry, str) and entry)
    return tuple(dict.fromkeys(fact_ids))


def _first_string(value: Mapping[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        item = value.get(key)
        if isinstance(item, str) and item:
            return item
    return None


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _claim_text(value: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key in ("quoted_evidence", "text", "statement", "claim", "summary"):
        item = value.get(key)
        if isinstance(item, str) and item:
            parts.append(item)
    return "\n".join(parts)


def validate_payload(payload: Any) -> dict[str, Any]:
    """Validate a ``{"evidence_manifest": [...], "parent_synthesis": {...}}`` payload."""
    if not isinstance(payload, Mapping):
        return {
            "success": False,
            "error": "payload must be a JSON object with evidence_manifest and parent_synthesis",
        }
    evidence_manifest = payload.get("evidence_manifest")
    parent_synthesis = payload.get("parent_synthesis")
    if not isinstance(evidence_manifest, list):
        return {
            "success": False,
            "error": "evidence_manifest must be a list of evidence handle objects",
        }
    if not isinstance(parent_synthesis, Mapping):
        return {
            "success": False,
            "error": "parent_synthesis must be a structured JSON object",
        }

    result = validate_parent_synthesis(
        evidence_manifest=evidence_manifest,
        parent_synthesis=parent_synthesis,
    )
    return {
        "success": True,
        "traceguard": result.to_dict(),
        "normalized_evidence_manifest": list(
            normalize_allowed_evidence_manifest(evidence_manifest)
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate structured parent-synthesis claims against accepted "
            "child evidence handles."
        ),
    )
    parser.add_argument(
        "--input",
        default="-",
        help="Path to the JSON payload, or '-' to read stdin (default).",
    )
    args = parser.parse_args(argv)

    try:
        if args.input == "-":
            payload = json.load(sys.stdin)
        else:
            with open(args.input, encoding="utf-8") as handle:
                payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        print(json.dumps({"success": False, "error": f"could not read payload: {error}"}))
        return 2

    verdict = validate_payload(payload)
    print(json.dumps(verdict, ensure_ascii=False, indent=2))
    if not verdict["success"]:
        return 2
    return 0 if verdict["traceguard"]["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
