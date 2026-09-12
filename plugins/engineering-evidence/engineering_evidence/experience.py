"""Sanitized engineering learning candidates for the shared Vault."""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .receipts import validate_receipt


_SECRET_RE = re.compile(
    r"(?i)(token|password|secret|api[_-]?key|private[_-]?key)\s*([:=])\s*([^\s,;]+)"
)
_SECRET_NAME_RE = re.compile(r"(?i)\b(?:HERMES|GITHUB|OPENAI|AWS)_[A-Z0-9_]*(?:TOKEN|KEY|SECRET)\b")
_VALIDATOR_ROLE = "memory-validator"


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if not isinstance(value, str):
        return value
    value = _SECRET_RE.sub(lambda match: f"{match.group(1)}{match.group(2)}<redacted>", value)
    return _SECRET_NAME_RE.sub("<secret-name>", value)


def build_experience_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    candidate = _sanitize(candidate)
    return validate_receipt(
        {
            "schema_name": "engineering_evidence_v1",
            "receipt_type": "experience_candidate",
            "receipt_id": f"experience-{uuid.uuid4().hex}",
            "repository": candidate.get("repository", "unknown"),
            "head_sha": candidate.get("head_sha", ""),
            "authority": "diagnostic-only",
            "promotion_state": "proposed",
            "candidate": candidate,
        }
    )


def store_experience_candidate(vault_root: Path, candidate: dict[str, Any]) -> Path:
    """Write a candidate in the allowlisted Vault learning format atomically."""

    receipt = build_experience_candidate(candidate)
    record_id = receipt["receipt_id"]
    destination = vault_root.expanduser().resolve() / "Memories" / "Hermes" / "Engineering" / f"{record_id}.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(receipt["candidate"], indent=2, sort_keys=True)
    frontmatter = "\n".join(
        [
            "---",
            "schema_name: agent_learning_record_v1",
            f"record_id: {record_id}",
            f"title: Engineering experience candidate {record_id[-12:]}",
            "kind: memory",
            "agent: hermes",
            "memory_area: Engineering",
            "status: proposed",
            "classification: diagnostic-only",
            "authority: source-index",
            "sync_owned: true",
            "execution_status: not-applicable",
            "---",
            "",
        ]
    )
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(frontmatter + payload + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return destination


def _candidate_body(path: Path) -> dict[str, Any]:
    content = path.read_text(encoding="utf-8")
    try:
        _, body = content.split("\n---\n", 1)
        return json.loads(body)
    except (ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid experience candidate: {path}") from exc


def _promotion_path(candidate_path: Path) -> Path:
    return candidate_path.with_name(f"{candidate_path.name}.promotion.json")


def promote_experience_candidate(
    candidate_path: Path,
    *,
    validator_role: str,
    rationale: str,
) -> dict[str, Any]:
    """Create a separate validation record; the candidate itself remains immutable."""

    if validator_role != _VALIDATOR_ROLE:
        raise ValueError("validator role must be memory-validator")
    if not rationale.strip():
        raise ValueError("promotion rationale is required")
    candidate_path = candidate_path.expanduser().resolve()
    if candidate_path.parent.name != "Engineering" or candidate_path.parent.parent.name != "Hermes":
        raise ValueError("candidate must be inside the Hermes Engineering Vault")
    candidate = build_experience_candidate(_candidate_body(candidate_path))
    promotion = {
        "schema_name": "engineering_evidence_v1",
        "promotion_type": "experience_validation",
        "record_id": candidate_path.stem,
        "status": "validated",
        "validator_role": validator_role,
        "rationale": _sanitize(rationale),
        "source_reference": str(candidate_path),
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "authority": "diagnostic-only",
    }
    destination = _promotion_path(candidate_path)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(promotion, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return promotion


def search_experience_candidates(
    vault_root: Path,
    repository: str,
    head_sha: str,
    *,
    terms: list[str] | None = None,
    affected_files: list[str] | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Return only explicitly validated candidates matching the exact repo and HEAD."""

    if limit < 1 or limit > 50:
        raise ValueError("limit must be between 1 and 50")
    if not re.fullmatch(r"[0-9a-f]{40}", head_sha):
        raise ValueError("head_sha must be a 40-character hexadecimal commit")
    root = vault_root.expanduser().resolve() / "Memories" / "Hermes" / "Engineering"
    wanted = [item.strip().lower() for item in [*(terms or []), *(affected_files or [])] if item.strip()]
    matches: list[dict[str, Any]] = []
    if not root.is_dir():
        return matches
    for path in sorted(root.glob("experience-*.md")):
        promotion_path = _promotion_path(path)
        if not promotion_path.is_file():
            continue
        try:
            promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
            candidate = _candidate_body(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if promotion.get("status") != "validated":
            continue
        if candidate.get("repository") != repository or candidate.get("head_sha") != head_sha:
            continue
        haystack = json.dumps(candidate, sort_keys=True).lower()
        score = sum(haystack.count(term) for term in wanted)
        if wanted and score == 0:
            continue
        matches.append(
            {
                "receipt_id": promotion.get("record_id"),
                "source_reference": str(path),
                "promotion_reference": str(promotion_path),
                "score": score,
                "candidate": candidate,
            }
        )
    matches.sort(key=lambda item: (-int(item["score"]), str(item["receipt_id"])))
    return matches[:limit]
