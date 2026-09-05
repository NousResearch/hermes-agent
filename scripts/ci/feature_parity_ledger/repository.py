"""Strict JSON loading and repository-wide ledger validation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .core import (
    LEDGER_DIRECTORY,
    MAX_DOCUMENT_BYTES,
    REGISTRY_PATH,
    DuplicateJsonKeyError,
    LedgerValidationError,
    _DUPLICATE_KEY_HOOK,
    _canonical_repo_path,
    _is_int,
)
from .ledger import validate_ledger
from .registry import validate_contract_registry

def load_json_document(
    path: Path,
    *,
    max_bytes: int = MAX_DOCUMENT_BYTES,
) -> Mapping[str, Any]:
    """Load bounded, strict UTF-8 JSON with duplicate-key rejection."""
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise LedgerValidationError([f"cannot stat {path}: {exc}"]) from exc
    if size > max_bytes:
        raise LedgerValidationError(
            [f"{path} exceeds the {max_bytes}-byte document limit"]
        )
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise LedgerValidationError([f"cannot read {path}: {exc}"]) from exc
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LedgerValidationError([f"{path}: invalid UTF-8: {exc}"]) from exc
    try:
        document = json.loads(text, object_pairs_hook=_DUPLICATE_KEY_HOOK)
    except (json.JSONDecodeError, DuplicateJsonKeyError) as exc:
        raise LedgerValidationError([f"{path}: invalid JSON: {exc}"]) from exc
    if not isinstance(document, Mapping):
        raise LedgerValidationError([f"{path}: JSON root must be an object"])
    return document


def discover_ledgers(repository_root: Path) -> list[Path]:
    directory = repository_root / LEDGER_DIRECTORY
    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.glob("*.json")
        if path.name != REGISTRY_PATH.name
    )


def _verify_receipt(
    repository_root: Path,
    ledger_path: Path,
    row: Mapping[str, Any],
    errors: list[str],
) -> None:
    if row.get("delivery_state") != "released":
        return
    release = row.get("release_evidence")
    if not isinstance(release, Mapping):
        return
    receipt = release.get("live_receipt")
    if not isinstance(receipt, Mapping):
        return
    receipt_path = receipt.get("path")
    digest = receipt.get("sha256")
    path_errors: list[str] = []
    canonical = _canonical_repo_path(receipt_path, "live_receipt.path", path_errors)
    if path_errors or not canonical or not isinstance(digest, str):
        return
    candidate = (repository_root / canonical).resolve()
    try:
        candidate.relative_to(repository_root.resolve())
    except ValueError:
        errors.append(f"{ledger_path}: live receipt escapes repository root")
        return
    if not candidate.is_file():
        errors.append(f"{ledger_path}: live receipt does not exist: {canonical}")
        return
    try:
        actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
    except OSError as exc:
        errors.append(f"{ledger_path}: cannot read live receipt {canonical}: {exc}")
        return
    if actual != digest:
        errors.append(
            f"{ledger_path}: live receipt hash mismatch for {canonical}: expected {digest}, got {actual}"
        )


def validate_repository(repository_root: Path) -> list[str]:
    """Validate every ledger, the registry, cross-ledger ownership, and receipts."""
    root = repository_root.resolve()
    registry_path = root / REGISTRY_PATH
    try:
        registry = load_json_document(registry_path)
    except LedgerValidationError as exc:
        return list(exc.errors)

    errors = [f"{registry_path}: {error}" for error in validate_contract_registry(registry)]
    campaign_ids: list[tuple[str, Path]] = []
    trackers: list[tuple[int, Path]] = []
    authorities: list[tuple[str, int, str, Path]] = []

    ledgers = discover_ledgers(root)
    for path in ledgers:
        try:
            document = load_json_document(path)
        except LedgerValidationError as exc:
            errors.extend(exc.errors)
            continue
        ledger_errors = validate_ledger(document, contract_registry=registry)
        errors.extend(f"{path}: {error}" for error in ledger_errors)

        campaign = document.get("campaign")
        if isinstance(campaign, Mapping):
            campaign_id = campaign.get("id")
            tracker = campaign.get("tracker")
            repository = campaign.get("repository")
            if isinstance(campaign_id, str):
                campaign_ids.append((campaign_id, path))
            if _is_int(tracker):
                trackers.append((tracker, path))
            if isinstance(repository, str):
                for row in document.get("capabilities", []):
                    if not isinstance(row, Mapping):
                        continue
                    for publication in row.get("publications", []):
                        if not isinstance(publication, Mapping):
                            continue
                        if (
                            publication.get("role") == "authoritative"
                            and publication.get("kind") == "pull_request"
                            and _is_int(publication.get("number"))
                        ):
                            authorities.append(
                                (
                                    repository,
                                    publication["number"],
                                    str(row.get("id", "")),
                                    path,
                                )
                            )
                    _verify_receipt(root, path, row, errors)

    for campaign_id, count in Counter(item[0] for item in campaign_ids).items():
        if count > 1:
            paths = sorted(str(path) for value, path in campaign_ids if value == campaign_id)
            errors.append(f"campaign id {campaign_id!r} is duplicated across ledgers: {paths}")
    for tracker, count in Counter(item[0] for item in trackers).items():
        if count > 1:
            paths = sorted(str(path) for value, path in trackers if value == tracker)
            errors.append(f"tracker issue #{tracker} is duplicated across ledgers: {paths}")
    key_counts = Counter((repository, number) for repository, number, _, _ in authorities)
    for (repository, number), count in key_counts.items():
        if count > 1:
            owners = sorted(
                f"{capability_id}@{path}"
                for repo, pr_number, capability_id, path in authorities
                if repo == repository and pr_number == number
            )
            errors.append(
                f"authoritative pull request {repository}#{number} is claimed across ledgers: {owners}"
            )

    return errors


def validate_path(
    path: Path,
    *,
    contract_registry: Mapping[str, Any] | None = None,
) -> list[str]:
    try:
        document = load_json_document(path)
    except LedgerValidationError as exc:
        return list(exc.errors)
    return validate_ledger(document, contract_registry=contract_registry)
