"""Shared types, constants, and canonicalization helpers."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

SCHEMA_VERSION = 1
REGISTRY_SCHEMA_VERSION = 1
MAX_DOCUMENT_BYTES = 2_000_000

HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
CAPABILITY_ID = re.compile(r"^[A-Z][A-Z0-9]*[0-9]+$")
CAMPAIGN_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
GITHUB_REPOSITORY = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9_.-]*[A-Za-z0-9])?/[A-Za-z0-9_.-]+$"
)
GITHUB_LOGIN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?$")

PRODUCT_STATES = {
    "accepted",
    "existing",
    "pair_gap",
    "conditional",
    "deferred",
    "rejected",
}
DELIVERY_STATES = {
    "gap",
    "candidate_blocked",
    "candidate_unwired",
    "candidate_open",
    "on_main_unverified",
    "released",
    "superseded",
}
FORBIDDEN_DELIVERY_STATES = {
    "implemented_in_packet",
    "implemented_locally",
    "package_green",
    "patch_ready",
    "branch_exists",
}
PUBLICATION_ROLES = {"authoritative", "dependency", "superseded"}
PUBLICATION_KINDS = {"issue", "pull_request", "commit", "release"}
PUBLICATION_STATES = {"open", "closed", "merged", "superseded"}

ACTIVE_PUBLICATION_STATES = {
    "candidate_blocked",
    "candidate_unwired",
    "candidate_open",
    "on_main_unverified",
    "released",
}
CANDIDATE_STATES = {
    "candidate_blocked",
    "candidate_unwired",
    "candidate_open",
}
CONSUMER_REQUIRED_STATES = {
    "candidate_open",
    "on_main_unverified",
    "released",
}
MAIN_REQUIRED_STATES = {"on_main_unverified", "released"}
DECISION_GATED_PRODUCT_STATES = {
    "pair_gap",
    "conditional",
    "deferred",
    "rejected",
}
NON_PROMOTABLE_PRODUCT_STATES = {
    "pair_gap",
    "conditional",
    "deferred",
    "rejected",
}
REQUIRED_ROW_LISTS = {
    "implementation_paths",
    "test_paths",
    "consumers",
    "publications",
    "artifact_evidence",
}
REGISTRY_PATH = Path("docs/architecture/feature-parity/contracts.json")
LEDGER_DIRECTORY = Path("docs/architecture/feature-parity")


class LedgerValidationError(ValueError):
    """Raised when one or more validation errors prevent safe processing."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(errors)
        super().__init__("\n".join(self.errors))


class DuplicateJsonKeyError(ValueError):
    """Raised when JSON repeats an object key."""


class _DuplicateKeyHook:
    def __call__(self, pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DuplicateJsonKeyError(f"duplicate JSON key: {key!r}")
            result[key] = value
        return result


_DUPLICATE_KEY_HOOK = _DuplicateKeyHook()


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _append_required_string(
    mapping: Mapping[str, Any],
    key: str,
    field: str,
    errors: list[str],
) -> str:
    value = mapping.get(key)
    if not _non_empty_string(value):
        errors.append(f"{field} must be a non-empty string")
        return ""
    return value.strip()


def _required_list(
    mapping: Mapping[str, Any],
    key: str,
    field: str,
    errors: list[str],
) -> list[Any]:
    if key not in mapping:
        errors.append(f"{field} is required")
        return []
    value = mapping.get(key)
    if not isinstance(value, list):
        errors.append(f"{field} must be a list")
        return []
    return value


def _required_string_list(
    mapping: Mapping[str, Any],
    key: str,
    field: str,
    errors: list[str],
) -> list[str]:
    values = _required_list(mapping, key, field, errors)
    if any(not _non_empty_string(item) for item in values):
        errors.append(f"{field} must contain only non-empty strings")
        return []
    stripped = [str(item).strip() for item in values]
    duplicates = sorted(name for name, count in Counter(stripped).items() if count > 1)
    if duplicates:
        errors.append(f"{field} contains duplicates: {duplicates}")
    return stripped


def _parse_utc_timestamp(value: Any, field: str, errors: list[str]) -> None:
    if not isinstance(value, str) or not value.endswith("Z"):
        errors.append(f"{field} must be an explicit UTC timestamp ending in Z")
        return
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        errors.append(f"{field} must be a valid RFC 3339 UTC timestamp")


def _canonical_repo_path(value: Any, field: str, errors: list[str]) -> str:
    if not _non_empty_string(value):
        errors.append(f"{field} must be a non-empty repository-relative path")
        return ""
    path = value.strip()
    if "\\" in path:
        errors.append(f"{field} must use forward slashes")
        return ""
    if path.startswith("/") or re.match(r"^[A-Za-z]:", path):
        errors.append(f"{field} must be repository-relative")
        return ""
    if path.endswith("/") or "//" in path:
        errors.append(f"{field} is not canonical")
        return ""
    parts = PurePosixPath(path).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        errors.append(f"{field} contains an invalid path segment")
        return ""
    canonical = PurePosixPath(*parts).as_posix()
    if canonical != path:
        errors.append(f"{field} is not canonical; expected {canonical!r}")
        return ""
    return canonical


def _consumer_path(value: Any, field: str, errors: list[str]) -> str:
    if not _non_empty_string(value):
        errors.append(f"{field} must identify '<path>:<symbol>'")
        return ""
    raw = value.strip()
    path, separator, symbol = raw.partition(":")
    if not separator or not symbol.strip():
        errors.append(f"{field} must identify '<path>:<symbol>'")
        return ""
    canonical = _canonical_repo_path(path, f"{field}.path", errors)
    if any(character in symbol for character in "\r\n"):
        errors.append(f"{field}.symbol must be one line")
    return canonical


def _github_url(
    value: Any,
    field: str,
    repository: str,
    errors: list[str],
    *,
    expected_path_prefix: str | None = None,
) -> str:
    if not _non_empty_string(value):
        errors.append(f"{field} must be a non-empty GitHub URL")
        return ""
    raw = value.strip()
    parsed = urlparse(raw)
    if parsed.scheme != "https" or parsed.netloc != "github.com":
        errors.append(f"{field} must use https://github.com")
        return ""
    expected_prefix = f"/{repository}/"
    if not parsed.path.startswith(expected_prefix):
        errors.append(f"{field} must belong to {repository}")
        return ""
    if expected_path_prefix is not None:
        suffix = parsed.path[len(expected_prefix) :]
        matches = (
            suffix.startswith(expected_path_prefix)
            if expected_path_prefix.endswith("/")
            else suffix == expected_path_prefix
        )
        if not matches:
            errors.append(f"{field} must point to {expected_path_prefix}")
    return raw


def canonical_contract_payload(
    capabilities: Iterable[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Return the ordered immutable semantic identity for a campaign."""
    payload: list[dict[str, str]] = []
    for index, row in enumerate(capabilities):
        if not isinstance(row, Mapping):
            raise LedgerValidationError(
                [f"capabilities[{index}] must be an object before digesting"]
            )
        payload.append(
            {
                "id": str(row.get("id", "")),
                "name": str(row.get("name", "")),
                "product_state": str(row.get("product_state", "")),
                "source_anchor": str(row.get("source_anchor", "")),
            }
        )
    return payload


def canonical_contract_digest(capabilities: Iterable[Mapping[str, Any]]) -> str:
    payload = canonical_contract_payload(capabilities)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
