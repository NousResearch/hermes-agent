"""Task 29 — lexical artifact path validation (R5-03)."""

from __future__ import annotations

import hashlib
import json
import unicodedata

from htr.advisory_inspection_constants import (
    MAX_COMPONENT_UTF8_BYTES,
    MAX_PATH_COMPONENTS,
    MAX_PATH_UTF8_BYTES,
)

# Lexical stop statuses — L1 not attempted when any of these apply.
PATH_LEXICAL_FATAL = frozenset(
    {
        "path_not_applicable",
        "path_utf8_invalid",
        "path_budget_exceeded",
        "path_control_rejected",
        "path_backslash_rejected",
        "path_empty_rejected",
        "path_absolute_rejected",
        "path_separator_rejected",
        "path_dot_rejected",
        "path_dotdot_rejected",
    }
)


def lexical_validate_artifact_path(declared_path: object) -> tuple[str, tuple[str, ...] | None, list[str]]:
    """Validate manifest ``declared_path`` per R5-03 steps 1–15."""
    findings: list[str] = []

    if not isinstance(declared_path, str):
        return "path_not_applicable", None, findings

    for ch in declared_path:
        cp = ord(ch)
        if 0xD800 <= cp <= 0xDFFF:
            findings.append("path_surrogate_rejected")
            return "path_utf8_invalid", None, findings

    try:
        encoded = declared_path.encode("utf-8")
    except UnicodeEncodeError:
        return "path_utf8_invalid", None, findings

    if len(encoded) > MAX_PATH_UTF8_BYTES:
        return "path_budget_exceeded", None, findings

    for ch in declared_path:
        cp = ord(ch)
        if cp <= 0x1F or cp == 0x7F:
            return "path_control_rejected", None, findings

    if "\\" in declared_path:
        return "path_backslash_rejected", None, findings

    if declared_path == "":
        return "path_empty_rejected", None, findings

    if declared_path.startswith("/"):
        return "path_absolute_rejected", None, findings

    if declared_path.endswith("/"):
        return "path_separator_rejected", None, findings

    if "//" in declared_path:
        return "path_separator_rejected", None, findings

    parts = declared_path.split("/")
    if len(parts) == 0:
        return "path_empty_rejected", None, findings

    if len(parts) > MAX_PATH_COMPONENTS:
        return "path_budget_exceeded", None, findings

    for part in parts:
        if part == ".":
            return "path_dot_rejected", None, findings
        if part == "..":
            return "path_dotdot_rejected", None, findings
        if len(part.encode("utf-8")) > MAX_COMPONENT_UTF8_BYTES:
            return "path_budget_exceeded", None, findings

    if unicodedata.normalize("NFC", declared_path) != declared_path:
        findings.append("path_nfc_not_normalized")

    if parts[0] == "artifacts":
        status = "path_valid_attempt_relative"
    else:
        status = "path_valid_outside_artifacts_dir"

    return status, tuple(parts), findings


def path_identity_digest(
    *,
    run_id: str,
    task_id: str,
    attempt_id: str,
    declared_path: str,
    validated_components: tuple[str, ...],
) -> str:
    """Canonical path identity digest (R5-03)."""
    payload = {
        "attempt_id": attempt_id,
        "declared_path": declared_path,
        "run_id": run_id,
        "schema": "htr.task29.path_identity.v1",
        "task_id": task_id,
        "validated_components": list(validated_components),
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
