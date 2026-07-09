"""Artifact-first execution contracts and host-visible artifact checks."""

from __future__ import annotations

import hashlib
import os
import re
import sys
import zipfile
from pathlib import Path
from typing import Any, Iterable

_PATH_RE = re.compile(
    r"(?P<path>(?:[A-Za-z]:[\\/]|\\\\[A-Za-z][\\/]|/[A-Za-z0-9_.~-])[^\r\n\"'`<>]*)"
)
_REQUIRED_HINT_RE = re.compile(
    r"\b(required|output\s+file|final\s+report|final\s+synthesis|write|save|artifact|deliverable)\b",
    re.IGNORECASE,
)
_TRAILING_PATH_CHARS = " \t.,;:)]}"


def _clean_path_text(value: Any) -> str:
    text = str(value or "").strip().strip("'\"")
    return text.rstrip(_TRAILING_PATH_CHARS)


def is_windows_drive_alias_path(path_text: str) -> bool:
    """Return True for noncanonical aliases like ``\\d\\file.zip``."""

    text = str(path_text or "").strip()
    return bool(re.match(r"^\\\\[A-Za-z]\\\\", text) or re.match(r"^\\[A-Za-z]\\", text))


def _is_absolute_for_platform(path_text: str, *, platform: str | None = None) -> bool:
    platform_name = platform or sys.platform
    text = str(path_text or "").strip()
    if platform_name.startswith("win"):
        return bool(re.match(r"^[A-Za-z]:[\\/]", text)) and not is_windows_drive_alias_path(text)
    return Path(text).expanduser().is_absolute() and not is_windows_drive_alias_path(text)


def detect_required_artifacts(text: Any) -> list[dict[str, Any]]:
    """Detect explicit required artifact/output paths in prompts or Kanban cards."""

    seen: set[str] = set()
    contracts: list[dict[str, Any]] = []
    for line in str(text or "").splitlines():
        if not _REQUIRED_HINT_RE.search(line):
            continue
        for match in _PATH_RE.finditer(line):
            path_text = _clean_path_text(match.group("path"))
            if not path_text or path_text in seen:
                continue
            seen.add(path_text)
            contracts.append({"path": path_text, "required": True})
    return contracts


def write_required_artifacts(prompt: Any, draft_content: Any) -> list[dict[str, Any]]:
    """Write/update required artifact drafts named by ``prompt`` using useful stdout text."""

    draft = str(draft_content or "").strip()
    results: list[dict[str, Any]] = []
    for contract in detect_required_artifacts(prompt):
        path_text = contract["path"]
        result = {"path": path_text, "required": True, "status": "missing"}
        if is_windows_drive_alias_path(path_text):
            result["status"] = "noncanonical_path"
            results.append(result)
            continue
        if not draft:
            results.append(result)
            continue
        try:
            target = Path(path_text).expanduser()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(draft + ("\n" if not draft.endswith("\n") else ""), encoding="utf-8")
            if target.exists() and target.stat().st_size > 0:
                result["status"] = "written"
        except OSError:
            result["status"] = "write_failed"
        results.append(result)
    return results


def required_artifact_guard_action(
    *,
    prompt: Any,
    stdout_draft: Any,
    turns_remaining: int,
) -> dict[str, Any]:
    """Return the next artifact-first action near a turn/tool budget boundary."""

    contracts = detect_required_artifacts(prompt)
    if not contracts:
        return {"action": "continue"}
    draft = str(stdout_draft or "").strip()
    if int(turns_remaining) <= 1 and draft:
        return {"action": "write_required_artifact", "path": contracts[0]["path"]}
    return {"action": "continue", "path": contracts[0]["path"]}


def _issue(code: str, message: str, *, path: str | None = None, item: str | None = None) -> dict[str, Any]:
    issue: dict[str, Any] = {"code": code, "message": message}
    if path:
        issue["path"] = path
    if item:
        issue["item"] = item
    return issue


def _readable_by_runtime(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            handle.read(1)
        return True
    except OSError:
        return False


def verify_deploy_artifact_contract(
    *,
    artifact_path: str | os.PathLike[str],
    expected_sha256: str | None = None,
    required_entries: Iterable[str] | None = None,
    platform: str | None = None,
) -> dict[str, Any]:
    """Verify deploy/package artifact metadata against the host-visible file."""

    raw_path = _clean_path_text(artifact_path)
    required = [str(entry).replace("\\", "/").strip() for entry in (required_entries or []) if str(entry).strip()]
    issues: list[dict[str, Any]] = []
    checked: dict[str, Any] = {
        "path": raw_path,
        "canonical_path": raw_path,
        "byte_length": 0,
        "sha256": "",
        "required_entries": required,
        "runtime_can_read": False,
    }

    if not raw_path:
        issues.append(_issue("artifact_path_missing", "artifact path is required"))
        return {"ok": False, "issues": issues, "checked": checked}

    alias_path = is_windows_drive_alias_path(raw_path)
    if alias_path or not _is_absolute_for_platform(raw_path, platform=platform):
        issues.append(_issue(
            "noncanonical_path",
            "artifact path must be a canonical absolute path for the upload host",
            path=raw_path,
        ))

    if alias_path:
        issues.append(_issue(
            "host_visible_missing",
            "artifact path is not host-visible to this runtime",
            path=raw_path,
        ))
        return {"ok": False, "issues": issues, "checked": checked}

    path = Path(raw_path).expanduser()
    try:
        canonical = path.resolve(strict=False)
    except OSError:
        canonical = path.absolute()
    checked["canonical_path"] = str(canonical)

    if not path.exists():
        issues.append(_issue(
            "host_visible_missing",
            "artifact file does not exist on the host-visible filesystem",
            path=str(path),
        ))
        return {"ok": False, "issues": issues, "checked": checked}
    if not path.is_file():
        issues.append(_issue("artifact_not_file", "artifact path is not a file", path=str(path)))
        return {"ok": False, "issues": issues, "checked": checked}

    try:
        data = path.read_bytes()
    except OSError:
        data = b""
    checked["byte_length"] = len(data)
    checked["runtime_can_read"] = _readable_by_runtime(path)
    if len(data) <= 0:
        issues.append(_issue("artifact_empty", "artifact file must be non-empty", path=str(path)))
    if not checked["runtime_can_read"]:
        issues.append(_issue("runtime_cannot_read", "upload runtime cannot read artifact path", path=str(path)))

    actual_hash = hashlib.sha256(data).hexdigest() if data else ""
    checked["sha256"] = actual_hash
    if expected_sha256 and actual_hash.lower() != str(expected_sha256).strip().lower():
        issues.append(_issue("sha256_mismatch", "artifact sha256 does not match expected hash", path=str(path)))

    if required:
        try:
            with zipfile.ZipFile(path) as zf:
                names = {name.replace("\\", "/") for name in zf.namelist()}
        except zipfile.BadZipFile:
            names = set()
            issues.append(_issue("zip_unreadable", "artifact is not a readable zip archive", path=str(path)))
        for entry in required:
            if entry not in names:
                issues.append(_issue("zip_entry_missing", "required deploy archive entry is missing", path=str(path), item=entry))

    return {"ok": not issues, "issues": issues, "checked": checked}
