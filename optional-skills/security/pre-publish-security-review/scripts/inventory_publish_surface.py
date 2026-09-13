#!/usr/bin/env python3
"""Inventory a publish directory for deterministic secret-exposure risks.

The script reports paths, line numbers, and rule labels only. It never emits a
matched value or source snippet. It is intentionally a narrow preflight, not a
replacement for SAST or manual review.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

DEFAULT_MAX_FILE_BYTES = 2_000_000
VCS_DIRECTORIES = {".git", ".hg", ".svn"}
CREDENTIAL_DIRECTORIES = {".aws", ".gnupg", ".ssh", ".wrangler"}
SAFE_ENV_SUFFIXES = (".example", ".sample", ".template")
SENSITIVE_EXACT_NAMES = {
    ".dev.vars",
    ".envrc",
    ".netrc",
    ".npmrc",
    ".pypirc",
    ".secrets",
    "auth.json",
    "credentials.json",
    "credentials.yaml",
    "credentials.yml",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa",
    "secrets.json",
    "secrets.yaml",
    "secrets.yml",
}
SENSITIVE_SUFFIXES = (
    ".jks",
    ".key",
    ".kdbx",
    ".keystore",
    ".p12",
    ".pem",
    ".pfx",
)


@dataclass(frozen=True)
class Signature:
    rule: str
    pattern: re.Pattern[str]
    blocking: bool


SIGNATURES = (
    Signature(
        "private-key-header",
        re.compile(r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----"),
        True,
    ),
    Signature(
        "github-token",
        re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
        True,
    ),
    Signature("aws-access-key-id", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"), True),
    Signature("google-api-key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"), True),
    Signature("slack-token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"), True),
    Signature("stripe-live-key", re.compile(r"\b[rs]k_live_[A-Za-z0-9]{16,}\b"), True),
    Signature(
        "credential-in-url",
        re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s/:@]+:[^\s/@]+@", re.IGNORECASE),
        True,
    ),
    Signature(
        "literal-credential-assignment",
        re.compile(
            r"(?im)\b(?:api[_-]?key|client[_-]?secret|password|passwd|secret|token)"
            r"\s*[:=]\s*['\"][^'\"\r\n]{8,}['\"]"
        ),
        False,
    ),
)


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _is_sensitive_path(relative_path: str, *, is_directory: bool = False) -> str | None:
    path = Path(relative_path)
    name = path.name.lower()

    if name in VCS_DIRECTORIES:
        return "version-control-metadata"
    if is_directory and name in CREDENTIAL_DIRECTORIES:
        return "credential-directory"
    if (
        name == ".env"
        or name.endswith(".env")
        or (name.startswith("env.") and not name.endswith(SAFE_ENV_SUFFIXES))
        or (name.startswith(".env.") and not name.endswith(SAFE_ENV_SUFFIXES))
    ):
        return "environment-file"
    if name in SENSITIVE_EXACT_NAMES:
        return "credential-file"
    if name.endswith(SENSITIVE_SUFFIXES):
        return "private-key-or-keystore"
    if "service-account" in name and name.endswith(".json"):
        return "service-account-file"
    return None


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _iter_files(root: Path) -> tuple[list[Path], list[dict[str, object]]]:
    files: list[Path] = []
    path_findings: list[dict[str, object]] = []

    root_rule = _is_sensitive_path(root.name, is_directory=True)
    if root_rule:
        path_findings.append({"path": ".", "rule": root_rule})

    def on_walk_error(error: OSError) -> None:
        relative = "."
        if error.filename:
            try:
                relative = Path(error.filename).relative_to(root).as_posix()
            except ValueError:
                pass
        display_path = f"{relative}/" if relative != "." else relative
        path_findings.append({"path": display_path, "rule": "directory-read-error"})

    for current, dirnames, filenames in os.walk(
        root,
        topdown=True,
        followlinks=False,
        onerror=on_walk_error,
    ):
        current_path = Path(current)
        kept_dirs: list[str] = []
        for dirname in sorted(dirnames):
            full_path = current_path / dirname
            relative = _relative(full_path, root)
            rule = _is_sensitive_path(relative, is_directory=True)
            if rule:
                path_findings.append({"path": f"{relative}/", "rule": rule})
                continue
            if full_path.is_symlink():
                path_findings.append({"path": relative, "rule": "symlink-directory"})
                continue
            kept_dirs.append(dirname)
        dirnames[:] = kept_dirs

        for filename in sorted(filenames):
            path = current_path / filename
            relative = _relative(path, root)
            if path.is_symlink():
                path_findings.append({"path": relative, "rule": "symlink-file"})
                continue
            if path.is_file():
                files.append(path)
            else:
                path_findings.append({"path": relative, "rule": "non-regular-file"})

    files.sort(key=lambda path: _relative(path, root))
    path_findings.sort(key=lambda finding: (str(finding["path"]), str(finding["rule"])))
    return files, path_findings


def _scan_text(
    text: str,
    *,
    relative_path: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    blockers: list[dict[str, object]] = []
    review: list[dict[str, object]] = []

    for signature in SIGNATURES:
        for match in signature.pattern.finditer(text):
            finding: dict[str, object] = {
                "path": relative_path,
                "line": _line_number(text, match.start()),
                "rule": signature.rule,
            }
            (blockers if signature.blocking else review).append(finding)

    return blockers, review


def inventory(root: Path, max_file_bytes: int) -> dict[str, Any]:
    root = root.resolve()
    files, path_findings = _iter_files(root)
    secret_candidates: list[dict[str, object]] = []
    review_candidates: list[dict[str, object]] = []
    skipped_files: list[dict[str, object]] = []
    scanned_files = 0

    for path in files:
        relative = _relative(path, root)
        path_rule = _is_sensitive_path(relative)
        if path_rule:
            path_findings.append({"path": relative, "rule": path_rule})

        if path.suffix.lower() == ".map":
            review_candidates.append({"path": relative, "rule": "source-map"})

        try:
            with path.open("rb") as handle:
                data = handle.read(max_file_bytes + 1)
        except OSError:
            skipped_files.append({"path": relative, "reason": "read-error"})
            continue

        if len(data) > max_file_bytes:
            skipped_files.append({"path": relative, "reason": "size-limit"})
            review_candidates.append({"path": relative, "rule": "unscanned-oversize-file"})
            continue
        if b"\x00" in data[:8192]:
            skipped_files.append({"path": relative, "reason": "binary"})
            continue

        scanned_files += 1
        text = data.decode("utf-8", errors="replace")
        blockers, review = _scan_text(text, relative_path=relative)
        secret_candidates.extend(blockers)
        review_candidates.extend(review)

    path_findings.sort(key=lambda finding: (str(finding["path"]), str(finding["rule"])))
    def finding_key(finding: dict[str, object]) -> tuple[str, int, str]:
        line = finding.get("line", 0)
        return (
            str(finding["path"]),
            line if isinstance(line, int) else 0,
            str(finding["rule"]),
        )

    secret_candidates.sort(key=finding_key)
    review_candidates.sort(key=finding_key)
    skipped_files.sort(key=lambda finding: (str(finding["path"]), str(finding["reason"])))

    blockers = len(path_findings) + len(secret_candidates)
    return {
        "root": str(root),
        "summary": {
            "files_discovered": len(files),
            "files_scanned_as_text": scanned_files,
            "blocking_findings": blockers,
            "review_findings": len(review_candidates),
            "skipped_files": len(skipped_files),
        },
        "sensitive_paths": path_findings,
        "secret_candidates": secret_candidates,
        "review_candidates": review_candidates,
        "skipped_files": skipped_files,
        "decision": "HOLD" if blockers else "REVIEW",
        "note": (
            "Locations and rule labels only; matched values are never emitted. "
            "Zero blockers is not proof that the artifact is secure."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inventory a publish directory without printing matched secret values."
    )
    parser.add_argument("root", type=Path, help="Exact directory that will be published")
    parser.add_argument(
        "--max-file-bytes",
        type=int,
        default=DEFAULT_MAX_FILE_BYTES,
        help=f"Maximum bytes scanned per text file (default: {DEFAULT_MAX_FILE_BYTES})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_file_bytes < 1:
        print("error: --max-file-bytes must be positive", file=sys.stderr)
        return 2
    if not args.root.is_dir():
        print(f"error: publish directory does not exist: {args.root}", file=sys.stderr)
        return 2

    report = inventory(args.root, args.max_file_bytes)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report["summary"]["blocking_findings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
