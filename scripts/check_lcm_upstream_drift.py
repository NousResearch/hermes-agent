#!/usr/bin/env python3
"""Validate LCM vendoring provenance and report upstream security drift.

Offline mode validates the structured metadata in VENDORED_FROM.txt only. Online
mode also asks upstream Git/GitHub for current refs and compare metadata, then
reports drift for Apollo owner action. The checker is read-only and never updates
vendored code.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

SECTION_MARKER = "PRD-6 I-10 upstream security-drift metadata:"
REQUIRED_FIELDS = (
    "source_repository",
    "source_commit",
    "vendored_commit",
    "vendored_version",
    "ingest_audit_verdict",
    "last_upstream_security_check",
    "checked_by",
    "next_check_due",
)
SECURITY_KEYWORDS = (
    "auth",
    "credential",
    "cve",
    "injection",
    "permission",
    "redact",
    "secret",
    "security",
    "token",
    "vuln",
)
FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True)
class LcmMetadata:
    path: Path
    fields: dict[str, str]

    @property
    def source_repository(self) -> str:
        return self.fields["source_repository"]

    @property
    def source_commit(self) -> str:
        return self.fields["source_commit"]

    @property
    def vendored_commit(self) -> str:
        return self.fields["vendored_commit"]

    @property
    def vendored_version(self) -> str:
        return self.fields["vendored_version"]

    @property
    def ingest_audit_verdict(self) -> str:
        return self.fields["ingest_audit_verdict"]

    @property
    def last_upstream_security_check(self) -> str:
        return self.fields["last_upstream_security_check"]

    @property
    def checked_by(self) -> str:
        return self.fields["checked_by"]

    @property
    def next_check_due(self) -> str:
        return self.fields["next_check_due"]

    @property
    def checked_upstream_head(self) -> str | None:
        return self.fields.get("checked_upstream_head")

    @property
    def checked_upstream_tag(self) -> str | None:
        return self.fields.get(f"checked_upstream_tag_{self.vendored_version}")


@dataclass(frozen=True)
class UpstreamCommit:
    sha: str
    message: str

    @property
    def short_sha(self) -> str:
        return self.sha[:12]

    @property
    def security_relevant(self) -> bool:
        folded = self.message.lower()
        return any(keyword in folded for keyword in SECURITY_KEYWORDS)


@dataclass(frozen=True)
class UpstreamReport:
    head_commit: str | None
    tag_commit: str | None
    commits: list[UpstreamCommit]
    lookup_warnings: list[str]


@dataclass(frozen=True)
class CheckResult:
    status: str
    messages: list[str]

    @property
    def exit_code(self) -> int:
        if self.status == "FAIL":
            return 1
        return 0


def parse_metadata(path: str | Path) -> LcmMetadata:
    metadata_path = Path(path).expanduser().resolve()
    text = metadata_path.read_text(encoding="utf-8")
    fields = _parse_metadata_section(text, metadata_path)
    missing = [field for field in REQUIRED_FIELDS if not fields.get(field)]
    if missing:
        raise ValueError(f"{metadata_path} missing required metadata field(s): {', '.join(missing)}")
    _validate_metadata_shapes(fields, metadata_path)
    return LcmMetadata(metadata_path, fields)


def _parse_metadata_section(text: str, metadata_path: Path) -> dict[str, str]:
    lines = text.splitlines()
    section_start = next((i for i, line in enumerate(lines) if line.strip() == SECTION_MARKER), None)
    if section_start is None:
        raise ValueError(f"{metadata_path} missing metadata section: {SECTION_MARKER}")

    fields: dict[str, str] = {}
    for line in lines[section_start + 1 :]:
        if not line.strip():
            continue
        if not line.startswith((" ", "\t")):
            break
        key, sep, value = line.strip().partition(":")
        if not sep:
            continue
        fields[key.strip()] = value.strip()
    return fields


def _validate_metadata_shapes(fields: dict[str, str], metadata_path: Path) -> None:
    for field in ("source_commit", "vendored_commit"):
        value = fields[field]
        if not FULL_SHA_RE.fullmatch(value):
            raise ValueError(f"{metadata_path} field {field} must be a full 40-character git SHA")
    for field in ("last_upstream_security_check", "next_check_due"):
        value = fields[field]
        if not DATE_RE.fullmatch(value):
            raise ValueError(f"{metadata_path} field {field} must use YYYY-MM-DD")
        datetime.strptime(value, "%Y-%m-%d")
    if fields["ingest_audit_verdict"].upper() not in {"PASS", "WARN", "FAIL"}:
        raise ValueError(f"{metadata_path} field ingest_audit_verdict must be PASS, WARN, or FAIL")
    if not fields["vendored_version"].startswith("v"):
        raise ValueError(f"{metadata_path} field vendored_version must be a tag-like value such as v0.16.2")
    if fields["source_repository"] != "github.com/stephenschoettler/hermes-lcm":
        raise ValueError(
            f"{metadata_path} field source_repository must remain github.com/stephenschoettler/hermes-lcm"
        )


def run_check(metadata_path: str | Path, *, offline: bool = False) -> CheckResult:
    metadata = parse_metadata(metadata_path)
    messages = [
        f"PASS: metadata fields valid for {metadata.source_repository} "
        f"{metadata.vendored_version} ({metadata.vendored_commit[:12]})"
    ]
    if offline:
        messages.append("PASS: offline metadata validation only; upstream network checks skipped")
        return CheckResult("PASS", messages)

    upstream = query_upstream(metadata)
    status = "PASS"
    for warning in upstream.lookup_warnings:
        messages.append(f"WARN: {warning}")
        status = "WARN"

    if upstream.head_commit:
        if upstream.head_commit == metadata.vendored_commit:
            messages.append("PASS: upstream HEAD matches vendored commit")
        else:
            messages.append(
                "WARN: upstream HEAD differs from vendored commit "
                f"(vendored {metadata.vendored_commit[:12]}, head {upstream.head_commit[:12]}); "
                "Apollo must review drift before updating vendored code"
            )
            status = "WARN"
    if upstream.tag_commit:
        expected_tag = metadata.checked_upstream_tag
        if expected_tag and upstream.tag_commit != expected_tag:
            messages.append(
                "WARN: upstream version tag ref changed since last recorded check "
                f"({metadata.vendored_version}: recorded {expected_tag[:12]}, current {upstream.tag_commit[:12]})"
            )
            status = "WARN"
        else:
            messages.append(f"PASS: upstream {metadata.vendored_version} tag ref matches recorded metadata")

    security_commits = [commit for commit in upstream.commits if commit.security_relevant]
    if security_commits:
        status = "WARN"
        for commit in security_commits:
            messages.append(f"WARN: security-relevant upstream commit {commit.short_sha}: {commit.message}")
    elif upstream.commits:
        messages.append("PASS: upstream compare returned no security-keyword commits")
    return CheckResult(status, messages)


def query_upstream(metadata: LcmMetadata) -> UpstreamReport:
    warnings: list[str] = []
    head_commit: str | None = None
    tag_commit: str | None = None
    refs = _git_ls_remote(metadata)
    if isinstance(refs, str):
        warnings.append(refs)
    else:
        head_commit = refs.get("HEAD")
        tag_commit = refs.get(f"refs/tags/{metadata.vendored_version}") or refs.get(
            f"refs/tags/{metadata.vendored_version}^{{}}"
        )
        if head_commit is None:
            warnings.append("upstream HEAD ref was not returned by git ls-remote")
        if tag_commit is None:
            warnings.append(f"upstream tag {metadata.vendored_version} was not returned by git ls-remote")

    commits: list[UpstreamCommit] = []
    if head_commit and head_commit != metadata.vendored_commit:
        compare = _github_compare(metadata, head_commit)
        if isinstance(compare, str):
            warnings.append(compare)
        else:
            commits = compare
    return UpstreamReport(head_commit, tag_commit, commits, warnings)


def _git_ls_remote(metadata: LcmMetadata) -> dict[str, str] | str:
    repo_url = f"https://{metadata.source_repository}.git"
    wanted_refs = ["HEAD", f"refs/tags/{metadata.vendored_version}", f"refs/tags/{metadata.vendored_version}^{{}}"]
    try:
        proc = subprocess.run(
            ["git", "ls-remote", repo_url, *wanted_refs],
            check=False,
            capture_output=True,
            encoding="utf-8",
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"upstream lookup unavailable: {exc}"
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip() or f"git ls-remote exited {proc.returncode}"
        return f"upstream lookup unavailable: {detail}"
    refs: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        sha, _, ref = line.partition("\t")
        if sha and ref:
            refs[ref] = sha
    return refs


def _github_compare(metadata: LcmMetadata, head_commit: str) -> list[UpstreamCommit] | str:
    owner_repo = metadata.source_repository
    prefix = "github.com/"
    if owner_repo.startswith(prefix):
        owner_repo = owner_repo[len(prefix) :]
    url = f"https://api.github.com/repos/{owner_repo}/compare/{metadata.vendored_commit}...{head_commit}"
    request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "hermes-lcm-drift-check"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        return f"GitHub compare unavailable: {exc}"
    commits = payload.get("commits")
    if not isinstance(commits, list):
        return "GitHub compare response did not include a commits list"
    parsed: list[UpstreamCommit] = []
    for commit in commits:
        if not isinstance(commit, dict):
            continue
        sha = str(commit.get("sha", ""))
        commit_data = commit.get("commit", {})
        message = ""
        if isinstance(commit_data, dict):
            message = str(commit_data.get("message", "")).splitlines()[0]
        if sha and message:
            parsed.append(UpstreamCommit(sha=sha, message=message))
    return parsed


def print_result(result: CheckResult) -> None:
    for message in result.messages:
        print(message)
    print(f"status={result.status}")


def _parse_args(argv: Iterable[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, help="Path to plugins/context_engine/lcm/VENDORED_FROM.txt")
    parser.add_argument("--offline", action="store_true", help="Validate metadata only; skip upstream network checks")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = run_check(args.metadata, offline=args.offline)
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    print_result(result)
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
