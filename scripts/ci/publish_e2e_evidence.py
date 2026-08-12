#!/usr/bin/env python3
"""Publish validated E2E evidence as GitHub attachments and update its PR comment.

This script only runs from the trusted ``workflow_run`` publisher. It never
checks out PR code: it accepts the small evidence artifact produced by the
untrusted E2E workflow, validates its manifest and PNG bytes, uploads the
approved files as GitHub attachments, and replaces the placeholder in the
source PR's CI review comment with those attachment URLs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

API_BASE = "https://api.github.com"
EVIDENCE_START = "<!-- hermes-e2e-evidence:start -->"
EVIDENCE_END = "<!-- hermes-e2e-evidence:end -->"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MAX_FILES = 20
MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_TOTAL_BYTES = 20 * 1024 * 1024
MAX_DIMENSION = 8_000
COMMENT_LOOKUP_ATTEMPTS = 6
COMMENT_LOOKUP_DELAY_SECONDS = 2
_SAFE_FILE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.png$")
_ATTACHMENT_URL = re.compile(r"^!\[[^\]\r\n]*\]\((https://github\.com/user-attachments/assets/[0-9a-fA-F-]+)\)$")


@dataclass(frozen=True)
class EvidenceFile:
    """One validated PNG and the label used when rendering the PR comment."""

    filename: str
    label: str


def _api_request(
    url: str,
    token: str,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Send one authenticated GitHub API request and return its JSON object."""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "hermes-e2e-evidence-publisher",
        },
    )
    with urllib.request.urlopen(request) as response:
        parsed = json.loads(response.read())
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected an object from {url}")
    return parsed


def _read_png(path: Path) -> bytes:
    """Read a bounded PNG, rejecting corrupt and unexpectedly large images."""
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Evidence file is not a regular file: {path.name}")
    size = path.stat().st_size
    if size == 0 or size > MAX_FILE_BYTES:
        raise ValueError(f"Evidence file has invalid size: {path.name}")
    data = path.read_bytes()
    if not data.startswith(PNG_SIGNATURE) or len(data) < 24 or data[12:16] != b"IHDR":
        raise ValueError(f"Evidence file is not a PNG: {path.name}")
    width = int.from_bytes(data[16:20], "big")
    height = int.from_bytes(data[20:24], "big")
    if not 0 < width <= MAX_DIMENSION or not 0 < height <= MAX_DIMENSION:
        raise ValueError(f"Evidence image has invalid dimensions: {path.name}")
    return data


def _manifest_files(manifest: dict[str, Any]) -> list[EvidenceFile]:
    """Flatten a version-one manifest into ordered, reviewer-facing images."""
    if manifest.get("version") != 1:
        raise ValueError("Unsupported E2E evidence manifest version")

    files: list[EvidenceFile] = []
    screenshots = manifest.get("screenshots", [])
    diffs = manifest.get("diffs", [])
    if not isinstance(screenshots, list) or not isinstance(diffs, list):
        raise ValueError("Evidence manifest lists are malformed")

    for entry in screenshots:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str) or not isinstance(entry.get("file"), str):
            raise ValueError("Evidence screenshot entry is malformed")
        files.append(EvidenceFile(entry["file"], f"new screenshot: {entry['name']}"))

    for entry in diffs:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str) or not isinstance(entry.get("diff"), str):
            raise ValueError("Evidence visual-diff entry is malformed")
        files.append(EvidenceFile(entry["diff"], f"visual diff: {entry['name']}"))
        for kind in ("actual", "expected"):
            value = entry.get(kind)
            if value is not None:
                if not isinstance(value, str):
                    raise ValueError("Evidence visual-diff companion is malformed")
                files.append(EvidenceFile(value, f"visual {kind}: {entry['name']}"))

    names = [item.filename for item in files]
    if len(files) > MAX_FILES or len(set(names)) != len(names):
        raise ValueError("Evidence manifest has too many or duplicate files")
    if any(not _SAFE_FILE.fullmatch(name) for name in names):
        raise ValueError("Evidence manifest contains an unsafe filename")
    return files


def load_evidence(evidence_dir: Path) -> tuple[list[EvidenceFile], dict[str, bytes]]:
    """Load the manifest and return only the validated files it declares."""
    manifest_path = evidence_dir / "e2e-evidence.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("E2E evidence manifest is missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("E2E evidence manifest is not JSON") from exc
    if not isinstance(manifest, dict):
        raise ValueError("E2E evidence manifest is not an object")

    files = _manifest_files(manifest)
    payloads: dict[str, bytes] = {}
    total = 0
    for item in files:
        path = evidence_dir / item.filename
        if path.parent != evidence_dir:
            raise ValueError("Evidence file escaped its artifact directory")
        payload = _read_png(path)
        total += len(payload)
        if total > MAX_TOTAL_BYTES:
            raise ValueError("E2E evidence exceeds the total size limit")
        payloads[item.filename] = payload
    return files, payloads


def render_evidence(files: list[EvidenceFile], attachment_urls: dict[str, str]) -> str:
    """Render validated GitHub attachment URLs inside the review-comment marker."""
    blocks = [EVIDENCE_START]
    for item in files:
        url = attachment_urls.get(item.filename)
        if url is None:
            raise ValueError(f"Missing attachment URL for {item.filename}")
        blocks.extend((
            "<details>",
            f"<summary>{item.label}</summary>",
            "",
            f"![{item.label}]({url})",
            "",
            "</details>",
        ))
    blocks.append(EVIDENCE_END)
    return "\n".join(blocks)


def replace_evidence_marker(comment: str, evidence: str) -> str:
    """Replace exactly the pending-evidence region in a CI review comment."""
    pattern = re.compile(f"{re.escape(EVIDENCE_START)}.*?{re.escape(EVIDENCE_END)}", re.DOTALL)
    result, count = pattern.subn(evidence, comment, count=1)
    if count != 1:
        raise ValueError("CI review comment does not contain one evidence marker")
    return result


def _find_review_comment(comments: object) -> dict[str, Any] | None:
    """Find a live CI review comment only after it contains this marker."""
    if not isinstance(comments, list):
        raise ValueError("GitHub comments response is malformed")
    for item in comments:
        if not isinstance(item, dict):
            continue
        body = str(item.get("body", ""))
        if body.startswith("<!-- hermes-ci-review-bot -->") and EVIDENCE_START in body and EVIDENCE_END in body:
            return item
    return None


def _wait_for_review_comment(token: str, source_repo: str, pr_number: str) -> dict[str, Any]:
    """Wait briefly for GitHub's comment API to expose the completed marker."""
    request = urllib.request.Request(
        f"{API_BASE}/repos/{source_repo}/issues/{pr_number}/comments?per_page=100",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "hermes-e2e-evidence-publisher",
        },
    )
    for attempt in range(COMMENT_LOOKUP_ATTEMPTS):
        with urllib.request.urlopen(request) as response:
            comment = _find_review_comment(json.loads(response.read()))
        if comment is not None:
            return comment
        if attempt + 1 < COMMENT_LOOKUP_ATTEMPTS:
            time.sleep(COMMENT_LOOKUP_DELAY_SECONDS)
    raise ValueError("CI review comment with E2E evidence marker is missing")


def upload_evidence(
    files: list[EvidenceFile],
    evidence_dir: Path,
    source_repo: str,
    session_token: str,
) -> dict[str, str]:
    """Upload validated files through gh-image and accept only attachment URLs."""
    environment = os.environ.copy()
    environment["GH_SESSION_TOKEN"] = session_token
    attachment_urls: dict[str, str] = {}
    for item in files:
        result = subprocess.run(
            ["gh", "image", "--repo", source_repo, str(evidence_dir / item.filename)],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        match = _ATTACHMENT_URL.fullmatch(result.stdout.strip())
        if match is None:
            raise ValueError(f"gh-image returned an invalid attachment reference for {item.filename}")
        attachment_urls[item.filename] = match.group(1)
    return attachment_urls


def publish(
    token: str,
    source_repo: str,
    evidence_dir: Path,
    pr_number: str,
    session_token: str,
) -> bool:
    """Publish evidence and patch its source PR comment; false means nothing to show."""
    files, _ = load_evidence(evidence_dir)
    if not files:
        print("No inline E2E evidence to publish.")
        return False
    comment = _wait_for_review_comment(token, source_repo, pr_number)
    attachment_urls = upload_evidence(files, evidence_dir, source_repo, session_token)
    evidence = render_evidence(files, attachment_urls)
    body = replace_evidence_marker(str(comment.get("body", "")), evidence)
    _api_request(
        f"{API_BASE}/repos/{source_repo}/issues/comments/{comment['id']}",
        token,
        method="PATCH",
        payload={"body": body},
    )
    print(f"Published {len(files)} E2E evidence image attachment(s).")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--pr-number", required=True)
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        parser.error("GITHUB_TOKEN is required")
    session_token = os.environ.get("GH_SESSION_TOKEN", "")
    if not session_token:
        parser.error("GH_SESSION_TOKEN is required")
    publish(token, args.source_repo, args.evidence_dir, args.pr_number, session_token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
