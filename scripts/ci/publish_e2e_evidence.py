#!/usr/bin/env python3
"""Publish validated E2E evidence to a throwaway branch and update its PR comment.

This script only runs from the trusted ``workflow_run`` publisher. It never
checks out PR code: it accepts the small evidence artifact produced by the
untrusted E2E workflow, validates its manifest and PNG bytes, commits the
approved files to a run-scoped branch in the public evidence repository, and
replaces the placeholder in the source PR's CI review comment with raw URLs
pinned to that commit.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import time
import urllib.error
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


def render_evidence(files: list[EvidenceFile], evidence_repo: str, commit_sha: str) -> str:
    """Render commit-pinned raw images inside the review-comment marker."""
    blocks = [EVIDENCE_START]
    for item in files:
        url = f"https://raw.githubusercontent.com/{evidence_repo}/{commit_sha}/{item.filename}"
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


def _create_commit(
    token: str,
    evidence_repo: str,
    run_id: str,
    files: dict[str, bytes],
) -> str:
    """Create a run-scoped branch containing evidence via Git's object API."""
    repo_url = f"{API_BASE}/repos/{evidence_repo}"
    main_ref = _api_request(f"{repo_url}/git/ref/heads/main", token)
    base_sha = main_ref["object"]["sha"]
    base_commit = _api_request(f"{repo_url}/git/commits/{base_sha}", token)
    base_tree = base_commit["tree"]["sha"]

    tree_entries = []
    for filename, payload in files.items():
        blob = _api_request(
            f"{repo_url}/git/blobs",
            token,
            method="POST",
            payload={"content": base64.b64encode(payload).decode("ascii"), "encoding": "base64"},
        )
        tree_entries.append({"path": filename, "mode": "100644", "type": "blob", "sha": blob["sha"]})

    tree = _api_request(
        f"{repo_url}/git/trees",
        token,
        method="POST",
        payload={"base_tree": base_tree, "tree": tree_entries},
    )
    commit = _api_request(
        f"{repo_url}/git/commits",
        token,
        method="POST",
        payload={
            "message": f"ci: publish e2e evidence for run {run_id}",
            "tree": tree["sha"],
            "parents": [base_sha],
        },
    )
    branch = f"run-{run_id}"
    branch_url = f"{repo_url}/git/ref/heads/{branch}"
    try:
        _api_request(branch_url, token)
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
        _api_request(
            f"{repo_url}/git/refs",
            token,
            method="POST",
            payload={"ref": f"refs/heads/{branch}", "sha": commit["sha"]},
        )
    else:
        _api_request(branch_url, token, method="PATCH", payload={"sha": commit["sha"], "force": True})
    return str(commit["sha"])


def publish(
    token: str,
    source_repo: str,
    evidence_repo: str,
    evidence_dir: Path,
    run_id: str,
    pr_number: str,
) -> bool:
    """Publish evidence and patch its source PR comment; false means nothing to show."""
    files, payloads = load_evidence(evidence_dir)
    if not files:
        print("No inline E2E evidence to publish.")
        return False
    comment = _wait_for_review_comment(token, source_repo, pr_number)
    commit_sha = _create_commit(token, evidence_repo, run_id, payloads)
    evidence = render_evidence(files, evidence_repo, commit_sha)
    body = replace_evidence_marker(str(comment.get("body", "")), evidence)
    _api_request(
        f"{API_BASE}/repos/{source_repo}/issues/comments/{comment['id']}",
        token,
        method="PATCH",
        payload={"body": body},
    )
    print(f"Published {len(files)} E2E evidence image(s) at {commit_sha}.")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--evidence-repo", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--pr-number", required=True)
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        parser.error("GITHUB_TOKEN is required")
    publish(token, args.source_repo, args.evidence_repo, args.evidence_dir, args.run_id, args.pr_number)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
