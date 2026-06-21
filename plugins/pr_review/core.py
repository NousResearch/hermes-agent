"""Core helpers for the Hermes PR reviewer plugin."""

from __future__ import annotations

import base64
import fnmatch
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from hermes_constants import get_hermes_home


_PR_URL_RE = re.compile(r"https://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/pull/(?P<number>\d+)(?:\b|/)?")
_PR_SHORT_RE = re.compile(r"^(?P<owner>[^/\s#]+)/(?P<repo>[^/\s#]+)#(?P<number>\d+)$")

DEFAULT_DOC_PATHS = (
    "AGENTS.md",
    "CLAUDE.md",
    ".cursorrules",
    "README.md",
    "CONTRIBUTING.md",
    "docs/ARCHITECTURE.md",
    "docs/WORKFLOW.md",
    ".github/copilot-instructions.md",
)

DEFAULT_IGNORE_PATTERNS = (
    "**/package-lock.json",
    "**/pnpm-lock.yaml",
    "**/yarn.lock",
    "**/dist/**",
    "**/build/**",
    "**/generated/**",
    "**/*.min.js",
    "**/vendor/**",
)


@dataclass(frozen=True)
class PullRequestRef:
    owner: str
    repo: str
    number: int

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.repo}"

    @property
    def storage_name(self) -> str:
        return f"{self.owner}_{self.repo}".replace("/", "_")


def parse_pr_ref(raw: str) -> PullRequestRef:
    """Parse a GitHub PR URL or ``owner/repo#123`` reference."""
    value = (raw or "").strip()
    match = _PR_URL_RE.match(value) or _PR_SHORT_RE.match(value)
    if not match:
        raise ValueError("PR must be a GitHub URL or owner/repo#number")
    return PullRequestRef(
        owner=match.group("owner"),
        repo=match.group("repo"),
        number=int(match.group("number")),
    )


def artifacts_root() -> Path:
    return Path(get_hermes_home()) / "pr-reviewer" / "reviews"


def run_gh(args: Sequence[str], *, input_text: str | None = None, timeout: int = 120) -> str:
    """Run ``gh`` and return stdout, raising a useful RuntimeError on failure."""
    proc = subprocess.run(
        ["gh", *args],
        input=input_text,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        detail = stderr or stdout or f"exit {proc.returncode}"
        raise RuntimeError(f"gh {' '.join(args)} failed: {detail}")
    return proc.stdout


def run_gh_json(args: Sequence[str], *, timeout: int = 120) -> Any:
    return json.loads(run_gh(args, timeout=timeout) or "null")


def fetch_pr_metadata(ref: PullRequestRef) -> Dict[str, Any]:
    fields = [
        "number",
        "title",
        "body",
        "author",
        "url",
        "baseRefName",
        "headRefName",
        "headRefOid",
        "baseRefOid",
        "isDraft",
        "mergeStateStatus",
        "additions",
        "deletions",
        "changedFiles",
        "labels",
    ]
    return run_gh_json([
        "pr",
        "view",
        str(ref.number),
        "--repo",
        ref.full_name,
        "--json",
        ",".join(fields),
    ])


def fetch_pr_diff(ref: PullRequestRef) -> str:
    return run_gh(["pr", "diff", str(ref.number), "--repo", ref.full_name], timeout=180)


def fetch_pr_files(ref: PullRequestRef) -> List[Dict[str, Any]]:
    data = run_gh_json([
        "api",
        f"repos/{ref.full_name}/pulls/{ref.number}/files",
        "--paginate",
    ])
    return data if isinstance(data, list) else []


def _api_get_json(path: str) -> Any:
    return run_gh_json(["api", path], timeout=60)


def fetch_file_from_base(ref: PullRequestRef, path: str, base_ref: str) -> Optional[str]:
    """Fetch a file from the trusted base branch via GitHub Contents API."""
    api_path = f"repos/{ref.full_name}/contents/{path}?ref={base_ref}"
    try:
        data = _api_get_json(api_path)
    except Exception:
        return None
    if not isinstance(data, dict) or data.get("type") != "file":
        return None
    encoded = data.get("content") or ""
    try:
        return base64.b64decode(encoded, validate=False).decode("utf-8", errors="replace")
    except Exception:
        return None


def fetch_instruction_glob_from_base(ref: PullRequestRef, base_ref: str) -> Dict[str, str]:
    """Fetch .github/instructions/*.instructions.md from base branch when present."""
    out: Dict[str, str] = {}
    try:
        entries = _api_get_json(f"repos/{ref.full_name}/contents/.github/instructions?ref={base_ref}")
    except Exception:
        return out
    if not isinstance(entries, list):
        return out
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "")
        path = str(entry.get("path") or "")
        if name.endswith(".instructions.md") and path:
            text = fetch_file_from_base(ref, path, base_ref)
            if text:
                out[path] = text
    return out


def collect_trusted_docs(ref: PullRequestRef, base_ref: str, *, max_chars_per_doc: int = 20_000) -> Dict[str, str]:
    docs: Dict[str, str] = {}
    for path in DEFAULT_DOC_PATHS:
        text = fetch_file_from_base(ref, path, base_ref)
        if text:
            docs[path] = text[:max_chars_per_doc]
    docs.update(fetch_instruction_glob_from_base(ref, base_ref))
    return docs


def is_ignored_path(path: str, patterns: Iterable[str] = DEFAULT_IGNORE_PATTERNS) -> bool:
    # Match both raw path and a fake root-prefixed path so **/foo patterns also
    # cover root-level files with Python's fnmatch semantics.
    candidates = (path, f"root/{path}")
    return any(fnmatch.fnmatchcase(candidate, pattern) for pattern in patterns for candidate in candidates)


def filter_files(files: Iterable[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    included: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for item in files:
        filename = str(item.get("filename") or "")
        if filename and is_ignored_path(filename):
            skipped.append({**item, "skip_reason": "ignored_path"})
        else:
            included.append(item)
    return included, skipped


def truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars] + "\n\n[TRUNCATED by Hermes PR Reviewer context budget]\n", True


def artifact_dir(ref: PullRequestRef, head_sha: str) -> Path:
    safe_sha = (head_sha or "unknown")[:12]
    return artifacts_root() / ref.storage_name / str(ref.number) / safe_sha


def review_schema() -> Dict[str, Any]:
    finding = {
        "type": "object",
        "properties": {
            "severity": {"type": "string", "enum": ["critical", "warning", "suggestion"]},
            "path": {"type": "string"},
            "line": {"type": ["integer", "null"]},
            "title": {"type": "string"},
            "evidence": {"type": "string"},
            "why_it_matters": {"type": "string"},
            "suggested_fix": {"type": "string"},
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            "fingerprint_hint": {"type": "string"},
        },
        "required": ["severity", "path", "title", "evidence", "why_it_matters", "suggested_fix", "confidence"],
    }
    return {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["approve", "comment", "request_changes"]},
            "risk": {"type": "string", "enum": ["low", "medium", "high"]},
            "summary": {"type": "string"},
            "findings": {"type": "array", "items": finding},
            "verification_notes": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["verdict", "risk", "summary", "findings", "verification_notes"],
    }


def build_review_input(
    *,
    metadata: Dict[str, Any],
    diff: str,
    docs: Dict[str, str],
    included_files: List[Dict[str, Any]],
    skipped_files: List[Dict[str, Any]],
    max_diff_chars: int,
) -> tuple[str, Dict[str, Any]]:
    clipped_diff, diff_truncated = truncate_text(diff, max_diff_chars)
    manifest = {
        "repo": metadata.get("url", ""),
        "number": metadata.get("number"),
        "title": metadata.get("title"),
        "base_ref": metadata.get("baseRefName"),
        "head_ref": metadata.get("headRefName"),
        "head_sha": metadata.get("headRefOid"),
        "changed_files": metadata.get("changedFiles"),
        "additions": metadata.get("additions"),
        "deletions": metadata.get("deletions"),
        "docs_loaded": sorted(docs),
        "included_files": [f.get("filename") for f in included_files],
        "skipped_files": [{"filename": f.get("filename"), "reason": f.get("skip_reason")} for f in skipped_files],
        "diff_truncated": diff_truncated,
        "max_diff_chars": max_diff_chars,
    }
    sections = [
        "# Hermes PR Review Context",
        "",
        "## PR metadata",
        json.dumps({k: metadata.get(k) for k in sorted(metadata)}, indent=2, default=str),
        "",
        "## Trusted base-branch project docs",
    ]
    if docs:
        for path, text in docs.items():
            sections.extend([f"\n### {path}", text])
    else:
        sections.append("No trusted project docs found.")
    sections.extend([
        "",
        "## Included changed files",
        json.dumps(manifest["included_files"], indent=2),
        "",
        "## Skipped files",
        json.dumps(manifest["skipped_files"], indent=2),
        "",
        "## PR diff",
        clipped_diff,
    ])
    return "\n".join(sections), manifest


def render_markdown(review: Dict[str, Any], manifest: Dict[str, Any]) -> str:
    findings = review.get("findings") or []
    lines = [
        "<!-- hermes-pr-review:summary:v1 -->",
        "## Hermes PR Review",
        "",
        f"**Verdict:** {str(review.get('verdict', 'comment')).upper()}",
        f"**Risk:** {str(review.get('risk', 'low')).title()}",
        f"**Reviewed commit:** `{str(manifest.get('head_sha') or 'unknown')[:12]}`",
        f"**Findings:** {len(findings)}",
        "",
        "### Summary",
        str(review.get("summary") or "No summary provided."),
        "",
    ]
    if findings:
        lines.append("### Findings")
        for idx, finding in enumerate(findings, 1):
            sev = str(finding.get("severity") or "warning").upper()
            path = finding.get("path") or "unknown"
            line = finding.get("line")
            loc = f"{path}:{line}" if line else str(path)
            lines.extend([
                "",
                f"{idx}. **{sev} — {finding.get('title', 'Finding')}**",
                f"   - Location: `{loc}`",
                f"   - Confidence: {finding.get('confidence', 'medium')}",
                f"   - Evidence: {finding.get('evidence', '')}",
                f"   - Why it matters: {finding.get('why_it_matters', '')}",
                f"   - Suggested fix: {finding.get('suggested_fix', '')}",
            ])
    else:
        lines.extend(["### Findings", "No actionable findings reported."])

    notes = review.get("verification_notes") or []
    lines.extend(["", "### Verification / context used"])
    for note in notes:
        lines.append(f"- {note}")
    for path in manifest.get("docs_loaded") or []:
        lines.append(f"- Loaded trusted doc `{path}` from the base branch")
    if manifest.get("diff_truncated"):
        lines.append(f"- Diff was truncated at {manifest.get('max_diff_chars')} characters")
    skipped = manifest.get("skipped_files") or []
    if skipped:
        lines.append(f"- Skipped {len(skipped)} ignored/generated files")
    return "\n".join(lines).rstrip() + "\n"


def write_artifacts(out_dir: Path, *, context: str, manifest: Dict[str, Any], review: Dict[str, Any]) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "context": out_dir / "context.md",
        "manifest": out_dir / "context-manifest.json",
        "findings": out_dir / "findings.json",
        "review": out_dir / "review.md",
    }
    paths["context"].write_text(context, encoding="utf-8")
    paths["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    paths["findings"].write_text(json.dumps(review, indent=2, sort_keys=True), encoding="utf-8")
    paths["review"].write_text(render_markdown(review, manifest), encoding="utf-8")
    return {name: str(path) for name, path in paths.items()}


def stub_review(manifest: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "verdict": "comment",
        "risk": "low",
        "summary": "Dry-run context collection completed; LLM review was skipped.",
        "findings": [],
        "verification_notes": [
            "Fetched PR metadata, changed files, and diff through gh.",
            "Loaded trusted project docs from the base branch when available.",
            f"Prepared {len(manifest.get('included_files') or [])} included changed files for review.",
        ],
    }


def as_jsonable(value: Any) -> Any:
    if hasattr(value, "parsed"):
        return value.parsed
    if hasattr(value, "__dict__"):
        return asdict(value) if hasattr(value, "__dataclass_fields__") else value.__dict__
    return value
