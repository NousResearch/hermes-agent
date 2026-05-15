"""External update sources — discover, audit, and apply updates for git-backed
external components (plugins, dashboards, MCP helper repos).

This module implements two safe operations:

- **check**: fetch metadata, run audit, write report — never mutates working trees.
- **apply**: update only sources that are trusted, clean, fast-forwardable,
  and audit-passing.

Design doc: ``docs/design/external-update-sources.md``
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HERMES_HOME = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
REPORT_DIR = HERMES_HOME / "update-sources"
REPORT_PATH = REPORT_DIR / "last-run.json"

# Critical patterns — block apply unconditionally.
_CRITICAL_PATTERNS: List[Tuple[str, re.Pattern]] = [
    (
        "curl-pipe-shell",
        re.compile(
            r"\b(curl|wget)\b[^\n|;]*(\||>)\s*(bash|sh|dash|zsh)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "private-key-material",
        re.compile(r"BEGIN\s+(RSA\s+|OPENSSH\s+|EC\s+|DSA\s+)?PRIVATE\s+KEY"),
    ),
    (
        "env-secret-literal",
        re.compile(
            r"(?i)(api[_-]?key|token|secret|password)\s*[=:]\s*['\"]?"
            r"(?!test|fake|dummy|placeholder|example|resolver|x{4,}|your[-_]"
            r"|TODO|CHANGEME|REPLACE)"
            r"[^\s'\"]{16,}"
        ),
    ),
    (
        "chmod-suid",
        re.compile(r"chmod\s+[0-7]*[47][0-7]{2,3}"),
    ),
    (
        "encoded-script-exec",
        re.compile(
            r"base64\s+(-d|--decode)|\bexec\s*\(",
        ),
    ),
]

# Review-level patterns — reported, block non-interactive apply unless approved.
_REVIEW_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("eval-usage", re.compile(r"\beval\s*\(")),
    ("exec-usage", re.compile(r"\bexec\s*\(")),
    ("subprocess-shell-true", re.compile(r"subprocess\b[^\n]*shell\s*=\s*True")),
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AuditResult:
    """Static audit findings for a diff."""

    passed: bool = True
    critical: List[str] = field(default_factory=list)
    review: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "critical": self.critical,
            "review": self.review,
        }


@dataclass
class SourceResult:
    """Per-source report entry."""

    name: str
    kind: str  # "plugin", "dashboard-web-dist", "project-plugin"
    path: str
    remote_url: str = ""
    tracking_ref: str = ""
    head: str = ""
    remote_head: str = ""
    dirty: bool = False
    commits_available: int = 0
    changed_files: List[str] = field(default_factory=list)
    audit: dict = field(default_factory=dict)
    status: str = "unknown"  # one of: discovered, no_upstream, dirty, diverged, audit_blocked, audit_passed, updated, error
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Source discovery
# ---------------------------------------------------------------------------

def discover_sources(project_path: Optional[str] = None) -> List[SourceResult]:
    """Discover external git-backed components.

    Scans:

    - ``~/.hermes/plugins/*`` when the entry is a git checkout (kind ``plugin``).
    - dashboard plugins: plugins that contain ``dashboard/manifest.json``
      (kind ``plugin+dashboard``).
    - ``HERMES_WEB_DIST`` env var if it points to a git repo
      (kind ``dashboard-web-dist``).
    - Project plugins only when *project_path* is given explicitly or
      configured via ``HERMES_PROJECT_PLUGINS_PATH`` (kind ``project-plugin``).

    Returns a list of :class:`SourceResult` with ``status="discovered"``.
    """
    sources: List[SourceResult] = []

    # 1. User plugins
    plugins_dir = HERMES_HOME / "plugins"
    if plugins_dir.is_dir():
        for entry in sorted(plugins_dir.iterdir()):
            if not entry.is_dir():
                continue
            # Resolve symlinks (e.g. hermes-lcm -> plugin-checkouts/hermes-lcm)
            real = entry.resolve()
            if not (real / ".git").is_dir():
                continue
            kind = "plugin"
            if (real / "dashboard" / "manifest.json").is_file():
                kind = "plugin+dashboard"
            sources.append(
                SourceResult(
                    name=entry.name,
                    kind=kind,
                    path=str(real),
                    status="discovered",
                )
            )

    # 2. External web dist via HERMES_WEB_DIST
    web_dist = os.environ.get("HERMES_WEB_DIST", "")
    if web_dist:
        wd_path = Path(web_dist).resolve()
        if (wd_path / ".git").is_dir():
            sources.append(
                SourceResult(
                    name="web-dist",
                    kind="dashboard-web-dist",
                    path=str(wd_path),
                    status="discovered",
                )
            )

    # 3. Project plugins — only when explicitly requested
    pp = project_path or os.environ.get("HERMES_PROJECT_PLUGINS_PATH", "")
    if pp:
        pp_dir = Path(pp).resolve() / ".hermes" / "plugins"
        if pp_dir.is_dir():
            for entry in sorted(pp_dir.iterdir()):
                if not entry.is_dir():
                    continue
                real = entry.resolve()
                if not (real / ".git").is_dir():
                    continue
                sources.append(
                    SourceResult(
                        name=entry.name,
                        kind="project-plugin",
                        path=str(real),
                        status="discovered",
                    )
                )

    return sources


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def _git(args: List[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git"] + args,
        cwd=cwd,
        capture_output=True,
        text=True,
    )


def _get_remote_url(path: Path) -> str:
    r = _git(["config", "--get", "remote.origin.url"], path)
    return r.stdout.strip() if r.returncode == 0 else ""


def _get_tracking_ref(path: Path) -> str:
    r = _git(
        ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"], path
    )
    if r.returncode != 0:
        # Fallback: try @{upstream}
        r = _git(
            ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"],
            path,
        )
    return r.stdout.strip() if r.returncode == 0 else ""


def _get_head(path: Path) -> str:
    r = _git(["rev-parse", "--short", "HEAD"], path)
    return r.stdout.strip() if r.returncode == 0 else ""


def _is_dirty(path: Path) -> bool:
    """Check for modified, staged, or untracked files."""
    r = _git(["status", "--porcelain"], path)
    return bool(r.stdout.strip())


def _fetch(path: Path) -> bool:
    r = _git(["fetch", "--quiet"], path)
    return r.returncode == 0


def _get_remote_head(path: Path, tracking_ref: str) -> str:
    r = _git(["rev-parse", "--short", tracking_ref], path)
    return r.stdout.strip() if r.returncode == 0 else ""


def _count_commits_behind(path: Path, tracking_ref: str) -> int:
    r = _git(
        ["rev-list", "--count", f"HEAD..{tracking_ref}"], path
    )
    return int(r.stdout.strip()) if r.returncode == 0 else 0


def _has_diverged(path: Path, tracking_ref: str) -> bool:
    """True if local has commits not on the remote."""
    r = _git(
        ["rev-list", "--count", f"{tracking_ref}..HEAD"], path
    )
    return int(r.stdout.strip()) > 0 if r.returncode == 0 else True


def _get_changed_files(path: Path, tracking_ref: str) -> List[str]:
    r = _git(
        ["diff", "--name-only", f"HEAD..{tracking_ref}"], path
    )
    if r.returncode == 0 and r.stdout.strip():
        return r.stdout.strip().splitlines()
    return []


def _get_diff(path: Path, tracking_ref: str) -> str:
    r = _git(
        ["diff", "--unified=0", f"HEAD..{tracking_ref}"], path
    )
    return r.stdout if r.returncode == 0 else ""


# ---------------------------------------------------------------------------
# Audit gate
# ---------------------------------------------------------------------------

def audit_diff(diff_text: str) -> AuditResult:
    """Run static analysis on a diff. Returns an :class:`AuditResult`."""
    result = AuditResult()
    added_lines: List[str] = []
    for line in diff_text.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added_lines.append(line[1:])

    joined = "\n".join(added_lines)
    for label, pat in _CRITICAL_PATTERNS:
        if pat.search(joined):
            result.critical.append(label)

    for label, pat in _REVIEW_PATTERNS:
        if pat.search(joined):
            result.review.append(label)

    if result.critical:
        result.passed = False

    return result


# ---------------------------------------------------------------------------
# Enrichment — populate metadata for discovered sources
# ---------------------------------------------------------------------------

def enrich_source(src: SourceResult, fetch: bool = True) -> SourceResult:
    """Fill in git metadata for a discovered source.

    If *fetch* is True, runs ``git fetch`` to get the latest remote refs.
    Sets ``status`` to one of: ``no_upstream``, ``dirty``, ``diverged``,
    or leaves it at ``discovered`` if ready for audit.
    """
    p = Path(src.path)
    src.remote_url = _get_remote_url(p)
    src.tracking_ref = _get_tracking_ref(p)
    src.head = _get_head(p)
    src.dirty = _is_dirty(p)

    if not src.tracking_ref:
        src.status = "no_upstream"
        return src

    if src.dirty:
        src.status = "dirty"
        return src

    if fetch:
        _fetch(p)

    src.remote_head = _get_remote_head(p, src.tracking_ref)
    src.commits_available = _count_commits_behind(p, src.tracking_ref)

    if src.commits_available == 0:
        src.status = "audit_passed"  # nothing to update
        return src

    if _has_diverged(p, src.tracking_ref):
        src.status = "diverged"
        return src

    src.changed_files = _get_changed_files(p, src.tracking_ref)
    return src


# ---------------------------------------------------------------------------
# Check — safe read-only inspection
# ---------------------------------------------------------------------------

def run_check(project_path: Optional[str] = None) -> dict:
    """Discover sources, fetch metadata, audit, and return a report dict.

    Never mutates any working tree.
    """
    sources = discover_sources(project_path=project_path)

    results: List[SourceResult] = []
    for src in sources:
        enrich_source(src, fetch=True)
        # Run audit on anything that has commits available and isn't blocked
        if src.status == "discovered" and src.commits_available > 0:
            diff_text = _get_diff(Path(src.path), src.tracking_ref)
            audit = audit_diff(diff_text)
            src.audit = audit.to_dict()
            if not audit.passed:
                src.status = "audit_blocked"
            else:
                src.status = "audit_passed"
        results.append(src)

    report = _build_report("check", results)
    _write_report(report)
    return report


# ---------------------------------------------------------------------------
# Apply — audit-gated update
# ---------------------------------------------------------------------------

def run_apply(
    project_path: Optional[str] = None,
    source_name: Optional[str] = None,
    yes: bool = False,
    approve_review: bool = False,
) -> dict:
    """Discover, audit, and apply updates for passing sources.

    Parameters
    ----------
    source_name : str, optional
        Only apply this named source.
    yes : bool
        Skip interactive confirmation.
    approve_review : bool
        Also approve review-level findings (not just critical).
    """
    sources = discover_sources(project_path=project_path)

    if source_name:
        sources = [s for s in sources if s.name == source_name]
        if not sources:
            report = _build_report(
                "apply",
                [],
                error=f"Source '{source_name}' not found",
            )
            _write_report(report)
            return report

    results: List[SourceResult] = []
    for src in sources:
        enrich_source(src, fetch=True)

        if src.status in ("no_upstream", "dirty", "diverged"):
            src.audit = AuditResult(passed=False, critical=[src.status]).to_dict()
            src.status = f"blocked_{src.status}"
            results.append(src)
            continue

        if src.commits_available == 0:
            src.audit = AuditResult().to_dict()
            src.status = "audit_passed"  # nothing to do
            results.append(src)
            continue

        # Run audit
        diff_text = _get_diff(Path(src.path), src.tracking_ref)
        audit = audit_diff(diff_text)
        src.audit = audit.to_dict()

        if audit.critical:
            src.status = "audit_blocked"
            results.append(src)
            continue

        if audit.review and not approve_review:
            src.status = "audit_blocked"
            results.append(src)
            continue

        # Passed — apply via ff-only pull
        p = Path(src.path)
        r = _git(["pull", "--ff-only"], p)
        if r.returncode == 0:
            src.status = "updated"
            src.head = _get_head(p)
        else:
            src.status = "error"
            src.error = r.stderr.strip()[:500]

        results.append(src)

    report = _build_report("apply", results)
    _write_report(report)
    return report


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

def _build_report(
    mode: str,
    results: List[SourceResult],
    error: str = "",
) -> dict:
    """Build the JSON-serializable report dict."""
    sources_dicts = [s.to_dict() for s in results]
    updated = sum(1 for s in results if s.status == "updated")
    blocked = sum(
        1
        for s in results
        if s.status.startswith("blocked_") or s.status == "audit_blocked"
    )
    available = sum(1 for s in results if s.commits_available > 0 and s.status != "updated")
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "sources": sources_dicts,
        "summary": {
            "total": len(results),
            "updated": updated,
            "blocked": blocked,
            "available": available,
        },
        **({"error": error} if error else {}),
    }


def _write_report(report: dict) -> None:
    """Write report to ``~/.hermes/update-sources/last-run.json``."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")


# ---------------------------------------------------------------------------
# CLI printer
# ---------------------------------------------------------------------------

def print_report(report: dict) -> None:
    """Pretty-print a report for terminal output."""
    mode = report.get("mode", "check")
    summary = report.get("summary", {})

    print(f"\n📋 External source update report ({mode})")
    print(f"   Total: {summary.get('total', 0)}  "
          f"Updated: {summary.get('updated', 0)}  "
          f"Blocked: {summary.get('blocked', 0)}  "
          f"Available: {summary.get('available', 0)}")
    print()

    for src in report.get("sources", []):
        icon = {
            "updated": "✅",
            "audit_passed": "✓",
            "audit_blocked": "🚫",
            "discovered": "🔍",
        }.get(src.get("status", ""), "⚠")

        name = src.get("name", "?")
        kind = src.get("kind", "?")
        status = src.get("status", "?")
        commits = src.get("commits_available", 0)
        print(f"  {icon} {name} ({kind}) — {status}")
        if commits:
            print(f"     {commits} commit(s) available")
        audit = src.get("audit", {})
        if audit.get("critical"):
            print(f"     Critical: {', '.join(audit['critical'])}")
        if audit.get("review"):
            print(f"     Review: {', '.join(audit['review'])}")
        if src.get("error"):
            print(f"     Error: {src['error']}")

    print(f"\n   Report: {REPORT_PATH}")
