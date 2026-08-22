"""On-demand tirith scan for persistent Hermes instruction poisoning."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

BASELINE_SCHEMA_VERSION = 1
SEVERITY_ORDER = {"INFO": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
DEFAULT_INSTRUCTION_PATTERNS = (
    "SOUL.md",
    "AGENTS.md",
    "CLAUDE.md",
    ".cursorrules",
    "config.yaml",
    "mcp.json",
    ".mcp.json",
    "skills/*",
    "memories/*",
    "workspace/*",
    ".claude/*",
    ".cursor/*",
    ".hermes/*",
    ".github/copilot-instructions.md",
)


@dataclass(frozen=True)
class ConfigFinding:
    path: str
    rule_id: str
    severity: str
    title: str
    fingerprint: str

    def as_dict(self) -> dict[str, str]:
        return {
            "path": self.path,
            "rule_id": self.rule_id,
            "severity": self.severity,
            "title": self.title,
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class ScanSummary:
    scanned_count: int
    findings: list[ConfigFinding]
    incomplete_reasons: tuple[str, ...] = ()


def _finding_fingerprint(path: str, raw: dict[str, Any]) -> str:
    """Hash the complete finding without persisting evidence or file content."""
    canonical = json.dumps(
        {"path": path, "finding": raw},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _parse_tirith_output(payload: dict[str, Any]) -> ScanSummary:
    """Normalize tirith directory- and single-file JSON schemas."""
    files = payload.get("files")
    if not isinstance(files, list):
        files = [payload] if "path" in payload else []

    findings: list[ConfigFinding] = []
    for file_result in files:
        if not isinstance(file_result, dict):
            continue
        path = str(file_result.get("path") or "<unknown>")
        raw_findings = file_result.get("findings") or []
        if not isinstance(raw_findings, list):
            continue
        for raw in raw_findings:
            if not isinstance(raw, dict):
                continue
            findings.append(
                ConfigFinding(
                    path=path,
                    rule_id=str(raw.get("rule_id") or "unknown"),
                    severity=str(raw.get("severity") or "unknown").upper(),
                    title=str(raw.get("title") or raw.get("description") or ""),
                    fingerprint=_finding_fingerprint(path, raw),
                )
            )

    findings.sort(key=lambda item: (-SEVERITY_ORDER.get(item.severity, -1), item.path, item.rule_id))
    scanned_count = payload.get("scanned_count", 1 if "path" in payload else 0)
    incomplete_reasons: list[str] = []
    panic_count = payload.get("panic_count", 0)
    if isinstance(panic_count, int) and not isinstance(panic_count, bool) and panic_count > 0:
        incomplete_reasons.append(f"tirith reported {panic_count} rule panic(s)")
    if payload.get("truncated") is True:
        incomplete_reasons.append("tirith truncated the scan")
    if payload.get("analysis_incomplete") is True:
        reason = "tirith reported incomplete analysis"
        coverage_gaps = payload.get("coverage_gaps")
        if isinstance(coverage_gaps, list) and coverage_gaps:
            reason += f" ({len(coverage_gaps)} coverage gap(s))"
        incomplete_reasons.append(reason)
    return ScanSummary(
        scanned_count=int(scanned_count or 0),
        findings=findings,
        incomplete_reasons=tuple(incomplete_reasons),
    )


def _scan_path(
    tirith: str,
    path: Path,
    timeout: int,
    include_patterns: tuple[str, ...] = (),
) -> ScanSummary:
    command = [
        tirith,
        "scan",
        str(path),
        "--profile",
        "ai-agent-repo",
        "--fail-on",
        "high",
        "--json",
    ]
    for pattern in include_patterns:
        command.extend(("--include", pattern))
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"tirith scan failed for {path}: {exc}") from exc

    if result.returncode not in {0, 1, 2}:
        detail = result.stderr.strip() or f"exit code {result.returncode}"
        raise RuntimeError(f"tirith scan failed for {path}: {detail}")
    try:
        payload = json.loads(result.stdout)
    except (json.JSONDecodeError, TypeError) as exc:
        raise RuntimeError(f"tirith returned invalid JSON for {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"tirith returned an invalid result for {path}")
    return _parse_tirith_output(payload)


def _load_baseline(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"could not read baseline {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid baseline document in {path}")
    if payload.get("schema_version") != BASELINE_SCHEMA_VERSION:
        raise RuntimeError(f"unsupported baseline schema in {path}")
    fingerprints = payload.get("fingerprints")
    if not isinstance(fingerprints, list) or not all(isinstance(item, str) for item in fingerprints):
        raise RuntimeError(f"invalid baseline fingerprints in {path}")
    return set(fingerprints)


def _write_baseline(path: Path, findings: list[ConfigFinding]) -> None:
    if path.is_symlink():
        raise RuntimeError(f"refusing to replace symlinked baseline {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "fingerprints": sorted({finding.fingerprint for finding in findings}),
    }
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _render_human(
    *, scanned_count: int, findings: list[ConfigFinding], new_findings: list[ConfigFinding],
    baseline_path: Path, baseline_exists: bool, updated: bool,
    incomplete_reasons: list[str],
) -> str:
    lines = [f"Scanned {scanned_count} file(s); tirith reported {len(findings)} finding(s)."]
    if updated:
        lines.append(f"Baseline updated with {len(findings)} finding(s): {baseline_path}")
    elif not new_findings:
        lines.append("No new findings since the accepted baseline.")
    else:
        scope = "No baseline exists; treating all findings as new." if not baseline_exists else f"{len(new_findings)} new finding(s):"
        lines.append(scope)
        for finding in new_findings:
            lines.append(f"  [{finding.severity}] {finding.path}: {finding.rule_id} — {finding.title}")
        if not baseline_exists:
            lines.append("Review the findings, then rerun with --update-baseline to accept them.")
    if incomplete_reasons:
        lines.append(f"Scan incomplete: {'; '.join(incomplete_reasons)}.")
        lines.append("The baseline was not updated; resolve the incomplete analysis and retry.")
    return "\n".join(lines)


def cmd_security_scan(args: argparse.Namespace) -> int:
    """Implementation of ``hermes security scan``."""
    home = Path(get_hermes_home())
    explicit_paths = bool(getattr(args, "paths", None))
    paths = [Path(item).expanduser() for item in (getattr(args, "paths", None) or [home])]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        print(f"scan path does not exist: {missing[0]}", file=sys.stderr)
        return 2

    from tools.tirith_security import resolve_tirith_for_scan

    tirith, cfg = resolve_tirith_for_scan()
    if not tirith:
        print("tirith is disabled or unavailable; configure security.tirith_path", file=sys.stderr)
        return 2
    requested_timeout = getattr(args, "timeout", None)
    timeout = cfg.get("tirith_scan_timeout", 120) if requested_timeout is None else requested_timeout
    if not isinstance(timeout, int) or timeout <= 0:
        print("scan timeout must be a positive integer", file=sys.stderr)
        return 2

    baseline_path = Path(getattr(args, "baseline", None) or home / "security" / "tirith-scan-baseline.json")
    baseline_exists = baseline_path.exists()
    try:
        baseline = _load_baseline(baseline_path)
        include_patterns = () if explicit_paths else DEFAULT_INSTRUCTION_PATTERNS
        summaries = [_scan_path(tirith, path, timeout, include_patterns) for path in paths]
        findings = [finding for summary in summaries for finding in summary.findings]
        scanned_count = sum(summary.scanned_count for summary in summaries)
        incomplete_reasons = [
            reason for summary in summaries for reason in summary.incomplete_reasons
        ]
        new_findings = [finding for finding in findings if finding.fingerprint not in baseline]
        updated = bool(getattr(args, "update_baseline", False)) and not incomplete_reasons
        if updated:
            _write_baseline(baseline_path, findings)
    except (OSError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    output = {
        "scanned_count": scanned_count,
        "finding_count": len(findings),
        "new_finding_count": len(new_findings),
        "baseline": str(baseline_path),
        "baseline_updated": updated,
        "analysis_incomplete": bool(incomplete_reasons),
        "incomplete_reasons": incomplete_reasons,
        "new_findings": [finding.as_dict() for finding in new_findings],
    }
    if getattr(args, "json", False):
        print(json.dumps(output, indent=2))
    else:
        print(
            _render_human(
                scanned_count=scanned_count,
                findings=findings,
                new_findings=new_findings,
                baseline_path=baseline_path,
                baseline_exists=baseline_exists,
                updated=updated,
                incomplete_reasons=incomplete_reasons,
            )
        )

    if incomplete_reasons:
        return 2
    if updated:
        return 0
    threshold = SEVERITY_ORDER[str(getattr(args, "fail_on", "high")).upper()]
    return int(any(SEVERITY_ORDER.get(finding.severity, -1) >= threshold for finding in new_findings))