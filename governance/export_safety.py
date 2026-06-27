"""Export-mode and secret-scan scaffolding for safe bundle generation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Any, Dict, Iterable, List


class ExportMode(Enum):
    RUNTIME_CLEAN = ("runtime_clean", False, [
        r"(^|/)\.git(/|$)",
        r"(^|/)node_modules(/|$)",
        r"(^|/)\.venv(/|$)",
        r"(^|/)venv(/|$)",
        r"(^|/)logs?(/|$)",
        r"(^|/)sessions?(/|$)",
        r"(^|/)auth\.json$",
        r"(^|/)\.env(\..*)?$",
        r"(^|/)api-keys(/|$)",
    ])
    SHAREABLE_AUDIT_FULL = ("shareable_audit_full", True, [
        r"(^|/)\.git/objects(/|$)",
        r"(^|/)node_modules(/|$)",
    ])
    LOCAL_FULL_WITH_SECRETS = ("local_full_with_secrets", False, [])

    def __init__(self, label: str, requires_redaction: bool, deny_patterns: List[str]):
        self.label = label
        self.requires_redaction = requires_redaction
        self._deny_patterns = deny_patterns

    def allows_path(self, path: str) -> bool:
        normalized = path.replace("\\", "/")
        while normalized.startswith("./"):
            normalized = normalized[2:]
        return not any(re.search(pattern, normalized, re.IGNORECASE) for pattern in self._deny_patterns)


@dataclass
class SecretFinding:
    label: str
    start: int
    end: int
    preview: str


class SecretScanner:
    """Small stdlib-only scanner for common token/password shapes."""

    PATTERNS = [
        ("named_secret", re.compile(r"(?i)\b(?:[A-Z0-9_]*_)?(?:API[_-]?KEY|TOKEN|SECRET|PASSWORD|PASSWD|PRIVATE[_-]?KEY)\b\s*[:=]\s*['\"]?([^'\"\s]{6,})")),
        ("openai_key", re.compile(r"\bsk-[A-Za-z0-9._-]{6,}\b")),
        ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b")),
        ("bearer_token", re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{12,}\b")),
    ]

    @staticmethod
    def _safe_preview(label: str, start: int, end: int) -> str:
        """Return a non-leaking finding preview.

        Secret scan reports are evidence artifacts; they must not echo the
        matched secret bytes back into logs, chat, or exported audit bundles.
        """
        return f"[REDACTED:{label}:chars={end - start}]"

    def find(self, text: str) -> List[SecretFinding]:
        findings: List[SecretFinding] = []
        for label, pattern in self.PATTERNS:
            for match in pattern.finditer(text or ""):
                start, end = match.span()
                findings.append(SecretFinding(label=label, start=start, end=end, preview=self._safe_preview(label, start, end)))
        findings.sort(key=lambda item: (item.start, item.end))
        # Drop exact duplicate spans from overlapping patterns.
        unique: List[SecretFinding] = []
        seen = set()
        for finding in findings:
            key = (finding.start, finding.end)
            if key not in seen:
                unique.append(finding)
                seen.add(key)
        return unique

    def scan_text(self, text: str, *, mode: str = "report") -> Dict[str, Any]:
        findings = self.find(text)
        return {
            "status": "redaction_required" if findings else "clean",
            "mode": mode,
            "findings_count": len(findings),
            "findings": [finding.__dict__ for finding in findings],
        }

    def redact_text(self, text: str) -> str:
        findings = self.find(text)
        if not findings:
            return text
        pieces: List[str] = []
        cursor = 0
        for finding in findings:
            if finding.start < cursor:
                continue
            pieces.append(text[cursor:finding.start])
            pieces.append(f"[REDACTED:{finding.label}]")
            cursor = finding.end
        pieces.append(text[cursor:])
        return "".join(pieces)
