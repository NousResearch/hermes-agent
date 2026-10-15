#!/usr/bin/env python3
"""Deterministic secret/PII preflight before external egress (stdlib only).

High-precision on credentials/secrets; fail closed on API keys and private keys.
Does not claim complete PII detection.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    (
        "secrets",
        re.compile(
            r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----[\s\S]*?"
            r"-----END (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
            re.I,
        ),
        "block",
    ),
    ("credentials", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"), "block"),
    (
        "credentials",
        re.compile(r"\b(?:api[_-]?key|apikey)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}", re.I),
        "block",
    ),
    ("credentials", re.compile(r"\bBearer\s+[A-Za-z0-9\-._~+/]+=*", re.I), "block"),
    ("credentials", re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "block"),
    (
        "passwords_connection",
        re.compile(r"(?:postgres|mysql|mongodb|redis|amqp)://[^\s:]+:[^\s@]+@", re.I),
        "block",
    ),
    ("passwords_connection", re.compile(r"\bpassword\s*[:=]\s*\S+", re.I), "block"),
    (
        "financial",
        re.compile(r"\b(?:4\d{12}(?:\d{3})?|5[1-5]\d{14}|3[47]\d{13})\b"),
        "block",
    ),
]


def preflight(text: str) -> dict[str, Any]:
    if not text:
        return {
            "findings": [],
            "blocked_categories": [],
            "safe_for_egress": True,
            "redacted_text": text or "",
        }
    findings: list[dict[str, str]] = []
    blocked: set[str] = set()
    redacted = text
    for category, pattern, severity in _PATTERNS:
        for m in pattern.finditer(text):
            span = m.group(0)
            findings.append(
                {
                    "category": category,
                    "span_hint": span[:48] + ("…" if len(span) > 48 else ""),
                    "severity": severity,
                }
            )
            if severity == "block":
                blocked.add(category)
        if severity == "block":
            redacted = pattern.sub(f"[REDACTED:{category}]", redacted)
    return {
        "findings": findings,
        "blocked_categories": sorted(blocked),
        "safe_for_egress": len(blocked) == 0,
        "redacted_text": redacted,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Neon Genie privacy preflight")
    parser.add_argument("--text", default="")
    parser.add_argument("--file", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    text = args.text
    if args.file is not None:
        text = args.file.read_text(encoding="utf-8")
    result = preflight(text)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("safe_for_egress:", result["safe_for_egress"])
        print("blocked_categories:", ",".join(result["blocked_categories"]) or "(none)")
        for f in result["findings"]:
            print(f"  [{f['severity']}] {f['category']}: {f['span_hint']}")
    return 0 if result["safe_for_egress"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
