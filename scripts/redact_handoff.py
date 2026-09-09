#!/usr/bin/env python3
"""Redact credentials from the collect-logs handoff staging directory before it is zipped.

Reuses the agent's production redactor — ``agent.redact.redact_sensitive_text`` — the
same vocabulary that masks tool output and logs at runtime. No new pattern list is
invented here. Pattern-matching is a backstop, not a guarantee.

Every text file under ``<staging_dir>`` is rewritten in place with redactions applied.
After redaction the result is re-scanned with a strict AWS/GitHub/Slack matcher; if
anything still looks like a live credential it is reported as a SURVIVOR and the caller
(collect-logs) must NOT create the zip.

    redact_handoff.py <staging_dir>

Exit codes:
    0  clean — redactions may have been applied, nothing survived
    2  could not load the redactor, or bad arguments  (fail closed)
    3  a likely secret SURVIVED redaction — do not create the zip  (fail closed)

Writes ``<staging_dir>/redaction-report.txt``.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

_REPORT_NAME = "redaction-report.txt"
_MAX_BYTES = 8 * 1024 * 1024  # skip anything larger — handoff text files are small
_SKIP_DIRS = {".git"}
# Extensions we never treat as redactable text (belt; the NUL sniff is the real guard).
_BINARY_EXT = {
    ".zip", ".gz", ".tar", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf",
    ".ico", ".woff", ".woff2", ".ttf", ".otf", ".mp4", ".mov", ".sqlite", ".db",
    ".pyc", ".so", ".dll", ".exe",
}

# Strict survivor matcher — same three families as .githooks/content-scan, same
# placeholder carve-outs. Used ONLY to decide fail-closed after redaction ran.
_SURVIVOR_RE = re.compile(
    r"AKIA(?!IOSFODNN7EXAMPLE)(?!EXAMPLE)[0-9A-Z]{16}"
    r"|gh[pousr]_(?!(.)\1{19})[A-Za-z0-9]{36,}"
    r"|github_pat_[0-9A-Za-z_]{82}"
    r"|xox[bpars]-[0-9]{9,}-[0-9]{9,}-[A-Za-z0-9]{20,}"
    r"|xapp-1-[A-Z0-9]+-[0-9]+-[a-f0-9]{16,}"
)


def _load_redactor():
    """Import agent.redact.redact_sensitive_text, adding the repo root to sys.path."""
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from agent.redact import redact_sensitive_text  # noqa: E402

    return redact_sensitive_text


def _looks_binary(data: bytes) -> bool:
    return b"\x00" in data[:8192]


def _iter_text_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for name in filenames:
            if name == _REPORT_NAME:
                continue
            p = Path(dirpath) / name
            if p.suffix.lower() in _BINARY_EXT:
                continue
            try:
                if p.stat().st_size > _MAX_BYTES:
                    continue
                raw = p.read_bytes()
            except OSError:
                continue
            if _looks_binary(raw):
                continue
            yield p, raw


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        sys.stderr.write("usage: redact_handoff.py <staging_dir>\n")
        return 2
    root = Path(argv[1])
    if not root.is_dir():
        sys.stderr.write(f"redact_handoff: not a directory: {root}\n")
        return 2

    try:
        redact = _load_redactor()
    except Exception as exc:  # noqa: BLE001 — any import failure is fail-closed
        sys.stderr.write(
            "redact_handoff: cannot import agent.redact — the mandatory handoff "
            f"redaction cannot run ({exc.__class__.__name__}: {exc}).\n"
            "Fix: run scripts/bootstrap-north-forge.ps1, or use a Python that can "
            "import this repo, then re-run collect-logs.\n"
        )
        return 2

    touched: list[tuple[str, int]] = []
    survivors: list[str] = []
    scanned = 0

    for path, raw in _iter_text_files(root):
        scanned += 1
        try:
            original = raw.decode("utf-8")
        except UnicodeDecodeError:
            original = raw.decode("utf-8", "replace")
        # force=True: a safety boundary must redact regardless of config.
        redacted = redact(original, force=True)
        rel = path.relative_to(root).as_posix()

        if redacted != original:
            changed_lines = sum(
                1 for a, b in zip(original.splitlines(), redacted.splitlines()) if a != b
            )
            path.write_text(redacted, encoding="utf-8")
            touched.append((rel, changed_lines))

        for i, line in enumerate(redacted.splitlines(), 1):
            if _SURVIVOR_RE.search(line):
                survivors.append(f"{rel}:{i}: {line.strip()[:160]}")

    # --- write the report -------------------------------------------------
    report = root / _REPORT_NAME
    lines = [
        "handoff redaction report",
        "========================",
        "",
        "Redactor: agent.redact.redact_sensitive_text (the agent's production vocabulary).",
        "Pattern-matching is a BACKSTOP, not a guarantee — it will miss novel or obfuscated",
        "secrets. Treat the bundle as sensitive and check it before sharing.",
        "",
        f"text files scanned : {scanned}",
        f"files redacted     : {len(touched)}",
    ]
    if touched:
        lines.append("")
        lines.append("redacted files (changed lines):")
        lines += [f"  {rel}  ({n} line(s))" for rel, n in sorted(touched)]
    else:
        lines.append("")
        lines.append("no redactions were necessary.")
    lines.append("")
    if survivors:
        lines.append(f"SURVIVORS — {len(survivors)} line(s) still match a strict credential shape:")
        lines += [f"  {s}" for s in survivors]
        lines.append("")
        lines.append("The zip was NOT created. Remove/rotate these values and re-run collect-logs.")
    else:
        lines.append("survivors: none — nothing matched the strict AWS/GitHub/Slack recheck.")
    lines.append("")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if survivors:
        sys.stderr.write(
            f"redact_handoff: {len(survivors)} likely secret(s) SURVIVED redaction — "
            "not creating the zip:\n"
        )
        for s in survivors:
            sys.stderr.write(f"  {s}\n")
        return 3

    sys.stdout.write(
        f"redact_handoff: {scanned} file(s) scanned, {len(touched)} redacted, 0 survivors\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
