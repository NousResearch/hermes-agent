"""Validate Hermes provider-level Git/PR provenance.

The CLI entry point is ``hermes-provenance-check``.  It validates the
portable provenance contract documented by the bundled
``hermes-pr-provenance`` skill:

- Hermes-authored commits should have final, contiguous ``Writer:`` trailers.
- ``Refs:`` should be present when a branch is tied to a task/issue.
- ``Writer:`` values should be provider-level, not exact model names.
- PR bodies should contain a concise ``## Provenance`` block when requested.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

TRAILER_RE = re.compile(r"^([A-Za-z][A-Za-z0-9-]*):\s*(.*)$")
PROVENANCE_HEADING_RE = re.compile(r"^##\s+Provenance\s*$", re.IGNORECASE | re.MULTILINE)
NEXT_H2_RE = re.compile(r"^##\s+", re.MULTILINE)

DEFAULT_ALLOWED_WRITERS = frozenset(
    {
        "chat",  # Some repos use chat/human for existing local enums.
        "claude",
        "codex",
        "cursor",
        "gemini",
        "grok",
        "human",
        "openrouter",
        "user",
    }
)

MODELISH_WRITER_RE = re.compile(
    r"(?:[/.]|\d|gpt|opus|sonnet|haiku|flash|pro|reasoning|mini|large)",
    re.IGNORECASE,
)

REQUIRED_PR_PROVENANCE_FIELDS = (
    "GitHub actor",
    "PR created by",
    "Implemented by",
    "Writers from commit trailers",
    "Task ledger",
)


@dataclass(frozen=True)
class Commit:
    sha: str
    subject: str
    body: str

    @property
    def short_sha(self) -> str:
        return self.sha[:12]


@dataclass(frozen=True)
class TrailerBlock:
    trailers: dict[str, list[str]]
    raw_lines: tuple[str, ...]

    def values(self, key: str) -> list[str]:
        return self.trailers.get(key.lower(), [])


@dataclass(frozen=True)
class Finding:
    severity: str
    message: str
    commit: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        return {"severity": self.severity, "message": self.message, "commit": self.commit}


@dataclass
class CheckResult:
    range_spec: str | None = None
    commit_count: int = 0
    writers: set[str] = field(default_factory=set)
    refs: set[str] = field(default_factory=set)
    findings: list[Finding] = field(default_factory=list)
    pr_block: str | None = None

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "range": self.range_spec,
            "commit_count": self.commit_count,
            "writers": sorted(self.writers),
            "refs": sorted(self.refs),
            "findings": [f.as_dict() for f in self.findings],
            "pr_block": self.pr_block,
        }


def run_git(args: Sequence[str], cwd: Path | None = None) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        cmd = " ".join(["git", *args])
        raise RuntimeError(f"{cmd} failed: {completed.stderr.strip()}")
    return completed.stdout


def parse_final_trailer_block(message: str) -> TrailerBlock:
    """Return the final contiguous Git trailer block.

    This intentionally accepts only simple ``Key: value`` trailer lines at the
    end of the commit message. If a blank line appears between ``Writer:`` and
    ``Refs:``, the earlier trailer is outside the final block and therefore
    treated as missing by the validator.
    """

    lines = message.rstrip("\n").splitlines()
    raw_reversed: list[str] = []

    for line in reversed(lines):
        if not line.strip():
            break
        if not TRAILER_RE.match(line):
            break
        raw_reversed.append(line)

    raw_lines = tuple(reversed(raw_reversed))
    parsed: dict[str, list[str]] = {}
    for line in raw_lines:
        match = TRAILER_RE.match(line)
        if not match:
            continue
        key, value = match.group(1).lower(), match.group(2).strip()
        parsed.setdefault(key, []).append(value)
    return TrailerBlock(trailers=parsed, raw_lines=raw_lines)


def find_all_trailer_values(message: str, key: str) -> list[str]:
    needle = key.lower()
    values: list[str] = []
    for line in message.splitlines():
        match = TRAILER_RE.match(line)
        if match and match.group(1).lower() == needle:
            values.append(match.group(2).strip())
    return values


def parse_git_log(raw: str) -> list[Commit]:
    commits: list[Commit] = []
    for record in raw.split("\x1e"):
        record = record.strip("\n")
        if not record:
            continue
        parts = record.split("\x1f", 2)
        if len(parts) != 3:
            continue
        sha, subject, body = parts
        commits.append(Commit(sha=sha.strip(), subject=subject.strip(), body=body))
    return commits


def get_commits(range_spec: str, cwd: Path | None = None) -> list[Commit]:
    raw = run_git(["log", "--format=%H%x1f%s%x1f%B%x1e", range_spec], cwd=cwd)
    return parse_git_log(raw)


def normalize_allowed_writers(extra_writers: Iterable[str]) -> set[str]:
    allowed = set(DEFAULT_ALLOWED_WRITERS)
    allowed.update(writer.strip().lower() for writer in extra_writers if writer.strip())
    return allowed


def validate_writer_value(
    writer: str,
    allowed_writers: set[str],
    *,
    allow_any_writer: bool,
) -> str | None:
    if not writer:
        return "empty Writer trailer"
    normalized = writer.lower()
    if MODELISH_WRITER_RE.search(normalized):
        return (
            f"Writer value {writer!r} looks model-specific; use a provider-level "
            "value such as codex, grok, claude, or gemini"
        )
    if not re.fullmatch(r"[a-z][a-z0-9_-]*", normalized):
        return f"Writer value {writer!r} is not a simple provider token"
    if not allow_any_writer and normalized not in allowed_writers:
        return (
            f"Writer value {writer!r} is not in the allowed provider set; "
            "pass --allowed-writer for repository-specific enums"
        )
    return None


def validate_commits(
    commits: Sequence[Commit],
    *,
    range_spec: str | None = None,
    require_refs: bool = True,
    allowed_writers: set[str] | None = None,
    allow_any_writer: bool = False,
    allow_model_provenance: bool = False,
) -> CheckResult:
    allowed_writers = allowed_writers or normalize_allowed_writers(())
    result = CheckResult(range_spec=range_spec, commit_count=len(commits))

    for commit in commits:
        block = parse_final_trailer_block(commit.body)
        writers = block.values("writer")
        refs = block.values("refs")
        writer_models = find_all_trailer_values(commit.body, "writer-model")
        all_writers = find_all_trailer_values(commit.body, "writer")
        all_refs = find_all_trailer_values(commit.body, "refs")

        if writer_models and not allow_model_provenance:
            result.findings.append(
                Finding(
                    "error",
                    "Writer-Model trailer is present but exact model provenance is not enabled",
                    commit.short_sha,
                )
            )

        if all_writers and not writers:
            result.findings.append(
                Finding(
                    "error",
                    "Writer trailer exists but is not in the final contiguous trailer block",
                    commit.short_sha,
                )
            )
        if all_refs and not refs:
            result.findings.append(
                Finding(
                    "error",
                    "Refs trailer exists but is not in the final contiguous trailer block",
                    commit.short_sha,
                )
            )

        if not writers:
            result.findings.append(
                Finding("error", "missing final Writer trailer", commit.short_sha)
            )
        for writer in writers:
            result.writers.add(writer.lower())
            problem = validate_writer_value(
                writer,
                allowed_writers,
                allow_any_writer=allow_any_writer,
            )
            if problem:
                result.findings.append(Finding("error", problem, commit.short_sha))

        if require_refs and not refs:
            result.findings.append(
                Finding("error", "missing final Refs trailer", commit.short_sha)
            )
        for ref in refs:
            if not ref:
                result.findings.append(Finding("error", "empty Refs trailer", commit.short_sha))
            else:
                result.refs.add(ref)

    if not commits:
        result.findings.append(
            Finding("warning", "no commits found in range; nothing to validate")
        )

    return result


def extract_pr_provenance_block(body: str) -> str | None:
    match = PROVENANCE_HEADING_RE.search(body)
    if not match:
        return None
    next_match = NEXT_H2_RE.search(body, match.end())
    end = next_match.start() if next_match else len(body)
    return body[match.start() : end].strip()


def validate_pr_body(body: str, result: CheckResult) -> None:
    block = extract_pr_provenance_block(body)
    result.pr_block = block
    if block is None:
        result.findings.append(Finding("error", "PR body is missing a ## Provenance block"))
        return

    lower_block = block.lower()
    for field_name in REQUIRED_PR_PROVENANCE_FIELDS:
        if field_name.lower() not in lower_block:
            result.findings.append(
                Finding("error", f"PR provenance block is missing {field_name!r}")
            )

    missing_writers = [writer for writer in result.writers if writer not in lower_block]
    if missing_writers:
        result.findings.append(
            Finding(
                "warning",
                "PR provenance block does not mention detected writer(s): "
                + ", ".join(sorted(missing_writers)),
            )
        )


def emit_pr_block(result: CheckResult, github_actor: str | None = None) -> str:
    writers = ", ".join(sorted(result.writers)) if result.writers else "unknown"
    refs = ", ".join(sorted(result.refs)) if result.refs else "none"
    actor = github_actor or "<github-login-or-bot-account>"
    return "\n".join(
        [
            "## Provenance",
            "",
            f"- GitHub actor: {actor}",
            f"- PR created by: {writers}",
            f"- Implemented by: {writers}",
            f"- Writers from commit trailers: {writers}",
            f"- Task ledger: {refs}",
        ]
    )


def print_human_report(result: CheckResult, *, include_pr_block: bool = False) -> None:
    print("Hermes provenance check")
    if result.range_spec:
        print(f"Range: {result.range_spec}")
    print(f"Commits: {result.commit_count}")
    print(f"Writers: {', '.join(sorted(result.writers)) if result.writers else 'none'}")
    print(f"Refs: {', '.join(sorted(result.refs)) if result.refs else 'none'}")

    for finding in result.findings:
        prefix = finding.severity.upper()
        location = f" {finding.commit}" if finding.commit else ""
        print(f"{prefix}{location}: {finding.message}")

    print("PASS" if result.ok else "FAIL")

    if include_pr_block:
        print()
        print(result.pr_block or emit_pr_block(result))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate Hermes provider-level commit/PR provenance."
    )
    range_group = parser.add_mutually_exclusive_group()
    range_group.add_argument("--range", dest="range_spec", help="Explicit git range, e.g. main..HEAD")
    parser.add_argument("--base", default="origin/main", help="Base ref when --range is not set (default: origin/main)")
    parser.add_argument("--head", default="HEAD", help="Head ref when --range is not set (default: HEAD)")
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="Repository path (default: cwd)")
    parser.add_argument(
        "--no-require-refs",
        action="store_true",
        help="Do not fail commits that have Writer but no Refs trailer.",
    )
    parser.add_argument(
        "--allowed-writer",
        action="append",
        default=[],
        help="Allow an additional repository-specific Writer value. Repeatable.",
    )
    parser.add_argument(
        "--allow-any-writer",
        action="store_true",
        help="Allow any simple provider token, while still rejecting model-looking Writer values.",
    )
    parser.add_argument(
        "--allow-model-provenance",
        action="store_true",
        help="Allow Writer-Model trailers. Off by default because model provenance is deferred.",
    )
    parser.add_argument(
        "--pr-body",
        help="Validate a PR body markdown file. Use '-' to read from stdin.",
    )
    parser.add_argument(
        "--emit-pr-block",
        action="store_true",
        help="Print a starter ## Provenance block from detected writers/refs.",
    )
    parser.add_argument(
        "--github-actor",
        help="GitHub actor to include when emitting a PR provenance block.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser


def read_pr_body_arg(value: str) -> str:
    if value == "-":
        return sys.stdin.read()
    return Path(value).read_text(encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    range_spec = args.range_spec or f"{args.base}..{args.head}"
    result: CheckResult
    try:
        commits = get_commits(range_spec, cwd=args.repo)
        result = validate_commits(
            commits,
            range_spec=range_spec,
            require_refs=not args.no_require_refs,
            allowed_writers=normalize_allowed_writers(args.allowed_writer),
            allow_any_writer=args.allow_any_writer,
            allow_model_provenance=args.allow_model_provenance,
        )
        if args.pr_body:
            validate_pr_body(read_pr_body_arg(args.pr_body), result)
        if args.emit_pr_block:
            result.pr_block = emit_pr_block(result, args.github_actor)
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        result = CheckResult(range_spec=range_spec)
        result.findings.append(Finding("error", str(exc)))

    if args.json:
        print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    else:
        print_human_report(result, include_pr_block=args.emit_pr_block)

    return 0 if result.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
