#!/usr/bin/env python3
"""Guard the container supply chain: every downloaded artifact verified, every
FROM digest-pinned (#126942).

Fails when a repo Dockerfile:
  * streams a download into an interpreter/archiver (`curl|tar`, `curl|sh`,
    `wget|sh`, ...) — the bytes become code with no verification step, or
  * names a base image by a mutable tag with no `@sha256:` digest, so the
    same build inputs resolve different content over time.

Both disciplines were already the runtime Dockerfile's own standard; this
guard keeps every current and future Dockerfile on it. Exempt a line with
`supply-chain-guard: allow` when a download is verifiably pinned by another
mechanism (e.g. an npm integrity hash on the same line).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Lines that fetch remote bytes and pipe them into a consumer in one step:
# nothing between the pipe and the consumer can inspect or verify them.
STREAM_PATTERNS = re.compile(
    r"\b(curl|wget)\b[^|]*\|\s*(sudo\s+)?(tar|sh|bash|zsh|python3?|unzip|dd)\b"
)
# FROM lines not carrying a digest. Stage references (FROM python_deps AS ...),
# `scratch`, and the FROM ${VAR} indirection (checked via its feeding ARG below)
# carry no tag colon, so they never match.
FROM_LINE = re.compile(r"^\s*FROM\s+(?:--\S+\s+)*(\S+)")
# A build ARG whose default value is a plain `<name>:<tag>` image reference
# with no digest — the pattern the sandbox base pin used before #126942.
ARG_TAG_ONLY = re.compile(
    r"^\s*ARG\s+\w+\s*=\s*[\"\']?[a-z0-9./_-]+:[a-z0-9._-]+(?!.*@sha256:)"
)
EXEMPT = "supply-chain-guard: allow"


def dockerfiles() -> list[Path]:
    root = REPO_ROOT / "Dockerfile"
    files = [root] if root.is_file() else []
    files.extend(sorted((REPO_ROOT / "docker").glob("*.Dockerfile")))
    return files


def logical_lines(text: str) -> list[tuple[int, str]]:
    """Merge backslash continuations so a `curl ... \\` / `| tar` split across
    two physical lines is still one logical line to the stream check."""
    merged: list[tuple[int, str]] = []
    pending_start = 0
    pending = ""

    for lineno, line in enumerate(text.splitlines(), start=1):
        if pending:
            pending = f"{pending} {line.strip()}"
        else:
            pending_start = lineno
            pending = line

        if pending.rstrip().endswith("\\"):
            pending = pending.rstrip()[:-1]
            continue

        merged.append((pending_start, pending))
        pending = ""

    if pending:
        merged.append((pending_start, pending))

    return merged


def main() -> int:
    failures: list[str] = []

    for path in dockerfiles():
        for lineno, line in logical_lines(path.read_text()):
            if EXEMPT in line:
                continue

            if STREAM_PATTERNS.search(line):
                failures.append(
                    f"{path.relative_to(REPO_ROOT)}:{lineno}: downloads piped "
                    "straight into a consumer (no verification step) — fetch to "
                    "a file, checksum-verify, then extract; or exempt with "
                    f"'{EXEMPT}' if another mechanism pins it"
                )

            match = FROM_LINE.match(line)

            if match is not None:
                image = match.group(1)

                if ":" in image and "@sha256:" not in image:
                    failures.append(
                        f"{path.relative_to(REPO_ROOT)}:{lineno}: FROM {image} "
                        "without a `@sha256:` digest — a mutable tag lets the "
                        "base drift under unchanged inputs; pin the digest, or "
                        "feed the FROM an ARG and pin the ARG's default"
                    )

            if ARG_TAG_ONLY.match(line):
                failures.append(
                    f"{path.relative_to(REPO_ROOT)}:{lineno}: ARG default pins "
                    "an image by mutable tag with no `@sha256:` digest"
                )

    for failure in failures:
        print(f"::error {failure}", file=sys.stderr)

    if failures:
        print(
            f"\n{len(failures)} supply-chain guard failure(s). "
            "See .github/workflows/docker-lint.yml (docker-supply-chain job).",
            file=sys.stderr,
        )
        return 1

    print("supply-chain guard: every Dockerfile download verified, every base digest-pinned")
    return 0


if __name__ == "__main__":
    sys.exit(main())
