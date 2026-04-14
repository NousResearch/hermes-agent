#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib

TARGET_GLOBS = [
    "dist/plugin-sdk/pi-embedded-helpers-*.js",
    "dist/pi-embedded-helpers-*.js",
]

INSERT_AFTER = 'if (!trimmed) return "LLM request failed with an unknown error.";'
INSERT_BLOCK = '''
\tconst codexRequestMatch = trimmed.match(/^(?:codex\\s+)?request failed\\s*\\(([^)]+)\\)(?::\\s*([\\s\\S]+))?\\.?$/i);
\tif (codexRequestMatch) {
\t\tconst code = (codexRequestMatch[1] ?? "unknown_error").trim().toLowerCase() || "unknown_error";
\t\tconst detail = (codexRequestMatch[2] ?? "").trim();
\t\tconst combined = `${code} ${detail}`.trim();
\t\tconst transientCopy = formatRateLimitOrOverloadedErrorCopy(combined);
\t\tif (transientCopy) return transientCopy;
\t\tif (isTimeoutErrorMessage(combined) || code === "unknown_error" || code === "server_error" || code === "service_unavailable" || code === "api_connection_error") {
\t\t\treturn "The AI service is temporarily unavailable. Please try again in a moment.";
\t\t}
\t\tif (isAuthErrorMessage(combined)) return "LLM request unauthorized.";
\t\tif (isBillingErrorMessage(combined)) return BILLING_ERROR_USER_MESSAGE;
\t\tif (detail) return `LLM request failed (${code}): ${detail}`;
\t\treturn `LLM request failed (${code}).`;
\t}
'''.rstrip("\n")


def patch_regex_lines(text: str) -> str:
    text = text.replace(
        r"const ERROR_PAYLOAD_PREFIX_RE = /^(?:error|api\s*error|apierror|openai\s*error|anthropic\s*error|gateway\s*error)[:\s-]+/i;",
        r"const ERROR_PAYLOAD_PREFIX_RE = /^(?:error|api\s*error|apierror|openai\s*error|anthropic\s*error|gateway\s*error|codex\s*error|codex\s*request failed)[:\s-]+/i;",
    )
    text = text.replace(
        r"const ERROR_PAYLOAD_PREFIX_RE = /^(?:error|api\s*error|apierror|openai\s*error|anthropic\s*error|gateway\s*error|codex\s*error)[:\s-]+/i;",
        r"const ERROR_PAYLOAD_PREFIX_RE = /^(?:error|api\s*error|apierror|openai\s*error|anthropic\s*error|gateway\s*error|codex\s*error|codex\s*request failed)[:\s-]+/i;",
    )

    text = text.replace(
        r"const ERROR_PREFIX_RE = /^(?:error|api\s*error|openai\s*error|anthropic\s*error|gateway\s*error|request failed|failed|exception)[:\s-]+/i;",
        r"const ERROR_PREFIX_RE = /^(?:error|api\s*error|openai\s*error|anthropic\s*error|gateway\s*error|codex\s*error|codex\s*request failed|request failed|failed|exception)[:\s-]+/i;",
    )
    text = text.replace(
        r"const ERROR_PREFIX_RE = /^(?:error|api\s*error|openai\s*error|anthropic\s*error|gateway\s*error|codex\s*error|request failed|failed|exception)[:\s-]+/i;",
        r"const ERROR_PREFIX_RE = /^(?:error|api\s*error|openai\s*error|anthropic\s*error|gateway\s*error|codex\s*error|codex\s*request failed|request failed|failed|exception)[:\s-]+/i;",
    )

    text = text.replace(
        r"if (isLikelyHttpErrorText(raw) || isRawApiErrorPayload(raw)) return formatRawAssistantErrorForUi(raw);",
        "if (isLikelyHttpErrorText(raw) || isRawApiErrorPayload(raw) || /^(?:codex\\s+)?request failed\\s*\\(/i.test(raw)) return formatRawAssistantErrorForUi(raw);",
    )
    return text


def patch_formatter(text: str) -> str:
    if "const codexRequestMatch = trimmed.match(/^(?:codex\\s+)?request failed" in text:
        return text

    idx = text.find(INSERT_AFTER)
    if idx == -1:
        return text

    insert_at = idx + len(INSERT_AFTER)
    return text[:insert_at] + "\n" + INSERT_BLOCK + text[insert_at:]


def resolve_targets(root: pathlib.Path, explicit_files: list[str]) -> list[pathlib.Path]:
    if explicit_files:
        return [pathlib.Path(p).resolve() for p in explicit_files]

    found: list[pathlib.Path] = []
    for rel_glob in TARGET_GLOBS:
        found.extend(root.glob(rel_glob))
    return sorted(set(found))


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch OpenClaw helper bundles to sanitize Codex request-failed errors.")
    parser.add_argument("--root", default="/usr/lib/node_modules/openclaw", help="OpenClaw package root (default: /usr/lib/node_modules/openclaw)")
    parser.add_argument("--file", action="append", default=[], help="Patch an explicit file path (repeatable).")
    parser.add_argument("--dry-run", action="store_true", help="Report files that would change without writing.")
    args = parser.parse_args()

    targets = resolve_targets(pathlib.Path(args.root), args.file)

    patched: list[str] = []
    for path in targets:
        if not path.exists() or not path.is_file():
            continue
        original = path.read_text(encoding="utf-8")
        updated = patch_regex_lines(original)
        updated = patch_formatter(updated)
        if updated != original:
            if not args.dry_run:
                path.write_text(updated, encoding="utf-8")
            patched.append(str(path))

    print(f"patched_files={len(patched)}")
    for p in patched:
        print(p)


if __name__ == "__main__":
    main()
