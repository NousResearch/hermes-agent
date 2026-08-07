#!/usr/bin/env python3
"""
PR template compliance checker.

Validates that a PR body includes required sections and, for AI-assisted PRs,
an appropriate agent disclosure with a generic username placeholder.

Usage:
    python3 check-pr-template-compliance.py <pr-body-file>
    python3 check-pr-template-compliance.py --pr-body "## Summary..."
"""

import argparse
import re
import sys

REQUIRED_SECTIONS = ["Summary", "Test Plan", "Motivation", "Changes"]
DISCLOSURE_NOTE = (
    "> **Note for AI-assisted PRs:** If this PR was authored or assisted by an "
    "AI agent, include a disclosure with the repository username or handle. "
    "Use a generic placeholder format such as `<username>` or `${GITHUB_USERNAME}` "
    "at the end of the PR body:\n"
    ">\n"
    "> ```\n"
    "> AI disclosure: This PR was prepared with assistance from an AI agent "
    "on behalf of `<username>`.\n"
    "> ```\n"
    "> Never substitute a real account name or handle for the placeholder."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check PR body for required sections and compliance."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("pr_file", nargs="?", help="Path to file containing PR body")
    group.add_argument("--pr-body", help="PR body text (inline)")
    return parser.parse_args()


def load_pr_body(args: argparse.Namespace) -> str:
    if args.pr_file:
        try:
            with open(args.pr_file) as f:
                return f.read()
        except FileNotFoundError:
            print(f"❌ File not found: {args.pr_file}", file=sys.stderr)
            sys.exit(1)
    return args.pr_body or ""


def check_sections(body: str) -> list[str]:
    missing = []
    for section in REQUIRED_SECTIONS:
        # Match markdown headings like "## Summary" and "### Summary"
        if not re.search(rf"^#{{1,6}}\s*{section}\s*$", body, re.MULTILINE):
            missing.append(section)
    return missing


def check_disclosure(body: str) -> list[str]:
    """Check that AI disclosure, if present, uses a generic placeholder."""
    issues = []

    PLACEHOLDER_PATTERNS = {
        "<username>", "${GITHUB_USERNAME}", "$USERNAME", "<user>"
    }
    # Words that are expected after trigger phrases (not personal names)
    ALLOWED_WORDS = {"AI", "an", "a", "the", "this", "that"}
    TRIGGER_PHRASES = [
        "Authored by",
        "agent on behalf of",
        "Filed by",
        "AI agent on behalf of",
    ]

    for phrase in TRIGGER_PHRASES:
        pattern = re.compile(
            rf"{re.escape(phrase)}\s+(\S+)",
            re.MULTILINE,
        )
        for m in pattern.finditer(body):
            subject = m.group(1).rstrip(".!?,")
            if (
                subject not in PLACEHOLDER_PATTERNS
                and subject not in ALLOWED_WORDS
                and subject[0:1].isupper()
            ):
                issues.append(
                    f"⚠ Found hardcoded identity \"{subject}\" after "
                    f"\"{phrase}\" — use `<username>` or "
                    f"`${{GITHUB_USERNAME}}` instead."
                )

    return issues


def main() -> None:
    args = parse_args()
    body = load_pr_body(args)

    if not body.strip():
        print("❌ PR body is empty.")
        print("---")
        print("Compliance advice:")
        print(DISCLOSURE_NOTE)
        sys.exit(1)

    exit_code = 0

    missing = check_sections(body)
    if missing:
        print(f"❌ Missing required sections: {', '.join(missing)}")
        exit_code = 1
    else:
        print("✅ All required sections present.")

    disclosure_issues = check_disclosure(body)
    if disclosure_issues:
        for issue in disclosure_issues:
            print(f"❌ {issue}")
        exit_code = 1
        print("---")
        print("Compliance advice:")
        print(DISCLOSURE_NOTE)

    if exit_code == 0:
        print("✅ PR body is compliant.")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
