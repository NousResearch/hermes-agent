#!/usr/bin/env python3
"""Proposal Output Validator — verifies mashup-review HTML proposals before delivery.

Checks, per proposal file:
  1. Non-empty file, reasonable size
  2. At least one <section id="slug"> (max 5)
  3. Slug uniqueness
  4. Mandatory sections present: Concept, Value Prop, Trade-offs, Content Drafts,
     Recommendation
  5. Content draft completeness: Tweet, LinkedIn, Blog angle
  6. Every recommendation has effort (S/M/L), risk (Low/Medium/High), priority
  7. No raw placeholder text ("SLUG-HERE", "TBD", "lorem", "REPLACE_ME")
  8. At least one source link (arxiv.org or github.com)

Exit code 0 = all checks pass.  Exit code 1 = validation errors found.
The mashup-review cron prompt can append this as a post-check.

Usage:
  python3 proposal_validator.py [path-to-proposal.html ...]
  python3 proposal_validator.py --latest   # validate newest mashup-*.html
"""
import argparse
import re
import sys
from pathlib import Path

DEFAULT_GLOB = "mashup-*.html"
MAX_PROPOSALS = 5
PLACEHOLDER_RE = re.compile(
    r"(SLUG-HERE|REPLACE_ME|TBD|lorem ipsum|TODO|\[N\]|\.\.\.)", re.IGNORECASE
)
SECTION_RE = re.compile(r'<section\s+id="([^"]+)"[^>]*>(.*?)</section>', re.DOTALL | re.IGNORECASE)


def find_proposal_files(paths, base_dir=None):
    """Resolve explicit paths, or glob the default directory."""
    base = Path(base_dir or (Path.home() / ".hermes" / "runbooks" / "proposals"))
    if paths:
        return [Path(p) for p in paths if Path(p).exists()]
    if not base.exists():
        return []
    return sorted(base.glob(DEFAULT_GLOB), reverse=True)


def validate_file(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return [f"cannot read {path}: {e}"]

    if len(text) < 500:
        errors.append(f"{path.name}: file suspiciously small ({len(text)} bytes)")

    sections = SECTION_RE.findall(text)
    if not sections:
        errors.append(f"{path.name}: no <section id=\"...\"> found — proposals missing")
        return errors
    if len(sections) > MAX_PROPOSALS:
        errors.append(f"{path.name}: {len(sections)} proposals exceeds max {MAX_PROPOSALS}")

    seen_slugs = {}
    for slug, body in sections:
        if slug in seen_slugs:
            errors.append(f"{path.name}: duplicate slug '{slug}'")
        seen_slugs[slug] = True

        for required in ("Concept", "Value Prop", "Trade-offs", "Content Drafts", "Recommendation"):
            if required.lower() not in body.lower():
                errors.append(f"{path.name}: section '{slug}' missing '{required}'")

        for draft in ("Tweet", "LinkedIn", "Blog angle"):
            if draft.lower() not in body.lower():
                errors.append(f"{path.name}: section '{slug}' missing content draft '{draft}'")

        if not re.search(r"Effort:\s*[SML]", body, re.IGNORECASE):
            errors.append(f"{path.name}: section '{slug}' missing Effort S/M/L")
        if not re.search(r"Risk:\s*(Low|Medium|High)", body, re.IGNORECASE):
            errors.append(f"{path.name}: section '{slug}' missing Risk Low/Medium/High")
        if not re.search(r"Priority:\s*\d+", body, re.IGNORECASE):
            errors.append(f"{path.name}: section '{slug}' missing Priority")

        if not re.search(r"arxiv\.org|github\.com", body, re.IGNORECASE):
            errors.append(f"{path.name}: section '{slug}' missing source link (arxiv/github)")

    for m in PLACEHOLDER_RE.finditer(text):
        errors.append(f"{path.name}: placeholder text '{m.group(1)}' present")
        break  # one per file is enough

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("paths", nargs="*", help="Explicit proposal HTML paths (default: latest mashup-*.html)")
    parser.add_argument("--base", default=None, help="Proposals directory (default: ~/.hermes/runbooks/proposals)")
    args = parser.parse_args()

    files = find_proposal_files(args.paths, args.base)
    if not files:
        print("proposal_validator: no proposal files found")
        return 1

    all_errors = []
    for f in files:
        all_errors.extend(validate_file(f))

    if all_errors:
        for err in all_errors:
            print(f"FAIL: {err}")
        print(f"proposal_validator: {len(all_errors)} issue(s) in {len(files)} file(s)")
        return 1

    print(f"proposal_validator: OK — {len(files)} file(s) valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
