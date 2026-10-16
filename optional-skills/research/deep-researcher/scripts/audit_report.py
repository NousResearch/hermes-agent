#!/usr/bin/env python3
"""Audit a deep-research report's citation structure, coverage, and links."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import NamedTuple
from urllib.parse import urlsplit

BIBLIOGRAPHY_HEADING = re.compile(
    r"^#{2,4}\s+(?:Citation Links|References|Bibliography)\s*$", re.IGNORECASE
)
CITATION_MARKER = re.compile(r"\[([0-9][0-9,\-–\s]*)\]")
NUMBERED_ENTRY = re.compile(r"^(\d+)\.\s+(.*)$")
MARKDOWN_URL = re.compile(r"\]\((https?://[^)]+)\)")
BARE_URL = re.compile(r"(https?://\S+)")
FACT_SIGNAL = re.compile(
    r"(?:\b(?:18|19|20)\d{2}\b|\b\d+(?:\.\d+)?%\b|\$\s?\d|"
    r"\b\d+(?:\.\d+)?\s*(?:million|billion|trillion|percent|users?|stars?|days?|years?)\b|"
    r"\b(?:increased|decreased|launched|released|founded|acquired|outperform(?:s|ed)?|"
    r"largest|smallest|fastest|slowest|leading|only)\b)",
    re.IGNORECASE,
)


class LinkResult(NamedTuple):
    number: int
    url: str
    severity: str
    message: str


def split_document(text: str) -> tuple[str, list[str]]:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if BIBLIOGRAPHY_HEADING.match(line.strip()):
            return "\n".join(lines[:index]), lines[index + 1 :]
    return text, []


def analytical_body(text: str) -> str:
    """Exclude source inventories and methodology from heuristic claim checks."""
    return re.split(
        r"^##\s+(?:Sources analyzed|Methodology|Appendix|Considered but excluded or failed)\s*$",
        text,
        maxsplit=1,
        flags=re.IGNORECASE | re.MULTILINE,
    )[0]


def expand_citation_group(group: str) -> set[int]:
    result: set[int] = set()
    for part in re.split(r"\s*,\s*", group.replace("–", "-")):
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            if left.strip().isdigit() and right.strip().isdigit():
                start, end = int(left), int(right)
                if 0 < start <= end <= start + 100:
                    result.update(range(start, end + 1))
        elif part.strip().isdigit():
            result.add(int(part))
    return result


def citations_in(text: str) -> set[int]:
    result: set[int] = set()
    for match in CITATION_MARKER.finditer(text):
        result.update(expand_citation_group(match.group(1)))
    return result


def bibliography_entries(lines: list[str]) -> tuple[dict[int, str], list[str]]:
    entries: dict[int, str] = {}
    errors: list[str] = []
    for raw in lines:
        stripped = raw.strip()
        match = NUMBERED_ENTRY.match(stripped)
        if not match:
            continue
        number = int(match.group(1))
        url_match = MARKDOWN_URL.search(match.group(2)) or BARE_URL.search(
            match.group(2)
        )
        if not url_match:
            errors.append(f"Bibliography entry [{number}] has no HTTP(S) URL.")
            continue
        if number in entries:
            errors.append(f"Bibliography entry [{number}] is duplicated.")
        entries[number] = url_match.group(1).rstrip(".,")
    return entries, errors


def candidate_factual_lines(body: str) -> list[tuple[int, str]]:
    warnings: list[tuple[int, str]] = []
    in_fence = False
    for number, raw in enumerate(body.splitlines(), start=1):
        stripped = raw.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or not stripped or stripped.startswith(("#", ">")):
            continue
        if stripped.startswith("|") and stripped.endswith("|"):
            continue
        if re.fullmatch(r"\|?[\s:|-]+\|?", stripped):
            continue
        if FACT_SIGNAL.search(stripped) and not citations_in(stripped):
            warnings.append((number, stripped[:180]))
    return warnings


def uncited_analytical_rows(body: str) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    for number, raw in enumerate(body.splitlines(), start=1):
        stripped = raw.strip()
        if not (stripped.startswith("|") and stripped.endswith("|")):
            continue
        if re.fullmatch(r"\|?[\s:|-]+\|?", stripped):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) < 3 or all(len(cell.split()) <= 3 for cell in cells):
            continue
        if not citations_in(stripped) and not any(
            cell.lower() in {"source", "sources"} for cell in cells
        ):
            rows.append((number, stripped[:180]))
    return rows


def check_url(item: tuple[int, str], timeout: int) -> LinkResult:
    number, url = item
    request = urllib.request.Request(url, method="GET")
    request.add_header("User-Agent", "HermesDeepResearchAudit/2.0")
    request.add_header("Range", "bytes=0-1023")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read(1)
            status = getattr(response, "status", 200)
            return LinkResult(number, url, "ok", f"HTTP {status}")
    except urllib.error.HTTPError as exc:
        if exc.code in {404, 410}:
            return LinkResult(number, url, "error", f"HTTP {exc.code} (not found)")
        if exc.code in {401, 403, 405, 408, 425, 429} or 500 <= exc.code < 600:
            return LinkResult(
                number, url, "warning", f"HTTP {exc.code} (blocked or transient)"
            )
        return LinkResult(number, url, "warning", f"HTTP {exc.code}")
    except Exception as exc:  # noqa: BLE001
        return LinkResult(number, url, "warning", f"unverified: {str(exc)[:100]}")


def ledger_urls(path: Path) -> set[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        source.get("canonical_url") or source.get("url")
        for source in data.get("sources", [])
    }


def normalize_for_match(url: str) -> str:
    parts = urlsplit(url)
    path = (parts.path or "/").rstrip("/") or "/"
    return f"{parts.scheme.lower()}://{parts.netloc.lower()}{path}"


def run(args: argparse.Namespace) -> int:
    path = Path(args.report).expanduser()
    if not path.exists():
        print(f"ERROR: report not found: {path}")
        return 1
    body, bibliography_lines = split_document(path.read_text(encoding="utf-8"))
    cited = citations_in(body)
    entries, parse_errors = bibliography_entries(bibliography_lines)
    errors = list(parse_errors)
    warnings: list[str] = []

    if not bibliography_lines:
        errors.append(
            "No 'Citation Links', 'References', or 'Bibliography' section found."
        )
    if not cited:
        errors.append("No numbered citations were found in the report body.")
    missing = sorted(cited - set(entries))
    unused = sorted(set(entries) - cited)
    for number in missing:
        errors.append(
            f"Citation [{number}] is used in the body but missing from the bibliography."
        )
    for number in unused:
        warnings.append(
            f"Bibliography entry [{number}] is never cited in the report body."
        )
    if entries:
        expected = set(range(1, max(entries) + 1))
        gaps = sorted(expected - set(entries))
        if gaps:
            warnings.append(f"Bibliography numbering has gaps: {gaps}.")

    domains = Counter(urlsplit(url).hostname or "" for url in entries.values())
    if len(entries) >= 5 and domains:
        domain, count = domains.most_common(1)[0]
        if count / len(entries) > 0.5:
            warnings.append(
                f"Source concentration: {count}/{len(entries)} bibliography entries are from {domain}."
            )
    duplicate_urls: dict[str, list[int]] = {}
    for number, url in entries.items():
        duplicate_urls.setdefault(normalize_for_match(url), []).append(number)
    for url, numbers in duplicate_urls.items():
        if len(numbers) > 1:
            errors.append(f"Duplicate bibliography URL in entries {numbers}: {url}")

    analysis_text = analytical_body(body)
    factual = candidate_factual_lines(analysis_text)
    for line_number, preview in factual[: args.max_heuristic_warnings]:
        warnings.append(
            f"Possibly uncited factual claim at line {line_number}: {preview}"
        )
    if len(factual) > args.max_heuristic_warnings:
        warnings.append(
            f"{len(factual) - args.max_heuristic_warnings} additional possibly uncited lines suppressed."
        )
    for line_number, preview in uncited_analytical_rows(analysis_text)[
        : args.max_heuristic_warnings
    ]:
        warnings.append(
            f"Possibly uncited analytical table row at line {line_number}: {preview}"
        )

    if args.ledger:
        try:
            evidence_urls = {
                normalize_for_match(url)
                for url in ledger_urls(Path(args.ledger).expanduser())
                if url
            }
            for number, url in entries.items():
                if normalize_for_match(url) not in evidence_urls:
                    errors.append(
                        f"Bibliography entry [{number}] is absent from the evidence ledger."
                    )
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            errors.append(f"Could not read evidence ledger: {exc}")

    link_results: list[LinkResult] = []
    if not args.no_live and entries:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(args.workers, len(entries))
        ) as executor:
            link_results = sorted(
                executor.map(
                    lambda item: check_url(item, args.timeout), entries.items()
                ),
                key=lambda item: item.number,
            )
        for result in link_results:
            message = f"Link [{result.number}] {result.message}: {result.url}"
            if result.severity == "error":
                errors.append(message)
            elif result.severity == "warning":
                warnings.append(message)

    print(f"Report audit: {path.name}")
    print(f"  Body citations: {len(cited)} unique")
    print(f"  Bibliography:   {len(entries)} entries across {len(domains)} domain(s)")
    if link_results:
        print(
            f"  Live links:     {sum(item.severity == 'ok' for item in link_results)}/{len(link_results)} verified"
        )
    for message in errors:
        print(f"ERROR: {message}")
    for message in warnings:
        print(f"WARN: {message}")
    if not errors and not warnings:
        print("PASS: all deterministic report checks passed.")
    elif not errors:
        print(f"PASS WITH WARNINGS: {len(warnings)} item(s) require review.")
    else:
        print(f"FAIL: {len(errors)} error(s), {len(warnings)} warning(s).")
    return 1 if errors else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", help="Markdown report to audit")
    parser.add_argument("--ledger", help="Optional evidence-ledger JSON to cross-check")
    parser.add_argument("--no-live", action="store_true", help="Skip HTTP checks")
    parser.add_argument(
        "--timeout", type=int, default=10, help="Per-link timeout in seconds"
    )
    parser.add_argument("--workers", type=int, default=8, help="Concurrent link checks")
    parser.add_argument("--max-heuristic-warnings", type=int, default=12)
    return parser


def main() -> None:
    sys.exit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
