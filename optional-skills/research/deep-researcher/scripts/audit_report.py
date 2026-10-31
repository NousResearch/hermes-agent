#!/usr/bin/env python3
"""Audit a deep-research report's citation structure, coverage, and links."""

from __future__ import annotations

import argparse
import concurrent.futures
import http.client
import json
import re
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import NamedTuple
from urllib.parse import urlsplit

from evidence_ledger import canonical_url
from report_links import first_url

BIBLIOGRAPHY_HEADING = re.compile(
    r"^#{2,4}\s+(?:Citation Links|References|Bibliography)\s*$", re.IGNORECASE
)
CITATION_MARKER = re.compile(r"\[([0-9][0-9,\-–\s]*)\]")
NUMBERED_ENTRY = re.compile(r"^(\d+)\.\s+(.*)$")
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
        if not BIBLIOGRAPHY_HEADING.match(line.strip()):
            continue
        level = len(line.lstrip()) - len(line.lstrip().lstrip('#'))
        end = len(lines)
        for following in range(index + 1, len(lines)):
            if re.match(r'^#{1,' + str(level) + r'}\s+', lines[following].strip()):
                end = following
                break
        return '\n'.join(lines[:index] + lines[end:]), lines[index + 1:end]
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


def citations_in(text: str, known_numbers: set[int] | None = None) -> set[int]:
    result: set[int] = set()
    for match in CITATION_MARKER.finditer(text):
        group = match.group(1).strip()
        if group.isdigit() and 1800 <= int(group) <= 2199 and int(group) not in (known_numbers or set()):
            continue
        result.update(expand_citation_group(group))
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
        url = first_url(match.group(2))
        if not url:
            errors.append(f"Bibliography entry [{number}] has no HTTP(S) URL.")
            continue
        if number in entries:
            errors.append(f"Bibliography entry [{number}] is duplicated.")
        entries[number] = url
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
    except (OSError, ValueError, http.client.HTTPException) as exc:  # URLError, timeouts, bad URLs
        return LinkResult(number, url, "warning", f"unverified: {str(exc)[:100]}")


def ledger_urls(path: Path) -> set[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        source.get("url") or source.get("canonical_url")
        for source in data.get("sources", [])
    }


def normalize_for_match(url: str) -> str:
    return canonical_url(url)


def _audit_citations(
    cited: set[int], entries: dict[int, str], bibliography_lines: list[str],
    errors: list[str], warnings: list[str]
) -> None:
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



def _audit_source_distribution(
    entries: dict[int, str], errors: list[str], warnings: list[str]
) -> Counter[str]:
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

    return domains


def _audit_uncited_content(body: str, max_warnings: int, warnings: list[str]) -> None:
    analysis_text = analytical_body(body)
    factual = candidate_factual_lines(analysis_text)
    for line_number, preview in factual[: max_warnings]:
        warnings.append(
            f"Possibly uncited factual claim at line {line_number}: {preview}"
        )
    if len(factual) > max_warnings:
        warnings.append(
            f"{len(factual) - max_warnings} additional possibly uncited lines suppressed."
        )
    for line_number, preview in uncited_analytical_rows(analysis_text)[
        : max_warnings
    ]:
        warnings.append(
            f"Possibly uncited analytical table row at line {line_number}: {preview}"
        )



def _audit_ledger(entries: dict[int, str], ledger_path: str, errors: list[str]) -> None:
    try:
        evidence_urls = {
            normalize_for_match(url)
            for url in ledger_urls(Path(ledger_path).expanduser())
            if url
        }
        for number, url in entries.items():
            if normalize_for_match(url) not in evidence_urls:
                errors.append(
                    f"Bibliography entry [{number}] is absent from the evidence ledger."
                )
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        errors.append(f"Could not read evidence ledger: {exc}")



def _audit_live_links(
    entries: dict[int, str], args: argparse.Namespace, errors: list[str], warnings: list[str]
) -> list[LinkResult]:
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

    return link_results


def _print_audit(
    path: Path, cited: set[int], entries: dict[int, str], domains: Counter[str],
    link_results: list[LinkResult], errors: list[str], warnings: list[str]
) -> None:
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


def run(args: argparse.Namespace) -> int:
    path = Path(args.report).expanduser()
    if not path.exists():
        print(f"ERROR: report not found: {path}")
        return 1
    body, bibliography_lines = split_document(path.read_text(encoding="utf-8"))
    entries, parse_errors = bibliography_entries(bibliography_lines)
    cited = citations_in(body, set(entries))
    errors = list(parse_errors)
    warnings: list[str] = []
    _audit_citations(cited, entries, bibliography_lines, errors, warnings)
    domains = _audit_source_distribution(entries, errors, warnings)
    _audit_uncited_content(body, args.max_heuristic_warnings, warnings)
    if args.ledger:
        _audit_ledger(entries, args.ledger, errors)
    link_results = _audit_live_links(entries, args, errors, warnings)
    _print_audit(path, cited, entries, domains, link_results, errors, warnings)
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
