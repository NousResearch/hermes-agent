#!/usr/bin/env python3
"""Create and validate a compact evidence ledger for deep research."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

SOURCE_TYPES = ("primary", "secondary", "commentary", "community")
CLAIM_STATUSES = ("supported", "disputed", "single-source", "unsupported")
TRACKING_KEYS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ref",
    "source",
}


def canonical_url(value: str) -> str:
    """Normalize a URL enough to catch common duplicate-source entries."""
    parts = urlsplit(value.strip())
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise ValueError(f"Expected an absolute HTTP(S) URL: {value}")
    host = parts.hostname.lower() if parts.hostname else ""
    port = parts.port
    if port and not (
        (parts.scheme == "http" and port == 80)
        or (parts.scheme == "https" and port == 443)
    ):
        host = f"{host}:{port}"
    query = urlencode(
        sorted(
            (key, val)
            for key, val in parse_qsl(parts.query, keep_blank_values=True)
            if not key.lower().startswith("utm_") and key.lower() not in TRACKING_KEYS
        )
    )
    path = re.sub(r"/{2,}", "/", parts.path or "/")
    if path != "/":
        path = path.rstrip("/")
    return urlunsplit((parts.scheme.lower(), host, path, query, ""))


def read_ledger(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"ERROR: ledger does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"ERROR: invalid ledger JSON: {exc}") from exc
    if not isinstance(data, dict) or data.get("version") != 1:
        raise SystemExit("ERROR: unsupported or malformed evidence ledger")
    return data


def write_ledger(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    handle, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as temp:
            json.dump(data, temp, indent=2, ensure_ascii=False)
            temp.write("\n")
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def cmd_init(args: argparse.Namespace) -> None:
    path = Path(args.ledger).expanduser()
    if path.exists() and not args.force:
        raise SystemExit(
            f"ERROR: ledger already exists: {path} (use --force to replace)"
        )
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    data = {
        "version": 1,
        "topic": args.topic,
        "created_at": now,
        "updated_at": now,
        "research_questions": args.question,
        "sources": [],
        "claims": [],
    }
    write_ledger(path, data)
    print(f"Initialized {path} with {len(args.question)} research question(s).")


def cmd_add_source(args: argparse.Namespace) -> None:
    path = Path(args.ledger).expanduser()
    data = read_ledger(path)
    normalized = canonical_url(args.url)
    for source in data["sources"]:
        if source["canonical_url"] == normalized:
            raise SystemExit(
                f"ERROR: duplicate source URL already has ID {source['id']}"
            )
    source_id = max((item["id"] for item in data["sources"]), default=0) + 1
    source = {
        "id": source_id,
        "url": args.url,
        "canonical_url": normalized,
        "title": args.title,
        "publisher": args.publisher or urlsplit(normalized).hostname,
        "author": args.author or "",
        "published_date": args.published_date or "",
        "accessed_date": args.accessed_date
        or datetime.now(timezone.utc).date().isoformat(),
        "source_type": args.source_type,
        "independence_group": args.independence_group or None,
        "independence_assessed": bool(args.independence_group and args.independence_note),
        "independence_note": args.independence_note or "",
        "questions": args.question,
        "extraction_status": args.extraction_status,
        "summary": args.summary or "",
        "excerpts": args.excerpt,
        "limitations": args.limitation,
        "evidence": [],
    }
    for index, value in enumerate(args.evidence, 1):
        try:
            item = json.loads(value)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"ERROR: invalid evidence JSON: {exc}") from exc
        if not isinstance(item, dict) or any(
            not isinstance(item.get(key), str) or not item[key].strip()
            for key in ("excerpt", "locator")
        ):
            raise SystemExit("ERROR: evidence needs nonempty excerpt and locator strings")
        source["evidence"].append({
            "id": f"S{source_id}E{index}",
            "excerpt": item["excerpt"], "locator": item["locator"],
        })
    data["sources"].append(source)
    write_ledger(path, data)
    print(f"Added source [{source_id}] {args.title}")


def parse_source_ids(values: list[str]) -> list[int]:
    result: list[int] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                result.append(int(item))
    return sorted(set(result))


def cmd_add_claim(args: argparse.Namespace) -> None:
    path = Path(args.ledger).expanduser()
    data = read_ledger(path)
    claim_number = (
        max(
            (
                int(item["id"][1:])
                for item in data["claims"]
                if re.fullmatch(r"C\d+", item["id"])
            ),
            default=0,
        )
        + 1
    )
    claim = {
        "id": f"C{claim_number:03d}",
        "text": args.text,
        "question": args.question,
        "importance": args.importance,
        "status": args.status,
        "source_ids": parse_source_ids(args.source),
        "counter_source_ids": parse_source_ids(args.counter_source),
        "reasoning": args.reasoning or "",
        "caveat": args.caveat or "",
        "evidence_ids": args.evidence,
    }
    data["claims"].append(claim)
    write_ledger(path, data)
    print(f"Added claim {claim['id']} ({claim['status']})")


def validate(data: dict[str, Any]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    source_ids = [source.get("id") for source in data.get("sources", [])]
    source_id_set = set(source_ids)
    if len(source_ids) != len(source_id_set):
        errors.append("Duplicate source IDs exist.")
    canonical_seen: dict[str, int] = {}
    for source in data.get("sources", []):
        if source.get("source_type") not in SOURCE_TYPES:
            errors.append(
                f"Source [{source.get('id', '?')}] has invalid source_type {source.get('source_type')!r}."
            )
        try:
            normalized = canonical_url(source.get("url", ""))
        except ValueError as exc:
            errors.append(f"Source [{source.get('id', '?')}]: {exc}")
            continue
        if normalized in canonical_seen:
            errors.append(
                f"Sources [{canonical_seen[normalized]}] and [{source.get('id')}] duplicate {normalized}."
            )
        canonical_seen[normalized] = source.get("id")
        if source.get("extraction_status") != "extracted":
            warnings.append(
                f"Source [{source.get('id')}] was not successfully extracted."
            )
        if not source.get("summary"):
            warnings.append(
                f"Source [{source.get('id')}] has no compact evidence summary."
            )
        elif len(source["summary"]) > 1200:
            warnings.append(
                f"Source [{source.get('id')}] summary exceeds 1,200 characters; compact it."
            )
        if any(len(excerpt) > 500 for excerpt in source.get("excerpts", [])):
            warnings.append(
                f"Source [{source.get('id')}] has an excerpt over 500 characters; retain only decisive wording."
            )
    question_set = set(data.get("research_questions", []))
    question_sources = {question: set() for question in question_set}
    for source in data.get("sources", []):
        for question in source.get("questions", []):
            if question not in question_set:
                warnings.append(
                    f"Source [{source.get('id')}] references unknown question {question!r}."
                )
            else:
                question_sources[question].add(source.get("id"))
    for question, ids in question_sources.items():
        if len(ids) < 2:
            warnings.append(
                f"Research question {question!r} has fewer than two extracted sources."
            )
    claim_ids = [claim.get("id") for claim in data.get("claims", [])]
    if len(claim_ids) != len(set(claim_ids)):
        errors.append("Duplicate claim IDs exist.")
    source_by_id = {source.get("id"): source for source in data.get("sources", [])}
    for claim in data.get("claims", []):
        claim_id = claim.get("id", "?")
        supporting = set(claim.get("source_ids", []))
        counter = set(claim.get("counter_source_ids", []))
        missing = (supporting | counter) - source_id_set
        if missing:
            errors.append(
                f"Claim {claim_id} references missing source IDs: {sorted(missing)}."
            )
            continue
        if claim.get("question") not in question_set:
            errors.append(
                f"Claim {claim_id} references unknown question {claim.get('question')!r}."
            )
        if claim.get("status") not in CLAIM_STATUSES:
            errors.append(
                f"Claim {claim_id} has invalid status {claim.get('status')!r}."
            )
        if supporting & counter:
            errors.append(
                f"Claim {claim_id} lists source IDs as both support and counter-evidence: "
                f"{sorted(supporting & counter)}."
            )
        unusable = {
            source_id
            for source_id in supporting | counter
            if source_by_id[source_id].get("extraction_status") != "extracted"
        }
        if unusable:
            errors.append(
                f"Claim {claim_id} cites sources that were not extracted: {sorted(unusable)}."
            )
        if claim.get("status") == "unsupported" and supporting:
            warnings.append(
                f"Claim {claim_id} is unsupported but lists supporting sources."
            )
        if claim.get("status") != "unsupported" and not supporting:
            errors.append(
                f"Claim {claim_id} has status {claim.get('status')!r} but no supporting source."
            )
        groups = {
            source.get("independence_group")
            for source in data.get("sources", [])
            if source.get("id") in supporting
            and source.get("independence_assessed") is True
            and source.get("independence_note", "").strip()
            and source.get("independence_group")
        }
        if claim.get("importance") == "major" and len(groups) < 2:
            warnings.append(
                f"Major claim {claim_id} lacks two independent supporting sources."
            )
        if claim.get("status") == "disputed" and not counter:
            warnings.append(
                f"Disputed claim {claim_id} has no counter-evidence source."
            )
    evidence_by_id = {}
    for source in data.get("sources", []):
        if source.get("independence_assessed") is not True:
            warnings.append(f"Source [{source.get('id')}] independence is unverified.")
        for item in source.get("evidence", []):
            evidence_id = item.get("id")
            if not evidence_id or evidence_id in evidence_by_id:
                errors.append(f"Missing or duplicate evidence ID: {evidence_id!r}.")
            evidence_by_id[evidence_id] = source.get("id")
            if not item.get("excerpt") or not item.get("locator"):
                errors.append(f"Evidence {evidence_id} needs an excerpt and locator.")
            if len(item.get("excerpt", "")) > 500:
                warnings.append(f"Evidence {evidence_id} exceeds 500 characters; keep only the relevant passage.")
    for claim in data.get("claims", []):
        cited_sources = set(claim.get("source_ids", []) + claim.get("counter_source_ids", []))
        if cited_sources and not claim.get("evidence_ids"):
            warnings.append(f"Claim {claim.get('id')} has no precise evidence links.")
        for evidence_id in claim.get("evidence_ids", []):
            if evidence_id not in evidence_by_id or evidence_by_id[evidence_id] not in cited_sources:
                errors.append(f"Claim {claim.get('id')} references unknown or unrelated evidence {evidence_id}.")
        located_sources = {evidence_by_id[item] for item in claim.get("evidence_ids", []) if item in evidence_by_id}
        if cited_sources - located_sources:
            warnings.append(f"Claim {claim.get('id')} lacks precise passages for sources {sorted(cited_sources - located_sources)}.")
    return errors, warnings


def cmd_check(args: argparse.Namespace) -> None:
    data = read_ledger(Path(args.ledger).expanduser())
    errors, warnings = validate(data)
    print(
        f"Ledger: {len(data.get('sources', []))} source(s), "
        f"{len(data.get('claims', []))} claim(s), "
        f"{len(data.get('research_questions', []))} question(s)"
    )
    for message in errors:
        print(f"ERROR: {message}")
    for message in warnings:
        print(f"WARN: {message}")
    if not errors and not warnings:
        print("PASS: ledger structure and evidence coverage checks passed.")
    raise SystemExit(1 if errors else 0)


def render_brief(args: argparse.Namespace) -> None:
    data = read_ledger(Path(args.ledger).expanduser())
    sources = {source["id"]: source for source in data.get("sources", [])}
    print(f"# Evidence brief: {data.get('topic', '')}\n")
    for question in data.get("research_questions", []):
        print(f"## {question}")
        matching = [
            claim
            for claim in data.get("claims", [])
            if claim.get("question") == question
        ]
        if not matching:
            print("- GAP: no claims recorded")
        for claim in matching:
            refs = (
                ", ".join(f"[{item}]" for item in claim.get("source_ids", [])) or "none"
            )
            counter = ", ".join(
                f"[{item}]" for item in claim.get("counter_source_ids", [])
            )
            suffix = f"; counter: {counter}" if counter else ""
            print(
                f"- {claim['id']} ({claim['status']}, {claim['importance']}): {claim['text']} — {refs}{suffix}"
            )
            if claim.get("caveat"):
                print(f"  Caveat: {claim['caveat']}")
            if claim.get("evidence_ids"):
                print(f"  Evidence: {', '.join(claim['evidence_ids'])}")
        print()
    print("## Source notes")
    for source_id in sorted(sources):
        source = sources[source_id]
        print(
            f"- [{source_id}] {source['title']} ({source['source_type']}): {source.get('summary') or 'NO SUMMARY'}"
        )
        print(f"  Independence: {source.get('independence_group') if source.get('independence_assessed') else 'unverified'}")
        for limitation in source.get("limitations", []):
            print(f"  Limitation: {limitation}")
        for item in source.get("evidence", []):
            if any(item.get("id") in claim.get("evidence_ids", []) for claim in data.get("claims", [])):
                print(f"  {item['id']} @ {item['locator']}: {item['excerpt']}")


def cmd_brief(args: argparse.Namespace) -> None:
    if args.max_chars <= 0:
        raise SystemExit("ERROR: --max-chars must be positive")
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        render_brief(args)
    text = output.getvalue()
    if len(text) <= args.max_chars:
        print(text, end="")
    else:
        notice = "\n[BRIEF TRUNCATED: read relevant ledger records before synthesis.]\n"
        if args.max_chars < len(notice):
            raise SystemExit(f"ERROR: --max-chars must be at least {len(notice)} when truncation is needed")
        print(text[:args.max_chars - len(notice)] + notice, end="")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="Create a new ledger")
    init.add_argument("ledger")
    init.add_argument("--topic", required=True)
    init.add_argument("--question", action="append", required=True)
    init.add_argument("--force", action="store_true")
    init.set_defaults(func=cmd_init)

    source = subparsers.add_parser("add-source", help="Add an extracted source")
    source.add_argument("ledger")
    source.add_argument("--url", required=True)
    source.add_argument("--title", required=True)
    source.add_argument("--publisher")
    source.add_argument("--author")
    source.add_argument("--published-date")
    source.add_argument("--accessed-date")
    source.add_argument("--source-type", choices=SOURCE_TYPES, required=True)
    source.add_argument("--independence-group")
    source.add_argument("--independence-note", help="Why this source belongs to the stated origin group")
    source.add_argument("--evidence", action="append", default=[], help='JSON with excerpt and locator; repeatable')
    source.add_argument("--question", action="append", default=[])
    source.add_argument(
        "--extraction-status",
        choices=("pending", "extracted", "failed"),
        default="extracted",
    )
    source.add_argument("--summary")
    source.add_argument("--excerpt", action="append", default=[])
    source.add_argument("--limitation", action="append", default=[])
    source.set_defaults(func=cmd_add_source)

    claim = subparsers.add_parser(
        "add-claim", help="Add a claim and its evidence links"
    )
    claim.add_argument("ledger")
    claim.add_argument("--text", required=True)
    claim.add_argument("--question", required=True)
    claim.add_argument("--importance", choices=("major", "minor"), default="major")
    claim.add_argument("--status", choices=CLAIM_STATUSES, required=True)
    claim.add_argument("--source", action="append", default=[])
    claim.add_argument("--counter-source", action="append", default=[])
    claim.add_argument("--reasoning")
    claim.add_argument("--caveat")
    claim.add_argument("--evidence", action="append", default=[], help="Supporting or opposing evidence ID; repeatable")
    claim.set_defaults(func=cmd_add_claim)

    check = subparsers.add_parser(
        "check", help="Validate evidence structure and coverage"
    )
    check.add_argument("ledger")
    check.set_defaults(func=cmd_check)

    brief = subparsers.add_parser(
        "brief", help="Render a context-efficient evidence brief"
    )
    brief.add_argument("ledger")
    brief.add_argument("--max-chars", type=int, default=24000, help="Maximum brief characters, including truncation notice")
    brief.set_defaults(func=cmd_brief)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
