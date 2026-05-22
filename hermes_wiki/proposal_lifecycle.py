"""Lifecycle operations for queued LLM Wiki memory proposals.

This module manages proposal review artifacts only. It can list, show, and mark
proposal status, but it does not mutate canonical wiki pages, ingest sources, or
reindex vectors.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from hermes_wiki.config import WikiConfig
from hermes_wiki.frontmatter import read_page, slugify, write_page
from hermes_wiki.proposals import _load_explicit_wiki_config

_ALLOWED_STATUSES = {"proposed", "pending", "accepted", "rejected", "closed"}
_MUTATION_COMMANDS = {"accept", "reject", "close"}


@dataclass(frozen=True)
class ProposalRecord:
    """A queued memory proposal loaded from the proposals namespace."""

    slug: str
    title: str
    status: str
    path: Path
    frontmatter: dict[str, Any] = field(default_factory=dict)
    body: str = ""

    @property
    def target_pages(self) -> list[str]:
        value = self.frontmatter.get("target_pages", [])
        return [str(item) for item in value] if isinstance(value, list) else []

    @property
    def source_refs(self) -> list[str]:
        value = self.frontmatter.get("source_refs", [])
        return [str(item) for item in value] if isinstance(value, list) else []


def _proposal_root(config: WikiConfig) -> Path:
    return (config.wiki_path / "proposals").resolve()


def _safe_slug(raw_slug: str) -> str:
    original = str(raw_slug or "").strip()
    slug = slugify(original)
    if not slug or Path(original).is_absolute() or ".." in Path(original).parts or "/" in original or "\\" in original:
        raise ValueError("unsafe proposal slug")
    return slug


def _proposal_path(config: WikiConfig, slug: str) -> Path:
    safe = _safe_slug(slug)
    root = _proposal_root(config)
    path = (root / f"{safe}.md").resolve()
    if path.parent != root:
        raise ValueError("unsafe proposal slug")
    return path


def _record_from_path(config: WikiConfig, path: Path) -> ProposalRecord:
    root = _proposal_root(config)
    resolved = path.resolve()
    if resolved.parent != root or resolved.suffix != ".md":
        raise ValueError("proposal path must stay under proposals/")
    fm, body = read_page(resolved)
    slug = str(fm.get("slug") or resolved.stem).strip() or resolved.stem
    status = str(fm.get("status", "proposed") or "proposed")
    title = str(fm.get("title") or slug).strip()
    return ProposalRecord(slug=slug, title=title, status=status, path=resolved, frontmatter=fm, body=body)


def list_proposals(config: WikiConfig, *, status: str | None = None) -> list[ProposalRecord]:
    """List queued proposal records sorted by slug."""

    root = config.wiki_path / "proposals"
    if not root.exists():
        return []
    records = [_record_from_path(config, path) for path in sorted(root.glob("*.md"))]
    if status:
        records = [record for record in records if record.status == status]
    return sorted(records, key=lambda record: record.slug)


def read_proposal(config: WikiConfig, slug: str) -> ProposalRecord:
    """Read one proposal by safe slug."""

    path = _proposal_path(config, slug)
    if not path.exists():
        raise FileNotFoundError(f"Proposal not found: {slug}")
    return _record_from_path(config, path)


def update_proposal_status(config: WikiConfig, slug: str, status: str, *, note: str | None = None) -> ProposalRecord:
    """Update proposal frontmatter status only.

    This intentionally preserves the proposal body and never writes canonical wiki
    pages. Accepting a proposal means the review artifact has been accepted by a
    separate curated workflow; it does not imply this helper applied page edits.
    """

    normalized = str(status or "").strip().lower()
    if normalized not in _ALLOWED_STATUSES:
        raise ValueError(f"unsupported proposal status: {status}")
    record = read_proposal(config, slug)
    fm = dict(record.frontmatter)
    fm["status"] = normalized
    fm["reviewed"] = date.today().isoformat()
    if note:
        fm["review_note"] = str(note).strip()
    write_page(record.path, fm, record.body)
    return read_proposal(config, slug)


def proposal_record_to_dict(record: ProposalRecord, *, include_body: bool = False) -> dict[str, Any]:
    payload = {
        "slug": record.slug,
        "title": record.title,
        "status": record.status,
        "path": str(record.path),
        "target_pages": record.target_pages,
        "source_refs": record.source_refs,
        "created": record.frontmatter.get("created"),
        "reviewed": record.frontmatter.get("reviewed"),
        "review_note": record.frontmatter.get("review_note"),
    }
    if include_body:
        payload["body"] = record.body
    return payload


def render_proposal_record(record: ProposalRecord) -> str:
    return record.body if record.body else f"# {record.title}\n"


def _build_config(config_path: str | None) -> WikiConfig:
    return _load_explicit_wiki_config(config_path) if config_path else WikiConfig.from_hermes_config()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="List, show, and review LLM Wiki memory proposals")
    parser.add_argument("--config", help="Hermes config.yaml path to load wiki settings from")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List queued proposals")
    list_parser.add_argument("--status", help="Optional proposal status filter")
    list_parser.add_argument("--json", action="store_true", help="Emit JSON")

    show_parser = subparsers.add_parser("show", help="Show a proposal")
    show_parser.add_argument("slug")
    show_parser.add_argument("--json", action="store_true", help="Emit JSON")

    for command, help_text in (
        ("accept", "Mark a proposal accepted"),
        ("reject", "Mark a proposal rejected"),
        ("close", "Mark a proposal closed"),
    ):
        status_parser = subparsers.add_parser(command, help=help_text)
        status_parser.add_argument("slug")
        status_parser.add_argument("--note", help="Optional review note")

    args = parser.parse_args(argv)

    try:
        if args.command in _MUTATION_COMMANDS and not args.config:
            raise ValueError(f"{args.command} requires explicit --config to avoid writing to the wrong wiki")
        config = _build_config(args.config)
        if args.command == "list":
            records = list_proposals(config, status=args.status)
            if args.json:
                print(json.dumps([proposal_record_to_dict(record) for record in records], sort_keys=True))
            else:
                for record in records:
                    print(f"{record.slug}\t{record.status}\t{record.title}")
        elif args.command == "show":
            record = read_proposal(config, args.slug)
            if args.json:
                print(json.dumps(proposal_record_to_dict(record, include_body=True), sort_keys=True))
            else:
                print(render_proposal_record(record), end="")
        elif args.command == "accept":
            print(json.dumps(proposal_record_to_dict(update_proposal_status(config, args.slug, "accepted", note=args.note)), sort_keys=True))
        elif args.command == "reject":
            print(json.dumps(proposal_record_to_dict(update_proposal_status(config, args.slug, "rejected", note=args.note)), sort_keys=True))
        elif args.command == "close":
            print(json.dumps(proposal_record_to_dict(update_proposal_status(config, args.slug, "closed", note=args.note)), sort_keys=True))
    except (FileNotFoundError, ValueError, yaml.YAMLError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
