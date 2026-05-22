"""Safe memory proposal helpers for LLM Wiki.

A proposal is a review artifact, not durable accepted memory. The helpers here
render and optionally queue explicit proposals without ingesting, reindexing, or
modifying canonical wiki pages.
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
from hermes_wiki.frontmatter import slugify, write_page


@dataclass(frozen=True)
class MemoryProposal:
    """A proposed durable-memory update awaiting review."""

    title: str
    rationale: str
    proposed_changes: list[str]
    source_refs: list[str]
    target_pages: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    slug: str | None = None
    status: str = "proposed"


def _clean_list(values: list[str] | tuple[str, ...] | None) -> list[str]:
    return [str(value).strip() for value in values or [] if str(value).strip()]


def _validate_relative_page_paths(paths: list[str], *, field_name: str) -> list[str]:
    cleaned: list[str] = []
    for raw_path in paths:
        page = str(raw_path).strip()
        if not page:
            continue
        candidate = Path(page)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"{field_name} must contain relative wiki page paths")
        cleaned.append(page)
    return cleaned


def _proposal_slug(proposal: MemoryProposal) -> str:
    raw_slug = proposal.slug if proposal.slug is not None else proposal.title
    slug = slugify(raw_slug)
    if not slug or Path(str(raw_slug)).is_absolute() or ".." in Path(str(raw_slug)).parts:
        raise ValueError("unsafe proposal slug")
    return slug


def proposal_to_dict(proposal: MemoryProposal) -> dict[str, Any]:
    """Return stable, JSON-serializable proposal metadata."""

    title = str(proposal.title).strip()
    rationale = str(proposal.rationale).strip()
    proposed_changes = _clean_list(proposal.proposed_changes)
    source_refs = _clean_list(proposal.source_refs)
    target_pages = _validate_relative_page_paths(_clean_list(proposal.target_pages), field_name="target_pages")
    tags = _clean_list(proposal.tags)
    slug = _proposal_slug(proposal)

    if not title:
        raise ValueError("proposal title is required")
    if not rationale:
        raise ValueError("proposal rationale is required")
    if not proposed_changes:
        raise ValueError("proposal must include at least one proposed change")
    if not source_refs:
        raise ValueError("proposal must include at least one source reference")

    return {
        "type": "memory_proposal",
        "title": title,
        "slug": slug,
        "status": str(proposal.status or "proposed"),
        "created": date.today().isoformat(),
        "target_pages": target_pages,
        "source_refs": source_refs,
        "tags": tags,
    }


def render_proposal_markdown(proposal: MemoryProposal) -> str:
    """Render a human-reviewable memory proposal body."""

    metadata = proposal_to_dict(proposal)
    changes = _clean_list(proposal.proposed_changes)
    source_refs = metadata["source_refs"]
    target_pages = metadata["target_pages"]

    sections = [
        f"# {metadata['title']}",
        "",
        "## Rationale",
        str(proposal.rationale).strip(),
        "",
        "## Proposed changes",
        *(f"- {change}" for change in changes),
        "",
        "## Source references",
        *(f"- {ref}" for ref in source_refs),
        "",
        "## Target pages",
    ]
    if target_pages:
        sections.extend(f"- {page}" for page in target_pages)
    else:
        sections.append("- New page or reviewer-selected target")
    sections.extend(
        [
            "",
            "## Review checklist",
            "- [ ] Source-backed",
            "- [ ] Durable beyond the current week",
            "- [ ] Not a raw transcript dump",
            "- [ ] Safe to ingest into canonical wiki pages",
        ]
    )
    return "\n".join(sections).strip() + "\n"


def queue_proposal(config: WikiConfig, proposal: MemoryProposal, *, write: bool = False) -> Path:
    """Return or explicitly write a pending proposal path under `<wiki>/proposals`.

    With `write=False` this is a dry-run path calculation. With `write=True` the
    proposal is written as a review artifact only; canonical wiki pages and the
    vector index are not changed.
    """

    metadata = proposal_to_dict(proposal)
    root = (config.wiki_path / "proposals").resolve()
    path = (root / f"{metadata['slug']}.md").resolve()
    if path.parent != root:
        raise ValueError("unsafe proposal slug")
    if write:
        body = render_proposal_markdown(proposal)
        write_page(path, metadata, body)
    return path


def _load_explicit_wiki_config(config_path: str | Path) -> WikiConfig:
    path = Path(config_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Hermes config not found: {config_path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Hermes config must be a mapping: {config_path}")
    if "wiki" not in data:
        raise ValueError(f"Hermes config has no wiki section: {config_path}")
    return WikiConfig.from_dict(data)


def _build_config(config_path: str | None) -> WikiConfig:
    return _load_explicit_wiki_config(config_path) if config_path else WikiConfig.from_hermes_config()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Draft or queue an LLM Wiki memory proposal")
    parser.add_argument("--title", required=True, help="Proposal title")
    parser.add_argument("--rationale", required=True, help="Why this memory should exist")
    parser.add_argument("--change", action="append", required=True, help="Proposed durable-memory change")
    parser.add_argument("--source", action="append", required=True, help="Source reference supporting the proposal")
    parser.add_argument("--target", action="append", default=[], help="Relative wiki page path the proposal may update")
    parser.add_argument("--tag", action="append", default=[], help="Proposal tag")
    parser.add_argument("--slug", help="Optional proposal slug")
    parser.add_argument("--config", help="Hermes config.yaml path to load wiki settings from")
    parser.add_argument("--queue", action="store_true", help="Write the proposal under <wiki>/proposals")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of markdown when not queueing")
    args = parser.parse_args(argv)

    try:
        proposal = MemoryProposal(
            title=args.title,
            rationale=args.rationale,
            proposed_changes=args.change,
            source_refs=args.source,
            target_pages=args.target,
            tags=args.tag,
            slug=args.slug,
        )
        if args.queue:
            if not args.config:
                raise ValueError("--queue requires explicit --config to avoid writing to the wrong wiki")
            path = queue_proposal(_build_config(args.config), proposal, write=True)
            print(json.dumps({"queued": True, "path": str(path)}, sort_keys=True))
        elif args.json:
            print(json.dumps(proposal_to_dict(proposal), sort_keys=True))
        else:
            print(render_proposal_markdown(proposal), end="")
    except (FileNotFoundError, ValueError, yaml.YAMLError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
