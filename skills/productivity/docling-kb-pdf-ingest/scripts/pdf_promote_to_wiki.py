from __future__ import annotations

import argparse
import json
from pathlib import Path

from pdf_ingest_config import PROMOTION_CANDIDATE_FILENAME, WIKI_SOURCES_HOME
from pdf_ingest_lib import slugify


def _clean_lines(markdown_text: str) -> list[str]:
    return [line.strip() for line in markdown_text.splitlines() if line.strip() and line.strip() != "<!-- image -->"]


def _summary(markdown_text: str) -> str:
    for line in _clean_lines(markdown_text):
        if not line.startswith("#"):
            return line[:220]
    return "No summary available."


def _highlights(markdown_text: str, limit: int = 5) -> list[str]:
    picks: list[str] = []
    for line in _clean_lines(markdown_text):
        if line.startswith("#"):
            picks.append(line.lstrip("#").strip())
        elif len(line) > 40:
            picks.append(line[:180])
        if len(picks) >= limit:
            break
    return picks or ["No reliable highlights extracted."]


def source_page_slug(source_record: dict) -> str:
    return slugify(Path(source_record["original_filename"]).stem)


def source_page_path(source_record: dict) -> Path:
    return WIKI_SOURCES_HOME / f"{source_page_slug(source_record)}.md"


def build_promotion_candidate(*, source_record: dict, markdown_text: str) -> str:
    title = Path(source_record["original_filename"]).stem.replace("-", " ").replace("_", " ").strip() or source_record["original_filename"]
    summary = _summary(markdown_text)
    raw_rel = Path(source_record["raw_path"])
    return (
        f"# {title}\n\n"
        f"> {summary}\n\n"
        f"## Source metadata\n"
        f"- source_id: `{source_record['source_id']}`\n"
        f"- parser: `{source_record['parser_selected']}`\n"
        f"- route_reason: {source_record['route_reason']}\n"
        f"- raw_path: `{raw_rel}`\n\n"
        f"## Suggested action\n"
        f"- Source summary target: `wiki/sources/{source_page_slug(source_record)}.md`\n"
        f"- Review extracted markdown for citation-safe facts before promoting into concepts/entities.\n"
    )


def build_source_summary_page(*, source_record: dict, markdown_text: str) -> str:
    slug = source_page_slug(source_record)
    title = slug.replace("-", " ").strip().title()
    summary = _summary(markdown_text)
    highlights = _highlights(markdown_text)
    raw_rel = Path(source_record["raw_path"])
    lines = [
        "---",
        f"title: \"{title}\"",
        f"created: {source_record['parse_completed_at'][:10]}",
        f"updated: {source_record['parse_completed_at'][:10]}",
        "type: source",
        f"tags: [source, pdf, kb-ingest, {source_record['parser_selected']}]",
        f"sources: [\"{raw_rel}\"]",
        "---",
        "",
        f"# {title}",
        "",
        f"> {summary}",
        "",
        "## Content",
        "",
        f"This source page was auto-promoted from the PDF evidence layer. Parser: {source_record['parser_selected']}. Route reason: {source_record['route_reason']}",
        "",
        "## Highlights",
        "",
    ]
    for item in highlights:
        lines.append(f"- {item}")
    lines.extend([
        "",
        "## Related",
        "",
        "- [[research-cycle-log]]",
        "",
        "## Sources",
        "",
        f"- [{raw_rel.name}](../../raw/pdf/{raw_rel.parent.name}/{raw_rel.name})",
        f"- Evidence record: `{source_record['source_id']}`",
        "",
        "## Changelog",
        "",
        f"- {source_record['parse_completed_at'][:10]}: Auto-promoted source summary from PDF evidence ingest using {source_record['parser_selected']}.",
        "",
    ])
    return "\n".join(lines)


def write_promotion_candidate(*, output_dir: Path, source_record: dict, markdown_text: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / PROMOTION_CANDIDATE_FILENAME
    target.write_text(build_promotion_candidate(source_record=source_record, markdown_text=markdown_text), encoding="utf-8")
    return target


def promote_source_summary(*, source_record: dict, markdown_text: str) -> Path:
    WIKI_SOURCES_HOME.mkdir(parents=True, exist_ok=True)
    target = source_page_path(source_record)
    target.write_text(build_source_summary_page(source_record=source_record, markdown_text=markdown_text), encoding="utf-8")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write PDF promotion artifacts")
    parser.add_argument("source_record_json")
    parser.add_argument("markdown_path")
    parser.add_argument("output_dir")
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()

    source_record = json.loads(Path(args.source_record_json).read_text(encoding="utf-8"))
    markdown_text = Path(args.markdown_path).read_text(encoding="utf-8")
    payload = {
        "promotion_candidate_path": str(write_promotion_candidate(output_dir=Path(args.output_dir), source_record=source_record, markdown_text=markdown_text))
    }
    if args.promote:
        payload["source_page_path"] = str(promote_source_summary(source_record=source_record, markdown_text=markdown_text))
    print(json.dumps(payload, indent=2))
