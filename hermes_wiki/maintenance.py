"""Read-only maintenance reporting for LLM Wiki."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from hermes_wiki.config import WikiConfig
from hermes_wiki.frontmatter import extract_wikilinks, read_page, slugify, write_page
from hermes_wiki.models import IssueSeverity, LintIssue


@dataclass
class MaintenanceReport:
    issues: list[LintIssue] = field(default_factory=list)
    total_pages: int = 0
    total_sources: int = 0
    total_links: int = 0
    broken_links: int = 0
    orphan_pages: int = 0
    pending_proposals: int = 0
    pages_without_sources: int = 0

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == IssueSeverity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == IssueSeverity.WARNING)

    @property
    def info_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == IssueSeverity.INFO)


_PAGE_DIRS = ("entities", "concepts", "comparisons", "queries")
_PROVENANCE_RE = re.compile(r"\^\[(raw/[^\]]+\.md)\]")


def _page_files(config: WikiConfig) -> list[Path]:
    files: list[Path] = []
    for directory_name in _PAGE_DIRS:
        directory = config.wiki_path / directory_name
        if directory.exists():
            files.extend(sorted(directory.glob("*.md")))
    return files


def _source_files(config: WikiConfig) -> list[Path]:
    raw_dir = config.wiki_path / "raw"
    if not raw_dir.exists():
        return []
    return sorted(raw_dir.glob("**/*.md"))


def _relative(config: WikiConfig, path: Path) -> str:
    return str(path.relative_to(config.wiki_path))


def _has_source_coverage(frontmatter: dict[str, Any], body: str) -> bool:
    sources = frontmatter.get("sources")
    if isinstance(sources, list) and any(str(source).strip() for source in sources):
        return True
    return bool(_PROVENANCE_RE.search(body))


def _proposal_files(config: WikiConfig) -> list[Path]:
    proposals_dir = config.wiki_path / "proposals"
    if not proposals_dir.exists():
        return []
    return sorted(proposals_dir.glob("*.md"))


def generate_maintenance_report(config: WikiConfig) -> MaintenanceReport:
    """Generate a read-only wiki maintenance report.

    This scans local wiki markdown files only. It does not write logs, ingest
    sources, reindex vectors, call embeddings, or create directories.
    """

    report = MaintenanceReport()
    pages = _page_files(config)
    sources = _source_files(config)
    report.total_pages = len(pages)
    report.total_sources = len(sources)

    slug_to_path = {path.stem: path for path in pages}
    inbound_links = {slug: set() for slug in slug_to_path}

    for page_path in pages:
        fm, body = read_page(page_path)
        rel_path = _relative(config, page_path)
        links = extract_wikilinks(body)
        report.total_links += len(links)
        for link in links:
            link_slug = slugify(link)
            if link_slug in slug_to_path:
                inbound_links[link_slug].add(page_path.stem)
            else:
                report.broken_links += 1
                report.issues.append(
                    LintIssue(
                        severity=IssueSeverity.ERROR,
                        category="broken_link",
                        message=f"Broken wikilink [[{link}]]",
                        file_path=rel_path,
                        suggestion=f"Create page '{link}' or remove the link",
                    )
                )
        if not _has_source_coverage(fm, body):
            report.pages_without_sources += 1
            report.issues.append(
                LintIssue(
                    severity=IssueSeverity.WARNING,
                    category="missing_source_coverage",
                    message="Page has no frontmatter sources or body provenance markers",
                    file_path=rel_path,
                    suggestion="Add source-backed provenance or re-ingest from a curated source",
                )
            )

    for slug, inbound in inbound_links.items():
        if not inbound:
            report.orphan_pages += 1
            report.issues.append(
                LintIssue(
                    severity=IssueSeverity.WARNING,
                    category="orphan_page",
                    message=f"Page [[{slug}]] has no inbound links",
                    file_path=_relative(config, slug_to_path[slug]),
                    suggestion="Add wikilinks from related pages",
                )
            )

    for proposal_path in _proposal_files(config):
        fm, _body = read_page(proposal_path)
        if fm.get("status", "proposed") in {"proposed", "pending"}:
            report.pending_proposals += 1
            report.issues.append(
                LintIssue(
                    severity=IssueSeverity.INFO,
                    category="pending_proposal",
                    message=f"Pending proposal: {fm.get('title', proposal_path.stem)}",
                    file_path=_relative(config, proposal_path),
                    suggestion="Review, accept into canonical wiki pages, or close",
                )
            )

    return report


def maintenance_report_to_dict(report: MaintenanceReport) -> dict[str, Any]:
    return {
        "total_pages": report.total_pages,
        "total_sources": report.total_sources,
        "total_links": report.total_links,
        "broken_links": report.broken_links,
        "orphan_pages": report.orphan_pages,
        "pending_proposals": report.pending_proposals,
        "pages_without_sources": report.pages_without_sources,
        "errors": report.error_count,
        "warnings": report.warning_count,
        "infos": report.info_count,
        "issues": [
            {
                "severity": issue.severity.value,
                "category": issue.category,
                "message": issue.message,
                "file_path": issue.file_path,
                "suggestion": issue.suggestion,
            }
            for issue in report.issues
        ],
    }


def render_maintenance_report(report: MaintenanceReport) -> str:
    lines = [
        "# LLM Wiki Maintenance Report",
        "",
        "## Summary",
        f"- Pages: {report.total_pages}",
        f"- Sources: {report.total_sources}",
        f"- Links: {report.total_links}",
        f"- Broken links: {report.broken_links}",
        f"- Orphan pages: {report.orphan_pages}",
        f"- Pending proposals: {report.pending_proposals}",
        f"- Pages without source coverage: {report.pages_without_sources}",
        f"- Issues: {len(report.issues)} ({report.error_count} errors, {report.warning_count} warnings, {report.info_count} info)",
        "",
        "## Issues",
    ]
    if not report.issues:
        lines.append("No issues found.")
    else:
        for issue in report.issues:
            file_path = f" `{issue.file_path}`" if issue.file_path else ""
            suggestion = f" — {issue.suggestion}" if issue.suggestion else ""
            lines.append(f"- **{issue.severity.value} / {issue.category}**{file_path}: {issue.message}{suggestion}")
    return "\n".join(lines) + "\n"


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


def _validate_report_path(config: WikiConfig, report_path: str) -> Path:
    rel_path = Path(report_path)
    if rel_path.is_absolute() or ".." in rel_path.parts:
        raise ValueError("--write-report must be a relative path under reports/")
    if not rel_path.parts or rel_path.parts[0] != "reports" or rel_path.suffix != ".md":
        raise ValueError("--write-report must be a relative markdown path under reports/")
    path = (config.wiki_path / rel_path).resolve()
    reports_root = (config.wiki_path / "reports").resolve()
    if path.parent != reports_root and reports_root not in path.parents:
        raise ValueError("--write-report must stay inside reports/")
    if not path.is_relative_to(config.wiki_path.resolve()):
        raise ValueError("--write-report must stay inside the wiki")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a read-only LLM Wiki maintenance report")
    parser.add_argument("--config", help="Hermes config.yaml path to load wiki settings from")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of markdown")
    parser.add_argument("--write-report", help="Explicit relative path under the wiki for a markdown report")
    args = parser.parse_args(argv)

    try:
        config = _build_config(args.config)
        report = generate_maintenance_report(config)
        if args.write_report:
            if not args.config:
                raise ValueError("--write-report requires explicit --config to avoid writing to the wrong wiki")
            path = _validate_report_path(config, args.write_report)
            write_page(
                path,
                {"title": "LLM Wiki Maintenance Report", "type": "maintenance_report", "status": "generated"},
                render_maintenance_report(report),
            )
            print(json.dumps({"written": True, "path": str(path)}, sort_keys=True))
        elif args.json:
            print(json.dumps(maintenance_report_to_dict(report), sort_keys=True))
        else:
            print(render_maintenance_report(report), end="")
    except (FileNotFoundError, ValueError, yaml.YAMLError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
