from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Optional

from hermes_wiki.config import WikiConfig
from hermes_wiki.frontmatter import read_page
from hermes_wiki.models import LogEntry


class WikiIndexer:
    """Manages index.md and log.md for wiki navigation."""

    def __init__(self, config: WikiConfig):
        self.config = config

    def read_index(self) -> str:
        if self.config.index_path.exists():
            return self.config.index_path.read_text(encoding="utf-8")
        return ""

    def get_all_pages(self) -> dict[str, list[dict]]:
        """Scan filesystem and return all wiki pages grouped by type."""
        pages: dict[str, list[dict]] = {
            "entities": [],
            "concepts": [],
            "comparisons": [],
            "queries": [],
        }

        for page_type, dir_path in [
            ("entities", self.config.entities_dir),
            ("concepts", self.config.concepts_dir),
            ("comparisons", self.config.comparisons_dir),
            ("queries", self.config.queries_dir),
        ]:
            if not dir_path.exists():
                continue
            for md_file in sorted(dir_path.glob("*.md")):
                fm, _ = read_page(md_file)
                pages[page_type].append({
                    "slug": md_file.stem,
                    "title": fm.get("title", md_file.stem),
                    "tags": fm.get("tags", []),
                    "updated": str(fm.get("updated", "")),
                    "path": str(md_file.relative_to(self.config.wiki_path)),
                    "confidence": fm.get("confidence"),
                    "contested": fm.get("contested", False),
                })

        return pages

    def rebuild_index(self) -> int:
        """Rebuild index.md from filesystem. Returns total page count."""
        pages = self.get_all_pages()
        total = sum(len(v) for v in pages.values())

        sections = [
            f"# Wiki Index\n",
            f"> Content catalog. Every wiki page listed under its type with a one-line summary.",
            f"> Read this first to find relevant pages for any query.",
            f"> Last updated: {date.today().isoformat()} | Total pages: {total}\n",
        ]

        for section_name, entries in [
            ("Entities", pages["entities"]),
            ("Concepts", pages["concepts"]),
            ("Comparisons", pages["comparisons"]),
            ("Queries", pages["queries"]),
        ]:
            sections.append(f"\n## {section_name}\n")
            if entries:
                for entry in entries:
                    tags_str = f" [{', '.join(entry['tags'])}]" if entry['tags'] else ""
                    contested = " ⚠️ CONTESTED" if entry.get('contested') else ""
                    sections.append(f"- [[{entry['slug']}]] — {entry['title']}{tags_str}{contested}")
            else:
                sections.append("_No pages yet._")

        self.config.index_path.write_text("\n".join(sections) + "\n", encoding="utf-8")
        return total

    def add_to_index(self, slug: str, title: str, page_type: str, tags: list[str] | None = None) -> None:
        """Add a single entry to index.md without full rebuild."""
        if not self.config.index_path.exists():
            self.rebuild_index()
            return

        content = self.config.index_path.read_text(encoding="utf-8")

        if f"[[{slug}]]" in content:
            return

        type_to_section = {
            "entity": "## Entities",
            "concept": "## Concepts",
            "comparison": "## Comparisons",
            "query": "## Queries",
        }
        section_header = type_to_section.get(page_type, "## Entities")

        tags_str = f" [{', '.join(tags)}]" if tags else ""
        new_entry = f"- [[{slug}]] — {title}{tags_str}"

        if section_header in content:
            no_pages = "_No pages yet._"
            if no_pages in content.split(section_header)[1].split("\n##")[0]:
                content = content.replace(
                    f"{section_header}\n\n{no_pages}",
                    f"{section_header}\n\n{new_entry}",
                )
            else:
                pos = content.find(section_header)
                next_section = content.find("\n## ", pos + len(section_header))
                if next_section == -1:
                    content = content.rstrip() + f"\n{new_entry}\n"
                else:
                    content = content[:next_section] + f"{new_entry}\n" + content[next_section:]
        else:
            content = content.rstrip() + f"\n\n{section_header}\n\n{new_entry}\n"

        total_match = re.search(r"Total pages: (\d+)", content)
        if total_match:
            old_count = int(total_match.group(1))
            content = content.replace(
                f"Total pages: {old_count}",
                f"Total pages: {old_count + 1}",
            )

        content = re.sub(
            r"Last updated: \d{4}-\d{2}-\d{2}",
            f"Last updated: {date.today().isoformat()}",
            content,
        )

        self.config.index_path.write_text(content, encoding="utf-8")

    def append_log(self, action: str, subject: str, details: list[str] | None = None) -> None:
        """Append an entry to log.md."""
        entry = LogEntry(
            date=date.today(),
            action=action,
            subject=subject,
            details=details or [],
        )

        log_text = ""
        if self.config.log_path.exists():
            log_text = self.config.log_path.read_text(encoding="utf-8")

        log_text = log_text.rstrip() + "\n\n" + entry.to_markdown() + "\n"

        entry_count = log_text.count("## [")
        if entry_count > self.config.log_rotation_threshold:
            year = date.today().year
            archive_name = f"log-{year}.md"
            archive_path = self.config.wiki_path / archive_name
            suffix = 2
            while archive_path.exists():
                archive_name = f"log-{year}-{suffix}.md"
                archive_path = self.config.wiki_path / archive_name
                suffix += 1
            archive_path.write_text(log_text, encoding="utf-8")
            log_text = (
                "# Wiki Log\n\n"
                "> Chronological record of all wiki actions. Append-only.\n"
                "> Format: `## [YYYY-MM-DD] action | subject`\n"
                f"> Previous logs archived to {archive_name}\n\n"
                f"## [{date.today().isoformat()}] rotate | Log rotated, {entry_count} entries archived\n"
            )

        self.config.log_path.write_text(log_text, encoding="utf-8")

    def recent_log_entries(self, n: int = 20) -> str:
        """Get the last N log entries as text."""
        if not self.config.log_path.exists():
            return "No log entries yet."

        text = self.config.log_path.read_text(encoding="utf-8")
        entries = re.split(r"(?=^## \[)", text, flags=re.MULTILINE)
        entries = [e.strip() for e in entries if e.strip().startswith("## [")]
        recent = entries[-n:] if len(entries) > n else entries
        return "\n\n".join(recent) if recent else "No log entries yet."

    def list_entity_slugs(self) -> list[str]:
        """List all entity page slugs."""
        if not self.config.entities_dir.exists():
            return []
        return [f.stem for f in sorted(self.config.entities_dir.glob("*.md"))]

    def list_concept_slugs(self) -> list[str]:
        """List all concept page slugs."""
        if not self.config.concepts_dir.exists():
            return []
        return [f.stem for f in sorted(self.config.concepts_dir.glob("*.md"))]

    def list_all_slugs(self) -> list[str]:
        """List all wiki page slugs across all types."""
        slugs = []
        for subdir in ["entities", "concepts", "comparisons", "queries"]:
            dir_path = self.config.wiki_path / subdir
            if dir_path.exists():
                slugs.extend(f.stem for f in dir_path.glob("*.md"))
        return sorted(slugs)

    def list_sources(self) -> list[dict]:
        """List all raw sources with metadata."""
        sources = []
        for subdir in ["articles", "papers", "transcripts"]:
            dir_path = self.config.raw_dir / subdir
            if not dir_path.exists():
                continue
            for f in sorted(dir_path.glob("*.md")):
                fm, _ = read_page(f)
                sources.append({
                    "path": str(f.relative_to(self.config.wiki_path)),
                    "name": f.stem,
                    "source_url": fm.get("source_url"),
                    "ingested": str(fm.get("ingested", "")),
                    "subdir": subdir,
                })
        return sources
