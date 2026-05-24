"""Curator extension for domain knowledge notes in the 3-Tier Knowledge Center.

Extends the existing curator system to scan, review, and archive domain notes.
Also manages usage tracking for review priority ordering.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_USAGE_FILE = "knowledge_usage.json"


class DomainNoteCurator:
    """Curates domain knowledge notes — scans for stale notes and archives them."""

    def __init__(self, vault_path: Optional[Path] = None) -> None:
        if vault_path is None:
            vault_path = Path.home() / "ObsidianVault" / "HermesAgent"
        self.vault_path = Path(vault_path)
        self.domains_dir = self.vault_path / "domains"
        self.archive_dir = self.domains_dir / ".archive"

    def scan_domain_notes(self) -> List[Dict[str, Any]]:
        """Scan all domain notes and return metadata.

        Excludes README.md, index.md, mapping.md, and .review_queue.json.

        Returns:
            List of dicts with note metadata (path, domain, title, modified, frontmatter)
        """
        notes: List[Dict[str, Any]] = []
        if not self.domains_dir.exists():
            return notes

        for domain_dir in sorted(self.domains_dir.iterdir()):
            if not domain_dir.is_dir():
                continue
            if domain_dir.name.startswith("."):
                continue  # Skip .archive, etc.

            for note_file in sorted(domain_dir.glob("*.md")):
                if note_file.name in ("README.md",):
                    continue
                try:
                    text = note_file.read_text(encoding="utf-8")
                    # Extract frontmatter
                    fm_match = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
                    frontmatter = {}
                    if fm_match:
                        for line in fm_match.group(1).split("\n"):
                            if ":" in line:
                                key, val = line.split(":", 1)
                                frontmatter[key.strip()] = val.strip()

                    notes.append({
                        "path": str(note_file),
                        "domain": domain_dir.name,
                        "title": frontmatter.get("title", note_file.stem),
                        "modified": datetime.fromtimestamp(note_file.stat().st_mtime).strftime("%Y-%m-%d"),
                        "origin_project": frontmatter.get("origin_project", ""),
                        "promoted_at": frontmatter.get("promoted_at", ""),
                        "status": frontmatter.get("status", ""),
                        "frontmatter": frontmatter,
                    })
                except Exception as e:
                    logger.debug("Error reading %s: %s", note_file, e)

        return notes

    def is_agent_created(self, note_meta: Dict[str, Any]) -> bool:
        """Check if a note was created by the agent (has origin_project field)."""
        return bool(note_meta.get("origin_project"))

    def mark_stale(self, notes: List[Dict[str, Any]], stale_after_days: int = 30) -> List[Dict[str, Any]]:
        """Mark notes as stale based on age.

        Args:
            notes: List of note metadata
            stale_after_days: Days after which a note is considered stale

        Returns:
            List of stale note metadata
        """
        stale: List[Dict[str, Any]] = []
        cutoff = datetime.now()
        from datetime import timedelta
        cutoff = cutoff - timedelta(days=stale_after_days)

        for note in notes:
            try:
                modified = datetime.strptime(note["modified"], "%Y-%m-%d")
                if modified < cutoff:
                    note["stale"] = True
                    stale.append(note)
            except ValueError:
                pass  # Skip notes with invalid dates

        return stale

    def archive_note(self, note_path: str) -> Optional[str]:
        """Archive a domain note.

        Args:
            note_path: Path to the note to archive

        Returns:
            New archived path, or None on failure
        """
        src = Path(note_path)
        if not src.exists():
            return None

        self.archive_dir.mkdir(parents=True, exist_ok=True)
        dest = self.archive_dir / src.name
        # Handle duplicates
        counter = 1
        while dest.exists():
            dest = self.archive_dir / f"{src.stem}-{counter}{src.suffix}"
            counter += 1

        try:
            src.rename(dest)
            return str(dest)
        except Exception as e:
            logger.error("Failed to archive %s: %s", note_path, e)
            return None

    def restore_note(self, archived_path: str, target_domain: Optional[str] = None) -> Optional[str]:
        """Restore an archived note back to its domain directory.

        Args:
            archived_path: Path to the archived note
            target_domain: Target domain (auto-detected from frontmatter if not provided)

        Returns:
            New restored path, or None on failure
        """
        src = Path(archived_path)
        if not src.exists():
            return None

        # Determine target domain
        if not target_domain:
            try:
                text = src.read_text(encoding="utf-8")
                tag_match = re.search(r"tags:\s*\n\s*-\s*(\w+)", text)
                if tag_match:
                    target_domain = tag_match.group(1)
            except Exception:
                pass

        if not target_domain:
            return None

        domain_dir = self.domains_dir / target_domain
        domain_dir.mkdir(parents=True, exist_ok=True)
        dest = domain_dir / src.name

        try:
            src.rename(dest)
            return str(dest)
        except Exception as e:
            logger.error("Failed to restore %s: %s", archived_path, e)
            return None


class KnowledgeUsageTracker:
    """Tracks usage of domain knowledge notes for curator priority ordering."""

    def __init__(self, hermes_home: Optional[Path] = None) -> None:
        if hermes_home is None:
            hermes_home = get_hermes_home()
        self._usage_path = hermes_home / _USAGE_FILE
        self._usage: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if self._usage_path.exists():
            try:
                text = self._usage_path.read_text(encoding="utf-8")
                self._usage = json.loads(text)
            except (json.JSONDecodeError, Exception):
                self._usage = {}

    def _save(self) -> None:
        self._usage_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._usage_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._usage, indent=2), encoding="utf-8")
        tmp.rename(self._usage_path)

    def record_view(self, note_id: str, domain: str, origin_project: str) -> None:
        """Record a view of a knowledge note."""
        if note_id not in self._usage:
            self._usage[note_id] = {
                "view_count": 0,
                "use_count": 0,
                "last_viewed_at": "",
                "last_used_at": "",
                "domain": domain,
                "origin_project": origin_project,
            }
        self._usage[note_id]["view_count"] += 1
        self._usage[note_id]["last_viewed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._save()

    def record_use(self, note_id: str, domain: str, origin_project: str) -> None:
        """Record actual use of a knowledge note (agent referenced it in work)."""
        if note_id not in self._usage:
            self._usage[note_id] = {
                "view_count": 0,
                "use_count": 0,
                "last_viewed_at": "",
                "last_used_at": "",
                "domain": domain,
                "origin_project": origin_project,
            }
        self._usage[note_id]["use_count"] += 1
        self._usage[note_id]["last_used_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._save()

    def get_usage(self, note_id: str) -> Optional[Dict[str, Any]]:
        """Get usage data for a note."""
        return self._usage.get(note_id)

    def get_all_usage(self) -> Dict[str, Dict[str, Any]]:
        """Get all usage data."""
        return dict(self._usage)

    def get_priority_score(self, note_id: str, modified_date: str) -> float:
        """Calculate review priority score.

        Higher score = higher priority for review.
        Popular (high view count) but stale (old modified date) notes get highest priority.

        Args:
            note_id: Note identifier
            modified_date: Last modified date (YYYY-MM-DD)

        Returns:
            Priority score (0.0–1.0)
        """
        usage = self._usage.get(note_id, {})
        view_count = usage.get("view_count", 0)
        use_count = usage.get("use_count", 0)

        # Age factor: older = higher priority (up to a cap)
        try:
            modified = datetime.strptime(modified_date, "%Y-%m-%d")
            age_days = (datetime.now() - modified).days
            age_factor = min(age_days / 90, 1.0)  # Cap at 90 days
        except ValueError:
            age_factor = 0.5

        # Popularity factor
        total_views = view_count + use_count
        pop_factor = min(total_views / 10, 1.0)  # Cap at 10 views

        # Combined: 60% age, 40% popularity
        return 0.6 * age_factor + 0.4 * pop_factor
