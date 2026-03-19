"""
cosmos Quick Notes - Fast Note-Taking Utility

"Taking notes so you don't have to remember everything!"

Features:
- Quick capture with timestamps
- Tag-based organization
- Markdown support
- Search and filter
- Export to various formats
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import hashlib

from loguru import logger


@dataclass
class Note:
    """A quick note entry."""
    id: str
    content: str
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    pinned: bool = False
    archived: bool = False

    def matches(self, query: str) -> bool:
        """Check if note matches search query."""
        query_lower = query.lower()
        return (
            query_lower in self.content.lower() or
            any(query_lower in tag.lower() for tag in self.tags)
        )


class QuickNotes:
    """Fast note-taking manager."""

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir) / "notes"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.notes_file = self.data_dir / "quick_notes.json"
        self.notes: Dict[str, Note] = {}
        self._load()

    def _load(self):
        """Load notes from disk."""
        if self.notes_file.exists():
            try:
                with open(self.notes_file) as f:
                    data = json.load(f)
                for note_data in data.get("notes", []):
                    note = Note(**note_data)
                    self.notes[note.id] = note
            except Exception as e:
                logger.error(f"Failed to load notes: {e}")

    def _save(self):
        """Save notes to disk."""
        try:
            with open(self.notes_file, "w") as f:
                json.dump({
                    "notes": [asdict(n) for n in self.notes.values()],
                    "updated": datetime.now().isoformat(),
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save notes: {e}")

    def add(self, content: str, tags: List[str] = None) -> Note:
        """Add a new quick note."""
        note_id = hashlib.sha256(
            f"{content}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]

        note = Note(
            id=note_id,
            content=content,
            tags=tags or [],
        )
        self.notes[note_id] = note
        self._save()
        logger.info(f"QuickNotes: Added note {note_id[:8]}...")
        return note

    def update(self, note_id: str, content: str = None, tags: List[str] = None) -> Optional[Note]:
        """Update an existing note."""
        if note_id not in self.notes:
            return None

        note = self.notes[note_id]
        if content:
            note.content = content
        if tags is not None:
            note.tags = tags
        note.updated_at = datetime.now().isoformat()
        self._save()
        return note

    def delete(self, note_id: str) -> bool:
        """Delete a note."""
        if note_id in self.notes:
            del self.notes[note_id]
            self._save()
            return True
        return False

    def archive(self, note_id: str) -> bool:
        """Archive a note."""
        if note_id in self.notes:
            self.notes[note_id].archived = True
            self._save()
            return True
        return False

    def pin(self, note_id: str) -> bool:
        """Pin/unpin a note."""
        if note_id in self.notes:
            self.notes[note_id].pinned = not self.notes[note_id].pinned
            self._save()
            return True
        return False

    def search(self, query: str, include_archived: bool = False) -> List[Note]:
        """Search notes by content or tags."""
        results = []
        for note in self.notes.values():
            if not include_archived and note.archived:
                continue
            if note.matches(query):
                results.append(note)
        return sorted(results, key=lambda n: (not n.pinned, n.updated_at), reverse=True)

    def list_by_tag(self, tag: str) -> List[Note]:
        """Get all notes with a specific tag."""
        return [n for n in self.notes.values() if tag.lower() in [t.lower() for t in n.tags]]

    def list_recent(self, limit: int = 10) -> List[Note]:
        """Get most recent notes."""
        active = [n for n in self.notes.values() if not n.archived]
        return sorted(active, key=lambda n: n.updated_at, reverse=True)[:limit]

    def get_all_tags(self) -> List[str]:
        """Get all unique tags."""
        tags = set()
        for note in self.notes.values():
            tags.update(note.tags)
        return sorted(tags)

    def export_markdown(self, output_path: str = None) -> str:
        """Export all notes to markdown."""
        lines = ["# Quick Notes\n"]

        # Group by tags
        by_tag: Dict[str, List[Note]] = {"Untagged": []}
        for note in self.notes.values():
            if note.archived:
                continue
            if not note.tags:
                by_tag["Untagged"].append(note)
            else:
                for tag in note.tags:
                    if tag not in by_tag:
                        by_tag[tag] = []
                    by_tag[tag].append(note)

        for tag, notes in sorted(by_tag.items()):
            if not notes:
                continue
            lines.append(f"\n## {tag}\n")
            for note in sorted(notes, key=lambda n: n.created_at, reverse=True):
                pin_icon = "📌 " if note.pinned else ""
                lines.append(f"- {pin_icon}{note.content}")
                lines.append(f"  - *{note.created_at[:10]}*\n")

        content = "\n".join(lines)

        if output_path:
            Path(output_path).write_text(content)

        return content

    def stats(self) -> Dict[str, Any]:
        """Get notes statistics."""
        active = [n for n in self.notes.values() if not n.archived]
        return {
            "total": len(self.notes),
            "active": len(active),
            "archived": len(self.notes) - len(active),
            "pinned": sum(1 for n in active if n.pinned),
            "tags": len(self.get_all_tags()),
        }


# Global instance
quick_notes = QuickNotes()
