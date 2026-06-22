"""Wikilink graph traversal for Obsidian vault.

Builds a bidirectional link index from the vault and provides
graph-augmented retrieval: given a seed note, walk its neighbour
links to gather related context (Graph RAG).
"""

from __future__ import annotations

import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Set

from plugins.memory.obsidian.vault import WIKILINK_RE, Note


class WikilinkGraph:
    """Bidirectional wikilink index over a vault."""

    def __init__(self) -> None:
        # title/stem → [outgoing link titles]
        self._forward: Dict[str, List[str]] = defaultdict(list)
        # title/stem → [titles that link IN to this note]
        self._backward: Dict[str, List[str]] = defaultdict(list)
        # stem → resolved path (for lookup)
        self._stem_to_path: Dict[str, Path] = {}

    def build(self, notes: List[Note], vault: Path) -> None:
        """Index all outgoing links. Notes are pre-loaded Note objects."""
        self._forward.clear()
        self._backward.clear()
        self._stem_to_path.clear()

        for note in notes:
            stem = note.path.stem.lower()
            self._stem_to_path[stem] = note.path
            for link in note.links:
                target = link.strip().lower()
                self._forward[stem].append(target)
                self._backward[target].append(stem)

    def add_note(self, note: Note) -> None:
        stem = note.path.stem.lower()
        self._stem_to_path[stem] = note.path
        for link in note.links:
            target = link.strip().lower()
            if target not in self._forward[stem]:
                self._forward[stem].append(target)
            if stem not in self._backward[target]:
                self._backward[target].append(stem)

    def remove_note(self, note: Note) -> None:
        stem = note.path.stem.lower()
        for target in self._forward.get(stem, []):
            self._backward[target] = [
                s for s in self._backward.get(target, []) if s != stem
            ]
        self._forward.pop(stem, None)
        self._stem_to_path.pop(stem, None)

    def neighbours(self, title: str, *, depth: int = 1) -> List[str]:
        """Return all notes reachable within `depth` hops (forward + backward)."""
        stem = title.lower()
        visited: Set[str] = {stem}
        queue: deque[tuple[str, int]] = deque([(stem, 0)])
        result: List[str] = []

        while queue:
            node, d = queue.popleft()
            if d >= depth:
                continue
            for neighbour in (self._forward.get(node, []) + self._backward.get(node, [])):
                if neighbour not in visited:
                    visited.add(neighbour)
                    result.append(neighbour)
                    queue.append((neighbour, d + 1))

        return result

    def backlinks(self, title: str) -> List[str]:
        return list(self._backward.get(title.lower(), []))

    def outlinks(self, title: str) -> List[str]:
        return list(self._forward.get(title.lower(), []))

    def path_for(self, stem: str) -> Optional[Path]:
        return self._stem_to_path.get(stem.lower())

    def hub_notes(self, top_n: int = 10) -> List[tuple[str, int]]:
        """Return the most-linked-to notes by backlink count."""
        counts = [(stem, len(links)) for stem, links in self._backward.items()]
        return sorted(counts, key=lambda x: x[1], reverse=True)[:top_n]
