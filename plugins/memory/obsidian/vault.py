"""Obsidian vault I/O: read, write, list notes, frontmatter parsing.

All filesystem access goes through this module so the rest of the plugin
stays decoupled from path handling and encoding edge cases.
"""

from __future__ import annotations

import os
import re
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

WIKILINK_RE = re.compile(r"\[\[([^\[\]|#]+?)(?:[|#][^\[\]]*)?\]\]")
TAG_RE = re.compile(r"(?:^|\s)#([\w/-]+)", re.MULTILINE)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass
class Note:
    path: Path
    title: str
    raw: str                         # full file contents
    frontmatter: Dict[str, Any] = field(default_factory=dict)
    body: str = ""                   # raw minus frontmatter block
    links: List[str] = field(default_factory=list)   # outgoing [[wikilinks]]
    tags: List[str] = field(default_factory=list)
    headings: List[Tuple[int, str]] = field(default_factory=list)  # (level, text)
    mtime: float = 0.0

    @property
    def rel_path(self) -> str:
        return self.path.name  # caller computes relative path if needed

    def chunk_by_heading(self, max_chars: int = 1500) -> List[str]:
        """Split body into chunks at heading boundaries, capped at max_chars."""
        chunks: List[str] = []
        current: List[str] = []
        current_len = 0
        for line in self.body.splitlines(keepends=True):
            if HEADING_RE.match(line) and current_len >= max_chars // 4:
                if current:
                    chunks.append("".join(current).strip())
                current = [line]
                current_len = len(line)
            else:
                current.append(line)
                current_len += len(line)
                if current_len >= max_chars:
                    chunks.append("".join(current).strip())
                    current = []
                    current_len = 0
        if current:
            chunks.append("".join(current).strip())
        return [c for c in chunks if c]


# ---------------------------------------------------------------------------
# Frontmatter helpers
# ---------------------------------------------------------------------------

def _parse_frontmatter(raw: str) -> Tuple[Dict[str, Any], str]:
    m = FRONTMATTER_RE.match(raw)
    if not m:
        return {}, raw
    fm_text = m.group(1)
    body = raw[m.end():]
    if _YAML_AVAILABLE:
        try:
            fm = yaml.safe_load(fm_text) or {}
            if not isinstance(fm, dict):
                fm = {}
        except Exception:
            fm = {}
    else:
        fm = _simple_yaml_parse(fm_text)
    return fm, body


def _simple_yaml_parse(text: str) -> Dict[str, Any]:
    """Minimal YAML parser for simple key: value pairs (no PyYAML fallback)."""
    result: Dict[str, Any] = {}
    for line in text.splitlines():
        if ":" not in line or line.startswith(" ") or line.startswith("-"):
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if val.lower() == "true":
            result[key] = True
        elif val.lower() == "false":
            result[key] = False
        elif val.isdigit():
            result[key] = int(val)
        else:
            result[key] = val
    return result


def _build_frontmatter(fm: Dict[str, Any]) -> str:
    if not fm:
        return ""
    if _YAML_AVAILABLE:
        text = yaml.dump(fm, default_flow_style=False, allow_unicode=True).strip()
    else:
        lines = []
        for k, v in fm.items():
            if isinstance(v, list):
                lines.append(f"{k}:")
                for item in v:
                    lines.append(f"  - {item}")
            elif isinstance(v, bool):
                lines.append(f"{k}: {'true' if v else 'false'}")
            else:
                lines.append(f"{k}: {v}")
        text = "\n".join(lines)
    return f"---\n{text}\n---\n"


# ---------------------------------------------------------------------------
# VaultReader
# ---------------------------------------------------------------------------

class VaultReader:
    """Read notes from an Obsidian vault directory."""

    def __init__(self, vault_path: str | Path) -> None:
        self.vault = Path(vault_path).expanduser().resolve()

    def _load(self, path: Path) -> Note:
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            raw = ""
        fm, body = _parse_frontmatter(raw)
        title = fm.get("title") or path.stem
        links = WIKILINK_RE.findall(raw)
        tags_fm = fm.get("tags", [])
        if isinstance(tags_fm, str):
            tags_fm = [tags_fm]
        tags_inline = TAG_RE.findall(raw)
        tags = list({*tags_fm, *tags_inline})
        headings = [(len(m.group(1)), m.group(2).strip())
                    for m in HEADING_RE.finditer(body)]
        return Note(
            path=path,
            title=title,
            raw=raw,
            frontmatter=fm,
            body=body,
            links=links,
            tags=tags,
            headings=headings,
            mtime=path.stat().st_mtime if path.exists() else 0.0,
        )

    def read(self, rel_or_abs: str | Path) -> Optional[Note]:
        p = Path(rel_or_abs)
        if not p.is_absolute():
            p = self.vault / p
        if not p.exists():
            # fuzzy: search by stem
            candidates = list(self.vault.rglob(f"{p.stem}.md"))
            if not candidates:
                return None
            p = candidates[0]
        return self._load(p)

    def list_all(self, exclude_dirs: tuple[str, ...] = (".obsidian", ".git")) -> List[Path]:
        result = []
        for md in self.vault.rglob("*.md"):
            if any(part in exclude_dirs for part in md.parts):
                continue
            result.append(md)
        return result

    def list_folder(self, folder: str) -> List[Path]:
        target = self.vault / folder
        if not target.is_dir():
            return []
        return sorted(target.rglob("*.md"))

    def load_all(self, limit: int = 2000) -> List[Note]:
        paths = self.list_all()[:limit]
        return [self._load(p) for p in paths]


# ---------------------------------------------------------------------------
# VaultWriter
# ---------------------------------------------------------------------------

class VaultWriter:
    """Create and update notes in an Obsidian vault."""

    def __init__(self, vault_path: str | Path) -> None:
        self.vault = Path(vault_path).expanduser().resolve()

    def _resolve(self, rel: str) -> Path:
        return self.vault / rel

    def write(
        self,
        rel_path: str,
        content: str,
        frontmatter: Optional[Dict[str, Any]] = None,
        *,
        overwrite: bool = True,
    ) -> Path:
        path = self._resolve(rel_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not overwrite:
            return path
        fm_block = _build_frontmatter(frontmatter) if frontmatter else ""
        path.write_text(fm_block + content, encoding="utf-8")
        return path

    def append(self, rel_path: str, content: str, *, separator: str = "\n\n") -> Path:
        path = self._resolve(rel_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            existing = path.read_text(encoding="utf-8")
            path.write_text(existing.rstrip() + separator + content, encoding="utf-8")
        else:
            path.write_text(content, encoding="utf-8")
        return path

    def ensure_daily_note(self, folder: str = "Daily Notes") -> Tuple[Path, bool]:
        """Create today's daily note if it doesn't exist. Returns (path, created)."""
        today = date.today().isoformat()
        rel = f"{folder}/{today}.md"
        path = self._resolve(rel)
        if path.exists():
            return path, False
        fm = {
            "date": today,
            "tags": ["daily"],
            "created": datetime.now().isoformat(timespec="minutes"),
        }
        self.write(rel, f"# {today}\n\n", frontmatter=fm)
        return path, True

    def append_to_daily(self, content: str, *, section: str = "", folder: str = "Daily Notes") -> Path:
        path, _ = self.ensure_daily_note(folder)
        today = date.today().isoformat()
        ts = datetime.now().strftime("%H:%M")
        header = f"\n### {section} — {ts}\n" if section else f"\n### {ts}\n"
        return self.append(f"{folder}/{today}.md", header + content)

    def write_session_note(
        self,
        session_id: str,
        title: str,
        content: str,
        *,
        agent: str = "hermes",
        folder: str = "AI/Hermes/Sessions",
    ) -> Path:
        today = date.today().isoformat()
        safe_id = session_id[:8]
        rel = f"{folder}/{today}-{safe_id}.md"
        fm = {
            "session_id": session_id,
            "agent": agent,
            "date": today,
            "title": title,
            "tags": [f"ai/{agent}", "session"],
        }
        return self.write(rel, content, frontmatter=fm)

    def mirror_memory_entry(
        self,
        action: str,
        target: str,
        content: str,
        *,
        agent: str = "hermes",
    ) -> Path:
        folder = f"AI/{agent.capitalize()}/Memory"
        rel = f"{folder}/{target}.md"
        path = self._resolve(rel)
        ts = datetime.now().isoformat(timespec="minutes")
        entry = f"\n\n<!-- {action} @ {ts} -->\n{content}"
        if action == "remove":
            # Mark removed in frontmatter rather than deleting
            note_raw = path.read_text(encoding="utf-8") if path.exists() else ""
            fm, body = _parse_frontmatter(note_raw)
            fm["archived"] = True
            fm["archived_at"] = ts
            path.write_text(_build_frontmatter(fm) + body, encoding="utf-8")
            return path
        return self.append(rel, entry) if action in ("add", "replace") else path


# ---------------------------------------------------------------------------
# Utility: format a note as a retrieval result
# ---------------------------------------------------------------------------

def format_note_for_context(note: Note, vault: Path, *, max_chars: int = 800) -> str:
    rel = note.path.relative_to(vault) if note.path.is_relative_to(vault) else note.path
    snippet = textwrap.shorten(note.body.strip(), width=max_chars, placeholder="…")
    return f"[[{note.title}]] ({rel})\n{snippet}"
