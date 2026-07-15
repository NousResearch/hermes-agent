"""Vault-backed note catalog used by StudyOS projections and adapters.

``StudyNoteCatalog`` is the module interface. The function aliases below it
remain internal implementation helpers so the legacy model-tool adapter can
be migrated without changing its observable contract.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
except Exception:  # pragma: no cover - PyYAML is a project dependency.
    yaml = None  # type: ignore[assignment]


WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
_CN_NORMALIZE_RE = re.compile(r"[的与和之]")


def _safe_relative_path(vault: Path, rel: str | None) -> Path:
    if not rel or not str(rel).strip():
        return vault
    raw = Path(str(rel).strip())
    candidate = (
        raw.expanduser().resolve()
        if raw.is_absolute()
        else (vault / raw).resolve()
    )
    try:
        candidate.relative_to(vault)
    except ValueError as exc:
        raise ValueError(f"Path escapes vault: {rel}") from exc
    return candidate


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _append_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        existing = _read_text(path)
        separator = "" if existing.endswith("\n") else "\n"
        _write_text(path, existing + separator + content)
    else:
        _write_text(path, content)


def _parse_frontmatter(raw: str) -> tuple[dict[str, Any], str, str | None]:
    lines = raw.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, raw, None
    end_idx = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            end_idx = idx
            break
    if end_idx is None:
        return {}, "\n".join(lines[1:]), "Missing closing --- in frontmatter"
    fm_text = "\n".join(lines[1:end_idx])
    body = "\n".join(lines[end_idx + 1 :])
    if not fm_text.strip():
        return {}, body, None
    if yaml is None:
        return {}, body, "PyYAML unavailable; frontmatter not parsed"
    try:
        parsed = yaml.safe_load(fm_text) or {}
    except Exception as exc:
        return {}, body, f"Failed to parse frontmatter: {exc}"
    if not isinstance(parsed, dict):
        return {}, body, "Frontmatter is not a mapping"
    return parsed, body, None


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    return [str(value)]


def _strip_wikilink(value: str) -> str:
    text = value.strip()
    if text.startswith("[[") and text.endswith("]]"):
        text = text[2:-2]
    if "|" in text:
        text = text.split("|", 1)[0]
    if "#" in text:
        text = text.split("#", 1)[0]
    return text.strip()


def _clean_body_for_links(body: str) -> str:
    return CODE_BLOCK_RE.sub("", body)


def _extract_wikilinks(body: str) -> list[str]:
    links = []
    seen = set()
    for match in WIKILINK_RE.finditer(_clean_body_for_links(body)):
        target = match.group(1).strip()
        if not target or "://" in target or target in seen:
            continue
        seen.add(target)
        links.append(target)
    return links


def _extract_headings(body: str) -> list[dict[str, Any]]:
    headings = []
    clean = CODE_BLOCK_RE.sub("", body)
    for match in HEADING_RE.finditer(clean):
        headings.append(
            {"level": len(match.group(1)), "text": match.group(2).strip()}
        )
    return headings


def _excerpt(body: str, limit: int = 260) -> str:
    clean_lines = []
    for line in CODE_BLOCK_RE.sub("", body).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        clean_lines.append(stripped)
    text = " ".join(clean_lines)
    return text[:limit] + ("..." if len(text) > limit else "")


def _layer_from(path: Path, vault: Path, frontmatter: dict[str, Any]) -> str:
    note_type = str(frontmatter.get("type") or "").strip()
    if note_type:
        return note_type
    relative = path.relative_to(vault).as_posix()
    if "/examples/" in f"/{relative}" or relative.startswith("examples/"):
        return "example"
    if "Box/题型/" in relative or "Box/题型\\" in relative:
        return "pattern"
    if "/Box/" in f"/{relative}" or relative.startswith("Box/"):
        return "concept"
    return "note"


def parse_note(
    path: Path,
    vault: Path,
    include_body: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    raw = _read_text(path)
    frontmatter, body, warning = _parse_frontmatter(raw)
    if warning:
        warnings.append(f"{path.relative_to(vault).as_posix()}: {warning}")
    headings = _extract_headings(body)
    title = str(
        frontmatter.get("title")
        or (headings[0]["text"] if headings else path.stem)
    )
    data: dict[str, Any] = {
        "path": path.relative_to(vault).as_posix(),
        "basename": path.name,
        "title": title,
        "layer": _layer_from(path, vault, frontmatter),
        "frontmatter": frontmatter,
        "tags": _as_list(frontmatter.get("tags")),
        "concepts": [
            _strip_wikilink(value)
            for value in _as_list(frontmatter.get("concepts"))
        ],
        "patterns": [
            _strip_wikilink(value)
            for value in _as_list(frontmatter.get("patterns"))
        ],
        "aliases": _as_list(frontmatter.get("aliases")),
        "headings": headings,
        "wikilinks": _extract_wikilinks(body),
        "excerpt": _excerpt(body),
        "size": path.stat().st_size,
        "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(
            timespec="seconds"
        ),
    }
    if include_body:
        data["body"] = body
    return data, warnings


def _iter_markdown_notes(
    vault: Path,
    *,
    folder: str | None = None,
    file_glob: str | None = None,
    include_study_os: bool = False,
) -> Iterable[Path]:
    root = _safe_relative_path(vault, folder)
    if not root.exists():
        return []
    pattern = (file_glob or "**/*.md").strip() or "**/*.md"
    paths = sorted(root.glob(pattern) if root.is_dir() else [root])
    result = []
    for path in paths:
        if not path.is_file() or path.suffix.lower() != ".md":
            continue
        relative = path.resolve().relative_to(vault).as_posix()
        if not include_study_os and (
            relative == ".StudyOS" or relative.startswith(".StudyOS/")
        ):
            continue
        result.append(path.resolve())
    return result


def _note_subject(note: dict[str, Any]) -> str | None:
    """Return the top-level course folder for a note, when it has one."""
    parts = Path(str(note.get("path") or "")).parts
    if len(parts) < 2 or parts[0].startswith("."):
        return None
    return parts[0]


def _normalize_cn(text: str) -> str:
    return _CN_NORMALIZE_RE.sub("", text.casefold())


def _matches_note(
    note: dict[str, Any],
    *,
    query: str | None,
    tag: str | None,
    layer: str | None,
    search_body: bool = False,
    normalize: bool = False,
) -> bool:
    if layer and note.get("layer") != layer:
        return False
    if tag:
        wanted = tag.strip().lstrip("#")
        tags = {item.lstrip("#") for item in note.get("tags", [])}
        if wanted not in tags:
            return False
    if query:
        query_lower = query.casefold()
        haystacks = [
            str(note.get("path", "")),
            str(note.get("title", "")),
            str(note.get("excerpt", "")),
            " ".join(note.get("aliases", [])),
            " ".join(note.get("concepts", [])),
            " ".join(note.get("patterns", [])),
            " ".join(note.get("wikilinks", [])),
        ]
        if search_body:
            haystacks.append(str(note.get("body", "")))
        lowered = [item.casefold() for item in haystacks]
        if any(query_lower in item for item in lowered):
            return True
        if normalize:
            normalized_query = _normalize_cn(query_lower)
            if normalized_query and any(
                normalized_query in _normalize_cn(item) for item in lowered
            ):
                return True
        return False
    return True


def _find_note(
    vault: Path,
    note_ref: str,
    include_study_os: bool = False,
) -> tuple[Path | None, list[Path]]:
    ref = (note_ref or "").strip()
    if not ref:
        return None, []
    direct = _safe_relative_path(vault, ref)
    if direct.is_file():
        return direct, []
    if direct.with_suffix(".md").is_file():
        return direct.with_suffix(".md"), []
    ref_clean = _strip_wikilink(ref)
    matches = []
    for path in _iter_markdown_notes(vault, include_study_os=include_study_os):
        data, _warnings = parse_note(path, vault, include_body=False)
        candidates = {
            data["path"],
            Path(data["path"]).with_suffix("").as_posix(),
            data["basename"],
            Path(data["basename"]).stem,
            data["title"],
            *data.get("aliases", []),
        }
        if ref_clean in candidates:
            matches.append(path)
    if len(matches) == 1:
        return matches[0], []
    return None, matches


@dataclass(frozen=True)
class StudyNoteCatalog:
    """Parse, discover, and resolve notes inside one Vault."""

    vault: Path

    def iter(
        self,
        *,
        folder: str | None = None,
        file_glob: str | None = None,
        include_study_os: bool = False,
    ) -> Iterable[Path]:
        return _iter_markdown_notes(
            self.vault,
            folder=folder,
            file_glob=file_glob,
            include_study_os=include_study_os,
        )

    def parse(
        self,
        path: Path,
        *,
        include_body: bool = False,
    ) -> tuple[dict[str, Any], list[str]]:
        return parse_note(path, self.vault, include_body=include_body)

    def find(
        self,
        note_ref: str,
        *,
        include_study_os: bool = False,
    ) -> tuple[Path | None, list[Path]]:
        return _find_note(
            self.vault,
            note_ref,
            include_study_os=include_study_os,
        )

    @staticmethod
    def subject(note: dict[str, Any]) -> str | None:
        return _note_subject(note)
