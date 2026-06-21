"""Safe Obsidian vault folder-browser helpers for gateway adapters."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote


class ObsidianBrowserError(RuntimeError):
    """Raised when the Obsidian browser cannot resolve or list a folder."""


@dataclass(frozen=True)
class ObsidianBrowserEntry:
    name: str
    relative_path: str
    is_dir: bool
    url: str | None = None


@dataclass(frozen=True)
class ObsidianBrowserPayload:
    vault: Path
    relative_path: str
    current_path: Path
    parent_path: str | None
    dirs: list[ObsidianBrowserEntry]
    files: list[ObsidianBrowserEntry]
    total_dirs: int
    total_files: int
    truncated: bool = False


def obsidian_vault_root() -> Path:
    """Return the configured Obsidian vault root.

    The gateway/plugin convention is ``OBSIDIAN_VAULT_PATH``.  We intentionally
    do not hardcode a user path here so this helper remains profile-safe and
    usable outside the current machine.
    """

    raw = os.environ.get("OBSIDIAN_VAULT_PATH", "").strip()
    if not raw:
        raise ObsidianBrowserError(
            "Obsidian vault path is not configured. Set OBSIDIAN_VAULT_PATH."
        )
    vault = Path(raw).expanduser().resolve()
    if not vault.exists() or not vault.is_dir():
        raise ObsidianBrowserError(f"Obsidian vault not found: {vault}")
    return vault


def normalize_relative_path(value: str | os.PathLike[str] | None) -> str:
    """Normalize a user/callback path into a vault-relative POSIX path."""

    raw = str(value or "").strip().replace("\\", "/")
    while raw.startswith("/"):
        raw = raw[1:]
    parts = [part for part in raw.split("/") if part not in {"", "."}]
    if any(part == ".." for part in parts):
        raise ObsidianBrowserError("Path traversal is not allowed.")
    return "/".join(parts)


def resolve_vault_path(relative_path: str | os.PathLike[str] | None = "") -> tuple[Path, str, Path]:
    """Resolve a vault-relative folder path and prove it stays inside the vault."""

    vault = obsidian_vault_root()
    rel = normalize_relative_path(relative_path)
    candidate = (vault / rel).resolve()
    try:
        candidate.relative_to(vault)
    except ValueError as exc:
        raise ObsidianBrowserError("Path escapes the Obsidian vault.") from exc
    if not candidate.exists() or not candidate.is_dir():
        display = f"/{rel}" if rel else "/"
        raise ObsidianBrowserError(f"Obsidian folder not found: {display}")
    return vault, rel, candidate


def obsidian_open_url(vault: Path, relative_path: str) -> str:
    """Build an obsidian:// open URL for a vault-relative path."""

    vault_name = quote(vault.name, safe="")
    note_path = quote(normalize_relative_path(relative_path), safe="/")
    if note_path:
        return f"obsidian://open?vault={vault_name}&path={note_path}"
    return f"obsidian://open?vault={vault_name}"


def _sort_key(path: Path) -> tuple[str, str]:
    return (path.name.casefold(), path.name)


def _safe_iterdir(path: Path) -> list[Path]:
    try:
        return list(path.iterdir())
    except OSError as exc:
        raise ObsidianBrowserError(f"Failed to list Obsidian folder: {exc}") from exc


def _entry_relative_path(vault: Path, entry: Path) -> str:
    return entry.relative_to(vault).as_posix()


def build_obsidian_browser_payload(
    relative_path: str | os.PathLike[str] | None = "",
    *,
    max_dirs: int = 18,
    max_files: int = 6,
) -> ObsidianBrowserPayload:
    """List folders and markdown notes for a Telegram inline-button browser."""

    vault, rel, current = resolve_vault_path(relative_path)
    children = [p for p in _safe_iterdir(current) if not p.name.startswith(".")]
    dir_paths = sorted((p for p in children if p.is_dir()), key=_sort_key)
    file_paths = sorted((p for p in children if p.is_file() and p.suffix.lower() == ".md"), key=_sort_key)

    dir_entries = [
        ObsidianBrowserEntry(
            name=path.name,
            relative_path=_entry_relative_path(vault, path),
            is_dir=True,
        )
        for path in dir_paths[: max(0, max_dirs)]
    ]
    file_entries = [
        ObsidianBrowserEntry(
            name=path.name,
            relative_path=_entry_relative_path(vault, path),
            is_dir=False,
            url=obsidian_open_url(vault, _entry_relative_path(vault, path)),
        )
        for path in file_paths[: max(0, max_files)]
    ]

    parent_path: str | None = None
    if rel:
        parent = Path(rel).parent.as_posix()
        parent_path = "" if parent == "." else parent

    return ObsidianBrowserPayload(
        vault=vault,
        relative_path=rel,
        current_path=current,
        parent_path=parent_path,
        dirs=dir_entries,
        files=file_entries,
        total_dirs=len(dir_paths),
        total_files=len(file_paths),
        truncated=(len(dir_entries) < len(dir_paths) or len(file_entries) < len(file_paths)),
    )


def render_obsidian_browser_text(payload: ObsidianBrowserPayload) -> str:
    """Render a short Markdown message for the folder browser."""

    display_path = f"/{payload.relative_path}" if payload.relative_path else "/"
    title = "🗂 Obsidian Root" if not payload.relative_path else f"🗂 Obsidian {display_path}"
    lines = [
        title,
        f"- Path: `{display_path}`",
        f"- Folders: {payload.total_dirs} | Notes: {payload.total_files}",
    ]
    if payload.truncated:
        lines.append("- 일부 항목만 표시 중입니다. 하위 폴더로 들어가서 좁혀 보세요.")
    if not payload.dirs and not payload.files:
        lines.append("- 표시할 하위 폴더나 Markdown 노트가 없습니다.")
    return "\n".join(lines)
