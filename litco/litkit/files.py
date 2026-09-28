"""Where LitKit tools read and write on the matter host.

Every file a LitKit tool writes lands under the current turn's working directory
(``shared/`` for a channel thread, ``users/<id>/`` for a private thread). Names that come
from LitKit (Bates numbers, filenames) are reduced to one safe path segment, and any path
the agent supplies is resolved and refused if it escapes the working directory.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

from litco.homes import matter_home, safe_segment
from litco.litkit.context import current_turn

SPILL_CHARS = 12_000
PREVIEW_CHARS = 1_500
PERSISTED_OUTPUT_TAG = "<persisted-output>"
PERSISTED_OUTPUT_CLOSING_TAG = "</persisted-output>"
TEXT_SEPARATOR = "=" * 60


class PathOutsideWorkDir(ValueError):
    pass


def work_dir() -> Path:
    """The current turn's working directory; outside a turn, the session cwd or ``<home>/shared``."""
    turn = current_turn()
    base: Optional[Path] = turn.cwd if turn is not None and turn.cwd else None
    if base is None:
        try:
            from agent.runtime_cwd import scoped_session_cwd
            declared = scoped_session_cwd()
        except Exception:
            declared = ""
        base = Path(declared) if declared else matter_home() / "shared"
    base = Path(base).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    return base


def inside(base: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def output_path(subdir: str, filename: str, *, base: Optional[Path] = None) -> Path:
    """``<work dir>/<subdir>/<safe filename>``; ``subdir`` may be nested but never escapes."""
    root = base or work_dir()
    parts = [safe_segment(p) for p in str(subdir or "").replace("\\", "/").split("/") if p and p not in (".", "..")]
    folder = root.joinpath(*parts) if parts else root
    target = folder / safe_segment(filename)
    if not inside(root, target):
        raise PathOutsideWorkDir(f"refusing to write outside the working directory: {target}")
    folder.mkdir(parents=True, exist_ok=True)
    return target


def output_dir(subdir: str, *, base: Optional[Path] = None) -> Path:
    root = base or work_dir()
    parts = [safe_segment(p) for p in str(subdir or "").replace("\\", "/").split("/") if p and p not in (".", "..")]
    folder = root.joinpath(*parts) if parts else root
    if not inside(root, folder):
        raise PathOutsideWorkDir(f"refusing to write outside the working directory: {folder}")
    folder.mkdir(parents=True, exist_ok=True)
    return folder


class InputFileMissing(ValueError):
    """The agent named a file that is not on disk (usually: it has not been written yet)."""


def input_path(raw: str, *, tool: str = "this tool") -> Path:
    """A file the agent names for upload: relative to the working directory, or absolute inside
    the working directory or the matter home. Must exist."""
    if not raw or not str(raw).strip():
        raise ValueError("a file path is required")
    base = work_dir()
    path = Path(os.path.expanduser(str(raw).strip()))
    path = (path if path.is_absolute() else base / path).resolve()
    if not (inside(base, path) or inside(matter_home(), path)):
        raise PathOutsideWorkDir(f"{raw} is outside the matter's working directories")
    if path.is_dir():
        raise ValueError(f"{raw} is a folder, not a file; name the file itself")
    if not path.is_file():
        raise InputFileMissing(
            f"the file {raw} does not exist (looked for {path}). Write the file first, confirm it exists "
            f"(for example with ls or read_file), then call {tool} again with that path.")
    return path


def relative(path: Path) -> str:
    """Path shown to the agent: relative to the working directory when possible."""
    try:
        return str(Path(path).resolve().relative_to(work_dir()))
    except ValueError:
        return str(path)


def dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def generate_preview(content: str, max_chars: int = PREVIEW_CHARS) -> tuple:
    if len(content) <= max_chars:
        return content, False
    last_nl = content.rfind("\n", 0, max_chars)
    return content[: last_nl + 1 if last_nl > max_chars // 2 else max_chars], True


def spill(content: str, tool_name: str, *, threshold: int = SPILL_CHARS) -> str:
    """Hermes's spill convention: an oversized result is saved to a file and replaced by a
    ``<persisted-output>`` block with a preview and the path (``Full output saved to: ...``).
    The file lives under the working directory in ``litkit/results/``."""
    if len(content) <= threshold:
        return content
    stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
    target = output_path("litkit/results", f"{tool_name}-{stamp}.json")
    target.write_text(content, encoding="utf-8")
    preview, has_more = generate_preview(content)
    size_kb = len(content) / 1024
    return (f"{PERSISTED_OUTPUT_TAG}\n"
            f"This tool result was too large ({len(content):,} characters, {size_kb:.1f} KB).\n"
            f"Full output saved to: {target}\n"
            "Use the read_file tool with offset and limit to access specific sections of this output, or "
            "process it with execute_code. Do not re-request the same data from LitKit; the full result is "
            "already on disk.\n\n"
            f"Preview (first {len(preview)} chars):\n{preview}{chr(10) + '...' if has_more else ''}\n"
            f"{PERSISTED_OUTPUT_CLOSING_TAG}")
