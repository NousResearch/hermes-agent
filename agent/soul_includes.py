"""SOUL.md include/import expansion.

This module is intentionally small and filesystem-focused. Prompt assembly stays
in ``agent.prompt_builder``; this module only expands explicit directive lines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Callable, Optional
import glob
import re

_INCLUDE_RE = re.compile(r"^@(include|import)\s+(.+?)\s*$")
_FENCE_RE = re.compile(r"^(?P<indent> {0,3})(?P<fence>`{3,}|~{3,})(?P<rest>.*)$")
_GLOB_CHARS = set("*?[")

MAX_DEPTH = 16
MAX_FILES = 128

ScanContent = Callable[[str, str], str]


@dataclass
class SoulIncludeState:
    """Mutable expansion state for cycle and limit checks."""

    active_stack: list[Path] = field(default_factory=list)
    files_loaded: int = 0


def parse_include_directive(line: str) -> Optional[str]:
    """Return include target for a column-zero directive line."""
    if line.startswith(r"\@include") or line.startswith(r"\@import"):
        return None
    match = _INCLUDE_RE.match(line)
    if not match:
        return None
    target = match.group(2).strip()
    return target or None


def has_soul_include_directive(content: str) -> bool:
    """Return True when *content* contains an active include directive."""
    fence: tuple[str, int] | None = None
    for line in content.splitlines():
        if fence is not None:
            if _closes_fence(line, fence):
                fence = None
            continue
        opened = _opens_fence(line)
        if opened is not None:
            fence = opened
            continue
        if parse_include_directive(line) is not None:
            return True
    return False


def _opens_fence(line: str) -> tuple[str, int] | None:
    match = _FENCE_RE.match(line)
    if not match:
        return None
    fence = match.group("fence")
    return fence[0], len(fence)


def _closes_fence(line: str, opened: tuple[str, int]) -> bool:
    match = _FENCE_RE.match(line)
    if not match:
        return False
    fence = match.group("fence")
    rest = match.group("rest").strip()
    return fence[0] == opened[0] and len(fence) >= opened[1] and not rest


def expand_soul_includes(
    content: str,
    *,
    root_path: Path,
    hermes_home: Path,
    scan_content: ScanContent,
    max_depth: int = MAX_DEPTH,
    max_files: int = MAX_FILES,
) -> str:
    """Return SOUL markdown with ``@include``/``@import`` expanded in place."""
    home = hermes_home.resolve()
    root = root_path.resolve(strict=True)
    state = SoulIncludeState(active_stack=[root], files_loaded=1)
    scanned = scan_content(content, _label_for(home, root))
    if _is_blocked(scanned):
        return scanned
    return _expand_lines(
        scanned,
        current_path=root,
        hermes_home=home,
        scan_content=scan_content,
        state=state,
        depth=0,
        max_depth=max_depth,
        max_files=max_files,
    ).strip()


def _expand_file(
    path: Path,
    *,
    hermes_home: Path,
    scan_content: ScanContent,
    state: SoulIncludeState,
    depth: int,
    max_depth: int,
    max_files: int,
    source_label: str,
    line_no: int,
) -> str:
    resolved = path.resolve(strict=True)
    label = _label_for(hermes_home, resolved)
    if resolved in state.active_stack:
        chain = " -> ".join([_label_for(hermes_home, p) for p in [*state.active_stack, resolved]])
        return _placeholder(source_label, line_no, f"cycle detected: {chain}")
    if depth > max_depth:
        return _placeholder(source_label, line_no, f"max include depth exceeded ({max_depth})")
    if state.files_loaded >= max_files:
        return _placeholder(source_label, line_no, f"max include files exceeded ({max_files})")

    state.files_loaded += 1
    try:
        raw = resolved.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as exc:
        return _placeholder(
            source_label,
            line_no,
            f"could not read include {label}: {exc.__class__.__name__}",
        )
    if not raw:
        return ""
    scanned = scan_content(raw, label)
    if _is_blocked(scanned):
        return scanned

    state.active_stack.append(resolved)
    try:
        return _expand_lines(
            scanned,
            current_path=resolved,
            hermes_home=hermes_home,
            scan_content=scan_content,
            state=state,
            depth=depth,
            max_depth=max_depth,
            max_files=max_files,
        ).strip()
    finally:
        state.active_stack.pop()


def _expand_lines(
    content: str,
    *,
    current_path: Path,
    hermes_home: Path,
    scan_content: ScanContent,
    state: SoulIncludeState,
    depth: int,
    max_depth: int,
    max_files: int,
) -> str:
    source_label = _label_for(hermes_home, current_path)
    out: list[str] = []
    fence: tuple[str, int] | None = None
    for line_no, line in enumerate(content.splitlines(), start=1):
        if fence is not None:
            out.append(line)
            if _closes_fence(line, fence):
                fence = None
            continue
        opened = _opens_fence(line)
        if opened is not None:
            fence = opened
            out.append(line)
            continue
        target = parse_include_directive(line)
        if target is None:
            out.append(line)
            continue
        targets, error = _resolve_targets(target, hermes_home=hermes_home)
        if error:
            out.append(_placeholder(source_label, line_no, error))
            continue
        expanded_parts = [
            _expand_file(
                path,
                hermes_home=hermes_home,
                scan_content=scan_content,
                state=state,
                depth=depth + 1,
                max_depth=max_depth,
                max_files=max_files,
                source_label=source_label,
                line_no=line_no,
            )
            for path in targets
        ]
        out.append("\n".join(part for part in expanded_parts if part))
    return "\n".join(out)


def _resolve_targets(raw_target: str, *, hermes_home: Path) -> tuple[list[Path], Optional[str]]:
    target = raw_target.strip().strip('"\'')
    if not target:
        return [], "empty include target"
    if target.startswith("~"):
        return [], f"tilde paths are not allowed: {raw_target}"
    target_path = Path(target)
    if target_path.is_absolute():
        return [], f"absolute paths are not allowed: {raw_target}"
    if ".." in PurePosixPath(target.replace("\\", "/")).parts:
        return [], f"parent traversal is not allowed: {raw_target}"
    if "**" in PurePosixPath(target.replace("\\", "/")).parts:
        return [], f"recursive glob is not allowed: {raw_target}"

    has_glob = any(ch in target for ch in _GLOB_CHARS)
    if not has_glob and Path(target).suffix.lower() != ".md":
        return [], f"not a markdown include: {raw_target}"

    if has_glob:
        candidates = [Path(p) for p in glob.glob(str(hermes_home / target))]
        candidates = [p for p in candidates if p.suffix.lower() == ".md"]
    else:
        candidates = [hermes_home / target]

    safe: list[Path] = []
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError:
            continue
        except OSError as exc:
            return [], f"could not resolve include {raw_target}: {exc}"
        if not resolved.is_file():
            continue
        if resolved.suffix.lower() != ".md":
            continue
        try:
            resolved.relative_to(hermes_home)
        except ValueError:
            return [], f"symlink escapes HERMES_HOME: {raw_target}"
        safe.append(resolved)

    if not safe:
        return [], f"no include matches: {raw_target}"

    safe.sort(key=lambda p: p.relative_to(hermes_home).as_posix())
    return safe, None


def _placeholder(source_label: str, line_no: int, message: str) -> str:
    return f"[INCLUDE ERROR: {source_label}:{line_no}: {message}]"


def _is_blocked(content: str) -> bool:
    return content.startswith("[BLOCKED:")


def _label_for(hermes_home: Path, path: Path) -> str:
    try:
        return path.relative_to(hermes_home).as_posix()
    except ValueError:
        return path.name
