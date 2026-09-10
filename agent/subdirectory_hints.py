"""Progressive subdirectory hint discovery: as the agent navigates into
subdirectories via tool calls, load project context files (AGENTS.md, CLAUDE.md,
.cursorrules) from them and append to the tool result — context arrives without
touching the system prompt (prompt caching preserved). Complements the startup
CWD-only loading in ``prompt_builder.py``."""

import hashlib
import logging
import os
import shlex
from pathlib import Path
from typing import Dict, Any, Optional, Set

from agent.prompt_builder import _read_text_with_timeout, _scan_context_content, _truncate_content
from agent.search_policy import SEARCH_PRUNE_DIR_NAMES

logger = logging.getLogger(__name__)

# Same filenames as prompt_builder.py, in priority order (first match wins per dir).
_HINT_FILENAMES = ["AGENTS.override.md", "AGENTS.md", "agents.md", "CLAUDE.md", "claude.md", ".cursorrules"]
# Per-file ceiling for on-demand subdirectory hints. 32 KiB matches Codex's `project_doc_max_bytes` default
# (Claude Code and Cursor apply none); it is a guard against a stray huge CLAUDE.md in a vendored tree, not a
# target — keep area AGENTS.md files well under it (~8k) because this text lands in a tool result on the first
# touch of that directory. Over the ceiling: head+tail kept, marker with the path so the agent can read_file it,
# and a WARNING in the log (the old 8k silent tail-chop cut apps/desktop/AGENTS.md for months unnoticed).
_MAX_HINT_CHARS = 32_000
_PATH_ARG_KEYS = {"path", "file_path", "workdir"}
_COMMAND_TOOLS = {"terminal"}
_MAX_ANCESTOR_WALK = 5  # ancestor levels walked per path — bounds deep-path scans

# Shared with broad recursive search probes so context discovery and search never drift into
# different dependency/cache/build trees (those hold *copies* of context files, never authoritative ones).
_EXCLUDED_DIR_NAMES = SEARCH_PRUNE_DIR_NAMES


def _digest(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _first_hint_file(directory: Path):
    """``(path, stripped content)`` of the first readable non-empty hint file
    in *directory* (priority order), or None. Unreadable files are skipped."""
    for filename in _HINT_FILENAMES:
        candidate = directory / filename
        try:
            if not candidate.is_file():
                continue
            content = candidate.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            continue
        return candidate, content
    return None


class SubdirectoryHintTracker:
    """Track which directories the agent visits and load hints on first access.

    Usage: after each tool call, ``hints = tracker.check_tool_call(name, args)``
    and append the returned text to the tool result.
    """

    def __init__(self, working_dir: Optional[str] = None):
        self.working_dir = Path(working_dir or os.getcwd()).resolve()
        # The working dir is pre-marked loaded (startup context handles it).
        self._loaded_dirs: Set[Path] = {self.working_dir}
        # Content digests already injected: the same file reached through
        # symlinks/hardlinks/copies is never re-sent. Seeded with the CWD hint
        # file prompt_builder already loaded.
        self._loaded_digests: Set[str] = set()
        found = _first_hint_file(self.working_dir)
        if found and found[1]:
            self._loaded_digests.add(_digest(found[1]))

    def check_tool_call(self, tool_name: str, tool_args: Dict[str, Any]) -> Optional[str]:
        """Return formatted hint text for newly visited directories, or None."""
        all_hints = [h for d in self._extract_directories(tool_name, tool_args) if (h := self._load_hints_for_directory(d))]
        return "\n\n" + "\n\n".join(all_hints) if all_hints else None

    def _extract_directories(self, tool_name: str, args: Dict[str, Any]) -> list:
        """Extract directory paths from tool call arguments."""
        candidates: Set[Path] = set()
        for key in _PATH_ARG_KEYS:
            val = args.get(key)
            if isinstance(val, str) and val.strip():
                self._add_path_candidate(val, candidates)
        cmd = args.get("command", "") if tool_name in _COMMAND_TOOLS else None
        if isinstance(cmd, str):
            self._extract_paths_from_command(cmd, candidates)
        return list(candidates)

    def _add_path_candidate(self, raw_path: str, candidates: Set[Path]):
        """Add a raw path's directory and its ancestors (up to ``_MAX_ANCESTOR_WALK``
        levels, stopping at the first already-loaded dir) so reading
        ``project/src/main.py`` still discovers ``project/AGENTS.md``."""
        try:
            p = Path(raw_path).expanduser()
            if not p.is_absolute():
                p = self.working_dir / p
            p = p.resolve()
            if p.suffix or (p.exists() and p.is_file()):
                p = p.parent
            for _ in range(_MAX_ANCESTOR_WALK):
                if p in self._loaded_dirs:
                    break
                if self._is_valid_subdir(p):
                    candidates.add(p)
                if p.parent == p:
                    break  # filesystem root
                p = p.parent
        except (OSError, ValueError, RuntimeError):
            pass

    def _extract_paths_from_command(self, cmd: str, candidates: Set[Path]):
        """Extract path-like tokens (contain / or .; not flags or URLs) from a shell command."""
        try:
            tokens = shlex.split(cmd)
        except ValueError:
            tokens = cmd.split()
        for token in tokens:
            if token.startswith(("-", "http://", "https://", "git@")) or ("/" not in token and "." not in token):
                continue
            self._add_path_candidate(token, candidates)

    def _within_working_dir(self, path: Path) -> bool:
        """Reject paths outside the working-dir tree: loading ~/.codex/AGENTS.md
        or ~/.claude/CLAUDE.md would mix another agent's instructions into this
        session. Falls back to an ancestor check when ``is_relative_to`` fails."""
        try:
            return path.is_relative_to(self.working_dir)
        except (OSError, ValueError):
            try:
                path.relative_to(self.working_dir)
                return True
            except ValueError:
                return False

    def _is_valid_subdir(self, path: Path) -> bool:
        """Directory inside the working-dir tree, not yet loaded, not an excluded copy dir."""
        try:
            if not path.is_dir():
                return False
        except OSError:
            return False
        return path not in self._loaded_dirs and self._within_working_dir(path) and not self._is_excluded(path)

    def _is_excluded(self, path: Path) -> bool:
        """True when a segment *below* the working dir is an excluded copy dir
        (a user deliberately working inside ``vendor/`` keeps that segment legitimate)."""
        try:
            rel_parts = path.relative_to(self.working_dir).parts
        except ValueError:
            return True  # outside the tree — already rejected upstream
        return any(part in _EXCLUDED_DIR_NAMES for part in rel_parts)

    def _load_hints_for_directory(self, directory: Path) -> Optional[str]:
        """Load the first hint file in *directory*; formatted text or None."""
        self._loaded_dirs.add(directory)
        if not self._within_working_dir(directory):
            logger.debug("Skipping hint files in %s — outside working_dir %s", directory, self.working_dir)
            return None
        for filename in _HINT_FILENAMES:
            hint_path = directory / filename
            try:
                if not hint_path.is_file():
                    continue
            except OSError:
                continue
            try:
                content = (_read_text_with_timeout(hint_path) or "").strip()
                if not content:
                    continue
                digest = _digest(content)
                if digest in self._loaded_digests:
                    logger.debug("Skipping duplicate hint content at %s (digest %s)", hint_path, digest[:12])
                    return None
                self._loaded_digests.add(digest)
                # Same security scan as startup context loading.
                content = _scan_context_content(content, filename)
                rel_path = self._display_path(hint_path)
                content = _truncate_content(content, filename, max_chars=_MAX_HINT_CHARS, read_path=rel_path)
                logger.debug("Loaded subdirectory hints from %s: %s", directory, [rel_path])
                return f"[Subdirectory context discovered: {rel_path}]\n{content}"  # first match wins per directory
            except Exception as exc:
                logger.debug("Could not read %s: %s", hint_path, exc)
        return None

    def _display_path(self, hint_path: Path) -> str:
        """Working-dir-relative, else ``~/``-relative (POSIX rendering so Windows
        never shows ``~/AppData\\Local\\...`` chimeras), else absolute."""
        try:
            return str(hint_path.relative_to(self.working_dir))
        except (ValueError, RuntimeError):
            pass
        try:
            return "~/" + hint_path.relative_to(Path.home()).as_posix()
        except (ValueError, RuntimeError):
            return str(hint_path)
