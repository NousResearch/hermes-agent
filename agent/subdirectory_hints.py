"""Progressive subdirectory hint discovery.

As the agent navigates into subdirectories via tool calls (read_file, terminal,
search_files, etc.), this module discovers and loads project context files
(AGENTS.md, CLAUDE.md, .cursorrules) from those directories.  Discovered hints
are appended to the tool result so the model gets relevant context at the moment
it starts working in a new area of the codebase.

This complements the startup context loading in ``prompt_builder.py`` which only
loads from the CWD.  Subdirectory hints are discovered lazily and injected into
the conversation without modifying the system prompt (preserving prompt caching).

Inspired by Block/goose's SubdirectoryHintTracker.
"""

import hashlib
import logging
import os
import shlex
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Set

from agent.prompt_builder import _scan_context_content

logger = logging.getLogger(__name__)

# Context files to look for in subdirectories, in priority order.
# Same filenames as prompt_builder.py but we load ALL found (not first-wins)
# since different subdirectories may use different conventions.
_HINT_FILENAMES = [
    "AGENTS.override.md",
    "AGENTS.md", "agents.md",
    "CLAUDE.md", "claude.md",
    ".cursorrules",
]

# Maximum chars per hint file to prevent context bloat
_MAX_HINT_CHARS = 8_000

# Hard cap on how long a single hint-file open/read may block the tool path.
# Evicted FileProvider placeholders (iCloud Drive, etc.) hang forever on
# open/read; a short timeout treats them as missing so the gateway stays live.
_HINT_READ_TIMEOUT_SECONDS = 2.0

# Tool argument keys that typically contain file paths
_PATH_ARG_KEYS = {"path", "file_path", "workdir"}

# Tools that take shell commands where we should extract paths
_COMMAND_TOOLS = {"terminal"}

# How many parent directories to walk up when looking for hints.
# Prevents scanning all the way to / for deeply nested paths.
_MAX_ANCESTOR_WALK = 5

# Directory names that never contain authoritative project context.
# Backups, vendored deps, VCS internals, and caches routinely hold *copies* of
# AGENTS.md; loading those duplicates real context and inflates the prompt.
_EXCLUDED_DIR_NAMES = frozenset({
    "node_modules", "venv", ".venv", "__pycache__",
    ".git", ".hg", ".svn",
    ".Trash", ".cache", ".tox", ".mypy_cache", ".pytest_cache",
    "site-packages", "dist-packages",
    "backups", "backup", ".backups",
    "vendor", "third_party",
})

# Directory names that sit directly under a `Library` path component and mark
# a FileProvider-backed subtree. Mirrors cron/lifecycle_guard.py so iCloud
# Drive (`Mobile Documents`) and third-party providers under `CloudStorage`
# (Dropbox / OneDrive / Google Drive / Box) are refused before open.
_CLOUD_PLACEHOLDER_MARKERS = frozenset({"Mobile Documents", "CloudStorage"})

# Broader refuse-list substrings from the 2026-08-18 gateway wedge
# (task 20260818_092117_51a51a54): bare Path.read_text hung 53+ min on an
# iCloud-backed AGENTS.md. Match anywhere in the path string.
_CLOUD_PATH_REFUSE_SUBSTRINGS = (
    "Mobile Documents",
    "iCloud",
    ".icloud",
    "com.apple.fileprovider",
    "CloudStorage",
)


def _is_ancestor_or_same(a: Path, b: Path) -> bool:
    """Check if *a* is the same as or an ancestor of *b* (parent directory check)."""
    try:
        b.relative_to(a)
        return True
    except ValueError:
        return False


def _is_cloud_backed_path(path: Path) -> bool:
    """True when *path* is under a macOS FileProvider / iCloud subtree.

    Opening an evicted placeholder can block indefinitely. Refuse from path
    metadata alone — never open, stat, or resolve into the cloud tree first.
    """
    parts = path.parts
    if any(
        parts[index - 1] == "Library" and part in _CLOUD_PLACEHOLDER_MARKERS
        for index, part in enumerate(parts)
        if index
    ):
        return True
    path_str = str(path)
    return any(marker in path_str for marker in _CLOUD_PATH_REFUSE_SUBSTRINGS)


def _read_hint_text(path: Path) -> Optional[str]:
    """Bounded UTF-8 read of a hint file. Returns None on refuse/timeout/error.

    Never uses bare ``Path.read_text``: iCloud / FileProvider placeholders
    hang forever on open (gateway wedge 2026-08-18). Preflight refuse + a
    short timeout treat the file as missing so the tool call path stays live.
    """
    if _is_cloud_backed_path(path):
        logger.debug("Skipping iCloud/FileProvider-backed hint file: %s", path)
        return None

    box: Dict[str, Any] = {"content": None, "error": None}

    def _do_read() -> None:
        try:
            # Explicit open/read (not Path.read_text) so the blocking call is
            # confined to this daemon worker and can be abandoned on timeout.
            with open(path, "r", encoding="utf-8") as fh:
                box["content"] = fh.read().strip()
        except BaseException as exc:  # noqa: BLE001 — marshalled back to caller
            box["error"] = exc

    thread = threading.Thread(
        target=_do_read,
        name="subdirectory-hint-read",
        daemon=True,
    )
    thread.start()
    thread.join(timeout=_HINT_READ_TIMEOUT_SECONDS)
    if thread.is_alive():
        logger.warning(
            "Timed out reading hint file after %.1fs (treating as missing): %s",
            _HINT_READ_TIMEOUT_SECONDS,
            path,
        )
        return None

    err = box["error"]
    if err is not None:
        if isinstance(err, (OSError, UnicodeDecodeError)):
            logger.debug("Could not read %s: %s", path, err)
            return None
        # Unexpected errors should still surface in tests / logs as debug and
        # behave like a missing hint — never wedge the tool path.
        logger.debug("Could not read %s: %s", path, err)
        return None
    return box["content"]


class SubdirectoryHintTracker:
    """Track which directories the agent visits and load hints on first access.

    Usage::

        tracker = SubdirectoryHintTracker(working_dir="/path/to/project")

        # After each tool call:
        hints = tracker.check_tool_call("read_file", {"path": "backend/src/main.py"})
        if hints:
            tool_result += hints  # append to the tool result string
    """

    def __init__(self, working_dir: Optional[str] = None):
        self.working_dir = Path(working_dir or os.getcwd()).resolve()
        self._loaded_dirs: Set[Path] = set()
        # Content digests already injected — prevents re-sending the same file
        # reachable through symlinks, hardlinks, or duplicated copies.
        self._loaded_digests: Set[str] = set()
        # Pre-mark the working dir as loaded (startup context handles it)
        self._loaded_dirs.add(self.working_dir)
        self._seed_working_dir_digest()

    def _seed_working_dir_digest(self) -> None:
        """Record the CWD context file's digest so it is never re-injected.

        ``prompt_builder`` already loads the working directory's context file at
        startup.  Seeding its digest here means the same content reached through
        a different path (a symlink farm, a shared workspace) is recognised as a
        duplicate instead of being sent a second time.
        """
        for filename in _HINT_FILENAMES:
            candidate = self.working_dir / filename
            # Refuse cloud paths before is_file()/open — both can hang on
            # FileProvider placeholders (same class of wedge as the load path).
            if _is_cloud_backed_path(candidate):
                logger.debug(
                    "Skipping iCloud/FileProvider-backed CWD hint seed: %s",
                    candidate,
                )
                continue
            try:
                if not candidate.is_file():
                    continue
            except OSError:
                continue
            content = _read_hint_text(candidate)
            if content:
                self._loaded_digests.add(
                    hashlib.sha256(content.encode("utf-8")).hexdigest()
                )
            break  # first match wins, mirroring startup loading

    def check_tool_call(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
    ) -> Optional[str]:
        """Check tool call arguments for new directories and load any hint files.

        Returns formatted hint text to append to the tool result, or None.
        """
        dirs = self._extract_directories(tool_name, tool_args)
        if not dirs:
            return None

        all_hints = []
        for d in dirs:
            hints = self._load_hints_for_directory(d)
            if hints:
                all_hints.append(hints)

        if not all_hints:
            return None

        return "\n\n" + "\n\n".join(all_hints)

    def _extract_directories(
        self, tool_name: str, args: Dict[str, Any]
    ) -> list:
        """Extract directory paths from tool call arguments."""
        candidates: Set[Path] = set()

        # Direct path arguments
        for key in _PATH_ARG_KEYS:
            val = args.get(key)
            if isinstance(val, str) and val.strip():
                self._add_path_candidate(val, candidates)

        # Shell commands — extract path-like tokens
        if tool_name in _COMMAND_TOOLS:
            cmd = args.get("command", "")
            if isinstance(cmd, str):
                self._extract_paths_from_command(cmd, candidates)

        return list(candidates)

    def _add_path_candidate(self, raw_path: str, candidates: Set[Path]):
        """Resolve a raw path and add its directory + ancestors to candidates.

        Walks up from the resolved directory toward the filesystem root,
        stopping at the first directory already in ``_loaded_dirs`` (or after
        ``_MAX_ANCESTOR_WALK`` levels).  This ensures that reading
        ``project/src/main.py`` discovers ``project/AGENTS.md`` even when
        ``project/src/`` has no hint files of its own.
        """
        try:
            p = Path(raw_path).expanduser()
            if not p.is_absolute():
                p = self.working_dir / p
            p = p.resolve()
            # Use parent if it's a file path (has extension or doesn't exist as dir)
            if p.suffix or (p.exists() and p.is_file()):
                p = p.parent
            # Walk up ancestors — stop at already-loaded or root
            for _ in range(_MAX_ANCESTOR_WALK):
                if p in self._loaded_dirs:
                    break
                if self._is_valid_subdir(p):
                    candidates.add(p)
                parent = p.parent
                if parent == p:
                    break  # filesystem root
                p = parent
        except (OSError, ValueError, RuntimeError):
            pass

    def _extract_paths_from_command(self, cmd: str, candidates: Set[Path]):
        """Extract path-like tokens from a shell command string."""
        try:
            tokens = shlex.split(cmd)
        except ValueError:
            tokens = cmd.split()

        for token in tokens:
            # Skip flags
            if token.startswith("-"):
                continue
            # Must look like a path (contains / or .)
            if "/" not in token and "." not in token:
                continue
            # Skip URLs
            if token.startswith(("http://", "https://", "git@")):
                continue
            self._add_path_candidate(token, candidates)

    def _is_valid_subdir(self, path: Path) -> bool:
        """Check if path is a valid directory to scan for hints.

        Only allow subdirectories within the working directory tree.
        This prevents loading AGENTS.md from outside the active workspace
        (e.g. ~/.codex/AGENTS.md, ~/.claude/CLAUDE.md), which causes
        cross-agent context contamination and instruction mixup.
        """
        try:
            if not path.is_dir():
                return False
        except OSError:
            return False
        if path in self._loaded_dirs:
            return False
        # Reject paths outside the working directory tree.
        # path.resolve() may differ from working_dir.resolve() due to symlinks,
        # but path.is_relative_to(working_dir) handles both absolute and
        # symlinked paths correctly on Python 3.9+.
        try:
            if not path.is_relative_to(self.working_dir):
                return False
        except (OSError, ValueError):
            # Older Python or path resolution error — fall back to parent
            # check as a best-effort safeguard.
            if not _is_ancestor_or_same(self.working_dir, path):
                return False
        if self._is_excluded(path):
            return False
        return True

    def _is_excluded(self, path: Path) -> bool:
        """True when the path sits inside a directory that holds copies, not context.

        Directories the user is deliberately working inside are never excluded —
        if ``working_dir`` is itself under ``vendor/``, that segment is legitimate
        and only segments *below* the working dir are screened.
        """
        try:
            rel_parts = path.relative_to(self.working_dir).parts
        except ValueError:
            # Paths outside the working dir are already rejected by
            # _is_valid_subdir before this runs; treat as excluded defensively.
            return True
        return any(part in _EXCLUDED_DIR_NAMES for part in rel_parts)

    def _load_hints_for_directory(self, directory: Path) -> Optional[str]:
        """Load hint files from a directory. Returns formatted text or None.

        Only loads hints from directories within the working directory tree.
        """
        self._loaded_dirs.add(directory)

        # Reject paths outside the working directory tree.
        try:
            if not directory.is_relative_to(self.working_dir):
                logger.debug(
                    "Skipping hint files in %s — outside working_dir %s",
                    directory, self.working_dir,
                )
                return None
        except (OSError, ValueError):
            if not _is_ancestor_or_same(self.working_dir, directory):
                logger.debug(
                    "Skipping hint files in %s — outside working_dir %s",
                    directory, self.working_dir,
                )
                return None

        found_hints = []
        for filename in _HINT_FILENAMES:
            hint_path = directory / filename
            # Preflight refuse before is_file()/open — FileProvider paths hang.
            if _is_cloud_backed_path(hint_path):
                logger.debug(
                    "Skipping iCloud/FileProvider-backed hint file: %s",
                    hint_path,
                )
                continue
            try:
                if not hint_path.is_file():
                    continue
            except OSError:
                continue
            try:
                content = _read_hint_text(hint_path)
                if not content:
                    continue
                # Skip content we've already injected. The same AGENTS.md is
                # routinely reachable through several paths (symlinked shared
                # workspaces, hardlinks, copied backups); re-sending it burns
                # context for zero new information.
                digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
                if digest in self._loaded_digests:
                    logger.debug(
                        "Skipping duplicate hint content at %s (digest %s)",
                        hint_path,
                        digest[:12],
                    )
                    break
                self._loaded_digests.add(digest)
                # Same security scan as startup context loading
                content = _scan_context_content(content, filename)
                if len(content) > _MAX_HINT_CHARS:
                    content = (
                        content[:_MAX_HINT_CHARS]
                        + f"\n\n[...truncated {filename}: {len(content):,} chars total]"
                    )
                # Best-effort relative path for display
                rel_path = str(hint_path)
                try:
                    rel_path = str(hint_path.relative_to(self.working_dir))
                except (ValueError, RuntimeError):
                    try:
                        rel_path = str(hint_path.relative_to(Path.home()))
                        rel_path = "~/" + rel_path
                    except (ValueError, RuntimeError):
                        pass  # keep absolute
                found_hints.append((rel_path, content))
                # First match wins per directory (like startup loading)
                break
            except Exception as exc:
                logger.debug("Could not read %s: %s", hint_path, exc)

        if not found_hints:
            return None

        sections = []
        for rel_path, content in found_hints:
            sections.append(
                f"[Subdirectory context discovered: {rel_path}]\n{content}"
            )

        logger.debug(
            "Loaded subdirectory hints from %s: %s",
            directory,
            [h[0] for h in found_hints],
        )
        return "\n\n".join(sections)
