#!/usr/bin/env python3
"""
Project Change Tracking Module

Tracks file modifications, pending edits, and edit sessions to enable
conflict detection and semantic change understanding for the Hermes agent.

This module never proposes conflicting edits to the same function by
maintaining awareness of WHAT files changed, WHEN, and WHAT changed.
"""

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import ast
import subprocess

logger = logging.getLogger(__name__)


class ProjectChangeTracker:
    """
    Tracks file modifications, pending edits, and edit sessions.

    Thread-safe implementation using locks for concurrent tool calls.
    """

    def __init__(self):
        self._modified_files: Dict[str, float] = {}  # path (realpath) -> mtime
        self._edit_sessions: List[Dict[str, Any]] = []  # list of edit session records
        self._pending_edited_files: Set[str] = set()  # files written but not re-read
        self._file_content_cache: Dict[str, tuple[float, str]] = {}  # path -> (mtime, content)
        self._read_mtimes: Dict[str, float] = {}  # path -> mtime at last read
        self._patches: List[Dict[str, Any]] = []  # record of patches applied

        self._lock = threading.RLock()

    # -------------------------------------------------------------------------
    # Path normalization helpers
    # -------------------------------------------------------------------------

    def _normalize_path(self, path: str) -> str:
        """Normalize a path using realpath to handle symlinks and relative paths."""
        try:
            return os.path.realpath(os.path.expanduser(path))
        except (OSError, ValueError):
            return os.path.normpath(os.path.expanduser(path))

    # -------------------------------------------------------------------------
    # Mark methods - called by tools after operations
    # -------------------------------------------------------------------------

    def mark_file_read(self, path: str, content: str, mtime: float) -> None:
        """
        Call when a file is read. Updates cache and clears pending edit flag.

        Args:
            path: File path (will be normalized)
            content: Content that was read
            mtime: Modification time of the file at read time
        """
        with self._lock:
            normalized = self._normalize_path(path)
            self._file_content_cache[normalized] = (mtime, content)
            self._read_mtimes[normalized] = mtime
            # Clear pending flag since we've now read the file
            self._pending_edited_files.discard(normalized)
            logger.debug("Marked file read: %s (mtime=%.3f)", normalized, mtime)

    def mark_file_written(self, path: str, mtime: float) -> None:
        """
        Call when a file is written. Marks as pending so next read is forced.

        Args:
            path: File path (will be normalized)
            mtime: Modification time after write
        """
        with self._lock:
            normalized = self._normalize_path(path)
            self._modified_files[normalized] = mtime
            self._pending_edited_files.add(normalized)
            # Update cached mtime
            if normalized in self._file_content_cache:
                old_content = self._file_content_cache[normalized][1]
                self._file_content_cache[normalized] = (mtime, old_content)
            else:
                self._file_content_cache[normalized] = (mtime, "")
            logger.debug("Marked file written: %s (mtime=%.3f)", normalized, mtime)

    def mark_file_patched(self, path: str, old_string: str, new_string: str) -> None:
        """
        Call when patch() is used. Records the edit for context.

        Args:
            path: File path (will be normalized)
            old_string: The string that was replaced
            new_string: The replacement string
        """
        with self._lock:
            normalized = self._normalize_path(path)
            try:
                current_mtime = os.path.getmtime(normalized)
            except OSError:
                current_mtime = time.time()

            self._modified_files[normalized] = current_mtime
            self._pending_edited_files.add(normalized)

            patch_record = {
                "path": normalized,
                "old_string": old_string,
                "new_string": new_string,
                "timestamp": time.time(),
                "mtime": current_mtime,
            }
            self._patches.append(patch_record)
            logger.debug("Marked file patched: %s", normalized)

    # -------------------------------------------------------------------------
    # Query methods
    # -------------------------------------------------------------------------

    def get_changed_since(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Return files changed since session started.

        Args:
            session_id: The session identifier

        Returns:
            List of dicts with path, mtime, and type of change
        """
        with self._lock:
            result = []
            for session in self._edit_sessions:
                if session.get("task_id") == session_id:
                    session_start = session.get("started_at", 0)
                    break
            else:
                # Session not found, return all modified files
                session_start = 0

            for path, mtime in self._modified_files.items():
                if mtime > session_start:
                    result.append({
                        "path": path,
                        "mtime": mtime,
                        "type": "modified",
                        "pending": path in self._pending_edited_files,
                    })

            return result

    def check_conflict(self, path: str) -> bool:
        """
        Return True if file was modified externally since last read.

        A conflict exists if:
        1. The file has been read before
        2. The file's current mtime is different from its mtime at last read
        3. The file is NOT in the pending edited files set (we already know about our own writes)

        Args:
            path: File path to check

        Returns:
            True if a conflict is detected
        """
        with self._lock:
            normalized = self._normalize_path(path)

            # If never read, no conflict possible
            if normalized not in self._read_mtimes:
                return False

            # If we have a pending edit for this file, no conflict
            if normalized in self._pending_edited_files:
                return False

            last_read_mtime = self._read_mtimes[normalized]

            try:
                current_mtime = os.path.getmtime(normalized)
            except OSError:
                # File might have been deleted or inaccessible
                return False

            return current_mtime != last_read_mtime

    def get_file_mtime(self, path: str) -> Optional[float]:
        """
        Get current mtime of a file.

        Args:
            path: File path

        Returns:
            Current mtime or None if file doesn't exist or can't be accessed
        """
        with self._lock:
            normalized = self._normalize_path(path)
            try:
                return os.path.getmtime(normalized)
            except OSError:
                return None

    def get_pending_files(self) -> List[str]:
        """Return list of files that have been written/patched but not re-read."""
        with self._lock:
            return list(self._pending_edited_files)

    def get_session_summary(self, session_idx: Optional[int] = None) -> Dict[str, Any]:
        """
        Get summary of an edit session.

        Args:
            session_idx: Optional session index. If None, returns all sessions.

        Returns:
            Dict with session information
        """
        with self._lock:
            if session_idx is not None:
                if 0 <= session_idx < len(self._edit_sessions):
                    return dict(self._edit_sessions[session_idx])
                return {}
            else:
                return {
                    "total_sessions": len(self._edit_sessions),
                    "sessions": [dict(s) for s in self._edit_sessions],
                }

    def get_modified_files(self) -> List[Dict[str, Any]]:
        """Return all modified files with their mtimes."""
        with self._lock:
            return [
                {"path": path, "mtime": mtime, "pending": path in self._pending_edited_files}
                for path, mtime in self._modified_files.items()
            ]

    def get_session_edits(self, session_idx: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Return all edits (writes/patches) for a session or all sessions.

        Args:
            session_idx: Optional session index. If None, returns all edits.

        Returns:
            List of dicts with path, old_string, new_string, timestamp
        """
        with self._lock:
            if session_idx is not None:
                if 0 <= session_idx < len(self._edit_sessions):
                    session = self._edit_sessions[session_idx]
                    patches = [p for p in self._patches
                               if p.get("timestamp", 0) >= session.get("started_at", 0)
                               and (session.get("ended_at") is None or p.get("timestamp", 0) <= session.get("ended_at", float("inf")))]
                    return patches
                return []
            return list(self._patches)

    def get_file_content_near_changes(self, path: str, context_lines: int = 10) -> Optional[str]:
        """
        Read file content near where changes were made.

        Args:
            path: File path to read
            context_lines: Number of lines of context around changes to include

        Returns:
            File content string with change context, or None if file can't be read
        """
        with self._lock:
            normalized = self._normalize_path(path)

            # Find patches for this file
            file_patches = [p for p in self._patches if p.get("path") == normalized]
            if not file_patches:
                return None

            try:
                with open(normalized, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            except (OSError, IOError):
                return None

            if not lines:
                return ""

            # Find line ranges affected by patches
            affected_ranges = []
            for patch in file_patches:
                old_str = patch.get("old_string", "")
                new_str = patch.get("new_string", "")

                # Simple heuristic: find where old_string appears
                old_lines = old_str.split("\n") if old_str else []
                if len(old_lines) > 1:
                    # Multi-line: find approximate region
                    # This is a rough approximation based on patch record
                    for idx, line in enumerate(lines):
                        if old_lines[0] in line:
                            start = max(0, idx - context_lines)
                            end = min(len(lines), idx + len(old_lines) + context_lines)
                            affected_ranges.append((start, end))
                            break

            if not affected_ranges:
                # Fallback: return full file if we can't find specific regions
                return "".join(lines)

            # Merge overlapping ranges and extract content
            affected_ranges.sort()
            merged = []
            for start, end in affected_ranges:
                if merged and start <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                else:
                    merged.append((start, end))

            result_lines = []
            for start, end in merged:
                if start > 0:
                    result_lines.append(f"... (lines {start}-{end}) ...")
                result_lines.append("".join(lines[start:end]))
                if end < len(lines):
                    result_lines.append(f"... (lines {end}) ...")

            return "\n".join(result_lines)

    # -------------------------------------------------------------------------
    # Edit session management
    # -------------------------------------------------------------------------

    def start_edit_session(self, task_id: str) -> int:
        """
        Begin a new edit session, return session index.

        Args:
            task_id: Unique task identifier

        Returns:
            Session index that can be used to end the session
        """
        with self._lock:
            session = {
                "task_id": task_id,
                "started_at": time.time(),
                "ended_at": None,
                "files_modified": [],
                "patches_applied": [],
            }
            session_idx = len(self._edit_sessions)
            self._edit_sessions.append(session)
            logger.debug("Started edit session %d for task %s", session_idx, task_id)
            return session_idx

    def end_edit_session(self, session_idx: int) -> None:
        """
        Close an edit session.

        Args:
            session_idx: Index returned by start_edit_session
        """
        with self._lock:
            if 0 <= session_idx < len(self._edit_sessions):
                self._edit_sessions[session_idx]["ended_at"] = time.time()
                logger.debug("Ended edit session %d", session_idx)

    def add_file_to_session(self, session_idx: int, file_path: str) -> None:
        """Add a file to an edit session's modification list."""
        with self._lock:
            if 0 <= session_idx < len(self._edit_sessions):
                if file_path not in self._edit_sessions[session_idx]["files_modified"]:
                    self._edit_sessions[session_idx]["files_modified"].append(file_path)

    # -------------------------------------------------------------------------
    # Utility methods
    # -------------------------------------------------------------------------

    def clear(self) -> None:
        """Clear all tracking state. Use with caution."""
        with self._lock:
            self._modified_files.clear()
            self._pending_edited_files.clear()
            self._file_content_cache.clear()
            self._read_mtimes.clear()
            self._patches.clear()
            # Keep edit sessions but mark them as ended
            now = time.time()
            for session in self._edit_sessions:
                if session.get("ended_at") is None:
                    session["ended_at"] = now



# -------------------------------------------------------------------------
# Git status and diff methods
# -------------------------------------------------------------------------

    def _git_status(self) -> Dict[str, Any]:
        """
        Run git status --porcelain and git diff --stat to get changed files.
    
        Returns:
            Dict with changed files and their change types (M/A/D/R)
        """
        with self._lock:
            result: Dict[str, Any] = {
                "changed_files": [],
                "summary": {},
                "error": None,
            }
            try:
                import subprocess
    
                # Get porcelain status
                status_proc = subprocess.run(
                    ["git", "status", "--porcelain"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if status_proc.returncode != 0:
                    result["error"] = f"git status failed: {status_proc.stderr}"
                    return result
    
                # Parse porcelain format: XY filename
                # X = index status, Y = working tree status
                # M = modified, A = added, D = deleted, R = renamed, ?? = untracked
                changed_files = []
                for line in status_proc.stdout.strip().split("\n"):
                    if not line:
                        continue
                    if len(line) < 3:
                        continue
                    status_code = line[:2]
                    filepath = line[3:].strip()
    
                    # Determine primary change type
                    if status_code[0] in "MADR":
                        change_type = status_code[0]
                    elif status_code[1] in "MADR":
                        change_type = status_code[1]
                    elif status_code == "??":
                        change_type = "??"
                    else:
                        change_type = status_code
    
                    changed_files.append({
                        "path": filepath,
                        "index_status": status_code[0],
                        "worktree_status": status_code[1],
                        "change_type": change_type,
                    })
    
                result["changed_files"] = changed_files
    
                # Get diff stat
                stat_proc = subprocess.run(
                    ["git", "diff", "--stat"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if stat_proc.returncode == 0:
                    result["diff_stat"] = stat_proc.stdout.strip()
                else:
                    result["diff_stat"] = f"git diff --stat failed: {stat_proc.stderr}"
    
                # Summary counts
                summary: Dict[str, int] = {}
                for f in changed_files:
                    ct = f["change_type"]
                    summary[ct] = summary.get(ct, 0) + 1
                result["summary"] = summary
    
            except subprocess.TimeoutExpired:
                result["error"] = "git command timed out"
            except FileNotFoundError:
                result["error"] = "git command not found"
            except Exception as e:
                result["error"] = str(e)
    
            return result
    
    def _parse_hunk_diff(self, path: str) -> Dict[str, Any]:
        """
        Run git diff for a specific path to get hunk-level changes with line numbers.
    
        Args:
            path: File path to get diff for
    
        Returns:
            Dict with hunks, line numbers, and change statistics
        """
        with self._lock:
            result: Dict[str, Any] = {
                "path": path,
                "hunks": [],
                "error": None,
            }
            try:
                import subprocess
    
                normalized = self._normalize_path(path)
    
                # Get full diff for the file
                diff_proc = subprocess.run(
                    ["git", "diff", "--", normalized],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
    
                if diff_proc.returncode not in (0, 1):
                    # 1 means diff found differences, which is OK
                    result["error"] = f"git diff failed: {diff_proc.stderr}"
                    return result
    
                diff_text = diff_proc.stdout
                if not diff_text:
                    result["hunks"] = []
                    return result
    
                # Parse unified diff hunks
                # Format: @@ -old_start,old_count +new_start,new_count @@
                import re
    
                hunks = []
                current_hunk: Dict[str, Any] = {}
                old_line_start: int = 0
                new_line_start: int = 0
                old_lines: List[str] = []
                new_lines: List[str] = []
    
                for line in diff_text.split("\n"):
                    if line.startswith("@@"):
                        # Save previous hunk if exists
                        if current_hunk:
                            current_hunk["old_lines"] = old_lines
                            current_hunk["new_lines"] = new_lines
                            hunks.append(current_hunk)
    
                        # Parse hunk header
                        m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
                        if m:
                            old_start = int(m.group(1))
                            old_count = int(m.group(2)) if m.group(2) else 1
                            new_start = int(m.group(3))
                            new_count = int(m.group(4)) if m.group(4) else 1
    
                            current_hunk = {
                                "old_start": old_start,
                                "old_count": old_count,
                                "new_start": new_start,
                                "new_count": new_count,
                                "header": line,
                                "changes": [],
                            }
                            old_line_start = old_start
                            new_line_start = new_start
                            old_lines = []
                            new_lines = []
                    elif current_hunk:
                        if line.startswith("-"):
                            old_lines.append(line[1:])
                            current_hunk["changes"].append({
                                "type": "deletion",
                                "old_line": old_line_start,
                                "content": line[1:],
                            })
                            old_line_start += 1
                        elif line.startswith("+"):
                            new_lines.append(line[1:])
                            current_hunk["changes"].append({
                                "type": "addition",
                                "new_line": new_line_start,
                                "content": line[1:],
                            })
                            new_line_start += 1
                        elif line.startswith(" ") or line == "":
                            # Context line
                            old_lines.append(line[1:] if len(line) > 1 else "")
                            new_lines.append(line[1:] if len(line) > 1 else "")
                            old_line_start += 1
                            new_line_start += 1
    
                # Don't forget the last hunk
                if current_hunk:
                    current_hunk["old_lines"] = old_lines
                    current_hunk["new_lines"] = new_lines
                    hunks.append(current_hunk)
    
                result["hunks"] = hunks
                result["hunk_count"] = len(hunks)
    
            except subprocess.TimeoutExpired:
                result["error"] = "git diff timed out"
            except FileNotFoundError:
                result["error"] = "git command not found"
            except Exception as e:
                result["error"] = str(e)
    
            return result
    
    # -------------------------------------------------------------------------
    # Semantic Python analysis methods
    # -------------------------------------------------------------------------
    
    def _semantic_python_changes(self, path: str) -> Dict[str, Any]:
        """
        Use Python AST to parse a file and return function/class/import changes.
    
        For new files, returns all definitions as "added".
        For modified files, attempts to diff with git version.
    
        Args:
            path: File path to analyze
    
        Returns:
            Dict with added functions, removed functions, renamed functions,
            changed signatures, and added imports
        """
        with self._lock:
            result: Dict[str, Any] = {
                "path": path,
                "added_functions": [],
                "removed_functions": [],
                "renamed_functions": [],
                "changed_signatures": [],
                "added_imports": [],
                "added_classes": [],
                "removed_classes": [],
                "error": None,
            }
    
            try:
                import ast
                import subprocess
    
                normalized = self._normalize_path(path)
    
                # Read current file content
                try:
                    with open(normalized, "r", encoding="utf-8") as f:
                        current_content = f.read()
                except Exception as e:
                    result["error"] = f"Could not read file: {e}"
                    return result
    
                # Get the git version of the file (if it exists in git)
                git_content: Optional[str] = None
                try:
                    git_proc = subprocess.run(
                        ["git", "show", f"HEAD:{normalized}"],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    if git_proc.returncode == 0:
                        git_content = git_proc.stdout
                except Exception:
                    pass  # File might be new or git failed
    
                # Parse current content
                try:
                    current_tree = ast.parse(current_content, filename=normalized)
                except SyntaxError as e:
                    result["error"] = f"Syntax error in file: {e}"
                    return result
    
                current_info = self._extract_python_definitions(current_tree)
    
                if git_content is None:
                    # New file - everything is added
                    result["added_functions"] = current_info["functions"]
                    result["added_classes"] = current_info["classes"]
                    result["added_imports"] = current_info["imports"]
                else:
                    # Compare with git version
                    try:
                        git_tree = ast.parse(git_content, filename=normalized)
                    except SyntaxError:
                        # If git content can't be parsed, treat all as added
                        result["added_functions"] = current_info["functions"]
                        result["added_classes"] = current_info["classes"]
                        result["added_imports"] = current_info["imports"]
                        return result
    
                    git_info = self._extract_python_definitions(git_tree)
    
                    # Find added functions (in current but not in git)
                    current_funcs = {f["name"]: f for f in current_info["functions"]}
                    git_funcs = {f["name"]: f for f in git_info["functions"]}
    
                    for name, func in current_funcs.items():
                        if name not in git_funcs:
                            result["added_functions"].append(func)
                        else:
                            # Check if signature changed
                            if func["signature"] != git_funcs[name]["signature"]:
                                result["changed_signatures"].append({
                                    "name": name,
                                    "old_signature": git_funcs[name]["signature"],
                                    "new_signature": func["signature"],
                                })
    
                    # Find removed functions (in git but not in current)
                    for name, func in git_funcs.items():
                        if name not in current_funcs:
                            result["removed_functions"].append(func)
    
                    # Find renamed functions (similar signatures but different names)
                    for curr_name, curr_func in current_funcs.items():
                        for git_name, git_func in git_funcs.items():
                            if (curr_name != git_name and
                                curr_func["signature"] == git_func["signature"] and
                                curr_name not in git_funcs and
                                git_name not in current_funcs):
                                result["renamed_functions"].append({
                                    "old_name": git_name,
                                    "new_name": curr_name,
                                    "signature": curr_func["signature"],
                                })
    
                    # Find added classes
                    current_classes = {c["name"]: c for c in current_info["classes"]}
                    git_classes = {c["name"]: c for c in git_info["classes"]}
    
                    for name, cls in current_classes.items():
                        if name not in git_classes:
                            result["added_classes"].append(cls)
    
                    for name, cls in git_classes.items():
                        if name not in current_classes:
                            result["removed_classes"].append(cls)
    
                    # Find added imports
                    for imp in current_info["imports"]:
                        if imp not in git_info["imports"]:
                            result["added_imports"].append(imp)
    
            except subprocess.TimeoutExpired:
                result["error"] = "git command timed out"
            except FileNotFoundError:
                result["error"] = "git command not found"
            except Exception as e:
                result["error"] = str(e)
    
            return result
    
    def _extract_python_definitions(self, tree: ast.AST) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract function, class, and import definitions from an AST tree.
    
        Args:
            tree: Parsed AST tree
    
        Returns:
            Dict with 'functions', 'classes', and 'imports' lists
        """
        functions: List[Dict[str, Any]] = []
        classes: List[Dict[str, Any]] = []
        imports: List[str] = []
    
        for node in ast.walk(tree):
            # Handle imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name if alias.asname is None else f"{alias.name} as {alias.asname}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    if alias.asname:
                        imports.append(f"from {module} import {alias.name} as {alias.asname}")
                    else:
                        imports.append(f"from {module} import {alias.name}")
    
            # Handle function definitions
            elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                args = node.args
                arg_names: List[str] = []
                for arg in args.args:
                    arg_names.append(arg.arg)
                for arg in args.kwonlyargs:
                    arg_names.append(arg.arg)
                if args.vararg:
                    arg_names.append(f"*{args.vararg.arg}")
                if args.kwarg:
                    arg_names.append(f"**{args.kwarg.arg}")
    
                functions.append({
                    "name": node.name,
                    "signature": f"{node.name}({', '.join(arg_names)})",
                    "line": node.lineno,
                    "is_async": isinstance(node, ast.AsyncFunctionDef),
                })
    
            # Handle class definitions
            elif isinstance(node, ast.ClassDef):
                base_names: List[str] = []
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        base_names.append(base.id)
                    elif isinstance(base, ast.Attribute):
                        base_names.append(ast.unparse(base))
    
                classes.append({
                    "name": node.name,
                    "bases": base_names,
                    "line": node.lineno,
                })
    
        return {
            "functions": functions,
            "classes": classes,
            "imports": imports,
        }
    
    # -------------------------------------------------------------------------
    # Impact analysis methods
    # -------------------------------------------------------------------------
    
    def _find_impacted_files(self, path: str) -> Dict[str, Any]:
        """
        Search for files that import from or reference this path.
    
        Args:
            path: File path to find impacted files for
    
        Returns:
            Dict with list of impacted files and their import statements
        """
        with self._lock:
            result: Dict[str, Any] = {
                "target_path": path,
                "impacted_files": [],
                "error": None,
            }
    
            try:
                import re
                import subprocess
    
                normalized = self._normalize_path(path)
    
                # Get the module name from the path
                # e.g., /path/to/project/agent/tools/diff_patch.py -> agent.tools.diff_patch
                parts = Path(normalized).parts
    
                # Try to construct module name
                module_name: Optional[str] = None
                if normalized.endswith(".py"):
                    # Remove .py and try to find module path
                    module_path = normalized[:-3]
                    # Look for common base paths
                    possible_bases = []
                    try:
                        git_proc = subprocess.run(
                            ["git", "rev-parse", "--show-toplevel"],
                            capture_output=True,
                            text=True,
                            timeout=5,
                        )
                        if git_proc.returncode == 0:
                            git_root = git_proc.stdout.strip()
                            possible_bases.append(git_root)
                    except Exception:
                        pass
    
                    for base in possible_bases:
                        if module_path.startswith(base):
                            rel_path = module_path[len(base):].lstrip("/")
                            # Convert to module notation
                            module_name = rel_path.replace("/", ".").replace("\\", ".")
                            break
    
                if module_name is None:
                    # Fallback: use the filename without extension as import name
                    module_name = Path(normalized).stem
    
                # Find all Python files that might import this module
                search_patterns = [
                    f"from {re.escape(module_name)}",
                    f"import {re.escape(module_name)}",
                    f"from {re.escape(Path(normalized).name[:-3])}",
                    f"import {re.escape(Path(normalized).name[:-3])}",
                ]
    
                impacted: Dict[str, List[str]] = {}
    
                for pattern in search_patterns:
                    try:
                        grep_proc = subprocess.run(
                            ["grep", "-r", "-l", pattern, "--include=*.py"],
                            capture_output=True,
                            text=True,
                            timeout=30,
                        )
                        if grep_proc.returncode == 0:
                            for filepath in grep_proc.stdout.strip().split("\n"):
                                if filepath and filepath != normalized:
                                    if filepath not in impacted:
                                        impacted[filepath] = []
                                    impacted[filepath].append(pattern)
                    except Exception:
                        continue
    
                result["impacted_files"] = [
                    {"path": path, "matched_imports": matches}
                    for path, matches in impacted.items()
                ]
                result["count"] = len(result["impacted_files"])
    
            except subprocess.TimeoutExpired:
                result["error"] = "search timed out"
            except FileNotFoundError:
                result["error"] = "grep command not found"
            except Exception as e:
                result["error"] = str(e)
    
            return result
    
    # -------------------------------------------------------------------------
    # External modification detection
    # -------------------------------------------------------------------------
    
    def _detect_external_modifications(self) -> Dict[str, Any]:
        """
        Check if any tracked files were modified externally since last read.
    
        Compares current mtime vs recorded mtime in _read_mtimes.
    
        Returns:
            Dict with externally modified files and their mtime changes
        """
        with self._lock:
            result: Dict[str, Any] = {
                "externally_modified": [],
                "checked_count": 0,
                "error": None,
            }
    
            try:
                externally_modified: List[Dict[str, Any]] = []
    
                for path, last_read_mtime in self._read_mtimes.items():
                    result["checked_count"] += 1
    
                    try:
                        current_mtime = os.path.getmtime(path)
                    except OSError:
                        # File might have been deleted
                        externally_modified.append({
                            "path": path,
                            "status": "deleted",
                            "last_read_mtime": last_read_mtime,
                            "current_mtime": None,
                        })
                        continue
    
                    if current_mtime != last_read_mtime:
                        # Check if this is one of our own pending edits
                        if path in self._pending_edited_files:
                            continue
    
                        externally_modified.append({
                            "path": path,
                            "status": "modified",
                            "last_read_mtime": last_read_mtime,
                            "current_mtime": current_mtime,
                            "delta_seconds": current_mtime - last_read_mtime,
                        })
    
                result["externally_modified"] = externally_modified
                result["count"] = len(externally_modified)
    
            except Exception as e:
                result["error"] = str(e)
    
            return result
    
    
    
    # Project Context Tool - Updated with new actions
    # =============================================================================
    
    

_PROJECT_CONTEXT_TRACKER: Optional["ProjectChangeTracker"] = None
_tracker_lock = threading.Lock()


def get_project_tracker() -> "ProjectChangeTracker":
    """Get or create the global ProjectChangeTracker instance."""
    global _PROJECT_CONTEXT_TRACKER
    if _PROJECT_CONTEXT_TRACKER is None:
        with _tracker_lock:
            if _PROJECT_CONTEXT_TRACKER is None:
                _PROJECT_CONTEXT_TRACKER = ProjectChangeTracker()
    return _PROJECT_CONTEXT_TRACKER


def set_project_tracker(tracker: "ProjectChangeTracker") -> None:
    """Set the global ProjectChangeTracker instance (for testing)."""
    global _PROJECT_CONTEXT_TRACKER
    with _tracker_lock:
        _PROJECT_CONTEXT_TRACKER = tracker


def project_context_tool(action: str, path: str = None, task_id: str = None) -> str:
    """
    Query project change tracking information.

    Args:
        action: One of:
            - "list_changed": List files modified since session/task started
            - "check_file": Check if a specific file has conflicts
            - "session_summary": Get summary of edit sessions
            - "pending": List files that were written/patched but not re-read
            - "git_status": Get git status with changed files and change types
            - "semantic_changes": Get AST-based semantic changes for a Python file
            - "impact": Find files that import from the given path
            - "external_changes": Detect files modified externally since last read
        path: (optional) Specific file to check for check_file action
        task_id: (optional) Task/session identifier for context

    Returns:
        JSON string with results
    """
    tracker = get_project_tracker()

    try:
        if action == "list_changed":
            if task_id:
                changes = tracker.get_changed_since(task_id)
            else:
                changes = tracker.get_modified_files()
            return json.dumps({
                "action": "list_changed",
                "changed_files": changes,
                "count": len(changes),
            }, ensure_ascii=False)

        elif action == "check_file":
            if not path:
                return json.dumps({"error": "path is required for check_file action"})
            has_conflict = tracker.check_conflict(path)
            current_mtime = tracker.get_file_mtime(path)
            is_pending = path in tracker.get_pending_files()
            return json.dumps({
                "action": "check_file",
                "path": path,
                "has_conflict": has_conflict,
                "current_mtime": current_mtime,
                "is_pending": is_pending,
            }, ensure_ascii=False)

        elif action == "session_summary":
            summary = tracker.get_session_summary()
            return json.dumps({
                "action": "session_summary",
                **summary,
            }, ensure_ascii=False)

        elif action == "pending":
            pending = tracker.get_pending_files()
            return json.dumps({
                "action": "pending",
                "pending_files": pending,
                "count": len(pending),
            }, ensure_ascii=False)

        elif action == "git_status":
            status = tracker._git_status()
            return json.dumps({
                "action": "git_status",
                **status,
            }, ensure_ascii=False)

        elif action == "semantic_changes":
            if not path:
                return json.dumps({"error": "path is required for semantic_changes action"})
            changes = tracker._semantic_python_changes(path)
            return json.dumps({
                "action": "semantic_changes",
                **changes,
            }, ensure_ascii=False)

        elif action == "impact":
            if not path:
                return json.dumps({"error": "path is required for impact action"})
            impact = tracker._find_impacted_files(path)
            return json.dumps({
                "action": "impact",
                **impact,
            }, ensure_ascii=False)

        elif action == "external_changes":
            external = tracker._detect_external_modifications()
            return json.dumps({
                "action": "external_changes",
                **external,
            }, ensure_ascii=False)

        else:
            return json.dumps({
                "error": f"Unknown action: {action}",
                "valid_actions": [
                    "list_changed", "check_file", "session_summary",
                    "pending", "git_status", "semantic_changes", "impact",
                    "external_changes"
                ],
            })

    except Exception as e:
        logger.error("project_context_tool error: %s", str(e))
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# Update the schema to include new actions
PROJECT_CONTEXT_SCHEMA = {
    "name": "project_context",
    "description": """Query project change tracking information.

Use this tool to understand what files have been modified in the project,
check for conflicts, and see pending edits that haven't been read back.

Actions:
- list_changed: List all files modified since tracking started or since a session
- check_file: Check if a specific file has been modified externally (conflict detection)
- session_summary: Get information about edit sessions
- pending: List files that were written/patched but not yet re-read
- git_status: Get detailed git status with file change types (M/A/D/R)
- semantic_changes: Get AST-based semantic changes for Python files (functions, classes, imports)
- impact: Find files that import from or reference the given path
- external_changes: Detect files that were modified externally since last read""",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "list_changed", "check_file", "session_summary",
                    "pending", "git_status", "semantic_changes", "impact",
                    "external_changes"
                ],
                "description": "The action to perform",
            },
            "path": {
                "type": "string",
                "description": "File path to check (required for check_file, semantic_changes, impact actions)",
            },
            "task_id": {
                "type": "string",
                "description": "Task/session identifier for context",
            },
        },
        "required": ["action"],
    },
}
def _handle_project_context(args, **kw) -> str:
    """Handler for the project_context tool."""
    tid = kw.get("task_id") or "default"
    tracker = get_project_tracker()
    return project_context_tool(
        action=args.get("action", ""),
        path=args.get("path"),
        task_id=tid,
    )



# =============================================================================
# Register as a Hermes model tool via the central registry
# =============================================================================
try:
    from tools.registry import registry
    registry.register(
        name="project_context",
        toolset="hermes-core",
        schema=PROJECT_CONTEXT_SCHEMA,
        handler=lambda args, **kw: _handle_project_context(args, **kw),
        description="Query project change tracking — list modified files, detect conflicts, check pending edits",
        emoji="📋",
        is_read_only=True,
        is_concurrency_safe=True,
    )
except Exception as e:
    import logging
    logging.getLogger(__name__).warning("Could not register project_context tool: %s", e)
