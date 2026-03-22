"""Parse @-mentions in user messages and expand them to real content.

Supported mentions:
  @diff          -> git diff output (staged + unstaged)
  @git-log, @log -> last 20 git log entries (--oneline)
  @tree          -> directory tree (depth 3, excluding common dirs)
  @problems, @errors -> run ruff check or detected linter
  @file:<path>   -> contents of the specified file
  @search:<term> -> ripgrep search results for the term
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Tuple


def _run(cmd: list[str], cwd: str, timeout: int = 10) -> str:
    """Run a command and return stdout, or an error note."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=cwd, timeout=timeout
        )
        return (result.stdout or "").strip() or (result.stderr or "").strip() or "(no output)"
    except FileNotFoundError:
        return f"(command not found: {cmd[0]})"
    except subprocess.TimeoutExpired:
        return "(command timed out)"
    except Exception as exc:
        return f"(error: {exc})"


def _expand_diff(cwd: str) -> str:
    staged = _run(["git", "diff", "--staged"], cwd)
    unstaged = _run(["git", "diff"], cwd)
    parts = []
    if staged and staged != "(no output)":
        parts.append(f"=== Staged changes ===\n{staged}")
    if unstaged and unstaged != "(no output)":
        parts.append(f"=== Unstaged changes ===\n{unstaged}")
    return "\n\n".join(parts) if parts else "(no changes)"


def _expand_log(cwd: str) -> str:
    return _run(["git", "log", "--oneline", "-20"], cwd)


def _expand_tree(cwd: str) -> str:
    excludes = [".git", "node_modules", "__pycache__", ".venv", "venv", ".tox"]
    # Try 'tree' command first, fall back to 'find'
    tree_out = _run(
        ["tree", "-L", "3", "--noreport"]
        + [item for d in excludes for item in ("-I", d)],
        cwd,
    )
    if "command not found" in tree_out:
        # Fallback: use find
        find_cmd = ["find", ".", "-maxdepth", "3", "-not", "-path", "./.git/*"]
        for d in excludes[1:]:
            find_cmd.extend(["-not", "-path", f"./{d}/*"])
        tree_out = _run(find_cmd, cwd)
    return tree_out


def _expand_problems(cwd: str) -> str:
    # Check if ruff is available and pyproject.toml exists
    if (Path(cwd) / "pyproject.toml").exists() or (Path(cwd) / "setup.py").exists():
        out = _run(["ruff", "check", "."], cwd, timeout=30)
        if "command not found" not in out:
            return out
    # Try flake8 as fallback
    out = _run(["flake8", "."], cwd, timeout=30)
    if "command not found" not in out:
        return out
    return "(no linter found — install ruff or flake8)"


def _expand_file(path: str, cwd: str) -> str:
    target = Path(cwd) / path
    if not target.exists():
        return f"(file not found: {path})"
    try:
        content = target.read_text(encoding="utf-8", errors="replace")
        # Truncate very large files
        if len(content) > 50_000:
            content = content[:50_000] + "\n... (truncated at 50K chars)"
        return content
    except Exception as exc:
        return f"(error reading {path}: {exc})"


def _expand_search(term: str, cwd: str) -> str:
    return _run(
        ["rg", "--no-heading", "--line-number", "-m", "30", term],
        cwd,
        timeout=15,
    )


# Regex to match @-mentions.  Order matters: file/search with colon first.
_MENTION_PATTERN = re.compile(
    r"@(file):([^\s]+)"       # @file:path/to/file
    r"|@(search):([^\s]+)"    # @search:term
    r"|@(diff)\b"             # @diff
    r"|@(git-log|log)\b"      # @git-log or @log
    r"|@(tree)\b"             # @tree
    r"|@(problems|errors)\b", # @problems or @errors
    re.IGNORECASE,
)


def expand_mentions(text: str, cwd: str) -> Tuple[str, str]:
    """Parse @-mentions in *text* and expand them to real content.

    Returns:
        (cleaned_text, expanded_context_block)
        - cleaned_text: the original text with @-mentions removed
        - expanded_context_block: a formatted block with expanded content
          (empty string if no mentions were found)
    """
    if not cwd:
        cwd = os.getcwd()

    sections: list[str] = []
    mentions_found: list[str] = []

    for m in _MENTION_PATTERN.finditer(text):
        if m.group(1):  # @file:path
            path = m.group(2)
            mentions_found.append(m.group(0))
            content = _expand_file(path, cwd)
            sections.append(f"📄 File: {path}\n```\n{content}\n```")
        elif m.group(3):  # @search:term
            term = m.group(4)
            mentions_found.append(m.group(0))
            content = _expand_search(term, cwd)
            sections.append(f"🔍 Search: {term}\n```\n{content}\n```")
        elif m.group(5):  # @diff
            mentions_found.append(m.group(0))
            content = _expand_diff(cwd)
            sections.append(f"📝 Git Diff\n```diff\n{content}\n```")
        elif m.group(6):  # @git-log / @log
            mentions_found.append(m.group(0))
            content = _expand_log(cwd)
            sections.append(f"📋 Git Log (last 20)\n```\n{content}\n```")
        elif m.group(7):  # @tree
            mentions_found.append(m.group(0))
            content = _expand_tree(cwd)
            sections.append(f"🌳 Directory Tree\n```\n{content}\n```")
        elif m.group(8):  # @problems / @errors
            mentions_found.append(m.group(0))
            content = _expand_problems(cwd)
            sections.append(f"⚠️ Linter Output\n```\n{content}\n```")

    if not sections:
        return text, ""

    # Remove mentions from text
    cleaned = _MENTION_PATTERN.sub("", text).strip()
    # Collapse multiple spaces
    cleaned = re.sub(r"  +", " ", cleaned)

    expanded = "[Expanded context from @-mentions]\n\n" + "\n\n".join(sections)
    return cleaned, expanded
