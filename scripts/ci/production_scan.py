"""Shared production Python inventory and reasoned line markers for CI guards."""
from __future__ import annotations
import io
import tokenize
import re
import subprocess
from pathlib import Path

SCAN_TREES = ("agent", "cron", "gateway", "hermes_cli", "plugins", "tools", "tui_gateway", "acp_adapter", "pm")
SCAN_ROOT_FILES = ("cli.py", "run_agent.py", "hermes_state.py", "utils.py", "hermes_constants.py")
SKIP_PARTS = {"tests", "__pycache__", "evals", "node_modules"}

def comment_lines(text: str, marker: str) -> set[int]:
    if marker not in text:
        return set()
    pattern = re.compile(re.escape(marker) + r"\s*(?:—|-+)\s*\S")
    return {token.start[0] for token in tokenize.generate_tokens(io.StringIO(text).readline)
            if token.type == tokenize.COMMENT and pattern.search(token.string)}


def marked(src_lines: list[str], lineno: int, marker: str, *, comments: set[int] | None = None,
           standalone_previous: bool = True) -> bool:
    if comments is None:
        comments = comment_lines('\n'.join(src_lines), marker)
    return lineno in comments or (lineno > 1 and lineno - 1 in comments and
                                 (not standalone_previous or src_lines[lineno - 2].lstrip().startswith("#")))


def tracked_files(root: Path) -> list[Path]:
    try:
        out = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "--", *SCAN_TREES, *SCAN_ROOT_FILES],
                             capture_output=True, text=True, check=True, stdin=subprocess.DEVNULL, timeout=60).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        # Fail closed: scanning nothing would report "no detached spawns".
        raise RuntimeError(f"cannot list the tracked files under {root}: {exc}") from exc
    return [root / rel for rel in out.split("\0") if rel.endswith(".py")]


def in_scope(rel: str) -> bool:
    parts = Path(rel).parts
    return not (SKIP_PARTS & set(parts)) and not Path(rel).name.startswith("test_")
