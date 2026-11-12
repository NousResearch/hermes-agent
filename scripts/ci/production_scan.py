"""Shared production Python inventory and reasoned line markers for CI guards."""
from __future__ import annotations
import re
import subprocess
from pathlib import Path

SCAN_TREES = ("agent", "cron", "gateway", "hermes_cli", "plugins", "tools", "tui_gateway", "acp_adapter", "pm")
SCAN_ROOT_FILES = ("cli.py", "run_agent.py", "hermes_state.py", "utils.py", "hermes_constants.py")
SKIP_PARTS = {"tests", "__pycache__", "evals", "node_modules"}

def marked(src_lines: list[str], lineno: int, marker: str) -> bool:
    pattern = re.compile(re.escape(marker) + r"\s*(?:—|-+)\s*\S")
    here = src_lines[lineno - 1] if lineno - 1 < len(src_lines) else ""
    above = src_lines[lineno - 2] if lineno >= 2 else ""
    return bool(pattern.search(here) or pattern.search(above))

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
