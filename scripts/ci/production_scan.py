from __future__ import annotations
import io
import re
import subprocess
import tokenize
from pathlib import Path

from hermes_agent.compat.paths import PathMoves
from scripts.code_health import gitio

SCAN_TREES = ("agent", "cron", "gateway", "hermes_cli", "plugins", "tools", "tui_gateway", "acp_adapter", "pm")
SCAN_ROOT_FILES = ("cli.py", "run_agent.py", "hermes_state.py", "utils.py", "hermes_constants.py")
SKIP_PARTS = {"tests", "__pycache__", "evals", "node_modules"}


def in_scope(rel: str) -> bool:
    parts = Path(rel).parts
    return not (SKIP_PARTS & set(parts)) and not Path(rel).name.startswith("test_")


def is_production(original_path: str) -> bool:
    return (original_path.endswith(".py") and Path(original_path).parts[0] in (*SCAN_TREES, *SCAN_ROOT_FILES)
            and in_scope(original_path))


def original_paths(root: Path, tree: str | None):
    return PathMoves(gitio.tree_moves(root, tree)).original


def production_files(root: Path, tree: str | None = None, *, untracked: bool = False) -> dict[str, str]:
    try:
        if tree is not None:
            listed = set(gitio.tracked_files(root, tree))
        else:
            others = ["--others", "--exclude-standard"] if untracked else []
            listed = set(gitio.git(root, "ls-files", "--cached", *others, "-z").split("\0"))
            listed -= set(gitio.git(root, "ls-files", "--deleted", "-z").split("\0"))
        original = original_paths(root, tree)
        generated = gitio.generated_files(root, tree)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"cannot list the tracked files under {root}: {exc}") from exc
    files = {rel: original(rel) for rel in listed if rel and rel not in generated}
    return {rel: old for rel, old in sorted(files.items()) if is_production(old)}


def comment_lines(text: str, marker: str) -> set[int]:
    if marker not in text:
        return set()
    pattern = re.compile(re.escape(marker) + r"\s*(?:—|-+)\s*[^\s—-]")
    try:
        return {token.start[0] for token in tokenize.generate_tokens(io.StringIO(text).readline)
                if token.type == tokenize.COMMENT and pattern.search(token.string)}
    except (tokenize.TokenError, SyntaxError):
        return set()
