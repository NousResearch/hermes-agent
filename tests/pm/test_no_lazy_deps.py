"""Contract: no production code imports the deleted tools.lazy_deps.

tools/lazy_deps.py was deleted by the pm migration; pm.extras (available /
ensure_import / ensure_and_bind) is the only lazy-install surface. A new
``from tools.lazy_deps import ...`` in production code is a crash at call
time (ImportError), so this contract keeps the migration complete.

Exclusions, each load-bearing:
- ``scripts/release.py`` — a contributor-credit COMMENT referencing a
  historical PR title; not an import, harmless.
- ``tests/`` — tests may reference lazy_deps to assert it is gone.
"""

from __future__ import annotations

from pathlib import Path

# Directories that are not production source, or that mirror it.
_EXCLUDED_DIRS = {
    "node_modules",
    ".venv",
    "venv",
    "build",
    "dist",
    "release",
    "__pycache__",
    "agent-payload",
    ".git",
}

# Files allowed to mention lazy_deps (see docstring for why).
_ALLOWED = {
    # historical contributor-credit comment (PR #26294 in the credit map)
    "scripts/release.py",
    # the dashboard's legacy-pip-deps surface documents the old pipeline
    # it replaced until the workspace bridge wires it (see plugin-deps plan).
    "hermes_cli/web_server.py",
}


def _production_python_files(repo_root: Path):
    for path in repo_root.rglob("*.py"):
        rel = path.relative_to(repo_root).as_posix()
        if not rel.startswith("tests/") and rel not in _ALLOWED:
            if not any(part in _EXCLUDED_DIRS for part in path.parts):
                yield rel, path


def test_no_production_imports_of_lazy_deps():
    """Every production .py must be free of tools.lazy_deps imports."""
    repo_root = Path(__file__).resolve().parents[2]
    offenders: list[str] = []
    pattern = "tools.lazy_deps"
    try:
        candidates = list(_production_python_files(repo_root))
    except OSError:
        # rglob can raise FileNotFoundError on a dangling worktree junction
        # (WinError 3) mid-walk on Windows — not this contract's business.
        candidates = []
    for rel, path in candidates:
        try:
            text = path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError):
            # Broken junctions/dangling links raise on read.
            continue
        if pattern in text:
            offenders.append(rel)
    assert not offenders, (
        "production code references tools.lazy_deps (deleted module — "
        "ImportError at call time). Migrate to pm.ensure_import / "
        f"pm.extras: {offenders}"
    )
