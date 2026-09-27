"""Packaging invariant: every root-level module that packaged code imports ships in the wheel.

``packages.find`` only sees directories with ``__init__.py``; root single-file modules
(``run_agent``, ``hermes_state``, ``toolsets``...) reach the wheel through
``setup.py::_root_py_modules()``, which derives the list from the tree at build time. A
static list drifted every time the root layout changed and broke installed wheels with
``ModuleNotFoundError`` on ``import hermes_state``. This test pins the two halves of that
contract: the derived list covers every root module packaged code can import, and the
helper stays the single source (no static ``py-modules`` creeping back into pyproject).
"""
from __future__ import annotations

import ast
import importlib.util
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGES = ("agent", "tools", "hermes_cli", "gateway", "tui_gateway", "cron", "acp_adapter", "plugins", "providers")


def _root_py_modules() -> set[str]:
    spec = importlib.util.spec_from_file_location("_hermes_setup_py", REPO_ROOT / "setup.py")
    mod = importlib.util.module_from_spec(spec)
    saved = sys.argv
    sys.argv = ["setup.py", "--name"]  # setup() must not try to build anything on import
    try:
        try:
            spec.loader.exec_module(mod)
        except SystemExit:
            pass
    finally:
        sys.argv = saved
    return set(mod._root_py_modules())


def _imported_root_names(paths) -> dict[str, set[str]]:
    root_files = {p.stem for p in REPO_ROOT.glob("*.py")}
    hits: dict[str, set[str]] = {}
    for path in paths:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except (SyntaxError, OSError):  # sibling tests drop scratch root modules mid-run
            continue
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module.split(".")[0]]
            for n in names:
                if n in root_files:
                    hits.setdefault(str(path.relative_to(REPO_ROOT)), set()).add(n)
    return hits


def test_pyproject_has_no_static_py_modules_list():
    cfg = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "py-modules" not in cfg["tool"]["setuptools"], (
        "root modules are derived in setup.py::_root_py_modules(); a static py-modules list drifts "
        "from the tree and breaks installed wheels. Do not add it back."
    )


def test_every_root_module_imported_by_packaged_code_is_shipped():
    # Scratch modules other tests write at the repo root (``_test_*.py``) are not packaged.
    shipped = {n for n in _root_py_modules() if not n.startswith("_test_")}
    paths = [p for n in shipped if (p := REPO_ROOT / f"{n}.py").is_file()]
    for pkg in PACKAGES:
        paths.extend((REPO_ROOT / pkg).rglob("*.py"))
    missing = {f: sorted(n for n in names if n not in shipped) for f, names in _imported_root_names(paths).items()}
    missing = {f: v for f, v in missing.items() if v}
    assert not missing, f"packaged code imports root modules the wheel would not ship: {missing}"
    assert "hermes_state" in shipped and "setup" not in shipped
