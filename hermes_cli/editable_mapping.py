"""Keep a PEP 660 editable install's module map in step with the tree.

The editable finder setuptools writes (``__editable___*_finder.py``) resolves
first-party top-level names from a ``MAPPING`` snapshotted when that editable
install was last written. ``hermes update`` pulls new files into the tree, so a
pull that adds a top-level module leaves the snapshot behind: the file is in the
tree and unimportable through the finder, and anything importing it fails with
it — including mapped modules that import it (``hermes_state`` importing
``hermes_state_pidns``, instance) (#134413).

``hermes_bootstrap.harden_import_path()`` masks this on the shipped entry points
by putting the repo root on ``sys.path``; processes that import Hermes modules
through the venv interpreter without bootstrap see it directly.

Stdlib only, no import of the tree being inspected.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

__all__ = ["expected_tops", "finder_path", "mapped_tops", "refresh_mapping", "stale_tops"]


def finder_path(root: Path) -> Path | None:
    """The editable finder for an install rooted at *root*, if it has one."""
    root = Path(root)
    for lib in (root / "venv" / "lib", root / "venv" / "Lib"):
        if not lib.is_dir():
            continue
        sites = sorted(lib.glob("python*/site-packages")) or [lib / "site-packages"]
        for site in sites:
            for cand in site.glob("__editable___*_finder.py"):
                return cand
    return None


def expected_tops(root: Path) -> set[str]:
    """Top-level names the packaging config says are shipped.

    Mirrors the two authorities: ``setup.py::_root_py_modules()`` (every root
    ``.py`` except ``setup.py``) and ``[tool.setuptools.packages.find]``'s
    ``include`` list. Not "any directory with an ``__init__.py``": that counts
    ``tests/`` and ``evals/``, which are not distributed tops.
    """
    root = Path(root)
    tops = {p.stem for p in root.glob("*.py") if p.name != "setup.py"}
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        text = pyproject.read_text(encoding="utf-8-sig", errors="replace")
        table = re.search(r"\[tool\.setuptools\.packages\.find\](.*?)(?=\n\[)", text, re.S)
        if table:
            include = re.search(r"include\s*=\s*\[(.*?)\]", table.group(1), re.S)
            if include:
                listed = re.findall(r'"([^"]+)"', include.group(1))
                listed += re.findall(r"'([^']+)'", include.group(1))
                tops |= {v.split(".")[0] for v in listed}
    return tops


def mapped_tops(finder_src: str) -> set[str]:
    """Top-level names the finder already resolves."""
    tops: set[str] = set()
    mapping = re.search(
        r"MAPPING\s*(?::\s*dict\[str,\s*str\])?\s*=\s*\{(.*?)\}\s*\n", finder_src, re.S
    )
    if mapping:
        tops |= set(re.findall(r"'([A-Za-z_][A-Za-z0-9_]*)':", mapping.group(1)))
    namespaces = re.search(
        r"NAMESPACES\s*(?::\s*dict\[str,\s*list\[str\]\])?\s*=\s*\{(.*?)\}\s*\n",
        finder_src, re.S,
    )
    if namespaces:
        tops |= {
            key.split(".")[0]
            for key in re.findall(r"'([A-Za-z_][A-Za-z0-9_.\-]*)':", namespaces.group(1))
        }
    return tops


def stale_tops(root: Path) -> list[str]:
    """Shipped top-level names the finder cannot resolve. Empty means in step."""
    path = finder_path(root)
    if path is None:
        return []
    src = path.read_text(encoding="utf-8-sig", errors="replace")
    expected = {
        top for top in expected_tops(root)
        if (root / top).exists() or (root / f"{top}.py").exists()
    }
    return sorted(expected - mapped_tops(src))


def refresh_mapping(root: Path) -> list[str]:
    """Add the missing tops to the finder's ``MAPPING``; returns what was added.

    Rewrites only the ``MAPPING`` literal, so the file keeps the shape
    setuptools generated and the next editable reinstall overwrites it wholesale
    as before. Parses the result before returning so a malformed edit cannot
    leave a finder that breaks every import.
    """
    root = Path(root)
    path = finder_path(root)
    if path is None:
        return []
    missing = stale_tops(root)
    if not missing:
        return []
    src = path.read_text(encoding="utf-8-sig", errors="replace")
    literal = re.search(
        r"(MAPPING\s*(?::\s*dict\[str,\s*str\])?\s*=\s*\{)(.*?)(\}\s*\n)", src, re.S
    )
    if literal is None:
        return []
    body = literal.group(2).rstrip()
    if not body.endswith(","):
        body += ","
    body += " " + ", ".join(f"{top!r}: {str(root / top)!r}" for top in missing)
    patched = src[: literal.start(2)] + body + src[literal.end(2):]
    ast.parse(patched)
    path.write_text(patched, encoding="utf-8")
    return missing
