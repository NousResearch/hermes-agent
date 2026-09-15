"""Bake the payload's bytecode cache at staging time.

A sealed payload ships no __pycache__: the runtime redirects bytecode
writes to a user-level cache (signature-breaking on macOS, read-only
mount on AppImage/MSIX), so every fresh install pays a cold-compile
stall on first launch. Baking turns that into a warm read:

* compileall runs with the payload's OWN staged interpreter (a pyc
  minted by any other interpreter is silently ignored — the magic tag
  is part of the cache filename);
* ``--invalidation-mode unchecked-hash``: the pyc is trusted without
  source validation and never rewritten, so staging-repack mtimes
  cannot invalidate it and a stale source cannot trigger a rewrite
  into the sealed tree;
* the baked pycs are chmodded read-only BEFORE packaging, so no
  incidental write can touch them; the dirs stay writable because
  in-place rebuilds rmtree the tree, and asserted coverage means no
  cache-miss write can target them;
* the runtime's sys.pycache_prefix redirect is dropped when the baked
  marker is present (the prefix relocates READS too — see
  scripts/build/launcher_wrapper.py), so imports read the source-
  adjacent baked pycs, which is Python's default multi-root lookup:
  payload modules read theirs; plugin/user modules keep caching beside
  their own sources under HERMES_HOME.

Coverage is the performance contract: every parseable module under
the baked roots must have a pyc. Unparseable fixtures (deliberately
invalid test data) are counted and skipped, not failed.
"""

import os
import py_compile
import subprocess
from pathlib import Path

MARKER = ".hermes-baked-pycache"


def _import_roots(root: Path) -> list[Path]:
    """The payload's import roots, in launcher order (repo, venv site, PM)."""
    roots = [root / "hermes-agent"]
    for source_root in roots + [root / "venv", root / "pm-runtime"]:
        if not source_root.is_dir():
            raise FileNotFoundError(f"missing import root: {source_root}")
    sites = sorted((root / "venv").glob("lib/python*/site-packages")) + \
        sorted((root / "venv").glob("Lib/site-packages"))
    if len(sites) != 1:
        raise ValueError(f"expected exactly one staged venv site-packages, found {len(sites)}")
    roots.append(sites[0])
    pm_sites = sorted((root / "pm-runtime").glob("lib/python*/site-packages")) + \
        sorted((root / "pm-runtime").glob("Lib/site-packages"))
    if len(pm_sites) != 1:
        raise ValueError(f"expected exactly one PM runtime site-packages, found {len(pm_sites)}")
    roots.append(pm_sites[0])
    return roots


def _uncovered(root: Path) -> tuple[list[Path], int]:
    """Parseable .py files without a pyc, and unparseable fixtures skipped."""
    from importlib.util import cache_from_source
    missing: list[Path] = []
    unparseable = 0
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        if Path(cache_from_source(path)).exists():
            continue
        try:
            compile(path.read_bytes(), str(path), "exec")
        except SyntaxError:
            # Deliberately invalid test fixtures ship in the snapshot; they
            # can never be imported, so no pyc is expected.
            unparseable += 1
        else:
            missing.append(path)
    return missing, unparseable


def bake_bytecode(root: Path, python: Path) -> dict:
    """Compile every payload module with the staged interpreter and seal the
    caches read-only. ``root`` is the payload root; ``python`` MUST be the
    payload's own staged interpreter binary."""
    root = Path(root).resolve()
    python = Path(python).resolve()
    if not python.is_file():
        raise FileNotFoundError(f"staged interpreter is missing: {python}")
    total = 0
    for source_root in _import_roots(root):
        subprocess.run(
            [str(python), "-I", "-m", "compileall", "-q",
             "--invalidation-mode", "unchecked-hash", str(source_root)],
            check=True, timeout=30 * 60, stdin=subprocess.DEVNULL)
        missing, unparseable = _uncovered(source_root)
        if missing:
            sample = ", ".join(str(m.relative_to(root)) for m in missing[:5])
            raise ValueError(f"{len(missing)} payload modules have no bytecode pyc "
                             f"({sample}…): a fresh install would cold-compile them "
                             "every launch")
        total += sum(1 for _ in source_root.rglob("__pycache__/*.pyc"))
    # Pycs are read-only BEFORE packaging so a stale-source rewrite (which
    # unchecked-hash never triggers anyway) or any incidental write can never
    # touch them and the macOS signature never observes a pyc change. The
    # __pycache__ DIRS stay writable: in-place payload rebuilds rmtree the
    # tree (native._prepare_native), and complete asserted coverage plus
    # unchecked-hash means no cache-miss write can ever target these dirs.
    for source_root in _import_roots(root):
        for cache in source_root.rglob("__pycache__"):
            for pyc in cache.iterdir():
                if pyc.is_file():
                    pyc.chmod(0o444)
    (root / MARKER).write_text("unchecked-hash\n", encoding="utf-8")
    return {"modules": total, "marker": str(root / MARKER)}
