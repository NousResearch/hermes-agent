"""Detect when the gateway is running stale code after a hot ``git pull``.

The gateway is a single long-lived process; its ``sys.modules`` is frozen at
boot. If the checkout is updated underneath it (a manual ``git pull``, or the
window before ``hermes update``'s graceful restart fires), a first-time lazy
import on a new code path can resolve a freshly-pulled consumer module against a
stale cached dependency -> ImportError (see
``tests/test_stale_utils_module_import.py`` for the exact failure).

We snapshot the checkout revision at gateway startup and compare on demand, so
risky callers (e.g. ``/model`` switching) can refuse with a clear "restart the
gateway" message instead of crashing on a cryptic import error.

If the revision can't be read (non-git install, IO error), the boot snapshot
stays ``None`` and skew detection no-ops — it never produces a false positive.

**Precision (2026-07-03):** the crash this guards against is an ImportError on a
first-time lazy import of a *Python module*. A checkout can advance without any
importable Python changing — a docs-only, locale-only, skill-only, YAML-only, or
test-only deploy moves the SHA but cannot cause a stale-module import crash.

**In-process import oracle (2026-07-07):** the 2026-07-03 tree-shape heuristic
(``tests/``+``docs/`` suppression) still refused on *any* other ``*.py`` diff —
including files the running gateway never imports (an unloaded provider, an
``eval/`` harness, a ``scripts/`` tool, another profile's plugin). That is the
residual false-refuse. The process already knows exactly which files it loaded:
``sys.modules`` is the ground truth, and every loaded first-party module has a
real ``__file__`` under the checkout root. So we now refuse a ``/model`` switch
only when a changed ``*.py`` is **either** already imported in-process **or** is
an in-tree submodule of an already-loaded package (the conservative lazy-import
ring — a loaded package can lazily import a sibling on a new code path). A
changed ``*.py`` that is neither cannot produce the stale-import ImportError the
guard names, so it no longer refuses.

**Fail-safe ladder (a broken oracle must REFUSE, never wave through):**
  1. in-process ``sys.modules`` oracle — refuse iff a changed ``*.py`` is loaded
     or is a submodule of a loaded package;
  2. if the oracle can't be computed (introspection error, or no first-party
     module is loaded — anomalous, don't trust it), fall back to the 2026-07-03
     ``tests/``+``docs/`` tree-shape heuristic;
  3. if even the file-level diff can't be computed, fall back to "any drift
     refuses".
Each rung is strictly more conservative — a needless refuse is a minor
annoyance, a missed skew is a crash — so the floor stays "refuse".
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_boot_fingerprint: str | None = None

# Path prefixes whose ``*.py`` files are NOT imported by the running gateway, so
# a change confined to them cannot cause a stale-module import crash. This is
# the *fallback* heuristic (rung 2) used only when the in-process import oracle
# can't be computed. Kept deliberately small and conservative — anything not
# listed here that is a ``*.py`` file is treated as runtime code (fail toward
# refusing).
_NON_RUNTIME_PY_PREFIXES: tuple[str, ...] = ("tests/", "docs/")


def _fingerprint() -> str | None:
    """Current checkout fingerprint, reusing the CLI's git-rev reader.

    ``hermes_cli.main`` is always already imported in a gateway process (it's
    the entry point), so this import is free and avoids duplicating the
    worktree-aware ref resolution.
    """
    try:
        from hermes_cli.main import _read_git_revision_fingerprint

        return _read_git_revision_fingerprint(_PROJECT_ROOT)
    except Exception:
        return None


def record_boot_fingerprint() -> None:
    """Snapshot the checkout revision at gateway startup (idempotent)."""
    global _boot_fingerprint
    if _boot_fingerprint is None:
        _boot_fingerprint = _fingerprint()


def _short(fingerprint: str) -> str:
    """Render a ``git:<ref>:<sha>`` fingerprint as a compact label."""
    sha = fingerprint.rsplit(":", 1)[-1]
    if sha and sha != "unresolved" and len(sha) > 10:
        return sha[:10]
    return sha or fingerprint


def _sha(fingerprint: str) -> str | None:
    """Extract the raw commit sha from a ``git:<ref>:<sha>`` fingerprint.

    Returns ``None`` when the sha is missing or ``unresolved`` (a ref whose
    object we couldn't read) — those can't seed a ``git diff``.
    """
    sha = fingerprint.rsplit(":", 1)[-1].strip()
    if not sha or sha == "unresolved":
        return None
    return sha


def _rel_to_root(path_str: str) -> str | None:
    """Resolve an absolute ``__file__``/``__path__`` entry to a repo-relative
    POSIX path under ``_PROJECT_ROOT``, or ``None`` if it's outside the tree.

    The git diff emits repo-relative POSIX paths, so the oracle must speak the
    same dialect to compare.
    """
    if not path_str:
        return None
    try:
        rel = Path(path_str).resolve().relative_to(_PROJECT_ROOT)
    except (ValueError, OSError, RuntimeError):
        return None
    return rel.as_posix()


def _loaded_first_party_paths() -> tuple[frozenset[str], tuple[str, ...]] | None:
    """Introspect ``sys.modules`` for first-party files loaded in this process.

    Returns ``(loaded_files, package_dirs)`` where ``loaded_files`` is the set
    of repo-relative ``*.py`` paths whose module is currently imported, and
    ``package_dirs`` is the set of repo-relative directories that back an
    already-loaded *package* (so a not-yet-imported submodule of a loaded
    package can still be recognized as reachable — the conservative
    lazy-import ring).

    Returns ``None`` if introspection raises, or if **no** first-party module
    is loaded (anomalous for a live gateway — don't trust an empty oracle;
    fall back to the heuristic). This is the fail-safe boundary: a broken or
    empty oracle degrades to the more-conservative rung, never to "allow".
    """
    try:
        files: set[str] = set()
        pkg_dirs: set[str] = set()
        for mod in list(sys.modules.values()):
            if mod is None:
                continue
            f = getattr(mod, "__file__", None)
            if isinstance(f, str) and f.endswith(".py"):
                rel = _rel_to_root(f)
                if rel is not None:
                    files.add(rel)
            # A package's __path__ lists the directories that back it; a lazy
            # import of a sibling submodule resolves through these.
            p = getattr(mod, "__path__", None)
            if p is not None:
                try:
                    entries = list(p)
                except TypeError:
                    entries = []
                for entry in entries:
                    if isinstance(entry, str):
                        rel = _rel_to_root(entry)
                        if rel is not None:
                            pkg_dirs.add(rel)
    except Exception:
        return None
    if not files and not pkg_dirs:
        # No first-party module resolved under the checkout root — the oracle
        # has nothing to say. Treat as inconclusive so the caller falls back to
        # the conservative heuristic rather than waving the switch through.
        return None
    return frozenset(files), tuple(pkg_dirs)


def _is_runtime_python(path: str) -> bool:
    """True if ``path`` is a Python module the gateway could import at runtime.

    Fallback (rung 2) heuristic used only when the in-process import oracle is
    unavailable. Only ``*.py`` files count (a docs/locale/YAML/skill change
    can't cause a stale-import crash), and files under the test/docs trees are
    excluded (they are never imported by the running gateway).
    """
    if not path.endswith(".py"):
        return False
    return not path.startswith(_NON_RUNTIME_PY_PREFIXES)


def _changed_py_risks_stale_import(changed: list[str]) -> bool:
    """Decide whether any changed path can cause a stale-module import crash.

    Rung 1 (in-process oracle): refuse iff a changed ``*.py`` is already loaded
    in ``sys.modules`` or is a submodule of an already-loaded package. Rung 2
    (fallback, when the oracle is unavailable): the ``tests/``+``docs/``
    tree-shape heuristic. A non-``*.py`` diff never risks a stale import.
    """
    py_changed = [p for p in changed if p.endswith(".py")]
    if not py_changed:
        # No Python changed at all (docs/skill/locale/YAML/test-data only) —
        # cannot cause a stale-module import crash.
        return False

    oracle = _loaded_first_party_paths()
    if oracle is None:
        # Fail-safe rung 2: oracle unavailable -> tree-shape heuristic, which
        # itself fails toward refusing (any runtime .py refuses).
        return any(_is_runtime_python(p) for p in py_changed)

    loaded_files, pkg_dirs = oracle
    for p in py_changed:
        if p in loaded_files:
            return True  # a currently-imported module changed -> real crash risk
        # Submodule of a loaded package (conservative lazy-import ring).
        if any(p == d or p.startswith(d + "/") for d in pkg_dirs):
            return True
    # Every changed .py is a file this process never imported and cannot reach
    # as a lazy import of a loaded package -> no stale-import crash possible.
    return False


def _runtime_python_changed(boot_sha: str, disk_sha: str) -> bool | None:
    """Whether any changed file between two revisions risks a stale import.

    Returns ``True``/``False`` on a successful diff, or ``None`` if the diff
    could not be computed (git error, missing object) so the caller can fall
    back to the conservative "any drift is skew" behavior.
    """
    try:
        proc = subprocess.run(
            ["git", "-C", str(_PROJECT_ROOT), "diff", "--name-only",
             f"{boot_sha}..{disk_sha}"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    changed = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    # An empty diff between two distinct sha strings is unexpected (they compared
    # unequal upstream); treat "no files" as inconclusive so we stay conservative.
    if not changed:
        return None
    return _changed_py_risks_stale_import(changed)


def detect_code_skew() -> tuple[str, str] | None:
    """Return ``(boot_rev, disk_rev)`` short labels if the checkout drifted
    since boot *in a way that risks a stale-module import crash*, else ``None``.

    A SHA change alone is not enough: only a change to a Python module this
    process actually imports (or could lazily import as a submodule of a loaded
    package) can cause the ImportError this guards against. When the sha drifts
    we diff the two revisions and suppress the skew if no such module changed
    (docs/skill/locale/test/YAML-only deploys, or ``*.py`` files this gateway
    never loaded). If the file-level diff can't be computed we conservatively
    report the skew (a needless refuse beats a missed crash).
    """
    if _boot_fingerprint is None:
        return None
    current = _fingerprint()
    if current is None or current == _boot_fingerprint:
        return None

    boot_sha = _sha(_boot_fingerprint)
    disk_sha = _sha(current)
    if boot_sha is not None and disk_sha is not None:
        if boot_sha == disk_sha:
            # Same commit, only the ref/branch label differs — no code changed,
            # so there's nothing to diff and no import risk. Don't refuse.
            return None
        runtime_changed = _runtime_python_changed(boot_sha, disk_sha)
        if runtime_changed is False:
            # Drift confined to files this process never imports (docs/skills/
            # tests/locale/YAML, or an unloaded provider/eval/scripts module).
            return None
        # runtime_changed is True (real risk) or None (couldn't tell) -> refuse.

    return _short(_boot_fingerprint), _short(current)
