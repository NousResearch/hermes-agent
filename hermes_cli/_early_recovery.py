"""Dependency-light venv recovery that runs BEFORE hermes_cli.main's imports.

The ``hermes`` console entry point is ``hermes_cli.main:main``.  Importing
``hermes_cli.main`` pulls in third-party packages at module level (``dotenv``
via ``hermes_cli.env_loader``, ``yaml`` via ``hermes_cli.config``, ...).  In
the exact failure state the update-recovery markers exist for — a failed lazy
backend refresh or interrupted core install that wiped a core package's
import files (#57828) — a normal launch crashes *while importing main.py*,
before ``_recover_from_interrupted_install()`` can run.  The marker system is
unreachable precisely when it is needed most.

This module is deliberately **stdlib-only** so importing it can never fail on
a corrupted venv.  ``hermes_cli.main`` imports and calls
:func:`recover_if_needed` at the very top of its module body, before any
third-party import.

Scope: this early pass only repairs enough for ``hermes_cli.main`` to become
importable again (force-reinstall of the known-fragile core packages, using
the pins from pyproject.toml).  It NEVER clears the recovery markers — the
full, confirmed marker lifecycle stays with ``_recover_from_interrupted_install()``
in main.py, which runs right after import succeeds.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import time
from pathlib import Path

# Core packages a failed lazy ``uv pip install`` is known to leave with intact
# distribution metadata but wiped import files (#57828).  ``module`` is what we
# probe via a real import; ``attr`` guards against an empty/stub module.
# main.py's marker-recovery path reuses these tables — keep them here (the
# dependency-light module) so both layers probe and repair the same set.
LAZY_REFRESH_IMPORT_PROBES: tuple[tuple[str, str], ...] = (
    ("yaml", "SafeDumper"),
    ("dotenv", "load_dotenv"),
    ("click", "Command"),
    ("certifi", "contents"),
    ("rich", "print"),
    ("cryptography", "__version__"),
    ("jwt", "encode"),
)

LAZY_REFRESH_REPAIR_PACKAGES: dict[str, str] = {
    "yaml": "PyYAML",
    "dotenv": "python-dotenv",
    "click": "click",
    "certifi": "certifi",
    "rich": "rich",
    "cryptography": "cryptography",
    "jwt": "PyJWT",
}


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _pinned_specs(packages: list[str], project_root: Path) -> list[str]:
    """Map bare package names to their pinned specs from pyproject.toml.

    Stdlib-only (tomllib + naive requirement-head parsing — ``packaging`` may
    itself be broken in the failure state this module exists for).  Unknown
    packages fall back to their bare name.
    """
    pyproject = project_root / "pyproject.toml"
    if not pyproject.is_file():
        return packages
    try:
        import tomllib

        with open(pyproject, "rb") as f:
            raw_deps = tomllib.load(f).get("project", {}).get("dependencies", []) or []
    except Exception:
        return packages

    name_to_spec: dict[str, str] = {}
    for spec in raw_deps:
        head = spec.split(";", 1)[0].strip()
        bare = head
        for op in ("==", ">=", "<=", "~=", ">", "<", "!="):
            if op in bare:
                bare = bare.split(op, 1)[0]
                break
        key = bare.strip().split("[", 1)[0].strip().lower()
        if key:
            name_to_spec[key] = head
    return [name_to_spec.get(pkg.lower(), pkg) for pkg in packages]


def _probe_broken_packages() -> list[str]:
    """Import-probe the fragile core packages in THIS process.

    Returns repair package names (deduped, probe order) for modules that fail
    to import or lack their sentinel attribute.  Failed imports leave nothing
    in ``sys.modules``, so a post-repair retry in the same process works.
    """
    broken: list[str] = []
    for mod_name, attr in LAZY_REFRESH_IMPORT_PROBES:
        try:
            mod = importlib.import_module(mod_name)
            if not hasattr(mod, attr):
                raise ImportError(f"{mod_name} missing {attr}")
        except Exception:
            pkg = LAZY_REFRESH_REPAIR_PACKAGES.get(mod_name)
            if pkg and pkg not in broken:
                broken.append(pkg)
    return broken


def _run_repair_install(specs: list[str], project_root: Path) -> bool:
    """ensurepip + ``pip install --force-reinstall`` the given specs.

    Streams nothing to stdout (``hermes acp`` speaks JSON-RPC on stdout);
    output is captured and replayed to stderr only on failure.  Never raises.
    """
    try:
        subprocess.run(
            [sys.executable, "-m", "ensurepip", "--upgrade", "--default-pip"],
            cwd=project_root,
            capture_output=True,
        )
    except Exception:
        pass
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--force-reinstall", *specs],
            cwd=project_root,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        print(f"  ✗ Early venv repair could not run pip: {exc}", file=sys.stderr)
        return False
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "")[-2000:]
        if tail:
            print(tail, file=sys.stderr)
        return False
    return True


def recover_if_needed(
    project_root: Path | None = None,
    argv: list[str] | None = None,
) -> None:
    """Repair wiped core packages so ``hermes_cli.main`` can import at all.

    Fast path (no marker present) is two ``lstat`` calls.  Only acts when a
    recovery marker from a prior ``hermes update`` exists AND an import probe
    confirms a core package is actually broken.  Markers are intentionally
    NOT cleared here — ``_recover_from_interrupted_install()`` in main.py owns
    the confirmed marker lifecycle and runs immediately after import succeeds.

    Never raises: on any failure the import of main.py proceeds and surfaces
    the real error.
    """
    try:
        args = sys.argv[1:] if argv is None else argv
        # Same deliberately-loose match as main(): the real update flow writes
        # and clears its own markers — a recovery install must not race it.
        if "update" in args:
            return
        root = _project_root() if project_root is None else project_root
        core_marker = root / ".update-incomplete"
        lazy_marker = root / ".lazy-refresh-incomplete"
        if not core_marker.exists() and not lazy_marker.exists():
            return
        # Managed/Docker/PyPI installs have no source tree here — the marker
        # is not ours to act on; main.py's recovery clears it.
        if not (root / "pyproject.toml").is_file():
            return

        broken = _probe_broken_packages()
        if not broken:
            # Imports are fine — main.py will load and run full recovery.
            return

        # Single-flight: share main.py's recovery lock so an early repair
        # never races a concurrent full recovery into the same shared venv.
        lock_path = root / ".update-incomplete.lock"
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()}\n".encode())
            os.close(fd)
        except FileExistsError:
            try:
                if time.time() - lock_path.stat().st_mtime > 3600:
                    lock_path.unlink()
            except OSError:
                pass
            return
        except OSError:
            pass  # read-only fs / perms — proceed unlocked, install surfaces it

        try:
            specs = _pinned_specs(broken, root)
            print(
                "⚠ Core package(s) broken by an interrupted update — "
                f"repairing before launch: {', '.join(broken)}",
                file=sys.stderr,
            )
            if _run_repair_install(specs, root) and not _probe_broken_packages():
                print("  ✓ Core packages repaired.", file=sys.stderr)
            else:
                print(
                    "  ✗ Automatic repair incomplete. Recover manually with:",
                    file=sys.stderr,
                )
                print(
                    f"    {sys.executable} -m pip install --force-reinstall "
                    + " ".join(specs),
                    file=sys.stderr,
                )
        finally:
            try:
                lock_path.unlink()
            except OSError:
                pass
    except Exception:
        # Never block launch — the import of main.py will surface the truth.
        pass
