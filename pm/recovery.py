"""Restore dependency generations without importing the damaged environment."""
from __future__ import annotations

import contextlib
import subprocess
import sys
from pathlib import Path

from pm.package import InstallError


STARTUP_IMPORTS = (
    ("ruamel.yaml", "ruamel.yaml", "YAML"),
    ("python-dotenv", "dotenv", "load_dotenv"),
    ("click", "click", "Command"),
    ("certifi", "certifi", "contents"),
    ("rich", "rich", "print"),
    ("cryptography", "cryptography.hazmat.bindings._rust", "openssl"),
    ("PyJWT", "jwt", "encode"),
)


def validate_environment(python: Path, *, env: dict, cwd: Path) -> None:
    """Run startup import checks in the candidate, never the repairing process."""
    script = (
        "import importlib, importlib.metadata, pathlib, re, tomllib\n"
        "project = tomllib.loads(pathlib.Path('pyproject.toml').read_text(encoding='utf-8-sig'))['project']\n"
        "required = {re.split(r'[\\[<>=!~; @]', dep, 1)[0].lower().replace('_', '-')\n"
        "            for dep in project.get('dependencies', [])}\n"
        f"checks = {STARTUP_IMPORTS!r}\n"
        "for distribution, module, attribute in checks:\n"
        "    try:\n"
        "        importlib.metadata.distribution(distribution)\n"
        "    except importlib.metadata.PackageNotFoundError:\n"
        "        if distribution.lower().replace('_', '-') in required:\n"
        "            raise\n"
        "        continue\n"
        "    loaded = importlib.import_module(module)\n"
        "    getattr(loaded, attribute)\n"
        "    if module == 'certifi':\n"
        "        bundle = pathlib.Path(loaded.where())\n"
        "        assert bundle.is_file() and bundle.stat().st_size >= 1024, 'CA bundle is missing'\n"
    )
    result = subprocess.run([str(python), "-I", "-c", script], cwd=cwd, env=env,
                            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    if result.returncode:
        raise InstallError("venv", f"startup validation failed: {result.stderr.strip()[-1000:]}")


def repair_dependencies(project_root: Path) -> None:
    """Restore this installation's recorded set; never repair a foreign tree."""
    from pm.client import sync_venv
    from pm.paths import repo_root

    if Path(project_root).resolve() != repo_root().resolve():
        raise InstallError("venv", "recovery root does not match this PM installation")
    with contextlib.redirect_stdout(sys.stderr):
        sync_venv(repair=True)


def refresh_dependencies(project_root: Path) -> str:
    """Re-resolve the durable selection against inputs an external update replaced.

    A container image swaps the code and lock under a selection recorded on the data volume; a
    generation resolved against the previous lock must never boot the new code. Rebuilds the
    recorded extras and plugins, or on failure boots the image's own environment while keeping
    them recorded, so the next boot or install rebuilds them. Returns what happened.
    """
    from hermes_cli.runtime_state import runtime_lock
    from pm.client import sync_venv
    from pm.environments import runtime_facts_path
    from pm.install import venv_is_current
    from pm.lock import Facts
    from pm.paths import repo_root

    root = Path(project_root).resolve()
    if root != repo_root().resolve():
        raise InstallError("venv", "refresh root does not match this PM installation")
    if not runtime_facts_path(root).is_file():
        return "base"
    if venv_is_current(project_root=root):
        return "current"
    try:
        with contextlib.redirect_stdout(sys.stderr):
            sync_venv(explicit=True)
        return "rebuilt"
    except Exception as exc:
        print(f"dependency refresh failed: {exc}", file=sys.stderr)
    with runtime_lock(root, timeout=None):
        facts = Facts(runtime_facts_path(root), strict=True)
        fact = facts.get("venv") or {}
        facts.record_state("venv", fact.get("stamp") or "stale", list(fact.get("extras") or []))
    return "fallback"
