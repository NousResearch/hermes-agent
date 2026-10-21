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


def validate_environment(python: Path | list[str], *, env: dict, cwd: Path) -> None:
    """Run startup import checks in the candidate, never the repairing process.

    *python* is the candidate's interpreter or its argv prefix (``venv_command``)."""
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
    command = [str(python), "-I"] if isinstance(python, (str, Path)) else [*python]
    result = subprocess.run([*command, "-c", script], cwd=cwd, env=env,
                            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60, check=False)
    if result.returncode:
        raise InstallError("venv", f"startup validation failed: {result.stderr.strip()[-1000:]}")


def repair_dependencies(project_root: Path) -> None:
    """Restore this installation's recorded set; never repair a foreign tree."""
    from hermes_cli.venv_sync import collect_superseded_generations
    from pm.client import sync_venv
    from pm.paths import repo_root

    if Path(project_root).resolve() != repo_root().resolve():
        raise InstallError("venv", "recovery root does not match this PM installation")
    with contextlib.redirect_stdout(sys.stderr):
        sync_venv(repair=True)
    collect_superseded_generations(project_root)


def _baseline_gaps(root: Path) -> list[str]:
    """Extras the store's shipped selection carries that the recorded selection lacks.

    A volume whose first generation was built before the image recorded its selection holds
    only what was installed on use (``["fal"]``, or ``[]``), and that generation replaces the
    image's environment with one missing everything the image ships.
    """
    from pm.environments import runtime_facts_path
    from pm.install import _extra_key, _facts, _still_declared
    from pm.lock import Facts
    from pm.packages import Venv

    fact = Facts(runtime_facts_path(root), strict=True).get("venv")
    if fact is None:
        return []
    # The same checks venv_is_current applies; a gap must not skip them.
    recorded = fact.get("extras") if isinstance(fact, dict) else None
    stamp = fact.get("stamp") if isinstance(fact, dict) else None
    if (not isinstance(recorded, list) or any(not isinstance(extra, str) for extra in recorded)
            or not isinstance(stamp, str) or not stamp):
        raise ValueError("invalid recorded dependency state")
    shipped = (_facts().get("venv") or {}).get("extras") or []
    have = {_extra_key(extra) for extra in recorded}
    return sorted({extra for extra in _still_declared(Venv(root), shipped) if _extra_key(extra) not in have})


def refresh_dependencies(project_root: Path) -> str:
    """Re-resolve the durable selection against inputs an external update replaced.

    A container image swaps the code and lock under a selection recorded on the data volume; a
    generation resolved against the previous lock must never boot the new code. Rebuilds the
    recorded extras and plugins, or on failure boots the image's own environment while keeping
    them recorded, so the next boot or install rebuilds them. A recorded selection missing
    extras the image ships gets them back. Returns what happened.
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
    missing = _baseline_gaps(root)
    if not missing and venv_is_current(project_root=root):
        return "current"
    try:
        with contextlib.redirect_stdout(sys.stderr):
            sync_venv(missing or None, explicit=True)
        return f"restored {', '.join(missing)}" if missing else "rebuilt"
    except Exception as exc:
        print(f"dependency refresh failed: {exc}", file=sys.stderr)
    with runtime_lock(root, timeout=None):
        facts = Facts(runtime_facts_path(root), strict=True)
        fact = facts.get("venv") or {}
        # Keep the restored extras recorded too: a lazy install before the next boot
        # unions onto this selection and must not publish a generation without them.
        recorded = list(fact.get("extras") or [])
        facts.record_state("venv", fact.get("stamp") or "stale",
                           recorded + [extra for extra in missing if extra not in recorded])
    return "fallback"
