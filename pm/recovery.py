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
