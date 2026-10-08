"""Restore dependency generations without importing the damaged environment."""
from __future__ import annotations

import contextlib
import os
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from pm.package import InstallError


def record_boot_dependency_failure(home: Path, error: Exception, *, fallback: bool = False) -> None:
    """Persist only a fixed diagnostic category, never untrusted resolver output."""
    detail = str(error).lower()
    if "incompatible" in detail or "requires_hermes" in detail:
        category = "plugin-incompatible"
    elif "no solution found" in detail or "conflict" in detail or "unsatisfiable" in detail:
        category = "resolver-conflict"
    elif "permission denied" in detail or "no space left" in detail:
        category = "volume-permission-or-space"
    elif any(word in detail for word in ("connect", "network", "dns", "offline", "index")):
        category = "index-or-network"
    else:
        category = "unclassified"
    try:
        logs = home / "logs"
        logs.mkdir(mode=0o700, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(logs / "dependency-refresh.log", flags, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            if not stat.S_ISREG(os.fstat(output.fileno()).st_mode):
                return
            os.fchmod(output.fileno(), 0o600)
            timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
            state = "fallback" if fallback else "boot-refused"
            output.write(f"{timestamp} state={state} category={category}\n")
    except OSError:
        # A diagnostic must never turn an optional extras fallback into a hard failure.
        pass


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
                            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
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


def refresh_dependencies(project_root: Path) -> str:
    """Re-resolve the durable selection against inputs an external update replaced.

    A container image swaps the code and lock under a selection recorded on the data volume; a
    generation resolved against the previous lock must never boot the new code. Pre-generation
    images also leave selected plugin declarations in the volume without a PM facts file. Build
    those through the same atomic PM transaction before boot, never mutate the sealed image venv.
    A selected plugin's failed build must stop startup rather than silently boot without its tools.
    Extras-only failures retain the image-environment fallback for the next boot.
    """
    from hermes_cli.runtime_state import runtime_lock
    from pm.client import sync_venv
    from pm.environments import runtime_facts_path
    from pm.install import venv_is_current
    from pm.lock import Facts
    from pm.paths import repo_root
    from pm.plugin_declarations import read_python_declaration
    from pm.workspace import enabled_plugin_entries, enabled_member_dirs

    root = Path(project_root).resolve()
    if root != repo_root().resolve():
        raise InstallError("venv", "refresh root does not match this PM installation")
    # Unlike update eviction, Docker boot must preserve every profile's active selection.
    # Fail on malformed secondary config rather than silently publish a smaller union.
    selected_plugins = []
    for _plugins_dir, name, plugin_dir in enabled_plugin_entries(skip_invalid_secondary=False):
        declaration = read_python_declaration(plugin_dir)
        if declaration.is_member:
            from hermes_cli.plugins_manifest import requires_hermes_error
            if reason := requires_hermes_error(declaration.manifest):
                raise InstallError("venv", f"selected plugin {name!r} is incompatible: {reason}")
            selected_plugins.append(plugin_dir)
    # Validate manifest contracts before an image with no facts is allowed to boot its base venv.
    enabled_member_dirs()
    if not runtime_facts_path(root).is_file() and not selected_plugins:
        return "base"
    if venv_is_current(project_root=root):
        return "current"
    try:
        with contextlib.redirect_stdout(sys.stderr):
            sync_venv(explicit=True)
        return "rebuilt"
    except Exception as exc:
        if selected_plugins:
            raise
        from hermes_constants import get_hermes_home
        record_boot_dependency_failure(get_hermes_home(), exc, fallback=True)
    with runtime_lock(root, timeout=None):
        facts = Facts(runtime_facts_path(root), strict=True)
        fact = facts.get("venv") or {}
        facts.record_state("venv", fact.get("stamp") or "stale", list(fact.get("extras") or []))
    return "fallback"
