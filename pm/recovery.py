"""Restore dependency generations without importing the damaged environment."""
from __future__ import annotations

import contextlib
import shutil
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

# What a rebuild takes beyond merging the image's uv cache onto the data volume: the generation
# hardlinks its packages from that cache, so this is its bytecode, the build's temporaries and
# the booted gateway's first writes. Below it the volume is too full to boot on at all.
REBUILD_HEADROOM_BYTES = 256 * 1024 ** 2


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
    from pm.install import _carried, _extra_key, _facts
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
    return sorted({extra for extra in _carried(Venv(root), shipped) if _extra_key(extra) not in have})


def _image_covers_selection(root: Path, missing: list[str]) -> bool:
    """Whether the image's own environment is exactly the selection a rebuild would reach.

    The target is the one ``sync_venv`` computes (recorded extras still declared, the
    restored baseline, the enabled plugin members), compared by stamp with the selection
    the image recorded for its sealed venv.
    """
    from pm.environments import runtime_facts_path
    from pm.install import _facts, _target_selection
    from pm.lock import Facts
    from pm.packages import Venv

    image = _facts().get("venv")
    if not isinstance(image, dict) or not image.get("stamp"):
        return False
    fact = Facts(runtime_facts_path(root), strict=True).get("venv") or {}
    _, stamp, _ = _target_selection(Venv(root), fact, extras=missing, inputs={}, repair=False,
                                    shipped=None, frozen=None)
    return stamp == image["stamp"]


def _select_image_environment(root: Path) -> None:
    """Drop the volume's selection so the image environment boots, and the uv cache only its builds used.

    The superseded generation is left to ``collect_generations``. The cache is merged back from
    the image on the next build that needs it.
    """
    from hermes_cli.runtime_state import runtime_lock
    from pm.environments import runtime_facts_path
    from pm.lock import Facts
    from pm.packages import uv_cache_path

    with runtime_lock(root, timeout=None):
        # Removed before the record is written: on a volume the cache already filled, that
        # write is what would fail.
        shutil.rmtree(uv_cache_path(), ignore_errors=True)
        Facts(runtime_facts_path(root), strict=True).forget(["venv"])


def _rebuild_shortfall() -> str:
    """Why the data volume cannot take a rebuild, or "" when it can.

    Measured, not guessed: the merge the build starts with is the bulk of it (hosted agents
    with 650-775 MB free filled their volume there), and it is near zero once this image's
    cache is merged.
    """
    from pm.environments import dependency_home_root
    from pm.packages import uv_cache_seed_bytes

    free = shutil.disk_usage(dependency_home_root()).free
    need = uv_cache_seed_bytes() + REBUILD_HEADROOM_BYTES
    return "" if free >= need else f"{free >> 20} MB free on the data volume, a rebuild needs {need >> 20} MB"


def _free_a_full_volume() -> None:
    """Delete the uv cache when the volume is too full to boot on, before the fallback record.

    Otherwise that record is the write that fails, and every boot fails the same way. Nothing
    reads the cache at boot, every generation holds its own links or copies of its files, and
    the next build merges the image's part back; a plugin's own wheels download again.
    """
    from pm.environments import dependency_home_root
    from pm.packages import uv_cache_path

    free = shutil.disk_usage(dependency_home_root()).free
    if free < REBUILD_HEADROOM_BYTES:
        print(f"{free >> 20} MB free on the data volume: removing the uv cache", file=sys.stderr)
        shutil.rmtree(uv_cache_path(), ignore_errors=True)


def refresh_dependencies(project_root: Path) -> str:
    """Re-resolve the durable selection against inputs an external update replaced.

    A container image swaps the code and lock under a selection recorded on the data volume; a
    generation resolved against the previous lock must never boot the new code. A selection the
    image environment already is boots that environment. Anything else is rebuilt, or, on
    failure or without room for the rebuild, the image's own environment boots while the
    selection stays recorded, so the next boot or install rebuilds it. A recorded selection
    missing extras the image ships gets them back. Returns what happened.
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
    # Everything past the recorded selection's own validity ends in the fallback when it raises
    # (plugin discovery reads every enabled manifest): escaping here would leave the previous
    # image's generation selected under the new code.
    try:
        if not missing and venv_is_current(project_root=root):
            return "current"
        if _image_covers_selection(root, missing):
            _select_image_environment(root)
            return "image"
        shortfall = _rebuild_shortfall()
        if shortfall:
            print(f"dependency refresh skipped: {shortfall}", file=sys.stderr)
        else:
            with contextlib.redirect_stdout(sys.stderr):
                # The image's baseline, not a user request: it never undoes a `--without`.
                sync_venv(missing or None, explicit=True, automatic=missing)
            return f"restored {', '.join(missing)}" if missing else "rebuilt"
    except Exception as exc:
        print(f"dependency refresh failed: {exc}", file=sys.stderr)
    with runtime_lock(root, timeout=None):
        _free_a_full_volume()
        facts = Facts(runtime_facts_path(root), strict=True)
        fact = facts.get("venv") or {}
        # Keep the restored extras recorded too: a lazy install before the next boot
        # unions onto this selection and must not publish a generation without them.
        recorded = list(fact.get("extras") or [])
        facts.record_state("venv", fact.get("stamp") or "stale",
                           recorded + [extra for extra in missing if extra not in recorded])
    return "fallback"
