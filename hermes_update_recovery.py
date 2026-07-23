"""Stdlib-only update recovery that runs *before* ``hermes_cli.main`` imports.

``hermes_cli.main`` pulls in ``hermes_cli.env_loader`` → ``dotenv`` at module
import time. If a failed lazy refresh emptied ``python-dotenv`` (#57828), the
console entry would crash before ``main()`` could call
``_recover_from_interrupted_install()``.

This module uses only the standard library (+ subprocess to ``uv``/``pip``) so
marker-driven recovery can heal probed packages — including ``dotenv`` —
before the heavy CLI import graph loads (#58004 review).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

UPDATE_MARKER = ".update-incomplete"
LAZY_MARKER = ".lazy-refresh-incomplete"
LOCK_NAME = ".update-incomplete.lock"

# (import_name, attr, pip_distribution_name)
IMPORT_PROBES: tuple[tuple[str, str, str], ...] = (
    ("yaml", "SafeDumper", "PyYAML"),
    ("dotenv", "load_dotenv", "python-dotenv"),
    ("click", "Command", "click"),
    ("certifi", "contents", "certifi"),
    ("rich", "print", "rich"),
    ("cryptography", "__version__", "cryptography"),
    ("jwt", "encode", "PyJWT"),
)


def update_marker_path(root: Path | None = None) -> Path:
    return (root or PROJECT_ROOT) / UPDATE_MARKER


def lazy_refresh_marker_path(root: Path | None = None) -> Path:
    return (root or PROJECT_ROOT) / LAZY_MARKER


def markers_present(root: Path | None = None) -> bool:
    base = root or PROJECT_ROOT
    return update_marker_path(base).exists() or lazy_refresh_marker_path(base).exists()


def repair_specs(packages: list[str], root: Path | None = None) -> list[str]:
    """Map distribution names to pinned specs from ``pyproject.toml``."""
    if not packages:
        return []
    pyproject = (root or PROJECT_ROOT) / "pyproject.toml"
    if not pyproject.is_file():
        return list(packages)
    try:
        import tomllib
    except ImportError:  # pragma: no cover
        return list(packages)
    try:
        with open(pyproject, "rb") as f:
            raw_deps = tomllib.load(f).get("project", {}).get("dependencies", []) or []
    except Exception:
        return list(packages)

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


def _clear_marker(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


def _write_marker(path: Path) -> None:
    try:
        path.write_text(f"started={time.time()}\npid={os.getpid()}\n", encoding="utf-8")
    except OSError:
        pass


def _venv_python(root: Path) -> Path | None:
    scripts = root / "venv" / ("Scripts" if sys.platform == "win32" else "bin")
    candidate = scripts / ("python.exe" if sys.platform == "win32" else "python")
    return candidate if candidate.is_file() else None


def _find_uv() -> str | None:
    found = shutil.which("uv")
    if found:
        return found
    # Managed uv used by hermes_cli.managed_uv — best-effort without importing it.
    home = Path.home()
    for candidate in (
        home / ".hermes" / "bin" / ("uv.exe" if sys.platform == "win32" else "uv"),
        home / ".local" / "bin" / ("uv.exe" if sys.platform == "win32" else "uv"),
    ):
        if candidate.is_file():
            return str(candidate)
    return None


def _install_target(root: Path) -> tuple[list[str], dict[str, str] | None]:
    uv = _find_uv()
    venv = root / "venv"
    if uv and venv.is_dir():
        return [uv, "pip"], {**os.environ, "VIRTUAL_ENV": str(venv)}
    return [sys.executable, "-m", "pip"], None


def detect_broken_imports(
    root: Path | None = None,
    *,
    env: dict[str, str] | None = None,
) -> list[str] | None:
    """Return broken pip names, ``[]`` if healthy, or ``None`` if indeterminate."""
    base = root or PROJECT_ROOT
    python = _venv_python(base)
    if python is None:
        return None

    probe_lines = "\n".join(
        f"    ({mod!r}, {attr!r}, {pkg!r})," for mod, attr, pkg in IMPORT_PROBES
    )
    script = (
        "broken = []\n"
        "probes = [\n"
        f"{probe_lines}\n"
        "]\n"
        "for mod, attr, pkg in probes:\n"
        "    try:\n"
        "        imported = __import__(mod)\n"
        "        if not hasattr(imported, attr):\n"
        "            broken.append(pkg)\n"
        "    except Exception:\n"
        "        broken.append(pkg)\n"
        "print('\\n'.join(broken))\n"
    )
    try:
        result = subprocess.run(
            [str(python), "-c", script],
            capture_output=True,
            text=True,
            check=False,
            env=env,
            cwd=str(base),
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def force_reinstall_packages(
    packages: list[str],
    root: Path | None = None,
    *,
    env: dict[str, str] | None = None,
) -> bool:
    """Force-reinstall packages with pyproject pins. Never raises."""
    if not packages:
        return True
    base = root or PROJECT_ROOT
    prefix, default_env = _install_target(base)
    run_env = env if env is not None else default_env
    specs = repair_specs(packages, base)
    try:
        subprocess.run(
            prefix + ["install", "--force-reinstall", *specs],
            cwd=str(base),
            check=True,
            env=run_env,
        )
    except (subprocess.CalledProcessError, OSError):
        return False
    after = detect_broken_imports(base, env=run_env)
    return after == []


def _quarantine_hermes_exe(root: Path) -> None:
    """Best-effort rename of live Windows shims so pip can rewrite them."""
    if sys.platform != "win32":
        return
    scripts = root / "venv" / "Scripts"
    if not scripts.is_dir():
        return
    stamp = int(time.time() * 1000)
    for name in ("hermes.exe", "hermes-gateway.exe", "hermes-agent.exe", "hermes-acp.exe"):
        shim = scripts / name
        if not shim.is_file():
            continue
        dest = scripts / f"{name}.old.{stamp}"
        try:
            shim.rename(dest)
        except OSError:
            pass


def recover_lazy_marker(root: Path | None = None) -> str:
    """Heal ``.lazy-refresh-incomplete``. Returns status string."""
    base = root or PROJECT_ROOT
    marker = lazy_refresh_marker_path(base)
    if not marker.exists():
        return "absent"

    print(
        "⚠ A previous lazy-backend refresh may have left the venv unhealthy — "
        "running import-based package repair (pre-main bootstrap)...",
        file=sys.stderr,
    )
    prefix, env = _install_target(base)
    broken = detect_broken_imports(base, env=env)
    if broken is None:
        print(
            "  ⚠ Import probes unavailable — leaving `.lazy-refresh-incomplete`.",
            file=sys.stderr,
        )
        return "indeterminate"
    if not broken:
        _clear_marker(marker)
        print("✓ Lazy-refresh probes clean — cleared recovery marker.", file=sys.stderr)
        return "healthy"
    print(
        f"  → Bootstrap repairing: {', '.join(broken)}",
        file=sys.stderr,
    )
    if force_reinstall_packages(broken, base, env=env):
        _clear_marker(marker)
        print("✓ Lazy-refresh bootstrap repair confirmed.", file=sys.stderr)
        return "repaired"

    specs = " ".join(repair_specs(broken, base))
    print("  ⚠ Bootstrap lazy repair incomplete. Run manually:", file=sys.stderr)
    print(
        f"    {' '.join(prefix)} install --force-reinstall {specs}",
        file=sys.stderr,
    )
    return "failed"


def recover_core_marker(root: Path | None = None) -> str:
    """Heal ``.update-incomplete`` via full editable install. Returns status."""
    base = root or PROJECT_ROOT
    marker = update_marker_path(base)
    if not marker.exists():
        return "absent"

    print(
        "⚠ A previous `hermes update` was interrupted mid-install — "
        "finishing dependency installation (pre-main bootstrap)...",
        file=sys.stderr,
    )

    # First aid: heal probe packages (incl. dotenv) so a later main import works
    # even if the full reinstall is still needed. Does NOT clear the core marker.
    prefix, env = _install_target(base)
    broken = detect_broken_imports(base, env=env)
    if broken:
        print(
            f"  → Bootstrap first-aid reinstall: {', '.join(broken)}",
            file=sys.stderr,
        )
        force_reinstall_packages(broken, base, env=env)

    try:
        subprocess.run(
            [sys.executable, "-m", "ensurepip", "--upgrade", "--default-pip"],
            cwd=str(base),
            capture_output=True,
            check=False,
        )
    except Exception:
        pass

    _quarantine_hermes_exe(base)
    try:
        subprocess.run(
            prefix + ["install", "-e", ".[all]"],
            cwd=str(base),
            check=True,
            env=env,
        )
    except (subprocess.CalledProcessError, OSError) as exc:
        print(f"✗ Bootstrap core recovery failed: {exc}", file=sys.stderr)
        print("  Recover manually with:", file=sys.stderr)
        print(f"    cd {base}", file=sys.stderr)
        print(f"    {sys.executable} -m ensurepip --upgrade", file=sys.stderr)
        print(f"    {' '.join(prefix)} install -e '.[all]'", file=sys.stderr)
        return "failed"

    _clear_marker(marker)
    print("✓ Core dependency bootstrap recovery succeeded.", file=sys.stderr)
    return "repaired"


def maybe_recover(
    root: Path | None = None,
    *,
    argv: list[str] | None = None,
) -> bool:
    """Run marker recovery if needed. Returns True when any recovery ran.

    Skips when ``update`` appears in argv (same rule as ``hermes_cli.main``)
    so recovery never races the real update flow.
    """
    args = sys.argv[1:] if argv is None else argv
    if "update" in args:
        return False

    base = root or PROJECT_ROOT
    if not markers_present(base):
        return False

    if not (base / "pyproject.toml").is_file():
        _clear_marker(update_marker_path(base))
        _clear_marker(lazy_refresh_marker_path(base))
        return False

    lock_path = base / LOCK_NAME
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
        return False
    except OSError:
        pass

    try:
        if lazy_refresh_marker_path(base).exists():
            recover_lazy_marker(base)
        if update_marker_path(base).exists():
            recover_core_marker(base)
        return True
    finally:
        try:
            lock_path.unlink()
        except OSError:
            pass
