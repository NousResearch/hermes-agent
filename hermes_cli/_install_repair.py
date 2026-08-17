"""Dependency install execution shared between early recovery and full recovery.

Both callers need to run the same core ``.[all]`` reinstall:

- ``hermes_cli._early_recovery.recover_if_needed`` — stdlib-only, runs BEFORE
  ``hermes_cli.main``'s third-party imports, so it can complete a pending
  update while no native extension is mapped yet (#83569).
- ``hermes_cli.main._recover_core_update_marker_locked`` — the historical
  post-import recovery path. Kept as a fallback for installs the early pass
  could not complete (marker left in place on failure).

This module is deliberately **stdlib-only** so importing it can never fail in
the corrupted-venv state it exists to repair. ``hermes_cli.main`` imports
``managed_uv``, ``hermes_constants``, and friends only in its late path; the
early path must not. Where the late path uses ``managed_uv.ensure_uv`` to
bootstrap uv if missing, the early path uses the stdlib
:func:`hermes_cli._early_recovery._find_uv_binary` lookup and falls back to
plain pip when uv is absent — a degraded but working installer (the late
recovery will bootstrap uv on the next launch if it ever matters).
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Single source of truth for the recovery-lock lifecycle and uv lookup —
# _early_recovery already owns both, and importing it is free (stdlib-only).
from hermes_cli import _early_recovery as _er


def _is_windows() -> bool:
    return sys.platform == "win32"


def _is_termux_env(env: dict | None = None) -> bool:
    """Stdlib Termux probe (hermes_cli.main's version lives behind imports)."""
    env = env if env is not None else os.environ
    try:
        if env.get("TERMUX_VERSION"):
            return True
        prefix = env.get("PREFIX", "")
        return "com.termux" in prefix
    except Exception:
        return False


@contextlib.contextmanager
def _stdout_to_stderr():
    """Route fd 1 (and sys.stdout) to stderr for the duration of an install.

    ``hermes acp`` speaks JSON-RPC on stdout; an inherited-fd install child
    writing there would corrupt the protocol. Mirrors
    ``main.py::_recover_from_interrupted_install``.
    """
    saved_fd = None
    saved_sys_stdout = sys.stdout
    try:
        saved_fd = os.dup(1)
        os.dup2(2, 1)
    except OSError:
        saved_fd = None
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = saved_sys_stdout
        if saved_fd is not None:
            try:
                os.dup2(saved_fd, 1)
            except OSError:
                pass
            try:
                os.close(saved_fd)
            except OSError:
                pass


def _resolve_install_target(root: Path) -> tuple[list[str], dict | None]:
    """(install_cmd_prefix, env) for the project venv — stdlib uv lookup.

    Mirrors ``main.py::_default_venv_install_target`` but without
    ``managed_uv``. ``VIRTUAL_ENV`` steers ``uv pip`` at the project venv even
    when invoked from the base interpreter (the early-recovery case).
    Termux strips leaked interpreter-path env vars so uv resolves the venv
    correctly.
    """
    uv_bin = _er._find_uv_binary()
    if uv_bin:
        env = {**os.environ, "VIRTUAL_ENV": str(root / "venv")}
        if _is_termux_env(env):
            env.pop("PYTHONPATH", None)
            env.pop("PYTHONHOME", None)
        return [uv_bin, "pip"], env
    return [sys.executable, "-m", "pip"], None


def _venv_scripts_dir(root: Path) -> Path | None:
    """Project venv Scripts/bin dir, when present. stdlib-only."""
    venv_dir = root / "venv"
    if not venv_dir.is_dir():
        return None
    # hermes_constants is stdlib-only, so the canonical layout helper is safe
    # to use from this corrupted-venv repair path (#76105: never open-code
    # the Scripts/bin split).
    from hermes_constants import venv_bin_dir

    scripts = venv_bin_dir(venv_dir, windows=_is_windows())
    return scripts if scripts.is_dir() else None


def _load_console_script_names(root: Path) -> list[str]:
    """``[project.scripts]`` names from pyproject.toml (tomllib, 3.11+)."""
    try:
        import tomllib
    except ImportError:  # pragma: no cover
        return []
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file():
        return []
    try:
        with open(pyproject, "rb") as f:
            data = tomllib.load(f)
        scripts = data.get("project", {}).get("scripts", {}) or {}
        return [str(name) for name in scripts if name]
    except Exception:
        return []


def _quarantine_running_hermes_exe(scripts_dir: Path) -> list[tuple[Path, Path]]:
    """Rename live hermes*.exe shims aside so the installer can rewrite them.

    Windows blocks REPLACE on a running .exe but allows RENAME. Best-effort:
    silently skips anything that cannot be renamed. Returns (original,
    quarantined) pairs. stdlib-only — the console-script set comes from
    pyproject ``[project.scripts]`` (fallback: the well-known trio).
    """
    if not _is_windows():
        return []
    names = set(_load_console_script_names(scripts_dir.parent.parent)) or {
        "hermes",
        "hermes-agent",
        "hermes-acp",
    }
    names.add("hermes-gateway")
    moved: list[tuple[Path, Path]] = []
    for name in sorted(names):
        shim = scripts_dir / f"{name}.exe"
        if not shim.exists():
            continue
        quarantined = shim.with_name(f"{name}.exe.old.{int(time.time() * 1000)}")
        try:
            os.rename(shim, quarantined)
            moved.append((shim, quarantined))
        except OSError:
            pass
    return moved


def _restore_quarantined_exes(moved: list[tuple[Path, Path]]) -> None:
    """Put quarantined shims back when the installer did not replace them."""
    for original, quarantined in moved:
        if original.exists():
            continue  # installer wrote a fresh shim — the .old one is garbage
        try:
            os.rename(quarantined, original)
        except OSError:
            pass


def _run_install_cmd(cmd: list[str], *, env: dict | None, root: Path) -> None:
    """Run an install command with quarantine protection for venv shims.

    Raises CalledProcessError on install failure (callers implement the
    per-extra fallback ladder).
    """
    scripts_dir = _venv_scripts_dir(root) if _is_windows() else None
    moved = _quarantine_running_hermes_exe(scripts_dir) if scripts_dir else []
    try:
        subprocess.run(cmd, cwd=root, check=True, env=env)
    finally:
        # Restore runs on success AND failure: a SUCCESSFUL install can still
        # skip the entry-points step entirely (uv audits an already-satisfied
        # editable install as a no-op and rewrites nothing), which would leave
        # the quarantined shims renamed aside and `hermes` gone from PATH
        # (#75584). _restore_quarantined_exes only renames back when the
        # installer did NOT write a fresh shim, so this is safe in both cases.
        if scripts_dir is not None:
            _restore_quarantined_exes(moved)


def _load_installable_optional_extras(root: Path, group: str) -> list[str]:
    """Optional extras referenced by a dependency group (all / termux-all)."""
    try:
        import tomllib

        with (root / "pyproject.toml").open("rb") as handle:
            project = tomllib.load(handle).get("project", {})
    except Exception:
        return []
    optional_deps = project.get("optional-dependencies", {})
    if not isinstance(optional_deps, dict):
        return []
    refs = optional_deps.get(group, [])
    referenced: list[str] = []
    for ref in refs:
        if "[" in ref and "]" in ref:
            name = ref.split("[", 1)[1].split("]", 1)[0]
            if name in optional_deps:
                referenced.append(name)
    return referenced


def run_core_install(root: Path) -> None:
    """Full core ``.[all]`` editable reinstall — the recovery install.

    Equal in behavior to the install half of
    ``main.py::_recover_core_update_marker_locked``:

    - bootstrap pip via ensurepip (a killed install can leave the venv with no
      pip module at all)
    - prefer ``uv pip`` with VIRTUAL_ENV pointed at the project venv; fall back
      to ``python -m pip`` when no uv binary is available
    - target ``.[all]`` (or ``.[termux-all]`` on Termux) with the per-extra
      fallback ladder when the combined extras resolve fails
    - quarantine live ``hermes*.exe`` shims on Windows so they can be replaced
    - route ALL install output to stderr (acp/JSON-RPC safety)
    - Termux strips leaked PYTHONPATH/PYTHONHOME from the uv env

    Raises ``subprocess.CalledProcessError`` when even the base install fails;
    callers own marker lifecycle (clear on success, keep on failure).
    """
    prefix, env = _resolve_install_target(root)
    group = "termux-all" if _is_termux_env(env) else "all"

    with _stdout_to_stderr():
        try:
            subprocess.run(
                [sys.executable, "-m", "ensurepip", "--upgrade", "--default-pip"],
                cwd=root,
                capture_output=True,
            )
        except Exception:
            pass

        try:
            _run_install_cmd(
                prefix + ["install", "-e", f".[{group}]"], env=env, root=root
            )
            return
        except subprocess.CalledProcessError:
            print(
                "  ⚠ Optional extras failed, reinstalling base dependencies "
                "and retrying extras individually..."
            )

        _run_install_cmd(prefix + ["install", "-e", "."], env=env, root=root)

        failed_extras: list[str] = []
        installed_extras: list[str] = []
        for extra in _load_installable_optional_extras(root, group):
            try:
                _run_install_cmd(
                    prefix + ["install", "-e", f".[{extra}]"], env=env, root=root
                )
                installed_extras.append(extra)
            except subprocess.CalledProcessError:
                failed_extras.append(extra)
        if installed_extras:
            print(
                "  ✓ Reinstalled optional extras individually: "
                + ", ".join(installed_extras)
            )
        if failed_extras:
            print(
                "  ⚠ Skipped optional extras that still failed: "
                + ", ".join(failed_extras)
            )


# ---------------------------------------------------------------------------
# Marker metadata (attempt counter for early-pass retry backoff)
# ---------------------------------------------------------------------------


def bump_marker_attempts(marker_path: Path) -> int:
    """Increment an attempts counter stored inside the marker file.

    The marker's existence is the signal; opportunistic JSON body carries the
    retry count so a persistently failing install can back off instead of
    reinstall-hammering every launch. Corrupt/missing bodies restart at 1.
    Returns the new attempt count. Never raises.
    """
    attempts = 0
    try:
        raw = marker_path.read_text(encoding="utf-8", errors="replace").strip()
        if raw:
            try:
                attempts = int(json.loads(raw).get("attempts", 0))
            except (ValueError, AttributeError):
                attempts = 0
    except OSError:
        attempts = 0
    attempts += 1
    try:
        marker_path.write_text(json.dumps({"attempts": attempts}), encoding="utf-8")
    except OSError:
        pass
    return attempts
