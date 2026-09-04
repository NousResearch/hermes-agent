"""Windows launcher staging: `hermes` / `hermes-acp` launchers that boot the
pm STORE python with ``PYTHONPATH=<repo>;<venv>/site-packages`` — never
``venv\\Scripts\\python.exe`` (pm work item 3, "no boot through the venv";
``pyvenv.cfg`` is inert dead config).

Two consumers:

- ``scripts/install.ps1`` Stage-Path (bootstrap) calls
  :func:`ensure_install_launchers` through the venv python — install-time
  use of the venv interpreter is fine; it is the materializer, not a boot
  path. On a fresh install the pm store interpreter does not exist yet, so
  a runtime-resolving ``.cmd`` delegator is staged and
  ``hermes_cli._install_repair.ensure_windows_bin_launchers`` upgrades it
  to an exe once ``hermes pm install`` lands the store python.
- ``hermes_cli/_install_repair.py`` calls the same machinery per-name when
  a launcher is missing or still boots through the venv.

Store-root resolution mirrors ``pm/paths.py`` (HERMES_RUNTIME_DIR →
install-stamp.json ``runtimeDir`` → the pm default root); it is re-implemented
here because this module must stay importable from a bare interpreter
(stdlib + distlib only) during repair/bootstrap where ``pm`` imports are not
guaranteed. If pm's layout changes, keep the two in lockstep.

The exe form uses distlib's ScriptMaker with a customized script template
that inserts the repo root and the venv's site-packages into ``sys.path``
before importing the entry point — distlib is taken from pip's vendored
copy when the standalone package is absent (uv-synced venvs carry pip but
rarely standalone distlib). When distlib is unavailable, a ``.cmd``
delegator carrying the same PYTHONPATH composition is written instead.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

#: Launcher command names — keep in lockstep with scripts/install.ps1
#: Stage-Path and hermes_cli/_install_repair.py.
WINDOWS_BIN_LAUNCHERS = ("hermes", "hermes-acp")

#: command name -> (entry module, callable) — mirrors pyproject.toml
#: [project.scripts].
ENTRY_POINTS = {
    "hermes": ("hermes_cli.main", "main"),
    "hermes-acp": ("acp_adapter.entry", "main"),
}


def _is_windows() -> bool:
    return os.name == "nt"


def project_venv_dir(repo_root: Path) -> Path | None:
    """The uv-sync target venv of the install (``venv`` preferred, ``.venv``
    for dev checkouts)."""
    for name in ("venv", ".venv"):
        candidate = Path(repo_root) / name
        if candidate.is_dir():
            return candidate
    return None


def store_root(repo_root: Path) -> Path:
    """The pm byte store, resolved exactly like pm/paths.store_root()."""
    env = os.environ.get("HERMES_RUNTIME_DIR")
    if env:
        return Path(env)
    stamp = Path(repo_root) / "install-stamp.json"
    if stamp.is_file():
        try:
            runtime_dir = json.loads(stamp.read_text(encoding="utf-8-sig")).get(
                "runtimeDir"
            )
        except (OSError, ValueError):
            runtime_dir = None
        if runtime_dir:
            return Path(runtime_dir)
    from hermes_constants import get_default_hermes_root

    return Path(get_default_hermes_root()) / "tools"


def resolve_store_python(repo_root: Path) -> Path | None:
    """The interpreter the store installed from the ``python`` package
    (facts.json entry first, newest ``python-*`` entry as fallback), or
    None when `hermes pm install` has not materialized one yet."""
    runtime = store_root(repo_root)
    rel = "python.exe" if _is_windows() else "bin/python3"

    facts = runtime / "facts.json"
    if facts.is_file():
        try:
            packages = json.loads(facts.read_text(encoding="utf-8-sig")).get(
                "packages", {}
            )
            entry = (packages.get("python") or {}).get("entry")
        except (OSError, ValueError):
            entry = None
        if entry:
            candidate = runtime / entry / rel
            if candidate.is_file():
                return candidate

    for entry_dir in sorted(runtime.glob("python-*"), key=lambda p: p.name):
        candidate = entry_dir / rel
        if candidate.is_file():
            return candidate
    return None


def venv_site_packages(venv_dir: Path) -> Path | None:
    """The venv directory uv sync fills (Windows ``Lib\\site-packages``;
    POSIX ``lib/python3.X/site-packages``)."""
    venv_dir = Path(venv_dir)
    if _is_windows():
        candidate = venv_dir / "Lib" / "site-packages"
        return candidate if candidate.is_dir() else None
    for candidate in sorted(venv_dir.glob("lib/python3.*/site-packages")):
        if candidate.is_dir():
            return candidate
    return None


def _load_script_maker():
    """distlib's ScriptMaker — standalone first, then pip's vendored copy."""
    try:
        from distlib.scripts import ScriptMaker

        return ScriptMaker
    except ImportError:
        pass
    try:
        from pip._vendor.distlib.scripts import ScriptMaker

        return ScriptMaker
    except ImportError:
        return None


def exe_is_venv_bound(exe: Path, venv_dir: Path | None) -> bool:
    """True when an existing launcher exe embeds the venv interpreter —
    i.e. it is a copied venv console-script trampoline from the pre-pm
    installer, which boots through ``venv\\Scripts\\python.exe`` and must be
    replaced. distlib launchers append the interpreter shebang after the zip
    payload; both encodings are scanned to be safe."""
    if venv_dir is None:
        return False
    needles = set()
    for interpreter in (
        venv_dir / "Scripts" / "hermes.exe",
        venv_dir / "Scripts" / "hermes-acp.exe",
        venv_dir / "Scripts" / "python.exe",
        venv_dir / "bin" / "python3",
        venv_dir / "bin" / "hermes",
        venv_dir / "bin" / "hermes-acp",
    ):
        for enc in ("utf-8", "utf-16-le"):
            try:
                needles.add(str(interpreter).encode(enc))
            except Exception:
                continue
    try:
        data = Path(exe).read_bytes()
    except OSError:
        return False
    return any(needle in data for needle in needles)


def _write_atomic(target: Path, write) -> Path | None:
    """Stage under a pid-suffixed name then os.replace, so a concurrent
    process start never sees a torn launcher."""
    staging = target.with_name(f"{target.name}.stage.{os.getpid()}")
    try:
        write(staging)
        os.replace(staging, target)
        return target
    except OSError:
        try:
            staging.unlink()
        except OSError:
            pass
        return None


def mint_launcher(
    name: str,
    repo_root: Path,
    out_dir: Path,
    python_exe: Path,
    site_packages: Path | None,
) -> Path | None:
    """Write one launcher for *name* into out_dir. Returns the written path
    or None. Prefers a distlib exe trampoline whose embedded script inserts
    the repo root and site-packages before importing the entry point; falls
    back to a .cmd delegator (same interpreter, same PYTHONPATH) when
    distlib is unavailable."""
    module, func = ENTRY_POINTS[name]
    out_dir = Path(out_dir)

    script_maker_cls = _load_script_maker()
    if script_maker_cls is not None:
        # The template is %-formatted by distlib with (module, import_name,
        # func); the install-time paths are baked in as literals. The repo
        # root goes FIRST — its packages win over site-packages (same
        # ordering as the desktop shim's PYTHONPATH composition).
        esc = lambda p: str(p).replace("%", "%%")  # noqa: E731
        site_line = (
            f"sys.path.append({esc(site_packages)!r})\n" if site_packages else ""
        )

        class _PathedScriptMaker(script_maker_cls):  # type: ignore[misc,valid-type]
            script_template = (
                "# -*- coding: utf-8 -*-\n"
                "import re\n"
                "import sys\n"
                f"sys.path.insert(0, {esc(repo_root)!r})\n"
                f"{site_line}"
                "if __name__ == '__main__':\n"
                "    from %(module)s import %(import_name)s\n"
                "    sys.argv[0] = re.sub(r'(-script\\.pyw|\\.exe)?$', '', sys.argv[0])\n"
                "    sys.exit(%(func)s())\n"
            )

        maker = _PathedScriptMaker(None, str(out_dir), add_launchers=True)
        maker.executable = str(python_exe)
        maker.variants = {""}
        maker.clobber = True
        try:
            written = maker.make(f"{name} = {module}:{func}")
        except Exception:
            written = []
        for path in written:
            if Path(path).suffix.lower() == ".exe":
                return Path(path)
        # distlib ran but produced no exe (unexpected) — fall through to cmd.

    site = f";{site_packages}" if site_packages else ""
    code = f"import sys; from {module} import {func}; sys.exit({func}())"
    body = (
        "@echo off\r\n"
        "chcp 65001 >nul\r\n"
        f'set "PYTHONPATH={repo_root}{site}"\r\n'
        'set "PYTHONHOME="\r\n'
        f'"{python_exe}" -c "{code}" %*\r\n'
    )
    return _write_atomic(out_dir / f"{name}.cmd", lambda p: p.write_text(body, encoding="utf-8"))


def stage_launcher(name: str, repo_root: Path, out_dir: Path) -> Path | None:
    """Stage/refresh ONE launcher for the install rooted at repo_root,
    writing what the CURRENT state supports:

    - store interpreter present → exe (or .cmd if distlib is missing) bound
      to it, with the repo-first PYTHONPATH baked in;
    - store interpreter absent → a runtime-resolving .cmd that finds the
      store python at boot and fails with a clear message until
      `hermes pm install` materializes it.

    Never raises; returns the written path or None."""
    repo_root = Path(repo_root)
    venv_dir = project_venv_dir(repo_root)
    site_packages = venv_site_packages(venv_dir) if venv_dir else None
    store_python = resolve_store_python(repo_root)
    if store_python is not None:
        path = mint_launcher(name, repo_root, out_dir, store_python, site_packages)
        if path is not None:
            return path
    return _write_runtime_cmd(name, repo_root, site_packages, out_dir)


def ensure_install_launchers(repo_root: Path, out_dir: Path) -> list[str]:
    """Stage/refresh every launcher in WINDOWS_BIN_LAUNCHERS — see
    :func:`stage_launcher` for the per-name contract."""
    repo_root = Path(repo_root)
    written: list[str] = []
    for name in WINDOWS_BIN_LAUNCHERS:
        path = stage_launcher(name, repo_root, Path(out_dir))
        if path is not None:
            written.append(str(path))
    return written


def _write_runtime_cmd(
    name: str,
    repo_root: Path,
    site_packages: Path | None,
    out_dir: Path,
) -> Path | None:
    """The boot-time-resolving .cmd fallback (fresh install, no store
    interpreter yet): glob ``<runtime>\\python-*`` for the store python at
    boot, compose PYTHONPATH, and fail with a clear message when the store
    is empty. Never references the venv interpreter. The ``endlocal & set``
    idiom hoists the boot-resolved interpreter and PYTHONPATH out of the
    setlocal scope (percent expansion happens while setlocal is still
    active, before endlocal executes)."""
    module, func = ENTRY_POINTS[name]
    site = ""
    if site_packages is not None:
        site = (
            ";%HERMES_REPO%\\"
            + str(site_packages.relative_to(repo_root)).replace("/", "\\")
        )
    body = (
        "@echo off\r\n"
        "chcp 65001 >nul\r\n"
        "setlocal\r\n"
        f'set "HERMES_REPO={repo_root}"\r\n'
        'set "PM_RT=%HERMES_RUNTIME_DIR%"\r\n'
        'if not defined PM_RT set "PM_RT=%LOCALAPPDATA%\\hermes\\tools"\r\n'
        'set "PM_PY="\r\n'
        'for /d %%D in ("%PM_RT%\\python-*") do if exist "%%D\\python.exe" set "PM_PY=%%D\\python.exe"\r\n'
        'if not defined PM_PY (\r\n'
        "  echo hermes: no pm store interpreter under %PM_RT% - run \"hermes pm install\" first 1>&2\r\n"
        "  exit /b 1\r\n"
        ")\r\n"
        f'endlocal & set "PYTHONPATH=%HERMES_REPO%{site}" & set "PM_PY=%PM_PY%"\r\n'
        'set "PYTHONHOME="\r\n'
        f'set "PM_ENTRY=import sys; from {module} import {func}; sys.exit({func}())"\r\n'
        '"%PM_PY%" -c "%PM_ENTRY%" %*\r\n'
    )
    return _write_atomic(
        Path(out_dir) / f"{name}.cmd",
        lambda p: p.write_text(body, encoding="utf-8"),
    )
