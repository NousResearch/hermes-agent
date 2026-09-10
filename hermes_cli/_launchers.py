"""Source-install launchers shared by setup, installers, and Windows repair.

Launchers execute store Python in isolated mode. They set the install's
default home and load hermes_bootstrap before the entry point. Bootstrap
reads the selected dependency generation at each start.

Windows uses distlib executables or a command-file fallback. POSIX uses
an executable shell wrapper. The standalone writer requires PM's store
interpreter before it publishes either command.
"""

from __future__ import annotations

import json
import os
import shlex
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hermes_constants import get_hermes_home, project_venv_dir
from hermes_cli.runtime_paths import site_packages, store_root

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
    """Write a native launcher with the shared bootstrap script, or return None."""
    module, func = ENTRY_POINTS[name]
    out_dir = Path(out_dir)
    script = _launcher_script(name, Path(repo_root), site_packages)

    if not _is_windows():
        return _mint_shell_launcher(name, out_dir, python_exe, script)

    script_maker_cls = _load_script_maker()
    if script_maker_cls is not None:
        class _PathedScriptMaker(script_maker_cls):  # type: ignore[misc,valid-type]
            def _get_script_text(self, entry):
                return script

        maker = _PathedScriptMaker(None, str(out_dir), add_launchers=True)
        maker.executable = str(python_exe)
        maker.variants = {""}
        maker.clobber = True
        try:
            written = maker.make(f"{name} = {module}:{func}", {"interpreter_args": ["-I"]})
        except Exception:
            written = []
        for path in written:
            if Path(path).suffix.lower() == ".exe":
                return Path(path)
        # distlib ran but produced no exe (unexpected) — fall through to cmd.

    # The script is data to Python, not interpolated shell source.
    import base64
    encoded = base64.b64encode(script.encode("utf-8")).decode("ascii")
    code = f"import base64; exec(base64.b64decode('{encoded}'))"
    body = (
        "@echo off\r\n"
        "chcp 65001 >nul\r\n"
        f'"{python_exe}" -I -c "{code}" %*\r\n'
    )
    return _write_atomic(out_dir / f"{name}.cmd", lambda p: p.write_text(body, encoding="utf-8"))


def _launcher_script(name: str, repo_root: Path, dependencies: Path | None) -> str:
    module, func = ENTRY_POINTS[name]
    return (
        "import os, re, sys\n"
        f"os.environ['HERMES_HOME'] = os.environ.get('HERMES_HOME') or {str(get_hermes_home())!r}\n"
        "os.environ.pop('PYTHONHOME', None)\n"
        "os.environ.pop('PYTHONPATH', None)\n"
        f"sys.path.insert(0, {str(repo_root.resolve())!r})\n"
        + (f"sys.path.append({str(dependencies)!r})\n" if dependencies else "")
        + "import hermes_bootstrap\n"
        f"from {module} import {func}\n"
        "sys.argv[0] = re.sub(r'(-script\\.pyw|\\.exe)?$', '', sys.argv[0])\n"
        f"sys.exit({func}())\n"
    )


def _mint_shell_launcher(name: str, out_dir: Path, python_exe: Path, script: str) -> Path | None:
    command = shlex.join([str(python_exe), "-I", "-c", script])

    def write(staging: Path) -> None:
        staging.write_text(f'#!/bin/sh\nexec {command} "$@"\n', encoding="utf-8", newline="\n")
        staging.chmod(0o755)

    return _write_atomic(out_dir / name, write)


def stage_launcher(name: str, repo_root: Path, out_dir: Path) -> Path | None:
    """Publish one launcher bound to the current store interpreter.

    Windows repair retains a command-file fallback when the store is absent.
    The standalone install writer refuses that incomplete state.
    """
    repo_root = Path(repo_root)
    venv_dir = project_venv_dir(repo_root)
    dependencies = site_packages(venv_dir) if venv_dir else None
    store_python = resolve_store_python(repo_root)
    if store_python is not None:
        path = mint_launcher(name, repo_root, out_dir, store_python, dependencies)
        if path is not None:
            return path
    if not _is_windows():
        return None
    return _write_runtime_cmd(name, repo_root, dependencies, out_dir)


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
        f'set "PM_ENTRY=import sys; import hermes_bootstrap; from {module} import {func}; sys.exit({func}())"\r\n'
        '"%PM_PY%" -c "%PM_ENTRY%" %*\r\n'
    )
    return _write_atomic(
        Path(out_dir) / f"{name}.cmd",
        lambda p: p.write_text(body, encoding="utf-8"),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Publish source-install launchers.")
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    if resolve_store_python(repo_root) is None:
        parser.exit(1, "hermes: store interpreter is missing; finish pm install before publishing launchers\n")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = ensure_install_launchers(repo_root, args.out_dir)
    if len(written) != len(ENTRY_POINTS):
        parser.exit(1, "hermes: launcher publication failed\n")
    print("\n".join(written))
