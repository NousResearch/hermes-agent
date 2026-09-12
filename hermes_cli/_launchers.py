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

from hermes_constants import get_hermes_home
from hermes_cli.runtime_paths import store_root


def runtime_command(repo_root: Path, args=(), *, module: str = "hermes_cli.main",
                    code: str | None = None, python: str | Path | None = None,
                    home: str | Path | None = None) -> list[str]:
    """An installation-bound command, safe to persist across dependency GC.

    Store Python owns the ABI; bootstrap selects and leases dependencies at
    child start. Nix and developer interpreters retain their external owner.
    No selected generation or ambient PYTHONPATH is captured in the command.
    """
    root = Path(repo_root).resolve()
    python = python or resolve_store_python(root) or Path(sys.executable)
    entry = f"exec({code!r})" if code is not None else (
        f"runpy.run_module({module!r}, run_name='__main__', alter_sys=True)")
    bootstrap = (
        "import os, sys, runpy; "
        f"os.environ['HERMES_HOME'] = os.environ.get('HERMES_HOME') or {str(home or get_hermes_home())!r}; "
        "os.environ.pop('PYTHONHOME', None); os.environ.pop('PYTHONPATH', None); "
        "os.environ.pop('VIRTUAL_ENV', None); "
        f"sys.path.insert(0, {str(root)!r}); "
        "import hermes_bootstrap; "
        + entry
    )
    return [str(python), "-I", "-c", bootstrap, *args]


def print_runtime_command(repo_root: Path, argv: list[str]) -> None:
    """Machine boundary for consumers holding the exact published launcher."""
    import argparse

    parser = argparse.ArgumentParser(description="Resolve this installation's launch command.")
    parser.add_argument("--module", default="hermes_cli.main")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    options = parser.parse_args(argv)
    args = options.args[1:] if options.args[:1] == ["--"] else options.args
    print(json.dumps(runtime_command(repo_root, args, module=options.module)))


def installation_command(repo_root: Path, args=(), *, module: str = "hermes_cli.main",
                         python: str | Path | None = None, home: str | Path | None = None) -> list[str]:
    """Persist a source launcher, never the versioned tool it currently uses.

    External/Nix installs retain their externally owned interpreter contract.
    Source installation/update publication refreshes the local launcher when
    the managed Python pin changes.
    """
    root = Path(repo_root)
    if resolve_store_python(root) is None:
        return runtime_command(root, args, module=module, python=python, home=home)
    prefix = [] if module == "hermes_cli.main" else ["--run-module", module]
    return [str(root / ".hermes" / "bin" / "hermes"), *prefix, *args]

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
    """Read PM's committed Python tool, without adopting unrecorded bytes."""
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
        "if sys.argv[1:2] == ['--print-runtime-command']:\n"
        "    sys.dont_write_bytecode = True\n"
        "    from pathlib import Path\n"
        "    from hermes_cli._launchers import print_runtime_command\n"
        f"    print_runtime_command(Path({str(repo_root.resolve())!r}), sys.argv[2:])\n"
        "    sys.exit(0)\n"
        "import hermes_bootstrap\n"
        "if sys.argv[1:2] == ['--run-module']:\n"
        "    import runpy\n"
        "    if len(sys.argv) < 3: sys.exit('hermes: --run-module needs a module')\n"
        "    module = sys.argv.pop(2)\n"
        "    del sys.argv[1]\n"
        "    runpy.run_module(module, run_name='__main__', alter_sys=True)\n"
        "    sys.exit(0)\n"
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
    """Publish one launcher bound to store Python, or refuse missing tools."""
    repo_root = Path(repo_root)
    store_python = resolve_store_python(repo_root)
    if store_python is not None:
        path = mint_launcher(name, repo_root, out_dir, store_python, None)
        if path is not None and path.suffix == ".cmd":
            # cmd.exe prefers .exe. An older launcher must not shadow the
            # newly published command when distlib is unavailable.
            try:
                (Path(out_dir) / f"{name}.exe").unlink(missing_ok=True)
            except OSError:
                return None
        return path
    return None


def ensure_install_launchers(repo_root: Path, out_dir: Path) -> list[str]:
    """Publish exact-install commands and their user-bin conveniences.

    Installers/updaters use the local command, since a shared HOME/bin may
    have been repointed to another checkout. Return the requested outputs.
    """
    repo_root = Path(repo_root)
    local = repo_root / ".hermes" / "bin"
    local.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for name in WINDOWS_BIN_LAUNCHERS:
        if Path(out_dir).resolve() != local.resolve():
            if stage_launcher(name, repo_root, local) is None:
                continue
        path = stage_launcher(name, repo_root, Path(out_dir))
        if path is not None:
            written.append(str(path))
    return written


if __name__ == "__main__":
    import argparse

    if sys.argv[1:2] == ["--print-runtime-command"]:
        print_runtime_command(Path(__file__).resolve().parents[1], sys.argv[2:])
        raise SystemExit(0)

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
