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
from pm.environments import dependency_home_root, store_root


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
#: Publish-UserCommand and hermes_cli/_install_repair.py.
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
    # Profile boot repairs shared launchers: their default must stay at the
    # install's dependency root, not whichever profile triggered publication.
    return (
        "import os, re, sys\n"
        f"os.environ['HERMES_HOME'] = os.environ.get('HERMES_HOME') or {str(dependency_home_root())!r}\n"
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


def _write_shell(target: Path, command: list[str]) -> Path | None:
    body = f'#!/bin/sh\nexec {shlex.join(command)} "$@"\n'
    try:
        if not target.is_symlink() and target.read_bytes() == body.encode("utf-8"):
            if os.access(target, os.X_OK):
                return target
    except OSError:
        pass

    def write(staging: Path) -> None:
        staging.write_text(body, encoding="utf-8", newline="\n")
        staging.chmod(0o755)

    return _write_atomic(target, write)


def _mint_shell_launcher(name: str, out_dir: Path, python_exe: Path, script: str) -> Path | None:
    return _write_shell(out_dir / name, [str(python_exe), "-I", "-c", script])


def _owns_launcher(target: Path, root: Path) -> bool:
    """Recognize our old source/venv launchers, never a mere mention in a comment."""
    if target.is_symlink():
        return target.resolve().is_relative_to(root)
    try:
        tokens = shlex.split(target.read_text(encoding="utf-8-sig"), comments=True)
    except (OSError, UnicodeError, ValueError):
        return False
    paths = {str(root / p) for p in (
        "hermes", "run_agent.py", "venv/bin/python", "venv/bin/python3",
        ".hermes/bin/hermes", ".hermes/bin/hermes-acp",
    )}
    # Current store launchers pass this Python bootstrap as one shell argument.
    bootstrap = f"sys.path.insert(0, {str(root)!r})"
    if paths.intersection(tokens) or any(bootstrap in token for token in tokens):
        return True
    # The historical updater wrote ACP as a sibling-hermes forwarder. Adopt
    # it only when that sibling demonstrably belongs to this installation.
    if target.name == "hermes-acp":
        sibling = target.with_name("hermes")
        return (tokens == ["exec", str(sibling), "acp", "$@"]
                and _owns_launcher(sibling, root))
    return False


def _publish_conveniences(root: Path, out_dir: Path, names, *, create: bool = True) -> dict[Path, bool]:
    """User-bin commands forward to durable local launchers, not a Python pin."""
    if create:
        out_dir.mkdir(parents=True, exist_ok=True)
    elif not out_dir.is_dir():
        return {}
    published = {}
    for name in names:
        target = out_dir / name
        if (not create or target.exists() or target.is_symlink()) and not _owns_launcher(target, root):
            continue
        before = target.lstat().st_mtime_ns if target.exists() or target.is_symlink() else None
        command = ([str(root / ".hermes/bin/hermes"), "--run-module", "run_agent"]
                   if name == "hermes-agent" else [str(root / ".hermes/bin" / name)])
        if _write_shell(target, command) is None:
            raise OSError(f"could not publish launcher {target}")
        published[target] = before != target.lstat().st_mtime_ns
    return published


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
    """Publish exact-install commands; conveniences follow them across Python repins."""
    root = Path(repo_root).resolve()
    local = root / ".hermes" / "bin"
    local.mkdir(parents=True, exist_ok=True)
    written = [str(path) for name in WINDOWS_BIN_LAUNCHERS
               if (path := stage_launcher(name, root, local)) is not None]
    if Path(out_dir).resolve() == local:
        return written
    if len(written) != len(WINDOWS_BIN_LAUNCHERS):
        return []
    if not _is_windows():
        return [str(path) for path in _publish_conveniences(root, Path(out_dir), WINDOWS_BIN_LAUNCHERS)]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    return [str(path) for name in WINDOWS_BIN_LAUNCHERS
            if (path := stage_launcher(name, root, Path(out_dir))) is not None]


def expose_cli(project_root: Path | None = None, *, create: bool = True) -> dict:
    """Repair PATH conveniences without taking over another installation's files.

    macOS CLI-first launches link the bundle's signed shims directly; Electron
    need not have run. Source installs converge on the store launcher owner.
    Shell rc/PATH registration remains installer-owned.

    Before dependency sync succeeds, create=False maintains only commands we
    already own, without loading application config or enabling new exposure.
    """
    # Resolved before the platform branch: the Windows path needs it too.
    from pm.paths import install_root

    root = Path(project_root or install_root()).resolve()
    if _is_windows():
        # The installer stages the user-facing commands into $HERMES_HOME\bin
        # and registers that directory in the User PATH. An update skipped both
        # (this used to answer "windows-installer-owned"), so a machine updated
        # from a release predating that convention kept the old
        # venv\Scripts entry and never converged on it. Mirror the installer.
        return _expose_windows_user_bin(root, create=create)
    if create:
        try:
            from hermes_cli.config import load_config
        except ImportError:
            # Installers expose after sync; never invent a config reader here.
            return {"ok": True, "skipped": "config-unavailable"}
        cli_cfg = (load_config() or {}).get("cli", {})
        if isinstance(cli_cfg, dict) and not cli_cfg.get("expose_on_path", True):
            return {"ok": True, "skipped": "config-disabled"}
    from hermes_cli.steward import read_install_stamp

    if _is_bundled_payload(root):
        if create and sys.platform == "darwin":
            return _symlink_sealed_launchers(root.parent / "bin")
        return {"ok": True, "skipped": "bundle-owns-launchers"}
    if read_install_stamp(root).get("updateMechanism") == "external":
        return {"ok": True, "skipped": "externally-owned"}
    if resolve_store_python(root) is None:
        return {"ok": True, "skipped": "no-store-python"}
    try:
        local = root / ".hermes" / "bin"
        if len(ensure_install_launchers(root, local)) != len(WINDOWS_BIN_LAUNCHERS):
            return {"ok": False, "error": "source launcher publication failed"}
        dirs = [Path.home() / ".local" / "bin"]
        # Repair existing FHS/custom-home exposure, but never create new global
        # entries or reclaim a convenience that was repointed to another root.
        from hermes_constants import get_default_hermes_root
        for directory in (get_default_hermes_root() / "bin", Path("/usr/local/bin")):
            if directory not in dirs and (not create or _owns_launcher(directory / "hermes", root)):
                dirs.append(directory)
        written = []
        for directory in dirs:
            published = _publish_conveniences(root, directory, (*WINDOWS_BIN_LAUNCHERS, "hermes-agent"), create=create)
            written.extend(path.name for path, changed in published.items() if changed)
        return {"ok": True, "written": written}
    except OSError as exc:
        return {"ok": False, "error": str(exc)}


def _merge_user_path(existing: str, entry: str) -> str | None:
    """``entry`` first, or None when the PATH already names it.

    Windows paths are case-insensitive and may carry a trailing separator, so
    compare that way rather than by exact string.
    """
    wanted = entry.rstrip("\\/")
    parts = [part for part in (existing or "").split(";") if part]
    if any(part.rstrip("\\/").casefold() == wanted.casefold() for part in parts):
        return None
    return ";".join([entry, *parts])


def _register_windows_user_path(entry: Path) -> str:
    """Put ``entry`` on the User PATH. Returns 'present' or 'added'.

    Preserves the stored value type: install.ps1 writes expandable entries
    (``%LOCALAPPDATA%\\...``), and rewriting the value as a plain string would
    freeze those.
    """
    import winreg  # type: ignore

    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0,
                        winreg.KEY_QUERY_VALUE | winreg.KEY_SET_VALUE) as key:
        try:
            current, kind = winreg.QueryValueEx(key, "Path")
        except FileNotFoundError:
            current, kind = "", winreg.REG_EXPAND_SZ
        merged = _merge_user_path(str(current), str(entry))
        if merged is None:
            return "present"
        if kind not in (winreg.REG_SZ, winreg.REG_EXPAND_SZ):
            kind = winreg.REG_EXPAND_SZ
        winreg.SetValueEx(key, "Path", 0, kind, merged)
    _broadcast_environment_change()
    return "added"


def _broadcast_environment_change() -> None:
    """Tell running processes the environment changed (best effort)."""
    import ctypes

    try:
        ctypes.windll.user32.SendMessageTimeoutW(0xFFFF, 0x1A, 0, "Environment", 0x0002, 5000, None)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - never fail an update over the broadcast
        pass


def _expose_windows_user_bin(root: Path, *, create: bool) -> dict:
    """Windows twin of expose_cli's POSIX half.

    Stage the user-facing commands into ``$HERMES_HOME\\bin`` and register that
    directory in the User PATH -- the same two things scripts/install.ps1 does --
    so an update converges a machine installed under the older venv\\Scripts
    convention instead of leaving it there forever.
    """
    from hermes_constants import get_default_hermes_root

    directory = get_default_hermes_root() / "bin"
    try:
        if not create:
            # Bootstrap: repair only commands we already own, enable nothing new.
            published = _publish_conveniences(root, directory, WINDOWS_BIN_LAUNCHERS, create=False)
            return {"ok": True, "written": [path.name for path, changed in published.items() if changed]}
        directory.mkdir(parents=True, exist_ok=True)
        written = ensure_install_launchers(root, directory)
        if len(written) != len(WINDOWS_BIN_LAUNCHERS):
            return {"ok": False, "error": "source launcher publication failed"}
        return {"ok": True, "path": _register_windows_user_path(directory),
                "written": [Path(path).name for path in written]}
    except OSError as exc:
        return {"ok": False, "error": str(exc)}


def _is_bundled_payload(root: Path) -> bool:
    """Is ``root`` a desktop bundle's agent payload? The stamp is the
    authority (payload marker / desktop-app distribution), never a
    sibling-directory sniff; a .git tree is a dev checkout regardless."""
    if (root / ".git").exists():
        return False
    from hermes_cli.steward import STEWARD_DESKTOP, read_install_stamp

    stamp = read_install_stamp(root)
    if not stamp:
        return False
    return bool(stamp.get("payload")) or stamp.get("distribution") == STEWARD_DESKTOP


def _symlink_sealed_launchers(payload_bin) -> dict:
    """Link ~/.local/bin/{hermes,hermes-agent,hermes-acp} at a sealed
    bundle's own prebuilt shims (macOS only).

    Symlinks, not copies: the shims are signed as part of the app bundle,
    and a copy would both orphan the signature's context and go stale on
    every app update — a symlink into the .app follows the bundle's
    content wherever Squirrel.Mac swaps it.

    Ownership guard mirrors expose_cli's wrapper logic: an existing
    entry is replaced only when it is ours — a symlink into THIS app
    bundle's payload — or missing/broken. A user's own `hermes` (pipx,
    another checkout's wrapper) is never touched.
    """
    link_dir = Path.home() / ".local" / "bin"
    payload_root = payload_bin.parent
    written: list[str] = []
    try:
        link_dir.mkdir(parents=True, exist_ok=True)
        for name in ("hermes", "hermes-agent", "hermes-acp"):
            source = payload_bin / name
            if not source.is_file():
                continue
            target = link_dir / name
            if target.is_symlink():
                current = os.readlink(target)
                if current == str(source):
                    continue  # already ours and current
                # Ours if it points into this payload (stale app path from
                # a previous version counts — resolve() of a dangling link
                # still yields the old path text) — or dangling entirely.
                points_into_payload = str(Path(current)).startswith(str(payload_root) + os.sep)
                if not points_into_payload and target.exists():
                    continue  # a live foreign link — user's arrangement
            elif target.exists():
                continue  # a real file we did not write — never clobber
            target.unlink(missing_ok=True)
            target.symlink_to(source)
            written.append(name)
    except OSError as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "written": written, "mode": "sealed-symlinks"}


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
