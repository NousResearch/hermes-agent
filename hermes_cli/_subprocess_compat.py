"""Windows subprocess compatibility helpers.

Hermes is developed on Linux / macOS and tested natively on Windows too.
Several common subprocess patterns break silently-or-loudly on Windows:

* ``["npm", "install", ...]`` — on Windows ``npm`` is ``npm.cmd``, a batch
  shim.  ``subprocess.Popen(["npm", ...])`` fails with WinError 193
  ("not a valid Win32 application") because CreateProcessW can't run a
  ``.cmd`` file without ``shell=True`` or PATHEXT resolution.

* ``start_new_session=True`` — on POSIX, this maps to ``os.setsid()`` and
  actually detaches the child.  On Windows it's silently ignored; the
  Windows equivalent is the ``CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW``
  creationflags bundle, which Python only applies when you pass it
  explicitly.

* Console-window flashes — every ``subprocess.Popen`` of a ``.exe`` on
  Windows spawns a cmd window briefly unless ``CREATE_NO_WINDOW`` is
  passed.  Cosmetic but jarring for background daemons.

This module centralizes the platform-branching logic so the rest of the
codebase doesn't sprinkle ``if sys.platform == "win32":`` everywhere.

**All helpers are no-ops on non-Windows** — calling them in Linux/macOS
code paths is safe by design.  That's the "do no damage on POSIX"
guarantee.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path, PureWindowsPath
from typing import Mapping, NamedTuple, Sequence

__all__ = [
    "IS_WINDOWS",
    "resolve_node_command",
    "split_command_line",
    "suppress_platform_ver_console",
    "windows_detach_flags",
    "windows_detach_flags_without_breakaway",
    "windows_hide_flags",
    "windows_detach_popen_kwargs",
    "bounded_git_probe",
    "bounded_probe_run",
    "noninteractive_git_env",
]


IS_WINDOWS = sys.platform == "win32"

# Private launcher-to-child metadata. This is diagnostic state, not user config.
_WINDOWS_GATEWAY_BREAKAWAY_ENV = "_HERMES_GATEWAY_BREAKAWAY"


def split_command_line(line: str) -> list[str]:
    """Split a user-supplied command line into tokens, Windows-safely.

    ``shlex.split(line)`` (posix=True) treats every backslash as an escape
    character, so Windows paths are silently mangled: ``C:\\Users\\me\\out.txt``
    becomes ``C:Usersmeout.txt`` — no error, just a wrong path that then
    "succeeds" against a mangled relative filename (#83934) or makes a valid
    hook script report "not executable" (#78293).

    On Windows this uses ``posix=False``, which preserves backslashes while
    still honoring double-quoted tokens ("path with spaces"). The trade-off
    is that posix=False keeps surrounding quotes on quoted tokens, so we
    strip one layer of matching double quotes per token — that matches how
    Windows command lines are conventionally parsed. On POSIX the behavior
    is exactly ``shlex.split``.

    Raises ValueError for unbalanced quotes, same as ``shlex.split``.
    """
    if not IS_WINDOWS:
        import shlex

        return shlex.split(line)
    import shlex

    tokens = shlex.split(line, posix=False)
    out: list[str] = []
    for tok in tokens:
        if len(tok) >= 2 and tok[0] == tok[-1] and tok[0] in ("'", '"'):
            tok = tok[1:-1]
        out.append(tok)
    return out


# -----------------------------------------------------------------------------
# Node ecosystem launcher resolution
# -----------------------------------------------------------------------------


_NPM_CMD_SHIM_HEADER = (
    "@echo off",
    "goto start",
    ":find_dp0",
    "set dp0=%~dp0",
    "exit /b",
    ":start",
    "setlocal",
    "call :find_dp0",
)
_NPM_CMD_SHIM_GENERATED_COMMENT = ":: created by npm, please don't edit manually."
_NPM_CMD_SHIM_LAUNCH_PREFIXES = (
    "endlocal & goto #_undefined_# 2>nul || title %comspec% & ",
    "endlocal & (call) || title %comspec% & ",
)
_NPM_CMD_SHIM_TARGET = re.compile(
    r'^"%_prog%"\s+"(?P<target>%dp0%\\[^"\r\n]+)"\s+%\*$', re.I
)
_LEGACY_NODE_CMD_SHIM_LOCAL = re.compile(
    r'^"%~dp0\\node\.exe"\s+"(?P<target>%~dp0\\[^"\r\n]+)"\s+%\*$',
    re.I,
)
_LEGACY_NODE_CMD_SHIM_PATH = re.compile(
    r'^node\s+"(?P<target>%~dp0\\[^"\r\n]+)"\s+%\*$',
    re.I,
)


class _NodeShimTarget(NamedTuple):
    entrypoint: Path
    prefer_adjacent_node: bool
    package_name: str | None = None
    removes_interior_js_from_pathext: bool = False


def _generated_cmd_shim_body(lines: list[str]) -> list[str]:
    """Remove only npm's known generated preamble and leading blank lines."""
    index = 0
    while index < len(lines) and not lines[index].strip():
        index += 1
    if (
        index < len(lines)
        and lines[index].strip().casefold() == _NPM_CMD_SHIM_GENERATED_COMMENT
    ):
        index += 1
        while index < len(lines) and not lines[index].strip():
            index += 1
    return lines[index:]


def _modern_npm_cmd_shim_entrypoint(lines: list[str]) -> str | None:
    """Extract a target from the current npm ``cmd-shim`` Node template."""
    if tuple(line.casefold() for line in lines[:8]) != _NPM_CMD_SHIM_HEADER:
        return None

    body = [line.strip() for line in lines[8:] if line.strip()]
    # Do not skip generated @SET declarations here. Native argv conversion
    # cannot reproduce their environment without changing this API's contract.
    index = 0
    required_prefix = [
        'if exist "%dp0%\\node.exe" (',
        'set "_prog=%dp0%\\node.exe"',
        ") else (",
        'set "_prog=node"',
    ]
    if [line.casefold() for line in body[index : index + 4]] != required_prefix:
        return None

    index += 4
    if index < len(body) and body[index].casefold() == (
        "set pathext=%pathext:;.js;=;%"
    ):
        index += 1
    if index >= len(body) or body[index] != ")":
        return None
    index += 1
    if len(body) != index + 1:
        return None

    launch = body[index]
    folded_launch = launch.casefold()
    launch_prefix = next(
        (
            prefix
            for prefix in _NPM_CMD_SHIM_LAUNCH_PREFIXES
            if folded_launch.startswith(prefix)
        ),
        None,
    )
    if launch_prefix is None:
        return None
    native_command = launch[len(launch_prefix) :]
    if native_command.casefold().startswith("set pathext=%pathext:;.js;=;% & "):
        native_command = native_command[len("set PATHEXT=%PATHEXT:;.JS;=;% & ") :]
    match = _NPM_CMD_SHIM_TARGET.fullmatch(native_command)
    return match.group("target") if match else None


def _npm_cli_cmd_shim_entrypoint(lines: list[str], command: str) -> str | None:
    """Recognize npm/npx's exact release launcher and return its CLI file.

    npm CLI ships hand-maintained Windows launchers rather than the generic
    ``cmd-shim`` template. Match the complete non-blank template shipped from
    npm 11.12.1 through 12.0.2 so a custom batch file cannot gain native
    execution merely by declaring the same variables.
    """
    if command not in {"npm", "npx"}:
        return None
    cli = command.upper()
    cli_file = f"{command}-cli.js"
    body = [line.strip() for line in lines if line.strip()]
    expected = [
        _NPM_CMD_SHIM_GENERATED_COMMENT,
        "@ECHO OFF",
        "SETLOCAL",
        'SET "NODE_EXE=%~dp0\\node.exe"',
        'IF NOT EXIST "%NODE_EXE%" (',
        'SET "NODE_EXE=node"',
        ")",
        'SET "NPM_PREFIX_JS=%~dp0\\node_modules\\npm\\bin\\npm-prefix.js"',
        f'SET "{cli}_CLI_JS=%~dp0\\node_modules\\npm\\bin\\{cli_file}"',
        'FOR /F "delims=" %%F IN (\'CALL "%NODE_EXE%" "%NPM_PREFIX_JS%"\') DO (',
        f'SET "NPM_PREFIX_{cli}_CLI_JS=%%F\\node_modules\\npm\\bin\\{cli_file}"',
        ")",
        f'IF EXIST "%NPM_PREFIX_{cli}_CLI_JS%" (',
        f'SET "{cli}_CLI_JS=%NPM_PREFIX_{cli}_CLI_JS%"',
        ")",
        f'"%NODE_EXE%" "%{cli}_CLI_JS%" %*',
    ]
    if [line.casefold() for line in body] != [line.casefold() for line in expected]:
        return None
    return cli_file


def _yarn_classic_cmd_shim_entrypoint(lines: list[str]) -> str | None:
    """Extract Yarn Classic 1.22.22's exact direct-node target."""
    body = [line.strip() for line in lines if line.strip()]
    if [line.casefold() for line in body] != [
        "@echo off",
        'node "%~dp0\\yarn.js" %*',
    ]:
        return None
    return "%~dp0\\yarn.js"


def _yarn_classic_delegated_entrypoint(
    shim_path: Path, lines: list[str]
) -> str | None:
    """Resolve Yarn Classic's exact ``yarnpkg.cmd`` one-hop delegation."""
    body = [line.strip() for line in lines if line.strip()]
    if [line.casefold() for line in body] != [
        "@echo off",
        '"%~dp0\\yarn.cmd" %*',
    ]:
        return None

    yarn_shim = shim_path.parent / "yarn.cmd"
    try:
        yarn_lines = yarn_shim.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError):
        return None
    return _yarn_classic_cmd_shim_entrypoint(yarn_lines)


def _corepack_cmd_shim_entrypoint(lines: list[str]) -> str | None:
    """Extract a current Corepack ``@zkochan/cmd-shim`` target.

    Corepack 0.34.6 ships this compact direct-node template both within its
    npm package and in the ``nodewin`` layout copied into Node distributions.
    No interpreter options or user environment assignments are accepted.
    Package metadata still has to bind the returned target to the command.
    """
    body = [line.strip() for line in lines if line.strip()]
    if len(body) != 7:
        return None
    expected = [
        "@setlocal",
        '@if exist "%~dp0\\node.exe" (',
        None,
        ") else (",
        "@set pathext=%pathext:;.js;=;%",
        None,
        ")",
    ]
    folded = [line.casefold() for line in body]
    if any(
        value is not None and folded[index] != value
        for index, value in enumerate(expected)
    ):
        return None

    local_match = _LEGACY_NODE_CMD_SHIM_LOCAL.fullmatch(body[2])
    path_match = _LEGACY_NODE_CMD_SHIM_PATH.fullmatch(body[5])
    if not local_match or not path_match:
        return None
    local_target = local_match.group("target")
    path_target = path_match.group("target")
    return local_target if local_target.casefold() == path_target.casefold() else None


def _legacy_node_cmd_shim_entrypoint(lines: list[str]) -> str | None:
    """Extract a target from npm/Yarn's historical direct-node template."""
    # A leading NODE_PATH declaration is intentionally not discarded: leaving
    # the shim unresolved is safer than silently changing its launch semantics.
    body = [line.strip() for line in lines if line.strip()]
    folded = [line.casefold() for line in body]
    if len(body) == 5:
        structural = [
            '@if exist "%~dp0\\node.exe" (',
            None,
            ") else (",
            None,
            ")",
        ]
        local_index, path_index = 1, 3
    elif len(body) == 7:
        structural = [
            '@if exist "%~dp0\\node.exe" (',
            None,
            ") else (",
            "@setlocal",
            "@set pathext=%pathext:;.js;=;%",
            None,
            ")",
        ]
        local_index, path_index = 1, 5
    else:
        return None
    if any(
        expected is not None and folded[index] != expected
        for index, expected in enumerate(structural)
    ):
        return None

    local_match = _LEGACY_NODE_CMD_SHIM_LOCAL.fullmatch(body[local_index])
    path_match = _LEGACY_NODE_CMD_SHIM_PATH.fullmatch(body[path_index])
    if not local_match or not path_match:
        return None
    local_target = local_match.group("target")
    path_target = path_match.group("target")
    return local_target if local_target.casefold() == path_target.casefold() else None


def _package_bin_entrypoint(
    package_json: Path, command: str, package_name: str | None = None
) -> Path | None:
    """Return a contained, existing package ``bin`` entry for ``command``."""
    try:
        package = json.loads(package_json.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(package, dict):
        return None
    if package_name is not None and str(package.get("name") or "").casefold() != (
        package_name.casefold()
    ):
        return None

    bins = package.get("bin")
    relative: object = None
    if isinstance(bins, dict):
        relative = next(
            (
                value
                for name, value in bins.items()
                if str(name).casefold() == command
            ),
            None,
        )
    elif isinstance(bins, str):
        declared_name = str(package.get("name") or "").rsplit("/", 1)[-1]
        if declared_name.casefold() == command:
            relative = bins
    if not isinstance(relative, str) or not relative:
        return None

    package_dir = package_json.parent.resolve()
    entrypoint = (package_dir / relative).resolve()
    try:
        entrypoint.relative_to(package_dir)
    except ValueError:
        return None
    return entrypoint if entrypoint.is_file() else None


def _node_executable(
    shim_path: Path,
    prefer_adjacent: bool,
    cwd: str | os.PathLike[str] | None = None,
    *,
    search_cwd: bool = True,
    removes_interior_js_from_pathext: bool = False,
) -> str | None:
    """Resolve the native Node executable used by a supported launcher."""
    adjacent_node = shim_path.parent / "node.exe"
    if prefer_adjacent and adjacent_node.is_file():
        return str(adjacent_node.resolve())
    if IS_WINDOWS and cwd is not None:
        node = _which_windows_command_from_cwd(
            "node",
            cwd,
            search_cwd=search_cwd,
            removes_interior_js_from_pathext=removes_interior_js_from_pathext,
        )[0]
    else:
        node = shutil.which("node.exe") or shutil.which("node")
    if node is None:
        return None
    if IS_WINDOWS:
        return node if Path(node).suffix.casefold() in {".com", ".exe"} else None
    return None if node.casefold().endswith((".cmd", ".bat")) else node


def _npm_cli_selected_entrypoint(
    shim_path: Path,
    command: str,
    cli_file: str,
    cwd: str | os.PathLike[str] | None,
    *,
    search_cwd: bool,
) -> Path | None:
    """Reproduce npm's prefix probe and preserve the CLI it selects."""
    local_package = shim_path.parent / "node_modules" / "npm"
    package_json = local_package / "package.json"
    local_entrypoint = _package_bin_entrypoint(package_json, command, "npm")
    expected_local = (local_package / "bin" / cli_file).resolve()
    if local_entrypoint is None or local_entrypoint != expected_local:
        return None

    prefix_probe = (local_package / "bin" / "npm-prefix.js").resolve()
    try:
        prefix_probe.relative_to(local_package.resolve())
    except ValueError:
        return None
    if not prefix_probe.is_file():
        return None

    node = _node_executable(
        shim_path,
        prefer_adjacent=True,
        cwd=cwd,
        search_cwd=search_cwd,
    )
    if node is None:
        return None
    try:
        result = subprocess.run(
            [node, str(prefix_probe)],
            capture_output=True,
            check=False,
            creationflags=windows_hide_flags(),
            cwd=os.fspath(cwd) if cwd is not None else None,
            shell=False,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError, UnicodeError):
        return None

    stdout = result.stdout or ""
    prefixes = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not prefixes:
        return local_entrypoint
    if len(prefixes) != 1:
        return None

    selected_prefix = Path(prefixes[0])
    if not selected_prefix.is_absolute():
        return None
    selected_entrypoint = (
        selected_prefix / "node_modules" / "npm" / "bin" / cli_file
    ).resolve()
    # npm's batch launcher changes targets only when this exact file exists.
    return selected_entrypoint if selected_entrypoint.is_file() else local_entrypoint


def _npm_cmd_shim_entrypoint(
    shim_path: Path,
    cwd: str | os.PathLike[str] | None,
    *,
    search_cwd: bool,
) -> _NodeShimTarget | None:
    """Return the native target semantics of a supported Node batch shim."""
    try:
        lines = shim_path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError):
        return None

    command = shim_path.stem.casefold()
    npm_cli_file = _npm_cli_cmd_shim_entrypoint(lines, command)
    if npm_cli_file is not None:
        selected = _npm_cli_selected_entrypoint(
            shim_path,
            command,
            npm_cli_file,
            cwd,
            search_cwd=search_cwd,
        )
        if selected is None:
            return None
        return _NodeShimTarget(selected, True, "npm")

    if command == "yarn":
        target = _yarn_classic_cmd_shim_entrypoint(lines)
        if target is not None:
            relative_target = target[len("%~dp0\\") :]
            entrypoint = (
                shim_path.parent / relative_target.replace("\\", os.sep)
            ).resolve()
            return _NodeShimTarget(entrypoint, False, "yarn")
    elif command == "yarnpkg":
        target = _yarn_classic_delegated_entrypoint(shim_path, lines)
        if target is not None:
            relative_target = target[len("%~dp0\\") :]
            entrypoint = (
                shim_path.parent / relative_target.replace("\\", os.sep)
            ).resolve()
            return _NodeShimTarget(entrypoint, False, "yarn")

    body = _generated_cmd_shim_body(lines)
    target = _modern_npm_cmd_shim_entrypoint(body)
    if target is None:
        target = _corepack_cmd_shim_entrypoint(body)
    if target is None:
        target = _legacy_node_cmd_shim_entrypoint(body)
    if target is None:
        return None

    folded_target = target.casefold()
    prefix = "%dp0%\\" if folded_target.startswith("%dp0%\\") else "%~dp0\\"
    relative_target = target[len(prefix) :]
    entrypoint = (shim_path.parent / relative_target.replace("\\", os.sep)).resolve()
    removes_interior_js_from_pathext = any(
        "set pathext=%pathext:;.js;=;%" in line.casefold() for line in body
    )
    return _NodeShimTarget(
        entrypoint,
        True,
        removes_interior_js_from_pathext=removes_interior_js_from_pathext,
    )


def _node_package_entrypoint(
    shim: str,
    cwd: str | os.PathLike[str] | None,
    *,
    search_cwd: bool,
) -> list[str] | None:
    """Return ``[node.exe, script]`` for an npm-generated Windows shim.

    Batch shims cannot safely receive arbitrary argv: Windows can route them
    through ``cmd.exe`` even when the caller passes ``shell=False``. npm's
    shims are only launch adapters for a package ``bin`` entry, so resolve the
    same package metadata and invoke that JavaScript entrypoint with Node
    directly. Every subsequent value then remains in the native argv channel.
    """
    shim_path = Path(shim)
    shim_target = _npm_cmd_shim_entrypoint(
        shim_path,
        cwd,
        search_cwd=search_cwd,
    )
    if shim_target is None:
        return None
    shim_entrypoint = shim_target.entrypoint
    command = shim_path.stem.casefold()
    package_roots = [shim_path.parent / "node_modules"]
    if shim_path.parent.name.casefold() == ".bin":
        package_roots.insert(0, shim_path.parent.parent)

    package_jsons: list[Path] = []
    selected_package = shim_entrypoint.parent.parent / "package.json"
    if selected_package.is_file():
        package_jsons.append(selected_package)
    packaged_corepack = shim_path.parent.parent / "package.json"
    if packaged_corepack.is_file():
        package_jsons.append(packaged_corepack)
    for root in package_roots:
        direct = root / command / "package.json"
        if direct.is_file():
            package_jsons.append(direct)
        if root.is_dir():
            package_jsons.extend(sorted(root.glob("*/package.json")))
            package_jsons.extend(sorted(root.glob("@*/*/package.json")))

    seen: set[Path] = set()
    for package_json in package_jsons:
        if package_json in seen:
            continue
        seen.add(package_json)
        entrypoint = _package_bin_entrypoint(
            package_json, command, shim_target.package_name
        )
        if entrypoint is None:
            continue
        if str(entrypoint).casefold() != str(shim_entrypoint).casefold():
            continue

        node = _node_executable(
            shim_path,
            prefer_adjacent=shim_target.prefer_adjacent_node,
            cwd=cwd,
            search_cwd=search_cwd,
            removes_interior_js_from_pathext=(
                shim_target.removes_interior_js_from_pathext
            ),
        )
        if node:
            return [node, str(entrypoint)]
    return None


def _windows_command_candidates(
    name: str,
    *,
    removes_interior_js_from_pathext: bool = False,
) -> list[str]:
    """Return *name* candidates in native Windows PATHEXT order."""
    pathext_source = os.environ.get("PATHEXT") or ".COM;.EXE;.BAT;.CMD"
    if removes_interior_js_from_pathext:
        # Corepack's ``%PATHEXT:;.JS;=;%`` is a literal substring
        # replacement. It does not remove .JS when that token lacks a leading
        # or trailing semicolon at a PATHEXT boundary.
        pathext_source = re.sub(r";\.JS;", ";", pathext_source, flags=re.I)
    pathext = [extension.strip() for extension in pathext_source.split(";")]
    pathext = [extension for extension in pathext if extension]
    candidates = [f"{name}{extension}" for extension in pathext]
    if PureWindowsPath(name).suffix:
        candidates.insert(0, name)
    return candidates


def _which_windows_explicit_command(
    name: str,
    *,
    allowed_root: str | os.PathLike[str] | None = None,
) -> str | None:
    """Apply Python 3.12-style PATHEXT lookup to a path-qualified command."""
    root = Path(allowed_root).resolve() if allowed_root is not None else None
    for candidate_name in _windows_command_candidates(name):
        candidate = Path(candidate_name)
        if not candidate.is_file():
            continue
        if root is None:
            return os.fspath(candidate)
        resolved = candidate.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                "resolved Windows executable is outside its allowed root"
            ) from exc
        return os.fspath(resolved)
    return None


def _which_windows_command_from_cwd(
    name: str,
    cwd: str | os.PathLike[str],
    *,
    search_cwd: bool = True,
    removes_interior_js_from_pathext: bool = False,
) -> tuple[str | None, str]:
    """Resolve a bare Windows command as though *cwd* were already active.

    ``shutil.which`` may implicitly search the parent process's current
    directory on Windows, even when a different child ``cwd`` will be passed
    to ``subprocess``. Probe the intended child directory and each PATH entry
    through an explicit path instead, retaining PATHEXT handling without
    exposing Hermes's unrelated launch directory. Windows suppresses the
    initial child-directory probe when ``NoDefaultCurrentDirectoryInExePath``
    is present or ``search_cwd`` is false. Relative PATH entries remain
    interpreted against the intended child directory; empty entries retain
    that meaning only while current-directory search is enabled.

    The second return value is an explicit, non-searching fallback. Passing it
    to ``CreateProcessW`` fails closed when no candidate exists instead of
    giving the operating system another chance to search the parent cwd.
    """
    child_cwd = Path(cwd).resolve()
    searches_current_directory = search_cwd and (
        "NoDefaultCurrentDirectoryInExePath" not in os.environ
    )
    search_dirs = [child_cwd] if searches_current_directory else []
    for entry in os.get_exec_path():
        if not entry and not searches_current_directory:
            continue
        directory = Path(entry) if entry else child_cwd
        if not directory.is_absolute():
            directory = child_cwd / directory
        search_dirs.append(directory)

    candidates = _windows_command_candidates(
        name,
        removes_interior_js_from_pathext=removes_interior_js_from_pathext,
    )

    seen: set[str] = set()
    # Reuse the first directory that this helper will explicitly probe. This
    # leaves CreateProcessW an absolute, non-searching miss without pointing it
    # back at the repository when native cwd search is disabled. os.get_exec_path
    # normally supplies at least one entry; retain an invalid Win32 component as
    # the fail-closed boundary for a mocked or otherwise empty search path.
    fallback_directory = search_dirs[0] if search_dirs else child_cwd / "<PATH>"
    fallback = os.fspath(fallback_directory / name)
    for directory in search_dirs:
        key = os.path.normcase(os.path.abspath(os.fspath(directory))).casefold()
        if key in seen:
            continue
        seen.add(key)
        for candidate_name in candidates:
            candidate = directory / candidate_name
            if candidate.is_file():
                return os.fspath(candidate), fallback
    return None, fallback


def resolve_node_command(
    name: str,
    argv: Sequence[str],
    *,
    cwd: str | os.PathLike[str] | None = None,
    search_cwd: bool = True,
    allowed_root: str | os.PathLike[str] | None = None,
) -> list[str]:
    """Resolve a Node-ecosystem command name to an absolute-path argv.

    On Windows, commands like ``npm``, ``npx``, ``yarn``, ``pnpm``,
    ``playwright``, ``prettier`` ship as ``.cmd`` files (batch shims).
    ``subprocess.Popen(["npm", "install"])`` fails with WinError 193
    because CreateProcessW doesn't execute batch files directly.

    ``shutil.which`` *does* resolve ``.cmd`` via PATHEXT. When a Windows
    ``cwd`` is supplied, bare commands reproduce cmd.exe's child-directory
    search unless ``NoDefaultCurrentDirectoryInExePath`` opts out, then search
    PATH without consulting Hermes's parent-process cwd. npm-generated batch
    shims are resolved further to their package ``bin`` JavaScript entrypoint
    and invoked with ``node.exe``. That avoids the implicit ``cmd.exe`` path,
    where metacharacters in otherwise legitimate arguments can be
    reinterpreted as shell syntax.

    On POSIX ``shutil.which`` also returns a fully-qualified path when
    found.  That's a small change from bare-name resolution (the OS does
    its own PATH search) but functionally identical and has the side
    benefit of making the argv reproducible in logs.

    Behavior when the command is not on PATH:
    - On Windows with ``cwd``: return an explicit path in the first searched
      directory so a subsequent CreateProcessW call fails closed without
      searching the parent-process directory.
    - On Windows without ``cwd``: return the bare name — caller can still try
      with ``shell=True`` as a last resort, OR the subsequent Popen will raise
      FileNotFoundError with a readable error we want to surface.
    - On POSIX: same.  Bare ``npm`` on a Linux box without npm installed
      fails the same way it did before this function existed.

    Args:
        name: The command name to resolve (``npm``, ``npx``, ``node`` …).
        argv: The remaining arguments.  Must NOT include ``name`` itself —
            this function builds the full argv list.
        cwd: The child working directory that governs bare executable search
            and npm/npx prefix selection. Defaults to the current process
            directory and ordinary ``shutil.which`` behavior.
        search_cwd: Whether bare Windows commands may resolve from ``cwd``
            before PATH. Defaults to native Windows search semantics.
        allowed_root: Optional containment root for a path-qualified Windows
            executable. The selected PATHEXT candidate must resolve within it.

    Returns:
        A list suitable for passing to subprocess.Popen/run/call.
    """
    fallback = name
    windows_name = PureWindowsPath(name)
    is_bare_windows_name = (
        IS_WINDOWS
        and cwd is not None
        and "/" not in name
        and "\\" not in name
        and not windows_name.drive
    )
    if is_bare_windows_name:
        resolved, fallback = _which_windows_command_from_cwd(
            name,
            cwd,
            search_cwd=search_cwd,
        )
    elif IS_WINDOWS and (
        "/" in name or "\\" in name or bool(windows_name.drive)
    ):
        resolved = _which_windows_explicit_command(
            name,
            allowed_root=allowed_root,
        )
    else:
        resolved = shutil.which(name)
    if resolved:
        if IS_WINDOWS and resolved.lower().endswith((".cmd", ".bat")):
            native = _node_package_entrypoint(
                resolved,
                cwd,
                search_cwd=search_cwd,
            )
            if native:
                return [*native, *argv]
        return [resolved, *argv]
    return [fallback, *argv]


# -----------------------------------------------------------------------------
# Detached / hidden process creation
# -----------------------------------------------------------------------------


# Win32 CreationFlags — defined here rather than imported from subprocess
# because CREATE_NO_WINDOW and DETACHED_PROCESS aren't guaranteed to be
# present on stdlib subprocess on older Pythons or non-Windows builds.
_CREATE_NEW_PROCESS_GROUP = 0x00000200
# DETACHED_PROCESS is intentionally NOT part of any flag bundle here — do not
# re-add it.  Two reasons (the recurring console-flash bug #54220 / #56747):
#
# 1. MSDN (Process Creation Flags): CREATE_NO_WINDOW "is ignored if used with
#    either CREATE_NEW_CONSOLE or DETACHED_PROCESS".  Combining them means
#    DETACHED_PROCESS governs and the no-window bit is dead.
# 2. A DETACHED_PROCESS child has NO console at all, so every console-subsystem
#    descendant it ever spawns (git, gh, cmd, node, wmic, powershell, …) must
#    allocate its OWN console — a visible flash per spawn, including spawns
#    inside third-party libraries that no per-call-site CREATE_NO_WINDOW sweep
#    can reach.  A CREATE_NO_WINDOW child instead OWNS a hidden console that
#    all descendants inherit, making "no flashing windows" a property of the
#    one daemon launch.  Root cause isolated + A/B verified on Windows 11 by
#    the desktop backend fix (commit aa2ae36c3f): with per-site hide flags
#    neutered, naive git/gh/cmd spawns don't flash under a hidden-console
#    parent and do flash under a console-less one.
_DETACHED_PROCESS = 0x00000008  # kept for reference; must stay out of bundles
_CREATE_NO_WINDOW = 0x08000000
# Escape any Win32 job object the parent process belongs to. Without this,
# a detached child still inherits its parent's job object membership, and
# when that parent (Electron, Tauri, Windows Terminal, the Desktop GUI's
# bootstrap-installer) dies, the OS tears down the whole job — taking the
# "detached" child with it. Critical for the post-update gateway watcher:
# Electron spawns the Tauri updater inside its own job, the updater spawns
# the watcher subprocess; without BREAKAWAY the watcher dies the instant
# Electron exits, so the gateway never gets respawned after a `hermes
# update` triggered from the GUI. See fix/windows-gateway-reliability.
_CREATE_BREAKAWAY_FROM_JOB = 0x01000000


def windows_detach_flags() -> int:
    """Return Win32 creationflags that detach a child from the parent
    console and process group without leaving it console-less.  0 on
    non-Windows.

    Pair with ``start_new_session=False`` (default) when calling
    subprocess.Popen — on POSIX use ``start_new_session=True`` instead,
    which maps to ``os.setsid()`` in the child.

    Rationale:
    - ``CREATE_NEW_PROCESS_GROUP`` — child has its own process group so
      Ctrl+C in the parent console doesn't propagate.
    - ``CREATE_NO_WINDOW`` — the child gets its own fresh console that is
      never shown.  This both detaches it from the parent's console
      lifetime (closing the launching terminal doesn't CTRL_CLOSE it) AND
      gives every console-subsystem descendant (git, gh, cmd, node, …) a
      console to inherit, so they don't allocate visible flashing ones.
      This deliberately replaces the old ``DETACHED_PROCESS`` approach:
      MSDN specifies CREATE_NO_WINDOW is *ignored* when combined with
      DETACHED_PROCESS, and a truly console-less daemon re-creates the
      per-descendant console-flash bug (#54220/#56747) at every spawn —
      see the note on ``_DETACHED_PROCESS`` above.
    - ``CREATE_BREAKAWAY_FROM_JOB`` — escape any job object the parent is
      in.  Electron (Desktop app) and Tauri (bootstrap installer) wrap
      their children in job objects; without breakaway, those children
      die when the parent process exits even though they have their own
      console.  This was the missing flag that made the post-update
      gateway respawn watcher silently die alongside the Tauri updater
      after the Electron Desktop's update flow finished.

    If a process is in a job that disallows breakaway (rare —
    JOB_OBJECT_LIMIT_BREAKAWAY_OK isn't set), CreateProcess returns
    ERROR_ACCESS_DENIED.  Python surfaces that as ``PermissionError``
    on the ``subprocess.Popen`` call.  Callers in this codebase already
    wrap detached spawns in ``try/except OSError`` and fall back to a
    cmd.exe wrapper, so the breakaway-denied case degrades gracefully
    rather than crashing.
    """
    if not IS_WINDOWS:
        return 0
    return (
        _CREATE_NEW_PROCESS_GROUP
        | _CREATE_NO_WINDOW
        | _CREATE_BREAKAWAY_FROM_JOB
    )


def windows_detach_flags_without_breakaway() -> int:
    """Same as :func:`windows_detach_flags` minus ``CREATE_BREAKAWAY_FROM_JOB``.

    The docstring on :func:`windows_detach_flags` notes that a process in
    a job which disallows breakaway (no ``JOB_OBJECT_LIMIT_BREAKAWAY_OK``)
    will see ``ERROR_ACCESS_DENIED`` from CreateProcess, surfacing as
    ``OSError`` (``PermissionError``) on the ``subprocess.Popen`` call.
    Callers that want to recover — by retrying without the breakaway
    bit — can pair the two helpers symbolically rather than coding the
    ``& ~0x01000000`` magic at every site:

    .. code-block:: python

        try:
            subprocess.Popen(argv, creationflags=windows_detach_flags(), …)
        except OSError:
            subprocess.Popen(
                argv,
                creationflags=windows_detach_flags_without_breakaway(),
                …,
            )

    See ``gateway_windows.py::_spawn_detached`` for the canonical
    implementation of this pattern.  Returns 0 on non-Windows.
    """
    if not IS_WINDOWS:
        return 0
    return _CREATE_NEW_PROCESS_GROUP | _CREATE_NO_WINDOW


def windows_hide_flags() -> int:
    """Return Win32 creationflags that merely hide the child's console
    window without detaching the child.  0 on non-Windows.

    Use for short-lived console apps spawned as part of a larger
    operation (``taskkill``, ``where``, version probes) where we want no
    flash but also want to collect stdout/exit code synchronously.

    The difference from :func:`windows_detach_flags`: no
    ``CREATE_NEW_PROCESS_GROUP`` / ``CREATE_BREAKAWAY_FROM_JOB`` — the
    child stays in the parent's process group and job so Ctrl+C and job
    teardown propagate normally, as a short-lived helper wants.  Stdio
    handles are inherited either way, so ``capture_output=True`` works
    with both bundles.
    """
    if not IS_WINDOWS:
        return 0
    return _CREATE_NO_WINDOW


def suppress_platform_ver_console() -> None:
    """Stub out ``platform._syscmd_ver`` on Windows so it can never flash a
    console window.  No-op on non-Windows.

    CPython's ``platform.win32_ver()`` — reached by ``platform.uname()``,
    ``platform.version()``, and ``platform.platform()`` — unconditionally
    shells out ``cmd /c ver`` via ``subprocess.check_output(..., shell=True)``
    with no ``CREATE_NO_WINDOW``.  From a windowless parent (the pythonw
    gateway and every kanban worker it spawns) that allocates a fresh
    *visible* console: one flashing ``cmd`` window per process, triggered by
    any dependency that merely touches ``platform.uname()`` at import time.

    With ``_syscmd_ver`` stubbed to return its inputs, ``win32_ver()`` hits
    the documented ``ValueError`` fallback and reads the version from
    ``sys.getwindowsversion().platform_version`` — same information, queried
    in-process, no subprocess, no window.  Verified equivalent on
    CPython 3.11 (``platform()`` → ``Windows-10-10.0.xxxxx-SP0`` either way).

    Call early, before heavyweight imports — the flash typically happens
    during a dependency's import, not from Hermes' own code.
    """
    if not IS_WINDOWS:
        return
    try:
        import platform

        if hasattr(platform, "_syscmd_ver"):
            def _quiet_syscmd_ver(system="", release="", version="",
                                  supported_platforms=("win32", "win16", "dos")):
                return system, release, version

            platform._syscmd_ver = _quiet_syscmd_ver
    except Exception:
        # Purely cosmetic hardening — never let it break startup.
        pass


def windows_detach_popen_kwargs() -> dict:
    """Return a dict of Popen kwargs that detach a child on Windows and
    fall back to the POSIX equivalent (``start_new_session=True``) on
    Linux/macOS.

    Usage pattern:

    .. code-block:: python

        subprocess.Popen(
            argv,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            close_fds=True,
            **windows_detach_popen_kwargs(),
        )

    This replaces the unsafe-on-Windows pattern:

    .. code-block:: python

        subprocess.Popen(..., start_new_session=True)

    which silently fails to detach on Windows (the flag is accepted but
    has no effect — the child stays attached to the parent's console
    and dies when the console closes).
    """
    if IS_WINDOWS:
        return {"creationflags": windows_detach_flags()}
    return {"start_new_session": True}


# -----------------------------------------------------------------------------
# Non-interactive git environment (credential-prompt hang guard)
# -----------------------------------------------------------------------------


def noninteractive_git_env(
    base: "Mapping[str, str] | None" = None,
) -> dict[str, str]:
    """Environment for *internal* git invocations that must never prompt.

    Hermes shells out to git from many non-interactive contexts — MCP catalog
    installs, plugin install/update, profile distribution staging, worktree
    base fetches, desktop review-pane fetch/push. When the remote is private,
    misconfigured, or requires auth, git's default behavior is to prompt on
    the inherited terminal (or via an askpass helper), which silently hangs
    the operation until its timeout — or forever at call sites without one.
    Ported from openai/codex#34540 / #34612 ("detach non-interactive
    subprocesses from stdin"): a background tool invocation must fail fast
    with a readable error, not wait for input nobody can type.

    Returns a copy of ``base`` (default ``os.environ``) with:

    * ``GIT_TERMINAL_PROMPT=0`` — git fails with "terminal prompts disabled"
      instead of prompting for credentials.
    * ``GCM_INTERACTIVE=Never`` — Git Credential Manager (the default
      credential helper on Windows installs) never pops its own dialog.

    ``GIT_ASKPASS`` / ``SSH_ASKPASS`` are deliberately left alone: when the
    user has a *working* askpass helper or ssh-agent configured, auth should
    still succeed non-interactively. The env only disables paths that block
    on a human.

    Pair with ``stdin=subprocess.DEVNULL`` so git (and any credential helper
    it spawns) also can't read the parent's inherited stdin.

    This is for internal plumbing calls only — the agent-facing terminal tool
    has its own policy layer and user-visible PTY, where prompting can be
    legitimate.
    """
    env = dict(base if base is not None else os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GCM_INTERACTIVE"] = "Never"
    return env


# -----------------------------------------------------------------------------
# Bounded, fail-open git probing (Windows post-kill deadlock guard)
# -----------------------------------------------------------------------------


def kill_process_tree(proc: "subprocess.Popen") -> None:
    """Best-effort terminate *proc* and its descendants on both platforms.

    ``proc.kill()`` alone only terminates the direct child. On Windows a
    suspended descendant (e.g. ``git.exe``) can survive holding duplicates of the
    captured pipe handles, which keeps the pipes from reaching EOF and leaks two
    reader threads + the process per fired timeout — ``taskkill /T /F`` takes the
    whole tree down so the bounded drain that follows can actually reach EOF.
    On POSIX the same class exists: killing the launcher leaves descendants
    (credential helpers, ``git-remote-https``, hook children) running and
    holding the pipe write ends. Callers spawn the child in its own process
    group (``process_group=0``, Python ≥3.11), so when — and only
    when — the child leads its own group (``pgid == pid``), the entire group is
    signalled with ``os.killpg``. The ownership check means a fallback spawn
    that shares our group can never cause us to kill unrelated processes.
    Ported from openai/codex#36793 ("Terminate timed-out Git process trees");
    generalized for the shell-hook runner via openai/codex#37527
    ("Terminate timed-out hook process trees").

    All failures are swallowed — this is cleanup on an already-failing path, and
    the caller's contract is to fail open. ``kill()`` can raise (access denied,
    already reaped); an unhandled raise here would escape the caller's ``except``
    handler and break that contract. The ``taskkill`` spawn itself cannot
    re-enter the deadlock class it fixes: it captures no pipes (DEVNULL), so its
    own timeout cleanup has no reader threads to join.
    """
    if not IS_WINDOWS:
        # Group-kill first: verify the child actually leads its own process
        # group before signalling it, so we never blast a shared group.
        try:
            import signal as _signal

            pgid = os.getpgid(proc.pid)
            if pgid == proc.pid:
                os.killpg(pgid, _signal.SIGKILL)  # windows-footgun: ok — inside `if not IS_WINDOWS` gate
        except Exception:
            pass
    try:
        proc.kill()
    except OSError:
        pass
    if IS_WINDOWS:
        try:
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                timeout=2,
                check=False,
                creationflags=windows_hide_flags(),
            )
        except Exception:
            pass


def bounded_probe_run(
    argv: Sequence[str],
    *,
    timeout: float,
    errors: str = "replace",
) -> "subprocess.CompletedProcess[str] | None":
    """Deadlock-safe ``subprocess.run(argv, capture_output=True, timeout=...)``
    for fail-open probe call sites. Returns a ``CompletedProcess`` when the
    child finished within *timeout* (any exit code), or ``None`` on spawn
    failure or timeout.

    Why not ``subprocess.run``: on Windows, ``run()``'s post-timeout cleanup
    calls an *unbounded* ``communicate()`` after killing the direct child.
    Killing it can leave a descendant (``git.exe`` under a launcher shim,
    ``conhost.exe`` under wmic/powershell) holding duplicates of the captured
    stdout/stderr handles, so the pipes never reach EOF and the reader-thread
    join blocks forever. The wmic / ``Get-CimInstance Win32_Process`` gateway
    scan hit exactly this during ``hermes update`` on slow-WMI machines
    (#87134); the git probes hit it first (#68609 / #66037).

    The bounded flow: an explicit ``communicate(timeout)``, then on any
    failure a tree-kill (see :func:`kill_process_tree`) plus a bounded 1s
    post-kill drain; if the pipes are still held after that, they're abandoned
    (the orphaned reader threads are daemonic and cost nothing).

    The spawn contract mirrors the ``run`` calls it replaces: PIPE/PIPE/DEVNULL,
    ``text`` with UTF-8 decoding (*errors* configurable — the process scans use
    ``"ignore"``), and the hidden-window ``creationflags`` on Windows only. On
    POSIX the child is placed in its own process group (``process_group=0``,
    Python ≥3.11) so timeout cleanup can take down descendants with the
    launcher instead of orphaning them.
    """
    _popen_kwargs: dict = {"creationflags": windows_hide_flags()} if IS_WINDOWS else {"process_group": 0}
    try:
        proc = subprocess.Popen(
            list(argv),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors=errors,
            **_popen_kwargs,
        )
    except Exception:
        return None
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except Exception:
        # Timeout OR any other communicate() failure (torn-down pipe, decode
        # error): terminate the child + descendants and drain bounded. Leaving
        # it running would leak the same suspended-descendant class this guards.
        kill_process_tree(proc)
        try:
            proc.communicate(timeout=1)
        except Exception:
            pass
        return None
    return subprocess.CompletedProcess(list(argv), proc.returncode, stdout, stderr)


def bounded_git_probe(argv: Sequence[str], *, timeout: float) -> str:
    """Run a short, throwaway ``git`` probe and return stripped stdout, or ``""``
    on ANY failure (nonzero exit, timeout, spawn error, decode error).

    This is the shared, deadlock-safe replacement for
    ``subprocess.run(["git", ...], timeout=...)`` at fail-open probe call sites
    (``tui_gateway.git_probe.run_git``, ``agent.coding_context._git``).

    Why not ``subprocess.run``: on Windows, ``run()``'s post-timeout cleanup
    calls an *unbounded* ``communicate()`` after killing git. Killing the
    PATH-resolved launcher can leave a suspended descendant ``git.exe`` holding
    duplicates of the captured stdout/stderr handles, so the pipes never reach
    EOF and the reader-thread join blocks forever. On the Desktop agent-build
    path (``_start_agent_build → _session_info → branch() → run_git``) that turned
    an optional branch label into ``agent initialization timed out``
    (issues #68609 / #66037).

    The bounded flow: an explicit ``communicate(timeout)``, then on any failure a
    tree-kill (see :func:`_kill_git_process_tree`) plus a bounded 1s post-kill
    drain; if the pipes are still held after that, they're abandoned (the orphaned
    reader threads are daemonic and cost nothing).

    The normal-path spawn contract mirrors the previous ``run`` call byte-for-byte:
    PIPE/PIPE/DEVNULL, ``text`` with UTF-8 ``errors="replace"`` decoding, and the
    hidden-window ``creationflags`` on Windows only. On POSIX the probe is
    additionally placed in its own process group (``process_group=0``,
    Python ≥3.11) so timeout cleanup can take down descendants — credential
    helpers, ``git-remote-https``, hook children — with the launcher instead of
    orphaning them (see :func:`_kill_git_process_tree`; port of
    openai/codex#36793). ``process_group`` only changes which group the child
    belongs to; it does not detach the terminal or alter the fast path.
    """
    result = bounded_probe_run(argv, timeout=timeout)
    if result is None or result.returncode != 0:
        return ""
    return (result.stdout or "").strip()


# Backward-compat alias — existing call sites/tests import the historical name.
_kill_git_process_tree = kill_process_tree
