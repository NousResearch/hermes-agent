"""Windows subprocess compatibility helpers.

* ``["npm", ...]`` — on Windows ``npm`` is ``npm.cmd``, a batch shim; ``Popen`` fails with
  WinError 193 because CreateProcessW can't run a ``.cmd`` without ``shell=True``/PATHEXT.
* ``start_new_session=True`` — POSIX ``os.setsid()`` detach; silently ignored on Windows, whose
  equivalent is the ``CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW`` creationflags bundle.
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

from hermes_constants import find_node_executable

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
    "NO_DRIVER_DIFF_FLAGS",
    "pid_is_hermes",
]

# Flags that neutralize *attribute-scoped* diff drivers on any diff-rendering git command. A
# malicious repo can name a driver in ``.gitattributes`` (``* diff=evil``) and point it at an
# arbitrary program via ``[diff "evil"] command=/textconv=`` in ``.git/config``; because the
# attacker chooses the name, ``GIT_CONFIG_KEY`` overrides in ``noninteractive_git_env`` cannot
# enumerate it — only these flags do. ``--no-ext-diff`` kills ``command=``; ``--no-textconv`` kills
# ``textconv=``; each alone leaves the other live. Smudge/clean filters are neutralized by the env
# layer's ``core.hooksPath`` + running against the index without checkout.
NO_DRIVER_DIFF_FLAGS = ("--no-ext-diff", "--no-textconv")

# Only these subcommands accept ``NO_DRIVER_DIFF_FLAGS`` — ``status`` and friends reject them
# (``unknown option``), so the helper gates on this set rather than blanket-prepending.
_DIFF_RENDERING_SUBCOMMANDS = frozenset({"diff", "show", "log", "blame"})

# Options that consume the FOLLOWING token, so that value is never mistaken for the subcommand
# (``-C diff`` is a path; ``-c diff=x`` is a config pair).
_GIT_VALUE_OPTS = {"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--exec-path"}


def harden_git_argv(args: Sequence[str]) -> list[str]:
    """Copy of subcommand-first git *args* (no leading ``"git"``) with :data:`NO_DRIVER_DIFF_FLAGS`
    inserted right after a diff-rendering subcommand; other subcommands are returned unchanged.

    Pair with :func:`noninteractive_git_env`: the env layer disables fsmonitor/hooks/pager/editor/
    credential sinks, this closes the one class (attacker-named attribute drivers) env cannot reach.
    """
    out = list(args)
    i = 0
    while i < len(out):
        tok = out[i]
        if tok in _GIT_VALUE_OPTS:
            i += 2
            continue
        if tok.startswith("-"):
            i += 1
            continue
        if tok in _DIFF_RENDERING_SUBCOMMANDS:
            return out[: i + 1] + list(NO_DRIVER_DIFF_FLAGS) + out[i + 1 :]
        return out  # first non-option token is a non-diff subcommand
    return out


IS_WINDOWS = sys.platform == "win32"

# Private launcher-to-child metadata. This is diagnostic state, not user config.
_WINDOWS_GATEWAY_BREAKAWAY_ENV = "_HERMES_GATEWAY_BREAKAWAY"


def split_command_line(line: str) -> list[str]:
    """Split a user-supplied command line into tokens, Windows-safely.

    ``shlex.split`` (posix=True) treats every backslash as an escape, mangling Windows paths. On
    Windows use ``posix=False`` and strip one layer of matching quotes per token; on POSIX this is
    exactly ``shlex.split``. Raises ValueError on unbalanced quotes.

    ``shlex.split(line)`` (posix=True) treats every backslash as an escape character, so Windows paths are
    silently mangled: ``C:\\Users\\me\\out.txt`` becomes ``C:Usersmeout.txt`` — no error, just a wrong path
    that then "succeeds" against a mangled relative filename (#83934) or makes a valid hook script report
    "not executable" (#78293).
    """
    import shlex

    if not IS_WINDOWS:
        return shlex.split(line)
    out: list[str] = []
    for tok in shlex.split(line, posix=False):
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
        node = find_node_executable("node")
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


# Win32 CreationFlags — defined here because CREATE_NO_WINDOW / DETACHED_PROCESS aren't guaranteed
# to exist on stdlib subprocess for older Pythons or non-Windows builds.
_CREATE_NEW_PROCESS_GROUP = 0x00000200
# DETACHED_PROCESS (0x00000008) is intentionally NOT part of any flag bundle — do not re-add it
# (the recurring console-flash bug #54220 / #56747): (1) MSDN: CREATE_NO_WINDOW "is ignored if used with either
# CREATE_NEW_CONSOLE or DETACHED_PROCESS"; (2) a DETACHED_PROCESS child has NO console, so every
# console-subsystem descendant (git, gh, cmd, node, powershell, …) allocates its own — a visible
# flash per spawn, including inside third-party libraries no per-site sweep can reach. A
# CREATE_NO_WINDOW child instead OWNS a hidden console all descendants inherit (A/B verified on
# Windows 11 by the desktop backend fix, commit aa2ae36c3f: with per-site hide flags neutered,
# naive git/gh/cmd spawns don't flash under a hidden-console parent and do under a console-less one).
# 1. Combining them means DETACHED_PROCESS governs and the no-window bit is dead. 2. See #54220, #56747.
_CREATE_NO_WINDOW = 0x08000000
# Escape any Win32 job object the parent belongs to. Without this a detached child inherits the
# parent's job, and when that parent (Electron, Tauri, Windows Terminal, the Desktop bootstrap
# installer) dies the OS tears down the whole job — taking the "detached" child with it. Critical
# for the post-update gateway watcher spawned from inside Electron's job.
_CREATE_BREAKAWAY_FROM_JOB = 0x01000000


def windows_detach_flags() -> int:
    """Win32 creationflags detaching a child from the parent console/group; 0 elsewhere.

    Pair with the default ``start_new_session=False`` (POSIX uses ``start_new_session=True``).
    CREATE_NEW_PROCESS_GROUP stops Ctrl+C propagating; CREATE_NO_WINDOW gives the child a hidden
    console descendants (git, gh, cmd, node, …) inherit so they don't flash — deliberately replacing
    the old DETACHED_PROCESS approach, which re-created the per-descendant console-flash bug
    (#54220/#56747) at every spawn; CREATE_BREAKAWAY_FROM_JOB escapes Electron/Tauri job objects. A
    job that forbids breakaway yields PermissionError from Popen — callers catch OSError and fall
    back to :func:`windows_detach_flags_without_breakaway`.

    Rationale: This both detaches it from the parent's console lifetime (closing the launching terminal
    doesn't CTRL_CLOSE it) AND gives every console-subsystem descendant (git, gh, cmd, node, …) a console to
    inherit, so they don't allocate visible flashing ones. This deliberately replaces the old
    ``DETACHED_PROCESS`` approach: MSDN specifies CREATE_NO_WINDOW is *ignored* when combined with
    DETACHED_PROCESS, and a truly console-less daemon re-creates the per-descendant console-flash bug
    (#54220/#56747) at every spawn — see the note on ``_DETACHED_PROCESS`` above. Electron (Desktop app) and
    Tauri (bootstrap installer) wrap their children in job objects; without breakaway, those children die
    when the parent process exits even though they have their own console. This was the missing flag that
    made the post-update gateway respawn watcher silently die alongside the Tauri updater after the Electron
    Desktop's update flow finished.
    """
    if not IS_WINDOWS:
        return 0
    return _CREATE_NEW_PROCESS_GROUP | _CREATE_NO_WINDOW | _CREATE_BREAKAWAY_FROM_JOB


def windows_detach_flags_without_breakaway() -> int:
    """:func:`windows_detach_flags` minus ``CREATE_BREAKAWAY_FROM_JOB``; 0 on non-Windows."""
    if not IS_WINDOWS:
        return 0
    return _CREATE_NEW_PROCESS_GROUP | _CREATE_NO_WINDOW


def windows_hide_flags() -> int:
    """Win32 creationflags hiding the child's console without detaching it; 0 elsewhere.

    For short-lived synchronous helpers (``taskkill``, ``where``, version probes): no flash, but the
    child stays in the parent's process group and job so Ctrl+C and job teardown still propagate.
    Stdio is inherited, so ``capture_output=True`` works.
    """
    return _CREATE_NO_WINDOW if IS_WINDOWS else 0


def suppress_platform_ver_console() -> None:
    """Stub ``platform._syscmd_ver`` on Windows so it never flashes a console. No-op elsewhere.

    ``platform.win32_ver()`` shells out ``cmd /c ver`` without CREATE_NO_WINDOW, so a windowless
    parent (pythonw gateway, kanban workers) flashes a cmd window whenever a dependency touches
    ``platform.uname()`` at import. With the stub, ``win32_ver()`` takes its documented fallback to
    ``sys.getwindowsversion()`` — same data, in-process. Call before heavy imports.
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
        pass  # Purely cosmetic hardening — never let it break startup.


def windows_detach_popen_kwargs() -> dict:
    """Popen kwargs detaching a child on Windows, or ``start_new_session=True`` on POSIX.

    Bare ``start_new_session=True`` is accepted but has no effect on Windows: the child stays
    attached to the parent console and dies when it closes.
    """
    if IS_WINDOWS:
        return {"creationflags": windows_detach_flags()}
    return {"start_new_session": True}


# GIT_CONFIG_KEY_n/VALUE_n overrides for internal git children: no credential/askpass prompts, no
# repo-configured fsmonitor/hooks/pager/editor/external-diff programs.
_GIT_CONFIG_INJECT_PREFIXES = ("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_")
_GIT_CONFIG_OVERRIDES = {
    "credential.helper": "",
    "core.askPass": "",
    "core.fsmonitor": "false",
    "core.untrackedCache": "false",
    "core.hooksPath": os.devnull,
    "core.pager": "cat",
    "core.editor": "true",
    "sequence.editor": "true",
    "diff.external": "",
    # ssh itself bypasses stdin=DEVNULL/GIT_TERMINAL_PROMPT and opens /dev/tty directly — an
    # unknown host key (or password auth) prompts there and steals the caller's terminal (#104591).
    # BatchMode makes ssh fail instead of prompting; a working ssh-agent still succeeds. Injected
    # at the config layer so an explicit user GIT_SSH_COMMAND (env) still takes precedence.
    "core.sshCommand": "ssh -o BatchMode=yes",
}


def _safe_directory_cache_key(env: "Mapping[str, str]") -> tuple:
    """Everything that decides which files ``git config --system/--global`` reads, plus the
    global candidates' mtimes so an edit to ``~/.gitconfig`` is picked up without a restart."""
    home = env.get("HOME", "")
    xdg = env.get("XDG_CONFIG_HOME") or os.path.join(home, ".config")
    candidates = (
        env.get("GIT_CONFIG_SYSTEM") or "/etc/gitconfig",
        env.get("GIT_CONFIG_GLOBAL") or os.path.join(home, ".gitconfig"),
        os.path.join(xdg, "git", "config"),
    )
    stamps = []
    for path in candidates:
        try:
            stamps.append(os.stat(path).st_mtime_ns)
        except OSError:
            stamps.append(None)
    return (
        env.get("GIT_CONFIG_GLOBAL"), env.get("GIT_CONFIG_SYSTEM"), env.get("GIT_CONFIG_NOSYSTEM"),
        home, env.get("XDG_CONFIG_HOME"), env.get("PATH"), *stamps,
    )


_safe_directory_cache: dict[tuple, list[str]] = {}


def _user_safe_directories(base_env: "Mapping[str, str]") -> list[str]:
    """The user's configured ``safe.directory`` values, in git's own effective order.

    Read with ``git config -z --get-all`` under *base_env* (the caller's untouched environment) so
    an explicit ``GIT_CONFIG_GLOBAL``/``GIT_CONFIG_SYSTEM`` still points at the file the user means.
    Best-effort: any failure (git missing, malformed config, timeout) yields no entries and leaves
    the caller exactly as it behaved before. Memoised per process on the inputs that select the
    config files (and the global file's mtime): ``noninteractive_git_env()`` runs on every internal
    git call, including the startup banner probe, and two ``git config`` children per call is
    ~10 ms against ~0.2 ms for the rest of the function.

    ``safe.directory`` is an *ordered* multi-valued setting and an empty value resets every entry
    seen so far, so a user can revoke a system-wide ``safe.directory=*`` and then name only the
    repositories they actually trust. Order and empty resets are therefore policy, not formatting:
    scopes are read lowest-precedence first (system, then global) and every value is preserved
    verbatim -- no de-duplication (it is a sequence, not a set) and no dropping of the reset
    marker, either of which would resurrect a revoked wildcard and widen trust. ``-z`` keeps a
    value containing whitespace or a newline as the single entry git reads it as.
    """
    cache_key = _safe_directory_cache_key(base_env)
    cached = _safe_directory_cache.get(cache_key)
    if cached is not None:
        return list(cached)
    env = dict(base_env)
    # --get-all itself must not be derailed by ambient injection or an interactive prompt.
    for key in list(env):
        if key == "GIT_CONFIG_PARAMETERS" or key.startswith(_GIT_CONFIG_INJECT_PREFIXES):
            env.pop(key, None)
    env.pop("GIT_CONFIG_COUNT", None)
    env["GIT_TERMINAL_PROMPT"] = "0"
    values: list[str] = []
    for scope in ("--system", "--global"):
        try:
            proc = subprocess.run(
                ["git", "config", scope, "-z", "--get-all", "safe.directory"],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=5, stdin=subprocess.DEVNULL, env=env, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if proc.returncode != 0:
            continue
        # -z terminates every value with NUL, so the trailing split field is always empty and is
        # not a config entry; interior empty fields are real reset markers and must survive.
        records = proc.stdout.split("\0")
        if records and records[-1] == "":
            records.pop()
        values.extend(records)
    _safe_directory_cache[cache_key] = list(values)
    return values


def noninteractive_git_env(base: "Mapping[str, str] | None" = None) -> dict[str, str]:
    """Environment for *internal* git invocations that must never prompt.

    Copy of ``base`` (default ``os.environ``) with ``GIT_TERMINAL_PROMPT=0`` (fail instead of
    prompting), ``GCM_INTERACTIVE=Never`` (no Git Credential Manager dialog), and isolated git
    config: inherited ``GIT_CONFIG_*`` injection, global/system config, pagers, editors, fsmonitor,
    external diff and hooks are all disabled so a user's repo/global config cannot hang or mutate
    Hermes's plumbing calls. ``core.sshCommand`` is pinned to ``ssh -o BatchMode=yes`` so the ssh
    child of a fetch/ls-remote fails instead of prompting — ssh bypasses ``stdin=DEVNULL`` and
    opens ``/dev/tty`` directly (#104591); an agent-authenticated ssh still succeeds, and an
    explicit user ``GIT_SSH_COMMAND`` env var still takes precedence over this config-layer pin.
    ``GIT_ASKPASS``/``SSH_ASKPASS`` env vars are left alone, but OpenSSH BatchMode disables
    passphrase/password prompts, including SSH askpass. Usable keys and ssh-agent authentication
    still work; Git's own working askpass helper is unaffected. Pair with
    ``stdin=subprocess.DEVNULL``. Internal plumbing only — the agent-facing terminal tool has its
    own policy layer and visible PTY.

    Hermes shells out to git from many non-interactive contexts — MCP catalog installs, plugin
    install/update, profile distribution staging, worktree base fetches, desktop review-pane fetch/push.
    When the remote is private, misconfigured, or requires auth, git's default behavior is to prompt on the
    inherited terminal (or via an askpass helper), which silently hangs the operation until its timeout — or
    forever at call sites without one. Ported from openai/codex#34540 / #34612 ("detach non-interactive
    subprocesses from stdin"): a background tool invocation must fail fast with a readable error, not wait
    for input nobody can type.
    """
    env = dict(base if base is not None else os.environ)
    # Captured before the isolation below rewrites GIT_CONFIG_GLOBAL/SYSTEM to /dev/null --
    # reading after that point would resolve the user's config to an empty file.
    safe_directories = _user_safe_directories(base if base is not None else os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GCM_INTERACTIVE"] = "Never"
    # Drop caller-supplied config injection; the GIT_CONFIG_COUNT block is rebuilt below so
    # ambient -c values cannot re-enable pagers, hooks, fsmonitor, editors or credential prompts.
    for key in list(env):
        if key == "GIT_CONFIG_PARAMETERS" or key.startswith(_GIT_CONFIG_INJECT_PREFIXES):
            env.pop(key, None)
    env.pop("GIT_CONFIG_COUNT", None)
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_CONFIG_SYSTEM"] = os.devnull
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env["GIT_PAGER"] = "cat"
    env["PAGER"] = "cat"
    env["GIT_EDITOR"] = "true"
    overrides = list(_GIT_CONFIG_OVERRIDES.items())
    # safe.directory is honoured ONLY from global/system config (git rejects it from repo-level
    # config so a hostile repo cannot self-authorise), and both are blanked just above. Without
    # re-injection every internal git call fails "detected dubious ownership" on any repo whose
    # st_uid != geteuid() -- NFS/CIFS mounts without idmapping, shared checkouts, containers with
    # a remapped uid -- even though the user's own `git config --global --add safe.directory` is
    # correctly set and their interactive git works fine. Carried over the GIT_CONFIG_KEY_n
    # channel, which survives GIT_CONFIG_GLOBAL=/dev/null. Read-only and non-widening: the values
    # are replayed in git's own effective order, empty reset markers included (see
    # _user_safe_directories), so a global reset still revokes a system-wide wildcard exactly as it
    # does for the user's interactive git. Appended last, but the hardening overrides above are
    # distinct keys, so they are unaffected by ordering within safe.directory.
    overrides.extend(("safe.directory", value) for value in safe_directories)
    env["GIT_CONFIG_COUNT"] = str(len(overrides))
    for idx, (key, value) in enumerate(overrides):
        env[f"GIT_CONFIG_KEY_{idx}"] = key
        env[f"GIT_CONFIG_VALUE_{idx}"] = value
    return env


def _process_start_time(pid: int) -> int | None:
    """The repository's stable process-start fingerprint, if available."""
    try:
        from gateway.status import get_process_start_time

        return get_process_start_time(pid)
    except Exception:
        return None


def _text_names_hermes(text: str) -> bool:
    r"""True when *text* names Hermes at a path-segment / token boundary.

    A bare ``"hermes" in text`` substring test would also match unrelated processes whose paths
    merely contain the letters (``...\shermesa\...``) — the false-positive class this prevents.
    """
    return any(token.startswith(("hermes", ".hermes"))
               for token in re.split(r"[\\/\s=,;\"']+", text.lower()))


def _process_command_is_hermes(pid: int) -> bool:
    """Best-effort check that *pid* currently runs Hermes code."""
    try:
        import psutil

        process = psutil.Process(pid)
        command = " ".join(process.cmdline() or [])
        executable = process.exe() or ""
        return _text_names_hermes(f"{command} {executable}")
    except Exception:
        return False


def pid_is_hermes(pid: int, *, expected_start_time: int | None = None) -> bool:
    """Whether it is safe to use ``taskkill`` for *pid*.

    The PID must be valid, currently exist, and identify a Hermes process. When the caller captured
    a start-time fingerprint before the destructive action, the live process must still have the
    same ``(pid, start_time)`` identity. Any ambiguity fails closed.
    """
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    if not IS_WINDOWS:
        if expected_start_time is None:
            return True
        try:
            return _process_start_time(pid) == expected_start_time
        except Exception:
            return False
    try:
        current_start_time = _process_start_time(pid)
    except Exception:
        return False
    if current_start_time is None:
        return False
    if expected_start_time is not None and current_start_time != expected_start_time:
        return False
    try:
        return _process_command_is_hermes(pid)
    except Exception:
        return False


def kill_process_tree(proc: "subprocess.Popen") -> None:
    """Best-effort terminate *proc* and its descendants on both platforms; never raises.

    ``proc.kill()`` alone only terminates the direct child. This is cleanup on an already-failing
    path whose contract is to fail open, so every failure (access denied, already reaped) is
    swallowed rather than escaping the caller's ``except``.

    On Windows a suspended descendant (e.g. ``git.exe``) can survive holding duplicates of the captured pipe
    handles, which keeps the pipes from reaching EOF and leaks two reader threads + the process per fired
    timeout — ``taskkill /T /F`` takes the whole tree down so the bounded drain that follows can actually
    reach EOF. On POSIX the same class exists: killing the launcher leaves descendants (credential helpers,
    ``git-remote-https``, hook children) running and holding the pipe write ends. Callers spawn the child in
    its own process group (``process_group=0``, Python ≥3.11), so when — and only when — the child leads its
    own group (``pgid == pid``), the entire group is signalled with ``os.killpg``. The ownership check means
    a fallback spawn that shares our group can never cause us to kill unrelated processes. Ported from
    openai/codex#36793 ("Terminate timed-out Git process trees"); generalized for the shell-hook runner via
    openai/codex#37527 ("Terminate timed-out hook process trees").
    """
    try:
        from agent.deadline import kill_process_tree as _deadline_kill_tree

        _deadline_kill_tree(proc.pid)
    except Exception:
        _legacy_kill_process_tree(proc)
        return
    # Ensure Popen's own bookkeeping sees the exit so communicate()/wait() cannot hang.
    try:
        proc.kill()
    except OSError:
        pass


def _legacy_kill_process_tree(proc: "subprocess.Popen") -> None:
    """Local tree-kill fallback when agent.deadline is unavailable (partial install, cycle)."""
    if not IS_WINDOWS:
        # Verify the child leads its own process group before signalling, never a shared group.
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
        # No identity guard on purpose: *proc* is our own retained Popen handle, so the PID cannot
        # be recycled while we hold it. The fail-closed ``pid_is_hermes`` guard is for BARE pids.
        try:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           stdin=subprocess.DEVNULL, timeout=2, check=False,
                           creationflags=windows_hide_flags())
        except Exception:
            pass


def bounded_probe_run(
    argv: Sequence[str], *, timeout: float, errors: str = "replace",
    env: "Mapping[str, str] | None" = None,
) -> "subprocess.CompletedProcess[str] | None":
    """Deadlock-safe ``subprocess.run(argv, capture_output=True, timeout=…)`` for fail-open probes.

    Returns a ``CompletedProcess`` when the child finished within *timeout* (any exit code), or
    ``None`` on spawn failure or timeout.

    Why not ``subprocess.run``: on Windows, ``run()``'s post-timeout cleanup calls an *unbounded*
    ``communicate()`` after killing the direct child. Killing it can leave a descendant (``git.exe`` under a
    launcher shim, ``conhost.exe`` under wmic/powershell) holding duplicates of the captured stdout/stderr
    handles, so the pipes never reach EOF and the reader-thread join blocks forever. The wmic /
    ``Get-CimInstance Win32_Process`` gateway scan hit exactly this during ``hermes update`` on slow-WMI
    machines (#87134); the git probes hit it first (#68609 / #66037).
    """
    _popen_kwargs: dict = {"creationflags": windows_hide_flags()} if IS_WINDOWS else {"process_group": 0}
    try:
        proc = subprocess.Popen(
            list(argv), stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
            text=True, encoding="utf-8", errors=errors,
            env=dict(env) if env is not None else None, **_popen_kwargs)
    except Exception:
        return None
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except Exception:
        # Timeout OR any other communicate() failure (torn-down pipe, decode error): tree-kill and
        # drain bounded — leaving it running would leak the suspended-descendant class this guards.
        kill_process_tree(proc)
        try:
            proc.communicate(timeout=1)
        except Exception:
            pass
        return None
    return subprocess.CompletedProcess(list(argv), proc.returncode, stdout, stderr)


def bounded_git_probe(argv: Sequence[str], *, timeout: float) -> str:
    """Run a short ``git`` probe and return stripped stdout, or ``""`` on ANY failure.

    On Windows ``run()``'s post-timeout cleanup calls an unbounded ``communicate()``; a suspended
    descendant git.exe holding the pipe handles then blocks forever. Here: bounded ``communicate``,
    tree-kill plus a 1s drain, then abandon the pipes; on POSIX the probe gets its own process
    group so cleanup also takes down credential/remote helpers.

    Security (GHSA-7x36-8jrh-v4pw): these probes run automatically against whatever directory the
    session sits in, before any tool call or trust prompt, and an index refresh executes the
    repo-configured ``core.fsmonitor`` program. Every probe therefore runs under
    :func:`noninteractive_git_env`; diff-rendering callers additionally pass
    :data:`NO_DRIVER_DIFF_FLAGS` (attribute-scoped drivers can't be disabled via env).

    Killing the PATH-resolved launcher can leave a suspended descendant ``git.exe`` holding duplicates of
    the captured stdout/stderr handles, so the pipes never reach EOF and the reader-thread join blocks
    forever. On the Desktop agent-build path (``_start_agent_build → _session_info → branch() → run_git``)
    that turned an optional branch label into ``agent initialization timed out`` (issues #68609 / #66037).
    The normal-path spawn contract mirrors the previous ``run`` call byte-for-byte: PIPE/PIPE/DEVNULL,
    ``text`` with UTF-8 ``errors="replace"`` decoding, and the hidden-window ``creationflags`` on Windows
    only. On POSIX the probe is additionally placed in its own process group (``process_group=0``, Python
    ≥3.11) so timeout cleanup can take down descendants — credential helpers, ``git-remote-https``, hook
    children — with the launcher instead of orphaning them (see :func:`kill_process_tree`; port of
    openai/codex#36793). ``process_group`` only changes which group the child belongs to; it does not detach
    the terminal or alter the fast path.
    """
    result = bounded_probe_run(argv, timeout=timeout, env=noninteractive_git_env())
    if result is None or result.returncode != 0:
        return ""
    return (result.stdout or "").strip()
