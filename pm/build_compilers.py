"""Point native source builds under PM's pinned interpreter at tools this host has.

python-build-standalone records the toolchain it was built with in sysconfig:
on Linux ``CC = 'clang -pthread'``, ``CXX = 'clang++ -pthread'``, linkers that
start with the same commands, and an ``AR`` under its deleted ``/tools/llvm``
build directory; on macOS only that archiver is foreign. setuptools compiles
every extension with those values unless the environment overrides them, so a
Debian/Ubuntu/WSL host with only gcc/g++ failed any sdist build with
``No such file or directory: 'clang++'`` (python-olm, via the matrix extra).

Stdlib only: ``pm.environment.classify_uv_failure`` imports this on the
bootstrap runner's pre-3.11 system python.
"""
from __future__ import annotations

from collections.abc import Mapping
import functools
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

# setuptools derives LDSHARED from an overridden CC but never LDCXXSHARED from
# CXX, so each compiler's own linker is rewritten with it.
_COMPILERS = (("CC", "LDSHARED"), ("CXX", "LDCXXSHARED"))
_GCC_FAMILY = {"clang": "gcc", "clang++": "g++"}
_RECORDED = ("CC", "CXX", "LDSHARED", "LDCXXSHARED", "AR")

_MISSING_COMPILER = (
    re.compile(r"No such file or directory: '([^']+)'"),
    re.compile(r"command '([^']+)' failed"),
    re.compile(r"No CMAKE_(?:C|CXX)_COMPILER could be found"),
)
_COMPILER_NAME = re.compile(r"(?:clang(?:\+\+)?|g?cc|g\+\+|c\+\+)(?:-[\d.]+)?")
_COMPILER_REMEDY = ("{what}: install a C/C++ compiler (Debian/Ubuntu/WSL: `sudo apt install build-essential`; "
                    "Fedora: `sudo dnf install gcc gcc-c++`; macOS: `xcode-select --install`) "
                    "or point CC/CXX at one, then retry")


@functools.lru_cache(maxsize=None)  # noqa: UP033 -- functools.cache is 3.9+; the bootstrap runner can be older
def _recorded_toolchain(python: str) -> dict[str, str]:
    code = ("import json, sysconfig; "
            f"print(json.dumps({{k: sysconfig.get_config_var(k) or '' for k in {_RECORDED!r}}}))")
    try:
        result = subprocess.run([python, "-I", "-c", code], capture_output=True, text=True,
                                timeout=30, check=False)
        return json.loads(result.stdout) if result.returncode == 0 else {}
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return {}


def _linux_defaults(recorded: Mapping[str, str], env: Mapping[str, str]) -> dict[str, str]:
    search = env.get("PATH")
    defaults: dict[str, str] = {}
    for compiler_var, linker_var in _COMPILERS:
        command = recorded.get(compiler_var, "")
        program = command.split(" ", 1)[0]
        substitute = _GCC_FAMILY.get(program)
        if (compiler_var in env or substitute is None or shutil.which(program, path=search)
                or not shutil.which(substitute, path=search)):
            continue
        defaults[compiler_var] = substitute + command[len(program):]
        linker = recorded.get(linker_var, "")
        if linker_var not in env and linker.split(" ", 1)[0] == program:
            defaults[linker_var] = substitute + linker[len(program):]
    archiver = recorded.get("AR", "")
    if archiver and not shutil.which(archiver, path=search) and shutil.which("ar", path=search):
        defaults["AR"] = "ar"
    return defaults


def pinned_build_defaults(python: Path, env: Mapping[str, str]) -> dict[str, str]:
    """Defaults for build variables the pinned interpreter records but this host lacks.

    Callers ``setdefault`` them, so a CC/CXX/AR the user set always wins. A recorded
    compiler with no gcc-family equivalent on PATH is left alone; the build then
    fails and ``missing_compiler_remedy`` names the fix.
    """
    if sys.platform == "darwin":
        return {"AR": "/usr/bin/ar"}
    if sys.platform != "linux":
        return {}
    return _linux_defaults(_recorded_toolchain(str(python)), env)


def missing_compiler_remedy(output: str) -> str:
    """The install hint for a build that failed because no compiler exists, else ''."""
    for pattern in _MISSING_COMPILER:
        for match in pattern.finditer(output):
            if not match.groups():
                return _COMPILER_REMEDY.format(what="this package compiles native code and no compiler was found")
            if _COMPILER_NAME.fullmatch(Path(match.group(1)).name):
                return _COMPILER_REMEDY.format(what=f"the build needs `{match.group(1)}`, which is not on PATH")
    return ""
