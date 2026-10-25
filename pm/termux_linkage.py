"""Link on-device extensions to libpython before validation runs (bionic).

bionic's ``dlopen`` resolves a symbol only inside an object's own DT_NEEDED
closure, unlike glibc, which also searches the host program. Extensions
built or shipped without libpython recorded there fail to import with
``cannot locate symbol "PyExc_Warning"``. The wheelhouse build repairs this
(``scripts/termux/build_wheels.py``), but an on-device ``uv sync`` installs
straight into the venv, so ``PythonEnvironment`` runs the same repair itself
— otherwise startup validation discards the generation it just validated.

Self-contained on purpose: a generation's workspace snapshot carries the
``pm`` package roots only, never ``scripts/``.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path


def libpython() -> Path:
    return (
        Path(sys.base_prefix) / "lib" / str(sysconfig.get_config_var("LDLIBRARY") or "")
    )


def _elf_output(argv: list[str]) -> str:
    return subprocess.check_output(argv, text=True)


def _elf_patch(argv: list[str]) -> None:
    subprocess.check_call(argv)


def _needs_libpython(extension: Path, reader: str) -> bool:
    symbols = _elf_output([reader, "--dyn-syms", "--wide", str(extension)])
    return any(
        " UND " in line and line.split()[-1].startswith(("Py", "_Py"))
        for line in symbols.splitlines()
    )


def _link(extension: Path, reader: str, patcher: str, soname: str) -> int:
    """Patch one extension; the caller decides what survives a failure."""
    if not _needs_libpython(extension, reader):
        return 0
    needed = _elf_output([patcher, "--print-needed", str(extension)]).splitlines()
    foreign = [
        name for name in needed if name.startswith("libpython") and name != soname
    ]
    if foreign:
        # A different interpreter is an ABI mismatch patchelf cannot fix; leave
        # it for startup validation to report instead of a broken closure.
        return 0
    if soname in needed:
        return 0
    _elf_patch([patcher, "--add-needed", soname, str(extension)])
    return 1


def repair(venv: Path, *, library: Path | None = None) -> int:
    """Add libpython to DT_NEEDED of every extension in *venv* that needs it.

    Returns the number of extensions patched. A no-op — never an error — on
    non-bionic hosts, without the ELF tools, or without a SONAME-bearing
    libpython: the environment is already built and validation still guards it.
    """
    if sys.platform != "android":
        return 0
    reader = shutil.which("llvm-readelf") or shutil.which("readelf")
    patcher = shutil.which("patchelf")
    if reader is None or patcher is None:
        return 0
    library = library if library is not None else libpython()
    if not library.is_file():
        return 0
    soname = _elf_output([patcher, "--print-soname", str(library)]).strip()
    if not soname:
        return 0
    patched = 0
    for site in sorted(venv.glob("lib/python*/site-packages")):
        for extension in sorted(site.rglob("*.so")):
            try:
                patched += _link(extension, reader, patcher, soname)
            except (subprocess.CalledProcessError, OSError):
                # One unreadable member must not forfeit the rest of the sweep.
                continue
    return patched
