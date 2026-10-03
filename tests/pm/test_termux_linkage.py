"""pm.termux_linkage: the on-device bionic libpython repair."""

from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

import pytest

from pm import termux_linkage
from pm.environment import InstallError, PythonEnvironment

SONAME = "libpython3.14.so"
UNDEFINED_PY = (
    "   1: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT  UND PyExc_Warning\n"
)


def _extension(root: Path, rel: str) -> Path:
    path = root / "lib" / "python3.14" / "site-packages" / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


class FakeElf:
    """Reads per extension; one shared soname for the library itself."""

    def __init__(self, monkeypatch, *, symbols="", needed="", soname=SONAME):
        self.symbols = {None: symbols}
        self.needed = {None: needed}
        self.soname = soname
        self.patches = []
        self.raise_on: set[str] = set()
        monkeypatch.setattr(
            termux_linkage.shutil,
            "which",
            lambda name: {"llvm-readelf": "llvm-readelf", "patchelf": "patchelf"}.get(
                name
            ),
        )
        monkeypatch.setattr(termux_linkage, "_elf_output", self._output)
        monkeypatch.setattr(termux_linkage, "_elf_patch", self._patch)

    def for_extension(self, path: Path, *, symbols="", needed=""):
        self.symbols[str(path)] = symbols
        self.needed[str(path)] = needed

    def _output(self, argv):
        if str(Path(argv[-1])) in self.raise_on:
            raise subprocess.CalledProcessError(1, argv)
        if argv[1] == "--dyn-syms":
            return self.symbols[str(Path(argv[-1]))]
        if argv[1] == "--print-needed":
            return self.needed[str(Path(argv[-1]))]
        assert argv[1] == "--print-soname", argv
        return self.soname + "\n"

    def _patch(self, argv):
        self.patches.append(argv)


@pytest.fixture
def android(monkeypatch):
    monkeypatch.setattr(sys, "platform", "android")


@pytest.fixture
def library(tmp_path):
    path = tmp_path / "libpython3.14.so"
    path.write_bytes(b"")
    return path


def test_non_bionic_host_is_a_no_op(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        termux_linkage.shutil, "which", lambda name: pytest.fail("no tools needed")
    )
    assert termux_linkage.repair(tmp_path) == 0


def test_missing_elf_tools_is_a_no_op(android, monkeypatch, tmp_path):
    monkeypatch.setattr(termux_linkage.shutil, "which", lambda name: None)
    assert termux_linkage.repair(tmp_path, library=tmp_path / "libpython.so") == 0


def test_missing_libpython_is_a_no_op(android, monkeypatch, tmp_path):
    FakeElf(monkeypatch)
    assert termux_linkage.repair(tmp_path, library=tmp_path / "absent.so") == 0


def test_sonameless_libpython_is_a_no_op(android, monkeypatch, tmp_path, library):
    elf = FakeElf(monkeypatch, soname="")
    extension = _extension(tmp_path, "pkg/_.so")
    elf.for_extension(extension, symbols=UNDEFINED_PY, needed="libc.so\n")
    assert termux_linkage.repair(tmp_path, library=library) == 0
    assert elf.patches == []


def test_links_extension_missing_libpython(android, monkeypatch, tmp_path, library):
    elf = FakeElf(monkeypatch)
    extension = _extension(tmp_path, "cryptography/hazmat/bindings/_rust.abi3.so")
    elf.for_extension(extension, symbols=UNDEFINED_PY, needed="libc.so\nlibdl.so\n")
    assert termux_linkage.repair(tmp_path, library=library) == 1
    assert elf.patches == [["patchelf", "--add-needed", SONAME, str(extension)]]


def test_already_linked_extension_is_untouched(android, monkeypatch, tmp_path, library):
    elf = FakeElf(monkeypatch)
    extension = _extension(tmp_path, "pkg/_.so")
    elf.for_extension(extension, symbols=UNDEFINED_PY, needed=f"libc.so\n{SONAME}\n")
    assert termux_linkage.repair(tmp_path, library=library) == 0
    assert elf.patches == []


def test_foreign_interpreter_extension_is_skipped(
    android, monkeypatch, tmp_path, library
):
    elf = FakeElf(monkeypatch)
    extension = _extension(tmp_path, "pkg/_.so")
    elf.for_extension(extension, symbols=UNDEFINED_PY, needed="libpython3.13.so\n")
    assert termux_linkage.repair(tmp_path, library=library) == 0
    assert elf.patches == []


def test_extension_without_python_symbols_is_skipped(
    android, monkeypatch, tmp_path, library
):
    elf = FakeElf(monkeypatch)
    extension = _extension(tmp_path, "pkg/_.so")
    elf.for_extension(extension, symbols="   1: ... UND evpow\n", needed="libc.so\n")
    assert termux_linkage.repair(tmp_path, library=library) == 0
    assert elf.patches == []


def test_broken_member_does_not_stop_the_sweep(android, monkeypatch, tmp_path, library):
    elf = FakeElf(monkeypatch)
    broken = _extension(tmp_path, "pkg/_broken.so")
    healthy = _extension(tmp_path, "pkg/_healthy.so")
    elf.for_extension(broken, symbols=UNDEFINED_PY, needed="libc.so\n")
    elf.for_extension(healthy, symbols=UNDEFINED_PY, needed="libc.so\n")
    elf.raise_on.add(str(broken))
    assert termux_linkage.repair(tmp_path, library=library) == 1
    assert elf.patches == [["patchelf", "--add-needed", SONAME, str(healthy)]]


def test_files_outside_site_packages_are_ignored(
    android, monkeypatch, tmp_path, library
):
    elf = FakeElf(monkeypatch)
    stray = tmp_path / "lib" / "python3.14" / "_hidden.so"
    stray.parent.mkdir(parents=True, exist_ok=True)
    stray.write_bytes(b"")
    elf.for_extension(stray, symbols=UNDEFINED_PY, needed="libc.so\n")
    assert termux_linkage.repair(tmp_path, library=library) == 0
    assert elf.patches == []


# ---------------------------------------------------------------------------
# The PythonEnvironment hooks
# ---------------------------------------------------------------------------


def _environment(tmp_path, output=None):
    return PythonEnvironment(
        uv=Path("/usr/bin/uv"),
        python=Path("/usr/bin/python3"),
        destination=tmp_path / "venv",
        cache=tmp_path / "cache",
        env={},
        output=output,
    )


def _installed(monkeypatch, *, returncode=0):
    calls = []

    def fake_run(self, args, *, cwd, timeout):
        calls.append(args)
        return subprocess.CompletedProcess(
            args, returncode, stdout="", stderr="boom" if returncode else ""
        )

    monkeypatch.setattr(PythonEnvironment, "_run", fake_run)
    return calls


def test_sync_repairs_linkage_before_returning(monkeypatch, tmp_path):
    calls = _installed(monkeypatch)
    repaired = []
    monkeypatch.setattr(
        termux_linkage, "repair", lambda venv: repaired.append(venv) or 0
    )
    _environment(tmp_path).sync(tmp_path / "source")
    assert repaired == [tmp_path / "venv"]
    assert calls


def test_pip_install_repairs_linkage_before_returning(monkeypatch, tmp_path):
    _installed(monkeypatch)
    repaired = []
    monkeypatch.setattr(
        termux_linkage, "repair", lambda venv: repaired.append(venv) or 0
    )
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("cryptography\n")
    _environment(tmp_path)._install_requirements_file(requirements)
    assert repaired == [tmp_path / "venv"]


def test_failed_install_does_not_repair(monkeypatch, tmp_path):
    _installed(monkeypatch, returncode=1)
    monkeypatch.setattr(
        termux_linkage, "repair", lambda venv: pytest.fail("install failed")
    )
    with pytest.raises(InstallError):
        _environment(tmp_path).sync(tmp_path / "source")


def test_repair_failure_never_fails_the_install(monkeypatch, tmp_path):
    _installed(monkeypatch)
    monkeypatch.setattr(
        termux_linkage,
        "repair",
        lambda venv: (_ for _ in ()).throw(RuntimeError("patchelf exploded")),
    )
    _environment(tmp_path).sync(tmp_path / "source")


def test_patched_extensions_are_reported(monkeypatch, tmp_path):
    _installed(monkeypatch)
    monkeypatch.setattr(termux_linkage, "repair", lambda venv: 2)
    output = io.StringIO()
    _environment(tmp_path, output=output).sync(tmp_path / "source")
    assert "linked 2 extension(s) to libpython" in output.getvalue()
