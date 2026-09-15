"""Exercise the linker repair against a real native extension on Linux."""
from __future__ import annotations

import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

from scripts.termux import python_linkage


@pytest.mark.platforms("linux")
def test_python_symbols_gain_an_explicit_library_dependency(tmp_path):
    import _cffi_backend

    library = Path(sysconfig.get_config_var("LIBDIR")) / sysconfig.get_config_var("LDLIBRARY")
    extension = tmp_path / Path(_cffi_backend.__file__).name
    shutil.copyfile(_cffi_backend.__file__, extension)  # Writable scratch even from a read-only Nix store.
    original = subprocess.check_output(["patchelf", "--print-needed", str(extension)], text=True).splitlines()
    for name in original:
        if name.startswith("libpython"):
            subprocess.run(["patchelf", "--remove-needed", name, str(extension)], check=True)
    assert python_linkage.link_extension(extension, library)
    needed = subprocess.check_output(["patchelf", "--print-needed", str(extension)], text=True).splitlines()
    soname = subprocess.check_output(["patchelf", "--print-soname", str(library)], text=True).strip()
    assert soname in needed
    assert not python_linkage.link_extension(extension, library), "a second pass must not edit a correct extension"
    subprocess.run(
        [sys.executable, "-c", "import _cffi_backend; print(_cffi_backend.__file__)"],
        cwd=tmp_path, check=True,
    )


def test_wheel_rewrite_regenerates_record_for_changed_member(tmp_path):
    import zipfile
    from scripts.termux import retag_wheel
    from tests.termux_fixtures import write_wheel, verify_record

    wheel = write_wheel(tmp_path)

    def repair(path, library):
        assert library == tmp_path / "libpython.so"
        assert path.read_bytes() == b"\x7fELFfake"
        path.write_bytes(b"repaired native bytes")
        return True

    python_linkage.repair_wheel(wheel, tmp_path / "libpython.so", repair=repair)
    verify_record(wheel)  # Retagging must not hide a stale repair RECORD.
    with zipfile.ZipFile(wheel) as archive:
        assert archive.read("fakedep/_native.so") == b"repaired native bytes"
    retagged = retag_wheel.retag_wheel(str(wheel), "android_24_arm64_v8a")
    verify_record(retagged)
    with zipfile.ZipFile(retagged) as archive:
        assert archive.read("fakedep/_native.so") == b"repaired native bytes"
        assert b"Tag: py3-none-android_24_arm64_v8a\n" in archive.read("fakedep-1.2.3.dist-info/WHEEL")
