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
    shutil.copy2(_cffi_backend.__file__, extension)
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
    import base64
    import csv
    import hashlib
    import io
    import zipfile

    wheel = tmp_path / "sample-1.0-cp311-cp311-linux_aarch64.whl"
    record = "sample-1.0.dist-info/RECORD"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("sample/_native.so", b"unrepaired native bytes")
        archive.writestr("sample/__init__.py", b"")
        archive.writestr("sample-1.0.dist-info/WHEEL", "Wheel-Version: 1.0\nTag: cp311-cp311-linux_aarch64\n")
        archive.writestr(record, "")

    def repair(path, library):
        assert library == tmp_path / "libpython.so"
        path.write_bytes(b"repaired native bytes")
        return True

    python_linkage.repair_wheel(wheel, tmp_path / "libpython.so", repair=repair)
    with zipfile.ZipFile(wheel) as archive:
        assert archive.read("sample/_native.so") == b"repaired native bytes"
        rows = {r[0]: r[1:] for r in csv.reader(io.StringIO(archive.read(record).decode()))}
        for name in archive.namelist():
            if name == record:
                assert rows[name] == ["", ""]
                continue
            data = archive.read(name)
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            assert rows[name] == ["sha256=" + digest, str(len(data))]
