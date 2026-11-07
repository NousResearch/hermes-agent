"""Link Python extensions explicitly for bionic's local symbol lookup."""
from __future__ import annotations

import csv
import io
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile

try:
    from .retag_wheel import record_hash
except ImportError:
    from retag_wheel import record_hash


def link_extension(extension: Path, library: Path) -> bool:
    reader = shutil.which("llvm-readelf") or shutil.which("readelf")
    if reader is None:
        raise RuntimeError("a native ELF reader is required for Python linkage verification")
    symbols = subprocess.check_output(
        [reader, "--dyn-syms", "--wide", str(extension)], text=True,
    )
    if not any(
        " UND " in line and line.split()[-1].startswith(("Py", "_Py"))
        for line in symbols.splitlines()
    ):
        return False
    needed = subprocess.check_output(
        ["patchelf", "--print-needed", str(extension)], text=True,
    ).splitlines()
    soname = subprocess.check_output(
        ["patchelf", "--print-soname", str(library)], text=True,
    ).strip()
    if not soname:
        raise RuntimeError(f"libpython has no SONAME: {library}")
    foreign = [name for name in needed if name.startswith("libpython") and name != soname]
    if foreign:
        raise RuntimeError(f"{extension.name} links a different interpreter: {foreign}")
    if soname in needed:
        return False
    subprocess.run(["patchelf", "--add-needed", soname, str(extension)], check=True)
    return True


def repair_wheel(wheel: Path, library: Path, *, repair=link_extension) -> int:
    """Repair native members without extracting untrusted archive paths."""
    with tempfile.TemporaryDirectory(prefix="hermes-wheel-link-") as tmp:
        native = Path(tmp) / "extension.so"
        with zipfile.ZipFile(wheel) as archive:
            if not any(info.filename.endswith(".so") for info in archive.infolist()):
                return 0
            members = [(info, archive.read(info)) for info in archive.infolist() if not info.is_dir()]
        records = [info.filename for info, _ in members if info.filename.endswith(".dist-info/RECORD")]
        if len(records) != 1:
            raise RuntimeError(f"expected one wheel RECORD: {wheel.name}")
        changed = 0
        updated = []
        for info, data in members:
            if info.filename.endswith(".so"):
                native.write_bytes(data)
                if repair(native, library):
                    data = native.read_bytes()
                    changed += 1
            updated.append((info, data))
        if not changed:
            return 0
        rows = [
            (info.filename, record_hash(data), str(len(data)))
            for info, data in updated if info.filename != records[0]
        ]
        rows.append((records[0], "", ""))
        record = io.StringIO()
        csv.writer(record, lineterminator="\n").writerows(rows)
        fd, staged = tempfile.mkstemp(prefix=".linked-", suffix=".whl", dir=wheel.parent)
        os.close(fd)
        try:
            with zipfile.ZipFile(staged, "w", zipfile.ZIP_DEFLATED) as archive:
                for info, data in updated:
                    archive.writestr(info, record.getvalue().encode() if info.filename == records[0] else data)
            os.chmod(staged, wheel.stat().st_mode & 0o777)
            os.replace(staged, wheel)
        finally:
            Path(staged).unlink(missing_ok=True)
        return changed
