"""Small, structurally valid Termux archives with independent integrity checks."""
import base64
import csv
import hashlib
import io
import tarfile
import zipfile
from pathlib import Path


def build_deb(path, control, files=(), compression="gz"):
    members = [("debian-binary", b"2.0\n")]
    for kind, entries in (("control", {"control": "".join(f"{k}: {v}\n" for k, v in control.items()).encode()}),
                          ("data", dict(files))):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode=f"w:{compression}") as archive:
            for name, data in entries.items():
                info = name if isinstance(name, tarfile.TarInfo) else tarfile.TarInfo(name)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
        members.append((f"{kind}.tar.{compression}", buffer.getvalue()))
    result = bytearray(b"!<arch>\n")
    for name, data in members:
        result.extend(f"{name:<16}{0:<12}{0:<6}{0:<6}{'100644':<8}{len(data):<10}`\n".encode())
        result.extend(data)
        result.extend(b"\n" if len(data) % 2 else b"")
    path.write_bytes(result)


def record_hash(data):
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


def write_wheel(directory, distribution="fakedep", version="1.2.3", platform_tag="linux_aarch64",
                *, metadata_version=None, include_so=True):
    info = f"{distribution}-{version}.dist-info"
    members = {
        f"{distribution}/__init__.py": b"",
        f"{info}/METADATA": f"Metadata-Version: 2.1\nName: {distribution}\nVersion: {metadata_version or version}\n".encode(),
        f"{info}/WHEEL": f"Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: py3-none-{platform_tag}\n".encode(),
    }
    if include_so:
        members[f"{distribution}/_native.so"] = b"\x7fELFfake"
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="\n").writerows(
        [[name, record_hash(data), str(len(data))] for name, data in members.items()]
        + [[f"{info}/RECORD", "", ""]]
    )
    members[f"{info}/RECORD"] = buffer.getvalue().encode()
    path = Path(directory) / f"{distribution}-{version}-py3-none-{platform_tag}.whl"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return path


def verify_record(path):
    with zipfile.ZipFile(path) as archive:
        record, = [name for name in archive.namelist() if name.endswith(".dist-info/RECORD")]
        rows = list(csv.reader(io.StringIO(archive.read(record).decode())))
        assert len(rows) == len(archive.namelist())
        assert {row[0] for row in rows} == set(archive.namelist())
        for name, digest, size in rows:
            data = archive.read(name)
            assert (digest, size) == (("", "") if name == record else (record_hash(data), str(len(data))))
