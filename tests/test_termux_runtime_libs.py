"""Tests for scripts/termux/stage_runtime_libs.py — cache correctness.

No filesystem mocks: a real tiny .deb fixture (ar archive + data.tar) is
served over loopback HTTP and fetched through pm's hardened Download,
then unpacked through pm's DebPackage — the same production path.
"""
from __future__ import annotations

import hashlib
import io
import json
import tarfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "termux"))

import stage_runtime_libs as srl  # noqa: E402

PREFIX = srl.PREFIX_REL


# ---------------------------------------------------------------- fixtures

def _ar_header(name: str, size: int) -> bytes:
    hdr = name.ljust(16).encode()
    hdr += b"0".ljust(12)          # mtime
    hdr += b"0".ljust(6)           # uid
    hdr += b"0".ljust(6)           # gid
    hdr += b"644".ljust(8)         # mode
    hdr += str(size).ljust(10).encode()
    hdr += b"`\n"
    assert len(hdr) == 60
    return hdr


def _build_deb(path: Path, lib_name: str, content: bytes) -> None:
    """A minimal but real .deb: ar{debian-binary, control.tar.gz, data.tar}
    with data.tar carrying <PREFIX>/lib/<lib_name>."""
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w") as tf:
        lib_dir = f"{PREFIX}/lib/"
        ti = tarfile.TarInfo(lib_dir)
        ti.type = tarfile.DIRTYPE
        ti.mode = 0o755
        tf.addfile(ti)
        payload = f"FAKE-ELF {lib_name}\n".encode() + content
        ti = tarfile.TarInfo(lib_dir + lib_name)
        ti.size = len(payload)
        ti.mode = 0o755
        tf.addfile(ti, io.BytesIO(payload))

    control = io.BytesIO()
    with tarfile.open(fileobj=control, mode="w") as tf:
        stanza = f"Package: pkg-{lib_name}\nVersion: 1.0\n".encode()
        ti = tarfile.TarInfo("control")
        ti.size = len(stanza)
        tf.addfile(ti, io.BytesIO(stanza))

    members = [
        (b"debian-binary/", b"2.0\n"),
        (b"control.tar.gz/", control.getvalue()),
        (b"data.tar/", data.getvalue()),
    ]
    out = b"!<arch>\n"
    for name, body in members:
        out += _ar_header(name.decode().rstrip("/"), len(body))
        out += body
        if len(body) % 2:
            out += b"\n"
    path.write_bytes(out)


class _Server:
    """Serves files from a dir over loopback HTTP."""

    def __init__(self, root: Path) -> None:
        self.root = root

        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                f = self.server.root / self.path.lstrip("/")  # type: ignore[attr-defined]
                if not f.is_file():
                    self.send_error(404)
                    return
                body = f.read_bytes()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Accept-Ranges", "none")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.httpd.root = root  # type: ignore[attr-defined]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}"

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture()
def lib_source(tmp_path: Path):
    """Two packages; libb.so shared by both (identical bytes). Returns
    (server, table, files) where files maps staged soname -> bytes."""
    src = tmp_path / "debs"
    src.mkdir()
    shared = b"shared-impl-v1"
    _build_deb(src / "liba.deb", "liba.so", b"impl-a")
    _build_deb(src / "libb.deb", "libb.so", shared)
    _build_deb(src / "libc.deb", "libb.so", shared)  # identical collision

    server = _Server(src)
    table = {
        "liba": {"url": f"{server.url}/liba.deb",
                 "sha256": hashlib.sha256((src / "liba.deb").read_bytes()).hexdigest(),
                 "version": "1.0"},
        "libb": {"url": f"{server.url}/libb.deb",
                 "sha256": hashlib.sha256((src / "libb.deb").read_bytes()).hexdigest(),
                 "version": "2.0"},
        "libc": {"url": f"{server.url}/libc.deb",
                 "sha256": hashlib.sha256((src / "libc.deb").read_bytes()).hexdigest(),
                 "version": "3.0"},
    }
    yield server, table
    server.stop()


def _stage(tmp_path: Path, table: dict) -> Path:
    return srl.stage(tmp_path / "payload", table)


# ------------------------------------------------------------------ tests

@pytest.mark.parametrize("corruption", ["missing", "extra", "stale"])
def test_stage_cache_correctness(tmp_path, lib_source, corruption):
    """Initial stage, true cache hit with the source gone, and rebuild on
    missing / extra / corrupted output."""
    server, table = lib_source
    out = _stage(tmp_path, table)
    names = {"liba.so", "libb.so"}
    assert {p.name for p in out.glob("*.so*")} == names
    manifest = json.loads(out.parent.joinpath("manifest.json").read_text())

    # True cache hit: source deleted, no downloads possible.
    server.stop()
    import shutil
    shutil.rmtree(tmp_path / "payload" / ".work")
    assert srl.stage(tmp_path / "payload", table) == out
    assert {p.name for p in out.glob("*.so*")} == names

    # Corrupt the cache and restore the source so a rebuild is possible.
    if corruption == "missing":
        (out / "liba.so").unlink()
    elif corruption == "extra":
        (out / "libjunk.so").write_bytes(b"bogus")
    else:
        (out / "libb.so").write_bytes(b"corrupted bytes")

    src = tmp_path / "debs"
    server2 = _Server(src)
    for name in table:
        table[name]["url"] = f"{server2.url}/{name}.deb"
    try:
        result = srl.stage(tmp_path / "payload", table)
        assert result == out
    finally:
        server2.stop()
    assert {p.name for p in out.glob("*.so*")} == names
    for name in names:
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == \
            manifest["files"][name]


def test_collision_identical_ok_conflicting_raises(tmp_path, lib_source):
    """Co-installed packages may share a soname only with identical bytes."""
    server, table = lib_source
    out = _stage(tmp_path, table)
    # libb + libc both ship libb.so with identical bytes: merged once.
    assert (out / "libb.so").read_bytes().startswith(b"FAKE-ELF libb.so")

    # Now a conflicting rebuild: same soname, different bytes.
    _build_deb(tmp_path / "debs" / "libc.deb", "libb.so", b"DIFFERENT-IMPL")
    table["libc"]["sha256"] = hashlib.sha256(
        (tmp_path / "debs" / "libc.deb").read_bytes()).hexdigest()

    # Force a miss: current manifest no longer validates for libc's bytes.
    with pytest.raises(Exception):
        srl.stage(tmp_path / "payload", table)


def test_rebuild_removes_superseded_license_files(tmp_path, lib_source):
    _, table = lib_source
    out = _stage(tmp_path, table)
    stale = out.parent / "share/doc/obsolete/copyright"
    stale.parent.mkdir(parents=True)
    stale.write_text("old package notice", encoding="utf-8")
    table["liba"]["version"] = "next"
    srl.stage(tmp_path / "payload", table)
    assert not stale.exists()
    assert (out / "liba.so").is_file()


def test_failed_download_identifies_the_pin_without_publishing(tmp_path, lib_source):
    _, table = lib_source
    (tmp_path / "debs/libb.deb").unlink()
    payload = tmp_path / "payload"
    with pytest.raises(srl.StageError) as caught:
        srl.stage(payload, table)
    message = str(caught.value)
    assert "libb" in message and table["libb"]["version"] in message
    assert table["libb"]["url"] in message
    assert "404" in message
    assert not (payload / "runtime-libs/manifest.json").exists()


def test_stale_scratch_cannot_poison_a_rebuilt_cache(tmp_path, lib_source):
    _, table = lib_source
    out = _stage(tmp_path, table)
    expected = (out / "liba.so").read_bytes()
    scratch = tmp_path / "payload/.work/runtime-libs/extract/liba" / PREFIX / "lib/liba.so"
    scratch.write_bytes(b"corrupted scratch bytes")
    (out / "liba.so").unlink()
    srl.stage(tmp_path / "payload", table)
    assert (out / "liba.so").read_bytes() == expected
