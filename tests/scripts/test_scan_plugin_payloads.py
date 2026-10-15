"""Unit tests for the catalog payload scanner (README rule 17).

``scripts/scan_plugin_payloads.py`` refuses compiled binaries, committed Python
bytecode and archives hiding either, while leaving plain-text source bundles and
inert assets (fonts, images) alone. Invariants, not line counts:
one benign shape stays clean, one attack shape per bypass still fails.
"""
from __future__ import annotations

import importlib.util
import io
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "scan_plugin_payloads.py"

_spec = importlib.util.spec_from_file_location("scan_plugin_payloads", SCRIPT)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def _scan(d: Path) -> list[str]:
    return mod.scan(d)


def _run(d: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), str(d)], capture_output=True, text=True, check=False)


def test_plain_source_tree_is_clean(tmp_path):
    (tmp_path / "__init__.py").write_text("x = 1\n")
    (tmp_path / "README.md").write_text("# hi\n")
    (tmp_path / "data.json").write_text("{}\n")
    assert _scan(tmp_path) == []
    assert _run(tmp_path).returncode == 0


@pytest.mark.parametrize("name,head", [
    ("evil.exe", b"MZ\x90\x00"),
    ("renamed.js", b"MZ\x90\x00"),       # PE header, source extension
    ("lib.so", b"\x7fELF\x02\x01\x01"),
    ("odd.dat", b"\x7fELF"),             # ELF, unknown extension
    ("mod.wasm", b"\x00asm\x01\x00"),
    ("bin.dylib", b"\xcf\xfa\xed\xfe"),
])
def test_compiled_binary_is_flagged_by_header(tmp_path, name, head):
    (tmp_path / name).write_bytes(head + b"\x00" * 32)
    findings = _scan(tmp_path)
    assert any(name in f and "compiled binary" in f for f in findings), findings


@pytest.mark.parametrize("name", ["cached.pyc", "sub/thing.pyo"])
def test_committed_bytecode_is_flagged(tmp_path, name):
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x00" * 16)
    findings = _scan(tmp_path)
    assert any(name.split("/")[-1] in f and "bytecode" in f for f in findings), findings


def test_source_tarball_is_clean(tmp_path):
    arc = tmp_path / "bundle.tar.gz"
    with tarfile.open(arc, "w:gz") as tf:
        for member_name, body in [("a.py", b"x=1\n"), ("docs/readme.md", b"# hi\n"),
                                  ("Dockerfile", b"FROM scratch\n"), ("logs.jsonl", b"{}\n"),
                                  ("font.ttf", b"\x00\x01\x00\x00font")]:
            ti = tarfile.TarInfo(member_name); ti.size = len(body)
            tf.addfile(ti, io.BytesIO(body))
    assert _scan(tmp_path) == []


def test_tarball_hiding_a_binary_fails(tmp_path):
    arc = tmp_path / "bundle.tar.gz"
    with tarfile.open(arc, "w:gz") as tf:
        body = b"\x7fELF" + b"\x00" * 40     # ELF under a .py member name
        ti = tarfile.TarInfo("mod.py"); ti.size = len(body)
        tf.addfile(ti, io.BytesIO(body))
    findings = _scan(tmp_path)
    assert any("compiled binary" in f for f in findings), findings


def test_tarball_with_unknown_blob_member_fails(tmp_path):
    arc = tmp_path / "bundle.tgz"
    with tarfile.open(arc, "w:gz") as tf:
        body = b"\x00\x01\x02opaque"
        ti = tarfile.TarInfo("payload.bin"); ti.size = len(body)
        tf.addfile(ti, io.BytesIO(body))
    findings = _scan(tmp_path)
    assert any("not reviewable source" in f for f in findings), findings


@pytest.mark.parametrize("name, body, expect", [
    ("words.tsv.gz", "haus\thäuser\n".encode(), None),       # gzipped UTF-8 data: reviewable
    ("tool.py.gz", b"import os\n", "not plain data"),          # code may not hide compressed
    ("words.txt.gz", b"\x7fELF" + b"\x00" * 40, "not text"),   # binary under a data name
    ("payload.bin.gz", b"opaque", "not plain data"),
])
def test_single_compressed_file_is_judged_by_what_it_decompresses_to(tmp_path, name, body, expect):
    import gzip
    (tmp_path / name).write_bytes(gzip.compress(body))
    findings = _scan(tmp_path)
    assert (findings == []) if expect is None else any(expect in f for f in findings), findings


def test_zip_hiding_a_binary_fails(tmp_path):
    arc = tmp_path / "bundle.zip"
    with zipfile.ZipFile(arc, "w") as zf:
        zf.writestr("inner.exe", b"MZ\x90\x00")
    findings = _scan(tmp_path)
    assert any("not reviewable source" in f for f in findings), findings


def test_symlinks_are_not_followed(tmp_path):
    (tmp_path / "real.py").write_text("x=1\n")
    try:
        (tmp_path / "link").symlink_to("/bin/ls")
    except OSError:
        pytest.skip("symlinks unsupported")
    assert _scan(tmp_path) == []


def test_git_checkout_judges_the_commit_not_bytecode_validate_wrote(tmp_path):
    """The catalog gate imports the plugin (validate) before scanning, which writes
    untracked __pycache__; only bytecode the pinned commit tracks fails rule 17."""
    git = ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "__init__.py").write_text("x = 1\n")
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-qm", "src"], check=True)
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "__init__.cpython-314.pyc").write_bytes(b"\x00" * 16)
    assert _scan(tmp_path) == []
    subprocess.run([*git, "add", "-f", "__pycache__"], check=True)
    subprocess.run([*git, "commit", "-qm", "bytecode"], check=True)
    assert any("committed Python bytecode" in f for f in _scan(tmp_path))
