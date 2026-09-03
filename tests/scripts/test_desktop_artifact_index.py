"""CLI/helpers for the Desktop prebuilt artifact index used by CI."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "desktop_artifact_index.py"
_SPEC = importlib.util.spec_from_file_location("desktop_artifact_index", _SCRIPT)
_MOD = importlib.util.module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
_SPEC.loader.exec_module(_MOD)
build_fragment = _MOD.build_fragment
merge_index_docs = _MOD.merge_index_docs
sha256_file = _MOD.sha256_file
zip_unpacked_dir = _MOD.zip_unpacked_dir


def test_build_and_merge_fragments(tmp_path: Path):
    a = build_fragment(
        commit="A" * 40,
        platform="linux",
        arch="x64",
        url="https://example.invalid/a.zip",
        sha256="ab" * 32,
        tag="v1",
        filename="a.zip",
    )
    b = build_fragment(
        commit="A" * 40,
        platform="linux",
        arch="x64",
        url="https://example.invalid/b.zip",
        sha256="cd" * 32,
        tag="v1",
        filename="b.zip",
        compatibility_window=80,
    )
    c = build_fragment(
        commit="B" * 40,
        platform="darwin",
        arch="arm64",
        url="https://example.invalid/c.zip",
        sha256="ef" * 32,
        tag="v1",
        filename="c.zip",
    )
    merged = merge_index_docs([a, b, c])
    assert merged["compatibility_window"] == 80
    rows = {(row["platform"], row["arch"], row["filename"]) for row in merged["artifacts"]}
    assert ("linux", "x64", "b.zip") in rows
    assert ("linux", "x64", "a.zip") not in rows
    assert ("darwin", "arm64", "c.zip") in rows


def test_zip_dir_roots_unpacked_name_and_hashes(tmp_path: Path):
    unpacked = tmp_path / "linux-unpacked"
    (unpacked / "nested").mkdir(parents=True)
    (unpacked / "Hermes").write_bytes(b"exe")
    (unpacked / "nested" / "res.txt").write_text("ok", encoding="utf-8")
    dest = tmp_path / "out.zip"
    zip_unpacked_dir(unpacked, dest)
    digest = sha256_file(dest)
    assert len(digest) == 64
    import zipfile

    with zipfile.ZipFile(dest) as zf:
        names = set(zf.namelist())
    assert "linux-unpacked/Hermes" in names
    assert "linux-unpacked/nested/res.txt" in names
