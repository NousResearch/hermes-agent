"""The Windows payload carries no file name MakeAppx refuses to pack."""
from __future__ import annotations

import pytest

from pm.package import InstallError
from scripts.bundles.msix_names import drop_msix_reserved_names

# python-docx 1.2.0's template layout, as the payload venv and uv cache hold it.
DOCX_TEMPLATE = ("default.docx", "default-styles.xml", "default-docx-template/[Content_Types].xml",
                 "default-docx-template/_rels/.rels", "default-docx-template/word/document.xml")


def _write(root, *names):
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")


def test_python_docx_unzipped_template_leaves_venv_and_uv_cache(tmp_path):
    site = "venv/Lib/site-packages/docx/templates"
    cache = "uv-cache/archive-v0/cXkCcX-PYzjQ7QIz/docx/templates"
    _write(tmp_path, *(f"{base}/{name}" for base in (site, cache) for name in DOCX_TEMPLATE),
           "hermes-agent/contributors/emails/[bot]@example.com")

    removed = drop_msix_reserved_names(tmp_path)

    assert sorted(p.as_posix() for p in removed) == [f"{cache}/default-docx-template",
                                                    f"{site}/default-docx-template"]
    names = [p.name.lower() for p in tmp_path.rglob("*")]
    assert "[content_types].xml" not in names
    # What python-docx loads at runtime, and other bracketed names MakeAppx packs fine, stay.
    for base in (site, cache):
        assert (tmp_path / base / "default.docx").is_file()
        assert (tmp_path / base / "default-styles.xml").is_file()
    assert (tmp_path / "hermes-agent/contributors/emails/[bot]@example.com").is_file()


def test_unknown_reserved_name_fails_staging_and_names_the_file(tmp_path):
    _write(tmp_path, "venv/Lib/site-packages/somepkg/data/[Content_Types].XML")

    with pytest.raises(InstallError, match=r"venv/Lib/site-packages/somepkg/data/\[Content_Types\]\.XML"):
        drop_msix_reserved_names(tmp_path)
    assert (tmp_path / "venv/Lib/site-packages/somepkg/data/[Content_Types].XML").is_file()
