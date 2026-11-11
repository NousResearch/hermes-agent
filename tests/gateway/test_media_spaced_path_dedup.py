"""A ``MEDIA:`` tag whose known-extension path contains a space is ONE attachment (#78383).

The extension-less scanner also walks that tag, stops at the first space, and validates the
truncated prefix on disk. Through a symlinked directory (or with Windows separators) that prefix
resolves to the same file under another spelling, and the string dedupe uploaded it twice. When the
prefix is itself a real file, a second, unrelated file was attached.
"""

import os

from gateway.platforms.base import BasePlatformAdapter


def _allow(monkeypatch, root):
    monkeypatch.setattr("gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS", (str(root),))
    monkeypatch.delenv("HERMES_MEDIA_DELIVERY_STRICT", raising=False)


def test_spaced_path_through_a_symlink_is_one_attachment(tmp_path, monkeypatch):
    real = tmp_path / "real"
    (real / "Hermes Agent").mkdir(parents=True)
    (real / "Hermes Agent" / "notes.md").write_text("hi", encoding="utf-8")
    link = tmp_path / "link"
    link.symlink_to(real)
    _allow(monkeypatch, tmp_path)

    media, cleaned = BasePlatformAdapter.extract_media(f"Done.\nMEDIA:{link}/Hermes Agent/notes.md\n")

    assert media == [(f"{link}/Hermes Agent/notes.md", False)]
    assert cleaned == "Done."


def test_spaced_path_prefix_that_is_another_file_is_not_attached(tmp_path, monkeypatch):
    (tmp_path / "My Documents").mkdir()
    (tmp_path / "My Documents" / "x.md").write_text("wanted", encoding="utf-8")
    (tmp_path / "My").write_text("unrelated", encoding="utf-8")  # the path cut at its first space
    _allow(monkeypatch, tmp_path)

    media, cleaned = BasePlatformAdapter.extract_media(f"Here.\nMEDIA:{tmp_path}/My Documents/x.md\n")

    assert [os.path.basename(p) for p, _ in media] == ["x.md"]
    assert cleaned == "Here."
