"""Cached attachment filenames must be NFC so the agent can open the path it is given."""
import unicodedata

from gateway.platforms import base


def test_cached_document_name_is_nfc(tmp_path, monkeypatch):
    monkeypatch.setattr(base, "get_document_cache_dir", lambda: tmp_path)
    nfd_name = unicodedata.normalize("NFD", "뉴스클리핑.pdf")
    assert not unicodedata.is_normalized("NFC", nfd_name)

    path = base.cache_document_from_bytes(b"x", nfd_name)

    assert path.endswith(unicodedata.normalize("NFC", "뉴스클리핑.pdf"))
    assert unicodedata.is_normalized("NFC", path)
    assert (tmp_path / path.rsplit("/", 1)[-1]).read_bytes() == b"x"


def test_ascii_name_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(base, "get_document_cache_dir", lambda: tmp_path)
    path = base.cache_document_from_bytes(b"x", "lecture2nd.pdf")
    assert path.endswith("_lecture2nd.pdf")
