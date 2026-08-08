"""Regression tests for the wave-1 extraction of ``gateway/platforms/base.py``.

Shard s2 move clusters covered (verbatim extraction):

* ``c2`` MEDIA: tag parsing / directive stripping
    -> ``gateway/platforms/media_tag_parsing.py`` (21 move votes)
* ``c3`` document cache helpers
    -> ``gateway/platforms/document_cache.py`` (12 move votes)

Two contracts are asserted here:

1. Behavior of the moved pure helpers (unchanged semantics).
2. Re-export parity: every moved function is still importable from
   ``gateway.platforms.base`` and is the *same object* as the one in its new
   module, so all existing ``from gateway.platforms.base import ...`` call
   sites (adapters, media_cache, run.py, tests) keep working.
"""

import os
import re
import time
from pathlib import Path

import pytest

from gateway.platforms.base import (
    MEDIA_EXTENSIONLESS_TAG_RE,
    MEDIA_TAG_CLEANUP_RE,
    _match_extensionless_path,
    _merge_spans,
    _normalize_media_tag_path,
    _path_lacks_deliverable_extension,
    _resolve_extensionless_candidate,
    _strip_media_directives,
    _strip_media_tag_directives,
    cache_document_from_bytes,
    cleanup_document_cache,
    get_document_cache_dir,
)
from gateway.platforms.document_cache import (
    cache_document_from_bytes as dc_cache_document_from_bytes,
    cleanup_document_cache as dc_cleanup_document_cache,
    get_document_cache_dir as dc_get_document_cache_dir,
)
from gateway.platforms.media_tag_parsing import (
    _match_extensionless_path as mtp_match_extensionless_path,
    _merge_spans as mtp_merge_spans,
    _normalize_media_tag_path as mtp_normalize_media_tag_path,
    _path_lacks_deliverable_extension as mtp_path_lacks_deliverable_extension,
    _resolve_extensionless_candidate as mtp_resolve_extensionless_candidate,
    _strip_media_directives as mtp_strip_media_directives,
    _strip_media_tag_directives as mtp_strip_media_tag_directives,
)

# ---------------------------------------------------------------------------
# Re-export parity: base re-exports the same objects the new modules define
# ---------------------------------------------------------------------------

PARITY_PAIRS = [
    (_match_extensionless_path, mtp_match_extensionless_path),
    (_merge_spans, mtp_merge_spans),
    (_normalize_media_tag_path, mtp_normalize_media_tag_path),
    (_path_lacks_deliverable_extension, mtp_path_lacks_deliverable_extension),
    (_resolve_extensionless_candidate, mtp_resolve_extensionless_candidate),
    (_strip_media_directives, mtp_strip_media_directives),
    (_strip_media_tag_directives, mtp_strip_media_tag_directives),
    (cache_document_from_bytes, dc_cache_document_from_bytes),
    (cleanup_document_cache, dc_cleanup_document_cache),
    (get_document_cache_dir, dc_get_document_cache_dir),
]


@pytest.mark.parametrize("base_fn,module_fn", PARITY_PAIRS)
def test_reexport_parity(base_fn, module_fn):
    assert base_fn is module_fn


# ---------------------------------------------------------------------------
# MEDIA: tag parsing helpers (cluster c2)
# ---------------------------------------------------------------------------

def test_normalize_media_tag_path_strips_quotes_and_punctuation():
    assert _normalize_media_tag_path("`/tmp/x.png`") == "/tmp/x.png"
    assert _normalize_media_tag_path('"/tmp/x.png"') == "/tmp/x.png"
    assert _normalize_media_tag_path("/tmp/x.png,;") == "/tmp/x.png"
    assert _normalize_media_tag_path("") == ""
    assert _normalize_media_tag_path(None) == ""


def test_merge_spans_merges_overlapping_and_nested():
    assert _merge_spans([(1, 3), (2, 5)]) == [(1, 5)]
    assert _merge_spans([(1, 5), (2, 3)]) == [(1, 5)]
    assert _merge_spans([(0, 2), (4, 6)]) == [(0, 2), (4, 6)]
    assert _merge_spans([]) == []


def test_path_lacks_deliverable_extension():
    assert _path_lacks_deliverable_extension("Caddyfile") is True
    assert _path_lacks_deliverable_extension("Makefile") is True
    assert _path_lacks_deliverable_extension("notes.log") is True
    assert _path_lacks_deliverable_extension("photo.png") is False
    assert _path_lacks_deliverable_extension("report.pdf") is False


def test_resolve_extensionless_candidate():
    assert _resolve_extensionless_candidate("") is None
    assert _resolve_extensionless_candidate(None) is None


def test_strip_media_tag_directives_removes_tags_and_markers():
    text = "see [[audio_as_voice]] MEDIA:/tmp/a.png for the file"
    cleaned = _strip_media_tag_directives(text)
    assert "MEDIA:" not in cleaned
    assert "[[audio_as_voice]]" not in cleaned


def test_strip_media_tag_directives_passthrough_when_nothing_to_strip():
    text = "plain text, no directives"
    assert _strip_media_tag_directives(text) == text


def test_strip_media_directives_delegates():
    text = "x [[as_document]] MEDIA:/tmp/b.png y"
    cleaned = _strip_media_directives(text)
    assert "MEDIA:" not in cleaned
    assert "[[as_document]]" not in cleaned
    assert _strip_media_directives("") == ""


def test_match_extensionless_path_rejects_unknown_match():
    # A MEDIA_EXTENSIONLESS_TAG_RE match whose path fails validation (the
    # path does not exist on disk) must resolve to None, never to a bogus
    # path — the validation oracle is the contract (#24032).
    match = MEDIA_EXTENSIONLESS_TAG_RE.match("MEDIA:/definitely/not/here.txt")
    assert match is not None
    assert _match_extensionless_path("MEDIA:/definitely/not/here.txt", match) is None


def test_regex_constants_still_exported_from_base():
    assert MEDIA_TAG_CLEANUP_RE.search("MEDIA:/x.png") is not None
    assert MEDIA_EXTENSIONLESS_TAG_RE.search("MEDIA:/Caddyfile") is not None


# ---------------------------------------------------------------------------
# Document cache helpers (cluster c3)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _redirect_document_cache(tmp_path, monkeypatch):
    """Point the module-level DOCUMENT_CACHE_DIR to a fresh tmp_path."""
    monkeypatch.setattr(
        "gateway.platforms.base.DOCUMENT_CACHE_DIR", tmp_path / "doc_cache"
    )


def test_get_document_cache_dir_creates_directory(tmp_path):
    d = get_document_cache_dir()
    assert isinstance(d, Path)
    assert d.is_dir()
    assert d == (tmp_path / "doc_cache").resolve() or d == tmp_path / "doc_cache"


def test_cache_document_from_bytes_writes_prefixed_file():
    path = cache_document_from_bytes(b"hello", "report.pdf")
    assert Path(path).is_file()
    name = Path(path).name
    assert name.startswith("doc_")
    assert name.endswith("_report.pdf")
    assert Path(path).read_bytes() == b"hello"


def test_cache_document_from_bytes_sanitizes_filename():
    path = cache_document_from_bytes(b"data", "")
    assert Path(path).is_file()
    assert Path(path).name.startswith("doc_")
    assert Path(path).name.endswith("_document")


def test_cache_document_from_bytes_strips_directory_components():
    # The sanitizer keeps only the basename, so a traversal-looking filename
    # can never escape the cache dir (the ValueError guard is a backstop).
    path = cache_document_from_bytes(b"data", "../../escape.txt")
    assert "/" not in Path(path).name and "\\" not in Path(path).name
    assert Path(path).is_file()


def test_cleanup_document_cache_removes_only_old_files():
    cache_dir = get_document_cache_dir()
    old = cache_dir / "doc_old.bin"
    new = cache_dir / "doc_new.bin"
    old.write_bytes(b"x")
    new.write_bytes(b"y")
    cutoff = time.time() - (25 * 3600)
    os.utime(old, (cutoff, cutoff))
    removed = cleanup_document_cache(max_age_hours=24)
    assert removed == 1
    assert not old.exists()
    assert new.exists()
