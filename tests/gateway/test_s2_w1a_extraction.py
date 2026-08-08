"""Regression tests for the s2-w1a extraction of gateway/platforms/base.py.

Wave-1 blind implementation moved two unanimous move-clusters out of the
god-file into sibling modules (per shard-plan s2):

* cluster c3 -> ``gateway/platforms/document_cache.py``
  (get_document_cache_dir, cache_document_from_bytes, cleanup_document_cache)
* cluster c8 -> ``gateway/platforms/send_errors.py``
  (_error_blob, classify_send_error, is_chat_level_not_found)

``gateway/platforms/base.py`` re-exports the public names so existing import
sites (adapters, gateway/delivery.py, tests) keep working unchanged.  These
tests pin the re-export identity and the pure behavior of the moved code.
"""

import os
import time
from pathlib import Path

import pytest

from gateway.platforms import base as base_mod
from gateway.platforms.base import (
    SEND_ERROR_KINDS,
    cache_document_from_bytes,
    classify_send_error,
    cleanup_document_cache,
    get_document_cache_dir,
    is_chat_level_not_found,
)
from gateway.platforms.document_cache import (
    cache_document_from_bytes as moved_cache_document,
    cleanup_document_cache as moved_cleanup_document,
    get_document_cache_dir as moved_get_doc_dir,
)
from gateway.platforms.send_errors import (
    _error_blob,
    classify_send_error as moved_classify,
    is_chat_level_not_found as moved_chat_level,
)


# ---------------------------------------------------------------------------
# Re-export identity: base.py must keep exposing the moved names as the SAME
# objects (existing callers import them from gateway.platforms.base).
# ---------------------------------------------------------------------------

class TestBaseReExports:
    def test_document_cache_names_are_the_moved_objects(self):
        assert base_mod.get_document_cache_dir is moved_get_doc_dir
        assert base_mod.cache_document_from_bytes is moved_cache_document
        assert base_mod.cleanup_document_cache is moved_cleanup_document

    def test_send_error_names_are_the_moved_objects(self):
        assert base_mod.classify_send_error is moved_classify
        assert base_mod.is_chat_level_not_found is moved_chat_level
        assert base_mod.SEND_ERROR_KINDS is SEND_ERROR_KINDS


# ---------------------------------------------------------------------------
# Send-error classification (cluster c8) — pure behavior in the new module
# ---------------------------------------------------------------------------

class TestClassifySendError:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Message_too_long", "too_long"),
            ("Bad Request: message is too long", "too_long"),
            ("Bad Request: can't parse entities: unsupported start tag", "bad_format"),
            ("Bad Request: can't find end of the entity", "bad_format"),
            ("Forbidden: bot was blocked by the user", "forbidden"),
            ("Forbidden: user is deactivated", "forbidden"),
            ("Bad Request: not enough rights to send text messages", "forbidden"),
            ("Bad Request: chat not found", "not_found"),
            ("Bad Request: message to edit not found", "not_found"),
            ("Too Many Requests: retry after 12", "rate_limited"),
            ("Flood control exceeded", "rate_limited"),
            ("ConnectError: connection refused", "transient"),
            ("ConnectTimeout", "transient"),
            ("some entirely novel provider message", "unknown"),
            ("", "unknown"),
        ],
    )
    def test_classify_send_error_text(self, text, expected):
        assert classify_send_error(None, text) == expected
        # The moved module must agree with the base re-export exactly.
        assert moved_classify(None, text) == expected

    def test_every_classification_is_in_the_vocabulary(self):
        for s in [
            "message_too_long",
            "can't parse entities",
            "forbidden",
            "chat not found",
            "flood",
            "connecterror",
            "mystery",
            "",
        ]:
            assert classify_send_error(None, s) in SEND_ERROR_KINDS

    def test_retryable_patterns_still_resolve_from_base(self):
        # _RETRYABLE_ERROR_PATTERNS lives in base.py; classify_send_error
        # must still see it through the local import.
        assert classify_send_error(None, "ConnectionResetError: broken pipe") == "transient"
        assert classify_send_error(None, "RemoteDisconnected") == "transient"


class TestIsChatLevelNotFound:
    def test_chat_level_is_true(self):
        assert is_chat_level_not_found(None, "Bad Request: chat not found") is True

    def test_subchat_only_is_false(self):
        assert is_chat_level_not_found(None, "Bad Request: message to edit not found") is False
        assert is_chat_level_not_found(None, "thread not found") is False

    def test_subchat_wins_when_both_present(self):
        # Conservative: a sub-chat marker means the parent chat may still be
        # reachable, so the target must NOT be marked dead.
        assert (
            is_chat_level_not_found(None, "chat not found: message to edit not found")
            is False
        )

    def test_matches_agree_with_classifier(self):
        # classify_send_error collapses both families into "not_found".
        assert classify_send_error(None, "Bad Request: chat not found") == "not_found"
        assert classify_send_error(None, "message to edit not found") == "not_found"


class TestErrorBlob:
    def test_includes_exc_class_and_text_lowercased(self):
        exc = ValueError("Bad Request: chat not found")
        blob = _error_blob(exc)
        assert blob == "bad request: chat not found valueerror"

    def test_error_text_only(self):
        assert _error_blob(None, "Forbidden: bot was blocked") == "forbidden: bot was blocked"

    def test_empty(self):
        assert _error_blob() == ""


# ---------------------------------------------------------------------------
# Document cache (cluster c3) — pure behavior in the new module
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _redirect_cache(tmp_path, monkeypatch):
    """Point the module-level DOCUMENT_CACHE_DIR at a fresh tmp_path (same
    seam the original tests/gateway/test_document_cache.py uses)."""
    monkeypatch.setattr(
        "gateway.platforms.base.DOCUMENT_CACHE_DIR", tmp_path / "doc_cache"
    )


class TestGetDocumentCacheDir:
    def test_creates_directory(self):
        cache_dir = get_document_cache_dir()
        assert cache_dir.exists()
        assert cache_dir.is_dir()
        assert get_document_cache_dir() is not None


class TestCacheDocumentFromBytes:
    def test_basic_caching(self):
        data = b"hello world"
        path = cache_document_from_bytes(data, "test.txt")
        assert os.path.exists(path)
        assert Path(path).read_bytes() == data

    def test_filename_preserved_in_path(self):
        path = cache_document_from_bytes(b"data", "report.pdf")
        assert "report.pdf" in os.path.basename(path)
        assert os.path.basename(path).startswith("doc_")

    def test_empty_filename_uses_fallback(self):
        path = cache_document_from_bytes(b"data", "")
        assert "document" in os.path.basename(path)

    def test_moved_module_agrees(self):
        path = moved_cache_document(b"data", "moved.txt")
        assert "moved.txt" in os.path.basename(path)


class TestCleanupDocumentCache:
    def test_removes_old_files(self):
        cache_dir = get_document_cache_dir()
        old_file = cache_dir / "old.txt"
        old_file.write_text("old")
        old_mtime = time.time() - 48 * 3600
        os.utime(old_file, (old_mtime, old_mtime))

        removed = cleanup_document_cache(max_age_hours=24)
        assert removed == 1
        assert not old_file.exists()
