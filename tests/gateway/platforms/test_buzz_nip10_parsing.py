"""Unit tests for Buzz adapter NIP-10 thread-root and reply-parent parsing.

These tests cover the static helper methods ``_extract_thread_root`` and
``_extract_reply_parent`` added by the ``thread_replies`` config feature.
They exercise real NIP-10 tag shapes without importing the full adapter
(the methods are ``@staticmethod``).
"""

from __future__ import annotations

import pytest

from tests.gateway._plugin_adapter_loader import load_plugin_adapter


@pytest.fixture(scope="module")
def buzz_adapter():
    """Load the Buzz adapter module in isolation."""
    return load_plugin_adapter("buzz")


# ── _extract_thread_root ──────────────────────────────────────────────


class TestExtractThreadRoot:
    """BuzzAdapter._extract_thread_root — NIP-10 root event extraction."""

    def test_root_marker(self, buzz_adapter):
        """Tagged ``root`` marker returns the root event id."""
        event = {
            "tags": [
                ["e", "aaa111", "wss://relay", "root"],
                ["e", "bbb222", "wss://relay", "reply"],
            ]
        }
        assert buzz_adapter.BuzzAdapter._extract_thread_root(event) == "aaa111"

    def test_first_e_tag_fallback(self, buzz_adapter):
        """When no markers are present, the first ``e`` tag is the root."""
        event = {
            "tags": [
                ["e", "aaa111", "wss://relay"],
                ["e", "bbb222", "wss://relay"],
            ]
        }
        assert buzz_adapter.BuzzAdapter._extract_thread_root(event) == "aaa111"

    def test_no_e_tags(self, buzz_adapter):
        """Top-level message with no ``e`` tags returns ``None``."""
        event = {"tags": [["p", "pubkey"]]}
        assert buzz_adapter.BuzzAdapter._extract_thread_root(event) is None

    def test_empty_tags(self, buzz_adapter):
        """Event with empty tags list returns ``None``."""
        assert buzz_adapter.BuzzAdapter._extract_thread_root({"tags": []}) is None

    def test_missing_tags(self, buzz_adapter):
        """Event with no ``tags`` key returns ``None``."""
        assert buzz_adapter.BuzzAdapter._extract_thread_root({}) is None

    def test_non_list_tags(self, buzz_adapter):
        """Non-list ``tags`` value returns ``None`` (defensive)."""
        assert buzz_adapter.BuzzAdapter._extract_thread_root({"tags": "not-a-list"}) is None

    def test_empty_event_id(self, buzz_adapter):
        """``e`` tag with empty event id is skipped."""
        event = {"tags": [["e", "", "wss://relay", "root"]]}
        assert buzz_adapter.BuzzAdapter._extract_thread_root(event) is None


# ── _extract_reply_parent ──────────────────────────────────────────────


class TestExtractReplyParent:
    """BuzzAdapter._extract_reply_parent — NIP-10 reply-to extraction."""

    def test_reply_marker(self, buzz_adapter):
        """Tagged ``reply`` marker returns the reply parent event id."""
        event = {
            "tags": [
                ["e", "aaa111", "wss://relay", "root"],
                ["e", "bbb222", "wss://relay", "reply"],
            ]
        }
        assert buzz_adapter.BuzzAdapter._extract_reply_parent(event) == "bbb222"

    def test_last_e_tag_fallback(self, buzz_adapter):
        """When no markers are present, the last ``e`` tag is the parent."""
        event = {
            "tags": [
                ["e", "aaa111", "wss://relay"],
                ["e", "bbb222", "wss://relay"],
            ]
        }
        assert buzz_adapter.BuzzAdapter._extract_reply_parent(event) == "bbb222"

    def test_no_e_tags(self, buzz_adapter):
        """Top-level message with no ``e`` tags returns ``None``."""
        event = {"tags": [["p", "pubkey"]]}
        assert buzz_adapter.BuzzAdapter._extract_reply_parent(event) is None

    def test_missing_tags(self, buzz_adapter):
        """Event with no ``tags`` key returns ``None``."""
        assert buzz_adapter.BuzzAdapter._extract_reply_parent({}) is None

    def test_empty_event_id(self, buzz_adapter):
        """``e`` tag with empty event id is skipped."""
        event = {"tags": [["e", "", "wss://relay", "reply"]]}
        assert buzz_adapter.BuzzAdapter._extract_reply_parent(event) is None