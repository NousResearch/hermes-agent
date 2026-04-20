"""Tests for feishu_comment_rules — 3-tier access control rule engine."""

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from gateway.platforms.feishu_comment_rules import (
    CommentsConfig,
    CommentDocumentRule,
    ResolvedCommentRule,
    _MtimeCache,
    _parse_document_rule,
    has_wiki_keys,
    is_user_allowed,
    load_config,
    pairing_add,
    pairing_list,
    pairing_remove,
    resolve_rule,
)


class TestCommentDocumentRuleParsing(unittest.TestCase):
    def test_parse_full_rule(self):
        rule = _parse_document_rule({
            "enabled": False,
            "policy": "allowlist",
            "allow_from": ["ou_a", "ou_b"],
        })
        self.assertFalse(rule.enabled)
        self.assertEqual(rule.policy, "allowlist")
        self.assertEqual(rule.allow_from, frozenset(["ou_a", "ou_b"]))

    def test_parse_partial_rule(self):
        rule = _parse_document_rule({"policy": "allowlist"})
        self.assertIsNone(rule.enabled)
        self.assertEqual(rule.policy, "allowlist")
        self.assertIsNone(rule.allow_from)

    def test_parse_empty_rule(self):
        rule = _parse_document_rule({})
        self.assertIsNone(rule.enabled)
        self.assertIsNone(rule.policy)
        self.assertIsNone(rule.allow_from)

    def test_invalid_policy_ignored(self):
        rule = _parse_document_rule({"policy": "invalid_value"})
        self.assertIsNone(rule.policy)


class TestResolveRule(unittest.TestCase):
    def test_exact_match(self):
        cfg = CommentsConfig(
            policy="pairing",
            allow_from=frozenset(["ou_top"]),
            documents={
                "docx:abc": CommentDocumentRule(policy="allowlist"),
            },
        )
        rule = resolve_rule(cfg, "docx", "abc")
        self.assertEqual(rule.policy, "allowlist")
        self.assertTrue(rule.match_source.startswith("exact:"))

    def test_wildcard_match(self):
        cfg = CommentsConfig(
            policy="pairing",
            documents={
                "*": CommentDocumentRule(policy="allowlist"),
            },
        )
        rule = resolve_rule(cfg, "docx", "unknown")
        self.assertEqual(rule.policy, "allowlist")
        self.assertEqual(rule.match_source, "wildcard")

    def test_top_level_fallback(self):
        cfg = CommentsConfig(policy="pairing", allow_from=frozenset(["ou_top"]))
        rule = resolve_rule(cfg, "docx", "whatever")
        self.assertEqual(rule.policy, "pairing")
        self.assertEqual(rule.allow_from, frozenset(["ou_top"]))
        self.assertEqual(rule.match_source, "top")

    def test_exact_overrides_wildcard(self):
        cfg = CommentsConfig(
            policy="pairing",
            documents={
                "*": CommentDocumentRule(policy="pairing"),
                "docx:abc": CommentDocumentRule(policy="allowlist"),
            },
        )
        rule = resolve_rule(cfg, "docx", "abc")
        self.assertEqual(rule.policy, "allowlist")
        self.assertTrue(rule.match_source.startswith("exact:"))

    def test_field_by_field_fallback(self):
        """Exact sets policy, wildcard sets allow_from, enabled from top."""
        cfg = CommentsConfig(
            enabled=True,
            policy="pairing",
            allow_from=frozenset(["ou_top"]),
            documents={
                "*": CommentDocumentRule(allow_from=frozenset(["ou_wildcard"])),
                "docx:abc": CommentDocumentRule(policy="allowlist"),
            },
        )
        rule = resolve_rule(cfg, "docx", "abc")
        self.assertEqual(rule.policy, "allowlist")
        self.assertEqual(rule.allow_from, frozenset(["ou_wildcard"]))
        self.assertTrue(rule.enabled)

    def test_explicit_empty_allow_from_does_not_fall_through(self):
        """allow_from=[] on exact should NOT inherit from wildcard or top."""
        cfg = CommentsConfig(
            allow_from=frozenset(["ou_top"]),
            documents={
                "*": CommentDocumentRule(allow_from=frozenset(["ou_wildcard"])),
                "docx:abc": CommentDocumentRule(
                    policy="allowlist",
                    allow_from=frozenset(),
                ),
            },
        )
        rule = resolve_rule(cfg, "docx", "abc")
        self.assertEqual(rule.allow_from, frozenset())

    def test_wiki_token_match(self):
        cfg = CommentsConfig(
            policy="pairing",
            documents={
                "wiki:WIKI123": CommentDocumentRule(policy="allowlist"),
            },
        )
        rule = resolve_rule(cfg, "docx", "obj_token", wiki_token="WIKI123")
        self.assertEqual(rule.policy, "allowlist")
        self.assertTrue(rule.match_source.startswith("exact:wiki:"))

    def test_exact_takes_priority_over_wiki(self):
        cfg = CommentsConfig(
            documents={
                "docx:abc": CommentDocumentRule(policy="allowlist"),
                "wiki:WIKI123": CommentDocumentRule(policy="pairing"),
            },
        )
        rule = resolve_rule(cfg, "docx", "abc", wiki_token="WIKI123")
        self.assertEqual(rule.policy, "allowlist")
        self.assertTrue(rule.match_source.startswith("exact:docx:"))

    def test_default_config(self):
        cfg = CommentsConfig()
        rule = resolve_rule(cfg, "docx", "anything")
        self.assertTrue(rule.enabled)
        self.assertEqual(rule.policy, "pairing")
        self.assertEqual(rule.allow_from, frozenset())


class TestHasWikiKeys(unittest.TestCase):
    def test_no_wiki_keys(self):
        cfg = CommentsConfig(documents={
            "docx:abc": CommentDocumentRule(policy="allowlist"),
            "*": CommentDocumentRule(policy="pairing"),
        })
        self.assertFalse(has_wiki_keys(cfg))

    def test_has_wiki_keys(self):
        cfg = CommentsConfig(documents={
            "wiki:WIKI123": CommentDocumentRule(policy="allowlist"),
        })
        self.assertTrue(has_wiki_keys(cfg))

    def test_empty_documents(self):
        cfg = CommentsConfig()
        self.assertFalse(has_wiki_keys(cfg))


class TestIsUserAllowed(unittest.TestCase):
    def test_allowlist_allows_listed(self):
        rule = ResolvedCommentRule(True, "allowlist", frozenset(["ou_a"]), "top")
        self.assertTrue(is_user_allowed(rule, "ou_a"))

    def test_allowlist_denies_unlisted(self):
        rule = ResolvedCommentRule(True, "allowlist", frozenset(["ou_a"]), "top")
        self.assertFalse(is_user_allowed(rule, "ou_b"))

    def test_allowlist_empty_denies_all(self):
        rule = ResolvedCommentRule(True, "allowlist", frozenset(), "top")
        self.assertFalse(is_user_allowed(rule, "ou_anyone"))

    def test_pairing_allows_in_allow_from(self):
        rule = ResolvedCommentRule(True, "pairing", frozenset(["ou_a"]), "top")
        self.assertTrue(is_user_allowed(rule, "ou_a"))

    def test_pairing_checks_store(self):
        rule = ResolvedCommentRule(True, "pairing", frozenset(), "top")
        with patch(
            "gateway.platforms.feishu_comment_rules._load_pairing_approved",
            return_value={"ou_approved"},
        ):
            self.assertTrue(is_user_allowed(rule, "ou_approved"))
            self.assertFalse(is_user_allowed(rule, "ou_unknown"))


class TestMtimeCache(unittest.TestCase):
    def test_returns_empty_dict_for_missing_file(self):
        cache = _MtimeCache(Path("/nonexistent/path.json"))
        self.assertEqual(cache.load(), {})

    def test_reads_file_and_caches(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"key": "value"}, f)
            f.flush()
            path = Path(f.name)
        try:
            cache = _MtimeCache(path)
            data = cache.load()
            self.assertEqual(data, {"key": "value"})
            # Second load should use cache (same mtime)
            data2 = cache.load()
            self.assertEqual(data2, {"key": "value"})
        finally:
            path.unlink()

    def test_reloads_on_mtime_change(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"v": 1}, f)
            f.flush()
            path = Path(f.name)
        try:
            cache = _MtimeCache(path)
            self.assertEqual(cache.load(), {"v": 1})
            # Modify file
            time.sleep(0.05)
            with open(path, "w") as f2:
                json.dump({"v": 2}, f2)
            # Force mtime change detection
            os.utime(path, (time.time() + 1, time.time() + 1))
            self.assertEqual(cache.load(), {"v": 2})
        finally:
            path.unlink()


class TestLoadConfig(unittest.TestCase):
    def test_load_with_documents(self):
        raw = {
            "enabled": True,
            "policy": "allowlist",
            "allow_from": ["ou_a"],
            "documents": {
                "*": {"policy": "pairing"},
                "docx:abc": {"policy": "allowlist", "allow_from": ["ou_b"]},
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(raw, f)
            path = Path(f.name)
        try:
            with patch("gateway.platforms.feishu_comment_rules.RULES_FILE", path):
                with patch("gateway.platforms.feishu_comment_rules._rules_cache", _MtimeCache(path)):
                    cfg = load_config()
            self.assertTrue(cfg.enabled)
            self.assertEqual(cfg.policy, "allowlist")
            self.assertEqual(cfg.allow_from, frozenset(["ou_a"]))
            self.assertIn("*", cfg.documents)
            self.assertIn("docx:abc", cfg.documents)
            self.assertEqual(cfg.documents["docx:abc"].policy, "allowlist")
        finally:
            path.unlink()

    def test_load_missing_file_returns_defaults(self):
        with patch("gateway.platforms.feishu_comment_rules._rules_cache", _MtimeCache(Path("/nonexistent"))):
            cfg = load_config()
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.policy, "pairing")
        self.assertEqual(cfg.allow_from, frozenset())
        self.assertEqual(cfg.documents, {})


class TestPairingStore(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._pairing_file = Path(self._tmpdir) / "pairing.json"
        with open(self._pairing_file, "w") as f:
            json.dump({"approved": {}}, f)
        self._patcher_file = patch("gateway.platforms.feishu_comment_rules.PAIRING_FILE", self._pairing_file)
        self._patcher_cache = patch(
            "gateway.platforms.feishu_comment_rules._pairing_cache",
            _MtimeCache(self._pairing_file),
        )
        self._patcher_file.start()
        self._patcher_cache.start()

    def tearDown(self):
        self._patcher_cache.stop()
        self._patcher_file.stop()
        if self._pairing_file.exists():
            self._pairing_file.unlink()
        os.rmdir(self._tmpdir)

    def test_add_and_list(self):
        self.assertTrue(pairing_add("ou_new"))
        approved = pairing_list()
        self.assertIn("ou_new", approved)

    def test_add_duplicate(self):
        pairing_add("ou_a")
        self.assertFalse(pairing_add("ou_a"))

    def test_remove(self):
        pairing_add("ou_a")
        self.assertTrue(pairing_remove("ou_a"))
        self.assertNotIn("ou_a", pairing_list())

    def test_remove_nonexistent(self):
        self.assertFalse(pairing_remove("ou_nobody"))


if __name__ == "__main__":
    unittest.main()
