"""Tests for feishu_comment_rules — 3-tier access control rule engine."""

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from plugins.platforms.feishu.feishu_comment_rules import (
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


class TestHasWikiKeys(unittest.TestCase):
    def test_no_wiki_keys(self):
        cfg = CommentsConfig(documents={
            "docx:abc": CommentDocumentRule(policy="allowlist"),
            "*": CommentDocumentRule(policy="pairing"),
        })
        self.assertFalse(has_wiki_keys(cfg))


class TestIsUserAllowed(unittest.TestCase):
    def test_allowlist_allows_listed(self):
        rule = ResolvedCommentRule(True, "allowlist", frozenset(["ou_a"]), "top")
        self.assertTrue(is_user_allowed(rule, "ou_a"))


    def test_pairing_checks_store(self):
        rule = ResolvedCommentRule(True, "pairing", frozenset(), "top")
        with patch(
            "plugins.platforms.feishu.feishu_comment_rules._load_pairing_approved",
            return_value={"ou_approved"},
        ):
            self.assertTrue(is_user_allowed(rule, "ou_approved"))
            self.assertFalse(is_user_allowed(rule, "ou_unknown"))


class TestMtimeCache(unittest.TestCase):
    def test_returns_empty_dict_for_missing_file(self):
        cache = _MtimeCache(Path("/nonexistent/path.json"))
        self.assertEqual(cache.load(), {})


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
            with patch("plugins.platforms.feishu.feishu_comment_rules._rules_file", return_value=path):
                with patch("plugins.platforms.feishu.feishu_comment_rules._rules_caches", {}):
                    cfg = load_config()
            self.assertTrue(cfg.enabled)
            self.assertEqual(cfg.policy, "allowlist")
            self.assertEqual(cfg.allow_from, frozenset(["ou_a"]))
            self.assertIn("*", cfg.documents)
            self.assertIn("docx:abc", cfg.documents)
            self.assertEqual(cfg.documents["docx:abc"].policy, "allowlist")
        finally:
            path.unlink()


class TestPairingStore(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._pairing_file = Path(self._tmpdir) / "pairing.json"
        with open(self._pairing_file, "w") as f:
            json.dump({"approved": {}}, f)
        self._patcher_file = patch(
            "plugins.platforms.feishu.feishu_comment_rules._pairing_file", return_value=self._pairing_file)
        self._patcher_caches = patch("plugins.platforms.feishu.feishu_comment_rules._pairing_caches", {})
        self._patcher_file.start()
        self._patcher_caches.start()

    def tearDown(self):
        self._patcher_caches.stop()
        self._patcher_file.stop()
        if self._pairing_file.exists():
            self._pairing_file.unlink()
        os.rmdir(self._tmpdir)

    def test_add_and_list(self):
        self.assertTrue(pairing_add("ou_new"))
        approved = pairing_list()
        self.assertIn("ou_new", approved)


class TestPerProfileScope(unittest.TestCase):
    """The rules/pairing caches must not freeze onto whichever profile's HERMES_HOME happened
    to trigger the module's first (lazy) import -- each profile's own files must stay isolated
    under multiplex. Regression for the #107620-adjacent gap left in #63962/#86905."""

    def setUp(self):
        self._tmpdir = Path(tempfile.mkdtemp())
        self._home_a = self._tmpdir / "profile_a"
        self._home_b = self._tmpdir / "profile_b"
        self._home_a.mkdir()
        self._home_b.mkdir()
        (self._home_a / "feishu_comment_rules.json").write_text(json.dumps({"policy": "allowlist"}))
        (self._home_b / "feishu_comment_rules.json").write_text(json.dumps({"policy": "pairing"}))
        import plugins.platforms.feishu.feishu_comment_rules as rules_mod
        self._rules_mod = rules_mod
        self._patcher_rules_caches = patch.object(rules_mod, "_rules_caches", {})
        self._patcher_pairing_caches = patch.object(rules_mod, "_pairing_caches", {})
        self._patcher_rules_caches.start()
        self._patcher_pairing_caches.start()
        self._current_home = self._home_a
        self._patcher_home = patch.object(rules_mod, "get_hermes_home", side_effect=lambda: self._current_home)
        # hermes_home_key() is imported directly into this module's namespace, so it must be
        # patched here too (patching hermes_constants.get_hermes_home wouldn't reach it) -- real
        # hermes_home_key(path=None) resolves via get_hermes_home() internally, which this test
        # cannot see through the patch above since it's a separate module-level binding.
        self._patcher_key = patch.object(
            rules_mod, "hermes_home_key", side_effect=lambda path=None: str(path or self._current_home))
        self._patcher_home.start()
        self._patcher_key.start()

    def tearDown(self):
        self._patcher_key.stop()
        self._patcher_home.stop()
        self._patcher_pairing_caches.stop()
        self._patcher_rules_caches.stop()
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_two_profiles_read_their_own_rules_file(self):
        self._current_home = self._home_a
        self.assertEqual(load_config().policy, "allowlist")
        self._current_home = self._home_b
        self.assertEqual(load_config().policy, "pairing")
        # Switching back to A must still see A's file, not a value cached under B's key.
        self._current_home = self._home_a
        self.assertEqual(load_config().policy, "allowlist")

    def test_two_profiles_have_independent_pairing_stores(self):
        self._current_home = self._home_a
        self.assertTrue(pairing_add("ou_a_only"))
        self._current_home = self._home_b
        self.assertNotIn("ou_a_only", pairing_list())
        self.assertTrue(pairing_add("ou_b_only"))
        self._current_home = self._home_a
        approved_a = pairing_list()
        self.assertIn("ou_a_only", approved_a)
        self.assertNotIn("ou_b_only", approved_a)


if __name__ == "__main__":
    unittest.main()
