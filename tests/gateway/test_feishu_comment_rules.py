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
            with patch("plugins.platforms.feishu.feishu_comment_rules.RULES_FILE", path):
                with patch("plugins.platforms.feishu.feishu_comment_rules._rules_cache", _MtimeCache(path)):
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
        self._patcher_file = patch("plugins.platforms.feishu.feishu_comment_rules.PAIRING_FILE", self._pairing_file)
        self._patcher_cache = patch(
            "plugins.platforms.feishu.feishu_comment_rules._pairing_cache",
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


class TestRulesFollowActiveProfile(unittest.TestCase):
    """The multiplexed gateway serves every profile from one process: the rules/pairing files and
    their mtime caches must follow the context-local HERMES_HOME override, one slot per profile."""

    def test_rules_and_pairing_follow_home_override(self):
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override
        from plugins.platforms.feishu import feishu_comment_rules as fcr

        def under(home, fn):
            token = set_hermes_home_override(str(home))
            try:
                return fn()
            finally:
                reset_hermes_home_override(token)

        with tempfile.TemporaryDirectory() as tmp:
            prof_a, prof_b = Path(tmp) / "A", Path(tmp) / "B"
            for home, enabled in ((prof_a, True), (prof_b, False)):
                home.mkdir()
                (home / "feishu_comment_rules.json").write_text(json.dumps({"enabled": enabled}))
            self.assertTrue(under(prof_a, fcr.load_config).enabled)  # warm A's slot
            self.assertFalse(under(prof_b, fcr.load_config).enabled)
            self.assertTrue(under(prof_a, fcr.load_config).enabled)  # A's slot survives B
            self.assertTrue(under(prof_b, lambda: fcr.pairing_add("ou_b")))
            self.assertTrue((prof_b / "feishu_comment_pairing.json").exists())
            self.assertFalse((prof_a / "feishu_comment_pairing.json").exists())
            self.assertNotIn("ou_b", under(prof_a, fcr.pairing_list))


if __name__ == "__main__":
    unittest.main()
