"""Hot-reload tests for feishu_group_rules (per-chat overlay on boot-time group_rules)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from plugins.platforms.feishu.feishu_comment_rules import _MtimeCache
from plugins.platforms.feishu.feishu_group_rules import (
    GROUP_RULES_FILE,
    _validate_entry,
    load_group_rules,
)
from tests.gateway.feishu_helpers import make_adapter_skeleton, make_sender


class TestValidateEntry(unittest.TestCase):
    def test_full_entry_normalized(self):
        entry = _validate_entry(
            "oc_1",
            {
                "policy": "Allowlist",
                "allowlist": ["ou_a", " ou_b ", ""],
                "blacklist": ["ou_x"],
                "require_mention": "false",
            },
        )
        self.assertEqual(entry["policy"], "allowlist")
        self.assertEqual(entry["allowlist"], {"ou_a", "ou_b"})
        self.assertEqual(entry["blacklist"], {"ou_x"})
        self.assertIs(entry["require_mention"], False)

    def test_partial_entry_keeps_only_present_keys(self):
        entry = _validate_entry("oc_1", {"require_mention": True})
        self.assertEqual(entry, {"require_mention": True})

    def test_unknown_policy_raises(self):
        with self.assertRaises(ValueError):
            _validate_entry("oc_1", {"policy": "yolo"})

    def test_non_list_allowlist_raises(self):
        with self.assertRaises(ValueError):
            _validate_entry("oc_1", {"allowlist": "ou_a"})

    def test_non_object_entry_raises(self):
        with self.assertRaises(ValueError):
            _validate_entry("oc_1", ["policy"])


class TestLoadGroupRules(unittest.TestCase):
    def _write(self, path: Path, payload) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    def test_malformed_entries_skipped_valid_served(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "group_rules.json"
            self._write(
                path,
                {
                    "oc_good": {"policy": "open", "require_mention": False},
                    "oc_bad_policy": {"policy": "nope"},
                    "oc_bad_list": {"allowlist": "ou_a"},
                    "oc_not_object": True,
                },
            )
            with (
                patch(
                    "plugins.platforms.feishu.feishu_group_rules.GROUP_RULES_FILE", path
                ),
                patch(
                    "plugins.platforms.feishu.feishu_group_rules._group_rules_cache",
                    _MtimeCache(path),
                ),
            ):
                rules = load_group_rules()
        self.assertEqual(list(rules), ["oc_good"])
        self.assertIs(rules["oc_good"]["require_mention"], False)

    def test_missing_file_yields_no_rules(self):
        with (
            patch(
                "plugins.platforms.feishu.feishu_group_rules.GROUP_RULES_FILE",
                Path("/nonexistent/g.json"),
            ),
            patch(
                "plugins.platforms.feishu.feishu_group_rules._group_rules_cache",
                _MtimeCache(Path("/nonexistent/g.json")),
            ),
        ):
            self.assertEqual(load_group_rules(), {})

    def test_edit_reread_on_mtime_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "group_rules.json"
            self._write(path, {"oc_1": {"require_mention": True}})
            os.utime(path, (1_000_000, 1_000_000))
            with (
                patch(
                    "plugins.platforms.feishu.feishu_group_rules.GROUP_RULES_FILE", path
                ),
                patch(
                    "plugins.platforms.feishu.feishu_group_rules._group_rules_cache",
                    _MtimeCache(path),
                ),
            ):
                self.assertIs(load_group_rules()["oc_1"]["require_mention"], True)
                self._write(path, {"oc_1": {"require_mention": False}})
                os.utime(path, (2_000_000, 2_000_000))
                self.assertIs(load_group_rules()["oc_1"]["require_mention"], False)


class TestEffectiveGroupRule(unittest.TestCase):
    def _adapter(self, **kwargs):
        from plugins.platforms.feishu.adapter import FeishuGroupRule

        adapter = make_adapter_skeleton(group_policy="allowlist")
        adapter._group_rules = {
            "oc_1": FeishuGroupRule(
                policy="allowlist",
                allowlist={"ou_alice"},
                blacklist=set(),
                require_mention=True,
            ),
        }
        return adapter

    def test_hot_require_mention_flips_mention_gate_without_restart(self):
        adapter = self._adapter()
        self.assertTrue(adapter._require_mention_for("oc_1"))
        with patch(
            "plugins.platforms.feishu.feishu_group_rules.load_group_rules",
            return_value={"oc_1": {"require_mention": False}},
        ):
            self.assertFalse(adapter._require_mention_for("oc_1"))
        # Overlay is read per admission: reverting the file restores boot behavior.
        self.assertTrue(adapter._require_mention_for("oc_1"))

    def test_partial_hot_entry_merges_into_boot_rule(self):
        adapter = self._adapter()
        with patch(
            "plugins.platforms.feishu.feishu_group_rules.load_group_rules",
            return_value={"oc_1": {"require_mention": False}},
        ):
            sender = make_sender(open_id="ou_alice")
            self.assertTrue(adapter._allow_group_message(sender.sender_id, "oc_1"))
            stranger = make_sender(open_id="ou_charlie")
            self.assertFalse(adapter._allow_group_message(stranger.sender_id, "oc_1"))

    def test_hot_entry_can_add_rule_for_unknown_chat(self):
        adapter = self._adapter()
        with patch(
            "plugins.platforms.feishu.feishu_group_rules.load_group_rules",
            return_value={"oc_new": {"policy": "disabled"}},
        ):
            sender = make_sender(open_id="ou_anyone")
            self.assertFalse(adapter._allow_group_message(sender.sender_id, "oc_new"))
        # Without the file, the chat falls back to the default policy.
        self.assertFalse(adapter._allow_group_message(sender.sender_id, "oc_new"))

    def test_hot_loader_failure_keeps_boot_rules(self):
        adapter = self._adapter()
        with patch(
            "plugins.platforms.feishu.feishu_group_rules.load_group_rules",
            side_effect=RuntimeError("boom"),
        ):
            self.assertTrue(adapter._require_mention_for("oc_1"))
            sender = make_sender(open_id="ou_alice")
            self.assertTrue(adapter._allow_group_message(sender.sender_id, "oc_1"))

    def test_empty_chat_id_never_touches_hot_rules(self):
        adapter = self._adapter()
        with patch(
            "plugins.platforms.feishu.feishu_group_rules.load_group_rules"
        ) as loader:
            self.assertIsNone(adapter._effective_group_rule(""))
            loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
