#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# ///
"""Unit tests for resolve_party.py — merge, alias, override, group resolution."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import resolve_party as rp  # noqa: E402

AGENTS = {
    "bmad-agent-analyst": {"name": "Mary", "icon": "📊", "title": "Analyst"},
    "bmad-agent-pm": {"name": "John", "icon": "📋", "title": "PM"},
}


class TestAlias(unittest.TestCase):
    def test_strips_known_prefixes(self):
        self.assertEqual(rp._alias("bmad-agent-analyst"), "analyst")
        self.assertEqual(rp._alias("bmad-foo"), "foo")

    def test_passes_through_unprefixed(self):
        self.assertEqual(rp._alias("morpheus"), "morpheus")


class TestBuildCollective(unittest.TestCase):
    def test_installed_agents_indexed_by_code_alias_and_name(self):
        col, idx, _ = rp.build_collective(AGENTS, [])
        self.assertEqual(set(col), {"bmad-agent-analyst", "bmad-agent-pm"})
        self.assertEqual(idx["analyst"], "bmad-agent-analyst")      # alias
        self.assertEqual(idx["mary"], "bmad-agent-analyst")         # name (ci)
        self.assertEqual(idx["bmad-agent-pm"], "bmad-agent-pm")     # full code
        self.assertEqual(col["bmad-agent-analyst"]["source"], "installed")

    def test_custom_member_appends(self):
        col, _, _ = rp.build_collective(AGENTS, [{"code": "morpheus", "name": "Morpheus", "persona": "riddles"}])
        self.assertIn("morpheus", col)
        self.assertEqual(col["morpheus"]["source"], "custom")
        self.assertEqual(col["morpheus"]["persona"], "riddles")

    def test_custom_overrides_installed_by_alias(self):
        col, _, _ = rp.build_collective(AGENTS, [{"code": "analyst", "name": "Mary-Custom", "persona": "p"}])
        # Override lands on the canonical installed code, not a new "analyst" entry.
        self.assertNotIn("analyst", col)
        self.assertEqual(col["bmad-agent-analyst"]["source"], "custom")
        self.assertEqual(col["bmad-agent-analyst"]["name"], "Mary-Custom")

    def test_member_without_code_skipped(self):
        col, _, _ = rp.build_collective(AGENTS, [{"name": "Nameless"}])
        self.assertEqual(set(col), {"bmad-agent-analyst", "bmad-agent-pm"})


class TestResolveMembers(unittest.TestCase):
    def setUp(self):
        self.col, self.idx, _ = rp.build_collective(AGENTS, [{"code": "morpheus", "name": "Morpheus"}])

    def test_resolves_in_listed_order_and_flags_unknowns(self):
        resolved, unresolved = rp.resolve_members(["morpheus", "analyst", "ghost"], self.col, self.idx)
        self.assertEqual([m["code"] for m in resolved], ["morpheus", "bmad-agent-analyst"])
        self.assertEqual(unresolved, ["ghost"])

    def test_empty(self):
        self.assertEqual(rp.resolve_members([], self.col, self.idx), ([], []))


class TestGroups(unittest.TestCase):
    GROUPS = [
        {"id": "wr", "name": "Writers", "members": ["analyst", "morpheus"]},
        {"id": "bad"},  # no name -> falls back to id; no members -> count 0
        {"name": "no-id"},  # dropped from menu
    ]

    def test_menu_is_names_only_with_counts_and_open_cast_flag(self):
        menu = rp.group_menu(self.GROUPS)
        self.assertEqual(menu, [
            {"id": "wr", "name": "Writers", "member_count": 2},
            {"id": "bad", "name": "bad", "member_count": 0, "open_cast": True},
        ])

    def test_find_group(self):
        self.assertEqual(rp.find_group(self.GROUPS, "wr")["name"], "Writers")
        self.assertIsNone(rp.find_group(self.GROUPS, "missing"))


class TestGroupDetail(unittest.TestCase):
    def setUp(self):
        self.col, self.idx, _ = rp.build_collective(AGENTS, [{"code": "morpheus", "name": "Morpheus"}])

    def test_scene_passes_through_when_present(self):
        g = {"id": "tos-10-forward", "name": "Ten Forward", "members": ["morpheus"],
             "scene": "Late evening, a few rounds in."}
        d = rp.group_detail(g, self.col, self.idx)
        self.assertEqual(d["scene"], "Late evening, a few rounds in.")
        self.assertEqual([m["code"] for m in d["members"]], ["morpheus"])

    def test_scene_omitted_when_absent_or_empty(self):
        for g in ({"id": "g", "members": ["morpheus"]},
                  {"id": "g", "members": ["morpheus"], "scene": ""}):
            self.assertNotIn("scene", rp.group_detail(g, self.col, self.idx))

    def test_anchored_group_is_not_open_cast(self):
        g = {"id": "g", "members": ["morpheus"]}
        self.assertNotIn("open_cast", rp.group_detail(g, self.col, self.idx))

    def test_open_cast_group_flagged_with_empty_members(self):
        g = {"id": "rebels", "name": "Star Wars Rebels",
             "scene": "Figures from the Rebels universe drop in as the topic calls for them."}
        d = rp.group_detail(g, self.col, self.idx)
        self.assertTrue(d["open_cast"])
        self.assertEqual(d["members"], [])
        self.assertEqual(d["scene"][:7], "Figures")

    def test_memory_enabled_follows_group_flag_and_defaults_off(self):
        on = rp.group_detail({"id": "g", "members": ["morpheus"], "memory": True}, self.col, self.idx)
        self.assertTrue(on["memory_enabled"])
        off = rp.group_detail({"id": "g", "members": ["morpheus"], "memory": False}, self.col, self.idx)
        self.assertFalse(off["memory_enabled"])
        absent = rp.group_detail({"id": "g", "members": ["morpheus"]}, self.col, self.idx)
        self.assertFalse(absent["memory_enabled"])  # opt-in per named group


class TestInstalledCodesIsDefaultRoom(unittest.TestCase):
    """The default room is installed agents only; pure customs stay in the pool."""

    def test_pure_custom_excluded_override_kept_in_default_room(self):
        col, _, installed = rp.build_collective(AGENTS, [
            {"code": "morpheus", "name": "Morpheus"},                 # pure custom
            {"code": "analyst", "name": "Mary-Custom", "persona": "p"},  # override
            {"code": "sec-hawk", "name": "Vex"},                      # shipped crew member
        ])
        # Pure customs are in the pool...
        self.assertIn("morpheus", col)
        self.assertIn("sec-hawk", col)
        # ...but NOT in the default room.
        self.assertEqual(installed, ["bmad-agent-analyst", "bmad-agent-pm"])
        default_room = [col[c]["code"] for c in installed]
        self.assertEqual(default_room, ["bmad-agent-analyst", "bmad-agent-pm"])
        # An override keeps its installed slot (and its custom content).
        self.assertEqual(col["bmad-agent-analyst"]["name"], "Mary-Custom")


if __name__ == "__main__":
    unittest.main()
