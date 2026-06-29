#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# ///
"""Unit tests for resolve_personas.py — pool merge, alias, party resolution."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import resolve_personas as rp  # noqa: E402

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


class TestBuildPool(unittest.TestCase):
    def test_installed_become_default_room_indexed_every_way(self):
        pool, idx, installed, custom = rp.build_pool(AGENTS, [])
        self.assertEqual(installed, ["bmad-agent-analyst", "bmad-agent-pm"])
        self.assertEqual(custom, [])
        self.assertEqual(idx["analyst"], "bmad-agent-analyst")   # alias
        self.assertEqual(idx["mary"], "bmad-agent-analyst")      # name (ci)
        self.assertEqual(pool["bmad-agent-analyst"]["source"], "installed")

    def test_pure_custom_member_stays_out_of_default_room(self):
        pool, _, installed, custom = rp.build_pool(
            AGENTS, [{"code": "morpheus", "name": "Morpheus", "persona": "riddles"}])
        self.assertEqual(custom, ["morpheus"])
        self.assertNotIn("morpheus", installed)
        self.assertEqual(pool["morpheus"]["persona"], "riddles")

    def test_custom_override_lands_on_installed_slot_not_a_new_face(self):
        pool, _, installed, custom = rp.build_pool(
            AGENTS, [{"code": "analyst", "name": "Mary-Custom", "persona": "p"}])
        self.assertNotIn("analyst", pool)
        self.assertEqual(custom, [])  # an override is not a new face
        self.assertEqual(pool["bmad-agent-analyst"]["source"], "custom")
        self.assertEqual(pool["bmad-agent-analyst"]["name"], "Mary-Custom")

    def test_member_without_code_skipped(self):
        pool, _, _, custom = rp.build_pool(AGENTS, [{"name": "Nameless"}])
        self.assertEqual(custom, [])
        self.assertEqual(set(pool), {"bmad-agent-analyst", "bmad-agent-pm"})

    def test_custom_rename_does_not_hijack_another_agents_name(self):
        # Override the analyst slot, renaming it to "John" — the PM's name.
        # The PM's name lookup must survive (last-writer-wins would corrupt it).
        _, idx, _, _ = rp.build_pool(AGENTS, [{"code": "analyst", "name": "John"}])
        self.assertEqual(idx["john"], "bmad-agent-pm")

    def test_brief_carries_model_and_capabilities(self):
        pool, _, _, _ = rp.build_pool(
            AGENTS, [{"code": "neo", "name": "Neo", "model": "opus", "capabilities": ["x"]}])
        brief = rp._brief(pool["neo"])
        self.assertEqual(brief["model"], "opus")
        self.assertEqual(brief["capabilities"], ["x"])

    def test_non_list_party_members_is_safe(self):
        pool, _, installed, custom = rp.build_pool(AGENTS, "not-a-list")
        self.assertEqual(custom, [])
        self.assertEqual(set(pool), {"bmad-agent-analyst", "bmad-agent-pm"})


class TestResolveParties(unittest.TestCase):
    def setUp(self):
        self.pool, self.idx, _, _ = rp.build_pool(
            AGENTS, [{"code": "shark", "name": "Marcus", "title": "CFO"}])

    def test_resolves_members_by_alias_and_custom_code(self):
        parties = rp.resolve_parties(
            [{"id": "tank", "name": "Tank", "scene": "hostile",
              "members": ["shark", "analyst"]}], self.pool, self.idx)
        self.assertEqual(len(parties), 1)
        self.assertEqual([m["name"] for m in parties[0]["members"]], ["Marcus", "Mary"])
        self.assertEqual(parties[0]["scene"], "hostile")

    def test_unknown_member_dropped_silently(self):
        parties = rp.resolve_parties(
            [{"id": "g", "members": ["analyst", "ghost"]}], self.pool, self.idx)
        self.assertEqual([m["name"] for m in parties[0]["members"]], ["Mary"])

    def test_member_resolution_is_case_insensitive(self):
        # A TOML author naturally writes "Analyst"/"Shark"; the filter accepts
        # them via the lowercase index, so resolution must too (no KeyError).
        parties = rp.resolve_parties(
            [{"id": "g", "members": ["Analyst", "Shark"]}], self.pool, self.idx)
        self.assertEqual([m["name"] for m in parties[0]["members"]], ["Mary", "Marcus"])

    def test_non_string_member_does_not_crash(self):
        # Malformed members (int, list) must drop silently, never raise.
        parties = rp.resolve_parties(
            [{"id": "g", "members": [123, ["x"], "analyst"]}], self.pool, self.idx)
        self.assertEqual([m["name"] for m in parties[0]["members"]], ["Mary"])

    def test_open_cast_group_flagged(self):
        parties = rp.resolve_parties(
            [{"id": "rebels", "name": "Rebels", "scene": "the Ghost"}], self.pool, self.idx)
        self.assertTrue(parties[0]["open_cast"])
        self.assertEqual(parties[0]["members"], [])

    def test_group_without_id_skipped(self):
        self.assertEqual(rp.resolve_parties([{"name": "no id"}], self.pool, self.idx), [])


class TestOverrideMergeFallback(unittest.TestCase):
    """When party-mode isn't installed, user override TOMLs are read directly."""

    def test_arrays_append_scalars_override(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            custom = Path(d) / "_bmad" / "custom"
            custom.mkdir(parents=True)
            (custom / "bmad-party-mode.toml").write_text(
                '[workflow]\ndefault_party = "a"\n'
                '[[workflow.party_members]]\ncode = "x"\nname = "X"\n')
            (custom / "bmad-party-mode.user.toml").write_text(
                '[workflow]\ndefault_party = "b"\n'
                '[[workflow.party_members]]\ncode = "y"\nname = "Y"\n')
            wf = rp.load_party_overrides(Path(d))
            self.assertEqual(wf["default_party"], "b")  # personal wins
            self.assertEqual([m["code"] for m in wf["party_members"]], ["x", "y"])  # appended


if __name__ == "__main__":
    unittest.main()
