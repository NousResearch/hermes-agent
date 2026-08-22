"""Skill-declared trigger phrases route free text to a skill's slash command.

The bug this closes: free text at the prompt is routed by the model, so a
skill invocation competes with every other skill for the model's attention.
A cheap model loses that race sometimes, and the failure is silent — a
plausible answer from the wrong skill rather than an error.
"""

import unittest

from agent.skill_commands import _normalize_triggers, match_skill_trigger


def _cmds(mapping):
    """Build a minimal scan_skill_commands()-shaped map."""
    return {
        key: {"name": key.lstrip("/"), "description": "", "triggers": triggers}
        for key, triggers in mapping.items()
    }


class TestNormalizeTriggers(unittest.TestCase):
    def test_absent_yields_empty(self):
        self.assertEqual(_normalize_triggers({}), [])
        self.assertEqual(_normalize_triggers({"triggers": None}), [])

    def test_scalar_is_accepted_as_one_trigger(self):
        self.assertEqual(_normalize_triggers({"triggers": "list trips"}), ["list trips"])

    def test_lowercased_and_whitespace_collapsed(self):
        got = _normalize_triggers({"triggers": ["  List   Trips  "]})
        self.assertEqual(got, ["list trips"])

    def test_short_triggers_rejected(self):
        # "go" would capture an enormous amount of ordinary prose.
        got = _normalize_triggers({"triggers": ["go", "hi", "brief"]})
        self.assertEqual(got, ["brief"])

    def test_duplicates_collapsed(self):
        got = _normalize_triggers({"triggers": ["brief", "BRIEF", " brief "]})
        self.assertEqual(got, ["brief"])

    def test_non_string_entries_ignored(self):
        got = _normalize_triggers({"triggers": ["brief", 42, None, {"a": 1}]})
        self.assertEqual(got, ["brief"])

    def test_malformed_type_yields_empty(self):
        self.assertEqual(_normalize_triggers({"triggers": 42}), [])


class TestMatchSkillTrigger(unittest.TestCase):
    def setUp(self):
        self.commands = _cmds({"/trip-brief": ["brief", "list trips"]})

    def test_exact_trigger_matches(self):
        self.assertEqual(
            match_skill_trigger("list trips", self.commands),
            "/trip-brief list trips",
        )

    def test_trigger_with_trailing_text_matches(self):
        self.assertEqual(
            match_skill_trigger("brief the wine trip", self.commands),
            "/trip-brief brief the wine trip",
        )

    def test_case_and_spacing_insensitive(self):
        self.assertEqual(
            match_skill_trigger("  LIST   TRIPS  ", self.commands),
            "/trip-brief LIST   TRIPS",
        )

    def test_original_text_is_preserved_in_rewrite(self):
        # The skill receives what the user actually typed, not the normalized
        # form used for matching.
        out = match_skill_trigger("Brief The Wine Trip", self.commands)
        self.assertEqual(out, "/trip-brief Brief The Wine Trip")

    # --- the safety cases -------------------------------------------------

    def test_word_prefix_does_not_match(self):
        # "briefing" is not "brief"; hijacking it would be a regression.
        self.assertIsNone(match_skill_trigger("briefing notes on Q3", self.commands))

    def test_trigger_mid_sentence_does_not_match(self):
        # Only a leading trigger is an invocation. Otherwise any sentence
        # mentioning the word gets captured.
        self.assertIsNone(
            match_skill_trigger("can you brief me on the merger", self.commands)
        )

    def test_unrelated_text_untouched(self):
        self.assertIsNone(
            match_skill_trigger("what restaurants are near the hotel", self.commands)
        )

    def test_existing_slash_command_untouched(self):
        self.assertIsNone(match_skill_trigger("/help", self.commands))
        self.assertIsNone(match_skill_trigger("  /trip-brief 2", self.commands))

    def test_empty_and_non_string_input(self):
        self.assertIsNone(match_skill_trigger("", self.commands))
        self.assertIsNone(match_skill_trigger("   ", self.commands))
        self.assertIsNone(match_skill_trigger(None, self.commands))
        self.assertIsNone(match_skill_trigger(12345, self.commands))

    def test_no_skills_declare_triggers_is_inert(self):
        # The default state of every existing install.
        inert = _cmds({"/a": [], "/b": None})
        self.assertIsNone(match_skill_trigger("brief the wine trip", inert))

    def test_triggers_are_literal_not_regex(self):
        # A skill is untrusted input; a trigger must never compile as a
        # pattern or any skill could declare ".*" and capture everything.
        commands = _cmds({"/regexy": [".*", "a+b"]})
        self.assertIsNone(match_skill_trigger("anything at all", commands))
        self.assertEqual(
            match_skill_trigger("a+b something", commands), "/regexy a+b something"
        )

    # --- precedence -------------------------------------------------------

    def test_longest_trigger_wins(self):
        commands = _cmds({"/deploy": ["deploy"], "/deploy-staging": ["deploy staging"]})
        self.assertEqual(
            match_skill_trigger("deploy staging now", commands),
            "/deploy-staging deploy staging now",
        )
        self.assertEqual(
            match_skill_trigger("deploy prod now", commands),
            "/deploy deploy prod now",
        )

    def test_equal_length_ties_resolve_to_first_slug(self):
        # Scan order varies with the filesystem; the winner must not.
        a = _cmds({"/aaa": ["build"], "/zzz": ["build"]})
        b = _cmds({"/zzz": ["build"], "/aaa": ["build"]})
        self.assertEqual(match_skill_trigger("build it", a), "/aaa build it")
        self.assertEqual(match_skill_trigger("build it", b), "/aaa build it")

    def test_non_dict_command_entry_is_skipped(self):
        commands = {"/junk": "not a dict", "/ok": {"triggers": ["brief"]}}
        self.assertEqual(match_skill_trigger("brief x", commands), "/ok brief x")

    def test_malformed_command_entries_are_survivable(self):
        commands = {"/broken": {}, "/ok": {"triggers": ["brief"]}}
        self.assertEqual(match_skill_trigger("brief x", commands), "/ok brief x")


if __name__ == "__main__":
    unittest.main()
