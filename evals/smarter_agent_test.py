#!/usr/bin/env python3
"""Test harness for the 4 smarter-agent improvements.

Covers:
  1. Semantic Tool Call Repair
  2. Smarter Memory Nudge
  3. Compression Preserves Corrections
  4. Predictive Delegation Skill

Usage:
    cd /home/gfive/.hermes/hermes-agent && python evals/smarter_agent_test.py

Each scenario defines input, expected behavior, and success criteria.
A baseline mode (--baseline) toggles improvements off for comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tool_call(name: str, arguments: dict | str) -> MagicMock:
    """Create a mock tool-call object matching the OpenAI-style structure."""
    tc = MagicMock()
    tc.function = MagicMock()
    tc.function.name = name
    if isinstance(arguments, dict):
        tc.function.arguments = json.dumps(arguments)
    else:
        tc.function.arguments = arguments
    return tc


def _make_agent(workspace_root: str = "/tmp/test-workspace") -> MagicMock:
    """Create a minimal mock agent for testing."""
    agent = MagicMock()
    agent.workspace_root = workspace_root
    agent.working_directory = workspace_root
    agent.cwd = workspace_root
    agent._buffer_vprint = MagicMock()
    agent._memory_nudge_interval = 10
    agent._turns_since_memory = 9  # Next tick triggers
    agent._memory_store = True
    agent.valid_tool_names = {"memory", "read_file", "write_file"}
    agent._pending_preference_distill = False
    return agent


# ===========================================================================
# Improvement 1: Semantic Tool Call Repair
# ===========================================================================

class TestSemanticToolCallRepair(unittest.TestCase):
    """Tests for agent/tool_call_repair.py integration."""

    def setUp(self):
        from agent.tool_call_repair import repair_tool_call_arguments, score_message_correction_weight
        self.repair = repair_tool_call_arguments
        self.score = score_message_correction_weight

    # -- Scenario 1: String-to-int coercion ----------------------------------
    def test_string_to_int_coercion(self):
        """Tool call with string '5' for an int param should be coerced."""
        agent = _make_agent()
        tc = _make_tool_call("terminal", {"command": "ls", "timeout": "30"})
        messages = []

        repairs = self.repair(agent, [tc], messages)

        args = json.loads(tc.function.arguments)
        self.assertEqual(args["timeout"], 30)
        self.assertGreaterEqual(repairs, 1)

    # -- Scenario 2: Boolean coercion ----------------------------------------
    def test_boolean_coercion(self):
        """String 'true'/'false' should be coerced to bool."""
        agent = _make_agent()
        tc = _make_tool_call("search", {"query": "test", "case_sensitive": "true"})
        messages = []

        repairs = self.repair(agent, [tc], messages)

        args = json.loads(tc.function.arguments)
        self.assertIs(args["case_sensitive"], True)
        self.assertGreaterEqual(repairs, 1)

    # -- Scenario 3: Correction scoring --------------------------------------
    def test_correction_scoring_detects_negative_feedback(self):
        """Messages with corrections should score > 1.0."""
        normal = self.score("Please read that file for me")
        correction = self.score("Don't do that, use the other approach")
        strong = self.score("No, stop doing that! That's wrong, I said use grep!")

        self.assertEqual(normal, 1.0)
        self.assertGreater(correction, 1.0)
        self.assertGreater(strong, correction)
        self.assertLessEqual(strong, 5.0)  # Cap at 5x


# ===========================================================================
# Improvement 2: Smarter Memory Nudge
# ===========================================================================

class TestSmarterMemoryNudge(unittest.TestCase):
    """Tests for the enhanced memory nudge in turn_finalizer.py."""

    # -- Scenario 4: Correction signals trigger preference distillation ------
    def test_correction_signals_set_preference_distill_flag(self):
        """When recent turns contain >= 2 correction signals, flag is set."""
        from agent.tool_call_repair import score_message_correction_weight

        messages = [
            {"role": "user", "content": "Read the config file"},
            {"role": "assistant", "content": "Here it is..."},
            {"role": "user", "content": "Don't include the secrets section"},
            {"role": "assistant", "content": "Fixed."},
            {"role": "user", "content": "No, stop doing that — never show passwords"},
            {"role": "assistant", "content": "Understood."},
        ]

        # Simulate the logic from turn_finalizer.py enhancement
        recent_user_msgs = [
            m for m in messages[-20:]
            if isinstance(m, dict) and m.get("role") == "user"
        ]
        correction_signals = sum(
            1 for m in recent_user_msgs
            if score_message_correction_weight(m.get("content", "")) > 1.5
        )

        self.assertGreaterEqual(correction_signals, 2)

    # -- Scenario 5: No false positive on normal conversation ----------------
    def test_normal_conversation_no_distill_flag(self):
        """Normal conversation without corrections should NOT trigger."""
        from agent.tool_call_repair import score_message_correction_weight

        messages = [
            {"role": "user", "content": "What time is it?"},
            {"role": "assistant", "content": "It's 3pm."},
            {"role": "user", "content": "Thanks, can you also check the weather?"},
        ]

        recent_user_msgs = [
            m for m in messages[-20:]
            if isinstance(m, dict) and m.get("role") == "user"
        ]
        correction_signals = sum(
            1 for m in recent_user_msgs
            if score_message_correction_weight(m.get("content", "")) > 1.5
        )

        self.assertEqual(correction_signals, 0)


# ===========================================================================
# Improvement 3: Compression Preserves Corrections
# ===========================================================================

class TestCompressionPreservesCorrections(unittest.TestCase):
    """Tests for correction tagging/stripping in context_compressor.py."""

    def setUp(self):
        from agent.context_compressor import (
            _tag_user_corrections,
            _strip_correction_markers,
            _USER_CORRECTION_MARKER,
        )
        self.tag = _tag_user_corrections
        self.strip = _strip_correction_markers
        self.marker = _USER_CORRECTION_MARKER

    # -- Scenario 6: Tagging adds markers to correction messages -------------
    def test_tagging_marks_correction_messages(self):
        """User messages with corrections get tagged; others don't."""
        messages = [
            {"role": "user", "content": "Read the file please"},
            {"role": "user", "content": "Don't do that, use cat instead"},
            {"role": "assistant", "content": "OK"},
            {"role": "user", "content": "That's wrong, I said use grep"},
        ]

        tagged_count = self.tag(messages)

        self.assertEqual(tagged_count, 2)
        self.assertTrue(messages[1]["content"].startswith(self.marker))
        self.assertTrue(messages[3]["content"].startswith(self.marker))
        self.assertFalse(messages[0]["content"].startswith(self.marker))

    # -- Scenario 7: Idempotent tagging --------------------------------------
    def test_tagging_is_idempotent(self):
        """Running tag twice should not double-tag."""
        messages = [
            {"role": "user", "content": "Never show raw passwords"},
        ]

        first = self.tag(messages)
        second = self.tag(messages)

        self.assertEqual(first, 1)
        self.assertEqual(second, 0)
        # Should have exactly one marker prefix
        self.assertEqual(
            messages[0]["content"].count(self.marker), 1
        )

    # -- Scenario 8: Stripping removes all markers ---------------------------
    def test_stripping_removes_markers(self):
        """After stripping, no markers remain in any message."""
        messages = [
            {"role": "user", "content": f"{self.marker} Don't do that"},
            {"role": "user", "content": "Normal message"},
            {"role": "user", "content": f"{self.marker} Stop using eval"},
        ]

        self.strip(messages)

        for msg in messages:
            self.assertNotIn(self.marker, msg.get("content", ""))
        # Original content preserved
        self.assertEqual(messages[0]["content"], "Don't do that")
        self.assertEqual(messages[1]["content"], "Normal message")


# ===========================================================================
# Improvement 4: Predictive Delegation Skill
# ===========================================================================

class TestPredictiveDelegationSkill(unittest.TestCase):
    """Tests for the predictive-delegation skill existence and structure."""

    # -- Scenario 9: Skill file exists with valid frontmatter ----------------
    def test_skill_file_exists_and_has_frontmatter(self):
        """SKILL.md must exist with required YAML frontmatter fields."""
        skill_path = _PROJECT_ROOT / "skills" / "software-development" / "predictive-delegation" / "SKILL.md"
        self.assertTrue(skill_path.exists(), f"Missing: {skill_path}")

        content = skill_path.read_text()
        # Must have YAML frontmatter
        self.assertTrue(content.startswith("---"), "Missing YAML frontmatter")
        parts = content.split("---", 2)
        self.assertGreaterEqual(len(parts), 3, "Malformed frontmatter")

        frontmatter = parts[1]
        self.assertIn("name:", frontmatter)
        self.assertIn("predictive-delegation", frontmatter)
        self.assertIn("description:", frontmatter)

    # -- Scenario 10: Skill contains trigger conditions and anti-patterns ----
    def test_skill_contains_required_sections(self):
        """Skill must document trigger conditions, decision framework, anti-patterns."""
        skill_path = _PROJECT_ROOT / "skills" / "software-development" / "predictive-delegation" / "SKILL.md"
        content = skill_path.read_text().lower()

        required_sections = [
            "trigger condition",
            "when not to use",
            "decision framework",
            "anti-pattern",
            "delegate_task",
        ]
        for section in required_sections:
            self.assertIn(section, content, f"Missing section: {section}")


# ===========================================================================
# Integration: Turn Tool Validation Hook
# ===========================================================================

class TestTurnToolValidationIntegration(unittest.TestCase):
    """Verify the repair hook is wired into turn_tool_validation."""

    def test_repair_import_in_validation_module(self):
        """turn_tool_validation.py must import and call repair_tool_call_arguments."""
        validation_path = _PROJECT_ROOT / "agent" / "turn_tool_validation.py"
        content = validation_path.read_text()

        self.assertIn("repair_tool_call_arguments", content)
        self.assertIn("tool_call_repair", content)


# ===========================================================================
# Runner
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Smarter Agent Test Harness")
    parser.add_argument(
        "--baseline", action="store_true",
        help="Baseline mode: skip improvement-specific tests (for comparison)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose test output",
    )
    args = parser.parse_args()

    if args.baseline:
        print("=" * 60)
        print("BASELINE MODE: Running only structural checks")
        print("=" * 60)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Always run structural/integration checks
    suite.addTests(loader.loadTestsFromTestCase(TestPredictiveDelegationSkill))
    suite.addTests(loader.loadTestsFromTestCase(TestTurnToolValidationIntegration))

    if not args.baseline:
        suite.addTests(loader.loadTestsFromTestCase(TestSemanticToolCallRepair))
        suite.addTests(loader.loadTestsFromTestCase(TestSmarterMemoryNudge))
        suite.addTests(loader.loadTestsFromTestCase(TestCompressionPreservesCorrections))

    verbosity = 2 if args.verbose else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 60)
    total = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    passed = total - failures - errors
    print(f"Results: {passed}/{total} passed, {failures} failed, {errors} errors")
    if args.baseline:
        print("(Baseline mode — improvement tests skipped)")
    print("=" * 60)

    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()