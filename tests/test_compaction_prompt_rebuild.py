"""Compaction ALWAYS rebuilds the system prompt from the live builder (#95681).

The old keep-prompt containment branch restored the stored bytes whenever the
reloaded memory blocks were embedded — so prompt-builder changes (guidance
diets, renames, new blocks) never reached long-lived sessions (Bot Mode
forever-chats, gateway channels). New contract:

1. builder output byte-equal  -> keep the ORIGINAL string object (identity
   preserved for KV/prefix caches keyed on it)
2. builder output differs     -> the rebuilt prompt wins, logged
3. plugin sections re-render at the same boundary; a RAISING plugin falls
   back to its last good bytes (fail-open), never silently vanishes
"""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent.system_prompt import invalidate_system_prompt


def _agent(**over):
    base = dict(
        _cached_system_prompt="OLD PROMPT",
        _cached_system_prompt_static="OLD",
        _memory_store=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


class TestInvalidateClearsPluginFreeze(unittest.TestCase):
    def test_invalidate_stashes_and_clears_plugin_snapshot(self):
        agent = _agent()
        agent._plugin_system_prompt_sections_snapshot = ("frozen-section",)
        invalidate_system_prompt(agent)
        self.assertFalse(hasattr(agent, "_plugin_system_prompt_sections_snapshot"))
        self.assertEqual(agent._plugin_system_prompt_sections_previous, ("frozen-section",))
        self.assertIsNone(agent._cached_system_prompt)

    def test_invalidate_without_snapshot_is_noop_for_plugins(self):
        agent = _agent()
        invalidate_system_prompt(agent)
        self.assertFalse(hasattr(agent, "_plugin_system_prompt_sections_snapshot"))


class TestPluginRerenderFailOpen(unittest.TestCase):
    def test_raising_plugin_render_falls_back_to_previous_bytes(self):
        from agent.system_prompt import _frozen_plugin_prompt_sections

        agent = _agent(_cached_system_prompt=None)
        agent._plugin_system_prompt_sections_previous = ("last-good",)
        with patch("hermes_cli.plugins.render_system_prompt_sections",
                   side_effect=RuntimeError("plugin exploded")):
            rendered = _frozen_plugin_prompt_sections(agent)
        self.assertEqual(rendered, ("last-good",))

    def test_raising_plugin_render_without_previous_is_empty(self):
        from agent.system_prompt import _frozen_plugin_prompt_sections

        agent = _agent(_cached_system_prompt=None)
        with patch("hermes_cli.plugins.render_system_prompt_sections",
                   side_effect=RuntimeError("plugin exploded")):
            rendered = _frozen_plugin_prompt_sections(agent)
        self.assertEqual(rendered, ())


class TestCommitAlwaysRebuilds(unittest.TestCase):
    """Source-level contract pins for the commit-site semantics."""

    def _src(self):
        import inspect
        from agent import conversation_compression as cc
        return inspect.getsource(cc)

    def test_keep_prompt_branch_requires_byte_equality(self):
        src = self._src()
        i = src.find("rebuilt_system_prompt = agent._build_system_prompt(")
        self.assertGreater(i, 0, "commit site must always run the live builder")
        window = src[i:i + 900]
        self.assertIn("rebuilt_system_prompt == cached_system_prompt", window,
                      "keep-prompt must be gated on BYTE EQUALITY of the "
                      "rebuilt output, not on memory containment")
        self.assertNotIn("_cached_prompt_reflects_builtin_memory(agent, cached_system_prompt)",
                         window,
                         "the containment keep-prompt gate must not return")

    def test_drift_rebuild_is_logged(self):
        src = self._src()
        self.assertIn("Compaction rebuilt a drifted system prompt", src)


if __name__ == "__main__":
    unittest.main()
