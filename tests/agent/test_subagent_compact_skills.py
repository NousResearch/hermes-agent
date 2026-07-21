"""Subagent compact skill index (delegate_task skill scoping).

The bug class under test: a delegate_task child inherits the parent's ENTIRE
skills index — measured at 85%+ of a child's system prompt on a large skills
tree — for descriptions it almost never reads. The fix demotes every category
to names-only for subagents (the "*" sentinel), re-promoting only skills the
dispatching brief named via delegate_task(skills=[...]).

Contracts:
  1. "*" in compact_categories demotes EVERY category to names-only.
  2. promoted_skills re-promote named skills to full descriptions inside
     demoted categories; the rest of the category stays names-only.
  3. Nothing is hidden: every skill name still appears in the index.
  4. The compact index is dramatically smaller than the full index.
  5. Cache correctness: different promoted sets produce different outputs
     (the cache key includes promoted_skills).
"""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock


def _make_skills_tree(root: Path, n_categories: int = 4, per_cat: int = 6) -> None:
    for c in range(n_categories):
        cat = root / f"cat{c}"
        cat.mkdir(parents=True)
        for s in range(per_cat):
            d = cat / f"skill-{c}-{s}"
            d.mkdir()
            (d / "SKILL.md").write_text(
                "---\n"
                f"name: skill-{c}-{s}\n"
                f"description: \"A long, detailed description for skill {c}-{s} "
                "explaining endpoints, workflows, pitfalls, and conventions that "
                "costs real tokens in every prompt that includes it verbatim.\"\n"
                "---\n\n# body\n"
            )


class SubagentCompactIndexTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="subskills-test-"))
        _make_skills_tree(self.tmp)
        # Patch the module's skills-dir resolution + kill the caches.
        import agent.prompt_builder as pb

        self.pb = pb
        self._patches = [
            mock.patch.object(pb, "get_skills_dir", return_value=self.tmp),
            mock.patch.object(pb, "get_all_skills_dirs", return_value=[self.tmp]),
            mock.patch.object(pb, "get_disabled_skill_names", return_value=set()),
        ]
        for p in self._patches:
            p.start()
        pb.clear_skills_system_prompt_cache(clear_snapshot=True)

    def tearDown(self):
        for p in self._patches:
            p.stop()
        self.pb.clear_skills_system_prompt_cache(clear_snapshot=True)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _build(self, **kw):
        return self.pb.build_skills_system_prompt(**kw)

    def test_star_sentinel_demotes_everything(self):
        full = self._build()
        compact = self._build(compact_categories=frozenset({"*"}))
        # 1. every category demoted -> no description text survives
        self.assertNotIn("A long, detailed description", compact)
        self.assertIn("[names only]", compact)
        # 3. nothing hidden: every skill name still present
        for c in range(4):
            for s in range(6):
                self.assertIn(f"skill-{c}-{s}", compact)
        # 4. dramatically smaller (compare the index block, not the fixed
        # instructional header that dominates at this small fixture size)
        def _index_block(s: str) -> str:
            i = s.find("<available_skills>")
            return s[i:] if i >= 0 else s

        self.assertLess(len(_index_block(compact)), len(_index_block(full)) * 0.5)

    def test_promoted_skills_keep_descriptions(self):
        compact = self._build(
            compact_categories=frozenset({"*"}),
            promoted_skills=frozenset({"skill-1-2", "skill-3-0"}),
        )
        # promoted entries render with full descriptions
        self.assertIn("skill-1-2: A long, detailed description", compact)
        self.assertIn("skill-3-0: A long, detailed description", compact)
        # non-promoted siblings stay names-only
        self.assertNotIn("skill-1-3: A long, detailed description", compact)
        self.assertIn("skill-1-3", compact)  # but the name is still there

    def test_compact_footer_explains_the_contract(self):
        compact = self._build(compact_categories=frozenset({"*"}))
        self.assertIn("compact index", compact)
        self.assertIn("skill_view", compact)

    def test_cache_distinguishes_promoted_sets(self):
        a = self._build(compact_categories=frozenset({"*"}))
        b = self._build(
            compact_categories=frozenset({"*"}),
            promoted_skills=frozenset({"skill-0-0"}),
        )
        self.assertNotEqual(a, b)
        # and a repeat call returns the cached identical string
        a2 = self._build(compact_categories=frozenset({"*"}))
        self.assertEqual(a, a2)

    def test_category_demotion_still_works_unchanged(self):
        """The coding-posture path (named categories) must be untouched."""
        out = self._build(compact_categories=frozenset({"cat0"}))
        self.assertIn("cat0 [names only]", out)
        # other categories keep descriptions
        self.assertIn("skill-1-0: A long, detailed description", out)


if __name__ == "__main__":
    unittest.main()
