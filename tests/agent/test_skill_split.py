"""Oversized-skill split tests (agent/skill_split.py).

Covers spec Phase 4 + Invariant 5:
- 130 KB fixture splits under the threshold with carve files created
- ORDERED-concatenation content equality (splice reconstruction), not a set
- frontmatter-identity (bytes unchanged)
- reversibility via the split manifest (join == original, byte-for-byte)
- heading-less blob falls back to part-NN.md, still content-preserving
- carve-slug collision with an existing references/ file is deduped
- terminal failure leaves the skill unsplit (never mangles frontmatter)
- pointer whitelist is the FIXED template regex (bounded, count-checked)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent import skill_split


FM = "---\nname: big-skill\ndescription: a big one\n---\n"


def _mk_body(sections: int = 30, sec_kb: int = 5) -> str:
    parts = ["Intro paragraph explaining the class-level idea.\n\n"]
    filler = ("lorem ipsum dolor sit amet " * 40).strip()
    for i in range(sections):
        title = f"pending-lesson-{i:02d}" if i % 3 == 0 else f"Topic {i:02d}"
        block = f"## {title}\n\n"
        while len(block) < sec_kb * 1024:
            block += filler + f" [{i}]\n"
        parts.append(block)
    return "".join(parts)


def _mk_skill(tmp_path: Path, text: str, name: str = "big-skill") -> Path:
    d = tmp_path / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(text, encoding="utf-8")
    return d


class TestPlanAndExecute:
    def test_under_threshold_skips(self, tmp_path):
        d = _mk_skill(tmp_path, FM + "tiny body\n")
        plan = skill_split.plan_split(d, 100)
        assert plan["action"] == "skip"

    def test_threshold_zero_disabled(self, tmp_path):
        d = _mk_skill(tmp_path, FM + _mk_body())
        plan = skill_split.plan_split(d, 0)
        assert plan["action"] == "skip"

    def test_130kb_fixture_splits_under_threshold(self, tmp_path):
        original = FM + _mk_body(sections=28, sec_kb=5)  # ~140 KB
        assert len(original.encode()) > 100 * 1024
        d = _mk_skill(tmp_path, original)
        plan = skill_split.plan_split(d, 100)
        assert plan["action"] == "split"
        written = skill_split.execute_split(d, plan)
        post = (d / "SKILL.md").read_text(encoding="utf-8")
        assert len(post.encode()) < 100 * 1024
        carve_files = [p for p in written if p.parent.name == "references"
                       and p.suffix == ".md"]
        assert carve_files, "carve files must be created"
        for cf in carve_files:
            assert cf.exists()
        assert (d / "references" / skill_split.MANIFEST_NAME).exists()

    def test_frontmatter_identity(self, tmp_path):
        original = FM + _mk_body()
        d = _mk_skill(tmp_path, original)
        plan = skill_split.plan_split(d, 100)
        skill_split.execute_split(d, plan)
        post = (d / "SKILL.md").read_text(encoding="utf-8")
        pre_fm, _ = skill_split.split_frontmatter(original)
        post_fm, _ = skill_split.split_frontmatter(post)
        assert pre_fm == post_fm
        assert "name: big-skill" in post_fm

    def test_ordered_content_preservation_via_reverse_join(self, tmp_path):
        """The strongest proof: join(split(x)) == x byte-for-byte."""
        original = FM + _mk_body()
        d = _mk_skill(tmp_path, original)
        plan = skill_split.plan_split(d, 100)
        skill_split.execute_split(d, plan)
        assert skill_split.join_split_skill(d) is True
        assert (d / "SKILL.md").read_text(encoding="utf-8") == original
        # carve files + manifest removed by the join
        refs = d / "references"
        assert not (refs / skill_split.MANIFEST_NAME).exists()

    def test_pointer_lines_match_fixed_template_and_count(self, tmp_path):
        original = FM + _mk_body()
        d = _mk_skill(tmp_path, original)
        plan = skill_split.plan_split(d, 100)
        skill_split.execute_split(d, plan)
        post = (d / "SKILL.md").read_text(encoding="utf-8")
        pointer_lines = [ln for ln in post.splitlines()
                         if skill_split.POINTER_RE.match(ln)]
        # bounded whitelist: exactly one pointer per carve, no more
        assert len(pointer_lines) == len(plan["carves"])

    def test_verify_rejects_dropped_duplicate_line(self):
        """Order/dup-sensitivity: dropping ONE copy of a duplicated line must
        fail verification (a set-of-hashes would pass it)."""
        body = "## A\n\ndup\ndup\nrest\n"
        original = FM + body
        carve_content = "## A\n\ndup\ndup\nrest\n"
        mangled_carve = "## A\n\ndup\nrest\n"  # dropped one duplicate
        pointer = skill_split._pointer_line("A", "a")
        lean = FM + pointer + "\n"
        ok = skill_split.verify_split(
            original, lean, [{"pointer": pointer, "content": carve_content}]
        )
        assert ok is None
        bad = skill_split.verify_split(
            original, lean, [{"pointer": pointer, "content": mangled_carve}]
        )
        assert bad is not None

    def test_verify_rejects_reordered_sections(self):
        body = "## A\naaa\n## B\nbbb\n"
        original = FM + body
        pa = skill_split._pointer_line("A", "a")
        pb = skill_split._pointer_line("B", "b")
        good = FM + pa + "\n" + pb + "\n"
        carves = [
            {"pointer": pa, "content": "## A\naaa\n"},
            {"pointer": pb, "content": "## B\nbbb\n"},
        ]
        assert skill_split.verify_split(original, good, carves) is None
        # swap the carve contents (reordering) → must fail
        swapped = [
            {"pointer": pa, "content": "## B\nbbb\n"},
            {"pointer": pb, "content": "## A\naaa\n"},
        ]
        assert skill_split.verify_split(original, good, swapped) is not None

    def test_verify_rejects_frontmatter_mangling(self):
        original = FM + "## A\nbody\n"
        pointer = skill_split._pointer_line("A", "a")
        mangled_fm = "---\nname: DIFFERENT\ndescription: a big one\n---\n"
        lean = mangled_fm + pointer + "\n"
        err = skill_split.verify_split(
            original, lean, [{"pointer": pointer, "content": "## A\nbody\n"}]
        )
        assert err == "frontmatter changed by split"

    def test_slug_collision_deduped_never_overwrites(self, tmp_path):
        original = FM + _mk_body(sections=8, sec_kb=3)
        d = _mk_skill(tmp_path, original)
        refs = d / "references"
        refs.mkdir()
        # pre-existing reference whose stem collides with a likely carve slug
        sentinel = refs / "pending-lesson-00.md"
        sentinel.write_text("PRE-EXISTING — do not clobber", encoding="utf-8")
        plan = skill_split.plan_split(d, 10)
        assert plan["action"] == "split"
        slugs = [c["slug"] for c in plan["carves"]]
        assert "pending-lesson-00" not in slugs  # deduped
        skill_split.execute_split(d, plan)
        assert sentinel.read_text(encoding="utf-8") == \
            "PRE-EXISTING — do not clobber"

    def test_headingless_blob_falls_back_to_partnn(self, tmp_path):
        blob = ("no headings here just a giant blob of text " * 4000)
        # ensure newlines exist for clean chunking
        blob = "\n".join(blob[i:i + 100] for i in range(0, len(blob), 100)) + "\n"
        original = FM + blob
        assert len(original.encode()) > 100 * 1024
        d = _mk_skill(tmp_path, original)
        plan = skill_split.plan_split(d, 100)
        assert plan["action"] == "split"
        assert plan.get("fallback") == "part-NN"
        skill_split.execute_split(d, plan)
        post = (d / "SKILL.md").read_text(encoding="utf-8")
        assert len(post.encode()) < 100 * 1024
        # reversible even in fallback mode — nothing silently dropped
        assert skill_split.join_split_skill(d) is True
        assert (d / "SKILL.md").read_text(encoding="utf-8") == original

    def test_terminal_failure_leaves_unsplit(self, tmp_path):
        """A skill whose preamble alone exceeds the cap must be left
        UNSPLIT and reported — never a mangled frontmatter/intro."""
        huge_preamble = "intro line\n" * 30000  # ~330 KB preamble, no headings
        # give it one tiny heading section so the heading path is exercised
        original = FM + huge_preamble + "## Only\ntiny\n"
        d = _mk_skill(tmp_path, original)
        plan = skill_split.plan_split(d, 100)
        assert plan["action"] == "unsplittable"
        assert (d / "SKILL.md").read_text(encoding="utf-8") == original

    def test_execute_aborts_on_stale_plan(self, tmp_path):
        original = FM + _mk_body()
        d = _mk_skill(tmp_path, original)
        plan = skill_split.plan_split(d, 100)
        (d / "SKILL.md").write_text(original + "CHANGED\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="changed since"):
            skill_split.execute_split(d, plan)
