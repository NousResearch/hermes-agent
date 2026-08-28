"""Tests for implicit skill prefetch (``agent/skill_prefetch.py``).

Ported semantic from OpenAI Codex's implicit skill invocation: when the
user prompt word-boundary-mentions a skill name, the turn prologue loads
that skill's full instructions into the prefetch cache so the model has
them on the first turn instead of round-tripping ``skill_view()``.

The detection logic is exercised as a pure function here; the integration
point (turn prologue appending to ``ext_prefetch_cache``) is covered in
``test_turn_context.py``.
"""

from __future__ import annotations

from unittest.mock import patch

from agent.skill_prefetch import (
    MAX_PREFETCH_SKILLS,
    MAX_SKILL_CHARS,
    MAX_TOTAL_CHARS,
    MIN_NAME_LENGTH,
    _read_skill_body,
    build_skill_prefetch,
    detect_mentioned_skill_names,
    _skill_pattern,
)


# ── skill_pattern ─────────────────────────────────────────────────────────


def test_skill_pattern_matches_bare_name():
    assert _skill_pattern("codex").search("research the codex harness")
    assert _skill_pattern("codex").search("codex!")


def test_skill_pattern_does_not_match_inside_word():
    # "codex" must not match inside "codexified" or as prefix of a
    # hyphenated compound (that compound is its own skill).
    assert not _skill_pattern("codex").search("codexified")
    assert not _skill_pattern("codex").search("codex-operations")


def test_skill_pattern_hyphen_matches_space():
    # People type "pr ready", the skill is `pr-ready`.
    assert _skill_pattern("pr-ready").search("the pr ready workflow")
    assert _skill_pattern("pr-ready").search("the pr-ready workflow")
    assert _skill_pattern("pr-ready").search("the pr_ready workflow")


def test_skill_pattern_is_case_insensitive():
    assert _skill_pattern("Codex").search("use CODEX for that")


# ── detect_mentioned_skill_names ──────────────────────────────────────────


def test_detect_returns_mentioned_skill():
    names = ["codex", "github-pr-workflow", "pdf"]
    hits = detect_mentioned_skill_names(
        "research the codex harness", names
    )
    assert hits == ["codex"]


def test_detect_returns_empty_for_no_mention():
    assert detect_mentioned_skill_names("hello there", ["codex", "obsidian"]) == []


def test_detect_empty_prompt():
    assert detect_mentioned_skill_names("", ["codex"]) == []
    assert detect_mentioned_skill_names(None, ["codex"]) == []


def test_detect_short_names_are_ignored():
    # `pdf` is 3 chars — too likely to be an ordinary English word.
    assert detect_mentioned_skill_names("make me a pdf", ["pdf"]) == []


def test_detect_sorts_longest_first():
    # "codex" alone cannot match the prefix of "codex-operations" (trailing
    # hyphen is excluded), so when both are skills the longer one wins.
    hits = detect_mentioned_skill_names(
        "use codex-operations please", ["codex", "codex-operations"]
    )
    assert hits == ["codex-operations"]
    # When the prompt names both, the longer skill sorts first.
    hits = detect_mentioned_skill_names(
        "use codex and codex-operations", ["codex", "codex-operations"]
    )
    assert hits == ["codex-operations", "codex"]


def test_detect_caps_at_max():
    names = [f"skill-{i}" for i in range(10)]
    prompt = " ".join(names)
    hits = detect_mentioned_skill_names(prompt, names)
    assert len(hits) == MAX_PREFETCH_SKILLS


# ── build_skill_prefetch / _read_skill_body ───────────────────────────────


def test_build_skill_prefetch_empty_for_no_mention():
    with patch(
        "tools.skills_tool._find_all_skills",
        return_value=[{"name": "codex"}],
    ):
        assert build_skill_prefetch("nothing here") == ""


def test_build_skill_prefetch_empty_prompt():
    assert build_skill_prefetch("") == ""
    assert build_skill_prefetch("  ") == ""


def test_build_skill_prefetch_returns_fenced_body(tmp_path):
    skill_dir = tmp_path / "codex"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: codex\ndescription: research the codex harness\n---\n\n# Codex\n\nFull instructions here.\n",
        encoding="utf-8",
    )
    with patch(
        "tools.skills_tool._find_all_skills",
        return_value=[{"name": "codex"}],
    ), patch(
        "agent.skill_prefetch._find_skill_md",
        return_value=skill_dir / "SKILL.md",
    ):
        out = build_skill_prefetch("research the codex harness")
    assert "Full instructions here." in out
    assert "[Implicitly loaded skill: codex]" in out


def test_build_skill_prefetch_missing_skill_is_noop(tmp_path):
    # Index names a skill that vanished before read — no crash, empty result.
    with patch(
        "tools.skills_tool._find_all_skills",
        return_value=[{"name": "vanished"}],
    ), patch(
        "agent.skill_prefetch._find_skill_md",
        return_value=None,
    ):
        assert build_skill_prefetch("use the vanished skill") == ""


def test_build_skill_prefetch_respects_char_budget(tmp_path):
    big_body = "x" * (MAX_SKILL_CHARS + 100)
    with patch(
        "tools.skills_tool._find_all_skills",
        return_value=[{"name": "codex"}],
    ), patch(
        "agent.skill_prefetch._find_skill_md",
        return_value=None,
    ), patch(
        "agent.skill_prefetch._read_skill_body",
        return_value=big_body,
    ):
        out = build_skill_prefetch("research the codex harness")
    assert len(out) <= MAX_TOTAL_CHARS


def test_read_skill_body_strips_frontmatter(tmp_path):
    skill_dir = tmp_path / "codex"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: codex\ndescription: x\n---\n\nBody text.",
        encoding="utf-8",
    )
    with patch(
        "agent.skill_utils.get_scan_ordered_skills_dirs",
        return_value=[tmp_path],
    ):
        body = _read_skill_body("codex")
    assert "Body text." in body
    assert "description: x" not in body


def test_read_skill_body_returns_empty_for_missing(tmp_path):
    with patch(
        "agent.skill_utils.get_scan_ordered_skills_dirs",
        return_value=[tmp_path],
    ):
        assert _read_skill_body("nope") == ""


def test_read_skill_body_rejects_traversal():
    assert _read_skill_body("../etc/passwd") == ""


# ── mention markers ────────────────────────────────────────────────────────


def test_marker_at_sign_triggers_short_skill():
    # @pdf bypasses the MIN_NAME_LENGTH=4 guard — explicit user intent wins.
    hits = detect_mentioned_skill_names("please @pdf this", ["pdf", "codex"])
    assert hits == ["pdf"]


def test_marker_skill_colon_triggers_short_skill():
    hits = detect_mentioned_skill_names("use skill: git here", ["git", "codex"])
    assert hits == ["git"]


def test_marker_slash_skill_triggers():
    hits = detect_mentioned_skill_names("/skill obsidian", ["obsidian", "pdf"])
    assert hits == ["obsidian"]


def test_marker_is_case_insensitive():
    # @OBSIDIAN should resolve to "obsidian" in the index.
    hits = detect_mentioned_skill_names("hi @OBSIDIAN", ["obsidian"])
    assert hits == ["obsidian"]


def test_marker_for_unknown_skill_is_dropped():
    # The user mentions a skill that isn't installed — silently drop, do not
    # break the turn. (Markers do not turn into phantom prefetches.)
    hits = detect_mentioned_skill_names("@does-not-exist", ["codex"])
    assert hits == []


def test_marker_does_not_match_email_or_path():
    # `foo@pdf.com` and `/skill/foo` should not trigger a marker match.
    # The pattern requires whitespace/start/paren/space before the marker.
    hits = detect_mentioned_skill_names(
        "send me at user@pdf.com please", ["pdf"]
    )
    assert hits == []
    hits = detect_mentioned_skill_names("look at /skill/pdf page", ["pdf"])
    # The leading space before "/skill" is the boundary, and "pdf" then has
    # to be the skill name (no leading "skill" word) — this is the slash-
    # command `/skill <name>`. A bare "/skill/pdf" reads as `/skill` then
    # `pdf` would be picked up by auto-match, but auto-match skips pdf
    # (length guard). The marker regex requires a space AFTER "/skill", so
    # "/skill/pdf" without a space does not match — that's the intended
    # slash-command discipline.
    assert hits == []


def test_marker_rejects_traversal_attempt():
    # Defense in depth: a marker whose captured name fails _safe_skill_name
    # must be dropped, even if the pattern matched.
    hits = detect_mentioned_skill_names("@../etc/passwd", ["codex"])
    assert hits == []


def test_marker_and_word_boundary_merge_dedup():
    # Both channels pick up "codex"; the result should not list it twice.
    hits = detect_mentioned_skill_names(
        "@codex please, then research the codex harness",
        ["codex", "obsidian"],
    )
    assert hits.count("codex") == 1
    assert "codex" in hits


def test_marker_hits_sort_longest_first_when_merged():
    # Auto-match picks up "codex" and "codex-operations"; marker explicitly
    # names "codex-operations". Longest wins overall.
    hits = detect_mentioned_skill_names(
        "@codex-operations and the codex harness",
        ["codex", "codex-operations"],
    )
    assert hits[0] == "codex-operations"
    assert hits.count("codex") == 1


def test_build_skill_prefetch_marker_bypasses_length_guard(tmp_path):
    # End-to-end: short skill hit via marker gets its body prefetched.
    skill_dir = tmp_path / "pdf"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: pdf\ndescription: merge pdfs\n---\n\nPDF merge instructions.",
        encoding="utf-8",
    )
    with patch(
        "tools.skills_tool._find_all_skills",
        return_value=[{"name": "pdf"}],
    ), patch(
        "agent.skill_prefetch._find_skill_md",
        return_value=skill_dir / "SKILL.md",
    ):
        out = build_skill_prefetch("please @pdf this file")
    assert "PDF merge instructions." in out
    assert "[Implicitly loaded skill: pdf]" in out
