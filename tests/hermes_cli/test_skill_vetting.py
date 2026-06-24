"""Unit tests for SKILL.md vetting (t_8a86fc9c).

Covers:

* ``is_vetted`` / ``vetting_state`` predicates
* ``validate_skill_content_quality`` (description length, body length,
  markdown headings, executable URLs)
* ``validate_skill_links_files`` (markdown link / image target existence,
  code-fence handling, external links ignored)
* ``stamp_skill_vetted`` (writes vetted_at / vetted_by, idempotent,
  rejects bad reviewer strings)
* ``run_vet`` orchestrator (runs all 4 validators, dry-run, stamp on
  success, doesn't stamp on failure)

These tests are hermes_cli-layer tests; they use ``tmp_path`` for
hermetic fixtures and never touch the user's real ``~/.hermes/skills/``
tree.  ``tools.skills_guard`` is the only system module we touch, and
it ships with hermes-agent so it's always importable in the test
environment.
"""

from __future__ import annotations

import re
from pathlib import Path
from textwrap import dedent

import pytest

from hermes_cli.skill_loader import (
    UNVETTED,
    is_vetted,
    parse_skill_frontmatter,
    parse_vetted_at,
    run_vet,
    stamp_skill_vetted,
    validate_skill_content_quality,
    validate_skill_frontmatter,
    validate_skill_links_files,
    vetting_state,
)


# ── Fixtures ───────────────────────────────────────────────────────────────


def _write_skill(skills_root: Path, name: str, body: str) -> Path:
    """Write a SKILL.md into ``skills_root/<name>/`` and return its path."""
    skill_dir = skills_root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text(dedent(body), encoding="utf-8")
    return skill_path


def _vettable_body(extras: str = "") -> str:
    """A SKILL.md body that passes all four validators.

    ``extras`` is appended to the body verbatim — use it to drop in
    referenced companion files (e.g. ``[examples](EXAMPLES.md)`` plus
    a real ``EXAMPLES.md`` on disk).
    """
    return dedent(f"""\
        ---
        name: vet-skill
        version: 1.0.0
        status: vetted
        description: A skill used by the vetting test suite to exercise
                     the four-validator vetting flow.
        author: Hermes Agent
        ---

        # Vet Skill

        This body is deliberately long enough (over 200 characters after
        dedent) and includes a markdown heading, references companion
        files, and uses no executable URLs so the four validators all
        pass without any trouble whatsoever on a freshly-written
        skill. Padding to be safe in case the validator is bumped up.
        {extras}
        """)


# ── is_vetted / vetting_state ─────────────────────────────────────────────


class TestIsVetted:
    def test_unstamped_is_unvetted(self):
        assert is_vetted({}) is False
        assert is_vetted({"name": "foo"}) is False

    def test_partial_stamp_is_unvetted(self):
        # Only vetted_at, no vetted_by → not vetted.
        assert is_vetted({"vetted_at": "2026-06-23T18:30:00Z"}) is False
        # Only vetted_by, no vetted_at → not vetted.
        assert is_vetted({"vetted_by": "wags-reviewer"}) is False

    def test_unvetted_sentinel_is_unvetted(self):
        # ``unvetted`` in vetted_by is the same as no stamp.
        assert is_vetted({"vetted_at": "x", "vetted_by": "unvetted"}) is False
        assert is_vetted({"vetted_at": "x", "vetted_by": UNVETTED}) is False

    def test_full_stamp_is_vetted(self):
        assert is_vetted(
            {"vetted_at": "2026-06-23T18:30:00Z", "vetted_by": "wags-reviewer"}
        ) is True

    def test_empty_strings_are_unvetted(self):
        # Empty strings are treated like missing fields.
        assert is_vetted({"vetted_at": "", "vetted_by": "wags-reviewer"}) is False
        assert is_vetted({"vetted_at": "x", "vetted_by": ""}) is False
        assert is_vetted({"vetted_at": "  ", "vetted_by": "  "}) is False


class TestVettingState:
    def test_unstamped_returns_sentinels(self):
        assert vetting_state({}) == {"vetted_at": UNVETTED, "vetted_by": UNVETTED}

    def test_partial_stamp_pads_missing(self):
        state = vetting_state({"vetted_by": "wags-reviewer"})
        assert state == {"vetted_at": UNVETTED, "vetted_by": "wags-reviewer"}

    def test_full_stamp_returns_both(self):
        state = vetting_state(
            {"vetted_at": "2026-06-23T18:30:00Z", "vetted_by": "wags-reviewer"}
        )
        assert state == {
            "vetted_at": "2026-06-23T18:30:00Z",
            "vetted_by": "wags-reviewer",
        }


class TestParseVettedAt:
    def test_parses_z_suffix(self):
        dt = parse_vetted_at({"vetted_at": "2026-06-23T18:30:00Z"})
        assert dt is not None
        assert dt.year == 2026 and dt.month == 6 and dt.day == 23

    def test_parses_offset(self):
        dt = parse_vetted_at({"vetted_at": "2026-06-23T18:30:00+00:00"})
        assert dt is not None

    def test_returns_none_for_unvetted(self):
        assert parse_vetted_at({}) is None
        assert parse_vetted_at({"vetted_at": UNVETTED}) is None
        assert parse_vetted_at({"vetted_at": "not-a-date"}) is None


# ── validate_skill_content_quality ────────────────────────────────────────


class TestContentQuality:
    def test_full_body_passes(self, tmp_path):
        path = _write_skill(tmp_path, "vet-skill", _vettable_body())
        ok, errors = validate_skill_content_quality(path)
        assert ok is True, errors
        assert errors == []

    def test_short_description_fails(self, tmp_path):
        body = dedent("""\
            ---
            name: vet-skill
            version: 1.0.0
            status: vetted
            description: TODO
            author: Hermes Agent
            ---

            # Skill

            """ + ("body text " * 30) + "\n")
        path = _write_skill(tmp_path, "vet-skill", body)
        ok, errors = validate_skill_content_quality(path)
        assert ok is False
        assert any("description is too short" in e for e in errors)

    def test_short_body_fails(self, tmp_path):
        body = dedent("""\
            ---
            name: vet-skill
            version: 1.0.0
            status: vetted
            description: A perfectly valid description.
            author: Hermes Agent
            ---

            # Skill

            too short
            """)
        path = _write_skill(tmp_path, "vet-skill", body)
        ok, errors = validate_skill_content_quality(path)
        assert ok is False
        assert any("body is too short" in e for e in errors)

    def test_no_headings_fails(self, tmp_path):
        # Body has enough text but no markdown heading.
        body = dedent("""\
            ---
            name: vet-skill
            version: 1.0.0
            status: vetted
            description: A perfectly valid description.
            author: Hermes Agent
            ---

            No headings here at all, just a wall of text that goes on
            for at least 200 characters so the length check passes but
            the heading check fails because we never start a line with
            a hash mark followed by a space. The body is intentionally
            long and uninteresting to trip the heading heuristic.
            """)
        path = _write_skill(tmp_path, "vet-skill", body)
        ok, errors = validate_skill_content_quality(path)
        assert ok is False
        assert any("no markdown headings" in e for e in errors)

    def test_script_tag_fails(self, tmp_path):
        body = dedent("""\
            ---
            name: vet-skill
            version: 1.0.0
            status: vetted
            description: A perfectly valid description.
            author: Hermes Agent
            ---

            # Skill

            <script>alert(1)</script>

            Body padding to satisfy the 200-character minimum requirement
            for the validator to even consider this body long enough to
            pass the content-quality gate when it is otherwise valid.
            """)
        path = _write_skill(tmp_path, "vet-skill", body)
        ok, errors = validate_skill_content_quality(path)
        assert ok is False
        assert any("<script>" in e for e in errors)

    def test_javascript_link_fails(self, tmp_path):
        body = dedent("""\
            ---
            name: vet-skill
            version: 1.0.0
            status: vetted
            description: A perfectly valid description.
            author: Hermes Agent
            ---

            # Skill

            [click me](javascript:alert(1))

            Body padding to satisfy the 200-character minimum requirement
            for the validator to even consider this body long enough to
            pass the content-quality gate when it is otherwise valid.
            """)
        path = _write_skill(tmp_path, "vet-skill", body)
        ok, errors = validate_skill_content_quality(path)
        assert ok is False
        assert any("executable URL" in e for e in errors)

    def test_prose_mention_of_javascript_is_fine(self, tmp_path):
        # Bare prose mention of "javascript:" inside a paragraph is OK —
        # we only flag executable shapes (in a link target, href=, etc.).
        body = dedent("""\
            ---
            name: vet-skill
            version: 1.0.0
            status: vetted
            description: A perfectly valid description.
            author: Hermes Agent
            ---

            # Skill

            This skill is safe: it does not contain any javascript:
            links, any data: URIs, or any active content whatsoever.
            The validator should not flag a bare prose mention of
            "javascript:" as executable code, because no link or HTML
            attribute uses the URL scheme.
            """)
        path = _write_skill(tmp_path, "vet-skill", body)
        ok, errors = validate_skill_content_quality(path)
        assert ok is True, errors

    def test_file_not_found(self, tmp_path):
        ok, errors = validate_skill_content_quality(tmp_path / "missing" / "SKILL.md")
        assert ok is False
        assert any("not found" in e for e in errors)


# ── validate_skill_links_files ────────────────────────────────────────────


class TestLinksFiles:
    def test_no_links_passes(self, tmp_path):
        path = _write_skill(tmp_path, "vet-skill", _vettable_body())
        ok, errors = validate_skill_links_files(path)
        assert ok is True, errors

    def test_existing_link_passes(self, tmp_path):
        # The skill directory is tmp_path/<name>/ and the SKILL.md lives
        # at tmp_path/<name>/SKILL.md per the _write_skill helper.
        # Drop the EXAMPLES.md in the same directory as SKILL.md so the
        # link-check resolves it.
        skill_dir = tmp_path / "vet-skill"
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "EXAMPLES.md").write_text("Examples here.\n" * 30)
        path = _write_skill(tmp_path, "vet-skill", _vettable_body(
            "See [examples](EXAMPLES.md) for usage details."
        ))
        ok, errors = validate_skill_links_files(path)
        assert ok is True, errors

    def test_missing_link_fails(self, tmp_path):
        path = _write_skill(tmp_path, "vet-skill", _vettable_body(
            "See [missing](NOT-A-REAL-FILE.md) for usage details."
        ))
        ok, errors = validate_skill_links_files(path)
        assert ok is False
        assert any("not found" in e for e in errors)

    def test_external_links_are_ignored(self, tmp_path):
        path = _write_skill(tmp_path, "vet-skill", _vettable_body(
            "See [the docs](https://example.com/docs.md) for usage details."
        ))
        ok, errors = validate_skill_links_files(path)
        assert ok is True, errors

    def test_anchor_links_are_ignored(self, tmp_path):
        path = _write_skill(tmp_path, "vet-skill", _vettable_body(
            "See [the section](#usage) for usage details."
        ))
        ok, errors = validate_skill_links_files(path)
        assert ok is True, errors

    def test_link_in_code_block_is_ignored(self, tmp_path):
        # A `[text](file.md)` inside a fenced code block is documentation
        # about a link, not an actual link — it should be skipped so a
        # tutorial that embeds example markdown doesn't fail.
        body = _vettable_body(
            "\n```\n# example: see [docs](NOT-REAL.md) for usage\n```\n"
        )
        path = _write_skill(tmp_path, "vet-skill", body)
        ok, errors = validate_skill_links_files(path)
        assert ok is True, errors

    def test_image_references_are_checked(self, tmp_path):
        path = _write_skill(tmp_path, "vet-skill", _vettable_body(
            "![logo](missing-image.png)"
        ))
        ok, errors = validate_skill_links_files(path)
        assert ok is False
        assert any("missing-image.png" in e for e in errors)


# ── stamp_skill_vetted ───────────────────────────────────────────────────


class TestStampVetted:
    def test_stamps_unstamped_skill(self, tmp_path):
        path = _write_skill(tmp_path, "vet-skill", _vettable_body())
        ts = stamp_skill_vetted(path, reviewer="wags-reviewer",
                                when="2026-06-23T18:30:00Z")
        assert ts == "2026-06-23T18:30:00Z"
        # The file now carries both stamp fields.
        fm, _ = parse_skill_frontmatter(path)
        assert fm["vetted_at"] == "2026-06-23T18:30:00Z"
        assert fm["vetted_by"] == "wags-reviewer"

    def test_idempotent_overwrite(self, tmp_path):
        path = _write_skill(tmp_path, "vet-skill", _vettable_body())
        stamp_skill_vetted(path, reviewer="reviewer-a",
                           when="2026-06-23T18:30:00Z")
        stamp_skill_vetted(path, reviewer="reviewer-b",
                           when="2026-06-24T01:00:00Z")
        fm, _ = parse_skill_frontmatter(path)
        # Re-stamping overwrites; no duplicates of vetted_at/vetted_by.
        assert fm["vetted_at"] == "2026-06-24T01:00:00Z"
        assert fm["vetted_by"] == "reviewer-b"
        # Ensure the file has exactly one of each (no duplicates from
        # a re-stamp).
        content = path.read_text()
        assert content.count("vetted_at:") == 1
        assert content.count("vetted_by:") == 1

    def test_rejects_unvetted_reviewer(self, tmp_path):
        path = _write_skill(tmp_path, "vet-skill", _vettable_body())
        with pytest.raises(ValueError, match="unvetted"):
            stamp_skill_vetted(path, reviewer="unvetted")

    def test_rejects_empty_reviewer(self, tmp_path):
        path = _write_skill(tmp_path, "vet-skill", _vettable_body())
        with pytest.raises(ValueError, match="non-empty"):
            stamp_skill_vetted(path, reviewer="")

    def test_rejects_bad_chars_in_reviewer(self, tmp_path):
        path = _write_skill(tmp_path, "vet-skill", _vettable_body())
        with pytest.raises(ValueError, match="characters outside"):
            stamp_skill_vetted(path, reviewer="has spaces")

    def test_rejects_overlong_reviewer(self, tmp_path):
        path = _write_skill(tmp_path, "vet-skill", _vettable_body())
        with pytest.raises(ValueError, match="64 characters"):
            stamp_skill_vetted(path, reviewer="a" * 65)

    def test_preserves_existing_fields(self, tmp_path):
        path = _write_skill(tmp_path, "vet-skill", _vettable_body())
        stamp_skill_vetted(path, reviewer="wags-reviewer",
                           when="2026-06-23T18:30:00Z")
        # Body is unchanged.
        assert "This body is deliberately long enough" in path.read_text()
        # The name/version/etc. are unchanged.
        fm, _ = parse_skill_frontmatter(path)
        assert fm["name"] == "vet-skill"
        assert fm["version"] == "1.0.0"
        assert fm["status"] == "vetted"


# ── run_vet orchestrator ─────────────────────────────────────────────────


class TestRunVet:
    def _vettable_skill(self, tmp_path, name="vet-skill", extras=""):
        # The skill directory is tmp_path/<name>/ and the SKILL.md lives
        # at tmp_path/<name>/SKILL.md per the _write_skill helper, so
        # EXAMPLES.md goes in the same directory as SKILL.md.
        skill_dir = tmp_path / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "EXAMPLES.md").write_text("Examples here.\n" * 30)
        return _write_skill(tmp_path, name, _vettable_body(extras))

    def test_dry_run_does_not_write(self, tmp_path):
        path = self._vettable_skill(tmp_path)
        result = run_vet(path, reviewer="wags-reviewer", dry_run=True)
        assert result["ok"] is True
        assert result["stamped"] is False
        # The file was not modified.
        assert "vetted_by" not in path.read_text()

    def test_real_vet_stamps(self, tmp_path):
        path = self._vettable_skill(tmp_path)
        result = run_vet(path, reviewer="wags-reviewer",
                         when="2026-06-23T18:30:00Z")
        assert result["ok"] is True
        assert result["stamped"] is True
        assert result["timestamp"] == "2026-06-23T18:30:00Z"
        fm, _ = parse_skill_frontmatter(path)
        assert is_vetted(fm) is True

    def test_validation_failure_prevents_stamp(self, tmp_path):
        # A SKILL.md with no body and no fields is invalid → stamp skipped.
        skill_dir = tmp_path / "broken-skill"
        skill_dir.mkdir()
        path = skill_dir / "SKILL.md"
        path.write_text("---\nname: broken-skill\n---\n\n# too short\n")
        result = run_vet(path, reviewer="wags-reviewer")
        assert result["ok"] is False
        assert result["stamped"] is False
        # The frontmatter must NOT carry vetted_at/vetted_by.
        fm, _ = parse_skill_frontmatter(path)
        assert "vetted_at" not in fm
        assert "vetted_by" not in fm

    def test_broken_link_fails_links_files_only(self, tmp_path):
        path = self._vettable_skill(tmp_path, extras="[missing](NO.md)")
        result = run_vet(path, reviewer="wags-reviewer", dry_run=True)
        assert result["ok"] is False
        assert result["validators"]["frontmatter"]["ok"] is True
        assert result["validators"]["content_quality"]["ok"] is True
        assert result["validators"]["links_files"]["ok"] is False

    def test_dry_run_with_validation_failure(self, tmp_path):
        path = self._vettable_skill(tmp_path, extras="[missing](NO.md)")
        result = run_vet(path, reviewer="wags-reviewer", dry_run=True)
        assert result["ok"] is False
        assert result["stamped"] is False

    def test_security_validator_runs(self, tmp_path):
        # A clean skill should pass the security validator.  This is a
        # smoke test for the orchestrator's "all 4 validators ran"
        # contract — the actual scanner is owned by tools.skills_guard
        # and covered by its own tests.
        path = self._vettable_skill(tmp_path)
        result = run_vet(path, reviewer="wags-reviewer", dry_run=True)
        assert "security" in result["validators"]

    def test_orchestrator_returns_dict_shape(self, tmp_path):
        # Lock down the orchestrator's return shape so downstream
        # consumers (CLI, future test framework) can rely on it.
        path = self._vettable_skill(tmp_path)
        result = run_vet(path, reviewer="wags-reviewer", dry_run=True)
        assert isinstance(result, dict)
        for key in ("path", "ok", "validators", "stamped", "timestamp", "reviewer"):
            assert key in result, f"missing key: {key}"
        assert isinstance(result["validators"], dict)
        for vname in ("frontmatter", "security", "content_quality", "links_files"):
            assert vname in result["validators"]
            assert "ok" in result["validators"][vname]
            assert "errors" in result["validators"][vname]


# ── Acceptance scenarios from the task body ──────────────────────────────


class TestAcceptance:
    """The four acceptance scenarios pinned in t_8a86fc9c's body."""

    def test_vet_clean_skill_stamps(self, tmp_path):
        # ``hermes skills vet agent-handoff --by wags-reviewer`` succeeds
        # on a clean skill and stamps frontmatter.
        path = _write_skill(tmp_path, "agent-handoff", _vettable_body().replace(
            "name: vet-skill", "name: agent-handoff"
        ))
        result = run_vet(path, reviewer="wags-reviewer",
                         when="2026-06-23T18:30:00Z")
        assert result["ok"] is True
        assert result["stamped"] is True
        fm, _ = parse_skill_frontmatter(path)
        assert fm["vetted_by"] == "wags-reviewer"
        assert fm["vetted_at"] == "2026-06-23T18:30:00Z"

    def test_vet_missing_field_does_not_stamp(self, tmp_path):
        # ``hermes skills vet`` on a skill with a missing required
        # field does NOT stamp vetted_by and reports the frontmatter
        # error.
        skill_dir = tmp_path / "broken-skill"
        skill_dir.mkdir()
        path = skill_dir / "SKILL.md"
        path.write_text(dedent("""\
            ---
            name: broken-skill
            ---

            # Body

            Long enough body that content_quality passes, but the
            frontmatter is missing version, status, description, and
            author so the frontmatter validator must fail.
            """))
        result = run_vet(path, reviewer="wags-reviewer", dry_run=False)
        assert result["validators"]["frontmatter"]["ok"] is False
        assert result["ok"] is False
        assert result["stamped"] is False
        # Verify the file was NOT modified.
        fm, _ = parse_skill_frontmatter(path)
        assert "vetted_by" not in fm or fm.get("vetted_by") == ""

    def test_vet_frontmatter_error_message_is_specific(self, tmp_path):
        # The frontmatter error is actionable so the operator can fix it.
        skill_dir = tmp_path / "broken-skill"
        skill_dir.mkdir()
        path = skill_dir / "SKILL.md"
        path.write_text(dedent("""\
            ---
            name: broken-skill
            ---

            # Body

            Long enough body that content_quality passes, but the
            frontmatter is missing version, status, description, and
            author so the frontmatter validator must fail.
            """))
        result = run_vet(path, reviewer="wags-reviewer", dry_run=True)
        errors = result["validators"]["frontmatter"]["errors"]
        # Every missing field shows up by name.
        for field in ("version", "status", "description", "author"):
            assert any(field in e for e in errors), (
                f"expected {field!r} in errors, got {errors}"
            )
