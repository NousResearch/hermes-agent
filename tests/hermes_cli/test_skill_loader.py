"""Unit tests for ``hermes_cli.skill_loader.validate_skill_frontmatter``.

Covers the acceptance criteria from kanban task t_9e321df6:
- missing required field  → specific actionable error
- invalid status           → "status 'X' is not one of {draft, vetted, deprecated}"
- invalid version          → semver error message
- name/dir mismatch        → "name 'X' does not match parent directory 'Y'"
- fully valid skill        → (True, [])
- file-not-found           → safe error
- validate_all_skills      → walks nested layouts

These tests are hermes_cli-layer tests; they use tmp_path for hermetic
fixtures and never touch the user's real ``~/.hermes/skills/`` tree.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from hermes_cli.skill_loader import (
    ALLOWED_STATUSES,
    REQUIRED_FIELDS,
    parse_skill_frontmatter,
    validate_all_skills,
    validate_skill_frontmatter,
)


# ── Fixtures ───────────────────────────────────────────────────────────────


def _write_skill(skills_root: Path, name: str, body: str) -> Path:
    """Write a SKILL.md into ``skills_root/<name>/`` and return its path."""
    skill_dir = skills_root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text(dedent(body), encoding="utf-8")
    return skill_path


VALID_BODY = """\
    ---
    name: my-skill
    version: 1.2.3
    status: vetted
    description: A perfectly valid skill used by the test suite.
    author: Hermes Agent
    ---

    # My Skill

    Body text goes here.
    """


# ── Happy path ─────────────────────────────────────────────────────────────


def test_fully_valid_skill_returns_ok(tmp_path):
    path = _write_skill(tmp_path, "my-skill", VALID_BODY)
    ok, errors = validate_skill_frontmatter(path)
    assert ok is True
    assert errors == []


def test_valid_skill_with_prerelease_version(tmp_path):
    body = VALID_BODY.replace("version: 1.2.3", "version: 0.1.0-rc.1")
    path = _write_skill(tmp_path, "my-skill", body)
    ok, errors = validate_skill_frontmatter(path)
    assert ok is True, errors


def test_valid_skill_with_build_metadata(tmp_path):
    body = VALID_BODY.replace("version: 1.2.3", "version: 1.2.3+build.42")
    path = _write_skill(tmp_path, "my-skill", body)
    ok, errors = validate_skill_frontmatter(path)
    assert ok is True, errors


def test_all_three_status_values_are_accepted(tmp_path):
    for status in ALLOWED_STATUSES:
        body = VALID_BODY.replace("status: vetted", f"status: {status}")
        path = _write_skill(tmp_path, "my-skill", body)
        ok, errors = validate_skill_frontmatter(path)
        assert ok is True, f"status={status!r} should be valid; got {errors}"


# ── Missing-field tests ────────────────────────────────────────────────────


@pytest.mark.parametrize("missing", REQUIRED_FIELDS)
def test_missing_required_field_is_specific(tmp_path, missing):
    body = VALID_BODY
    # Drop the field by removing its YAML line.
    body = "\n".join(
        line for line in body.splitlines()
        if not line.lstrip().startswith(f"{missing}:")
    )
    path = _write_skill(tmp_path, "my-skill", body)
    ok, errors = validate_skill_frontmatter(path)
    assert ok is False
    assert any(missing in e for e in errors), (
        f"expected error mentioning {missing!r}, got {errors!r}"
    )
    # Error must be actionable, not just "validation failed".
    assert any(f"missing required field '{missing}'" in e for e in errors), errors


def test_multiple_missing_fields_are_all_reported(tmp_path):
    body = VALID_BODY
    body = "\n".join(
        line for line in body.splitlines()
        if not line.lstrip().startswith(("version:", "author:"))
    )
    path = _write_skill(tmp_path, "my-skill", body)
    ok, errors = validate_skill_frontmatter(path)
    assert ok is False
    joined = "\n".join(errors)
    assert "version" in joined
    assert "author" in joined


def test_empty_string_field_counts_as_missing(tmp_path):
    body = VALID_BODY.replace("name: my-skill", 'name: ""')
    path = _write_skill(tmp_path, "my-skill", body)
    ok, errors = validate_skill_frontmatter(path)
    assert ok is False
    assert any("name" in e and "empty" in e for e in errors), errors


# ── Invalid status ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("bad_status", ["production", "stable", "live", "released", "alpha", "beta"])
def test_invalid_status_rejected(tmp_path, bad_status):
    body = VALID_BODY.replace("status: vetted", f"status: {bad_status!r}")
    path = _write_skill(tmp_path, "my-skill", body)
    ok, errors = validate_skill_frontmatter(path)
    assert ok is False
    assert any("not one of" in e for e in errors), errors
    # Error message should mention each allowed value.
    msg = " ".join(errors)
    for allowed in ALLOWED_STATUSES:
        assert allowed in msg, f"allowed status {allowed!r} missing from error: {msg!r}"


def test_status_is_case_insensitive(tmp_path):
    """Validator lowercases the status before checking the allow-list.

    ``DRAFT`` and ``Vetted`` are accepted; the spec only enumerates the
    canonical lowercase spellings but the check is intentionally
    case-tolerant so authors don't accidentally fail the gate by
    capitalizing a single word.
    """
    for status in ("DRAFT", "Draft", "vetted", "VETTED", "Deprecated"):
        body = VALID_BODY.replace("status: vetted", f"status: {status!r}")
        path = _write_skill(tmp_path, "my-skill", body)
        ok, errors = validate_skill_frontmatter(path)
        assert ok is True, f"status={status!r} should pass case-fold: {errors}"


# ── Invalid version ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "bad_version",
    [
        "1",                # missing minor/patch
        "1.2",              # missing patch
        "v1.2.3",           # leading v
        "1.2.3.4.5",        # 5 segments
        "01.2.3",           # leading zero in major
        "1.02.3",           # leading zero in minor
        "1.2.03",           # leading zero in patch
        "not-a-version",
        "1.2.3-",           # empty prerelease
        "1.2.3+",           # empty build metadata
        "1.2.3-+build",     # empty prerelease before build metadata
    ],
)
def test_invalid_version_rejected(tmp_path, bad_version):
    body = VALID_BODY.replace("version: 1.2.3", f"version: {bad_version!r}")
    path = _write_skill(tmp_path, "my-skill", body)
    ok, errors = validate_skill_frontmatter(path)
    assert ok is False
    assert any("not valid semver" in e for e in errors), errors


# ── name / parent directory mismatch ──────────────────────────────────────


def test_name_dir_mismatch_rejected(tmp_path):
    """name in frontmatter must match the parent directory name."""
    # _write_skill creates the directory with the name we pass, but we
    # intentionally diverge the frontmatter `name:` field.
    path = _write_skill(tmp_path, "real-dir-name", VALID_BODY)  # frontmatter says my-skill
    ok, errors = validate_skill_frontmatter(path)
    assert ok is False
    assert any(
        "name" in e and "real-dir-name" in e
        for e in errors
    ), errors


def test_name_dir_match_passes(tmp_path):
    body = VALID_BODY.replace("name: my-skill", "name: agent-handoff")
    path = _write_skill(tmp_path, "agent-handoff", body)
    ok, errors = validate_skill_frontmatter(path)
    assert ok is True, errors


# ── Malformed frontmatter ──────────────────────────────────────────────────


def test_missing_frontmatter_block_rejected(tmp_path):
    body = "# My Skill\n\nNo frontmatter here.\n"
    path = _write_skill(tmp_path, "my-skill", body)
    ok, errors = validate_skill_frontmatter(path)
    assert ok is False
    assert any("frontmatter" in e.lower() for e in errors), errors


def test_unterminated_frontmatter_block_rejected(tmp_path):
    body = "---\nname: my-skill\nversion: 1.2.3\n# no closing fence\n"
    path = _write_skill(tmp_path, "my-skill", body)
    ok, errors = validate_skill_frontmatter(path)
    assert ok is False


def test_file_not_found_is_safe(tmp_path):
    ok, errors = validate_skill_frontmatter(tmp_path / "does-not-exist" / "SKILL.md")
    assert ok is False
    assert len(errors) == 1
    assert "not found" in errors[0]


# ── parse_skill_frontmatter ────────────────────────────────────────────────


def test_parse_skill_frontmatter_returns_dict_and_body(tmp_path):
    path = _write_skill(tmp_path, "my-skill", VALID_BODY)
    fm, body = parse_skill_frontmatter(path)
    assert fm["name"] == "my-skill"
    assert fm["status"] == "vetted"
    assert "# My Skill" in body


def test_parse_skill_frontmatter_no_frontmatter_returns_empty(tmp_path):
    body = "# No frontmatter\n\nJust markdown.\n"
    p = tmp_path / "SKILL.md"
    p.write_text(body)
    fm, parsed_body = parse_skill_frontmatter(p)
    assert fm == {}
    assert "No frontmatter" in parsed_body


# ── validate_all_skills ────────────────────────────────────────────────────


def test_validate_all_skills_walks_top_level(tmp_path):
    valid_alpha = VALID_BODY.replace("name: my-skill", "name: alpha")
    valid_beta = VALID_BODY.replace("name: my-skill", "name: beta")
    bad_gamma = VALID_BODY.replace("name: my-skill", "name: gamma").replace(
        "status: vetted", "status: production"
    )
    _write_skill(tmp_path, "alpha", valid_alpha)
    _write_skill(tmp_path, "beta", valid_beta)
    _write_skill(tmp_path, "gamma", bad_gamma)

    results = validate_all_skills(tmp_path)
    assert len(results) == 3
    paths_by_name = {Path(p).parent.name: p for p in results}
    assert results[paths_by_name["alpha"]][0] is True, results[paths_by_name["alpha"]]
    assert results[paths_by_name["beta"]][0] is True, results[paths_by_name["beta"]]
    assert results[paths_by_name["gamma"]][0] is False
    assert any("not one of" in e for e in results[paths_by_name["gamma"]][1])


def test_validate_all_skills_walks_nested_categories(tmp_path):
    """Nested layouts (skills/<category>/<name>/SKILL.md) must also be walked."""
    apple_body = VALID_BODY.replace("name: my-skill", "name: macos-computer-use")
    devops_body = VALID_BODY.replace("name: my-skill", "name: totum-kanban-watch")
    _write_skill(tmp_path / "apple", "macos-computer-use", apple_body)
    _write_skill(tmp_path / "devops", "totum-kanban-watch", devops_body)

    results = validate_all_skills(tmp_path)
    assert len(results) == 2
    failures = {p: errs for p, (ok, errs) in results.items() if not ok}
    assert failures == {}, failures


def test_validate_all_skills_missing_root_returns_empty(tmp_path):
    results = validate_all_skills(tmp_path / "no-such-dir")
    assert results == {}