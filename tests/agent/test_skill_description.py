"""Tests for agent.skill_utils.extract_skill_description."""

import textwrap

from agent.skill_utils import extract_skill_description, parse_frontmatter


# ── Default-truncation behavior (current convention preserved) ────────────


def test_short_description_returned_unchanged():
    fm = {"description": "Short and sweet."}
    assert extract_skill_description(fm) == "Short and sweet."


def test_exactly_60_chars_returned_unchanged():
    desc = "x" * 60
    assert extract_skill_description({"description": desc}) == desc


def test_61_chars_truncated_to_57_plus_ellipsis():
    desc = "x" * 61
    result = extract_skill_description({"description": desc})
    assert result == "x" * 57 + "..."
    assert len(result) == 60


def test_long_description_truncated_to_57_plus_ellipsis():
    desc = "A really long skill description that goes on and on past the limit."
    result = extract_skill_description({"description": desc})
    assert result.endswith("...")
    assert len(result) == 60


def test_empty_description_returns_empty_string():
    assert extract_skill_description({"description": ""}) == ""
    assert extract_skill_description({}) == ""


def test_quoted_description_unwrapped():
    fm = {"description": "'wrapped in single quotes'"}
    assert extract_skill_description(fm) == "wrapped in single quotes"


def test_whitespace_stripped():
    fm = {"description": "   leading and trailing   "}
    assert extract_skill_description(fm) == "leading and trailing"


# ── description_full opt-in (new behavior) ─────────────────────────────────


def test_description_full_true_bypasses_truncation():
    long_desc = "A" * 500
    fm = {"description": long_desc, "description_full": True}
    assert extract_skill_description(fm) == long_desc


def test_description_full_true_with_short_description_unchanged():
    fm = {"description": "Short.", "description_full": True}
    assert extract_skill_description(fm) == "Short."


def test_description_full_false_preserves_truncation():
    long_desc = "x" * 200
    fm = {"description": long_desc, "description_full": False}
    result = extract_skill_description(fm)
    assert len(result) == 60
    assert result.endswith("...")


def test_description_full_absent_preserves_truncation():
    long_desc = "x" * 200
    fm = {"description": long_desc}
    result = extract_skill_description(fm)
    assert len(result) == 60
    assert result.endswith("...")


def test_description_full_only_python_True_opts_out():
    """The check is `is True`. Quoted strings, ints, and lists are rejected.

    Note: YAML 1.1 boolean synonyms (`yes`, `on`) are coerced to Python `True`
    by PyYAML *before* this function runs, so they DO opt out via the parser
    path. That coverage is in `test_yaml_boolean_synonyms_opt_out` below.
    Here we only verify the function-level contract: any value that is not
    Python `True` (after parsing) keeps the 60-char default.
    """
    long_desc = "x" * 200
    rejected_values = ("true", "yes", "on", "1", 1, ["true"], None, 0, "")
    for non_true_value in rejected_values:
        fm = {"description": long_desc, "description_full": non_true_value}
        result = extract_skill_description(fm)
        assert len(result) == 60, f"unexpected opt-out for value {non_true_value!r}"
        assert result.endswith("..."), f"unexpected opt-out for value {non_true_value!r}"


def test_description_full_with_empty_description_returns_empty():
    fm = {"description": "", "description_full": True}
    assert extract_skill_description(fm) == ""


def test_description_full_strips_quotes_and_whitespace():
    fm = {"description": "  'opt-in with quotes'  ", "description_full": True}
    assert extract_skill_description(fm) == "opt-in with quotes"


# ── Parser-path integration (covers the real SKILL.md → frontmatter → desc flow) ─


def _parse_frontmatter_block(yaml_block: str) -> dict:
    """Build a SKILL.md-shaped string from a YAML block and return parsed frontmatter."""
    content = f"---\n{yaml_block}---\n# Body"
    fm, _ = parse_frontmatter(content)
    return fm


def test_yaml_true_opts_out_via_parser():
    long_desc = "z" * 200
    yaml = textwrap.dedent(f"""\
        name: t1
        description: "{long_desc}"
        description_full: true
    """)
    fm = _parse_frontmatter_block(yaml)
    result = extract_skill_description(fm)
    assert result == long_desc


def test_yaml_boolean_synonyms_opt_out():
    """PyYAML 1.1 coerces unquoted `yes` and `on` to Python True.

    This is documented in the function docstring as a consequence of the
    `is True` check passing whatever the YAML parser yields. If upstream
    later moves to a stricter YAML 1.2 loader, these synonyms would no
    longer coerce, and these assertions would flip — that change would be
    a deliberate behavior shift, not a bug.
    """
    long_desc = "y" * 200
    for synonym in ("yes", "on", "True", "Yes", "TRUE"):
        yaml = textwrap.dedent(f"""\
            name: t-{synonym}
            description: "{long_desc}"
            description_full: {synonym}
        """)
        fm = _parse_frontmatter_block(yaml)
        result = extract_skill_description(fm)
        assert result == long_desc, (
            f"YAML synonym {synonym!r} did not opt out; "
            f"got {result[:30]!r}... (len {len(result)})"
        )


def test_yaml_quoted_true_does_not_opt_out():
    """A quoted string `"true"` parses to the str `'true'`, not Python True."""
    long_desc = "q" * 200
    yaml = textwrap.dedent(f"""\
        name: tq
        description: "{long_desc}"
        description_full: "true"
    """)
    fm = _parse_frontmatter_block(yaml)
    result = extract_skill_description(fm)
    assert len(result) == 60
    assert result.endswith("...")


def test_yaml_integer_one_does_not_opt_out():
    long_desc = "i" * 200
    yaml = textwrap.dedent(f"""\
        name: ti
        description: "{long_desc}"
        description_full: 1
    """)
    fm = _parse_frontmatter_block(yaml)
    result = extract_skill_description(fm)
    assert len(result) == 60
    assert result.endswith("...")


def test_yaml_false_keeps_truncation():
    long_desc = "f" * 200
    yaml = textwrap.dedent(f"""\
        name: tf
        description: "{long_desc}"
        description_full: false
    """)
    fm = _parse_frontmatter_block(yaml)
    result = extract_skill_description(fm)
    assert len(result) == 60
    assert result.endswith("...")


def test_yaml_no_flag_keeps_truncation():
    long_desc = "n" * 200
    yaml = textwrap.dedent(f"""\
        name: tn
        description: "{long_desc}"
    """)
    fm = _parse_frontmatter_block(yaml)
    result = extract_skill_description(fm)
    assert len(result) == 60
    assert result.endswith("...")
