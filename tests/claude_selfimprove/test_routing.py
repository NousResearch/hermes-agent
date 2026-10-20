from __future__ import annotations

from claude_selfimprove import routing


def _row(**overrides):
    row = {
        "scope": "global",
        "target_kind": "rule",
        "canonical_key": "never-force-push-main",
        "title": "Never force push main",
        "body": "Never force push to the main branch.",
    }
    row.update(overrides)
    return row


def test_repo_scope_never_routes(sandbox):
    row = _row(scope="repo")
    assert routing.resolve_target(row) is None


def test_rule_target_routes_to_rules_dir(sandbox):
    row = _row(target_kind="rule")
    decision = routing.resolve_target(row)
    assert decision.kind == "rule"
    assert decision.path == sandbox.claude_home / "rules" / "never-force-push-main.md"


def test_skill_target_routes_to_skill_dir(sandbox):
    row = _row(target_kind="skill", canonical_key="split-large-refactors")
    decision = routing.resolve_target(row)
    assert decision.kind == "skill"
    assert decision.path == sandbox.claude_home / "skills" / "split-large-refactors" / "SKILL.md"


def test_claude_md_block_target_stays_in_block_when_under_cap(sandbox):
    row = _row(target_kind="claude_md_block", body="short")
    decision = routing.resolve_target(row, current_claude_md_block_chars=0)
    assert decision.kind == "claude_md_block"
    assert decision.path is None


def test_claude_md_block_target_overflows_to_rule_when_over_cap(sandbox):
    row = _row(target_kind="claude_md_block", body="x" * 200)
    decision = routing.resolve_target(
        row, current_claude_md_block_chars=routing.MAX_MANAGED_BLOCK_CHARS - 50
    )
    assert decision.kind == "rule"
    assert decision.path == sandbox.claude_home / "rules" / "never-force-push-main.md"


def test_slugify_handles_messy_input():
    assert routing.slugify("Already-Kebab-Case") == "already-kebab-case"
    assert routing.slugify("weird__chars!!here") == "weird-chars-here"
    assert routing.slugify("") == "lesson"


def test_unknown_target_kind_returns_none(sandbox):
    row = _row(target_kind="something-unrecognized")
    assert routing.resolve_target(row) is None


# canonical_key is model-supplied text (classified from a transcript by a
# language model). rule_path_for/skill_path_for build real filesystem paths
# from it via slugify, so a hostile or malformed key must never survive as
# a "/" or a ".." in the output — that would let a written file land
# outside rules_dir()/skills_dir(). These lock in that guarantee.
_PATH_TRAVERSAL_INPUTS = [
    "../../etc/passwd",
    "../../../etc/passwd",
    "/etc/passwd",
    "..",
    "../",
    "....//....//etc/passwd",
    "a/../../b",
    "~/.ssh/id_rsa",
]


def test_slugify_never_produces_a_path_separator_or_traversal():
    for raw in _PATH_TRAVERSAL_INPUTS:
        slug = routing.slugify(raw)
        assert "/" not in slug, f"{raw!r} -> {slug!r} contains a path separator"
        assert "\\" not in slug, f"{raw!r} -> {slug!r} contains a path separator"
        assert ".." not in slug, f"{raw!r} -> {slug!r} still contains a traversal sequence"
        assert slug not in ("", ".", ".."), f"{raw!r} -> {slug!r} is not a safe slug"


def test_rule_path_for_stays_inside_rules_dir(sandbox):
    for raw in _PATH_TRAVERSAL_INPUTS:
        path = routing.rule_path_for(raw)
        path.relative_to(sandbox.claude_home / "rules")  # raises ValueError if it escaped


def test_skill_path_for_stays_inside_skills_dir(sandbox):
    for raw in _PATH_TRAVERSAL_INPUTS:
        path = routing.skill_path_for(raw)
        path.relative_to(sandbox.claude_home / "skills")  # raises ValueError if it escaped


def test_resolve_target_with_traversal_canonical_key_stays_contained(sandbox):
    for raw in _PATH_TRAVERSAL_INPUTS:
        row = _row(target_kind="rule", canonical_key=raw)
        decision = routing.resolve_target(row)
        decision.path.relative_to(sandbox.claude_home / "rules")
