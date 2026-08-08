from __future__ import annotations

from hermes_cli.provenance_check import (
    Commit,
    CheckResult,
    emit_pr_block,
    extract_pr_provenance_block,
    normalize_allowed_writers,
    parse_final_trailer_block,
    validate_commits,
    validate_pr_body,
)


def test_parse_final_trailer_block_requires_contiguous_final_block() -> None:
    message = """fix: example

Body.

Writer: codex

Refs: #42
"""

    block = parse_final_trailer_block(message)

    assert block.values("writer") == []
    assert block.values("refs") == ["#42"]


def test_validate_commits_accepts_provider_level_writer_and_refs() -> None:
    commit = Commit(
        sha="abcdef1234567890",
        subject="fix: example",
        body="""fix: example

Body.

Writer: codex
Refs: #42
""",
    )

    result = validate_commits([commit], range_spec="main..HEAD")

    assert result.ok
    assert result.writers == {"codex"}
    assert result.refs == {"#42"}


def test_validate_commits_rejects_model_specific_writer() -> None:
    commit = Commit(
        sha="abcdef1234567890",
        subject="fix: example",
        body="""fix: example

Body.

Writer: gpt-5.5
Refs: #42
""",
    )

    result = validate_commits([commit])

    assert not result.ok
    assert any("model-specific" in finding.message for finding in result.errors)


def test_validate_commits_rejects_writer_outside_final_trailer_block() -> None:
    commit = Commit(
        sha="abcdef1234567890",
        subject="fix: example",
        body="""fix: example

Body.

Writer: codex

Refs: #42
""",
    )

    result = validate_commits([commit])

    assert not result.ok
    assert any("not in the final contiguous trailer block" in finding.message for finding in result.errors)
    assert any("missing final Writer" in finding.message for finding in result.errors)


def test_validate_commits_supports_repository_specific_writer_enum() -> None:
    commit = Commit(
        sha="abcdef1234567890",
        subject="fix: example",
        body="""fix: example

Body.

Writer: auditbot
Refs: tb-123
""",
    )

    result = validate_commits(
        [commit],
        allowed_writers=normalize_allowed_writers(["auditbot"]),
    )

    assert result.ok
    assert result.writers == {"auditbot"}


def test_validate_commits_rejects_writer_model_by_default() -> None:
    commit = Commit(
        sha="abcdef1234567890",
        subject="fix: example",
        body="""fix: example

Body.

Writer: codex
Writer-Model: gpt-5.5
Refs: #42
""",
    )

    result = validate_commits([commit])

    assert not result.ok
    assert any("Writer-Model" in finding.message for finding in result.errors)


def test_validate_pr_body_requires_provenance_fields() -> None:
    result = CheckResult(writers={"codex"}, refs={"#42"})
    body = """## Summary
Fixes the bug.

## Provenance

- GitHub actor: hermesbot-almace
- PR created by: codex
- Implemented by: codex
- Writers from commit trailers: codex
- Task ledger: GitHub #42

Closes #42
"""

    validate_pr_body(body, result)

    assert result.ok
    assert extract_pr_provenance_block(body) is not None


def test_validate_pr_body_fails_without_provenance_block() -> None:
    result = CheckResult(writers={"codex"}, refs={"#42"})

    validate_pr_body("## Summary\nNo provenance here.\n", result)

    assert not result.ok
    assert any("missing a ## Provenance" in finding.message for finding in result.errors)


def test_emit_pr_block_uses_detected_writers_and_refs() -> None:
    result = CheckResult(writers={"codex"}, refs={"#42"})

    block = emit_pr_block(result, github_actor="hermesbot-almace")

    assert "GitHub actor: hermesbot-almace" in block
    assert "PR created by: codex" in block
    assert "Task ledger: #42" in block
