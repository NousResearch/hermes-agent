"""Behavior contracts for the bounded shared-learning Vault reader."""

from __future__ import annotations

from pathlib import Path

from agent.learning_vault import read_vault_learning


def _note(
    *,
    title: str,
    record_id: str,
    kind: str,
    agent: str,
    body: str,
    authority: str = "source-index",
    related: tuple[str, ...] = (),
) -> str:
    relation_lines = "\n".join(f"  - {item}" for item in related) or "  []"
    execution = "reference-only" if kind == "skill-reference" else "not-applicable"
    return f"""---
title: {title}
schema_name: agent_learning_record_v1
record_id: {record_id}
kind: {kind}
agent: {agent}
memory_area: Runtime
status: current
authority: {authority}
classification: diagnostic-only
sync_owned: true
execution_status: {execution}
verified_at: "2026-08-24"
related_record_ids:
{relation_lines}
source_label: should-never-be-opened.md
---

# {title}

{body}
"""


def test_reads_sync_owned_memories_and_reference_only_skills(tmp_path: Path) -> None:
    memory = tmp_path / "Memories" / "Codex" / "Runtime" / "lesson.md"
    skill = tmp_path / "Reference" / "Agent Skills" / "Claude" / "playwright.md"
    memory.parent.mkdir(parents=True)
    skill.parent.mkdir(parents=True)
    memory.write_text(
        _note(
            title="Bounded lesson",
            record_id="alr-memory",
            kind="memory",
            agent="codex",
            body="A" * 1400,
            related=("alr-skill",),
        ),
        encoding="utf-8",
    )
    skill.write_text(
        _note(
            title="Playwright",
            record_id="alr-skill",
            kind="skill-reference",
            agent="claude",
            body="Description only.",
        ),
        encoding="utf-8",
    )

    nodes, diagnostics = read_vault_learning(tmp_path)

    assert diagnostics == []
    assert {node.kind for node in nodes} == {"shared-memory", "skill-reference"}
    assert {node.record_id for node in nodes} == {"alr-memory", "alr-skill"}
    assert max(len(node.summary) for node in nodes) <= 1200
    assert next(
        node for node in nodes if node.record_id == "alr-memory"
    ).related_record_ids == ("alr-skill",)
    assert next(
        node for node in nodes if node.record_id == "alr-skill"
    ).execution_status == ("reference-only")


def test_rejects_unowned_missing_identity_and_runtime_authority(tmp_path: Path) -> None:
    root = tmp_path / "Memories" / "Hermes" / "Runtime"
    root.mkdir(parents=True)
    (root / "unowned.md").write_text("# Human note\n", encoding="utf-8")
    (root / "runtime.md").write_text(
        _note(
            title="Unsafe authority",
            record_id="alr-runtime",
            kind="memory",
            agent="hermes",
            body="Narrative only.",
            authority="runtime-truth",
        ),
        encoding="utf-8",
    )

    nodes, diagnostics = read_vault_learning(tmp_path)

    assert nodes == []
    assert {diagnostic.reason_code for diagnostic in diagnostics} == {
        "record_id_required",
        "unsupported_authority",
    }
    assert all("Narrative only" not in repr(diagnostic) for diagnostic in diagnostics)


def test_invalid_or_oversized_notes_are_diagnostic_not_fatal(tmp_path: Path) -> None:
    root = tmp_path / "Memories" / "Claude" / "Runtime"
    root.mkdir(parents=True)
    (root / "invalid.md").write_bytes(b"\xff\xfe")
    (root / "oversized.md").write_text("x" * (16 * 1024 + 1), encoding="utf-8")

    nodes, diagnostics = read_vault_learning(tmp_path)

    assert nodes == []
    assert {diagnostic.reason_code for diagnostic in diagnostics} == {
        "invalid_utf8",
        "note_too_large",
    }


def test_missing_vault_returns_bounded_diagnostic(tmp_path: Path) -> None:
    nodes, diagnostics = read_vault_learning(tmp_path / "missing")

    assert nodes == []
    assert [diagnostic.reason_code for diagnostic in diagnostics] == [
        "vault_unavailable"
    ]
