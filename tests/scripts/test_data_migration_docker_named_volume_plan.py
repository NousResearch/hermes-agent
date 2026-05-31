from __future__ import annotations

import re
from pathlib import Path


RUNBOOK = (
    Path(__file__).resolve().parents[2]
    / "notes"
    / "data-migration"
    / "docker-named-volume-backup-restore-plan.md"
)


def _read_runbook() -> str:
    return RUNBOOK.read_text(encoding="utf-8")


def _section(markdown: str, heading: str) -> str:
    match = re.search(
        rf"^## {re.escape(heading)}\n(?P<body>.*?)(?=^## |\Z)",
        markdown,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"missing section: {heading}"
    return match.group("body")


def _assert_contains_all(text: str, expected: list[str]) -> None:
    missing = [item for item in expected if item not in text]
    assert missing == []


def test_had_1158_runbook_has_required_sections_and_safety_language() -> None:
    text = _read_runbook()

    _assert_contains_all(
        text,
        [
            "# HAD-1158 Docker named-volume backup and restore plan",
            "## Scope and safety invariants",
            "## Named-volume inventory",
            "## Discovery commands",
            "## Backup plan",
            "## Restore plan",
            "## Boundary with `/data/hermes` coverage",
            "## Failure modes and rollback notes",
            "## Verification checklist and acceptance evidence",
            "## Unknowns and follow-ups",
            "This document does not execute backup or restore.",
            "No Docker backup, restore, volume create, volume remove, or volume overwrite command was run",
            "No service stop, start, restart, or container replacement is part of this artifact.",
        ],
    )


def test_had_1158_runbook_inventories_required_volumes_and_owners() -> None:
    inventory = _section(_read_runbook(), "Named-volume inventory")

    _assert_contains_all(
        inventory,
        [
            "`hadto-pipeline_pipeline-data`",
            "`ontology-platform_archivebox-data`",
            "`ontology-platform_oxigraph-data`",
            "`ontology-platform_source-materials-minio-data`",
            "`hadto-pipeline`",
            "`ontology-platform`",
            "`hadto-pipeline`",
            "`ontology-archivebox`",
            "`ontology-triplestore`",
            "`ontology-source-materials-blob-store`",
            "`/app/data`",
            "`/data`",
            "`critical-durable`",
            "`important-durable`",
        ],
    )


def test_had_1158_runbook_defines_discovery_backup_and_restore_evidence() -> None:
    text = _read_runbook()
    discovery = _section(text, "Discovery commands")
    backup = _section(text, "Backup plan")
    restore = _section(text, "Restore plan")
    verification = _section(text, "Verification checklist and acceptance evidence")

    _assert_contains_all(
        discovery,
        [
            "docker volume inspect",
            "docker inspect",
            "docker ps",
            "docker volume ls",
            "docker compose ls",
            "volume-inspect.json",
            "container-inspect.json",
            "compose-labels.txt",
        ],
    )
    _assert_contains_all(
        backup,
        [
            "Quiescence dependency",
            "Backup evidence to capture",
            "sha256sum",
            "tar --numeric-owner -cpf",
            "read-only mount",
            "writer quiescence",
            "`ontology-platform_source-materials-minio-data`",
        ],
    )
    _assert_contains_all(
        restore,
        [
            "Scratch-host rehearsal",
            "docker volume create",
            "tar --numeric-owner -xpf",
            "empty rehearsal volume",
            "Production restore target",
            "Rehearsal checks",
            "Production acceptance",
        ],
    )
    _assert_contains_all(
        verification,
        [
            "Per-volume quiescence evidence with secrets redacted.",
            "Backup readback checksum verification.",
            "Scratch-host restore log for every volume.",
            "Manager acceptance that Docker named-volume coverage closes the HAD-1155",
        ],
    )


def test_had_1158_runbook_keeps_data_hermes_boundary_explicit() -> None:
    boundary = _section(_read_runbook(), "Boundary with `/data/hermes` coverage")

    _assert_contains_all(
        boundary,
        [
            "`/data/hermes/profile-default`",
            "`/data/hermes/stacks`",
            "`/data/hermes/ctx-data`",
            "`/data/hermes/codex-home`",
            "They do not cover Docker named volumes stored under Docker's data root.",
            "Backup manifests must list each Docker named volume as a separate source.",
        ],
    )
