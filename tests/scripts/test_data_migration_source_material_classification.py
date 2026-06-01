from __future__ import annotations

import re
from pathlib import Path


CLASSIFICATION_DOC = (
    Path(__file__).resolve().parents[2]
    / "notes"
    / "data-migration"
    / "source-material-store-classification-had-1162.md"
)


def _read_doc() -> str:
    return CLASSIFICATION_DOC.read_text(encoding="utf-8")


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
    assert not missing, f"missing classification requirements: {missing}"


def test_had_1162_classification_records_read_only_evidence() -> None:
    evidence = _section(_read_doc(), "Evidence commands")

    _assert_contains_all(
        evidence,
        [
            "git status --short --branch",
            "rg --files notes/data-migration",
            "docker volume ls --format",
            "docker volume inspect hadto-pipeline_pipeline-data",
            "docker inspect ontology-archivebox hadto-ontology-workbench",
            "docker system df -v",
            "find /home/david/stacks",
            "du -sh /home/david/.hermes/tmp/ontology-source-candidates",
            "curl -fsS http://127.0.0.1:9100/minio/health/live",
            "Permission denied",
        ],
    )


def test_had_1162_classifies_required_docker_stores() -> None:
    classified = _section(_read_doc(), "Classified stores")

    _assert_contains_all(
        classified,
        [
            "ontology-platform_source-materials-minio-data",
            "ontology-platform_archivebox-data",
            "ontology-platform_oxigraph-data",
            "hadto-pipeline_pipeline-data",
            "ontology-platform_evolver-orsd",
            "`important-durable`",
            "`critical-durable`",
            "Docker-volume backup/restore plan",
            "credential-safe bucket inventory",
            "PRAGMA integrity_check",
        ],
    )


def test_had_1162_classifies_repo_adjacent_and_temp_source_stores() -> None:
    classified = _section(_read_doc(), "Classified stores")

    _assert_contains_all(
        classified,
        [
            "/home/david/stacks/smb-ontology-platform/research/source_store",
            "/home/david/stacks/smb-ontology-platform/research/manifests",
            "/home/david/stacks/smb-ontology-platform/research/archivebox_exports",
            "/home/david/stacks/smb-ontology-platform/orsd",
            "/home/david/stacks/hadto-decision-room-data/decision-room",
            "/home/david/stacks/hadto-ontology-workbench/data",
            "/home/david/stacks/hadto-pipeline/.local/data/data.db",
            "/home/david/.hermes/tmp/ontology-source-candidates",
            "/home/david/Downloads/taildrop/Keet*Ontology*",
            "/home/david/code/de-novo/ontology",
            "HAD-1155 permits tmp exclusions",
            "Covered by `/home/david/stacks`",
        ],
    )


def test_had_1162_names_unknowns_and_uncovered_regions() -> None:
    legacy = _section(_read_doc(), "Legacy or unknown Docker volumes")
    uncovered = _section(_read_doc(), "Uncovered regions")
    acceptance = _section(_read_doc(), "Acceptance checklist")

    _assert_contains_all(
        legacy,
        [
            "server_oxigraph-data",
            "server_pipeline-data",
            "ont040-archivebox-data",
            "default `important-durable` until owner review",
        ],
    )
    _assert_contains_all(
        uncovered,
        [
            "Docker volume file-level contents were not readable",
            "MinIO bucket inventory was not collected",
            "Old agent worktrees",
            "Active backup jobs and backup-storage readback were not inspected",
            "PhoneItIn MinIO/Postgres volumes",
        ],
    )
    _assert_contains_all(
        acceptance,
        [
            "Docker-backed object/blob stores are explicitly identified",
            "Repo-adjacent source/material stores are explicitly identified",
            "Temporary and external source-intake regions are not guessed away",
            "Unknown legacy Docker volumes are held as `important-durable`",
        ],
    )
