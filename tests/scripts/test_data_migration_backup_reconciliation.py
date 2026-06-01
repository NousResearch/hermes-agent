from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RECONCILIATION = (
    ROOT
    / "notes"
    / "data-migration"
    / "had-1161-backup-boundary-reconciliation.md"
)


def _read_reconciliation() -> str:
    return RECONCILIATION.read_text(encoding="utf-8")


def _assert_contains_all(text: str, expected: list[str]) -> None:
    missing = [item for item in expected if item not in text]
    assert missing == []


def test_had_1161_reconciliation_cites_source_artifacts() -> None:
    text = _read_reconciliation()

    _assert_contains_all(
        text,
        [
            "# HAD-1161 backup-boundary reconciliation after HAD-1147",
            "Issue: HAD-1161",
            "HAD-1147 / PR #132",
            "409081b533546af629d00e4c11b28fa787a58bb3",
            "`origin/main`",
            "`notes/data-migration/path-contract-inventory.md`",
            "`notes/data-migration/backup-boundary-restore-readiness-closeout.md`",
            "`scripts/data_migration_dry_run.py`",
            "`tests/scripts/test_data_migration_dry_run.py`",
            "`tests/scripts/test_data_migration_path_contract_inventory.py`",
        ],
    )


def test_had_1161_reconciliation_keeps_four_data_roots() -> None:
    text = _read_reconciliation()

    _assert_contains_all(
        text,
        [
            "No new `/data/hermes` subvolume is required after HAD-1147.",
            "boundary remains the four HAD-1155 roots",
            "`/data/hermes/profile-default`",
            "`/data/hermes/stacks`",
            "`/data/hermes/ctx-data`",
            "`/data/hermes/codex-home`",
            "`/home/david/.hermes`",
            "`/home/david/stacks`",
            "`/home/david/.ctx-data`",
            "`/home/david/.codex`",
        ],
    )


def test_had_1161_reconciliation_classifies_inventory_paths() -> None:
    text = _read_reconciliation()

    _assert_contains_all(
        text,
        [
            "Bind-mount backup through `/data/hermes/profile-default`",
            "Bind-mount backup through `/data/hermes/stacks`",
            "Bind-mount backup through `/data/hermes/ctx-data`",
            "Bind-mount backup through `/data/hermes/codex-home`",
            "Docker named-volume handling, not bind-mount backup through `/data/hermes`",
            "Explicit exclusion / no-move",
            "`/home/david/stacks/smb-ontology-platform/ops`",
            "`/home/david/stacks/hadto-decision-room-data/decision-room`",
            "`/home/david/.ops-agent/ops_heartbeat.sh`",
            "`/home/david/.local/share/uv/python`",
            "`/home/david/.config/gh`",
            "`/home/david/.config/git`",
            "`/opt/data`",
            "`/home/pn/.codex`",
            "`/workspace`",
            "`/tmp/hermes-results`",
        ],
    )


def test_had_1161_reconciliation_defers_docker_volumes_to_had_1158() -> None:
    text = _read_reconciliation()

    _assert_contains_all(
        text,
        [
            "HAD-1158",
            "`hadto-pipeline_pipeline-data`",
            "`ontology-platform_archivebox-data`",
            "`ontology-platform_oxigraph-data`",
            "`ontology-platform_source-materials-minio-data`",
            "outside this issue's bind-mount boundary",
        ],
    )


def test_had_1161_reconciliation_records_verification_command() -> None:
    text = _read_reconciliation()

    _assert_contains_all(
        text,
        [
            "## Verification",
            "python -m pytest tests/scripts/test_data_migration_backup_reconciliation.py",
            "classifies the HAD-1147 paths as bind-mount backup,",
            "Docker named-volume handling, or explicit exclusion",
        ],
    )
