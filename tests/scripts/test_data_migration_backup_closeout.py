from __future__ import annotations

import re
from pathlib import Path


CLOSEOUT_DOC = (
    Path(__file__).resolve().parents[2]
    / "notes"
    / "data-migration"
    / "backup-boundary-restore-readiness-closeout.md"
)


def _read_closeout() -> str:
    return CLOSEOUT_DOC.read_text(encoding="utf-8")


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
    assert not missing, f"missing closeout requirements: {missing}"


def _assert_in_order(text: str, expected: list[str]) -> None:
    cursor = 0
    for item in expected:
        next_index = text.find(item, cursor)
        assert next_index != -1, f"missing or out-of-order item: {item}"
        cursor = next_index + len(item)


def test_had_1155_closeout_defines_backup_classes() -> None:
    section = _section(_read_closeout(), "Backup classes")

    _assert_contains_all(
        section,
        [
            "`critical-durable`",
            "Required to resume Hermes, Codex, ctx, gateway, cron, or active project work.",
            "`important-durable`",
            "High-value history, notes, research, source materials, provenance, or generated records.",
            "`operational-history`",
            "Logs, cron output, reports, and evidence that improve auditability",
            "`rebuildable`",
            "Virtualenvs, dependency caches, build outputs, and downloaded toolchains",
            "`cache-excluded`",
            "Temporary caches, socket files, runtime locks, WAL/SHM residue",
            "`holdback-retained`",
            "Local rollback copy created during migration.",
        ],
    )


def test_had_1155_closeout_names_high_value_data_families() -> None:
    section = _section(_read_closeout(), "High-value data checklist")

    _assert_contains_all(
        section,
        [
            "Provenance records",
            "Ontology research",
            "Session state",
            "Cron output",
            "Project repos",
            "Source materials",
            "ctx data",
            "Codex metadata",
        ],
    )
    _assert_contains_all(
        section,
        [
            "`important-durable`",
            "`critical-durable`",
            "`operational-history`",
            "Docker service storage named in follow-ups",
            "secret-sensitive metadata",
        ],
    )


def test_had_1155_closeout_gates_holdbacks_and_orders_restore() -> None:
    markdown = _read_closeout()
    holdback = _section(markdown, "Holdback gate")
    restore_order = _section(markdown, "Restore-order assumptions after bare OS recovery")

    _assert_contains_all(
        holdback,
        [
            "/home/david/.hermes.pre-data-migration-holdback.${MIGRATION_ID}",
            "/home/david/stacks.pre-data-migration-holdback.${MIGRATION_ID}",
            "/home/david/.ctx-data.pre-data-migration-holdback.${MIGRATION_ID}",
            "/home/david/.codex.pre-data-migration-holdback.${MIGRATION_ID}",
            "backup verification evidence",
            "recurring backup includes every backed-up `/data/hermes`",
            "backup classes, restore order notes, exclusion decisions",
            "SQLite/session verification passes",
            "Repo readback for `/data/hermes/stacks`",
            "ctx and Codex metadata readback passes without exposing secrets",
            "Docker named-volume coverage is either completed through a follow-up issue",
            "manager explicitly accepts the backup verification evidence",
            "If any item fails, keep holdbacks and treat deletion as blocked.",
        ],
    )

    _assert_in_order(
        restore_order,
        [
            "1. Recreate the `david` user",
            "2. Mount the data disk at `/data`",
            "3. Restore `/data/hermes/profile-default`, `/data/hermes/stacks`,",
            "4. Recreate the logical path contracts",
            "5. Verify Hermes state while services are still stopped",
            "6. Verify `/home/david/stacks`",
            "7. Verify ctx and Codex metadata",
            "8. Restore service-specific Docker named volumes",
            "9. Start lower-level storage consumers before top-level intake",
            "10. Run application smoke checks and a backup readback check",
        ],
    )


def test_had_1155_closeout_assigns_follow_up_issue_ids() -> None:
    section = _section(_read_closeout(), "Follow-up issues to create or confirm")
    issue_ids = re.findall(r"^\| `?(HAD-\d+):", section, flags=re.MULTILINE)

    assert issue_ids == ["HAD-1158", "HAD-1159", "HAD-1160", "HAD-1161", "HAD-1162"]
    _assert_contains_all(
        section,
        [
            "Docker named-volume backup and restore plan",
            "Restore rehearsal for /data/hermes on a scratch host",
            "Backup verifier manifest for /data/hermes backup classes",
            "Reconcile backup boundary after HAD-1147 path-contract inventory merges",
            "Classify service-specific source-material stores outside /data/hermes",
        ],
    )
