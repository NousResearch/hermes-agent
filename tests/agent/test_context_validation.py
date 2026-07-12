"""Never-lose-important-context validation scenarios."""

import json
from pathlib import Path

import pytest

from agent.context_validation import (
    LocalNoteIndex,
    MemoryRecoveryWrite,
    build_context_validation_report,
    build_memory_backup_recovery_report,
    build_memory_startup_recovery_writes,
)
from tools.memory_tool import MemoryStore


@pytest.fixture()
def memory_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
    return tmp_path


def _new_store() -> MemoryStore:
    store = MemoryStore(memory_char_limit=1200, user_char_limit=1200)
    store.load_from_disk()
    return store


def test_preference_survives_compaction_and_restart_via_durable_memory(memory_dir):
    store = _new_store()
    assert store.add(
        "user",
        "User prefers concise morning briefs with revenue blockers first.",
    )["success"]

    # Simulate compaction/app restart: the old working conversation is gone, and
    # a fresh MemoryStore reloads only durable curated memory from disk.
    restarted = _new_store()
    report = build_context_validation_report(
        durable_memory_entries=restarted.user_entries,
        working_memory_entries=(),
        durable_expectations={"brief_style": ("concise", "morning briefs")},
        discarded_expectations={"smalltalk": ("weather banter",)},
    )

    assert report.ok
    assert "brief_style" in report.durable_memory
    assert report.working_memory == {}
    assert report.discarded_context["smalltalk"] == ("weather banter",)
    rendered = report.to_markdown()
    assert "Durable memory" in rendered
    assert "Working memory" in rendered
    assert "Discarded context" in rendered


def test_commitment_survives_garbage_collection_until_resolved(memory_dir):
    store = _new_store()
    assert store.add(
        "memory",
        "Active commitment: submit the Vanta application packet after Chris confirms final consent.",
    )["success"]

    # Filler is treated as garbage-collectable and must not leak into retained
    # surfaces, while the active commitment remains durable until resolved.
    restarted = _new_store()
    report = build_context_validation_report(
        durable_memory_entries=restarted.memory_entries,
        durable_expectations={"vanta_commitment": ("Active commitment", "Vanta", "final consent")},
        discarded_expectations={"filler": ("funny aside about lunch",)},
    )

    assert report.ok
    assert "vanta_commitment" in report.durable_memory
    assert "filler" in report.discarded_context


def test_conflicting_memory_requires_clarification_instead_of_overwrite():
    report = build_context_validation_report(
        conflict_candidates={
            "timezone": (
                "User timezone is America/New_York.",
                "User timezone is America/Los_Angeles.",
            )
        }
    )

    assert not report.ok
    assert report.requires_clarification
    assert report.unresolved_conflicts[0].key == "timezone"
    assert "requires clarification" in report.to_markdown()


def test_durable_note_index_recall_is_separate_from_curated_memory(tmp_path):
    vault = tmp_path / "vault"
    note = vault / "wiki" / "concepts" / "vanta-application.md"
    note.parent.mkdir(parents=True)
    note.write_text(
        "# Vanta application packet\n\n"
        "Remote U.S. Senior Software Engineer, Developer Experience.\n"
        "Packet includes Ashby form gates and resume-choice blockers.\n",
        encoding="utf-8",
    )

    report = build_context_validation_report(
        durable_memory_entries=(),
        note_index=LocalNoteIndex.from_path(Path(vault)),
        note_expectations={"vanta_packet_note": ("Vanta", "application packet", "Remote U.S.")},
    )

    assert report.ok
    hits = report.durable_notes["vanta_packet_note"]
    assert hits[0].surface == "durable_notes"
    assert hits[0].source == "wiki/concepts/vanta-application.md"
    assert report.durable_memory == {}


def test_memory_backup_recovery_report_requires_journal_note_and_index(tmp_path):
    vault = tmp_path / "vault"
    note = vault / "memories" / "client-preference.md"
    note.parent.mkdir(parents=True)
    note.write_text(
        "# Client preference\n\nPinned memory: Acme prefers callback windows after 3 PM.\n",
        encoding="utf-8",
    )

    report = build_memory_backup_recovery_report(
        [
            MemoryRecoveryWrite(
                id="mem-acme-callback",
                content="Acme prefers callback windows after 3 PM.",
                important=True,
                pinned=True,
                journaled=True,
                synced=True,
                local_indexed=True,
                durable_note_terms=("Acme", "callback windows", "3 PM"),
            )
        ],
        note_index=LocalNoteIndex.from_path(vault),
        last_successful_sync_at="2026-06-19T08:00:00Z",
        last_gc_run_at="2026-06-19T07:30:00Z",
    )

    assert report.ok
    assert report.protected_from_gc_ids == ("mem-acme-callback",)
    assert report.diagnostics["recovery_status"] == "ok"
    assert report.diagnostics["protected_memory_count"] == 1
    assert "Acme prefers" not in str(report.diagnostics)


def test_memory_backup_recovery_report_surfaces_retryable_missing_index(tmp_path):
    vault = tmp_path / "vault"
    note = vault / "memories" / "lead.md"
    note.parent.mkdir(parents=True)
    note.write_text(
        "# Lead\n\nImportant memory: Vanta packet waits on final consent.\n",
        encoding="utf-8",
    )

    report = build_memory_backup_recovery_report(
        [
            MemoryRecoveryWrite(
                id="mem-vanta-consent",
                content="Vanta packet waits on final consent.",
                important=True,
                journaled=True,
                synced=False,
                local_indexed=False,
                durable_note_terms=("Vanta", "final consent"),
            )
        ],
        note_index=LocalNoteIndex.from_path(vault),
    )

    assert not report.ok
    assert report.retryable_write_ids == ("mem-vanta-consent",)
    assert report.missing_local_index_ids == ("mem-vanta-consent",)
    assert report.recoverable_index_ids == ("mem-vanta-consent",)
    assert report.missing_journal_ids == ()
    assert report.missing_durable_note_ids == ()
    assert report.diagnostics["queued_write_count"] == 1
    checks = report.diagnostics["checks"]
    assert isinstance(checks, dict)
    assert checks["recoverable_index"] == 1


def test_memory_backup_recovery_report_names_sync_retry_plan_without_content(tmp_path):
    vault = tmp_path / "vault"
    note = vault / "memories" / "sync-retry.md"
    note.parent.mkdir(parents=True)
    note.write_text(
        "# Sync retry\n\nImportant memory: Quartz buyer phone token amethyst awaits provider sync.\n",
        encoding="utf-8",
    )

    report = build_memory_backup_recovery_report(
        [
            MemoryRecoveryWrite(
                id="mem-quartz-sync",
                content="Quartz buyer phone token amethyst awaits provider sync.",
                important=True,
                journaled=True,
                synced=False,
                local_indexed=True,
                durable_note_terms=("Quartz", "provider sync"),
                sync_retry_attempts=2,
                next_sync_retry_at="2026-07-12T16:05:00Z",
                last_sync_error_code="provider_503",
            )
        ],
        note_index=LocalNoteIndex.from_path(vault),
    )

    assert report.retryable_write_ids == ("mem-quartz-sync",)
    assert report.sync_retry_plan_by_id == {
        "mem-quartz-sync": {
            "attempts": 2,
            "next_retry_at": "2026-07-12T16:05:00Z",
            "last_error_code": "provider_503",
        }
    }
    assert report.diagnostics["checks"]["sync_retry_plan"] == 1
    rendered = report.to_markdown()
    assert "Sync retry plan" in rendered
    assert "mem-quartz-sync: attempts=2 next_retry_at=2026-07-12T16:05:00Z last_error_code=provider_503" in rendered
    assert "Quartz buyer" not in rendered
    assert "amethyst" not in rendered


def test_memory_backup_recovery_report_preserves_conflicting_facts():
    report = build_memory_backup_recovery_report(
        [
            MemoryRecoveryWrite(
                id="tz-east",
                content="User timezone is America/New_York.",
                important=True,
                pinned=True,
                journaled=True,
                synced=True,
                local_indexed=True,
                conflict_key="timezone",
            ),
            MemoryRecoveryWrite(
                id="tz-west",
                content="User timezone is America/Los_Angeles.",
                important=True,
                pinned=True,
                journaled=True,
                synced=True,
                local_indexed=True,
                conflict_key="timezone",
            ),
        ],
        note_index=LocalNoteIndex(()),
    )

    assert not report.ok
    assert report.unresolved_conflicts[0].key == "timezone"
    assert report.protected_from_gc_ids == ("tz-east", "tz-west")
    assert report.diagnostics["conflict_count"] == 1


def test_memory_backup_recovery_report_names_conflict_write_ids_without_values():
    report = build_memory_backup_recovery_report(
        [
            MemoryRecoveryWrite(
                id="tz-east",
                content="User timezone is America/New_York with private calendar anchors.",
                important=True,
                pinned=True,
                journaled=True,
                synced=True,
                local_indexed=True,
                conflict_key="timezone",
            ),
            MemoryRecoveryWrite(
                id="tz-west",
                content="User timezone is America/Los_Angeles with private calendar anchors.",
                important=True,
                pinned=True,
                journaled=True,
                synced=True,
                local_indexed=True,
                conflict_key="timezone",
            ),
        ],
        note_index=LocalNoteIndex(()),
    )

    assert not report.ok
    assert report.conflict_write_ids_by_key == {"timezone": ("tz-east", "tz-west")}
    assert report.diagnostics["checks"]["conflict_keys"] == 1

    rendered = report.to_markdown()
    assert "Conflict write ids" in rendered
    assert "timezone: tz-east, tz-west" in rendered
    assert "America/New_York" not in rendered
    assert "America/Los_Angeles" not in rendered
    assert "private calendar anchors" not in rendered


def test_memory_backup_recovery_report_blocks_protected_gc_without_rule_and_audit():
    report = build_memory_backup_recovery_report(
        [
            MemoryRecoveryWrite(
                id="mem-pinned-client",
                content="Pinned memory: Client Gamma retention clause is private.",
                important=True,
                pinned=True,
                journaled=True,
                synced=True,
                local_indexed=True,
            ),
            MemoryRecoveryWrite(
                id="mem-superseded-client",
                content="Important memory: Client Delta used an outdated escalation path.",
                important=True,
                journaled=True,
                synced=True,
                local_indexed=True,
            ),
        ],
        note_index=LocalNoteIndex(()),
        proposed_gc_delete_ids=("mem-pinned-client", "mem-superseded-client"),
        explicit_gc_rules={"mem-superseded-client": "user-confirmed-superseded-fact"},
        gc_audit_log_ids=("mem-superseded-client",),
    )

    assert not report.ok
    assert report.blocked_gc_delete_ids == ("mem-pinned-client",)
    assert report.approved_gc_delete_ids == ("mem-superseded-client",)
    assert report.gc_audit_by_id == {
        "mem-superseded-client": "user-confirmed-superseded-fact"
    }
    checks = report.diagnostics["checks"]
    assert isinstance(checks, dict)
    assert checks["blocked_gc_delete"] == 1
    assert checks["approved_gc_delete"] == 1

    rendered = report.to_markdown()
    assert "GC delete safety" in rendered
    assert "mem-pinned-client" in rendered
    assert "mem-superseded-client approved by user-confirmed-superseded-fact" in rendered
    assert "Client Gamma" not in rendered
    assert "Client Delta" not in rendered


def test_memory_backup_recovery_report_markdown_redacts_memory_content(tmp_path):
    vault = tmp_path / "vault"
    note = vault / "memories" / "private.md"
    note.parent.mkdir(parents=True)
    note.write_text(
        "# Private\n\nPinned memory: Client Alpha has a private launch codeword orchid.\n",
        encoding="utf-8",
    )

    report = build_memory_backup_recovery_report(
        [
            MemoryRecoveryWrite(
                id="mem-alpha-secret",
                content="Client Alpha has a private launch codeword orchid.",
                important=True,
                pinned=True,
                journaled=True,
                synced=False,
                local_indexed=False,
                durable_note_terms=("Client Alpha", "codeword orchid"),
            )
        ],
        note_index=LocalNoteIndex.from_path(vault),
        last_successful_sync_at="2026-07-12T09:30:00Z",
        last_gc_run_at="2026-07-12T09:00:00Z",
    )

    rendered = report.to_markdown()

    assert "# Memory backup recovery report" in rendered
    assert "recovery_status: needs_attention" in rendered
    assert "queued_write_count: 1" in rendered
    assert "mem-alpha-secret" in rendered
    assert "missing local index" in rendered
    assert "recoverable local index" in rendered
    assert "Client Alpha" not in rendered
    assert "codeword orchid" not in rendered


def test_memory_backup_recovery_report_tracks_refinement_run_without_values(tmp_path):
    vault = tmp_path / "vault"
    note = vault / "memories" / "epsilon.md"
    note.parent.mkdir(parents=True)
    note.write_text(
        "# Epsilon\n\nPinned memory: Client Epsilon has private renewal keyword moonstone.\n",
        encoding="utf-8",
    )

    report = build_memory_backup_recovery_report(
        [
            MemoryRecoveryWrite(
                id="mem-epsilon-secret",
                content="Client Epsilon has private renewal keyword moonstone.",
                important=True,
                pinned=True,
                journaled=True,
                synced=True,
                local_indexed=True,
                durable_note_terms=("Client Epsilon", "moonstone"),
            )
        ],
        note_index=LocalNoteIndex.from_path(vault),
        last_successful_sync_at="2026-07-12T15:20:00Z",
        last_gc_run_at="2026-07-12T15:10:00Z",
        last_refinement_run_at="2026-07-12T15:15:00Z",
    )

    assert report.ok
    assert report.diagnostics["last_refinement_run_at"] == "2026-07-12T15:15:00Z"
    rendered = report.to_markdown()
    assert "last_refinement_run_at: 2026-07-12T15:15:00Z" in rendered
    assert "Client Epsilon" not in rendered
    assert "moonstone" not in rendered


def test_memory_backup_recovery_report_flags_unrecoverable_index_gaps():
    report = build_memory_backup_recovery_report(
        [
            MemoryRecoveryWrite(
                id="mem-lost-cache",
                content="A critical memory that is missing every recovery surface.",
                important=True,
                pinned=True,
                journaled=False,
                synced=False,
                local_indexed=False,
                durable_note_terms=("critical memory", "recovery surface"),
            )
        ],
        note_index=LocalNoteIndex(()),
    )

    assert not report.ok
    assert report.missing_journal_ids == ("mem-lost-cache",)
    assert report.missing_durable_note_ids == ("mem-lost-cache",)
    assert report.missing_local_index_ids == ("mem-lost-cache",)
    assert report.recoverable_index_ids == ()
    assert report.unrecoverable_index_ids == ("mem-lost-cache",)
    checks = report.diagnostics["checks"]
    assert isinstance(checks, dict)
    assert checks["unrecoverable_index"] == 1
    rendered = report.to_markdown()
    assert "unrecoverable local index" in rendered
    assert "critical memory" not in rendered


def test_memory_backup_recovery_report_names_rebuild_sources_without_content(tmp_path):
    vault = tmp_path / "vault"
    note = vault / "memories" / "operator.md"
    note.parent.mkdir(parents=True)
    note.write_text(
        "# Operator\n\nImportant memory: Atlas escalation path stays in the pinned runbook.\n",
        encoding="utf-8",
    )

    report = build_memory_backup_recovery_report(
        [
            MemoryRecoveryWrite(
                id="mem-atlas-escalation",
                content="Atlas escalation path stays in the pinned runbook.",
                important=True,
                journaled=True,
                synced=True,
                local_indexed=False,
                durable_note_terms=("Atlas", "pinned runbook"),
            )
        ],
        note_index=LocalNoteIndex.from_path(vault),
    )

    assert report.recoverable_index_ids == ("mem-atlas-escalation",)
    assert report.recovery_sources_by_id == {
        "mem-atlas-escalation": ("journal", "durable_note")
    }
    rendered = report.to_markdown()
    assert "recovery sources" in rendered
    assert "mem-atlas-escalation: journal, durable_note" in rendered
    assert "Atlas escalation" not in rendered
    assert "pinned runbook" not in rendered


def test_memory_backup_recovery_report_exposes_redacted_startup_rebuild_tasks(tmp_path):
    vault = tmp_path / "vault"
    note = vault / "memories" / "support-window.md"
    note.parent.mkdir(parents=True)
    note.write_text(
        "# Support window\n\nPinned memory: Nova account prefers support callbacks after 4 PM.\n",
        encoding="utf-8",
    )

    report = build_memory_backup_recovery_report(
        [
            MemoryRecoveryWrite(
                id="mem-nova-callback",
                content="Nova account prefers support callbacks after 4 PM.",
                important=True,
                pinned=True,
                journaled=True,
                synced=True,
                local_indexed=False,
                durable_note_terms=("Nova", "support callbacks", "4 PM"),
            ),
            MemoryRecoveryWrite(
                id="mem-orphaned",
                content="Orphaned memory has no recovery surface.",
                important=True,
                pinned=True,
                journaled=False,
                synced=False,
                local_indexed=False,
                durable_note_terms=("Orphaned memory", "recovery surface"),
            ),
        ],
        note_index=LocalNoteIndex.from_path(vault),
    )

    assert [task.write_id for task in report.startup_recovery_tasks] == [
        "mem-nova-callback"
    ]
    task = report.startup_recovery_tasks[0]
    assert task.target_surface == "local_index"
    assert task.sources == ("journal", "durable_note")
    assert report.diagnostics["startup_recovery_task_count"] == 1

    rendered = report.to_markdown()
    assert "startup recovery tasks" in rendered
    assert "mem-nova-callback -> local_index via journal, durable_note" in rendered
    assert "mem-orphaned ->" not in rendered
    assert "Nova account" not in rendered
    assert "Orphaned memory" not in rendered


def test_memory_startup_snapshot_adapter_reads_journal_and_notes_without_content(tmp_path):
    memory_dir = tmp_path / "memories"
    memory_dir.mkdir()
    memory_text = "Pinned memory: Atlas launch codeword redwood must survive restart."
    (memory_dir / "MEMORY.md").write_text(memory_text, encoding="utf-8")
    journal_path = memory_dir / "memory-wal.jsonl"
    journal_path.write_text(
        json.dumps({"target": "memory", "content": memory_text}) + "\n",
        encoding="utf-8",
    )
    vault = tmp_path / "vault"
    note = vault / "memories" / "atlas.md"
    note.parent.mkdir(parents=True)
    note.write_text(f"# Atlas\n\n{memory_text}\n", encoding="utf-8")

    writes = build_memory_startup_recovery_writes(
        memory_dir,
        journal_path=journal_path,
        local_index_ids=(),
    )

    assert len(writes) == 1
    write = writes[0]
    assert write.important is True
    assert write.pinned is True
    assert write.journaled is True
    assert write.local_indexed is False
    assert write.id.startswith("memory-")
    assert write.id == build_memory_startup_recovery_writes(
        memory_dir,
        journal_path=journal_path,
        local_index_ids=(),
    )[0].id

    report = build_memory_backup_recovery_report(
        writes,
        note_index=LocalNoteIndex.from_path(vault),
    )

    assert report.recoverable_index_ids == (write.id,)
    assert report.startup_recovery_tasks[0].sources == ("journal", "durable_note")
    rendered = report.to_markdown()
    assert write.id in rendered
    assert "Atlas launch" not in rendered
    assert "redwood" not in rendered


def test_memory_startup_snapshot_adapter_recovers_from_corrupted_local_index_cache(tmp_path):
    memory_dir = tmp_path / "memories"
    memory_dir.mkdir()
    memory_text = "Pinned memory: Boreal renewal token bluebird must survive restart."
    (memory_dir / "MEMORY.md").write_text(memory_text, encoding="utf-8")
    journal_path = memory_dir / "memory-wal.jsonl"
    journal_path.write_text(
        json.dumps({"target": "memory", "content": memory_text}) + "\n",
        encoding="utf-8",
    )
    local_index_cache = memory_dir / "local-index-cache.json"
    local_index_cache.write_text("{not-valid-json", encoding="utf-8")
    vault = tmp_path / "vault"
    note = vault / "memories" / "boreal.md"
    note.parent.mkdir(parents=True)
    note.write_text(f"# Boreal\n\n{memory_text}\n", encoding="utf-8")

    writes = build_memory_startup_recovery_writes(
        memory_dir,
        journal_path=journal_path,
        local_index_cache_path=local_index_cache,
    )

    assert len(writes) == 1
    write = writes[0]
    assert write.local_indexed is False
    assert write.recovery_warnings == ("local_index_cache_unreadable",)

    report = build_memory_backup_recovery_report(
        writes,
        note_index=LocalNoteIndex.from_path(vault),
    )

    assert report.recoverable_index_ids == (write.id,)
    assert report.recovery_warnings_by_id == {
        write.id: ("local_index_cache_unreadable",)
    }
    rendered = report.to_markdown()
    assert "Recovery warnings" in rendered
    assert f"{write.id}: local_index_cache_unreadable" in rendered
    assert "Boreal renewal" not in rendered
    assert "bluebird" not in rendered
