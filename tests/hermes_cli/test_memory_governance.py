import hashlib
import json
import sqlite3
from pathlib import Path

from hermes_cli.memory_governance import (
    MemoryItem,
    build_proposal,
    classify_memory,
    execute_approved_proposals,
    load_builtin_items,
    load_json_items,
    render_report,
    write_proposals,
)


def test_classifies_task_state_as_move_to_linear():
    classification, reason, action = classify_memory(
        "Created Linear comment on PER-4601 with PR #123 evidence; Phase 2 done."
    )

    assert classification == "MOVE_TO_LINEAR"
    assert "Linear" in reason or "task" in reason
    assert "approval" in action


def test_classifies_procedure_as_move_to_skill():
    classification, reason, action = classify_memory(
        "When Mem0 profile is empty, run provider smoke and check search_fallback; pytest verifies the workflow."
    )

    assert classification == "MOVE_TO_SKILL"
    assert "procedural" in reason
    assert "skill" in action


def test_classifies_user_preference_as_keep_boot():
    classification, reason, action = classify_memory(
        "D prefers concise verified readback and expects exact approval before mutations."
    )

    assert classification == "KEEP_BOOT"
    assert "session start" in reason
    assert "No mutation" in action


def test_classifies_one_off_request_as_delete_candidate():
    classification, reason, action = classify_memory(
        "User попросил рекомендации по плану памяти."
    )

    assert classification == "DELETE_CANDIDATE"
    assert "one-off" in reason
    assert "approval" in action


def test_classifies_russian_linear_status_as_move_to_linear():
    classification, reason, action = classify_memory(
        "Рабочий пакет C — PER‑2950 lane permission matrix находится In Progress и обновлён сегодня."
    )

    assert classification == "MOVE_TO_LINEAR"
    assert "Linear" in reason or "task" in reason
    assert "approval" in action


def test_classifies_mixed_issue_state_and_durable_preference_as_update_candidate():
    classification, reason, action = classify_memory(
        "PER‑2941 остаётся главным umbrella, статус In Progress / HOLD, делегат = Cyrus; пользователь считает, что Cyrus не должен получать задачу umbrella/control‑tower, а только PR‑producing bounded implementation."
    )

    assert classification == "UPDATE_CANDIDATE"
    assert "durable preference" in reason
    assert "stable rule" in action


def test_classifies_russian_assistant_expectation_as_keep_boot():
    classification, reason, action = classify_memory(
        "Пользователь ожидает, что ассистент будет действовать как архитектор Linear."
    )

    assert classification == "KEEP_BOOT"
    assert "session start" in reason
    assert "No mutation" in action


def test_user_said_transient_want_is_delete_candidate_not_boot():
    classification, reason, action = classify_memory(
        "User said “делай”, indicating they want the assistant to proceed with the next action."
    )

    assert classification == "DELETE_CANDIDATE"
    assert "one-off" in reason
    assert "approval" in action


def test_build_proposal_has_stable_hash_and_approval_phrase():
    item = MemoryItem(source="mem0:get_all", row_number=1, memory_id="abc", text="User prefers concise responses.")

    first = build_proposal(item)
    second = build_proposal(item)

    assert first.proposal_id == second.proposal_id
    assert first.body_sha256 == second.body_sha256
    assert first.action_sha256 == second.action_sha256
    assert first.approval_phrase == f"approve memory {first.proposal_id} sha256 {first.action_sha256}"


def test_load_json_items_accepts_results_shape(tmp_path):
    fixture = tmp_path / "memories.json"
    fixture.write_text(
        json.dumps({"results": [{"id": "m1", "memory": "User prefers concise responses."}]}),
        encoding="utf-8",
    )

    items, source = load_json_items(fixture)

    assert source == "json:memories.json"
    assert len(items) == 1
    assert items[0].memory_id == "m1"
    assert items[0].text == "User prefers concise responses."


def test_load_builtin_items_reads_memory_and_user_files(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    memories = hermes_home / "memories"
    memories.mkdir(parents=True)
    (memories / "MEMORY.md").write_text("§\nMemory fact\n§\nLinear PER-1 Done", encoding="utf-8")
    (memories / "USER.md").write_text("§\nD prefers terse replies", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    items, source = load_builtin_items(target="all")

    assert source == "builtin:all"
    assert [item.source for item in items] == ["builtin:memory", "builtin:memory", "builtin:user"]
    assert items[0].memory_id.startswith("MEMORY.md:1:")
    assert items[2].text == "D prefers terse replies"


def test_load_builtin_items_can_target_user_only(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    memories = hermes_home / "memories"
    memories.mkdir(parents=True)
    (memories / "MEMORY.md").write_text("§\nMemory fact", encoding="utf-8")
    (memories / "USER.md").write_text("§\nUser fact", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    items, source = load_builtin_items(target="user")

    assert source == "builtin:user"
    assert len(items) == 1
    assert items[0].source == "builtin:user"
    assert items[0].text == "User fact"


def test_write_proposals_creates_sqlite_ledger(tmp_path):
    ledger = tmp_path / "ledger.db"
    proposal = build_proposal(
        MemoryItem(source="json:test", row_number=1, memory_id="row-1", text="Created Linear issue PER-123; done.")
    )

    run_id = write_proposals(ledger, [proposal], source="json:test")

    assert run_id.startswith("memgov_run_")
    with sqlite3.connect(ledger) as conn:
        rows = conn.execute(
            "SELECT proposal_id, classification, action_sha256, approval_phrase, status FROM proposals"
        ).fetchall()
        runs = conn.execute("SELECT run_id, dry_run FROM runs").fetchall()

    assert rows == [
        (
            proposal.proposal_id,
            proposal.classification,
            proposal.action_sha256,
            proposal.approval_phrase,
            "proposed",
        )
    ]
    assert runs == [(run_id, 1)]


def test_write_proposals_supersedes_old_proposed_rows_for_same_source(tmp_path):
    ledger = tmp_path / "ledger.db"
    old = build_proposal(
        MemoryItem(source="json:test", row_number=1, memory_id="old", text="User asked old one-off.")
    )
    new = build_proposal(
        MemoryItem(source="json:test", row_number=1, memory_id="new", text="User prefers concise responses.")
    )

    write_proposals(ledger, [old], source="json:test")
    write_proposals(ledger, [new], source="json:test")

    with sqlite3.connect(ledger) as conn:
        rows = dict(conn.execute("SELECT proposal_id, status FROM proposals").fetchall())

    assert rows[old.proposal_id] == "superseded"
    assert rows[new.proposal_id] == "proposed"


def test_write_proposals_supersedes_old_rows_for_mixed_builtin_sources(tmp_path):
    ledger = tmp_path / "ledger.db"
    old = build_proposal(
        MemoryItem(source="builtin:memory", row_number=1, memory_id="old", text="User asked old one-off.")
    )
    new = build_proposal(
        MemoryItem(source="builtin:user", row_number=1, memory_id="new", text="User prefers concise responses.")
    )

    write_proposals(ledger, [old], source="builtin:all")
    write_proposals(ledger, [new], source="builtin:all")

    with sqlite3.connect(ledger) as conn:
        rows = dict(conn.execute("SELECT proposal_id, status FROM proposals").fetchall())

    assert rows[old.proposal_id] == "superseded"
    assert rows[new.proposal_id] == "proposed"


def test_render_report_summarizes_counts_and_safety(tmp_path):
    proposals = [
        build_proposal(MemoryItem(source="json:test", row_number=1, memory_id="m1", text="User asked to record the plan in Linear issue PER-1.")),
        build_proposal(MemoryItem(source="json:test", row_number=2, memory_id="m2", text="D prefers terse verified readback.")),
    ]

    report = render_report(proposals, ledger_path=tmp_path / "ledger.db", run_id="run1", limit=5)

    assert "Hermes Memory Governance Report v0" in report
    assert "classification counts:" in report
    assert "MOVE_TO_LINEAR" in report
    assert "KEEP_BOOT" in report
    assert "approval examples" in report
    assert "no memory mutation executed" in report


class FakeMemoryClient:
    def __init__(self, memories):
        self.memories = dict(memories)
        self.deleted = []
        self.updated = []

    def get(self, memory_id):
        if memory_id not in self.memories:
            raise KeyError("Memory not found")
        return {"id": memory_id, "memory": self.memories[memory_id]}

    def delete(self, memory_id):
        if memory_id not in self.memories:
            raise KeyError("Memory not found")
        del self.memories[memory_id]
        self.deleted.append(memory_id)
        return {"message": "Memory deleted successfully!"}

    def update(self, memory_id, *args, **kwargs):
        if args:
            text = getattr(args[0], "text", None)
        else:
            text = kwargs.get("text")
        if memory_id not in self.memories:
            raise KeyError("Memory not found")
        self.memories[memory_id] = text
        self.updated.append((memory_id, text))
        return {"id": memory_id, "memory": text}


def test_execute_approved_delete_candidate_deletes_and_marks_executed(tmp_path):
    ledger = tmp_path / "ledger.db"
    proposal = build_proposal(
        MemoryItem(source="mem0:search", row_number=1, memory_id="m1", text="User said “делай”, indicating they want the assistant to proceed.")
    )
    write_proposals(ledger, [proposal], source="mem0:search")
    client = FakeMemoryClient({"m1": proposal.memory_text})

    results = execute_approved_proposals(
        ledger,
        approvals=[(proposal.proposal_id, proposal.action_sha256)],
        client=client,
    )

    assert [(r.proposal_id, r.status) for r in results] == [(proposal.proposal_id, "executed")]
    assert client.deleted == ["m1"]
    with sqlite3.connect(ledger) as conn:
        status = conn.execute("SELECT status FROM proposals WHERE proposal_id=?", (proposal.proposal_id,)).fetchone()[0]
    assert status == "executed"


def test_execute_approved_update_candidate_requires_replacement_and_updates(tmp_path):
    ledger = tmp_path / "ledger.db"
    proposal = build_proposal(
        MemoryItem(source="mem0:search", row_number=1, memory_id="m2", text="PER‑2941 статус In Progress; пользователь считает, что Cyrus не должен получать umbrella, только bounded implementation.")
    )
    assert proposal.classification == "UPDATE_CANDIDATE"
    write_proposals(ledger, [proposal], source="mem0:search")
    client = FakeMemoryClient({"m2": proposal.memory_text})
    replacement = "Cyrus should receive only bounded PR-producing implementation tasks."

    results = execute_approved_proposals(
        ledger,
        approvals=[(proposal.proposal_id, proposal.action_sha256)],
        replacements={proposal.proposal_id: replacement},
        replacement_shas={proposal.proposal_id: hashlib.sha256(replacement.encode("utf-8")).hexdigest()},
        client=client,
    )

    assert results[0].status == "executed"
    assert client.updated == [("m2", replacement)]
    assert client.memories["m2"] == replacement


def test_execute_approved_proposal_holds_on_sha_mismatch(tmp_path):
    ledger = tmp_path / "ledger.db"
    proposal = build_proposal(MemoryItem(source="mem0:search", row_number=1, memory_id="m1", text="User said old one-off."))
    write_proposals(ledger, [proposal], source="mem0:search")
    client = FakeMemoryClient({"m1": proposal.memory_text})

    results = execute_approved_proposals(ledger, approvals=[(proposal.proposal_id, "badsha")], client=client)

    assert results[0].status == "held"
    assert "sha" in results[0].detail.lower()
    assert client.deleted == []
    with sqlite3.connect(ledger) as conn:
        status = conn.execute("SELECT status FROM proposals WHERE proposal_id=?", (proposal.proposal_id,)).fetchone()[0]
    assert status == "proposed"


def test_execute_approved_proposal_holds_on_memory_drift(tmp_path):
    ledger = tmp_path / "ledger.db"
    proposal = build_proposal(MemoryItem(source="mem0:search", row_number=1, memory_id="m1", text="User said old one-off."))
    write_proposals(ledger, [proposal], source="mem0:search")
    client = FakeMemoryClient({"m1": "changed text"})

    results = execute_approved_proposals(
        ledger,
        approvals=[(proposal.proposal_id, proposal.action_sha256)],
        client=client,
    )

    assert results[0].status == "held"
    assert "drift" in results[0].detail.lower()
    assert client.deleted == []


def test_execute_approved_update_candidate_holds_without_replacement_sha(tmp_path):
    ledger = tmp_path / "ledger.db"
    proposal = build_proposal(
        MemoryItem(source="mem0:search", row_number=1, memory_id="m2", text="PER‑2941 статус In Progress; пользователь считает, что Cyrus не должен получать umbrella, только bounded implementation.")
    )
    write_proposals(ledger, [proposal], source="mem0:search")
    client = FakeMemoryClient({"m2": proposal.memory_text})
    replacement = "Cyrus should receive only bounded PR-producing implementation tasks."

    results = execute_approved_proposals(
        ledger,
        approvals=[(proposal.proposal_id, proposal.action_sha256)],
        replacements={proposal.proposal_id: replacement},
        client=client,
    )

    assert results[0].status == "held"
    assert "replacement sha256" in results[0].detail.lower()
    assert client.updated == []
    with sqlite3.connect(ledger) as conn:
        status = conn.execute("SELECT status FROM proposals WHERE proposal_id=?", (proposal.proposal_id,)).fetchone()[0]
    assert status == "held"


def test_execute_approved_update_candidate_holds_on_replacement_sha_mismatch(tmp_path):
    ledger = tmp_path / "ledger.db"
    proposal = build_proposal(
        MemoryItem(source="mem0:search", row_number=1, memory_id="m2", text="PER‑2941 статус In Progress; пользователь считает, что Cyrus не должен получать umbrella, только bounded implementation.")
    )
    write_proposals(ledger, [proposal], source="mem0:search")
    client = FakeMemoryClient({"m2": proposal.memory_text})
    replacement = "Cyrus should receive only bounded PR-producing implementation tasks."

    results = execute_approved_proposals(
        ledger,
        approvals=[(proposal.proposal_id, proposal.action_sha256)],
        replacements={proposal.proposal_id: replacement},
        replacement_shas={proposal.proposal_id: "0" * 64},
        client=client,
    )

    assert results[0].status == "held"
    assert "replacement sha256 mismatch" in results[0].detail.lower()
    assert client.updated == []


def test_execute_duplicate_approval_does_not_overwrite_executed_status(tmp_path):
    ledger = tmp_path / "ledger.db"
    proposal = build_proposal(
        MemoryItem(source="mem0:search", row_number=1, memory_id="m1", text="User said “делай”, indicating they want the assistant to proceed.")
    )
    write_proposals(ledger, [proposal], source="mem0:search")
    client = FakeMemoryClient({"m1": proposal.memory_text})

    results = execute_approved_proposals(
        ledger,
        approvals=[(proposal.proposal_id, proposal.action_sha256), (proposal.proposal_id, proposal.action_sha256)],
        client=client,
    )

    assert [(r.status, r.detail) for r in results] == [("executed", "deleted"), ("held", "duplicate approval ignored")]
    assert client.deleted == ["m1"]
    with sqlite3.connect(ledger) as conn:
        status = conn.execute("SELECT status FROM proposals WHERE proposal_id=?", (proposal.proposal_id,)).fetchone()[0]
    assert status == "executed"


def test_write_proposals_preserves_terminal_status_for_same_proposal_id(tmp_path):
    ledger = tmp_path / "ledger.db"
    proposal = build_proposal(
        MemoryItem(source="json:test", row_number=1, memory_id="row-1", text="User said old one-off.")
    )

    write_proposals(ledger, [proposal], source="json:test")
    with sqlite3.connect(ledger) as conn:
        conn.execute("UPDATE proposals SET status='executed' WHERE proposal_id=?", (proposal.proposal_id,))
    write_proposals(ledger, [proposal], source="json:test")

    with sqlite3.connect(ledger) as conn:
        status = conn.execute("SELECT status FROM proposals WHERE proposal_id=?", (proposal.proposal_id,)).fetchone()[0]
    assert status == "executed"


def test_write_proposals_reactivates_superseded_row_when_it_reappears(tmp_path):
    ledger = tmp_path / "ledger.db"
    old = build_proposal(MemoryItem(source="json:test", row_number=1, memory_id="old", text="User asked old one-off."))
    new = build_proposal(MemoryItem(source="json:test", row_number=1, memory_id="new", text="User prefers concise responses."))

    write_proposals(ledger, [old], source="json:test")
    write_proposals(ledger, [new], source="json:test")
    write_proposals(ledger, [old], source="json:test")

    with sqlite3.connect(ledger) as conn:
        rows = dict(conn.execute("SELECT proposal_id, status FROM proposals").fetchall())
    assert rows[old.proposal_id] == "proposed"
    assert rows[new.proposal_id] == "superseded"
