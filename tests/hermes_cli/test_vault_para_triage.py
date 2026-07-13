from __future__ import annotations

import json
from pathlib import Path

from hermes_cli import vault_para_triage as triage


def _make_vault(tmp_path: Path) -> Path:
    vault = tmp_path / "vault"
    (vault / "Inbox").mkdir(parents=True)
    (vault / "Projects").mkdir()
    (vault / "Areas" / "Health").mkdir(parents=True)
    (vault / "Resources" / "Reference").mkdir(parents=True)
    (vault / "Resources" / "_Inbox Review").mkdir(parents=True)
    (vault / "Archives").mkdir()
    return vault


def test_run_triage_adds_frontmatter_moves_note_and_writes_audit(tmp_path):
    vault = _make_vault(tmp_path)
    note = vault / "Inbox" / "doctor-follow-up.md"
    note.write_text("Need to call the doctor tomorrow about blood tests.\n", encoding="utf-8")

    summary = triage.run_triage(
        vault_path=vault,
        config_override={
            "routing": {
                "rules": [
                    {
                        "name": "health",
                        "match_any": ["doctor", "blood test"],
                        "target": "Areas/Health",
                        "confidence": 0.96,
                        "reason": "health note",
                    }
                ]
            }
        },
    )

    staged = vault / ".hermes" / "note-capture" / "staging" / "vault" / summary["processed"][0]["entry_id"] / "Areas" / "Health" / "doctor-follow-up.md"
    assert staged.exists()
    content = staged.read_text(encoding="utf-8")
    assert content.startswith("---\n")
    assert "title: doctor follow up" in content.lower()
    assert "para_target: Areas/Health" in content
    assert "para_triage_status: captured" in content

    audit_path = vault / ".hermes" / "para-triage" / "audit.jsonl"
    audit_rows = triage._read_jsonl(audit_path)
    assert len(audit_rows) == 1
    assert audit_rows[0]["path_before"] == "Inbox/doctor-follow-up.md"
    assert audit_rows[0]["path_after"] == "Areas/Health/doctor-follow-up.md"
    assert summary["counts"]["captured"] == 1
    assert not note.exists()

    capture_event = json.loads(
        (vault / ".hermes" / "note-capture" / "events" / f"{summary['processed'][0]['entry_id']}.json").read_text(encoding="utf-8")
    )
    assert capture_event["capture_model"] == "canonical_event_log"
    assert sorted(capture_event["projection"]["stores"]) == ["second_brain", "vault"]

    structure = json.loads((vault / ".hermes" / "para-triage" / "structure.json").read_text(encoding="utf-8"))
    assert "Areas/Health" in structure["targets"]


def test_apply_feedback_correct_relocates_note_and_learns_example(tmp_path):
    vault = _make_vault(tmp_path)
    note = vault / "Inbox" / "clinic-plan.md"
    note.write_text("Follow up with the clinic and update my records.\n", encoding="utf-8")

    summary = triage.run_triage(
        vault_path=vault,
        classifier=lambda **_: {
            "target": "Resources/Reference",
            "confidence": 0.31,
            "reason": "not sure",
            "needs_feedback": True,
        },
    )

    event = summary["processed"][0]
    initial = (
        vault
        / ".hermes"
        / "note-capture"
        / "staging"
        / "vault"
        / event["entry_id"]
        / "Resources"
        / "_Inbox Review"
        / "clinic-plan.md"
    )
    assert initial.exists()

    feedback = triage.apply_feedback(
        action="correct",
        entry_id=event["entry_id"],
        target="Areas/Health",
        vault_path=vault,
    )

    corrected = (
        vault
        / ".hermes"
        / "note-capture"
        / "staging"
        / "vault"
        / event["entry_id"]
        / "Areas"
        / "Health"
        / "clinic-plan.md"
    )
    assert corrected.exists()
    assert feedback["target"] == "Areas/Health"
    assert feedback["relocated_path"] == "Areas/Health/clinic-plan.md"

    pending = triage.list_pending_feedback(vault_path=vault)
    assert pending == []

    examples_path = vault / ".hermes" / "para-triage" / "routing_examples.json"
    examples = json.loads(examples_path.read_text(encoding="utf-8"))["examples"]
    assert examples[0]["entry_id"] == event["entry_id"]
    assert examples[0]["target"] == "Areas/Health"


def test_handle_feedback_command_lists_pending_items(tmp_path, monkeypatch):
    vault = _make_vault(tmp_path)
    note = vault / "Inbox" / "mystery-note.md"
    note.write_text("A note that should probably be reviewed later.\n", encoding="utf-8")

    triage.run_triage(
        vault_path=vault,
        classifier=lambda **_: {
            "target": "Resources/Reference",
            "confidence": 0.2,
            "reason": "uncertain",
            "needs_feedback": True,
        },
    )

    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(vault))
    output = triage.handle_feedback_command("list")
    assert "Pending PARA feedback" in output
    assert "Resources/_Inbox Review/mystery-note.md" in output


def test_low_confidence_action_inbox_keeps_note_in_inbox(tmp_path):
    vault = _make_vault(tmp_path)
    note = vault / "Inbox" / "unsure-note.md"
    note.write_text("Something ambiguous that should stay for review.\n", encoding="utf-8")

    summary = triage.run_triage(
        vault_path=vault,
        config_override={
            "routing": {
                "min_confidence": 0.8,
                "low_confidence_action": "inbox",
            }
        },
        classifier=lambda **_: {
            "target": "Resources/Reference",
            "confidence": 0.22,
            "reason": "too ambiguous",
            "needs_feedback": True,
        },
    )

    kept = vault / "Inbox" / "unsure-note.md"
    assert not kept.exists()
    content = (
        vault
        / ".hermes"
        / "note-capture"
        / "staging"
        / "vault"
        / summary["processed"][0]["entry_id"]
        / "Inbox"
        / "unsure-note.md"
    ).read_text(encoding="utf-8")
    assert "para_triage_status: needs-feedback" in content
    assert "para_target: Inbox" in content
    assert summary["processed"][0]["path_after"] == "Inbox/unsure-note.md"


def test_explicit_inbox_target_is_allowed(tmp_path):
    vault = _make_vault(tmp_path)
    note = vault / "Inbox" / "scratch-capture.md"
    note.write_text("Keep this as a raw inbox capture for later sorting.\n", encoding="utf-8")

    summary = triage.run_triage(
        vault_path=vault,
        classifier=lambda **_: {
            "target": "Inbox",
            "confidence": 0.93,
            "reason": "raw capture should remain in inbox",
            "needs_feedback": False,
        },
    )

    staged = (
        vault
        / ".hermes"
        / "note-capture"
        / "staging"
        / "vault"
        / summary["processed"][0]["entry_id"]
        / "Inbox"
        / "scratch-capture.md"
    )
    assert staged.exists()
    assert summary["processed"][0]["path_after"] == "Inbox/scratch-capture.md"

    capture_event = json.loads(
        (
            vault
            / ".hermes"
            / "note-capture"
            / "events"
            / f"{summary['processed'][0]['entry_id']}.json"
        ).read_text(encoding="utf-8")
    )
    assert capture_event["routing"]["target"] == "Inbox"


def test_feedback_status_includes_projection_summary(tmp_path):
    vault = _make_vault(tmp_path)
    note = vault / "Inbox" / "ops.md"
    note.write_text("Review the system and route this later.\n", encoding="utf-8")

    triage.run_triage(
        vault_path=vault,
        classifier=lambda **_: {
            "target": "Resources/Reference",
            "confidence": 0.9,
            "reason": "reference",
            "needs_feedback": False,
        },
    )

    status = triage.feedback_status(vault_path=vault)
    assert status["projection_status"]["sync_contract"]["capture_model"] == "canonical_event_log"
    assert status["projection_status"]["pending_events"] == 1


def test_format_run_report_includes_on_demand_audit_hint():
    report = triage.format_run_report(
        {
            "run_id": "20260712T020000Z",
            "vault_path": "/tmp/vault",
            "counts": {"captured": 1, "needs_feedback": 0},
            "processed": [
                {
                    "entry_id": "entry-1",
                    "path_before": "Inbox/example.md",
                    "path_after": "Projects/Example/example.md",
                    "confidence": 0.95,
                    "needs_feedback": False,
                }
            ],
        }
    )

    assert "On-demand audit: /para-feedback status | /para-feedback list" in report


def test_projection_store_path_prefix_is_applied_to_staged_relative_path(tmp_path):
    vault = _make_vault(tmp_path)
    note = vault / "Inbox" / "launch-summary.md"
    note.write_text("OpenAI launch notes and takeaways.\n", encoding="utf-8")

    summary = triage.run_triage(
        vault_path=vault,
        config_override={
            "projection": {
                "stores": {
                    "vault": {"enabled": True, "path_prefix": ""},
                    "second_brain": {"enabled": True, "path_prefix": "runtime-source"},
                }
            },
            "routing": {
                "rules": [
                    {
                        "name": "launch",
                        "match_any": ["launch", "takeaways"],
                        "target": "Resources/Reference",
                        "confidence": 0.99,
                        "reason": "reference note",
                    }
                ]
            },
        },
    )

    event_id = summary["processed"][0]["entry_id"]
    capture_event = json.loads(
        (vault / ".hermes" / "note-capture" / "events" / f"{event_id}.json").read_text(encoding="utf-8")
    )
    second_brain = capture_event["projection"]["stores"]["second_brain"]
    assert second_brain["target_relative_path"] == "runtime-source/Resources/Reference/launch-summary.md"
    assert second_brain["staged_relative_path"] == (
        f"staging/second_brain/{event_id}/runtime-source/Resources/Reference/launch-summary.md"
    )


def test_projection_summary_includes_memory_hint_and_enabled_stores(tmp_path):
    vault = _make_vault(tmp_path)
    note = vault / "Inbox" / "capture.md"
    note.write_text("Something to route.\n", encoding="utf-8")

    summary = triage.run_triage(
        vault_path=vault,
        config_override={
            "projection": {
                "stores": {
                    "vault": {"enabled": True, "path_prefix": ""},
                    "second_brain": {"enabled": False, "path_prefix": "runtime-source"},
                }
            },
            "routing": {
                "rules": [
                    {
                        "name": "capture",
                        "match_any": ["route"],
                        "target": "Resources/Reference",
                        "confidence": 0.95,
                        "reason": "reference note",
                    }
                ]
            },
        },
    )

    projection_status = summary["projection_status"]
    assert projection_status["sync_contract"]["stores"] == ["vault"]
    assert "downstream staged projections" in projection_status["memory_hint"]
