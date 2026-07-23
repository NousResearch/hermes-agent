"""Tests for explicit Hermes evolution run traces and mission manifests."""

from __future__ import annotations

import json


def test_start_trace_writes_profile_scoped_trace_json(tmp_path, monkeypatch):
    from agent import evolution_trace

    monkeypatch.setattr(evolution_trace, "get_hermes_home", lambda: tmp_path)

    result = evolution_trace.start_trace(
        mission="hermes-evolution",
        session_id="session-123",
        design_statement="Improve measured Hermes evolution.",
        skills_loaded=["hermes-agent", "test-driven-development"],
    )

    trace_path = tmp_path / "logs" / "evolution" / "runs" / result["run_id"] / "trace.json"
    assert result["trace_path"] == str(trace_path)
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    assert payload["run_id"] == result["run_id"]
    assert payload["mission"] == "hermes-evolution"
    assert payload["session_id"] == "session-123"
    assert payload["design_statement"] == "Improve measured Hermes evolution."
    assert payload["skills_loaded"] == ["hermes-agent", "test-driven-development"]
    assert payload["events"] == []
    assert payload["final_status"] == "running"


def test_record_event_updates_trace_aggregates(tmp_path, monkeypatch):
    from agent import evolution_trace

    monkeypatch.setattr(evolution_trace, "get_hermes_home", lambda: tmp_path)
    run_id = evolution_trace.start_trace(mission="software-dev")["run_id"]

    evolution_trace.record_event(
        run_id,
        "tool_call",
        tool="terminal",
        summary="pytest tests/agent/test_evolution_trace.py",
    )
    evolution_trace.record_event(
        run_id,
        "test_run",
        command="pytest tests/agent/test_evolution_trace.py",
        status="passed",
        summary="3 passed",
    )
    evolution_trace.record_event(
        run_id,
        "file_change",
        path="agent/evolution_trace.py",
        summary="add trace writer",
    )
    evolution_trace.record_event(
        run_id,
        "signal",
        signal="tests_passed",
        status="success",
        summary="trace tests passed",
    )

    payload = evolution_trace.read_trace(run_id)
    assert [e["type"] for e in payload["events"]] == ["tool_call", "test_run", "file_change", "signal"]
    assert payload["tools_called"] == ["terminal"]
    assert payload["tests_run"] == [
        {"command": "pytest tests/agent/test_evolution_trace.py", "status": "passed", "summary": "3 passed"}
    ]
    assert payload["files_changed"] == ["agent/evolution_trace.py"]
    assert payload["success_signals"] == ["tests_passed"]


def test_finish_trace_sets_terminal_status_and_summary(tmp_path, monkeypatch):
    from agent import evolution_trace

    monkeypatch.setattr(evolution_trace, "get_hermes_home", lambda: tmp_path)
    run_id = evolution_trace.start_trace(mission="hermes-evolution")["run_id"]

    result = evolution_trace.finish_trace(run_id, final_status="success", summary="Phase4 verified")

    payload = evolution_trace.read_trace(run_id)
    assert result["final_status"] == "success"
    assert payload["final_status"] == "success"
    assert payload["summary"] == "Phase4 verified"
    assert payload["completed_at"]


def test_load_mission_manifest_reads_guardrails(tmp_path, monkeypatch):
    from agent import evolution_trace

    monkeypatch.setattr(evolution_trace, "get_hermes_home", lambda: tmp_path)
    manifest_dir = tmp_path / "mission-manifests"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "hermes-evolution.yaml").write_text(
        """
name: hermes-evolution
design_statement: Improve Hermes through measured loops.
default_skills:
  - hermes-agent
  - test-driven-development
allowed_side_effects:
  - repo_file_edit
blocked_side_effects:
  - production_deploy
evaluation_gates:
  - tests_pass
""".strip(),
        encoding="utf-8",
    )

    manifest = evolution_trace.load_mission_manifest("hermes-evolution")

    assert manifest["name"] == "hermes-evolution"
    assert manifest["default_skills"] == ["hermes-agent", "test-driven-development"]
    assert manifest["allowed_side_effects"] == ["repo_file_edit"]
    assert manifest["blocked_side_effects"] == ["production_deploy"]
    assert manifest["evaluation_gates"] == ["tests_pass"]


def test_start_trace_from_manifest_copies_policy_fields(tmp_path, monkeypatch):
    from agent import evolution_trace

    monkeypatch.setattr(evolution_trace, "get_hermes_home", lambda: tmp_path)
    manifest_dir = tmp_path / "mission-manifests"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "genesis.yaml").write_text(
        """
name: genesis
design_statement: Evidence-first GENESIS decisions.
default_skills:
  - genesis-context-pack-compiler
allowed_side_effects:
  - research_files
blocked_side_effects:
  - live_trading
evaluation_gates:
  - evidence_pack_complete
""".strip(),
        encoding="utf-8",
    )

    result = evolution_trace.start_trace_from_manifest("genesis", session_id="s1")
    payload = evolution_trace.read_trace(result["run_id"])

    assert payload["mission"] == "genesis"
    assert payload["design_statement"] == "Evidence-first GENESIS decisions."
    assert payload["skills_loaded"] == ["genesis-context-pack-compiler"]
    assert payload["allowed_side_effects"] == ["research_files"]
    assert payload["blocked_side_effects"] == ["live_trading"]
    assert payload["evaluation_gates"] == ["evidence_pack_complete"]

def test_record_evidence_source_gate_and_decision_events(tmp_path, monkeypatch):
    from agent import evolution_trace

    monkeypatch.setattr(evolution_trace, "get_hermes_home", lambda: tmp_path)
    run_id = evolution_trace.start_trace(mission="genesis")["run_id"]

    evolution_trace.record_event(
        run_id,
        "evidence_source",
        source_type="obsidian_note",
        path="C:/vault/_master-index.md",
        title="Master Index",
        summary="MUST READ FIRST checked",
    )
    evolution_trace.record_event(
        run_id,
        "gate",
        gate="obsidian_master_index_checked",
        status="passed",
        summary="master index read",
    )
    evolution_trace.record_event(
        run_id,
        "decision",
        decision="HOLD",
        status="hold",
        summary="raw evidence still required",
    )

    payload = evolution_trace.read_trace(run_id)
    assert payload["evidence_sources"] == [
        {
            "source_type": "obsidian_note",
            "path": "C:/vault/_master-index.md",
            "title": "Master Index",
            "summary": "MUST READ FIRST checked",
        }
    ]
    assert payload["gate_results"] == [
        {"gate": "obsidian_master_index_checked", "status": "passed", "summary": "master index read"}
    ]
    assert payload["decisions"] == [
        {"decision": "HOLD", "status": "hold", "summary": "raw evidence still required"}
    ]
    assert payload["success_signals"] == ["obsidian_master_index_checked"]


def test_start_trace_from_manifest_copies_evidence_policy_fields(tmp_path, monkeypatch):
    from agent import evolution_trace

    monkeypatch.setattr(evolution_trace, "get_hermes_home", lambda: tmp_path)
    manifest_dir = tmp_path / "mission-manifests"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "genesis.yaml").write_text(
        """
name: genesis
default_skills:
  - obsidian
evidence_roots:
  - C:/Users/tkym1/OneDrive/ドキュメント/Obsidian Vault
must_read_notes:
  - _master-index.md
  - wiki/fx-genesis-operating-corrections-2026-06-18.md
evaluation_gates:
  - obsidian_master_index_checked
""".strip(),
        encoding="utf-8",
    )

    result = evolution_trace.start_trace_from_manifest("genesis")
    payload = evolution_trace.read_trace(result["run_id"])

    assert payload["evidence_roots"] == ["C:/Users/tkym1/OneDrive/ドキュメント/Obsidian Vault"]
    assert payload["must_read_notes"] == [
        "_master-index.md",
        "wiki/fx-genesis-operating-corrections-2026-06-18.md",
    ]



def test_record_context_pack_event_updates_aggregate(tmp_path, monkeypatch):
    from agent import evolution_trace

    monkeypatch.setattr(evolution_trace, "get_hermes_home", lambda: tmp_path)
    run_id = evolution_trace.start_trace(mission="genesis")["run_id"]

    evolution_trace.record_event(
        run_id,
        "context_pack",
        path="C:/tmp/genesis-pack.md",
        topic="GENESIS Context Pack Trace Integration",
        status="hold",
        source_count=3,
        missing_evidence="raw runtime proof still required",
        summary="MUST READ notes compressed into decision pack",
    )

    payload = evolution_trace.read_trace(run_id)
    assert payload["context_packs"] == [
        {
            "path": "C:/tmp/genesis-pack.md",
            "topic": "GENESIS Context Pack Trace Integration",
            "status": "hold",
            "source_count": 3,
            "missing_evidence": "raw runtime proof still required",
            "summary": "MUST READ notes compressed into decision pack",
        }
    ]


def test_start_trace_from_manifest_copies_generic_grill_gate(tmp_path, monkeypatch):
    from agent import evolution_trace

    monkeypatch.setattr(evolution_trace, "get_hermes_home", lambda: tmp_path)
    manifest_dir = tmp_path / "mission-manifests"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "hermes-evolution.yaml").write_text(
        """
name: hermes-evolution
grill_gate:
  purpose: Preserve ambiguity-killing questions before Hermes improvements.
  required_questions:
    - What exact Hermes capability is being strengthened?
    - What evidence is verified versus pointer?
    - What side effects are forbidden?
""".strip(),
        encoding="utf-8",
    )

    result = evolution_trace.start_trace_from_manifest("hermes-evolution")
    payload = evolution_trace.read_trace(result["run_id"])

    assert payload["grill_gate"] == {
        "purpose": "Preserve ambiguity-killing questions before Hermes improvements.",
        "required_questions": [
            "What exact Hermes capability is being strengthened?",
            "What evidence is verified versus pointer?",
            "What side effects are forbidden?",
        ],
    }


def test_grill_gate_required_questions_support_yaml_colons(tmp_path, monkeypatch):
    from agent import evolution_trace

    monkeypatch.setattr(evolution_trace, "get_hermes_home", lambda: tmp_path)
    manifest_dir = tmp_path / "mission-manifests"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "hermes-evolution.yaml").write_text(
        """
name: hermes-evolution
grill_gate:
  required_questions:
    - Which reviewer role should challenge this next: Codex, Claude Code, or Hermes?
""".strip(),
        encoding="utf-8",
    )

    manifest = evolution_trace.load_mission_manifest("hermes-evolution")

    assert manifest["grill_gate"]["required_questions"] == [
        "Which reviewer role should challenge this next: Codex, Claude Code, or Hermes?"
    ]


def test_record_change_packet_event_updates_aggregate(tmp_path, monkeypatch):
    from agent import evolution_trace

    monkeypatch.setattr(evolution_trace, "get_hermes_home", lambda: tmp_path)
    run_id = evolution_trace.start_trace(mission="hermes-evolution")["run_id"]

    evolution_trace.record_event(
        run_id,
        "change_packet",
        path="C:/tmp/hermes-change-packet.md",
        topic="Hermes Change Packet Compiler",
        objective="Bound implementation scope",
        status="hold",
        summary="Allowed files and tests defined",
    )

    payload = evolution_trace.read_trace(run_id)
    assert payload["change_packets"] == [
        {
            "path": "C:/tmp/hermes-change-packet.md",
            "topic": "Hermes Change Packet Compiler",
            "objective": "Bound implementation scope",
            "status": "hold",
            "summary": "Allowed files and tests defined",
        }
    ]
